# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Role-indexed AFD contract artifacts and fail-closed schema handling."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from aisimulate.afd_artifacts import AFDQualificationError, build_afd_qualification, write_afd_qualification_artifacts
from aisimulate.sweeper.afd_parallel import AFDTopology
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    ReplaySpec,
    RunnerCapabilities,
    canonical_json,
    resolve_afd_replay_identity,
)
from scripts import afd_caliber_matrix


@pytest.mark.parametrize("fail_on", [None, 1, 2])
def test_caliber_matrix_cleans_variant_configs_on_success_and_failure(tmp_path, monkeypatch, fail_on):
    base = tmp_path / "base.yaml"
    base.write_text("traffic:\n  load: {type: concurrency, concurrency: 1}\n")
    monkeypatch.setattr(sys, "argv", ["afd_caliber_matrix", "--config", str(base)])
    monkeypatch.setattr(afd_caliber_matrix, "OPEN_LOOP_LOADS", ())
    original_variant = afd_caliber_matrix._variant
    paths = []
    calls = []

    def variant(*args, **kwargs):
        path = original_variant(*args, **kwargs)
        paths.append(path)
        return path

    def run(path):
        assert path.is_file()
        pinned = yaml.safe_load(path.read_text())["engine"]["estimator_config"]["afd"]["frozen_decode_identity"]
        calls.append(pinned)
        if len(calls) == fail_on:
            raise RuntimeError("prediction failed")
        return {"completed_requests": 1}

    monkeypatch.setattr(afd_caliber_matrix, "_variant", variant)
    monkeypatch.setattr(afd_caliber_matrix, "_run", run)
    try:
        if fail_on is None:
            assert afd_caliber_matrix.main() == 0
            assert calls == [True, False]
        else:
            with pytest.raises(RuntimeError, match="prediction failed"):
                afd_caliber_matrix.main()
        assert len(paths) == (1 if fail_on == 1 else 2)
        assert all(not path.exists() for path in paths)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def source_metadata(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.afd_artifacts.validate_afd_model",
        lambda model: {
            "num_experts": 128,
            "layers": 48,
            "num_moe_layers": 48,
            "num_shared_experts": 0,
            "architecture": "Qwen3MoeForCausalLM",
            "extra_params": None,
            "moe_inter_size": 768,
            "raw_config": {},
        },
    )


def _spec(mode="agg", afd_roles=("aggregated",), layout="four_stage"):
    roles = ("aggregated",) if mode == "agg" else ("prefill", "decode")
    parallel, metadata, kwargs = {}, {}, {}
    for role in roles:
        prefix = "" if role == "aggregated" else f"{role}_"
        count = 1 if role in afd_roles else 2
        if role in afd_roles:
            topology = AFDTopology(
                n_a_nodes=1,
                n_f_nodes=1,
                gpus_per_node=8,
                a_tp_size=1,
                batch_size_per_a_replica=16,
                is_moe=True,
                num_experts=128,
                pipeline_layout=layout,
                a_system="h200_sxm",
                f_system="h200_sxm",
            )
            parallel[role] = {"type": "afd", **topology.provenance()}
            raw = {"afd": topology.provenance()["topology"]}
            metadata[role] = {
                "provider": "afd",
                "measurement_required": False,
                "model": "Qwen/Qwen3-30B-A3B",
                "config": topology.provenance()["topology"],
            }
        else:
            raw = {"timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}}
            parallel.update({prefix + "tp": 2, prefix + "pp": 1, prefix + "attention_dp": 1})
        parallel[prefix + "replicas"] = count
        kwargs["agg_engine_args" if role == "aggregated" else role + "_engine_args"] = raw
        kwargs["num_workers" if role == "aggregated" else "num_" + role + "_workers"] = count
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=mode,
            backend="sglang",
            backend_version="0.5.14",
            parallel_config=parallel,
            performance_model_metadata=metadata,
            **kwargs,
        ),
        workload={"isl": 16, "osl": 4, "concurrency": 2, "request_count": 4},
        goal={},
    )


@pytest.mark.parametrize(
    "workload,expected",
    [
        ({"isl": 16, "osl": 4}, True),
        ({"isl": 16, "osl": 4, "random_range_ratio": 0.5}, False),
        ({"source_type": "trace", "trace_path": "unused.jsonl"}, False),
        ({"source_type": "synthetic-session"}, False),
    ],
)
@pytest.mark.parametrize("explicit", [None, True, False])
def test_pricing_identity_is_pinned_saved_and_independent_of_environment(
    tmp_path, monkeypatch, workload, expected, explicit
):
    spec = replace(_spec("disagg", ("prefill", "decode")), workload=workload)
    for metadata in spec.backend_deployment.performance_model_metadata.values():
        metadata["estimator"] = {"estimator_config": {"afd": {"frozen_decode_identity": explicit}}}
    original = canonical_json(spec)
    resolved = resolve_afd_replay_identity(spec)
    assert canonical_json(spec) == original
    assert resolve_afd_replay_identity(resolved) is resolved
    for metadata in resolved.backend_deployment.performance_model_metadata.values():
        assert metadata["frozen_decode_identity"] is (expected if explicit is None else explicit)
    paths = write_afd_qualification_artifacts(tmp_path, spec)
    wire = json.loads(paths[0].read_text())
    wire["backend_deployment"] = BackendDeploymentSpec(**wire["backend_deployment"])
    restored = ReplaySpec(**wire)
    assert canonical_json(restored) == canonical_json(resolved)
    qualification = build_afd_qualification(restored)
    for value in ("0", "1", "invalid"):
        monkeypatch.setenv("AISIMULATE_AFD_FROZEN_DECODE_IDENTITY", value)
        assert canonical_json(resolve_afd_replay_identity(spec)) == canonical_json(resolved)
        assert build_afd_qualification(restored) == qualification


def test_different_explicit_pricing_choices_have_different_replay_hashes():
    hashes = []
    for explicit in (True, False):
        spec = _spec()
        spec.backend_deployment.performance_model_metadata["aggregated"]["estimator"] = {
            "estimator_config": {"afd": {"frozen_decode_identity": explicit}}
        }
        hashes.append(build_afd_qualification(spec)["identity"]["replay_spec_sha256"])
    assert hashes[0] != hashes[1]


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
def test_invalid_explicit_pricing_identity_is_rejected(value):
    spec = _spec()
    spec.backend_deployment.performance_model_metadata["aggregated"]["estimator"] = {
        "estimator_config": {"afd": {"frozen_decode_identity": value}}
    }
    with pytest.raises(ValueError, match="invalid estimator_config"):
        resolve_afd_replay_identity(spec)


@pytest.mark.parametrize("saved", [False, 1, "true", None])
def test_inconsistent_saved_pricing_identity_is_rejected(saved):
    spec = _spec()
    spec.backend_deployment.performance_model_metadata["aggregated"]["frozen_decode_identity"] = saved
    with pytest.raises(ValueError, match="frozen_decode_identity conflicts"):
        resolve_afd_replay_identity(spec)


def test_monolithic_api_one_consumer_rejects_afd_before_identity_resolution(monkeypatch):
    monkeypatch.setattr(
        "aisimulate_core._native.resolve_afd_decode_identity", lambda *args: pytest.fail("pricing resolved")
    )
    with pytest.raises(ValueError, match="does not support AFD worker type"):
        RunnerCapabilities(replay_spec_api_version=1).require_compatible(_spec())


@pytest.mark.parametrize("schema", [None, True, 1, 2.0, 3, "2"])
def test_afd_schema_is_validated_independently_of_replay_api_one(schema):
    spec = _spec()
    provenance = spec.backend_deployment.parallel_config["aggregated"]
    if schema is None:
        provenance.pop("schema_version")
    else:
        provenance["schema_version"] = schema
    with pytest.raises(ValueError, match="AFD topology schema"):
        RunnerCapabilities(supported_worker_types=("monolithic", "afd")).require_compatible(spec)


@pytest.mark.parametrize("mode", ["afd", "afd+pd"])
def test_replay_api_one_does_not_accept_obsolete_afd_deployment_modes(mode):
    spec = _spec()
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, deployment_mode=mode))
    with pytest.raises(ValueError, match="legacy AFD deployment modes"):
        RunnerCapabilities(supported_worker_types=("monolithic", "afd")).require_compatible(spec)


@pytest.mark.parametrize(
    "mode,afd_roles,total",
    [
        ("agg", ("aggregated",), 16),
        ("disagg", ("prefill",), 20),
        ("disagg", ("decode",), 20),
        ("disagg", ("prefill", "decode"), 32),
    ],
)
def test_qualification_accounts_for_every_role(mode, afd_roles, total):
    artifact = build_afd_qualification(_spec(mode, afd_roles))
    assert artifact["schema_version"] == 2
    plan = artifact["deployment_plan"]
    assert plan["gpu_accounting"]["total_gpus"] == total
    for role in afd_roles:
        assert plan["roles"][role]["schema_version"] == 2
        assert plan["roles"][role]["memory"]["phase"] == ("both" if role == "aggregated" else role)
    assert artifact["qualification"]["native_deployment_supported"] is False


@pytest.mark.parametrize("layout", ["four_stage", "two_stage"])
def test_role_preserves_pipeline_layout(layout):
    assert (
        build_afd_qualification(_spec(layout=layout))["deployment_plan"]["roles"]["aggregated"]["topology"][
            "pipeline_layout"
        ]
        == layout
    )


def test_pure_afd_qualification_matches_golden():
    rendered = json.dumps(build_afd_qualification(_spec()), indent=2, sort_keys=True) + "\n"
    assert rendered == (Path(__file__).parent / "golden" / "afd-qualification.json").read_text()


@pytest.mark.parametrize("mode,roles", [("agg", ("aggregated",)), ("disagg", ("prefill", "decode"))])
def test_qualification_normalizes_legacy_provider_without_mutating_saved_metadata(mode, roles):
    canonical = _spec(mode=mode, afd_roles=roles)
    saved = _spec(mode=mode, afd_roles=roles)
    for role in roles:
        saved.backend_deployment.performance_model_metadata[role]["provider"] = "aic-afd"
    wire = json.loads(json.dumps(asdict(saved)))
    wire["backend_deployment"] = BackendDeploymentSpec(**wire["backend_deployment"])
    restored = ReplaySpec(**wire)
    actual = build_afd_qualification(restored)
    expected = build_afd_qualification(canonical)
    # The fingerprint identifies the original saved input, including its spelling.
    assert actual["identity"].pop("replay_spec_sha256") != expected["identity"].pop("replay_spec_sha256")
    assert actual == expected
    for role in roles:
        assert restored.backend_deployment.performance_model_metadata[role]["provider"] == "aic-afd"


@pytest.mark.parametrize("schema", [None, True, "2", 2.0, 1, 3, 4, 5, 6])
def test_qualification_rejects_invalid_topology_schemas(schema):
    spec = _spec()
    spec.backend_deployment.parallel_config["aggregated"]["schema_version"] = schema
    with pytest.raises(AFDQualificationError, match="schema"):
        build_afd_qualification(spec)


def test_qualification_rejects_development_replay_api_two():
    with pytest.raises(AFDQualificationError, match="API version"):
        build_afd_qualification(replace(_spec(), api_version=2))


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_any_role_provenance_failure_rejects_the_whole_artifact(role):
    spec = _spec("disagg", ("prefill", "decode"))
    spec.backend_deployment.parallel_config[role]["gpu_accounting"]["afd_total_gpus"] = 1
    with pytest.raises(AFDQualificationError, match=role):
        build_afd_qualification(spec)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_missing_provider_identity_fails_for_either_role(role):
    spec = _spec("disagg", ("prefill", "decode"))
    spec.backend_deployment.performance_model_metadata[role]["measurement_required"] = True
    with pytest.raises(AFDQualificationError, match=role):
        build_afd_qualification(spec)


def test_routing_identity_conflict_is_not_normalized_away():
    spec = _spec()
    spec.backend_deployment.parallel_config["aggregated"]["topology"]["routing_location"] = "f"
    with pytest.raises(AFDQualificationError, match="provenance"):
        build_afd_qualification(spec)


def test_artifacts_are_deterministic_and_ordinary_replay_is_a_noop(tmp_path):
    paths = write_afd_qualification_artifacts(tmp_path, _spec("disagg", ("prefill", "decode")))
    before = [p.read_bytes() for p in paths]
    assert [
        p.read_bytes() for p in write_afd_qualification_artifacts(tmp_path, _spec("disagg", ("prefill", "decode")))
    ] == before
    assert write_afd_qualification_artifacts(tmp_path, _spec("agg", ())) is None


@pytest.mark.parametrize("schema", [1, 2.0])
def test_invalid_role_does_not_write_partial_artifacts(tmp_path, schema):
    spec = _spec("disagg", ("prefill", "decode"))
    spec.backend_deployment.parallel_config["decode"]["schema_version"] = schema
    with pytest.raises(AFDQualificationError):
        write_afd_qualification_artifacts(tmp_path, spec)
    assert list(tmp_path.iterdir()) == []


def test_current_saved_contract_qualifies_and_checks_accounting():
    spec = _spec()
    deployment = spec.backend_deployment
    provenance = deployment.parallel_config["aggregated"]
    assert provenance["schema_version"] == 2
    provenance["topology"].pop("f_gpus_per_node")
    # Fixture metadata and raw args deliberately share that same saved mapping.
    qualified = build_afd_qualification(spec)
    assert qualified["deployment_plan"]["roles"]["aggregated"]["topology"]["f_gpus_per_node"] == 8
    provenance["gpu_accounting"]["f_gpu_count"] = 4
    with pytest.raises(AFDQualificationError, match="provenance"):
        build_afd_qualification(spec)
