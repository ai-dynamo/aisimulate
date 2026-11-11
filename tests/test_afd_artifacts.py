# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Role-indexed AFD contract artifacts and fail-closed schema handling."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from aisimulate.afd_artifacts import AFDQualificationError, build_afd_qualification, write_afd_qualification_artifacts
from aisimulate.sweeper.afd_parallel import AFDTopology
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    EncoderPoolSpec,
    ReplaySpec,
    RunnerCapabilities,
    canonical_json,
    resolve_afd_replay_identity,
)
from scripts import afd_caliber_matrix


@pytest.mark.parametrize("fail_on", [None, 1])
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
        assert "frozen_decode_identity" not in path.read_text()
        calls.append(path)
        if len(calls) == fail_on:
            raise RuntimeError("prediction failed")
        return {"completed_requests": 1}

    monkeypatch.setattr(afd_caliber_matrix, "_variant", variant)
    monkeypatch.setattr(afd_caliber_matrix, "_run", run)
    try:
        if fail_on is None:
            assert afd_caliber_matrix.main() == 0
            assert len(calls) == 1
        else:
            with pytest.raises(RuntimeError, match="prediction failed"):
                afd_caliber_matrix.main()
        assert len(paths) == 1
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
    "workload",
    [
        {"isl": 16, "osl": 4},
        {"isl": 16, "osl": 4, "random_range_ratio": 0.5},
        {"source_type": "trace", "trace_path": "unused.jsonl"},
        {"source_type": "synthetic-session"},
        {"source_type": "agentic"},
    ],
)
def test_dynamic_identity_is_saved_without_a_pricing_switch(tmp_path, workload):
    spec = replace(_spec("disagg", ("prefill", "decode")), workload=workload)
    original = canonical_json(spec)
    assert resolve_afd_replay_identity(spec) is spec
    paths = write_afd_qualification_artifacts(tmp_path, spec)
    wire = json.loads(paths[0].read_text())
    wire["backend_deployment"] = BackendDeploymentSpec(**wire["backend_deployment"])
    restored = ReplaySpec(**wire)
    assert canonical_json(restored) == original
    assert "frozen_decode_identity" not in original
    assert build_afd_qualification(restored) == build_afd_qualification(spec)


@pytest.mark.parametrize("value", [None, True, False, 0, 1, "true", [], {}])
@pytest.mark.parametrize("location", ["estimator", "metadata"])
def test_every_retired_pricing_value_is_rejected_without_mutation(value, location, tmp_path):
    spec = _spec()
    metadata = spec.backend_deployment.performance_model_metadata["aggregated"]
    if location == "metadata":
        metadata["frozen_decode_identity"] = value
    else:
        metadata["estimator"] = {"estimator_config": {"afd": {"frozen_decode_identity": value}}}
    original = canonical_json(spec)
    with pytest.raises(ValueError, match="removed; regenerate"):
        resolve_afd_replay_identity(spec)
    with pytest.raises(AFDQualificationError, match="removed; regenerate"):
        write_afd_qualification_artifacts(tmp_path, spec)
    assert canonical_json(spec) == original
    assert not list(tmp_path.iterdir())


def test_monolithic_api_one_consumer_rejects_afd_before_identity_resolution():
    with pytest.raises(ValueError, match="does not support AFD worker type"):
        RunnerCapabilities(replay_spec_api_version=1).require_compatible(_spec())


@pytest.mark.parametrize("missing", ["prefill_tp", "prefill_pp", "prefill_attention_dp"])
def test_ordinary_companion_requires_complete_gpu_identity(missing):
    spec = _spec("disagg", ("decode",))
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, num_prefill_workers=1))
    spec.backend_deployment.parallel_config["prefill_replicas"] = 1
    # One ordinary TP2 worker plus sixteen AFD GPUs.
    assert build_afd_qualification(spec)["deployment_plan"]["gpu_accounting"]["total_gpus"] == 18
    del spec.backend_deployment.parallel_config[missing]
    with pytest.raises(AFDQualificationError, match="ordinary prefill GPU geometry"):
        build_afd_qualification(spec)


@pytest.mark.parametrize(
    "field,value",
    [("prefill_tp", 0), ("prefill_pp", -1), ("prefill_attention_dp", True), ("prefill_tp", 2.0), ("prefill_tp", "2")],
)
def test_ordinary_companion_rejects_invalid_gpu_identity(field, value):
    spec = _spec("disagg", ("decode",))
    spec.backend_deployment.parallel_config[field] = value
    with pytest.raises(AFDQualificationError, match="ordinary prefill GPU geometry"):
        build_afd_qualification(spec)


@pytest.mark.parametrize(
    "mode,afd_roles,field,role",
    [
        ("agg", ("aggregated",), "num_workers", "aggregated"),
        ("disagg", ("decode",), "num_prefill_workers", "prefill"),
        ("disagg", ("prefill",), "num_decode_workers", "decode"),
    ],
)
@pytest.mark.parametrize("count", [0, -1, True])
def test_qualification_requires_positive_worker_counts(mode, afd_roles, field, role, count):
    spec = _spec(mode, afd_roles)
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, **{field: count}))
    with pytest.raises(AFDQualificationError, match=f"{role} worker count must be positive"):
        build_afd_qualification(spec)


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
def test_qualification_requires_one_logical_worker_per_afd_pool(role):
    spec = _spec() if role == "aggregated" else _spec("disagg", ("prefill", "decode"))
    field = "num_workers" if role == "aggregated" else f"num_{role}_workers"
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, **{field: 2}))
    with pytest.raises(AFDQualificationError, match=f"{role} AFD pool requires one logical worker"):
        build_afd_qualification(spec)


@pytest.mark.parametrize("field,value", [("backend", "trtllm"), ("backend_version", ""), ("backend_version", None)])
def test_qualification_requires_resolved_sglang_backend(field, value):
    spec = _spec()
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, **{field: value}))
    with pytest.raises(AFDQualificationError, match="require sglang and a resolved backend_version"):
        build_afd_qualification(spec)


@pytest.mark.parametrize("vision", ["encoder", "images"])
def test_qualification_rejects_encoder_pools_and_images(vision):
    spec = _spec()
    if vision == "encoder":
        encoder = EncoderPoolSpec(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            system="h200_sxm",
            backend="sglang",
            backend_version="0.5.14",
            tp=1,
            batch_size=1,
            workers=1,
            latency_ms=1.0,
            throughput_rps=1000.0,
            memory_gib=1.0,
            rate_degradation=1.0,
            visual_tokens=256,
            image_height=224,
            image_width=224,
            image_count=1,
        )
        spec = replace(spec, backend_deployment=replace(spec.backend_deployment, encoder=encoder))
    else:
        spec = replace(spec, workload={**spec.workload, "images": {"count": 1, "height": 224, "width": 224}})
    with pytest.raises(AFDQualificationError, match="rejects encoder pools and images"):
        build_afd_qualification(spec)


@pytest.mark.parametrize("mode,roles,wrong_mode", [("agg", ("aggregated",), "disagg"), ("disagg", ("decode",), "agg")])
def test_qualification_rejects_afd_roles_outside_deployment_topology(mode, roles, wrong_mode):
    spec = _spec(mode, roles)
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, deployment_mode=wrong_mode))
    with pytest.raises(AFDQualificationError, match="AFD role is outside the configured deployment topology"):
        build_afd_qualification(spec)


def test_qualification_checks_expert_count_against_source_model():
    spec = _spec()
    deployment = spec.backend_deployment
    topology = AFDTopology.from_mapping({**deployment.agg_engine_args["afd"], "num_experts": 64})
    # Keep all saved topology identities consistent so only the source-model check can reject it.
    deployment.agg_engine_args["afd"] = topology.provenance()["topology"]
    deployment.parallel_config["aggregated"] = {"type": "afd", **topology.provenance()}
    deployment.performance_model_metadata["aggregated"]["config"] = topology.provenance()["topology"]
    with pytest.raises(AFDQualificationError, match="aggregated expert count disagrees with source model"):
        build_afd_qualification(spec)


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize("field", ["status", "phase"])
def test_qualification_requires_successful_memory_check_for_matching_role(role, field, tmp_path):
    spec = _spec() if role == "aggregated" else _spec("disagg", ("prefill", "decode"))
    memory = {
        name: {"status": "fits", "phase": "both" if name == "aggregated" else name}
        for name in spec.backend_deployment.afd_roles
    }
    artifact = build_afd_qualification(spec, memory=memory)
    assert artifact["deployment_plan"]["roles"][role]["memory"] == memory[role]
    memory[role][field] = "out_of_memory" if field == "status" else ("prefill" if role != "prefill" else "decode")
    with pytest.raises(AFDQualificationError, match=f"{role} requires a successful matching native memory check"):
        build_afd_qualification(spec, memory=memory)
    with pytest.raises(AFDQualificationError, match=f"{role} requires a successful matching native memory check"):
        write_afd_qualification_artifacts(tmp_path, spec, memory=memory)
    assert not list(tmp_path.iterdir())


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
