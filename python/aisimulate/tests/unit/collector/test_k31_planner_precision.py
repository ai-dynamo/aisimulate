# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic imported native proof; these tests do not qualify GPU execution."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from aisimulate_core.sdk.fpm_profile import load_fpm_profile
from collector.fpm_forward import planner, runtime_probe
from collector.fpm_forward.capabilities import resolve_model_capability
from collector.fpm_forward.runtime_instrumentation import load_instrumentation, runtime_binding
from collector.fpm_forward.runtime_observations import expected_ranks, validate_observations
from collector.fpm_forward.types import ParallelTopology

from .test_fpm_forward import _kimi_k31_config
from .test_fpm_profile_collection import no_models_or_timing_data  # noqa: F401
from .test_k31_runtime_observations import _complete_fixture
from .test_runtime_observations import _replace_launch, _write

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_models_or_timing_data")]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    index_path, launches = _complete_fixture(tmp_path, monkeypatch)
    config_path, _ = _kimi_k31_config(tmp_path)
    launch = launches["tp2"]
    launch["model_config"] = {"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    launch["topology"].update(dp=1, moe_tp=2, moe_ep=1)
    launch = runtime_probe.normalize_probe_launch(launch)
    launches["tp2"] = launch
    _replace_launch(index_path, launches)
    index = json.loads(index_path.read_text())
    attempt = index["configurations"]["tp2"]["attempts"][0]
    bundle = load_instrumentation(tmp_path / attempt["bundle"]["manifest"])
    phase = attempt["phases"]["decode"]
    context_path = tmp_path / phase["launch_manifest"]["path"]
    context = json.loads(context_path.read_text())
    context.update(expected_ranks=expected_ranks(launch["topology"]), instrumentation_binding=runtime_binding())
    phase["launch_manifest"]["sha256"] = _write(context_path, context)["sha256"]
    retained = []
    for ref in phase["artifacts"]:
        record_path = tmp_path / ref["path"]
        record = json.loads(record_path.read_text())
        if record["dp_rank"] != 0:
            continue
        record["model_config_sha256"] = launch["model_config"]["sha256"]
        record["resolved_config"]["parallel_config"].update(data_parallel_size=1, enable_expert_parallel=False)
        record["instrumentation_binding"] = {
            **runtime_binding(),
            **{key: record[key] for key in ("attempt_id", "configuration", "phase", "bundle_sha256")},
            "status": "verified",
            "expected_bundle_sha256": bundle.sha256,
            "hostname": "synthetic-host",
            "pid": 123,
            "root": "/frozen",
            "files": bundle.files,
            "imports": {
                "observer": {
                    "file": "observer.py",
                    "path": "/frozen/observer.py",
                    "sha256": bundle.files["observer.py"],
                }
            },
            "classes": {key: bundle.manifest[key] for key in ("worker_class", "scheduler_class")},
        }
        record["cpu_affinity_observation"] = {"status": "complete", "hostname": "synthetic-host", "pid": 123}
        ref["sha256"] = _write(record_path, record)["sha256"]
        retained.append(ref)
    phase["artifacts"] = retained
    index_path.write_text(json.dumps(index))
    result = validate_observations(index_path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    probe = runtime_probe.build_runtime_probe_plan("tp2", launch, bundle)
    capability = resolve_model_capability(
        backend="vllm",
        model_architecture=None,
        model_path=launch["identity"]["model"],
        system="gb300",
        selected_ops=set(),
        has_model_cases=False,
        requested_weight_quantizations=(),
        requested_kv_cache_dtypes=("fp8",),
        model_config_path=str(config_path),
        database_version=launch["identity"]["framework_version"],
        checkpoint_native_dtypes=True,
    )
    profile = {
        "schema_version": 1,
        "model": launch["identity"]["model"],
        "model_revision": launch["identity"]["model_revision"],
        "architecture": capability.architecture,
        "context_length": 4096,
        "num_experts": 8,
        "provenance": "Synthetic test only",
        "deployments": [
            {
                "system": "gb300",
                "backend": "vllm",
                "backend_version": launch["identity"]["framework_version"],
                **launch["topology"],
                **{key: value for key, value in launch["precision"].items() if key != "kvcache_quant_mode"},
                "kv_cache_dtype": launch["precision"]["kvcache_quant_mode"],
                "worker_type": "decode",
                "resources": result["resources"],
            }
        ],
    }
    return {
        "capability": capability,
        "profile": profile,
        "topologies": (ParallelTopology(**launch["topology"]),),
        "model_path": profile["model"],
        "system": "gb300",
        "backend": "vllm",
        "options": probe.options,
        "runtime_instrumentation": bundle,
        "runtime_launch": launch,
    }


def bind(inputs):
    return planner._observed_k31_capability(**{**inputs, "profile": load_fpm_profile(inputs["profile"])})


def edit_proof(inputs, mutation):
    memory = inputs["profile"]["deployments"][0]["resources"]["runtime_memory"]
    proof = json.loads(memory["provenance"])
    mutation(proof)
    memory["provenance"] = json.dumps(proof)


def test_valid_imported_native_proof_changes_only_moe_precision(inputs):
    original = inputs["capability"]
    assert original.dtype.moe_quant_mode == "w4a16_mxfp4"
    observed = bind(inputs)
    assert observed == replace(original, dtype=replace(original.dtype, moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm"))
    assert original.dtype.moe_quant_mode == "w4a16_mxfp4"


def test_formal_plan_uses_observed_precision_and_keeps_final_launch_validation(inputs, monkeypatch):
    calls = []
    validate = runtime_probe.validate_collection_probe_launch

    def record(*args, **kwargs):
        calls.append(True)
        return validate(*args, **kwargs)

    monkeypatch.setattr(runtime_probe, "validate_collection_probe_launch", record)
    plan = planner.build_collection_plan(
        backend=inputs["backend"],
        model_path=inputs["model_path"],
        system=inputs["system"],
        selected_ops=set(),
        options=inputs["options"],
        has_model_cases=False,
        model_config_path=inputs["runtime_launch"]["model_config"]["path"],
        fpm_profile=inputs["profile"],
        runtime_instrumentation=inputs["runtime_instrumentation"],
        runtime_launch=inputs["runtime_launch"],
        generator_overrides=runtime_probe.probe_generator_overrides(inputs["runtime_launch"]),
    )
    assert calls == [True]
    assert plan.cells and {cell.moe_quant_mode for cell in plan.cells} == {"w4a8_mxfp4_mxfp8_trtllm"}
    assert plan.runtime_instrumentation.sha256 == inputs["runtime_instrumentation"].sha256


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(source="declared"),
        lambda p: p.update(bundle_sha256="0" * 64),
        lambda p: p["runtime"].update(version="0.31.0"),
        lambda p: p["runtime"]["source_files"].pop(next(iter(p["runtime"]["source_files"]))),
        lambda p: p["launch"]["identity"].update(model_revision="different-checkpoint"),
        lambda p: p["launch"]["topology"].update(tp=1),
        lambda p: p["instrumentation"].update(import_binding="unverified_legacy"),
        lambda p: p["artifacts"].pop(0),
        lambda p: p["artifacts"].append(copy.deepcopy(p["artifacts"][0])),
        lambda p: p["artifacts"][0].update(sha256="not-a-hash"),
        lambda p: p["artifacts"][0]["evidence"].update(initialized_moe=None),
        lambda p: p["artifacts"][0]["evidence"]["initialized_moe"]["layers"][0]["descriptors"]["_a1"].update(
            dtype="bfloat16"
        ),
        lambda p: p["artifacts"][0]["evidence"].update(phase="prefill"),
        lambda p: p["artifacts"][0]["evidence"].update(unresolved_fields=["precision"]),
        lambda p: p["artifacts"][0]["evidence"]["lifecycle"].update(warmup_completed=False),
        lambda p: p["artifacts"][0]["evidence"]["instrumentation_binding"].update(status="unverified"),
        lambda p: p["rank_capacities"].pop(),
        lambda p: p["rank_capacities"].append(copy.deepcopy(p["rank_capacities"][0])),
        lambda p: p["rank_capacities"][0].update(kv_cache_bytes=9999),
    ],
)
def test_missing_stale_partial_or_relabelled_observations_cannot_override_precision(inputs, mutation):
    edit_proof(inputs, mutation)
    with pytest.raises(ValueError):
        bind(inputs)
    assert inputs["capability"].dtype.moe_quant_mode == "w4a16_mxfp4"


@pytest.mark.parametrize("field", ["runtime_instrumentation", "runtime_launch"])
def test_formal_binding_cannot_be_replaced_by_profile_label(inputs, field):
    inputs[field] = None
    with pytest.raises(ValueError, match="formal runtime instrumentation and launch"):
        bind(inputs)


def test_current_model_payload_is_bound_independently_of_public_name(inputs):
    capability = inputs["capability"]
    model = capability.model_config
    altered = dict(model.payload, test_only_changed_checkpoint=True)
    inputs["capability"] = replace(
        capability,
        model_config=type(model)(altered, source_kind=model.source_kind, source_reference=model.source_reference),
    )
    with pytest.raises(ValueError, match="current checkpoint"):
        bind(inputs)


def test_unobserved_checkpoint_native_identity_is_not_promoted(inputs):
    inputs["profile"]["deployments"][0]["moe_quant_mode"] = "w4a16_mxfp4"
    inputs["runtime_instrumentation"] = inputs["runtime_launch"] = None
    assert bind(inputs) is inputs["capability"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.update(max_num_tokens=2048), "resource envelope"),
        (lambda r: r.update(max_batch_size=32), "resource envelope"),
        (lambda r: r["runtime_memory"].update(max_model_len=8192), "resource envelope"),
        (lambda r: r["runtime_memory"].update(gpu_memory_utilization=0.8), "resource envelope"),
        (lambda r: r["cache_groups"][0].update(page_size_bytes=1024), "profile cache groups"),
        (lambda r: r["cache_groups"][-1].update(blocks_per_request=2), "profile cache groups"),
        (lambda r: r["cache_groups"].pop(), "profile cache groups"),
    ],
)
def test_resource_fields_cannot_be_relabelled_around_valid_precision_proof(inputs, mutation, match):
    mutation(inputs["profile"]["deployments"][0]["resources"])
    with pytest.raises(ValueError, match=match):
        bind(inputs)


def onboarding_wrapper(inputs):
    memory = inputs["profile"]["deployments"][0]["resources"]["runtime_memory"]
    proof = json.loads(memory["provenance"])
    index = Path(proof["observations_index"]["path"])
    (index.parent / "launch.json").write_text(json.dumps(inputs["runtime_launch"]))
    (index.parent / "model-config.json").write_bytes(
        Path(inputs["runtime_launch"]["model_config"]["path"]).read_bytes()
    )
    resources = copy.deepcopy(inputs["profile"]["deployments"][0]["resources"])
    resources.pop("provenance", None)
    resources["runtime_memory"].pop("provenance", None)
    wrapper = {
        "schema_version": "aisimulate-onboarding-runtime-probe/v1",
        "configuration": proof["configuration"],
        "checkpoint": str(index.parent / "checkpoint.json"),
        "model_metadata": {key: inputs["profile"][key] for key in ("architecture", "context_length", "num_experts")},
        "user_overrides": {},
        "observations_index": str(index),
        "source_artifacts": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(index.parent.rglob("*"))
            if path.is_file()
        ],
        "observed_resources": resources,
    }
    memory["provenance"] = json.dumps(wrapper)
    return wrapper


def test_canonical_onboarding_wrapper_revalidates_native_evidence(inputs):
    expected = bind(inputs)
    onboarding_wrapper(inputs)
    assert bind(inputs) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(configuration="missing"),
        lambda p: p["model_metadata"].update(num_experts=999),
        lambda p: p.update(user_overrides=None),
        lambda p: p.update(source_artifacts=[]),
        lambda p: p["source_artifacts"][0].update(sha256="0" * 64),
        lambda p: p["source_artifacts"].append(copy.deepcopy(p["source_artifacts"][0])),
        lambda p: p["observed_resources"]["runtime_memory"].update(kv_cache_bytes=9999),
    ],
)
def test_onboarding_wrapper_cannot_replace_immutable_native_proof(inputs, mutation):
    onboarding_wrapper(inputs)
    edit_proof(inputs, mutation)
    with pytest.raises(ValueError):
        bind(inputs)


def test_onboarding_wrapper_rejects_changed_source_even_if_inline_claim_is_valid(inputs):
    wrapper = onboarding_wrapper(inputs)
    index = Path(wrapper["observations_index"])
    index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError, match="source artifact changed"):
        bind(inputs)


def test_onboarding_wrapper_does_not_bypass_native_validator(inputs):
    onboarding_wrapper(inputs)
    inputs["runtime_launch"] = copy.deepcopy(inputs["runtime_launch"])
    inputs["runtime_launch"]["collection"]["warmup_iterations"] += 1
    with pytest.raises(ValueError, match="native evidence is incomplete"):
        bind(inputs)
