# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk import memory, perf_database
from aisimulate import aic
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.runner import _materialize_engine_role
from aisimulate.sweeper import kv_estimate

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


def test_materializer_sets_rank_local_capacity_without_forwarding_nextn(
    monkeypatch,
) -> None:
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 46000

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    lowered = aic.materialize_aic_num_gpu_blocks(
        {
            "engine_type": "vllm",
            "aic_backend": "vllm",
            "aic_system": "h200_sxm",
            "aic_model_path": "test-model",
            "aic_attention_dp_size": 2,
            "aic_pp_size": 3,
            "aic_nextn": 3,
            "systems_path": "/tmp/custom-systems.yaml",
            "block_size": 64,
            "max_num_seqs": 37,
        }
    )

    assert lowered["num_gpu_blocks"] == 46000
    assert lowered["dp_size"] == 2
    assert calls[0]["attention_dp_size"] == 2
    assert calls[0]["pp_size"] == 3
    assert calls[0]["systems_path"] == "/tmp/custom-systems.yaml"
    assert calls[0]["max_num_sequences"] == 37
    assert "nextn" not in calls[0]


def test_capacity_wrapper_owns_backend_defaults_and_quant_normalization(
    monkeypatch,
) -> None:
    calls = []

    def estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return 123

    from aiconfigurator_core.sdk import memory

    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", estimate)
    blocks = aic.estimate_num_gpu_blocks(
        backend_name="vllm",
        system="h200_sxm",
        model_path="test-model",
        tp_size=1,
        block_size=64,
        max_num_batched_tokens=4096,
        max_num_sequences=17,
        pp_size=3,
        gemm_dtype="int4",
        fmha_dtype="auto",
        systems_path="/tmp/custom-systems.yaml",
    )

    assert blocks == 123
    args, kwargs = calls[0]
    assert args == ("test-model", "h200_sxm", "vllm")
    assert kwargs["backend_version"] == "0.19.0"
    assert kwargs["memory_fraction_kind"] == "of_total"
    assert kwargs["memory_fraction_value"] == 0.9
    assert kwargs["pp_size"] == 3
    assert kwargs["max_batch_size"] == 17
    assert kwargs["systems_path"] == "/tmp/custom-systems.yaml"
    assert kwargs["gemm_quant_mode"] == "int4_wo"
    assert kwargs["fmha_quant_mode"] is None
    assert "nextn" not in kwargs


def test_explicit_capacity_is_preserved_without_estimation(monkeypatch) -> None:
    monkeypatch.setattr(
        aic,
        "estimate_num_gpu_blocks",
        lambda **_kwargs: pytest.fail("explicit capacity must not be estimated"),
    )

    raw = {
        "aic_backend": "vllm",
        "aic_model_path": "test-model",
        "num_gpu_blocks": 17,
    }
    assert aic.materialize_aic_num_gpu_blocks(raw) == raw


def test_materializer_preserves_explicit_zero_values(monkeypatch) -> None:
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    aic.materialize_aic_num_gpu_blocks(
        {
            "aic_backend": "sglang",
            "aic_model_path": "test-model",
            "aic_tp_size": 0,
            "aic_attention_dp_size": 0,
            "block_size": 0,
            "max_num_batched_tokens": 0,
            "gpu_memory_utilization": 0.0,
            "mem_fraction_static": 0.0,
            "free_gpu_memory_fraction": 0.0,
        }
    )

    assert calls[0]["tp_size"] == 0
    assert calls[0]["attention_dp_size"] == 0
    assert calls[0]["block_size"] == 0
    assert calls[0]["max_num_batched_tokens"] == 0
    assert calls[0]["gpu_memory_utilization"] == 0.0
    assert calls[0]["mem_fraction_static"] == 0.0
    assert calls[0]["free_gpu_memory_fraction"] == 0.0


def _prediction_config(backend="vllm", mode="aggregated", **engine_overrides):
    roles = ("aggregated",) if mode == "aggregated" else ("prefill", "decode")
    return CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": "Qwen/Qwen3-32B-FP8",
                "hardware": "h200_sxm",
                "backend": backend,
                "mode": mode,
                "context_length": 4096,
                "workers": {role: {} for role in roles},
                **engine_overrides,
            }
        }
    )


@pytest.mark.parametrize(
    ("backend", "mode"),
    [
        ("vllm", "aggregated"),
        ("sglang", "aggregated"),
        ("trtllm", "aggregated"),
        ("vllm", "disaggregated"),
        ("sglang", "disaggregated"),
    ],
)
def test_prediction_pins_shipped_current_version_for_memory_and_timing(monkeypatch, backend, mode):
    # Read the actual bundled slot policy: a hard-coded raw version must not
    # become the implicit default again when maintained slots change.
    slots = perf_database.get_version_slots("h200_sxm", backend)
    expected = slots["current"]
    config = _prediction_config(backend, mode)
    memory_versions = []

    def estimate(*args, **kwargs):
        memory_versions.append(kwargs["backend_version"])
        return 128

    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", estimate)
    deployment = prediction_to_replay_spec(config).backend_deployment
    assert deployment.backend_version == expected
    assert config.engine.backend_version is None
    roles = (
        [("aggregated", deployment.agg_engine_args)]
        if mode == "aggregated"
        else [("prefill", deployment.prefill_engine_args), ("decode", deployment.decode_engine_args)]
    )
    for role, args in roles:
        assert args["aic_backend_version"] == expected
        assert deployment.performance_model_metadata[role]["config"]["backend_version"] == expected
        lowered = _materialize_engine_role(backend, deployment.backend_version, deployment.parallel_config, args, role)
        assert lowered["rank"]["timing_model"]["config"]["backend_version"] == expected
    assert memory_versions == [expected] * len(roles)


@pytest.mark.parametrize("version", ["current", "next", "0.24.0", "0.19.0"])
def test_prediction_preserves_explicit_backend_version(monkeypatch, version):
    monkeypatch.setattr(
        kv_estimate,
        "get_latest_database_version",
        lambda *_args, **_kwargs: pytest.fail("an explicit version must not be replaced"),
    )
    deployment = prediction_to_replay_spec(_prediction_config(backend_version=version)).backend_deployment
    assert deployment.backend_version == version
    assert deployment.agg_engine_args["aic_backend_version"] == version


@pytest.mark.parametrize(("default_timing", "default_capacity"), [(True, False), (False, True), (False, False)])
def test_prediction_only_resolves_version_when_a_database_is_needed(monkeypatch, default_timing, default_capacity):
    calls = []

    def latest(system, backend):
        calls.append((system, backend))
        return "0.24.0"

    monkeypatch.setattr(kv_estimate, "get_latest_database_version", latest)
    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", lambda *_args, **_kwargs: 128)
    worker = {}
    if not default_timing:
        worker["timing"] = {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}
    if not default_capacity:
        worker["kv_cache"] = {"capacity": {"type": "fixed", "blocks": 128}}
    deployment = prediction_to_replay_spec(_prediction_config(workers={"aggregated": worker})).backend_deployment
    needs_database = default_timing or default_capacity
    assert calls == ([("h200_sxm", "vllm")] if needs_database else [])
    assert deployment.backend_version == ("0.24.0" if needs_database else "")


def test_prediction_without_a_database_fails_before_execution():
    with pytest.raises(RuntimeError, match="no perf database.*unsupported-test-gpu"):
        prediction_to_replay_spec(_prediction_config(hardware="unsupported-test-gpu"))
