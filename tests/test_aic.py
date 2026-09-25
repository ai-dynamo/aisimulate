# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aisimulate import capacity as aic

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    "version_fields",
    [
        {"backend_version": "test-version"},
        {"aic_backend_version": "test-version"},
        {"aic_backend_version": None, "backend_version": "test-version"},
    ],
)
def test_materializer_sets_rank_local_capacity_without_forwarding_nextn(
    monkeypatch,
    version_fields,
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
            **version_fields,
            "aic_system": "h200_sxm",
            "aic_model_path": "test-model",
            "aic_attention_dp_size": 2,
            "aic_pp_size": 3,
            "aic_nextn": 3,
            "cuda_graph_reserved_bytes": 14559947612,
            "systems_path": "/tmp/custom-systems.yaml",
            "block_size": 64,
            "max_num_seqs": 37,
        }
    )

    assert lowered["num_gpu_blocks"] == 46000
    assert lowered["dp_size"] == 2
    assert calls[0]["attention_dp_size"] == 2
    assert calls[0]["backend_version"] == "test-version"
    assert calls[0]["pp_size"] == 3
    assert calls[0]["systems_path"] == "/tmp/custom-systems.yaml"
    assert calls[0]["max_num_sequences"] == 37
    assert calls[0]["cuda_graph_reserved_bytes"] == 14559947612
    assert "nextn" not in calls[0]


def test_capacity_wrapper_owns_backend_defaults_and_quant_normalization(
    monkeypatch,
) -> None:
    calls = []

    def estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return 123

    from aisimulate_core.sdk import memory, perf_database

    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", estimate)
    version_calls = []

    def latest(system, backend, *, systems_paths):
        version_calls.append((system, backend, systems_paths))
        return "test-current"

    monkeypatch.setattr(perf_database, "get_latest_database_version", latest)
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
        cuda_graph_reserved_bytes=14559947612,
    )

    assert blocks == 123
    args, kwargs = calls[0]
    assert args == ("test-model", "h200_sxm", "vllm")
    assert kwargs["backend_version"] == "test-current"
    assert version_calls == [("h200_sxm", "vllm", "/tmp/custom-systems.yaml")]
    assert kwargs["memory_fraction_kind"] == "of_total"
    assert kwargs["memory_fraction_value"] == 0.9
    assert kwargs["pp_size"] == 3
    assert kwargs["max_batch_size"] == 17
    assert kwargs["systems_path"] == "/tmp/custom-systems.yaml"
    assert kwargs["cuda_graph_reserved_bytes"] == 14559947612
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
        "timing_model": {
            "type": "external",
            "provider": "aic",
            "config": {"systems_paths": ["/path/that/does/not/exist"]},
        },
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


def test_materialize_forwards_context_parallel_knobs(monkeypatch) -> None:
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    aic.materialize_aic_num_gpu_blocks(
        {
            "engine_type": "vllm",
            "aic_backend": "vllm",
            "aic_system": "h200_sxm",
            "aic_model_path": "test-model",
            "aic_cp_size": 2,
            "aic_dcp_size": 4,
            "block_size": 64,
        }
    )

    assert calls[0]["cp_size"] == 2
    assert calls[0]["dcp_size"] == 4


def test_materialize_defaults_context_parallel_knobs_to_one(monkeypatch) -> None:
    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(aic, "estimate_num_gpu_blocks", estimate)
    aic.materialize_aic_num_gpu_blocks(
        {
            "engine_type": "vllm",
            "aic_backend": "vllm",
            "aic_system": "h200_sxm",
            "aic_model_path": "test-model",
            "block_size": 64,
        }
    )

    assert calls[0]["cp_size"] == 1
    assert calls[0]["dcp_size"] == 1


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("timing", ["default", "fixed", "polynomial"])
def test_prediction_omitted_version_uses_current_database_for_capacity(monkeypatch, backend, timing):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate_core.sdk import perf_database

    expected = perf_database.get_version_slots("h200_sxm", backend)["current"]
    get_database = perf_database.get_database
    versions = []

    def recorded_database(*args, **kwargs):
        database = get_database(*args, **kwargs)
        versions.append(database.version)
        return database

    monkeypatch.setattr(perf_database, "get_database", recorded_database)
    timing_config = {"type": timing}
    if timing == "fixed":
        timing_config.update(prefill_ms=1, decode_ms=1)
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": "Qwen/Qwen3-32B-FP8",
                "hardware": "h200_sxm",
                "backend": backend,
                "context_length": 4096,
                "workers": {"aggregated": {"parallelism": {"tensor": 2}, "timing": timing_config}},
            }
        }
    )
    deployment = prediction_to_replay_spec(config).backend_deployment
    args = aic.materialize_aic_num_gpu_blocks(deployment.agg_engine_args)
    assert args["num_gpu_blocks"] > 0
    assert versions and set(versions) == {expected}
    assert config.engine.backend_version is None
    if timing == "default":
        assert args["timing_model"]["config"]["backend_version"] == expected
    else:
        assert args["timing_model"]["type"] == timing


@pytest.mark.parametrize("version", ["current", "next", "0.24.0", "0.19.0"])
def test_capacity_preserves_explicit_backend_version(monkeypatch, version):
    from aisimulate_core.sdk import memory, perf_database

    monkeypatch.setattr(
        perf_database,
        "get_latest_database_version",
        lambda *_args, **_kwargs: pytest.fail("explicit versions must not be replaced"),
    )
    versions = []

    def estimate(*args, **kwargs):
        versions.append(kwargs["backend_version"])
        return 128

    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", estimate)
    assert (
        aic.estimate_num_gpu_blocks(
            backend_name="vllm",
            backend_version=version,
            system="h200_sxm",
            model_path="test-model",
            tp_size=1,
            block_size=64,
            max_num_batched_tokens=4096,
        )
        == 128
    )
    assert versions == [version]


def test_capacity_without_a_database_fails_before_estimation(monkeypatch):
    from aisimulate_core.sdk import memory, perf_database

    monkeypatch.setattr(perf_database, "get_latest_database_version", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        memory,
        "estimate_num_gpu_blocks",
        lambda *_args, **_kwargs: pytest.fail("missing versions must fail before estimation"),
    )
    with pytest.raises(ValueError, match="no perf database.*unsupported-test-gpu.*vllm"):
        aic.estimate_num_gpu_blocks(
            backend_name="vllm",
            system="unsupported-test-gpu",
            model_path="test-model",
            tp_size=1,
            block_size=64,
            max_num_batched_tokens=4096,
        )
