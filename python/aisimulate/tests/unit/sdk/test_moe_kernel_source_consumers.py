# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact MoE source survives current CLI, Sweeper and Replay boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from aisimulate.compiler import _worker_engine_args
from aisimulate.config.engine import EnginePredictionConfig
from aisimulate.runner import _materialize_engine_role
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver
from aisimulate_core.sdk import RustForwardPassPerfModel
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError

pytestmark = pytest.mark.unit
SOURCE = "sglang_flashinfer_trtllm_moe"


@pytest.mark.parametrize("source", [None, SOURCE])
def test_cli_saved_config_and_replay_preserve_source(source):
    engine = EnginePredictionConfig(
        model="Qwen/Qwen3-30B-A3B",
        hardware="b200_sxm",
        backend="sglang",
        backend_version="0.5.17",
        moe_kernel_source=source,
        workers={"aggregated": {"kv_cache": {"capacity": {"type": "fixed", "blocks": 100}}}},
    )
    restored = EnginePredictionConfig.model_validate_json(engine.model_dump_json())
    assert restored.moe_kernel_source == source
    payload = _worker_engine_args(restored, restored.workers.aggregated, "aggregated", transfer_bytes_per_token=None)
    config = payload["timing_model"]["config"]
    assert config.get("moe_kernel_source") == source
    assert ("moe_kernel_source" in config) == (source is not None)
    role = _materialize_engine_role("sglang", "0.5.17", {}, payload, "aggregated")
    assert role["rank"]["timing_model"]["config"].get("moe_kernel_source") == source


@pytest.mark.parametrize("field", ["moe_kernel_source", "aic_moe_kernel_source"])
def test_flat_replay_source_reaches_timing(field):
    role = _materialize_engine_role(
        "sglang",
        "0.5.17",
        {},
        {"aic_model_path": "Qwen/Qwen3-30B-A3B", "aic_system": "b200_sxm", "num_gpu_blocks": 100, field: SOURCE},
        "aggregated",
    )
    assert role["rank"]["timing_model"]["config"]["moe_kernel_source"] == SOURCE
    assert field not in role["rank"]


def test_sweeper_source_is_round_tripped_resolved_and_cached(monkeypatch):
    space = SearchSpace(
        model_name="Qwen/Qwen3-30B-A3B", hardware_sku="b200_sxm", backend=["sglang"], moe_kernel_source=SOURCE
    )
    space = SearchSpace.model_validate_json(space.model_dump_json())
    resolver = ForwardPassEstimatorResolver(space)
    sample = {
        "backend": "sglang",
        "deployment_mode": "agg",
        "hardware_sku": "b200_sxm",
        "tp": 1,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "agg_block_size": 1,
    }
    calls = []

    def best_available(request):
        calls.append(request)
        resolved = request.to_dict() | {"backend_version": "0.5.17"}
        return SimpleNamespace(
            diagnostics=lambda: {"readiness": "ready", "provenance": {"config": resolved}}, close=lambda: None
        )

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", best_available)
    request = resolver._request(sample, "agg")
    assert request.moe_kernel_source == SOURCE
    assert resolver.resolve_candidate(sample)["agg"].config["moe_kernel_source"] == SOURCE
    resolver.resolve_candidate(sample)
    assert len(calls) == 1
    resolver._resolve(replace(request, moe_kernel_source=None), "agg")
    assert len(calls) == 2


def test_source_rejects_custom_timing_in_current_config():
    with pytest.raises(ValueError, match="engine model controls require default timing"):
        EnginePredictionConfig(
            model="Qwen/Qwen3-30B-A3B",
            hardware="b200_sxm",
            moe_kernel_source=SOURCE,
            workers={"aggregated": {"timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}}},
        )


def test_sweeper_capacity_preserves_source_through_model_config(monkeypatch):
    from aisimulate.sweeper import kv_estimate
    from aisimulate.sweeper.parallel_enum import ParallelShape
    from aisimulate.sweeper.search_space import _engine_memory_kwargs
    from aisimulate_core.sdk import memory

    space = SearchSpace(model_name="Qwen/Qwen3-30B-A3B", hardware_sku="b200_sxm", moe_kernel_source=SOURCE)
    captured = {}
    original = memory.build_model_config

    def build_config(**kwargs):
        captured["config"] = original(**kwargs)
        raise RuntimeError("stop before database access")

    monkeypatch.setattr(memory, "build_model_config", build_config)
    with pytest.raises(ValueError, match="stop before database access"):
        kv_estimate.estimate_kv_tokens(
            ParallelShape(1, 1, 1, 1),
            model_name=space.model_name,
            hardware_sku=space.hardware_sku,
            backend="sglang",
            backend_version="0.5.17",
            **_engine_memory_kwargs(space),
        )
    assert captured["config"].moe_kernel_source == SOURCE


def test_explicit_source_prevents_naive_capacity_fallback(monkeypatch):
    from aisimulate_core.sdk import memory

    def unavailable(*args, **kwargs):
        assert kwargs["moe_kernel_source"] == SOURCE
        raise RuntimeError("unavailable model")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", unavailable)
    with pytest.raises(ValueError, match="unavailable model"):
        memory.estimate_num_gpu_blocks(
            "Qwen/Qwen3-30B-A3B",
            "b200_sxm",
            "sglang",
            scheduler_block_size=1,
            max_num_tokens=1,
            max_batch_size=1,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            moe_kernel_source=SOURCE,
            allow_naive_fallback=True,
        )


@pytest.mark.parametrize("entry_point", ["compile", "native_memory", "memory_with_fallback"])
@pytest.mark.parametrize(
    ("model_path", "moe_backend", "message"),
    [
        ("Qwen/Qwen3-32B", None, "require an MoE model"),
        ("deepseek-ai/DeepSeek-V4-Pro", "megamoe", "MegaMoE"),
    ],
)
def test_unsupported_source_is_rejected_before_database_or_capacity_fallback(
    entry_point, model_path, moe_backend, message
):
    from aisimulate_core.sdk import engine, memory

    controls = {"moe_backend": moe_backend, "moe_kernel_source": SOURCE}
    with pytest.raises(InvalidEngineConfigurationError, match=message):
        if entry_point == "compile":
            engine.compile_engine(model_path, "missing-system", "sglang", **controls)
        elif entry_point == "native_memory":
            memory.KVCacheEstimator.from_request(
                model_path, "missing-system", "sglang", max_num_tokens=1, max_batch_size=1, **controls
            )
        else:
            memory.estimate_num_gpu_blocks(
                model_path,
                "missing-system",
                "sglang",
                scheduler_block_size=1,
                max_num_tokens=1,
                max_batch_size=1,
                memory_fraction_kind="of_total",
                memory_fraction_value=0.9,
                allow_naive_fallback=True,
                **controls,
            )


@pytest.mark.parametrize("estimation_mode", ["auto", "op_level"])
@pytest.mark.parametrize(
    ("model_path", "moe_backend", "message"),
    [
        ("Qwen/Qwen3-32B", None, "require an MoE model"),
        ("deepseek-ai/DeepSeek-V4-Pro", "megamoe", "MegaMoE"),
    ],
)
def test_canonical_source_validation_cannot_fall_back(estimation_mode, model_path, moe_backend, message):
    with pytest.raises(ValueError, match=f"invalid engine config:.*{message}"):
        RustForwardPassPerfModel.best_available(
            {
                "model": model_path,
                "system": "b200_sxm",
                "backend": "sglang",
                "worker_type": "aggregated",
                "estimation_mode": estimation_mode,
                "fallback_policy": "allow",
                "moe_backend": moe_backend,
                "moe_kernel_source": SOURCE,
            }
        )


def test_task_cannot_rewrite_an_explicit_moe_source_into_whole_forward_fpm():
    from aisimulate.sdk.task_v2 import Task
    from aisimulate_core.sdk.models import get_model

    task = Task(model_path="Qwen/Qwen3-30B-A3B", forward_model="fpm", moe_kernel_source=SOURCE)
    model_config = task.build_model_config(role="agg")
    with pytest.raises(InvalidEngineConfigurationError, match="moe_kernel_source.*forward_model='fpm'"):
        get_model(task.model_path, model_config, "sglang")


@pytest.mark.parametrize(
    ("forward_model", "message"),
    [
        ("op_level", "fpm_fmha_quant_mode requires forward_model='fpm'"),
        ("fpm", "moe_kernel_source is not supported with forward_model='fpm'"),
    ],
)
def test_independent_selectors_cannot_silently_discard_each_other(forward_model, message):
    from aisimulate_core.sdk.config_builders import build_model_config
    from aisimulate_core.sdk.models import get_model

    config = build_model_config(
        1,
        1,
        1,
        1,
        1,
        forward_model=forward_model,
        moe_kernel_source=SOURCE,
        fpm_fmha_quant_mode="fp8",
    )
    with pytest.raises(InvalidEngineConfigurationError, match=message):
        get_model("Qwen/Qwen3-30B-A3B", config, "sglang")
