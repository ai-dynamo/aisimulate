# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD-specific policy must not change ordinary model construction."""

from copy import deepcopy

import pytest

from aisimulate_core import _native
from aisimulate_core.sdk import common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.engine import _ops_json
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.models.afd import _get_afd_model_info, _resolve_afd_execution_quantization, build_afd_model
from aisimulate_core.sdk.models.helpers import _get_model_info, resolve_nvfp4_for_system
from aisimulate_core.sdk.perf_database import load_system_spec

pytestmark = pytest.mark.unit
GEMMA = "google/gemma-4-26B-A4B"


def test_gemma_shared_metadata_is_afd_local_and_does_not_change_ordinary_graph():
    public = _get_model_info(GEMMA)
    assert public["num_shared_experts"] == 0
    before = get_model(GEMMA, ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8), "sglang")
    expected = [_ops_json(getattr(before, phase)) for phase in ("context_ops", "generation_ops", "encoder_ops")]
    afd = _get_afd_model_info(GEMMA)
    assert afd is not public and afd["num_shared_experts"] == 1
    assert _get_model_info(GEMMA) is public and public["num_shared_experts"] == 0
    after = get_model(GEMMA, ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8), "sglang")
    assert [_ops_json(getattr(after, phase)) for phase in ("context_ops", "generation_ops", "encoder_ops")] == expected


@pytest.mark.parametrize("role,tp", [("a", 1), ("f", 8)])
def test_gemma_afd_keeps_shared_compute_on_a(role, tp):
    model = build_afd_model(GEMMA, ModelConfig(tp_size=tp, moe_tp_size=1, moe_ep_size=tp), "sglang", role)
    for phase in ("context_ops", "generation_ops"):
        names = [op._name for op in getattr(model, phase)]
        assert any("shared_mlp_gate_up" in name for name in names) is (role == "a")


@pytest.mark.parametrize("system,expected", [("h100_sxm", "nvfp4_wo"), ("b200_sxm", "nvfp4")])
def test_ordinary_and_afd_nvfp4_mapping_agree_on_supported_hardware(system, expected):
    ordinary = ModelConfig(gemm_quant_mode=common.GEMMQuantMode.nvfp4, moe_quant_mode=common.MoEQuantMode.nvfp4)
    afd = deepcopy(ordinary)
    resolve_nvfp4_for_system(ordinary, system)
    _resolve_afd_execution_quantization(afd, None, load_system_spec(system))
    assert ordinary.gemm_quant_mode.name == afd.gemm_quant_mode.name == expected
    assert ordinary.moe_quant_mode.name == afd.moe_quant_mode.name == expected


def test_strict_capability_rejection_is_afd_only():
    # Main's ordinary resolver does not reject FP8 based on SM capability.
    ordinary = ModelConfig(gemm_quant_mode=common.GEMMQuantMode.fp8, moe_quant_mode=common.MoEQuantMode.bfloat16)
    resolve_nvfp4_for_system(ordinary, "a100_sxm")
    assert ordinary.gemm_quant_mode == common.GEMMQuantMode.fp8
    with pytest.raises(ValueError, match="does not support FP8"):
        _resolve_afd_execution_quantization(deepcopy(ordinary), None, load_system_spec("a100_sxm"))
    spec = deepcopy(load_system_spec("h100_sxm"))
    spec["gpu"].pop("sm_version", None)
    with pytest.raises(ValueError, match="known GPU SM"):
        _resolve_afd_execution_quantization(ModelConfig(gemm_quant_mode=common.GEMMQuantMode.nvfp4), None, spec)


@pytest.mark.parametrize(
    "workload",
    [
        {"trace_path": "requests.jsonl"},
        {"source_type": "trace"},
        {"source_type": "synthetic-session"},
        {"source_type": "agentic"},
    ],
)
def test_trace_like_frozen_override_rejects_without_mutating_input(workload):
    import json

    original = deepcopy(workload)
    for explicit, expected in [(None, False), (False, False), (True, None)]:
        config = json.dumps({"afd": {"frozen_decode_identity": explicit}})
        if explicit:
            with pytest.raises(ValueError, match="use false or omit"):
                _native.resolve_afd_decode_identity(config, json.dumps(workload))
        else:
            assert _native.resolve_afd_decode_identity(config, json.dumps(workload)) is expected
    assert workload == original


def _mixed_prediction():
    return {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 3},
            "load": {"type": "concurrency", "concurrency": 2},
            "stop": {"requests": 4},
        },
        "engine": {
            "mode": "disaggregated",
            "model": "Qwen/Qwen3-30B-A3B",
            "hardware": "h200_sxm",
            "backend": "sglang",
            "backend_version": "0.5.14",
            "workers": {
                "prefill": {"parallelism": {"tensor": 1}, "timing": {"type": "fixed", "prefill_ms": 2, "decode_ms": 2}},
                "decode": {
                    "type": "afd",
                    "afd": {"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 8, "num_microbatches": 1},
                },
            },
        },
    }


def test_mixed_afd_companion_preserves_main_async_backend_validation():
    from aisimulate.config import CorePredictionConfig

    raw = _mixed_prediction()
    raw["engine"]["workers"]["prefill"]["scheduler"] = {"async_scheduling": True}
    with pytest.raises(ValueError, match="async_scheduling is supported only for backend=vllm"):
        CorePredictionConfig.model_validate(raw)


def test_mixed_afd_companion_rejects_native_pipeline_parallelism_before_pool_loading(monkeypatch):
    from aisimulate import runner
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig

    raw = _mixed_prediction()
    raw["engine"]["workers"]["prefill"]["parallelism"]["pipeline"] = 2

    def unreachable(*args, **kwargs):
        raise AssertionError("ordinary PP must reject before AFD pool preparation")

    monkeypatch.setattr(runner, "_materialize_afd_engine_role", unreachable)
    with pytest.raises(ValueError, match="pipeline parallelism"):
        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
        runner._materialize_engine_execution_spec(spec, trace_block_size=512, record_per_request=False)


def test_main_detokenization_control_rejects_role_based_afd():
    from aisimulate.config import CorePredictionConfig

    raw = _mixed_prediction()
    raw["engine"]["detokenization"] = {"base_ms": 1}
    with pytest.raises(ValueError, match="detokenization requires token replay"):
        CorePredictionConfig.model_validate(raw)
