# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash standalone per-op collection: case plan, routing and version gates."""

import json
from pathlib import Path

import pytest
from collector.case_generator import (
    get_common_kda_test_cases,
    get_common_moe_test_cases,
    get_gemm_case_specs,
    get_sglang_moe_backend,
    moe_model_allows_quantization,
)
from collector.model_cases import build_collection_case_plan

from .test_getter_deduplication import _install_vllm_stubs, _load_collector

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FP8 = "zai-org/GLM-5.3-Flash"
NVFP4 = "nvidia/GLM-5.3-Flash-NVFP4"
# Per-rank widths below/off the base grid at TP1/2/4/8 (TP8: vLLM KDA in_proj
# 3*1024+8+256, SGLang KDA b_proj 8, lm_head 154880/8).
GLM_OFFGRID_WIDTHS = {8, 16, 32, 288, 3336, 6416, 12576, 19360, 24576, 24896, 38720, 77440, 154880}


@pytest.mark.parametrize(
    ("backend", "ops"),
    [
        ("vllm", {"gemm", "moe", "kda", "mhc_module"}),
        # SGLang 0.5.21 is not an audited GLM KDA / mHC runtime: scheduling them only logs failures
        ("sglang", {"gemm", "moe"}),
    ],
)
@pytest.mark.parametrize("model_path", [FP8, NVFP4])
def test_plan_collects_the_audited_glm_ops(backend, ops, model_path):
    plan = build_collection_case_plan(backend=backend, model_path=model_path, gpu_type="gb300")
    assert plan.model_architecture == "Glm5NextForConditionalGeneration"
    assert plan.sm_version == 103
    assert set(plan.ops) == ops


def test_trtllm_plan_has_no_glm_moe_lane():
    plan = build_collection_case_plan(backend="trtllm", model_path=FP8, gpu_type="gb300")
    assert "moe" not in plan.ops


@pytest.mark.parametrize("model_path", [FP8, NVFP4])
def test_gemm_widths_reach_both_checkpoints_with_the_full_token_sweep(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    cases = get_gemm_case_specs()
    keys = {(case.x, case.n, case.k) for case in cases}
    assert len(keys) == len(cases)
    token_counts = {case.x for case in cases}
    for n in GLM_OFFGRID_WIDTHS:
        assert {x for x, nn, k in keys if (nn, k) == (n, 4096)} == token_counts, n
    # The base grid (base_ops: [gemm]) is still part of the plan.
    assert (8192, 12288, 4096) in keys


@pytest.mark.parametrize("backend", ["sglang", "trtllm", "vllm"])
@pytest.mark.parametrize(
    "model_path,allowed,blocked",
    [
        (FP8, "fp8_block", "nvfp4"),
        (NVFP4, "nvfp4", "fp8_block"),
    ],
)
def test_moe_rows_are_quant_distinct_per_checkpoint(model_path, backend, allowed, blocked):
    assert moe_model_allows_quantization(backend, model_path, allowed)
    assert not moe_model_allows_quantization(backend, model_path, blocked)
    assert not moe_model_allows_quantization(backend, model_path, "bfloat16")


@pytest.mark.parametrize("model_path", [FP8, NVFP4])
def test_sglang_moe_rows_carry_serving_routing_and_pure_tp(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    cases = [c for c in get_common_moe_test_cases() if c.model_name == model_path]
    assert cases
    assert {(c.hidden_size, c.inter_size, c.topk, c.num_experts) for c in cases} == {(4096, 2048, 8, 288)}
    assert {2, 4} <= {c.tp for c in cases if c.ep == 1}
    for case in cases:
        assert case.sglang_moe_swiglu_limit == 10.0
        assert case.sglang_moe_scoring_func == "sigmoid"
        assert case.sglang_moe_routed_scaling_factor == 2.5
        assert case.sglang_moe_num_expert_group == 1 and case.sglang_moe_topk_group == 1
        assert case.sglang_moe_has_correction_bias is True
        backends = case.sglang_moe_backends
        mode = "fp8_block" if model_path == FP8 else "nvfp4"
        assert backends[mode][103] == "flashinfer_trtllm"
        assert get_sglang_moe_backend(case, mode, 100) == "flashinfer_trtllm"
        if model_path == FP8:
            # Observed SGLang 0.5.20 SM90 serving dispatch (H200, TP4, auto MoE runner).
            assert backends[mode][90] == "triton"
            assert get_sglang_moe_backend(case, mode, 90) == "triton"


@pytest.mark.parametrize("backend", ["sglang", "trtllm", "vllm"])
@pytest.mark.parametrize("model_path", [FP8, NVFP4])
def test_pure_tp_moe_shards_are_planned(monkeypatch, backend, model_path):
    # Which TP fits is per-system capacity (NVFP4 TP1 on B300/GB300, FP8 TP8 on
    # Hopper); every pure-TP shard 1/2/4/8 with EP1 must be planned.
    from collector.op_plans import plan

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    cases = [c for c in plan("moe", backend, 100).cases if c.model_name == model_path]
    assert {1, 2, 4, 8} <= {c.tp for c in cases if c.ep == 1}


@pytest.mark.parametrize("model_path", [FP8, NVFP4])
def test_kda_plan_covers_the_tp1_to_tp8_head_shards(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    cases = [c for c in get_common_kda_test_cases() if c.model_name == FP8]
    # 64 KDA heads: 64 local heads at TP1, 32 at TP2, 16 at TP4, 8 at TP8. One
    # physical case per shard; the NVFP4 artifact is a shape-only alias.
    assert {c.num_v_heads for c in cases} == {64, 32, 16, 8}
    assert all(c.num_k_heads == c.num_v_heads and c.head_k_dim == c.head_v_dim == 128 for c in cases)
    for heads in (64, 32, 16, 8):
        assert {c.phase for c in cases if c.num_v_heads == heads} == {"context", "generation", "verify"}


def test_vllm_moe_resolves_glm_routing_without_touching_other_models(monkeypatch):
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    for model_path in (FP8, NVFP4):
        config = module._resolve_moe_runtime_config(model_path, {})
        assert config["use_grouped_topk"] is True
        assert (config["num_expert_group"], config["topk_group"]) == (1, 1)
        assert config["use_routing_bias"] is True
        assert config["router_logits_float32"] is True
        assert config["scoring_func"] == "sigmoid"
        assert config["routed_scaling_factor"] == 2.5
        assert config["swiglu_limit"] == 10.0
        assert config["apply_routed_scale_to_output"] is False
        assert config["renormalize"] is True
        assert config["activation"] == "silu"
    # GLM-5.x DSA keeps its own contract: no clamp, scale applied to output.
    glm52 = module._resolve_moe_runtime_config("zai-org/GLM-5.2-FP8", {})
    assert glm52["swiglu_limit"] is None
    assert glm52["apply_routed_scale_to_output"] is True


def test_vllm_nvfp4_uses_modelopt_only_for_the_glm53flash_checkpoint(monkeypatch):
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    assert module._uses_modelopt_nvfp4_checkpoint(NVFP4) is True
    assert module._uses_modelopt_nvfp4_checkpoint(FP8) is False
    # Other NVFP4 checkpoints keep the CompressedTensors construction.
    assert module._uses_modelopt_nvfp4_checkpoint("nvidia/GLM-5.2-NVFP4") is False
    qc = module._load_checkpoint_quantization_config(NVFP4)
    assert (qc["quant_method"], qc["quant_algo"]) == ("modelopt", "NVFP4")


def test_glm53flash_moe_rows_declare_their_checkpoint_modes():
    from collector.case_generator import _model_case_values

    modes = {
        row["model_path"]: row.get("quantization_modes") for row in _model_case_values("moe", apply_model_filter=False)
    }
    assert modes[FP8] == ["fp8_block"]
    assert modes[NVFP4] == ["nvfp4"]


def test_glm_case_file_declares_no_wideep_and_no_unregistered_ops():
    import yaml

    data = yaml.safe_load(
        (REPO_ROOT / "collector/cases/models/Glm5NextForConditionalGeneration_cases.yaml").read_text()
    )
    assert all(not row.get("wideep") for row in data["model_case_values"]["moe"])
    assert json.dumps(data).count("glm53flash_module") == 0
