# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash standalone per-op collection: case plan, routing and version gates."""

import ast
import json
import sys
import types
from pathlib import Path

import pytest
from collector.case_generator import get_common_moe_test_cases, get_gemm_case_specs, moe_model_allows_quantization
from collector.model_cases import build_collection_case_plan
from collector.sglang.registry import REGISTRY as SGLANG_REGISTRY
from collector.version_resolver import _check_compat, resolve_module

from .test_getter_deduplication import _install_vllm_stubs, _load_collector, _stub_module

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FP8 = "zai-org/GLM-5.3-Flash"
NVFP4 = "nvidia/GLM-5.3-Flash-NVFP4"
GLM_OFFGRID_WIDTHS = {16, 32, 160, 288, 6416, 12576, 24576, 24896, 38720, 77440, 154880}


def _compat(relative_path: str) -> str:
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    return next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__compat__" for t in node.targets)
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("model_path", [FP8, NVFP4])
def test_plan_collects_gemm_moe_and_compute_scale(backend, model_path):
    plan = build_collection_case_plan(backend=backend, model_path=model_path, gpu_type="gb300")
    assert plan.model_architecture == "Glm5NextForConditionalGeneration"
    assert plan.sm_version == 103
    assert set(plan.ops) == {"gemm", "moe", "compute_scale"}


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


@pytest.mark.parametrize(
    "model_path,backend,allowed,blocked",
    [
        (FP8, "vllm", "fp8_block", "nvfp4"),
        (FP8, "sglang", "fp8_block", "nvfp4"),
        (NVFP4, "vllm", "nvfp4", "fp8_block"),
        (NVFP4, "sglang", "nvfp4", "fp8_block"),
    ],
)
def test_moe_rows_are_quant_distinct_per_checkpoint(model_path, backend, allowed, blocked):
    assert moe_model_allows_quantization(backend, model_path, allowed)
    assert not moe_model_allows_quantization(backend, model_path, blocked)
    assert not moe_model_allows_quantization(backend, model_path, "bfloat16")
    assert not moe_model_allows_quantization("trtllm", model_path, allowed)


@pytest.mark.parametrize("model_path", [FP8, NVFP4])
def test_sglang_moe_rows_carry_serving_routing_and_pure_tp(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    cases = [c for c in get_common_moe_test_cases(backend="sglang") if c.model_name == model_path]
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


@pytest.mark.parametrize(
    "path",
    ["collector/vllm/collect_gemm.py", "collector/vllm/collect_moe.py", "collector/vllm/collect_computescale.py"],
)
@pytest.mark.parametrize(
    "version,accepted",
    [
        ("0.30.0", True),
        ("0.30.0+glm53tail.eb4704514fdf", True),
        ("0.28.0", False),
        ("0.29.0", False),
        ("0.30.1", False),
        ("0.24.0", True),
    ],
)
def test_vllm_collectors_admit_exactly_the_audited_0_30_0(path, version, accepted):
    assert _check_compat(_compat(path), version) is accepted


@pytest.mark.parametrize(
    "op,v2,v1",
    [
        ("gemm", "collector.sglang.collect_gemm_v2", "collector.sglang.collect_gemm_v1"),
        ("compute_scale", "collector.sglang.collect_computescale_v2", "collector.sglang.collect_computescale_v1"),
    ],
)
def test_sglang_0_5_20_routes_to_the_serving_built_fork(op, v2, v1):
    entry = next(e for e in SGLANG_REGISTRY if e.op == op)
    assert resolve_module(entry, "0.5.20") == v2
    assert resolve_module(entry, "0.5.17") == v1
    assert resolve_module(entry, "0.5.14") == v1
    v2_compat = _compat(v2.replace(".", "/") + ".py")
    v1_compat = _compat(v1.replace(".", "/") + ".py")
    assert _check_compat(v2_compat, "0.5.20")
    assert not _check_compat(v2_compat, "0.5.17")
    assert not _check_compat(v1_compat, "0.5.20")


def test_sglang_moe_admits_0_5_20_but_not_unaudited_releases():
    compat = _compat("collector/sglang/collect_moe.py")
    for version, accepted in [
        ("0.5.14", True),
        ("0.5.17", True),
        ("0.5.18", False),
        ("0.5.19", False),
        ("0.5.20", True),
    ]:
        assert _check_compat(compat, version) is accepted, version


def _load_sglang_gemm_v2(monkeypatch):
    _stub_module(monkeypatch, "torch", bfloat16="bfloat16", float32="float32")
    helper = types.ModuleType("collector.helper")
    helper.benchmark_with_power = None
    helper.log_perf = None
    helper.get_sm_version = lambda: 103
    monkeypatch.setitem(sys.modules, "collector.helper", helper)
    return _load_collector(monkeypatch, "collector.sglang.collect_gemm_v2", "collector/sglang/collect_gemm_v2.py")


def test_sglang_v2_queues_small_shapes_the_v1_fixme_skipped(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", FP8)
    module = _load_sglang_gemm_v2(monkeypatch)
    cases = module.get_gemm_test_cases()
    types_by_shape = {}
    for gemm_type, x, n, k in cases:
        types_by_shape.setdefault((n, k), set()).add(gemm_type)
    assert types_by_shape[(288, 4096)] == {"bfloat16", "fp8", "fp8_block", "nvfp4"}
    assert types_by_shape[(32, 4096)] == {"bfloat16", "fp8", "fp8_block", "nvfp4"}


@pytest.mark.parametrize(
    "traced,label",
    [
        ("gemm.fp8_scaled_mm:aot", "sglang_sgl_kernel_fp8_scaled_mm"),
        ("gemm.fp8_scaled_mm:torch", "sglang_torch_scaled_mm"),
        ("", "sglang_triton_scaled_mm"),
    ],
)
def test_sglang_v2_fp8_label_comes_from_the_executed_fused_op(monkeypatch, traced, label):
    module = _load_sglang_gemm_v2(monkeypatch)
    assert module._fp8_kernel_source(traced) == label


def test_sglang_v2_fp8_label_rejects_unknown_fused_ops(monkeypatch):
    module = _load_sglang_gemm_v2(monkeypatch)
    with pytest.raises(RuntimeError, match="unexpected fused ops"):
        module._fp8_kernel_source("gemm.fp8_scaled_mm:aot+gemm.other:jit")


def test_sglang_nvfp4_allowlist_admits_glm53flash_nvfp4_only():
    import yaml

    base = yaml.safe_load((REPO_ROOT / "collector/cases/base_ops/moe.yaml").read_text())
    modes = {m["name"]: m for m in base["common_case_values"]["moe_sglang"]["quantization_modes"]}
    assert NVFP4 in modes["nvfp4"]["allowed_model_paths"]
    assert FP8 not in modes["nvfp4"]["allowed_model_paths"]


def test_glm_case_file_declares_no_wideep_and_no_unregistered_ops():
    import yaml

    data = yaml.safe_load(
        (REPO_ROOT / "collector/cases/models/Glm5NextForConditionalGeneration_cases.yaml").read_text()
    )
    assert all(not row.get("wideep") for row in data["model_case_values"]["moe"])
    assert json.dumps(data).count("glm53flash_module") == 0
