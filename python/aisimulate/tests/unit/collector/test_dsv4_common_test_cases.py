# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest
from collector import case_generator as common_test_cases
from collector.op_plans import plan

pytestmark = pytest.mark.unit


_FLASH = "deepseek-ai/DeepSeek-V4-Flash"
_PRO = "deepseek-ai/DeepSeek-V4-Pro"
_FLASH_FP8 = "sgl-project/DeepSeek-V4-Flash-FP8"
_PRO_FP8 = "sgl-project/DeepSeek-V4-Pro-FP8"
_SUPPORTED_MODELS = (_FLASH, _PRO, _FLASH_FP8, _PRO_FP8)


def test_dsv4_no_filter_expands_flash_and_pro_modules_calib_stays_canonical(monkeypatch, capsys):
    """Full/raw plans expand BOTH default artifacts for module/sparse ops —
    their tables key model geometry ([native][local] / native heads) since
    #1423/#1431 — while topk calib stays on the single canonical model as a
    collection-cost policy (since #1460 the consumers key the DELTA per
    native geometry, so this is no longer a correctness gate), with the
    default-plan drop logged."""
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    module_cases = plan("dsv4_csa_context_module", "sglang", 100).cases
    assert module_cases
    assert {case.model_path for case in module_cases} == {_FLASH_FP8, _PRO_FP8}
    assert common_test_cases.get_dsv4_paged_mqa_logits_test_cases() == [
        [_FLASH_FP8, "paged_mqa_logits"],
        [_PRO_FP8, "paged_mqa_logits"],
    ]
    capsys.readouterr()
    assert common_test_cases.get_dsv4_topk_calib_test_cases() == [[_FLASH_FP8, "topk"]]
    assert "default calibration stays on" in capsys.readouterr().out


def test_dsv4_unrelated_model_filter_logs_the_skip(monkeypatch, capsys):
    """A non-DSV4 filter is a legitimate no-op for DSV4 getters, but the empty
    plan must be explainable from the log — a green job with an empty artifact
    is otherwise undiagnosable (#1460 review)."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "not-a/dsv4-model")
    monkeypatch.setattr(sys, "argv", ["pytest"])
    assert common_test_cases.get_dsv4_topk_calib_test_cases() == []
    assert "is not a DSV4 model; generating no DSV4 cases" in capsys.readouterr().out


def test_dsv4_targeted_model_collects_its_own_calib(monkeypatch):
    """A targeted run collects the selected model's own calibration — since
    #1460 the consumers key the DELTA per native geometry, so Pro calibration
    lands in its own bucket instead of overwriting Flash (#1468)."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _PRO)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    assert plan("dsv4_csa_context_module", "sglang", 100).cases
    assert common_test_cases.get_dsv4_topk_calib_test_cases() == [[_PRO, "topk"]]

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _FLASH_FP8)
    assert common_test_cases.get_dsv4_topk_calib_test_cases() == [[_FLASH_FP8, "topk"]]


@pytest.mark.parametrize("model_path", _SUPPORTED_MODELS)
def test_dsv4_module_plans_keep_each_backends_projection_gemms(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    for sm in (90, 100, 103, 120):
        sglang = {case.gemm_type for case in plan("dsv4_csa_context_module", "sglang", sm).cases}
        # SGLang loads a native checkpoint's FP8 projections only with its FP4
        # experts, which need Blackwell.
        native_below_blackwell = sm == 90 and model_path.startswith("deepseek-ai/")
        assert sglang == ({"bfloat16"} if native_below_blackwell else {"bfloat16", "fp8_block"})
        for backend in ("trtllm", "vllm"):
            assert {case.gemm_type for case in plan("dsv4_csa_context_module", backend, sm).cases} == {"fp8_block"}
    for sm in (80, 86, 89):
        assert plan("dsv4_csa_context_module", "sglang", sm).cases == []


@pytest.mark.parametrize("model_path", _SUPPORTED_MODELS)
def test_dsv4_module_plans_honor_model_filter(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    cases = plan("dsv4_hca_generation_module", "sglang", 100).cases

    assert cases
    assert {case.model_path for case in cases} == {model_path}
    assert {case.attn_kind for case in cases} == {"hca"}


def test_dsv4_module_points_follow_the_long_context_and_decode_grids(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _PRO_FP8)

    context = {(c.batch_size, c.seq_len, c.prefix_len) for c in plan("dsv4_csa_context_module", "vllm", 100).cases}
    assert all(b * s <= 16384 and b * (s + p) <= 1048576 and s + p <= 1048576 for b, s, p in context)
    # The top-k knees of DeepSeek-V4-Flash (512) and DeepSeek-V4-Pro (1024) stay prefix anchors.
    assert {512, 1024} <= {p for _b, _s, p in context}
    # From a 16K prefix on, only chunk-sized isl.
    assert {s for _b, s, p in context if p >= 16384} == {1, 512, 2048, 8192, 16384}

    decode = {(c.batch_size, c.seq_len) for c in plan("dsv4_csa_generation_module", "vllm", 100).cases}
    assert (1, 1048575) in decode and (2, 262144) in decode and (4, 131072) in decode and (64, 8192) in decode
    assert (2, 524288) not in decode and (8, 131072) not in decode and (128, 8192) not in decode


def test_dsv4_sparse_smoke_cases_honor_model_filter(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _FLASH)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--smoke"])

    cases = common_test_cases.get_dsv4_paged_mqa_logits_test_cases()

    # Sparse kernels are now one case per model ([model_path, kernel]); the worker
    # derives the (prefix, isl, bs) shapes 1:1 from the owning module CSV at
    # runtime instead of a separate smoke sweep grid.
    assert cases == [[_FLASH, "paged_mqa_logits"]]


@pytest.mark.parametrize("model_path", (_FLASH_FP8, _PRO_FP8))
def test_dsv4_mhc_preserves_requested_fp8_artifact(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    cases = common_test_cases.get_common_mhc_test_cases()

    assert len(cases) == 2
    assert {case.model_name for case in cases} == {model_path}


def test_dsv4_cases_skip_unrelated_model_filter(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "MiniMaxAI/MiniMax-M2.5")
    monkeypatch.setattr(sys, "argv", ["pytest"])

    assert plan("dsv4_csa_context_module", "sglang", 100).cases == []
    assert common_test_cases.get_dsv4_paged_mqa_logits_test_cases() == []


def test_sglang_sparse_modules_topk_calib_uses_canonical_model(monkeypatch, capsys):
    """The sglang registry's topk-calib getter (deepseekv4_sparse_modules)
    applies the same calib model policy as the shared case_generator getter:
    default plans stay canonical (cost policy, logged), targeted runs collect
    the selected model's own calibration (#1460 per-native calib keys)."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "collector" / "sglang" / "deepseekv4_sparse_modules.py"
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    wanted = {
        "_dsv4_sparse_kernel_cases",
        "get_dsv4_topk_calib_test_cases",
        "get_dsv4_paged_mqa_logits_test_cases",
    }
    ns = {
        "_dsv4_context_derived_shapes": lambda _m: [(0, 1, 1)],
        "_dsv4_generation_derived_shapes": lambda _m: [(0, 1, 1)],
    }
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in body} == wanted
    exec(compile(ast.Module(body=body, type_ignores=[]), str(src), "exec"), ns)

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    # full/raw: sparse kernels expand both default models, calib only
    # canonical (cost policy), with the default-plan drop logged
    assert {c[0] for c in ns["get_dsv4_paged_mqa_logits_test_cases"]()} == {_FLASH_FP8, _PRO_FP8}
    capsys.readouterr()
    assert {c[0] for c in ns["get_dsv4_topk_calib_test_cases"]()} == {_FLASH_FP8}
    assert "default calibration stays on" in capsys.readouterr().out
    # targeted run: the selected model collects its own calibration (#1460)
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _PRO)
    assert ns["get_dsv4_topk_calib_test_cases"]() == [[_PRO, "topk", 1]]
