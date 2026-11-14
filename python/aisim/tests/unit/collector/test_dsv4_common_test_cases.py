# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest
from collector import case_generator as common_test_cases
from collector.op_plans import plan, plan_cases

pytestmark = pytest.mark.unit


_FLASH = "deepseek-ai/DeepSeek-V4-Flash"
_PRO = "deepseek-ai/DeepSeek-V4-Pro"
_FLASH_FP8 = "sgl-project/DeepSeek-V4-Flash-FP8"
_PRO_FP8 = "sgl-project/DeepSeek-V4-Pro-FP8"
_SUPPORTED_MODELS = (_FLASH, _PRO, _FLASH_FP8, _PRO_FP8)


def test_dsv4_no_filter_expands_flash_and_pro_for_every_op(monkeypatch):
    """Full plans expand both default artifacts for the module and sparse ops and the
    top-k calibration, whose tables key the model geometry."""
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    module_cases = plan("dsv4_csa_context_module", "sglang", 100).cases
    assert module_cases
    assert {case.model_path for case in module_cases} == {_FLASH_FP8, _PRO_FP8}
    paged = plan("dsv4_paged_mqa_logits_module", "sglang", 100).cases
    assert {case.model_path for case in paged} == {_FLASH_FP8, _PRO_FP8}
    calib = plan("dsv4_csa_topk_calib", "sglang", 100)
    assert {case.model_path for case in calib.cases} == {_FLASH_FP8, _PRO_FP8}
    assert calib.removed == []


def test_dsv4_unrelated_model_filter_logs_the_skip(monkeypatch, capsys):
    """A non-DSV4 filter is a legitimate no-op for DSV4 getters, but the empty
    plan must be explainable from the log — a green job with an empty artifact
    is otherwise undiagnosable."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "not-a/dsv4-model")
    monkeypatch.setattr(sys, "argv", ["pytest"])
    assert plan("dsv4_csa_topk_calib", "sglang", 100).cases == []
    assert "is not a DSV4 model; generating no DSV4 cases" in capsys.readouterr().out


def test_dsv4_targeted_model_collects_its_own_calib(monkeypatch):
    """A targeted run collects the selected model's own calibration — the
    consumers key the DELTA per native geometry, so Pro calibration
    lands in its own bucket instead of overwriting Flash."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _PRO)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    assert plan("dsv4_csa_context_module", "sglang", 100).cases
    assert {case.model_path for case in plan("dsv4_csa_topk_calib", "sglang", 100).cases} == {_PRO}

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _FLASH_FP8)
    assert {case.model_path for case in plan("dsv4_csa_topk_calib", "sglang", 100).cases} == {_FLASH_FP8}


@pytest.mark.parametrize("model_path", _SUPPORTED_MODELS)
def test_dsv4_module_plans_keep_each_backends_projection_gemms(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    for sm in (90, 100, 103, 120):
        sglang = {case.gemm_type for case in plan("dsv4_csa_context_module", "sglang", sm).cases}
        # SGLang loads a native checkpoint's FP8 projections only with its FP4
        # experts, which need Blackwell.
        native_below_blackwell = sm == 90 and model_path.startswith("deepseek-ai/")
        assert sglang == ({"bfloat16"} if native_below_blackwell else {"bfloat16", "fp8_block"})
        assert {case.gemm_type for case in plan("dsv4_csa_context_module", "vllm", sm).cases} == {"fp8_block"}
        # TRT-LLM needs the fp8_ds_mla pool on SM120, which the module builds on SM90 only.
        trtllm = {case.gemm_type for case in plan("dsv4_csa_context_module", "trtllm", sm).cases}
        assert trtllm == (set() if sm == 120 else {"fp8_block"})
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
    # Every sequence length the chunk budget allows runs at long prefixes too.
    assert {s for _b, s, p in context if p >= 16384} >= {1, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16384}

    decode = {(c.batch_size, c.seq_len) for c in plan("dsv4_csa_generation_module", "vllm", 100).cases}
    assert (1, 1048575) in decode and (2, 262144) in decode and (4, 131072) in decode and (64, 8192) in decode
    assert (2, 524288) not in decode and (8, 131072) not in decode and (128, 8192) not in decode


def test_dsv4_paged_mqa_logits_plan_is_the_sparse_grid(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _FLASH)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--smoke"])

    cases = plan("dsv4_paged_mqa_logits_module", "vllm", 100).cases

    # The sparse lists within the 8192-token chunk and the 1M KV-token budget, the
    # short shapes included; --smoke samples rows in collect.py, not here.
    sweep = common_test_cases.get_dsv4_sweep()
    expected = {
        (b, s, p)
        for b in sweep.sparse_batch_sizes
        for s in sweep.sparse_input_lengths
        for p in sweep.sparse_past_kv_lengths
        if b * s <= 8192 and b * (s + p) <= 1048576
    }
    assert {(c.batch_size, c.seq_len, c.prefix_len) for c in cases} == expected
    assert (1, 1, 0) in expected
    assert {(c.tp_size, c.kernel, c.model_path) for c in cases} == {(1, "paged_mqa_logits", _FLASH)}


def test_dsv4_topk_calib_times_the_csa_module_points(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _FLASH_FP8)

    calib = {
        (c.batch_size, c.seq_len, c.prefix_len, c.variant) for c in plan("dsv4_csa_topk_calib", "sglang", 100).cases
    }
    context = {(c.batch_size, c.seq_len, c.prefix_len) for c in plan("dsv4_csa_context_module", "sglang", 100).cases}
    decode = {(c.batch_size, c.seq_len) for c in plan("dsv4_csa_generation_module", "sglang", 100).cases}
    # Prefill runs the v1 top-k, decode the v2 top-k with one new token after the KV length.
    assert calib == {(*point, "v1") for point in context} | {(b, 1, kv, "v2") for b, kv in decode}


def test_sglang_sparse_rows_carry_their_points_per_model_tp_and_batch(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    for op, kernel in (("dsv4_paged_mqa_logits_module", "paged_mqa_logits"), ("dsv4_csa_topk_calib", "topk")):
        op_plan = plan(op, "sglang", 90)
        rows = plan_cases(op, "sglang", 90)
        assert len({tuple(row[:4]) for row in rows}) == len(rows)
        assert {row[1] for row in rows} == {kernel} and {row[3] for row in rows} == {1}
        points = {(row[0], row[2], prefix, isl, variant) for row in rows for prefix, isl, variant in row[4]}
        assert points == {(c.model_path, c.batch_size, c.prefix_len, c.seq_len, c.variant) for c in op_plan.cases}


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
    assert plan("dsv4_paged_mqa_logits_module", "vllm", 100).cases == []
