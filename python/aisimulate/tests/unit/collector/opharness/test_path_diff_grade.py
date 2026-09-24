# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""path_diff.grade — the verdict rule as a pure function.

Each case below is a rule the gate has actually needed (dates in the source
comments): subset semantics, within-family kernel drift, the empty-capture
refusal, and the gemm-class infra-name grading.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, COMPONENTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


path_diff = _load("path_diff")

# a fixed toy taxonomy so the rule is tested, not the yaml
_LABELS = {
    "flash::FlashAttnFwdSm90": {"fa3"},
    "flash::FlashAttnFwdCombine": {"fa3"},
    "kernel_unified_attention": {"triton"},
    "deep_gemm::sm90_fp8_mqa_logits": {"dsa_nsa", "deepgemm"},
    "deep_gemm::sm90_fp8_paged_mqa_logits": {"dsa_nsa", "deepgemm"},
    "nvjet_sm90_tst_128x256": {"cublas"},
    "nvjet_sm90_tst_64x64": {"cublas"},
    "vllm::scaled_fp8_quant_kernel": {"vllm_kernel"},
    "per_token_group_quant_8bit_kernel": {"sgl_kernel"},
}


def label_kernels(kernels):
    labels, unmatched = set(), set()
    for k in kernels:
        if k in _LABELS:
            labels |= _LABELS[k]
        else:
            unmatched.add(k)
    return labels, unmatched


def backends(kernels):
    return label_kernels(kernels)[0]


def grade(col, srv, cap_error=None):
    return path_diff.grade(sorted(col), backends(col), set(srv), backends(srv), cap_error, label_kernels)


def test_subset_semantics_collector_one_op_serving_whole_model():
    col = ["flash::FlashAttnFwdSm90"]
    srv = ["flash::FlashAttnFwdSm90", "nvjet_sm90_tst_128x256", "per_token_group_quant_8bit_kernel"]
    g = grade(col, srv)
    assert g["verdict"] == "aligned" and g["only_col"] == [] and g["kernel_drift"] is None


def test_collector_only_signal_family_is_diverged():
    g = grade(["kernel_unified_attention"], ["flash::FlashAttnFwdSm90"])
    assert g["verdict"] == "diverged" and g["only_col"] == ["triton"]


def test_same_family_different_kernel_is_drift():
    # 0.29 DSA indexer: same dsa_nsa family, prefill vs paged logits kernel
    g = grade(["deep_gemm::sm90_fp8_mqa_logits"], ["deep_gemm::sm90_fp8_paged_mqa_logits"])
    assert g["verdict"] == "diverged"
    assert g["kernel_drift"]["dsa_nsa"]["collector_only_kernels"] == ["deep_gemm::sm90_fp8_mqa_logits"]


def test_crashed_capture_is_invalid_not_aligned():
    g = grade([], ["flash::FlashAttnFwdSm90"], cap_error="RuntimeError: boom")
    assert g["verdict"] == "invalid-capture"


def test_empty_capture_is_no_collector_signal():
    g = grade([], ["flash::FlashAttnFwdSm90"])
    assert g["verdict"] == "no-collector-signal"


def test_gemm_class_op_graded_on_infra_kernel_names():
    # cuBLAS IS the backend: aligned iff an instantiation serving launched appears
    g = grade(["nvjet_sm90_tst_128x256"], ["nvjet_sm90_tst_128x256", "flash::FlashAttnFwdSm90"])
    assert g["verdict"] == "aligned"
    assert g["infra_name_matches"] == {"cublas": ["nvjet_sm90_tst_128x256"]}


def test_gemm_class_op_without_shared_kernel_name_is_no_signal():
    # the compute_scale-vs-DeepSeek-V3.2 mispairing: both are quant kernels,
    # different ones (per-tensor scaled_fp8_quant vs block per_token_group_quant)
    g = grade(["vllm::scaled_fp8_quant_kernel"], ["per_token_group_quant_8bit_kernel"])
    assert g["verdict"] == "no-collector-signal" and g["infra_name_matches"] is None


def test_infra_families_never_count_as_signal():
    # serving ran a GEMM tile the collector did not: not a divergence
    g = grade(["flash::FlashAttnFwdSm90", "nvjet_sm90_tst_64x64"],
              ["flash::FlashAttnFwdSm90", "nvjet_sm90_tst_128x256"])
    assert g["verdict"] == "aligned"
