# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conclusions in the repo, evidence in a bundle (owner decision 2026-09-26):
verdict summaries carry identities, counts and — when red — the deciding
names, never kernel lists; the evidence bundle is content-addressed and
indexed."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, COMPONENTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _report(verdict):
    return {
        "verdict": verdict,
        "repo": "org/m",
        "framework": "vllm",
        "version": "0.29.0",
        "gate_name": "gemm_fp8_M",
        "target_roles": ["gemm"],
        "phase": None,
        "phase_scoped": False,
        "kv_dtype": "auto",
        "op_hint": "gemm",
        "capture_file": "facts/pathdiff/opcov_gemm_fp8.json",
        "capture_sha256": "abc",
        "capture_env": None,
        "capture_run_error": None,
        "serving_record": {"id": "70764220f4aa", "exec_fingerprint": "c04c55adbbf5", "evidence_status": "current"},
        "collector_backends": ["cutlass"],
        "serving_backends": ["cutlass", "fa3"],
        "collector_only_signal": ["gemm:cutlass"] if verdict != "aligned" else [],
        "kernel_drift": {"gemm": {"collector_only_kernels": ["k_wrong"], "serving_kernels": ["k_a", "k_b"]}}
        if verdict != "aligned"
        else None,
        "role_evidence": {"gemm": {"collector": ["k_wrong"], "serving": ["k_a", "k_b"], "matched": []}},
        "missing_roles": [],
        "aux_collector_only_roles": [],
        "serving_kernels_matched_in_collector": ["k_a"],
        "collector_unmatched_kernels": [],
    }


def test_verdict_summary_carries_identity_and_counts_not_kernel_lists():
    pdiff = _load("path_diff", "path_diff_layering")
    s = pdiff.summarize_report(_report("aligned"))
    assert s["serving_record"]["exec_fingerprint"] == "c04c55adbbf5" and s["capture_sha256"] == "abc"
    assert s["role_counts"] == {"gemm": {"collector": 1, "serving": 2, "matched": 0}}
    for k in ("role_evidence", "serving_kernels_matched_in_collector", "collector_unmatched_kernels", "kernel_drift"):
        assert k not in s
    # when red, the deciding names ARE part of the conclusion
    s = pdiff.summarize_report(_report("diverged"))
    assert s["kernel_drift"]["gemm"]["collector_only_kernels"] == ["k_wrong"]
    assert s["collector_only_signal"] == ["gemm:cutlass"]
