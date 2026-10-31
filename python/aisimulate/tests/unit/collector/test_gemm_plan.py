# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GEMM plan is one expansion shared by every CUDA backend.

The three backends queue the same cases on an SM, except whole dtype regions
that capabilities.yaml removes (hardware floors, backend gaps). The getters
only call the shared plan; they never filter per framework.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from collector.capabilities import _load_backend_gaps
from collector.case_generator import (
    GemmCase,
    get_gemm_case_specs,
    get_gemm_cases,
    get_gemm_type_specs,
    plan_gemm,
    summarize_typed_plan,
)
from collector.provenance import case_plan_hash

pytestmark = pytest.mark.unit

PKG_ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("sglang", "trtllm", "vllm")
SMS = (80, 89, 90, 100, 103, 120)


@pytest.fixture(autouse=True)
def _full_plan(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)


@pytest.mark.parametrize("sm", SMS)
def test_backends_differ_only_by_declared_backend_gaps(sm):
    plans = {backend: set(get_gemm_cases(backend, sm)) for backend in BACKENDS}
    union = set().union(*plans.values())
    for backend, cases in plans.items():
        gap_dtypes = {
            gap.dtype for gap in _load_backend_gaps() if (gap.backend, gap.op) == (backend, "gemm") and sm in gap.sms
        }
        assert {case.gemm_type for case in union - cases} <= gap_dtypes, backend


@pytest.mark.parametrize(
    ("sm", "dtypes"),
    [
        (80, ["bfloat16"]),
        (89, ["bfloat16", "fp8", "fp8_block"]),
        (90, ["bfloat16", "fp8", "fp8_block"]),
        (100, ["bfloat16", "fp8", "fp8_block", "nvfp4"]),
        (120, ["bfloat16", "fp8", "fp8_block", "nvfp4"]),
    ],
)
def test_hardware_floors_remove_whole_dtypes(sm, dtypes):
    cases, removed = plan_gemm("vllm", sm)
    shapes = len(get_gemm_case_specs())

    assert sorted({case.gemm_type for case in cases}) == sorted(dtypes)
    assert len(cases) == shapes * len(dtypes)
    assert sorted(region.dtype for region in removed) == sorted(set(get_gemm_type_specs()) - set(dtypes))
    assert all(region.cases == shapes and region.reason.startswith("dtype ") for region in removed)


def test_sglang_fp8_block_gap_covers_only_sm89_and_sm120():
    for sm in (89, 120):
        cases, removed = plan_gemm("sglang", sm)
        assert "fp8_block" not in {case.gemm_type for case in cases}
        assert [region.reason for region in removed if region.dtype == "fp8_block"] == [
            "backend gap G-sglang-gemm-fp8-block-deepgemm-only"
        ]
    for sm in (90, 100, 103):
        assert "fp8_block" in {case.gemm_type for case in get_gemm_cases("sglang", sm)}


def test_case_string_keeps_the_list_form_of_case_ids():
    case = GemmCase("fp8", 64, 4096, 7168)

    assert str(case) == repr(case) == "['fp8', 64, 4096, 7168]"
    assert str(case) == str(["fp8", 64, 4096, 7168])
    gemm_type, m, n, k = case
    assert (gemm_type, m, n, k) == (case.gemm_type, case.m, case.n, case.k)


def test_plan_is_weight_shape_major_with_m_descending():
    cases = get_gemm_cases("trtllm", 100)

    assert len(cases) == len(set(cases))
    weights = [(case.n, case.k) for case in cases]
    assert weights == sorted(weights, reverse=True)
    seen_groups = set()
    previous = None
    for case in cases:
        group = (case.gemm_type, case.n, case.k)
        if previous is not None and group == (previous.gemm_type, previous.n, previous.k):
            assert case.m < previous.m
        else:
            assert group not in seen_groups, f"group {group} is split"
            seen_groups.add(group)
        previous = case


@pytest.mark.parametrize("backend", BACKENDS)
def test_getter_only_calls_the_shared_plan(backend):
    source_path = PKG_ROOT / "collector" / backend / "collect_gemm.py"
    source = source_path.read_text()
    getter = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_gemm_test_cases"
    )

    assert ast.unparse(getter.body) == f"return get_gemm_cases('{backend}', get_sm_version())"
    assert "AIC_COLLECT_GEMM_TYPES" not in source


def test_summary_counts_per_key_and_hashes_the_planned_ids():
    cases, removed = plan_gemm("sglang", 120)
    summary = summarize_typed_plan("gemm", cases, removed, case_id=lambda case: f"sglang.gemm:run_gemm:{case}")
    shapes = len(get_gemm_case_specs())

    assert summary["cases"] == 3 * shapes
    assert summary["by_key"] == {"bfloat16": shapes, "fp8": shapes, "nvfp4": shapes}
    assert summary["groups"] == 3 * len({(spec.n, spec.k) for spec in get_gemm_case_specs()})
    assert summary["removed"] == [
        {"dtype": "fp8_block", "cases": shapes, "reason": "backend gap G-sglang-gemm-fp8-block-deepgemm-only"}
    ]
    assert summary["plan_hash"] == case_plan_hash([f"sglang.gemm:run_gemm:{case}" for case in cases])


def test_plan_only_reports_typed_plan_and_writes_the_table(tmp_path):
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(PKG_ROOT / "collector" / "collect.py"),
            *("--backend", "vllm", "--model-cases-full", "--sm", "89", "--ops", "gemm"),
            *("--plan-only", "--plan-table", str(tmp_path)),
        ],
        cwd=PKG_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    gemm = json.loads(result.stdout)["plans"]["gemm"]
    shapes = len(get_gemm_case_specs())
    assert gemm["by_key"] == {"bfloat16": shapes, "fp8": shapes, "fp8_block": shapes}
    assert gemm["removed"] == [{"dtype": "nvfp4", "cases": shapes, "reason": "dtype nvfp4 requires SM>=100"}]
    rows = (tmp_path / "vllm_sm89_gemm.csv").read_text().splitlines()
    assert rows[0] == "gemm_type,m,n,k"
    assert len(rows) == 1 + gemm["cases"]
