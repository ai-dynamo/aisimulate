# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every op plan is one expansion shared by every CUDA backend.

The tests run over every entry in ``OP_PLANS``, so moving an op onto a plan
needs no new test: each backend's getter must be the one-line plan call, the
backends may differ only by declared backend gaps, only whole regions are left
out, rows may not drop a case, and case strings keep the list form that case
IDs use.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from collector.capabilities import region_reason
from collector.op_plans import OP_PLANS, GemmCase, OpPlan, plan, summarize
from collector.provenance import case_plan_hash

pytestmark = pytest.mark.unit

PKG_ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("sglang", "trtllm", "vllm")
SMS = (80, 89, 90, 100, 103, 120)


@pytest.fixture(autouse=True)
def _full_plan(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)


def _region(op: str, case) -> dict:
    spec = OP_PLANS[op]
    return {"dtype": getattr(case, spec.dtype_field), **{field: getattr(case, field) for field in spec.region_fields}}


def _getters():
    for backend in BACKENDS:
        for entry in importlib.import_module(f"collector.{backend}.registry").REGISTRY:
            if entry.op in OP_PLANS:
                yield pytest.param(backend, entry, id=f"{entry.op}-{backend}")


@pytest.mark.parametrize(("backend", "entry"), list(_getters()))
def test_getter_is_the_one_line_plan_call(backend, entry):
    modules = [entry.module] if entry.module else [route.module for route in entry.versions]
    for module in modules:
        source = (PKG_ROOT / Path(*module.split("."))).with_suffix(".py").read_text()
        getter = next(
            node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == entry.get_func
        )
        assert ast.unparse(getter.body) == f"return plan_cases('{entry.op}', '{backend}', get_sm_version())"


@pytest.mark.parametrize("op", sorted(OP_PLANS))
@pytest.mark.parametrize("sm", SMS)
def test_backends_differ_only_by_declared_backend_gaps(op, sm):
    plans = {backend: plan(op, backend, sm) for backend in BACKENDS}
    union = set().union(*(set(op_plan.cases) for op_plan in plans.values()))
    for backend, op_plan in plans.items():
        for case in union - set(op_plan.cases):
            removal = region_reason(op, _region(op, case), backend, sm)
            assert removal is not None and removal[1].startswith("backend gap G-"), (backend, case)


@pytest.mark.parametrize("op", sorted(OP_PLANS))
@pytest.mark.parametrize("backend", BACKENDS)
def test_only_whole_regions_are_left_out(op, backend):
    expanded = OP_PLANS[op].expand()
    for sm in SMS:
        op_plan = plan(op, backend, sm)
        assert len(op_plan.cases) + sum(region.cases for region in op_plan.removed) == len(expanded)
        for region in op_plan.removed:
            assert region.reason.startswith(("op ", "dtype ", "backend gap G-")), region


@pytest.mark.parametrize("op", sorted(OP_PLANS))
def test_case_string_keeps_the_list_form_of_case_ids(op):
    case = OP_PLANS[op].expand()[0]

    assert str(case) == repr(case) == str(list(case))


@pytest.mark.parametrize("op", sorted(OP_PLANS))
def test_cases_sharing_a_module_build_are_contiguous(op):
    spec = OP_PLANS[op]
    cases = plan(op, "vllm", 100).cases
    assert len(cases) == len(set(cases))
    seen, previous = set(), None
    for case in cases:
        group = tuple(getattr(case, field) for field in spec.group_fields)
        if group != previous:
            assert group not in seen, f"group {group} is split"
            seen.add(group)
            previous = group


def test_rows_may_not_drop_a_case(monkeypatch):
    dropping = OpPlan(
        GemmCase,
        lambda: [GemmCase("bfloat16", 1, 128, 128)],
        dtype_field="gemm_type",
        group_fields=("n", "k"),
        rows={"vllm": lambda case, sm: []},
    )
    monkeypatch.setitem(OP_PLANS, "probe", dropping)

    with pytest.raises(ValueError, match="rows dropped case"):
        plan("probe", "vllm", 100)
    assert plan("probe", "sglang", 100).rows == [GemmCase("bfloat16", 1, 128, 128)]


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
def test_gemm_floors_remove_whole_dtypes(sm, dtypes):
    assert sorted({case.gemm_type for case in plan("gemm", "vllm", sm).cases}) == dtypes


def test_gemm_sglang_fp8_block_gap_covers_only_sm89_and_sm120():
    for sm in (89, 120):
        removed = plan("gemm", "sglang", sm).removed
        assert [(region.region, region.reason) for region in removed if region.region == "dtype=fp8_block"] == [
            ("dtype=fp8_block", "backend gap G-sglang-gemm-fp8-block-deepgemm-only")
        ]
    for sm in (90, 100, 103):
        assert "fp8_block" in {case.gemm_type for case in plan("gemm", "sglang", sm).cases}


def test_summary_counts_per_dtype_and_hashes_the_planned_ids():
    op_plan = plan("gemm", "sglang", 120)
    summary = summarize("gemm", op_plan, case_id=lambda row: f"sglang.gemm:run_gemm:{row}")
    shapes = summary["by_dtype"]["bfloat16"]

    assert summary["cases"] == 3 * shapes
    assert summary["by_dtype"] == {"bfloat16": shapes, "fp8": shapes, "nvfp4": shapes}
    assert summary["groups"] == len({(case.gemm_type, case.n, case.k) for case in op_plan.cases})
    assert summary["removed"] == [
        {"region": "dtype=fp8_block", "cases": shapes, "reason": "backend gap G-sglang-gemm-fp8-block-deepgemm-only"}
    ]
    assert summary["plan_hash"] == case_plan_hash([f"sglang.gemm:run_gemm:{case}" for case in op_plan.cases])
    assert "rows" not in summary


def test_plan_only_reports_op_plans_and_writes_the_table(tmp_path):
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
    shapes = gemm["by_dtype"]["bfloat16"]
    assert gemm["by_dtype"] == {"bfloat16": shapes, "fp8": shapes, "fp8_block": shapes}
    assert gemm["removed"] == [{"region": "dtype=nvfp4", "cases": shapes, "reason": "dtype nvfp4 requires SM>=100"}]
    rows = (tmp_path / "vllm_sm89_gemm.csv").read_text().splitlines()
    assert rows[0] == "gemm_type,m,n,k"
    assert len(rows) == 1 + gemm["cases"]
