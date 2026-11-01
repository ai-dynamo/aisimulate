# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every op plan is one expansion shared by every CUDA backend.

The tests run over every entry in ``OP_PLANS``, so moving an op onto a plan
needs no new test: each backend's getter must be the one-line plan call, the
backends may differ only by skips.py entries that name them, every left-out
case is counted under its entry, rows may not drop a case, batch rows time
every kept case exactly once, and case strings keep the list form that case
IDs use.
"""

from __future__ import annotations

import ast
import csv
import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from collector import skips
from collector.case_generator import without_models
from collector.op_plans import OP_PLANS, GemmCase, OpPlan, _Run, _runners, plan, summarize
from collector.provenance import case_plan_hash

pytestmark = pytest.mark.unit

PKG_ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("sglang", "trtllm", "vllm")
SMS = (80, 89, 90, 100, 103, 120)


@pytest.fixture(autouse=True)
def _full_plan(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)


def _entry(**keys) -> dict:
    """The skips.py entry with these keys and values."""
    return next(entry for entry in skips.SKIPS if all(entry.get(key) == value for key, value in keys.items()))


def _registered_ops(backend: str) -> list[str]:
    """The ops with an op plan that this backend's registry collects."""
    registry = importlib.import_module(f"collector.{backend}.registry").REGISTRY
    return sorted({entry.op for entry in registry} & set(OP_PLANS))


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
def test_backends_differ_only_by_entries_that_name_them(op, sm):
    spec = OP_PLANS[op]
    plans = {backend: plan(op, backend, sm) for backend in BACKENDS}
    union = set().union(*(set(op_plan.cases) for op_plan in plans.values()))
    for backend, op_plan in plans.items():
        missing = union - set(op_plan.cases)
        if not missing:
            continue
        with without_models(skips.models(backend, op, sm)):
            expanded = set(spec.expand())
        # A model entry leaves the case out before cases merge ...
        if missing - expanded:
            assert skips.model_entries(backend, op, sm), (backend, sorted(missing - expanded, key=str)[:3])

        # ... a case entry leaves out more ...
        dtypes_of = spec.dtypes
        _kept, skipped = skips.split(missing & expanded, backend, op, sm, dtypes_of, dispatch_entries=False)
        assert all("backend" in entry for _case, entry, _region in skipped), backend
        # ... and entries that name a dispatch leave out the rest on every kernel it runs on.
        left = (missing & expanded) - {case for case, _entry, _region in skipped}
        if left:
            dispatch = spec.dispatch[backend]
            runs = [_Run(case, runner) for case in left for runner in _runners(dispatch(case, sm))]
            kept_runs, skipped_runs = skips.split(
                runs, backend, op, sm, lambda run: dtypes_of(run.case), dispatch_entries=True
            )
            assert not kept_runs, (backend, kept_runs[:3])
            assert all("backend" in entry for _run, entry, _region in skipped_runs), backend


@pytest.mark.parametrize("op", sorted(OP_PLANS))
@pytest.mark.parametrize("backend", BACKENDS)
def test_every_left_out_case_is_counted_under_its_entry(op, backend):
    expanded = OP_PLANS[op].expand()
    entries = {repr(entry) for entry in skips.SKIPS}
    for sm in SMS:
        op_plan = plan(op, backend, sm)
        assert len(op_plan.cases) + sum(region.cases for region in op_plan.removed) == len(expanded)
        for region in [*op_plan.removed, *op_plan.removed_runs]:
            assert region.reason in entries, region


@pytest.mark.parametrize("op", sorted(OP_PLANS))
def test_case_string_keeps_the_list_form_of_case_ids(op):
    case = OP_PLANS[op].expand()[0]
    listed = ast.literal_eval(str(case))

    assert str(case) == repr(case)
    assert isinstance(listed, list)
    assert [tuple(value) if isinstance(value, list) else value for value in listed] == list(case)


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
        rows={"vllm": lambda case, sm, runner: []},
    )
    monkeypatch.setitem(OP_PLANS, "probe", dropping)

    with pytest.raises(ValueError, match="rows dropped case"):
        _ = plan("probe", "vllm", 100).rows
    assert plan("probe", "sglang", 100).rows == [GemmCase("bfloat16", 1, 128, 128)]


def test_batch_rows_must_time_every_kept_case_exactly_once(monkeypatch):
    cases = [GemmCase("bfloat16", 1, 128, 128), GemmCase("bfloat16", 2, 128, 128)]

    def probe(batch_rows):
        op_plan = OpPlan(
            GemmCase, lambda: cases, dtype_field="gemm_type", group_fields=("n", "k"), batch_rows={"vllm": batch_rows}
        )
        monkeypatch.setitem(OP_PLANS, "probe", op_plan)
        return plan("probe", "vllm", 100)

    assert probe(lambda kept, sm: [([[case.m for case in kept], 128, 128], tuple(kept))]).rows == [[[1, 2], 128, 128]]
    for wrong in (
        lambda kept, sm: [(["m=1"], (kept[0],))],
        lambda kept, sm: [(["all"], tuple(kept)), (["m=1 again"], (kept[0],))],
        lambda kept, sm: [(["all"], (*kept, GemmCase("fp8", 1, 128, 128)))],
    ):
        with pytest.raises(ValueError, match="exactly once"):
            _ = probe(wrong).rows


def test_a_backend_declares_batch_rows_or_rows_not_both():
    with pytest.raises(ValueError, match="not both"):
        OpPlan(
            GemmCase,
            list,
            dtype_field="gemm_type",
            group_fields=("n", "k"),
            rows={"vllm": lambda case, sm, runner: [case]},
            batch_rows={"vllm": lambda cases, sm: []},
        )


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


def test_gemm_sglang_fp8_block_skip_covers_only_sm89_and_sm120():
    for sm in (89, 120):
        removed = plan("gemm", "sglang", sm).removed
        assert [(region.region, region.reason) for region in removed if region.region == "dtype=fp8_block"] == [
            ("dtype=fp8_block", repr(_entry(backend="sglang", op="gemm")))
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
        {"region": "dtype=fp8_block", "cases": shapes, "reason": repr(_entry(backend="sglang", op="gemm"))}
    ]
    assert summary["plan_hash"] == case_plan_hash([f"sglang.gemm:run_gemm:{case}" for case in op_plan.cases])
    assert "rows" not in summary


def test_plan_only_reports_op_plans_and_writes_the_table(tmp_path, monkeypatch):
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(PKG_ROOT / "collector" / "collect.py"),
            *("--backend", "vllm", "--model-cases-full", "--sm", "89", "--ops", *_registered_ops("vllm")),
            *("--plan-only", "--plan-table", str(tmp_path)),
        ],
        cwd=PKG_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    plans = json.loads(result.stdout)["plans"]
    assert set(plans) == set(_registered_ops("vllm"))
    gemm = plans["gemm"]
    shapes = gemm["by_dtype"]["bfloat16"]
    assert gemm["by_dtype"] == {"bfloat16": shapes, "fp8": shapes, "fp8_block": shapes}
    nvfp4 = repr(_entry(dtype=["nvfp4", "w4a8_mxfp4_mxfp8"]))
    assert gemm["removed"] == [{"region": "dtype=nvfp4", "cases": shapes, "reason": nvfp4}]
    rows = (tmp_path / "vllm_sm89_gemm.csv").read_text().splitlines()
    assert rows[0] == "gemm_type,m,n,k"
    assert len(rows) == 1 + gemm["cases"]
    for op in plans:
        assert len((tmp_path / f"vllm_sm89_{op}.csv").read_text().splitlines()) == 1 + plans[op]["cases"], op

    # A run records an {id, params} row (mHC) under its own id, so the plan hash covers those ids.
    from collector.op_plans import plan_cases
    from collector.provenance import case_plan_hash

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    mhc_ids = [row["id"] for row in plan_cases("mhc_module", "vllm", 89)]
    assert plans["mhc_module"]["plan_hash"] == case_plan_hash(mhc_ids)


def test_planning_imports_no_framework():
    """--plan-only runs on any CPU: no op plan imports SGLang, vLLM, TensorRT-LLM or torch."""
    script = """
import sys
for name in ("sglang", "vllm", "tensorrt_llm", "torch"):
    sys.modules[name] = None
from collector.op_plans import OP_PLANS, plan
for op in OP_PLANS:
    for backend in ("sglang", "trtllm", "vllm"):
        assert plan(op, backend, 100).cases, (op, backend)
"""
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PKG_ROOT, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_moe_plan_reports_the_runner_sglang_picks_on_each_sm():
    sm100 = plan("moe", "sglang", 100)
    assert sum(sm100.by_dispatch.values()) == len(sm100.cases)
    assert {"triton", "flashinfer_trtllm", "flashinfer_mxfp4"} <= set(sm100.by_dispatch)
    assert "marlin" in plan("moe", "sglang", 90).by_dispatch
    assert plan("moe", "vllm", 100).by_dispatch == {}


def test_plan_table_carries_the_dispatch_of_each_case(tmp_path):
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(PKG_ROOT / "collector" / "collect.py"),
            *("--backend", "sglang", "--model-cases-full", "--sm", "90", "--ops", "moe"),
            *("--plan-only", "--plan-table", str(tmp_path)),
        ],
        cwd=PKG_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    with (tmp_path / "sglang_sm90_moe.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    plans = json.loads(result.stdout)["plans"]["moe"]
    assert len(table) == plans["cases"]
    assert Counter(row["dispatch"] for row in table) == Counter(plans["by_dispatch"])
    # SGLang serves Kimi-K2.5 INT4 on SM90 through Marlin.
    assert {row["dispatch"] for row in table if row["model_name"] == "moonshotai/Kimi-K2.5"} == {"marlin"}
