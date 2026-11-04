# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost accounting must not double-charge setup or lose unfinished reservations."""

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = pytest.mark.unit
_TOOL = Path(__file__).resolve().parents[3] / "tools/perf_database/collection_cost.py"
_SPEC = importlib.util.spec_from_file_location("collection_cost", _TOOL)
cost = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cost)


def table(directory, rows):
    directory.mkdir()
    with (directory / "vllm_sm103_attention_context.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch_size", "input_len", "dispatch"])
        writer.writeheader()
        writer.writerows(rows)


def test_delta_counts_old_lengths_with_new_batches_and_preserves_dispatch(tmp_path):
    old = [{"batch_size": "1", "input_len": "262144", "dispatch": "a"}]
    new = [
        *old,
        {"batch_size": "4", "input_len": "262144", "dispatch": "a"},
        {"batch_size": "1", "input_len": "262144", "dispatch": "b"},
    ]
    table(tmp_path / "old", old)
    table(tmp_path / "new", new)
    delta = cost.compare(tmp_path / "old", tmp_path / "new")
    result = next(iter(delta.values()))
    assert result["baseline"] == 1 and result["candidate"] == 3
    assert len(result["added"]) == 2 and not result["removed"]


def test_duplicate_or_mismatched_plan_tables_fail(tmp_path):
    row = {"batch_size": "1", "input_len": "262144", "dispatch": "a"}
    table(tmp_path / "duplicate", [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        cost.load_tables(tmp_path / "duplicate")


def test_reservations_bound_parallel_work_and_actual_time_releases_only_unused_budget(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    cost.reserve(ledger, "a", 60, 100)
    with pytest.raises(ValueError, match="exhausted"):
        cost.reserve(ledger, "b", 60, 100)
    cost.finish(ledger, "a", 20)
    cost.reserve(ledger, "b", 60, 100)
    with pytest.raises(ValueError, match="already reserved"):
        cost.reserve(ledger, "b", 1, 100)
    with pytest.raises(ValueError, match="exhausted"):
        cost.reserve(ledger, "c", 30, 100)
    assert len(cost.read_ledger(ledger)) == 3


def test_collection_wrapper_records_failed_time_and_never_imports_allocation_seconds(tmp_path):
    spec = {
        "argv": [sys.executable, "-c", "raise SystemExit(7)"],
        "cwd": str(tmp_path),
        "gpus": 4,
        "timeout_seconds": 5,
        "allocation_seconds": 1000,
    }
    result = cost.run(spec, tmp_path / "run")
    assert result["status"] == "failed" and result["exit_code"] == 7
    assert result["collection_gpu_hours"] == pytest.approx(result["collection_seconds"] * 4 / 3600)
    assert result["allocation_seconds_included"] is False
    assert json.loads((tmp_path / "run/result.json").read_text()) == result


def test_sampling_keeps_precision_dispatch_and_both_work_extremes():
    rows = [
        {"batch_size": "1", "input_len": str(i), "dispatch": dispatch, "kv_cache_dtype": dtype}
        for dispatch in ("a", "b")
        for dtype in ("fp8", "bfloat16")
        for i in range(1, 30)
    ]
    result = cost.select_anchors({"table": {"added": rows}}, 7)["table"]
    for dispatch in ("a", "b"):
        for dtype in ("fp8", "bfloat16"):
            group = [r for r in result if r["case"]["dispatch"] == dispatch and r["case"]["kv_cache_dtype"] == dtype]
            assert len(group) == 7
            assert {"1", "29"} <= {r["case"]["input_len"] for r in group}
    assert cost.select_anchors({"table": {"added": list(reversed(rows))}}, 7)["table"] == result


def test_native_selection_keeps_only_requested_dispatch_and_refuses_missing_case():
    path = _TOOL.with_name("collection_cost_anchor.py")
    spec = importlib.util.spec_from_file_location("collection_cost_anchor", path)
    anchor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anchor)

    class Case(NamedTuple):
        batch_size: int
        input_len: int

    @dataclass
    class Plan:
        cases: list
        dispatches: list = field(default_factory=list)

    planned = Plan([Case(1, 262144), Case(4, 262144)], [("a", "b"), ("a",)])
    selected = anchor.select_plan(planned, [{"batch_size": "1", "input_len": "262144", "dispatch": "b"}])
    assert selected.cases == [Case(1, 262144)] and selected.dispatches == [("b",)]
    assert len(planned.cases) == 2
    with pytest.raises(ValueError, match="absent"):
        anchor.select_plan(planned, [{"batch_size": "8", "input_len": "262144", "dispatch": "a"}])
