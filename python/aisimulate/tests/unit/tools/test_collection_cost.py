# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost accounting must not double-charge setup or lose unfinished reservations."""

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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


def test_grouped_token_extension_does_not_recharge_old_inner_cells():
    old = {"moe_type": "fp8", "num_tokens": "(16384, 32768)", "model_name": "model"}
    new = {**old, "num_tokens": "(16384, 32768, 65536)"}
    before = {cost.case_key(row) for row in cost.inner_cells(old)}
    after = {cost.case_key(row) for row in cost.inner_cells(new)}
    assert before <= after and len(after - before) == 1
    expanded = cost.inner_cells({"phase": "verify", "batch_sizes": "(1, 8)", "seq_lens": "(3, 5, 6)"})
    assert len(expanded) == 6 and expanded[0]["_task_case"]["batch_sizes"] == "(1, 8)"


def test_phase_parser_keeps_inclusive_time_and_does_not_sum_nested_bench(tmp_path):
    log = tmp_path / "log"
    log.write_text(
        "  [DSA point timing] decode b=32, kv=1048575: history=70.91s bench=0.05s bench_timed=0.01s total=75.43s\n"
    )
    points = cost.phase_timings(log)
    assert len(points) == 1 and points[0]["inclusive_seconds"] == 75.43
    assert points[0]["phase_seconds"]["history"] == 70.91
    assert points[0]["phase_seconds"]["bench_timed"] == 0.01


def test_shared_fixture_includes_standalone_mla_and_restores_baseline():
    path = _TOOL.with_name("collection_cost_inventory.py")
    spec = importlib.util.spec_from_file_location("collection_cost_inventory_test", path)
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    from collector import op_plans

    before_context = set(op_plans.plan("mla_context", "trtllm", 103).cases)
    before_generation = set(op_plans.plan("mla_generation", "trtllm", 103).cases)
    with inventory.shared_policy():
        expanded_context = set(op_plans.plan("mla_context", "trtllm", 103).cases)
        expanded_generation = set(op_plans.plan("mla_generation", "trtllm", 103).cases)
        assert before_context < expanded_context
        assert before_generation < expanded_generation
        assert any(case.input_len == 1048576 for case in expanded_context)
        assert any(case.input_len == 1048575 for case in expanded_generation)
    assert set(op_plans.plan("mla_context", "trtllm", 103).cases) == before_context
    assert set(op_plans.plan("mla_generation", "trtllm", 103).cases) == before_generation


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

    class Grouped(NamedTuple):
        phase: str
        batch_sizes: tuple
        seq_lens: tuple

    grouped = Plan([Grouped("verify", (1, 8), (2, 3, 4, 5, 6, 8))], [])
    selected = anchor.select_plan(
        grouped,
        [
            {"phase": "verify", "batch_sizes": "1", "seq_lens": "3", "dispatch": "", "_task_case": "attribution"},
            {"phase": "verify", "batch_sizes": "8", "seq_lens": "5", "dispatch": ""},
        ],
    )
    assert selected.cases == [Grouped("verify", (1,), (3,)), Grouped("verify", (8,), (5,))]
    with pytest.raises(ValueError, match="absent"):
        anchor.select_plan(grouped, [{"phase": "verify", "batch_sizes": "8", "seq_lens": "7", "dispatch": ""}])


def test_native_driver_uses_registry_arguments_and_retains_forward_latency(tmp_path, monkeypatch):
    path = _TOOL.with_name("collection_cost_anchor.py")
    spec = importlib.util.spec_from_file_location("collection_cost_anchor_run", path)
    anchor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anchor)

    class Case(NamedTuple):
        batch_size: int
        input_len: int

    @dataclass
    class Plan:
        cases: list
        dispatches: list
        rows: list

    calls = []

    def native(batch, length, *, perf_filename, device, skip_indexer):
        calls.append((batch, length, device, skip_indexer))
        Path(perf_filename).write_text("op_name,latency\ndsa_generation_module,0.7897\n")

    entry = SimpleNamespace(
        op="dsa_generation_module",
        module="native.module",
        run_func="native",
        unverified=False,
        unverified_sms=(),
        perf_filename="perf.txt",
        run_kwargs_dict={"skip_indexer": True},
        extra_perf_filenames=(),
    )
    modules = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                set_device=lambda n: None,
                get_device_name=lambda n: "NVIDIA B300 SXM6 AC",
                get_device_capability=lambda n: (10, 3),
                get_device_properties=lambda n: SimpleNamespace(total_memory=287428640768),
                synchronize=lambda: None,
            )
        ),
        "collector.op_plans": SimpleNamespace(plan=lambda *args, **kw: Plan([Case(32, 1048575)], [], [[32, 1048575]])),
        "tools.perf_database.collection_cost_inventory": SimpleNamespace(),
        "collector.sglang.registry": SimpleNamespace(REGISTRY=[entry]),
        "native.module": SimpleNamespace(native=native),
        "collector.collect": SimpleNamespace(
            _is_cuda_fatal_exception=lambda error, torch: isinstance(error, MemoryError)
        ),
        "collector.helper": SimpleNamespace(cuda_context_error=lambda device: None),
    }
    monkeypatch.setattr(anchor.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(anchor.importlib.metadata, "version", lambda name: "0.5.21")
    monkeypatch.setenv("AISIM_COLLECTOR_REF", "a" * 40)
    monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "true")
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "unrelated/inherited-model")
    monkeypatch.delenv("AIS_SM", raising=False)
    monkeypatch.delenv("AIC_SM", raising=False)
    request = {
        "collector_commit": "a" * 40,
        "backend": "sglang",
        "sm": 103,
        "gpu_name_contains": "NVIDIA B300",
        "package_version": "0.5.21",
        "scenario": "baseline",
        "op": "dsa_generation_module",
        "cases": [{"batch_size": "32", "input_len": "1048575", "dispatch": ""}],
    }
    result = anchor.run(request, tmp_path / "probe")
    assert result["status"] == "passed", result
    assert calls == [(32, 1048575, "cuda:0", True)]
    assert result["forward_rows"][0]["latency"] == "0.7897"
    assert result["calls"][0]["native_call_seconds"] >= 0
    assert result["effective_protocol"] == {"measure_power": False, "model_path": None, "coverage_scenario": "baseline"}
    entry.unverified = True
    rejected = anchor.run(request, tmp_path / "unverified")
    assert rejected["status"] == "failed" and "unverified" in rejected["error"]
    assert len(calls) == 1
    entry.unverified = False
    modules["collector.op_plans"].plan = lambda *args, **kw: Plan(
        [Case(32, 1048575), Case(64, 1048575)], [], [[32, 1048575], [64, 1048575]]
    )
    request["cases"].append({"batch_size": "64", "input_len": "1048575", "dispatch": ""})

    def failed_first(batch, length, **kwargs):
        if batch == 32:
            raise ValueError("ordinary failed shape")
        native(batch, length, **kwargs)

    modules["native.module"].native = failed_first
    recovered = anchor.run(request, tmp_path / "healthy-context")
    assert len(recovered["calls"]) == 2 and recovered["calls"][1]["status"] == "passed"
    modules["collector.helper"].cuda_context_error = lambda device: "sticky device error"
    poisoned = anchor.run(request, tmp_path / "poisoned-context")
    assert poisoned["worker_recycle_required"] is True
    assert len(poisoned["calls"]) == 1 and len(poisoned["unattempted_rows"]) == 1

    def allocation_failure(*args, **kwargs):
        raise MemoryError("allocation failed")

    modules["native.module"].native = allocation_failure
    modules["collector.helper"].cuda_context_error = lambda device: None
    oom = anchor.run(request, tmp_path / "oom-context")
    assert oom["worker_recycle_required"] is True and len(oom["calls"]) == 1
