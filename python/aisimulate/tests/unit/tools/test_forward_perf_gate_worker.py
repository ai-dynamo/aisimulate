# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the revision-local forward-performance worker."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.forward_perf_gate import PROTOCOL_VERSION, cases, measurement, worker
from tools.forward_perf_gate import run as gate_run
from tools.prediction_regression_gate import grid

from aiconfigurator.sdk.errors import (
    EmpiricalNotImplementedError,
    MissingSystemFlopsError,
    PerfDataNotAvailableError,
    SolNotImplementedError,
)

pytestmark = pytest.mark.unit


class SyntheticPanic(BaseException):
    pass


def _request() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "revision": "abc123",
        "warmup": 10,
        "iterations": 100,
        "case": cases.expand_cases()[0],
    }


def test_matrix_contains_unique_grid_points() -> None:
    expanded = cases.expand_cases()
    assert len(expanded) == 64
    assert len({case["case_id"] for case in expanded}) == len(expanded)
    assert sum(case["phase"] == "context" for case in expanded) == 31
    assert sum(case["phase"] == "generation" for case in expanded) == 33
    assert {case["database_mode"] for case in expanded} == {"SILICON", "EMPIRICAL"}
    assert all(case["database_mode"] == "SILICON" for case in expanded[36:])
    # Freeze the original 36 requests, including their IDs, values, and order.
    legacy = json.dumps(expanded[:36], sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(legacy).hexdigest() == "17b8247fc587711a5e7b00b9cb3dba86e093b9e210e8272112eb33ddea39eca2"


def test_additional_profiles_preserve_original_cache_groups() -> None:
    expanded = cases.expand_cases()
    original_groups = {worker._group_key(case) for case in expanded[:36]}
    additional_groups = {worker._group_key(case) for case in expanded[36:]}
    assert original_groups.isdisjoint(additional_groups)
    assert len(additional_groups) == 9
    for case in expanded[36:]:
        assert case["system_name"] in case["model_id"]
        assert f"{case['backend_name']}-{case['backend_version']}" in case["model_id"]
        assert f"-tp{case['tp_size']}-pp{case['pp_size']}-adp{case['attention_dp_size']}" in case["model_id"]
        assert f"-mtp{case['moe_tp_size']}-ep{case['moe_ep_size']}" in case["model_id"]
    assert [case["prefix"] for case in expanded if case["prefix"]] == [4096, 7168, 4096, 7168]
    dp_cases = [case for case in expanded if case["attention_dp_size"] == 8]
    assert [(case["batch_size"], case["isl"], case["tp_size"], case["moe_ep_size"]) for case in dp_cases] == [
        (32, 1024, 1, 8),
        (8, 32768, 1, 8),
    ]


def test_workflow_filters_cover_matrix_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    matches_path = runpy.run_path(str(repo_root / "scripts/select_forward_perf.py"))["matches_path"]
    dependencies = set()
    for case in cases.expand_cases():
        root = "python/aisimulate/src/aiconfigurator_core"
        model_config = f"{root}/model_configs/{case['model_path'].replace('/', '--')}_config.json"
        assert (repo_root / model_config).is_file()
        dependencies.add(model_config)
        dependencies.add(f"{root}/systems/data/{case['system_name']}/gemm/{case['backend_name']}/data.parquet")
    assert all(matches_path(path) for path in dependencies)
    for unrelated in (
        "docs/cli/user-guide.md",
        "python/aisimulate/src/aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json",
        "python/aisimulate/src/aiconfigurator_core/systems/data/a100_sxm/gemm/vllm/data.parquet",
    ):
        assert not matches_path(unrelated)


def test_case_hash_is_stable_across_key_order() -> None:
    case = cases.expand_cases()[0]
    reordered = dict(reversed(list(case.items())))
    assert worker.canonical_case_hash(case) == worker.canonical_case_hash(reordered)


def test_request_validation_rejects_protocol_and_case_drift() -> None:
    request = _request()
    assert worker.validate_request(request)[0]["case_id"] == request["case"]["case_id"]

    request["protocol_version"] = 2
    with pytest.raises(ValueError, match="protocol_version"):
        worker.validate_request(request)

    request = _request()
    request["case"] = {**request["case"], "unexpected": True}
    with pytest.raises(ValueError, match="unknown"):
        worker.validate_request(request)


def test_batch_request_validation_requires_unique_cases() -> None:
    expanded = cases.expand_cases()
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "revision": "abc123",
        "warmup": 10,
        "iterations": 100,
        "cases": expanded,
    }
    assert worker.validate_batch_request(request)[0] == expanded

    with pytest.raises(ValueError, match="non-empty array"):
        worker.validate_batch_request({**request, "cases": []})
    with pytest.raises(ValueError, match="unique"):
        worker.validate_batch_request({**request, "cases": [expanded[0], expanded[0]]})
    with pytest.raises(ValueError, match="mutually exclusive"):
        worker.validate_batch_request({**request, "case": expanded[0]})


@pytest.mark.parametrize(
    "error_type",
    [
        PerfDataNotAvailableError,
        EmpiricalNotImplementedError,
        MissingSystemFlopsError,
        SolNotImplementedError,
    ],
)
def test_worker_classifies_coverage_errors_as_data_miss(error_type: type[BaseException]) -> None:
    response = {}
    worker._record_error(response, error_type("missing data"))
    assert response["status"] == "DATA_MISS"
    assert response["error"]["type"] == error_type.__name__


def test_worker_classifies_chained_coverage_error_as_data_miss() -> None:
    response = {}
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = PerfDataNotAvailableError("missing data")
    worker._record_error(response, wrapped)
    assert response["status"] == "DATA_MISS"


def test_worker_handles_empty_exception_message() -> None:
    response = {}
    worker._record_error(response, MemoryError())
    assert response["status"] == "INVALID"
    assert response["error"]["message"] == ""


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit()])
def test_worker_reraises_control_flow(error: BaseException) -> None:
    with pytest.raises(type(error)):
        worker._reraise_control_flow(error)


def test_missing_database_is_a_data_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(measurement.perf_database, "get_database_view", lambda *args, **kwargs: None)
    with pytest.raises(PerfDataNotAvailableError, match="failed to load perf database"):
        measurement.build_session(
            measurement.BenchmarkCase(model_path="model"),
            suppress_loader_output=True,
            database_mode="SILICON",
        )


def test_batch_request_groups_setup_work_and_preserves_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expanded = cases.expand_cases()
    selected = [expanded[0], expanded[9], expanded[1], expanded[18]]
    groups = []

    def fake_run_case_group(group: list[dict], *, warmup: int, iterations: int, revision: str) -> list[dict]:
        groups.append([case["case_id"] for case in group])
        return [{"case_id": case["case_id"], "status": "OK"} for case in group]

    monkeypatch.setattr(worker, "_run_case_group", fake_run_case_group)
    response = worker.run_batch_request(
        {
            "protocol_version": PROTOCOL_VERSION,
            "revision": "abc123",
            "warmup": 1,
            "iterations": 3,
            "cases": selected,
        }
    )
    assert groups == [
        [expanded[0]["case_id"], expanded[1]["case_id"]],
        [expanded[9]["case_id"]],
        [expanded[18]["case_id"]],
    ]
    assert [result["case_id"] for result in response["results"]] == [case["case_id"] for case in selected]


def test_case_group_resets_and_builds_once_and_continues_after_case_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = cases.expand_cases()[:9]
    reset_calls = []
    setup_calls = []
    phase_calls = []
    measure_calls = 0
    runtime = measurement.config.RuntimeConfig(batch_size=1, isl=1024, osl=grid.CTX_OSL)

    monkeypatch.setattr(worker, "clear_caches", lambda case: reset_calls.append(case))
    monkeypatch.setattr(worker, "ensure_rust_library_present", lambda: None)

    def fake_setup(*args: object, **kwargs: object) -> tuple[float, object, object]:
        setup_calls.append(None)
        return 1.0, object(), runtime

    def fake_phase_call(session: object, runtime_config: object, *, phase: str, stride: int):
        phase_calls.append((phase, runtime_config.batch_size, runtime_config.isl))
        return lambda: 1.0

    def fake_measure(call: object, *, warmup: int, iterations: int):
        nonlocal measure_calls
        measure_calls += 1
        if measure_calls == 2:
            raise SyntheticPanic("case failed")
        return 1.0, 10.0, [5.0], {"call_median_us": 5.0}

    monkeypatch.setattr(worker, "measure_session_setup_ms", fake_setup)
    monkeypatch.setattr(worker, "phase_call", fake_phase_call)
    monkeypatch.setattr(worker, "measure_cold_and_warm", fake_measure)

    results = worker._run_case_group(selected, warmup=0, iterations=1, revision="abc123")

    assert len(reset_calls) == 1
    assert len(setup_calls) == 1
    assert phase_calls[:2] == [("context", 2, 2048), ("generation", 2, 2048)]
    assert len(phase_calls) == len(selected) + 2
    assert [result["case_id"] for result in results] == [case["case_id"] for case in selected]
    assert [result["status"] for result in results[:3]] == ["OK", "INVALID", "OK"]


def test_case_group_isolates_base_exception_during_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = cases.expand_cases()[:9]
    monkeypatch.setattr(worker, "clear_caches", lambda case: None)
    monkeypatch.setattr(worker, "ensure_rust_library_present", lambda: None)
    monkeypatch.setattr(
        worker,
        "measure_session_setup_ms",
        lambda *args, **kwargs: (_ for _ in ()).throw(SyntheticPanic("setup failed")),
    )

    results = worker._run_case_group(selected, warmup=0, iterations=1, revision="abc123")
    assert {result["status"] for result in results} == {"INVALID"}
    assert {result["error"]["type"] for result in results} == {"SyntheticPanic"}


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (PerfDataNotAvailableError("prime is missing"), "DATA_MISS"),
        (SyntheticPanic("prime panicked"), "INVALID"),
    ],
)
def test_priming_failure_is_limited_to_one_phase(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_status: str,
) -> None:
    selected = cases.expand_cases()[:9]
    runtime = measurement.config.RuntimeConfig(batch_size=1, isl=1024, osl=grid.CTX_OSL)

    monkeypatch.setattr(worker, "clear_caches", lambda case: None)
    monkeypatch.setattr(worker, "ensure_rust_library_present", lambda: None)
    monkeypatch.setattr(
        worker,
        "measure_session_setup_ms",
        lambda *args, **kwargs: (1.0, object(), runtime),
    )

    def fake_phase_call(session: object, runtime_config: object, *, phase: str, stride: int):
        def call() -> float:
            if phase == "context" and runtime_config.batch_size == 2 and runtime_config.isl == 2048:
                raise failure
            return 1.0

        return call

    monkeypatch.setattr(worker, "phase_call", fake_phase_call)
    monkeypatch.setattr(
        worker,
        "measure_cold_and_warm",
        lambda *args, **kwargs: (1.0, 10.0, [5.0], {"call_median_us": 5.0}),
    )

    results = worker._run_case_group(selected, warmup=0, iterations=1, revision="abc123")
    context = [result for result in results if result["case"]["phase"] == "context"]
    generation = [result for result in results if result["case"]["phase"] == "generation"]
    assert {result["status"] for result in context} == {expected_status}
    assert {result["error"]["type"] for result in context} == {"PRIMING_FAILED"}
    assert all(type(failure).__name__ in result["error"]["message"] for result in context)
    assert {result["status"] for result in generation} == {"OK"}
    assert {result["steady_state_setup_queries"] for result in generation} == {1}


def test_cold_is_unseen_query_after_steady_state_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter([0, 1_000, 2_000, 3_000, 4_000, 6_000])
    monkeypatch.setattr(measurement, "_perf_counter_ns", lambda: next(timestamps))
    call_count = 0
    events = []

    def call() -> float:
        nonlocal call_count
        call_count += 1
        events.append("target")
        return float(call_count)

    predicted, cold_us, samples, stats = measurement.measure_cold_and_warm(
        call,
        warmup=2,
        iterations=2,
    )
    assert predicted == 1.0
    assert cold_us == 1.0
    assert samples == [1.0, 2.0]
    assert stats["call_median_us"] == 1.5
    assert call_count == 5
    assert events == ["target", "target", "target", "target", "target"]


@pytest.mark.parametrize(
    ("phase", "points", "expected_osl"),
    [("context", grid.PREFILL_POINTS, 8), ("generation", grid.DECODE_POINTS, 256)],
)
def test_phase_priming_query_is_outside_the_measured_matrix(
    monkeypatch: pytest.MonkeyPatch, phase: str, points: list[tuple[int, int]], expected_osl: int
) -> None:
    monkeypatch.setattr(grid, "CTX_OSL", 16)
    monkeypatch.setattr(grid, "GEN_OSL", 512)
    runtime = measurement.config.RuntimeConfig(batch_size=1, isl=1024, osl=8, prefix=128)
    prime = measurement.priming_runtime_config(runtime, phase=phase)
    assert (prime.batch_size, prime.isl, prime.osl, prime.prefix) == (2, 2048, expected_osl, 0)
    assert (prime.batch_size, prime.isl) not in points


def test_cache_reset_uses_public_database_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(measurement.perf_database, "unload_database", lambda *args: calls.append(args))
    measurement.clear_caches(measurement.BenchmarkCase(model_path="model"))
    assert calls == [("b200_sxm", "vllm", "0.24.0")]


def test_run_worker_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="null\n", stderr="warning\n")
    monkeypatch.setattr(gate_run.subprocess, "run", lambda *args, **kwargs: completed)
    response = gate_run.run_worker(
        python=Path("/python"),
        worker=Path("/worker"),
        revision="abc123",
        case=cases.expand_cases()[0],
        warmup=0,
        iterations=1,
        cpu=0,
        timeout=1.0,
    )
    assert response["status"] == "WORKER_ERROR"
    assert response["error"]["message"] == "worker JSON must be an object, got NoneType"


@pytest.mark.parametrize(
    "result_ids",
    [["first"], ["first", "first"], ["first", "unexpected"], ["second", "first"]],
)
def test_run_worker_batch_rejects_incomplete_duplicate_or_reordered_results(
    monkeypatch: pytest.MonkeyPatch,
    result_ids: list[str],
) -> None:
    selected = cases.expand_cases()[:2]
    selected[0] = {**selected[0], "case_id": "first"}
    selected[1] = {**selected[1], "case_id": "second"}
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "revision": "abc123",
                "results": [{"case_id": case_id} for case_id in result_ids],
            }
        ),
        stderr="",
    )
    monkeypatch.setattr(gate_run.subprocess, "run", lambda *args, **kwargs: completed)
    results, error = gate_run.run_worker_batch(
        python=Path("/python"),
        worker=Path("/worker"),
        revision="abc123",
        cases=selected,
        warmup=0,
        iterations=1,
        cpu=0,
        timeout=1.0,
    )
    assert results == []
    assert "do not match the request" in error


def test_run_worker_batch_reports_process_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=7, stdout="", stderr="boom")
    monkeypatch.setattr(gate_run.subprocess, "run", lambda *args, **kwargs: completed)
    results, error = gate_run.run_worker_batch(
        python=Path("/python"),
        worker=Path("/worker"),
        revision="abc123",
        cases=cases.expand_cases()[:2],
        warmup=0,
        iterations=1,
        cpu=0,
        timeout=1.0,
    )
    assert results == []
    assert error == "WORKER_ERROR: exit 7: boom"


def test_full_controller_uses_twelve_batch_processes_and_alternates_case_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = cases.expand_cases()[:2]
    executable = tmp_path / "python"
    worker_path = tmp_path / "worker.py"
    executable.touch()
    worker_path.touch()
    output_dir = tmp_path / "results"
    calls = []

    def fake_batch(*, revision: str, cases: list[dict], **kwargs: object) -> tuple[list[dict], None]:
        calls.append((revision, [case["case_id"] for case in cases]))
        return (
            [
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "revision": revision,
                    "case_id": case["case_id"],
                    "case_hash": worker.canonical_case_hash(case),
                    "status": "OK",
                    "cold_us": 100_000.0,
                    "warm": {"call_median_us": 100.0},
                }
                for case in cases
            ],
            None,
        )

    monkeypatch.setattr(
        gate_run,
        "_parse_args",
        lambda: SimpleNamespace(
            base_python=executable,
            base_worker=worker_path,
            base_revision="base",
            head_python=executable,
            head_worker=worker_path,
            head_revision="head",
            output_dir=output_dir,
            rounds=5,
            warmup=10,
            iterations=100,
            worker_timeout=120.0,
            skip_prewarm=False,
            smoke=False,
        ),
    )
    monkeypatch.setattr(gate_run.shutil, "which", lambda command: "/usr/bin/taskset")
    monkeypatch.setattr(gate_run.case_matrix, "expand_cases", lambda: selected)
    monkeypatch.setattr(gate_run, "run_worker_batch", fake_batch)
    monkeypatch.setattr(gate_run, "_command_version", lambda command: "test")

    assert gate_run.main() == 0
    raw = json.loads((output_dir / "raw_results.json").read_text())
    forward = [case["case_id"] for case in selected]
    assert raw["worker_processes"] == 12
    assert len(calls) == 12
    assert calls[2:4] == [("base", forward), ("head", forward)]
    assert calls[4:6] == [("head", list(reversed(forward))), ("base", list(reversed(forward)))]


@pytest.mark.parametrize(
    ("smoke", "values", "expected"),
    [
        (False, (None, None, None), (5, 10, 100)),
        (True, (None, None, None), (1, 1, 3)),
        (True, (3, 2, 7), (3, 2, 7)),
        (True, (3, None, None), (3, 1, 3)),
    ],
)
def test_effective_counts_preserve_explicit_smoke_values(
    smoke: bool,
    values: tuple[int | None, int | None, int | None],
    expected: tuple[int, int, int],
) -> None:
    args = SimpleNamespace(smoke=smoke, rounds=values[0], warmup=values[1], iterations=values[2])
    assert gate_run._effective_counts(args) == expected


def test_smoke_writes_explicit_counts_to_raw_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = cases.expand_cases()[:1]
    executable = tmp_path / "python"
    worker_path = tmp_path / "worker.py"
    executable.touch()
    worker_path.touch()
    output_dir = tmp_path / "results"

    def fake_batch(*, revision: str, cases: list[dict], **kwargs: object) -> tuple[list[dict], None]:
        return (
            [
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "revision": revision,
                    "case_id": case["case_id"],
                    "case_hash": worker.canonical_case_hash(case),
                    "status": "OK",
                    "cold_us": 100_000.0,
                    "warm": {"call_median_us": 100.0},
                }
                for case in cases
            ],
            None,
        )

    monkeypatch.setattr(
        gate_run,
        "_parse_args",
        lambda: SimpleNamespace(
            base_python=executable,
            base_worker=worker_path,
            base_revision="base",
            head_python=executable,
            head_worker=worker_path,
            head_revision="head",
            output_dir=output_dir,
            rounds=3,
            warmup=2,
            iterations=7,
            worker_timeout=120.0,
            skip_prewarm=False,
            smoke=True,
        ),
    )
    monkeypatch.setattr(gate_run.shutil, "which", lambda command: "/usr/bin/taskset")
    monkeypatch.setattr(gate_run.case_matrix, "expand_cases", lambda: selected)
    monkeypatch.setattr(gate_run, "run_worker_batch", fake_batch)
    monkeypatch.setattr(gate_run, "_command_version", lambda command: "test")

    assert gate_run.main() == 0
    raw = json.loads((output_dir / "raw_results.json").read_text())
    assert raw["configuration"]["mode"] == "smoke"
    assert raw["configuration"]["rounds"] == 3
    assert raw["configuration"]["warmup"] == 2
    assert raw["configuration"]["iterations"] == 7


def test_raw_checkpoint_is_atomic(tmp_path: Path) -> None:
    gate_run._checkpoint({"status": "partial"}, tmp_path)
    assert (tmp_path / "raw_results.json").read_text() == '{\n  "status": "partial"\n}\n'
    assert not (tmp_path / "raw_results.json.tmp").exists()
