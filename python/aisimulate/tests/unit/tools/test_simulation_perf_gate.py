# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from tools.simulation_perf_gate import PROTOCOL_VERSION, cases, digest
from tools.simulation_perf_gate.compare import compare_case, validate
from tools.simulation_perf_gate.contract import MODEL_FIELDS

pytestmark = pytest.mark.unit
CASE = cases.case("case", "vllm", model="pinned", tp=1, requests=1, osl=2)


def response(side, phase="measure", wall=2000):
    identity = {
        **dict.fromkeys(MODEL_FIELDS),
        "model": "pinned",
        "system": "b200_sxm",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "worker_type": "aggregated",
        "tp": 1,
        "pp": 1,
        "attention_dp": 1,
        "kv_block_size": 64,
        "systems_paths": ["package:aisimulate_core/systems"],
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": CASE["case_id"],
        "case_hash": digest(CASE),
        "revision": side,
        "phase": phase,
        "status": "OK",
        "wall_time_ms": wall,
        "worker_elapsed_ms": wall + 1000,
        "model_identity": {"aggregated": identity},
        "model_provenance": {
            "aggregated": {
                **identity,
                "systems_paths": ["/env/aisimulate_core/systems"],
                "provider": "aic",
                "estimation_mode": "op_level",
                "fallback_policy": "deny",
                "database_mode": "SILICON",
                "enable_shared_layer": True,
            }
        },
        "report": {"num_requests": 1, "completed_requests": 1, "total_output_tokens": 2},
    }


def samples(head_times=(2000,) * 5, base_time=2000):
    return {
        "rounds": [
            {"round": index, "base": response("base", wall=base_time), "head": response("head", wall=wall)}
            for index, wall in enumerate(head_times, 1)
        ]
    }


def classify(value):
    return compare_case(CASE, value, revisions={"base": "base", "head": "head"}, rounds=5)["classification"]


@pytest.mark.parametrize(
    "times,base,expected",
    [
        ((2000,) * 5, 2000, "PASS"),
        ((2400,) * 4 + (2000,), 2000, "PERFORMANCE_REGRESSION"),
        ((2400,) * 3 + (2000,) * 2, 2000, "PASS"),
        ((2200,) * 5, 2000, "PASS"),
        ((200,) * 5, 100, "PASS"),
    ],
)
def test_threshold_and_consensus(times, base, expected):
    assert classify(samples(times, base)) == expected


def test_controller_runs_only_five_sequential_timing_pairs(tmp_path, monkeypatch):
    from tools.simulation_perf_gate import run

    calls = []

    def invoke(**kwargs):
        calls.append((kwargs["revision"], kwargs["phase"], kwargs["cpu"]))
        return response(kwargs["revision"], kwargs["phase"])

    monkeypatch.setattr(run, "invoke", invoke)
    monkeypatch.setattr(run, "expand_cases", lambda: [CASE])
    monkeypatch.setattr(run.os, "sched_getaffinity", lambda _: {7})
    monkeypatch.setattr(run.shutil, "which", lambda _: "/bin/taskset")
    arguments = ["run.py", "--output-dir", str(tmp_path)]
    for side in ("base", "head"):
        arguments += [f"--{side}-revision", side, f"--{side}-python", "/python", f"--{side}-worker", "/worker.py"]
    monkeypatch.setattr(run.sys, "argv", arguments)
    assert run.main() == 0
    assert calls == [
        (side, "measure", 7) for pair in range(5) for side in (("head", "base") if pair % 2 == 0 else ("base", "head"))
    ]
    raw = json.loads((tmp_path / "raw.json").read_text())
    assert set(raw["samples"][CASE["case_id"]]) == {"rounds"}
    assert not list(tmp_path.glob("*.gz"))


@pytest.mark.parametrize(
    "fault",
    [
        "missing_round",
        "protocol",
        "protocol_type",
        "phase",
        "hash",
        "revision",
        "incomplete",
        "output_tokens",
        "missing_data",
        "nonfinite",
        "diagnostic",
        "identity",
        "missing_provenance",
        "role",
        "policy",
        "malformed_report",
    ],
)
def test_invalid_measurements_never_pass(fault):
    value = samples()
    head = value["rounds"][0]["head"]
    if fault == "missing_round":
        value["rounds"].pop()
    elif fault in {"protocol", "phase", "hash", "revision"}:
        head[{"protocol": "protocol_version", "hash": "case_hash"}.get(fault, fault)] = "wrong"
    elif fault == "protocol_type":
        head["protocol_version"] = float(PROTOCOL_VERSION)
    elif fault == "incomplete":
        head["report"]["completed_requests"] = 0
    elif fault == "output_tokens":
        head["report"]["total_output_tokens"] = 1
    elif fault == "missing_data":
        head.update(status="ERROR", error={"message": "missing model data"})
    elif fault == "nonfinite":
        head["wall_time_ms"] = float("nan")
    elif fault == "diagnostic":
        head["report"]["diagnostic"] = {"nested": [float("inf")]}
    elif fault == "identity":
        head["model_identity"]["aggregated"]["model"] = "another"
    elif fault == "missing_provenance":
        del head["model_provenance"]
    elif fault == "role":
        head["model_provenance"]["prefill"] = head["model_provenance"].pop("aggregated")
    elif fault == "policy":
        head["model_provenance"]["aggregated"]["fallback_policy"] = "allow"
    else:
        head["report"] = []
    assert classify(value) == "INVALID_COMPARISON"


@pytest.mark.parametrize("wall,expected", [(2000, "PASS"), (4000, "PERFORMANCE_REGRESSION")])
def test_simulated_results_and_diagnostics_do_not_change_timing_verdict(wall, expected):
    value = samples((wall,) * 5)
    for index, pair in enumerate(value["rounds"]):
        pair["head"]["report"].update(duration_ms=index, p99_ttft_ms=index, prefix_cache_reused_ratio=0.5)
        pair["head"]["model_identity"]["aggregated"]["new_metadata"] = index
        pair["head"]["model_provenance"]["aggregated"]["new_metadata"] = index
    assert classify(value) == expected


@pytest.mark.parametrize("status,settled", [("incomplete", None), ("completed", None), ("completed", 3.0)])
def test_agentx_requires_complete_work(status, settled):
    case = {**CASE, "trace_sha256": "fixture"}
    value = response("head")
    value["case_hash"] = digest(case)
    value["report"]["agentic_play_outcomes"] = [{"status": status, "settled_at_ms": settled}]
    if status == "completed" and settled is not None:
        validate(value, case, "head", "measure")
    else:
        with pytest.raises(ValueError, match="AgentX"):
            validate(value, case, "head", "measure")


@pytest.mark.parametrize("payload", ["{", "[]", "null", '{"case": []}', '{"case": {"bad": NaN}}'])
def test_worker_malformed_input_returns_structured_error(monkeypatch, capsys, payload):
    import io

    from tools.simulation_perf_gate import worker

    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(payload))
    assert worker.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["status"] == "ERROR"
    assert result["error"]["type"] in {"ValueError", "JSONDecodeError"}
    assert result["error"]["message"]


@pytest.mark.parametrize(
    "protocol,phase", [(2, "measure"), (3, "measure"), (4.0, "measure"), (4, "availability"), (4, "equivalence")]
)
def test_worker_rejects_old_protocol_and_phase(protocol, phase):
    from tools.simulation_perf_gate.worker import run

    with pytest.raises(ValueError, match="protocol|phase"):
        run({"protocol_version": protocol, "phase": phase, "case": CASE})


def test_matrix_and_trace_are_complete_and_local():
    matrix = cases.expand_cases()
    assert len({item["case_id"] for item in matrix}) == 12
    trace = Path(cases.__file__).parent / "fixtures/agentx.jsonl"
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == cases.AGENTX_SHA256
    play = json.loads(trace.read_text())
    rows = [
        row for entry in play["requests"] for row in (entry["requests"] if entry["type"] == "subagent" else [entry])
    ]
    assert len(rows) == 129
    assert sum(entry["type"] == "subagent" for entry in play["requests"]) == 4
    for item in matrix:
        assert item["determinism"] == "canonical_v1"
        assert item["config"]["engine"]["estimation_mode"] == "op_level"
        assert item["config"]["engine"]["fallback_policy"] == "deny"
        assert "max_virtual_time_seconds" not in item["config"]["traffic"].get("stop", {})
        if item.get("trace_sha256"):
            assert item["expected_output_tokens"] == sum(row["out"] for row in rows)
        else:
            traffic = item["config"]["traffic"]
            session = traffic["source"]["type"] == "synthetic-session"
            assert item["expected_requests"] == (
                traffic["stop"]["sessions"] * 4 if session else traffic["stop"]["requests"]
            )
    assert cases.expand_cases() == deepcopy(matrix)


def test_qualification_requires_runtime_floor_and_budget():
    from tools.simulation_perf_gate.run import qualification_errors

    raw = {"elapsed_seconds": 901, "cases": [CASE]}
    result = {"classification": "PASS", "base_median_ms": 1999, "head_median_ms": 2001}
    assert len(qualification_errors(raw, [result])) == 2
    raw["elapsed_seconds"] = 899
    result["base_median_ms"] = 2000
    assert qualification_errors(raw, [result]) == []


def test_packaged_model_identity_is_independent_of_install_location(tmp_path):
    from tools.simulation_perf_gate.worker import portable_model_identity

    base = tmp_path / "base/site-packages/aisimulate_core/systems"
    head = tmp_path / "head/site-packages/aisimulate_core/systems"
    identity = response("base", "measure")["model_identity"]["aggregated"]
    assert portable_model_identity({**identity, "systems_paths": [str(base)]}, base) == portable_model_identity(
        {**identity, "systems_paths": [str(head)], "new_metadata": 1, "estimator_config": {"new_option": False}}, head
    )
    with pytest.raises(ValueError, match="packaged model data"):
        portable_model_identity({**identity, "systems_paths": [str(base)]}, head)


@pytest.mark.parametrize("failure", [None, "crash", "timeout", "malformed_error"])
def test_invoke_owns_process_timer_and_preserves_crash_details(tmp_path, monkeypatch, failure):
    from tools.simulation_perf_gate import run

    clock = [1.0]
    monkeypatch.setattr(run.time, "perf_counter", lambda: clock[0])
    original_loads = json.loads

    def slow_loads(payload):
        clock[0] += 10.0
        return original_loads(payload)

    monkeypatch.setattr(run.json, "loads", slow_loads)

    def execute(command, **kwargs):
        clock[0] += 2.0
        kwargs["stderr"].write("x" * 5000 + "panic details")
        if failure == "crash":
            raise subprocess.CalledProcessError(1, command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        payload = response("head", "measure") if failure is None else {"status": "ERROR", "error": "bad"}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload))

    monkeypatch.setattr(run.subprocess, "run", execute)
    log = tmp_path / "worker.log"
    result = run.invoke(
        python=Path("python"),
        worker=tmp_path / "src/python/aisimulate/tools/simulation_perf_gate/worker.py",
        case=CASE,
        revision="head",
        phase="measure",
        cpu=0,
        timeout=1,
        log=log,
    )
    assert result["worker_elapsed_ms"] == 2000.0
    if failure:
        assert result["status"] == "ERROR"
        assert result["error"]["log"] == str(log)
        assert len(result["error"]["stderr_tail"].encode()) == 4096
        assert result["error"]["stderr_tail"].endswith("panic details")
        assert len(log.read_bytes()) > 4096
        value = samples()
        value["rounds"][0]["head"] = result
        assert classify(value) == "INVALID_COMPARISON"
    else:
        assert result["status"] == "OK"
