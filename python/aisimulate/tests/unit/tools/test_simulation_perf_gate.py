# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from tools.simulation_perf_gate import PROTOCOL_VERSION, cases, digest
from tools.simulation_perf_gate.compare import compare_case, difference
from tools.simulation_perf_gate.contract import (
    COUNTS,
    METRICS,
    MODEL_FIELDS,
    REQUEST_FIELDS,
    ROUTING_FIELDS,
    TRAJECTORY_COUNTS,
    TRAJECTORY_METRICS,
    behavior,
)

pytestmark = pytest.mark.unit
CASE = {"case_id": "case", "expected_requests": 1, "expected_output_tokens": 2}


def response(side, phase, wall=2000):
    summary = {
        **dict.fromkeys(COUNTS, 1),
        **dict.fromkeys(METRICS, 3.0),
        "num_requests": 1,
        "completed_requests": 1,
        "total_input_tokens": 8,
        "total_output_tokens": 2,
        "committed_prefill_tokens": 8,
        "duration_ms": 3.0,
        "prefix_cache_reused_ratio": 0.0,
        "first_admission_prefix_cache_reused_ratio": 0.0,
    }
    if phase == "availability":
        summary["per_request"] = [
            {
                **dict.fromkeys(REQUEST_FIELDS),
                "uuid": "one",
                "request_id": "one",
                "terminal_status": "completed",
                "input_length": 8,
                "output_length": 2,
                "requested_output_length": 2,
                "reused_input_tokens": 0,
                "admission_count": 1,
                "readmission_count": 0,
                "routing_history": [
                    {**dict.fromkeys(ROUTING_FIELDS), "pool": "agg", "outcome": "immediate", "dp_rank": 0}
                ],
                "admission_history": [
                    {
                        "admission_ordinal": 0,
                        "pool_admission_ordinal": 0,
                        "pool": "agg",
                        "at_ms": 0.0,
                        "reused_input_tokens": 0,
                        "is_readmission": False,
                    }
                ],
            }
        ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": CASE["case_id"],
        "case_hash": digest(CASE),
        "revision": side,
        "phase": phase,
        "status": "OK",
        "wall_time_ms": wall,
        "worker_elapsed_ms": wall + 1000,
        "model_identity": {
            "aggregated": {
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
        },
        "behavior": summary,
    }


def samples(head_times=(2000,) * 5, base_time=2000):
    return {
        "availability": {side: response(side, "availability") for side in ("base", "head")},
        "rounds": [
            {"round": index, "base": response("base", "measure", base_time), "head": response("head", "measure", wall)}
            for index, wall in enumerate(head_times, 1)
        ],
    }


def classify(value):
    return compare_case(CASE, value, revisions={"base": "base", "head": "head"}, rounds=5)["classification"]


@pytest.mark.parametrize(
    "times,base,expected",
    [
        ((2000,) * 5, 2000, "PASS"),
        ((2400,) * 4 + (2000,), 2000, "PERFORMANCE_REGRESSION"),
        ((2400,) * 3 + (2000,) * 2, 2000, "PASS"),
        ((2200,) * 5, 2000, "PASS"),  # Strict relative threshold.
        ((200,) * 5, 100, "PASS"),  # Strict absolute threshold.
    ],
)
def test_threshold_and_consensus(times, base, expected):
    assert classify(samples(times, base)) == expected


def test_behavior_change_takes_precedence_over_slowdown():
    value = samples((3000,) * 5)
    for pair in [value["availability"], *value["rounds"]]:
        pair["head"]["behavior"]["duration_ms"] = 4.0
    assert classify(value) == "BEHAVIOR_CHANGED"
    value["availability"]["head"]["behavior"]["per_request"][0]["request_id"] = "other"
    assert classify(value) == "BEHAVIOR_CHANGED"


@pytest.mark.parametrize(
    "fault",
    [
        "missing_round",
        "protocol",
        "protocol_type",
        "hash",
        "revision",
        "incomplete",
        "missing_data",
        "nonfinite",
        "identity",
        "within_revision_drift",
        "missing_records",
        "malformed_behavior",
        "malformed_availability",
    ],
)
def test_invalid_comparisons_never_pass(fault):
    value = samples()
    head = value["rounds"][0]["head"]
    if fault == "missing_round":
        value["rounds"].pop()
    elif fault in {"protocol", "hash", "revision"}:
        head[{"protocol": "protocol_version", "hash": "case_hash", "revision": "revision"}[fault]] = "wrong"
    elif fault == "protocol_type":
        head["protocol_version"] = float(PROTOCOL_VERSION)
    elif fault == "incomplete":
        head["behavior"]["completed_requests"] = 0
    elif fault == "missing_data":
        head.update(status="ERROR", error={"message": "missing model data"})
    elif fault == "nonfinite":
        head["wall_time_ms"] = float("nan")
    elif fault == "identity":
        head["model_identity"]["aggregated"]["model"] = "another"
    elif fault == "within_revision_drift":
        for side in ("base", "head"):
            value["rounds"][0][side]["behavior"]["duration_ms"] = 4.0
    elif fault == "malformed_behavior":
        head["behavior"] = []
    elif fault == "malformed_availability":
        value["availability"]["head"]["behavior"] = None
    else:
        del value["availability"]["head"]["behavior"]["per_request"]
    assert classify(value) == "INVALID_COMPARISON"


def test_float_tolerance_does_not_hide_count_or_identity_changes():
    assert difference(1.0, 1.0 + 1e-8) is None
    assert difference(1, 1.0) is not None
    assert difference({"count": 1}, {"count": 2}) is not None
    assert difference("a", "b") is not None
    assert "missing_field" in difference({"missing_field": 1}, {})


def test_behavior_uses_fixed_fields_without_changing_raw_report():
    report = response("base", "measure")["behavior"]
    report.update(wall_time_ms=2000.0, processed_tokens_per_s=100.0, processed_output_tokens_per_s=25.0, power_w=400.0)
    result = behavior(report, per_request=False)
    assert not {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s", "power_w"} & result.keys()
    assert report["power_w"] == 400.0


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


def test_request_artifacts_preserve_behavior_comparison(tmp_path):
    import gzip

    from tools.simulation_perf_gate.run import retain_request_artifacts

    value = samples()
    value["availability"]["head"]["behavior"]["per_request"][0]["request_id"] = "changed"
    retain_request_artifacts(value["availability"], CASE, {"base": "base", "head": "head"}, tmp_path)
    assert "per_request" not in value["availability"]["base"]["behavior"]
    assert classify(value) == "BEHAVIOR_CHANGED"
    artifact = value["availability"]["head"]["per_request_artifact"]
    with gzip.open(tmp_path / artifact["path"], "rt") as source:
        records = json.load(source)
    assert digest(records) == artifact["sha256"]
    assert records[0]["request_id"] == "changed"
    del value["availability"]["per_request_difference"]
    assert classify(value) == "INVALID_COMPARISON"


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


@pytest.mark.parametrize("wall,expected", [(2000, "PASS"), (4000, "PERFORMANCE_REGRESSION")])
def test_additive_diagnostics_do_not_change_verdict_or_artifacts(tmp_path, wall, expected):
    import gzip

    from tools.simulation_perf_gate.run import retain_request_artifacts

    value = samples((wall,) * 5)
    for index, pair in enumerate([value["availability"], *value["rounds"]]):
        pair["head"]["behavior"]["new_diagnostic"] = index  # Also varies within one revision.
        pair["head"]["model_identity"]["aggregated"]["new_metadata"] = index
    row = value["availability"]["head"]["behavior"]["per_request"][0]
    row["diagnostic"] = {"extra": True}
    row["routing_history"][0]["diagnostic"] = 42
    row["admission_history"][0]["diagnostic"] = 42
    assert classify(value) == expected
    retain_request_artifacts(value["availability"], CASE, {"base": "base", "head": "head"}, tmp_path)
    assert classify(value) == expected
    artifact = value["availability"]["head"]["per_request_artifact"]
    with gzip.open(tmp_path / artifact["path"], "rt") as source:
        rows = json.load(source)
    assert rows[0]["diagnostic"] == {"extra": True}
    assert digest(rows) == artifact["sha256"]


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("latency", "BEHAVIOR_CHANGED"),
        ("routing", "BEHAVIOR_CHANGED"),
        ("admission", "BEHAVIOR_CHANGED"),
        ("missing_latency", "INVALID_COMPARISON"),
        ("missing_route", "INVALID_COMPARISON"),
        ("missing_identity", "INVALID_COMPARISON"),
        ("noninteger_rank", "INVALID_COMPARISON"),
    ],
)
def test_required_results_and_nested_records(fault, expected):
    value = samples()
    head = value["availability"]["head"]
    row = head["behavior"]["per_request"][0]
    if fault == "latency":
        for pair in [value["availability"], *value["rounds"]]:
            pair["head"]["behavior"]["p99_ttft_ms"] += 1.0
    elif fault == "routing":
        row["routing_history"][0]["dp_rank"] = 1
    elif fault == "admission":
        row["admission_history"][0]["reused_input_tokens"] = 1
    elif fault == "missing_latency":
        del head["behavior"]["p99_ttft_ms"]
    elif fault == "missing_route":
        del row["routing_history"][0]["dp_rank"]
    elif fault == "missing_identity":
        del head["model_identity"]["aggregated"]["tp"]
    else:
        row["routing_history"][0]["dp_rank"] = 0.0
    result = compare_case(CASE, value, revisions={"base": "base", "head": "head"}, rounds=5)
    assert result["classification"] == expected
    assert (result["invalid_reasons"] or result["behavior_changes"])[0]


def test_agentic_outcomes_and_identity_have_fixed_nested_fields():
    report = response("base", "availability")["behavior"]
    report.update(dict.fromkeys(TRAJECTORY_COUNTS, 1))
    report["incomplete_trajectories"] = 0
    report.update(dict.fromkeys(TRAJECTORY_METRICS, 3.0))
    report["agentic_play_outcomes"] = [
        {"play_id": "play", "status": "completed", "causal_terminal_ms": 3.0, "settled_at_ms": 3.0}
    ]
    row = report["per_request"][0]
    row.update(play_id="play", agentic={"request_id": "one", "play_id": "play", "conversation_id": "conversation"})
    expected = behavior(report, per_request=True, agentic=True)
    report["agentic_play_outcomes"][0]["diagnostic"] = 1
    row["agentic"]["diagnostic"] = 1
    assert behavior(report, per_request=True, agentic=True) == expected
    report["agentic_play_outcomes"][0]["settled_at_ms"] = 4.0
    assert "settled_at_ms" in difference(expected, behavior(report, per_request=True, agentic=True))
    del row["agentic"]["conversation_id"]
    with pytest.raises(ValueError, match="conversation_id"):
        behavior(report, per_request=True, agentic=True)


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
