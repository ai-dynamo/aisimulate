# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Coverage and arithmetic checks for the finite-cohort comparison."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("qwen_compare", Path(__file__).with_name("compare.py"))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def planned(repeat=0):
    return {
        "case_id": f"p4-l1-cold-r{repeat}",
        "initial_input_tokens": 4,
        "lanes": 1,
        "initial_cache": "cold",
        "repeat": repeat,
        "measured_turns_per_lane": 2,
        "planned_requests": 2,
    }


def plan(cases):
    return {
        "output_tokens": 2,
        "tool_tokens": 1,
        "cases": cases,
        "num_cases": len(cases),
        "planned_requests": 2 * len(cases),
    }


def evidence(root, case, functional=False):
    case_id = case["case_id"]
    observed_root, replay_root = root / "observed", root / "replay"
    rows = []
    native = []
    for turn in range(2):
        request_id = f"{case_id}:measured:l0:t{turn}"
        prompt = [1, 2, 3, 4] + [9, 10, 20] * turn
        rows.append(
            {
                "request_id": request_id,
                "cell_id": case_id,
                "lane_id": 0,
                "turn_index": turn,
                "phase": "measured",
                "status": "completed",
                "input_token_ids": prompt,
                "output_token_ids": [9, 10],
                "cached_tokens": 4 if turn else 0,
                "timing": {
                    "dispatch_ms": 140 * turn,
                    "completed_ms": 210 if turn else 40,
                    "ttft_ms": 20 if turn else 10,
                    "tpot_ms": 40 if turn else 20,
                },
            }
        )
        native.append(
            {
                "request_id": request_id,
                "terminal_status": "completed",
                "input_length": len(prompt),
                "output_length": 2,
                "first_admission_reused_input_tokens": 4 if turn else 0,
                "admission_count": 1,
                "arrival_time_ms": 144 * turn,
                "terminal_time_ms": 208 if turn else 44,
                "ttft_ms": 30 if turn else 12,
                "itl_ms": 30 if turn else 22,
            }
        )
    info = {**case, "cell_id": case_id, "status": "complete"}
    source = observed_root / case_id
    source.mkdir(parents=True)
    (source / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    write(source / "case.json", info)
    write(observed_root / "cache-geometry.json", {"geometry_id": "geometry"})
    mapping = {
        "input_digest": compare.digest(rows),
        "geometry_id": "geometry",
        "functional_only": functional,
        "requests": {
            row["request_id"]: {key: row[key] for key in ("cell_id", "lane_id", "turn_index", "phase")} for row in rows
        },
    }
    replay = replay_root / case_id
    write(replay / "mapping.json", mapping)
    write(replay / "native-report.json", {"per_request": native})
    write(
        replay / "cache-gate.json",
        {
            "input_digest": mapping["input_digest"],
            "geometry_id": "geometry",
            "functional_only": functional,
            "passed": True,
        },
    )
    write(replay / "runner-metadata.json", {"prediction_provenance": {"aggregated": {"timing_source": "measured"}}})
    return observed_root, replay_root


def test_request_weighted_errors_and_finite_cohort_throughput(tmp_path):
    case = planned()
    observed, replay = evidence(tmp_path, case)
    report = compare.compare(plan([case]), observed, replay)
    assert report["timing_comparison_complete"]
    assert report["accuracy_pass"] is None
    metrics = report["summary"]["metrics"]
    assert metrics["ttft_ms"]["observed_mean"] == 15
    assert metrics["ttft_ms"]["predicted_mean"] == 21
    assert metrics["ttft_ms"]["mape_percent"] == 35
    assert metrics["ttft_ms"]["mean_signed_error"] == 6
    assert metrics["tpot_ms"]["mape_percent"] == 17.5
    assert metrics["tpot_ms"]["mean_signed_error"] == -4
    assert metrics["output_tokens_per_s"]["observed_mean"] == pytest.approx(4 / 0.210)
    assert metrics["output_tokens_per_s"]["predicted_mean"] == pytest.approx(4 / 0.208)
    assert report["cases"][0]["requests"][1]["metrics"]["tpot_ms"]["signed_percentage_error"] == -25


def test_missing_case_and_functional_replay_remain_in_denominators(tmp_path):
    cases = [planned(repeat) for repeat in range(3)]
    observed, replay = evidence(tmp_path, cases[0])
    evidence(tmp_path, cases[1], functional=True)
    report = compare.compare(plan(cases), observed, replay)
    assert not report["timing_comparison_complete"]
    assert report["summary"]["planned_cases"] == 3
    assert report["summary"]["qualified_cases"] == 1
    assert report["summary"]["planned_requests"] == 6
    assert report["summary"]["observed_completed_requests"] == 4
    assert report["summary"]["metrics"]["ttft_ms"]["observed_count"] == 4
    assert report["summary"]["metrics"]["ttft_ms"]["paired_count"] == 2
    assert "functional_only_prediction" in report["cases"][1]["issues"]
    assert report["cases"][1]["metrics"]["ttft_ms"]["predicted_mean"] is None
    assert "missing_silicon_case" in report["cases"][2]["issues"]
    assert report["summary"]["cache_gate_counts"] == {"passed": 2, "failed": 0, "missing": 1}


def test_cache_mismatch_keeps_diagnostic_pairs_but_excludes_accuracy(tmp_path):
    case = planned()
    observed, replay = evidence(tmp_path, case)
    path = replay / case["case_id"] / "native-report.json"
    native = json.loads(path.read_text())
    native["per_request"][1]["first_admission_reused_input_tokens"] = 0
    write(path, native)
    report = compare.compare(plan([case]), observed, replay)
    assert "cached_token_mismatch" in report["cases"][0]["issues"]
    assert report["cases"][0]["metrics"]["ttft_ms"]["paired_count"] == 2
    assert report["summary"]["metrics"]["ttft_ms"]["paired_count"] == 0
    assert report["summary"]["metrics"]["ttft_ms"]["mape_percent"] is None
    assert report["summary"]["observed_completed_requests"] == 2
    assert report["summary"]["cache_gate_counts"]["failed"] == 1
    assert report["cases"][0]["cache_gate_reported"] is True


def test_changed_trace_identity_and_noncompleted_predictions_are_explicit(tmp_path):
    case = planned()
    observed, replay = evidence(tmp_path, case)
    path = replay / case["case_id"] / "native-report.json"
    native = json.loads(path.read_text())
    native["per_request"][1]["terminal_status"] = "failed"
    write(path, native)
    path = replay / case["case_id"] / "mapping.json"
    mapping = json.loads(path.read_text())
    mapping["input_digest"] = "wrong-trace"
    write(path, mapping)
    report = compare.compare(plan([case]), observed, replay)
    issues = report["cases"][0]["issues"]
    assert "missing_or_mismatched_source_identity" in issues
    assert "missing_or_incomplete_prediction" in issues
    assert report["cases"][0]["metrics"]["ttft_ms"]["paired_count"] == 0
    assert report["cases"][0]["predicted_throughput"]["output_tokens_per_s"] is None


def test_zero_observations_keep_signed_error_without_infinite_ape():
    summary = compare.summarize([(0, 5), (2, 3), (None, 10), (4, None)])
    assert summary["observed_count"] == summary["predicted_count"] == 3
    assert summary["paired_count"] == 2
    assert summary["mean_signed_error"] == 3
    assert summary["mape_percent"] == 50
    assert summary["ape_defined_count"] == summary["zero_observed_pairs"] == 1
    assert compare.error_pair(0, 5)["ape_percent"] is None
    assert compare.summarize([])["mape_percent"] is None


def test_fixed_estimator_cannot_score_timing_even_if_mapping_claims_otherwise(tmp_path):
    case = planned()
    observed, replay = evidence(tmp_path, case)
    write(
        replay / case["case_id"] / "runner-metadata.json",
        {"prediction_provenance": {"aggregated": {"estimator": "fixed", "timing_source": "configured"}}},
    )
    report = compare.compare(plan([case]), observed, replay)
    assert "configured_fixed_timing" in report["cases"][0]["issues"]
    assert report["summary"]["metrics"]["ttft_ms"]["paired_count"] == 0
    assert report["cases"][0]["cache_gate"] is True


def test_actual_36_case_plan_preserves_missing_2016_request_cohort(tmp_path):
    actual_plan = json.loads(Path(__file__).with_name("campaign.json").read_text())
    report = compare.compare(actual_plan, tmp_path / "missing-observed", tmp_path / "missing-replay")
    assert len(report["cases"]) == 36
    assert len(report["operating_points"]) == 12
    assert report["summary"]["planned_requests"] == 2016
    assert report["summary"]["qualified_cases"] == 0
    assert report["summary"]["observed_completed_requests"] == 0
    assert report["summary"]["cache_gate_counts"]["missing"] == 36
    assert report["summary"]["metrics"]["ttft_ms"]["mape_percent"] is None
