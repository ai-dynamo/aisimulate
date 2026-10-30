#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pair the planned silicon cohort with AISim replay, retaining coverage failures.

No numerical acceptance threshold is assumed. Functional-only fixed-timing
replays never contribute latency or throughput predictions. APE is undefined
when the observed value is zero; those pairs remain in the explicit counts.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

REQUEST_METRICS = ("ttft_ms", "tpot_ms", "e2e_latency_ms")
COHORT_METRICS = ("requests_per_s", "output_tokens_per_s", "cohort_makespan_ms")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def summarize(pairs):
    observed = [actual for actual, _ in pairs if number(actual)]
    predicted = [estimate for _, estimate in pairs if number(estimate)]
    matched = [(actual, estimate) for actual, estimate in pairs if number(actual) and number(estimate)]
    signed = [estimate - actual for actual, estimate in matched]
    percentages = [(estimate - actual) / actual * 100 for actual, estimate in matched if actual > 0]
    return {
        "observed_count": len(observed),
        "predicted_count": len(predicted),
        "paired_count": len(matched),
        "observed_mean": mean(observed) if observed else None,
        "predicted_mean": mean(predicted) if predicted else None,
        "paired_observed_mean": mean(actual for actual, _ in matched) if matched else None,
        "paired_predicted_mean": mean(estimate for _, estimate in matched) if matched else None,
        "mean_signed_error": mean(signed) if signed else None,
        "mean_signed_percentage_error": mean(percentages) if percentages else None,
        "mape_percent": mean(abs(value) for value in percentages) if percentages else None,
        "ape_defined_count": len(percentages),
        "zero_observed_pairs": len(matched) - len(percentages),
    }


def error_pair(actual, estimate):
    valid = number(actual) and number(estimate)
    return {
        "observed": actual,
        "predicted": estimate,
        "signed_error": estimate - actual if valid else None,
        "signed_percentage_error": (estimate - actual) / actual * 100 if valid and actual > 0 else None,
        "ape_percent": abs(estimate - actual) / actual * 100 if valid and actual > 0 else None,
    }


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def request_latency(row, predicted=False):
    if not row:
        return None
    start = row.get("arrival_time_ms") if predicted else row.get("timing", {}).get("dispatch_ms")
    end = row.get("terminal_time_ms") if predicted else row.get("timing", {}).get("completed_ms")
    return end - start if number(start) and number(end) and end >= start else None


def throughput(rows, expected, predicted=False):
    if len(rows) != expected or not all(
        row.get("terminal_status" if predicted else "status") == "completed" for row in rows
    ):
        return {
            "requests_per_s": None,
            "output_tokens_per_s": None,
            "duration_ms": None,
            "reason": "incomplete measured cohort",
        }
    starts = [row.get("arrival_time_ms") if predicted else row.get("timing", {}).get("dispatch_ms") for row in rows]
    ends = [row.get("terminal_time_ms") if predicted else row.get("timing", {}).get("completed_ms") for row in rows]
    if not all(number(value) for value in starts + ends) or max(ends) <= min(starts):
        return {
            "requests_per_s": None,
            "output_tokens_per_s": None,
            "duration_ms": None,
            "reason": "missing or invalid measured-cohort clock",
        }
    duration = max(ends) - min(starts)
    token_counts = [row.get("output_length") if predicted else len(row.get("output_token_ids", [])) for row in rows]
    if not all(type(value) is int and value >= 0 for value in token_counts):
        return {
            "requests_per_s": None,
            "output_tokens_per_s": None,
            "duration_ms": duration,
            "reason": "missing output token counts",
        }
    tokens = sum(token_counts)
    return {
        "requests_per_s": expected * 1000 / duration,
        "output_tokens_per_s": tokens * 1000 / duration,
        "duration_ms": duration,
        "reason": None,
    }


def compare_case(plan, observed_root, replay_root, output_tokens, tool_tokens):
    case_id = plan["case_id"]
    source = observed_root / case_id
    replay = replay_root / case_id
    path = source / "requests.jsonl"
    all_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    rows = [row for row in all_rows if row.get("phase") != "kernel_warmup"]
    measured = [row for row in rows if row.get("phase") == "measured"]
    info = read_json(source / "case.json")
    mapping = read_json(replay / "mapping.json")
    native = read_json(replay / "native-report.json")
    gate = read_json(replay / "cache-gate.json")
    provenance = read_json(replay / "runner-metadata.json")
    expected = plan["lanes"] * plan["measured_turns_per_lane"]
    expected_ids = {
        f"{case_id}:measured:l{lane}:t{turn}": (lane, turn)
        for lane in range(plan["lanes"])
        for turn in range(plan["measured_turns_per_lane"])
    }
    issues = []
    if plan["planned_requests"] != expected:
        issues.append("inconsistent_plan_count")
    if info is None or not path.exists():
        issues.append("missing_silicon_case")
    elif info.get("status") != "complete":
        issues.append("silicon_case_incomplete")
    if info and any(
        info.get(key) != plan[key]
        for key in ("initial_input_tokens", "lanes", "initial_cache", "measured_turns_per_lane")
    ):
        issues.append("silicon_configuration_mismatch")
    source_ids = [row.get("request_id") for row in rows]
    if len(set(source_ids)) != len(source_ids):
        issues.append("duplicate_source_request_ids")
    actual = {row.get("request_id"): row for row in measured}
    if set(actual) != set(expected_ids) or len(measured) != expected:
        issues.append("silicon_cohort_mismatch")
    if any(row.get("phase") not in ("warmup", "measured") for row in rows):
        issues.append("unexpected_source_phase")
    predictions = native.get("per_request", []) if native else []
    predicted = {row.get("request_id"): row for row in predictions}
    if native is None:
        issues.append("missing_prediction")
    elif len(predicted) != len(predictions) or set(predicted) != set(source_ids):
        issues.append("replay_cohort_mismatch")
    source_matches = bool(mapping) and mapping.get("input_digest") == digest(rows)
    if source_matches:
        identities = mapping.get("requests", {})
        source_matches = set(identities) == set(source_ids) and all(
            all(
                identities[row["request_id"]].get(key) == row.get(key)
                for key in ("cell_id", "lane_id", "turn_index", "phase")
            )
            for row in rows
        )
    if not source_matches:
        issues.append("missing_or_mismatched_source_identity")
    functional = mapping.get("functional_only") if mapping else None
    if functional is True:
        issues.append("functional_only_prediction")
    elif functional is not False:
        issues.append("missing_timing_mode_identity")
    if provenance is None or not provenance.get("prediction_provenance"):
        issues.append("missing_prediction_provenance")
    configured_fixed = bool(provenance) and any(
        item.get("estimator") == "fixed" for item in provenance.get("prediction_provenance", {}).values()
    )
    if configured_fixed:
        issues.append("configured_fixed_timing")
    timing_usable = functional is False and not configured_fixed
    gate_matches = (
        source_matches
        and bool(gate and mapping)
        and all(gate.get(key) == mapping.get(key) for key in ("input_digest", "geometry_id", "functional_only"))
    )
    if not gate_matches:
        issues.append("missing_or_mismatched_cache_gate")
    elif gate.get("passed") is not True:
        issues.append("cache_gate_failed")
    geometry = read_json(observed_root / "cache-geometry.json")
    if not geometry:
        issues.append("missing_cache_geometry")
    elif mapping and geometry.get("geometry_id") != mapping.get("geometry_id"):
        issues.append("cache_geometry_mismatch")
    recomputed_cache_matches = bool(rows) and all(
        row.get("status") == "completed"
        and predicted.get(row["request_id"], {}).get("terminal_status") == "completed"
        and row.get("cached_tokens") is not None
        and row["cached_tokens"] == predicted[row["request_id"]].get("first_admission_reused_input_tokens")
        and predicted[row["request_id"]].get("admission_count", 1) == 1
        and len(row.get("input_token_ids", [])) == predicted[row["request_id"]].get("input_length")
        and len(row.get("output_token_ids", [])) == predicted[row["request_id"]].get("output_length")
        for row in rows
    )
    if gate_matches and gate.get("passed") is True and not recomputed_cache_matches:
        issues.append("cache_gate_contradicts_request_evidence")
    request_pairs = []
    for request_id, (lane, turn) in expected_ids.items():
        observed, estimate = actual.get(request_id), predicted.get(request_id)
        request_issues = []
        observed_ok = bool(observed and observed.get("status") == "completed")
        estimate_ok = bool(estimate and estimate.get("terminal_status") == "completed")
        if not observed_ok:
            request_issues.append("missing_or_incomplete_silicon_request")
        if not estimate_ok:
            request_issues.append("missing_or_incomplete_prediction")
        if observed_ok:
            input_length = plan["initial_input_tokens"] + turn * (output_tokens + tool_tokens)
            if (observed.get("cell_id"), observed.get("lane_id"), observed.get("turn_index")) != (case_id, lane, turn):
                request_issues.append("request_identity_mismatch")
            if (
                len(observed.get("input_token_ids", [])) != input_length
                or len(observed.get("output_token_ids", [])) != output_tokens
            ):
                request_issues.append("silicon_token_count_mismatch")
            previous = actual.get(f"{case_id}:measured:l{lane}:t{turn - 1}")
            if previous:
                prefix = previous["input_token_ids"] + previous["output_token_ids"]
                if observed["input_token_ids"][: len(prefix)] != prefix:
                    request_issues.append("broken_literal_token_prefix")
        if observed_ok and estimate_ok:
            if estimate.get("input_length") != len(observed["input_token_ids"]) or estimate.get("output_length") != len(
                observed["output_token_ids"]
            ):
                request_issues.append("replay_token_count_mismatch")
            if observed.get("cached_tokens") is None or observed.get("cached_tokens") != estimate.get(
                "first_admission_reused_input_tokens"
            ):
                request_issues.append("cached_token_mismatch")
            if estimate.get("admission_count", 1) != 1:
                request_issues.append("preempted_request_requires_admission_comparison")
        metrics = {}
        for name, native_name in (("ttft_ms", "ttft_ms"), ("tpot_ms", "itl_ms"), ("e2e_latency_ms", None)):
            actual_value = (
                (request_latency(observed) if native_name is None else observed.get("timing", {}).get(name))
                if observed_ok
                else None
            )
            estimate_value = (
                (request_latency(estimate, predicted=True) if native_name is None else estimate.get(native_name))
                if estimate_ok and source_matches and timing_usable
                else None
            )
            if not number(actual_value):
                actual_value = None
                request_issues.append(f"missing_silicon_{name}")
            if not number(estimate_value):
                estimate_value = None
                request_issues.append(f"missing_prediction_{name}")
            metrics[name] = error_pair(actual_value, estimate_value)
        request_pairs.append(
            {
                "request_id": request_id,
                "lane_id": lane,
                "turn_index": turn,
                "issues": request_issues,
                "metrics": metrics,
            }
        )
    request_issue_counts = Counter(issue for row in request_pairs for issue in row["issues"])
    issues.extend(request_issue_counts)
    observed_throughput = throughput(list(actual.values()), expected)
    predicted_measured = [predicted[request_id] for request_id in expected_ids if request_id in predicted]
    predicted_throughput = (
        throughput(predicted_measured, expected, predicted=True)
        if source_matches and timing_usable
        else {
            "requests_per_s": None,
            "output_tokens_per_s": None,
            "duration_ms": None,
            "reason": "prediction identity missing or functional-only timing",
        }
    )
    if observed_throughput["reason"]:
        issues.append("missing_silicon_throughput")
    if predicted_throughput["reason"]:
        issues.append("missing_prediction_throughput")
    metrics = {
        name: summarize(
            [(row["metrics"][name]["observed"], row["metrics"][name]["predicted"]) for row in request_pairs]
        )
        for name in REQUEST_METRICS
    }
    for name in ("requests_per_s", "output_tokens_per_s"):
        metrics[name] = error_pair(observed_throughput[name], predicted_throughput[name])
    metrics["cohort_makespan_ms"] = error_pair(observed_throughput["duration_ms"], predicted_throughput["duration_ms"])
    return {
        "case_id": case_id,
        "initial_input_tokens": plan["initial_input_tokens"],
        "lanes": plan["lanes"],
        "initial_cache": plan["initial_cache"],
        "repeat": plan["repeat"],
        "status": "paired" if not issues else "incomplete_or_unqualified",
        "timing_comparison_qualified": not issues,
        "issues": sorted(set(issues)),
        "request_issue_counts": dict(request_issue_counts),
        "planned_requests": expected,
        "observed_measured_requests": len(measured),
        "observed_completed_requests": sum(row.get("status") == "completed" for row in measured),
        "replay_completed_requests": sum(row.get("terminal_status") == "completed" for row in predicted_measured),
        "unexpected_source_ids": sorted(set(actual) - set(expected_ids)),
        "unexpected_replay_ids": sorted(set(predicted) - set(source_ids)),
        "source_identity_matches": source_matches,
        "cache_gate": gate.get("passed") is True and recomputed_cache_matches if gate_matches else None,
        "cache_gate_reported": gate.get("passed") if gate else None,
        "prediction_provenance": provenance.get("prediction_provenance") if provenance else None,
        "launch_timing_boundary": mapping.get("launch_timing_boundary") if mapping else None,
        "observed_phase_launch_spread_ms": mapping.get("observed_phase_launch_spread_ms") if mapping else None,
        "observed_throughput": observed_throughput,
        "predicted_throughput": predicted_throughput,
        "metrics": metrics,
        "requests": request_pairs,
    }


def aggregate(cases):
    metrics = {}
    for name in REQUEST_METRICS:
        metrics[name] = summarize(
            [
                (
                    row["metrics"][name]["observed"],
                    row["metrics"][name]["predicted"] if case["timing_comparison_qualified"] else None,
                )
                for case in cases
                for row in case["requests"]
            ]
        )
    for name in COHORT_METRICS:
        metrics[name] = summarize(
            [
                (
                    case["metrics"][name]["observed"],
                    case["metrics"][name]["predicted"] if case["timing_comparison_qualified"] else None,
                )
                for case in cases
            ]
        )
    return {
        "planned_cases": len(cases),
        "qualified_cases": sum(case["timing_comparison_qualified"] for case in cases),
        "planned_requests": sum(case["planned_requests"] for case in cases),
        "observed_completed_requests": sum(case["observed_completed_requests"] for case in cases),
        "replay_completed_requests": sum(case["replay_completed_requests"] for case in cases),
        "cache_gate_counts": {
            name: sum(case["cache_gate"] is value for case in cases)
            for name, value in (("passed", True), ("failed", False), ("missing", None))
        },
        "case_issue_counts": dict(Counter(issue for case in cases for issue in case["issues"])),
        "metrics": metrics,
    }


def compare(plan, observed_root, replay_root):
    cases = [
        compare_case(case, observed_root, replay_root, plan["output_tokens"], plan["tool_tokens"])
        for case in plan["cases"]
    ]
    keys = sorted({(case["initial_input_tokens"], case["lanes"], case["initial_cache"]) for case in cases})
    groups = []
    for length, lanes, cache in keys:
        selected = [
            case
            for case in cases
            if (case["initial_input_tokens"], case["lanes"], case["initial_cache"]) == (length, lanes, cache)
        ]
        groups.append({"initial_input_tokens": length, "lanes": lanes, "initial_cache": cache, **aggregate(selected)})
    expected_ids = {case["case_id"] for case in plan["cases"]}
    extra_silicon = sorted(
        path.parent.name for path in observed_root.glob("*/requests.jsonl") if path.parent.name not in expected_ids
    )
    extra_replay = sorted(
        path.parent.name for path in replay_root.glob("*/native-report.json") if path.parent.name not in expected_ids
    )
    plan_errors = []
    if len(expected_ids) != len(cases) or plan.get("num_cases", len(cases)) != len(cases):
        plan_errors.append("inconsistent_case_count")
    if plan.get("planned_requests", sum(case["planned_requests"] for case in cases)) != sum(
        case["planned_requests"] for case in cases
    ):
        plan_errors.append("inconsistent_planned_request_count")
    return {
        "schema": "qwen35-agentic-comparison/v1",
        "numerical_acceptance_threshold": None,
        "accuracy_pass": None,
        "timing_comparison_complete": not plan_errors
        and not extra_silicon
        and not extra_replay
        and all(case["timing_comparison_qualified"] for case in cases),
        "plan_errors": plan_errors,
        "unexpected_silicon_cases": extra_silicon,
        "unexpected_replay_cases": extra_replay,
        "definitions": {
            "signed_error": "predicted minus observed; positive means overprediction",
            "mape": "mean absolute percentage error for request/cohort pairs; zero denominators excluded and counted",
            "group_latency": "request-weighted across repeats; aggregate predictions require qualified cases",
            "e2e_latency_ms": "observed completion minus dispatch; replay terminal time minus arrival",
            "observed_population": "all available measured requests; paired means use the qualified matched cohort",
            "group_throughput": "mean across repeated finite cohorts; excludes warmup; output tokens / cohort duration",
            "cohort_duration": "first measured dispatch to last completion; replay arrival to terminal time",
            "cohort_makespan_ms": "same measured-cohort duration, paired and averaged across repeats",
            "case_metrics": "raw diagnostic pairs retained when cache or coverage issues disqualify a case",
        },
        "summary": aggregate(cases),
        "operating_points": groups,
        "cases": cases,
        "cold_warm_comparisons": (read_json(observed_root / "collector.json") or {}).get("cold_warm_comparisons", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("campaign.json"))
    parser.add_argument("--observed", type=Path, required=True, help="Collector campaign directory")
    parser.add_argument("--replay", type=Path, required=True, help="Replay directory containing per-case results")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    args = parser.parse_args()
    report = compare(json.loads(args.plan.read_text()), args.observed, args.replay)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "timing_comparison_complete": report["timing_comparison_complete"],
                "qualified_cases": report["summary"]["qualified_cases"],
                "planned_cases": report["summary"]["planned_cases"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
