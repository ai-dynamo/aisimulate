# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score point 446063 using explicit source IDs and successful profile cohorts.

No timing or cache values are calibrated. Native namespace mapping is supplied
as evidence; this helper does not infer identity from lengths or timestamps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUEST_ID = re.compile(r"(weka:[0-9a-f]{16}):request:outer:(\d+)(?::inner:(\d+))?$")
BENCHMARK_URL = "https://inferencex.semianalysis.com/inference/agentic/446063"


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def ape(reference: Any, simulated: Any) -> float | None:
    if not finite(reference) or reference <= 0 or not finite(simulated) or simulated < 0:
        return None
    return abs(simulated - reference) / reference * 100


def percentage(numerator: int, denominator: int) -> float | None:
    return 100 * numerator / denominator if denominator else None


def phase(row: dict) -> str:
    result = row.get("phase", row.get("agentic_phase", "profile"))
    return "profile" if result == "profiling" else result


def reference_key(row: dict) -> tuple[str, int, int | None]:
    return row["source_trace_id"], row["source_outer_idx"], row.get("source_inner_idx")


def native_key(row: dict, reverse_namespaces: dict) -> tuple[str, int, int | None] | None:
    identity = row.get("request_id") or (row.get("agentic") or {}).get("request_id", "")
    match = REQUEST_ID.search(identity)
    if not match or match[1] not in reverse_namespaces:
        return None
    return reverse_namespaces[match[1]], int(match[2]), int(match[3]) if match[3] is not None else None


def key_fields(key: tuple | None, namespaces: dict) -> dict:
    if key is None:
        return {"source_trace_id": None, "source_outer_idx": None, "source_inner_idx": None, "source_namespace": None}
    trace, outer, inner = key
    return {
        "source_trace_id": trace,
        "source_outer_idx": outer,
        "source_inner_idx": inner,
        "source_namespace": namespaces.get(trace),
    }


def cohort(rows: list[dict], reference: bool) -> dict:
    """Recompute the successful cohort numerator and first-start/last-end span."""
    successful = [
        row
        for row in rows
        if phase(row) == "profile"
        and (
            not row.get("error")
            if reference
            else row.get("terminal_status") == "completed" and finite(row.get("first_admit_ms"))
        )
    ]
    start_key, end_key = ("start_ns", "end_ns") if reference else ("arrival_time_ms", "terminal_time_ms")
    all_times = all(finite(row.get(start_key)) and finite(row.get(end_key)) for row in successful)
    start = min((row[start_key] for row in successful), default=None) if all_times else None
    end = max((row[end_key] for row in successful), default=None) if all_times else None
    divisor = 1e9 if reference else 1000
    span = (end - start) / divisor if start is not None and end is not None and end >= start else None
    tokens = {}
    for label, key in [
        ("input", "input_tokens" if reference else "input_length"),
        ("output", "output_tokens" if reference else "output_length"),
    ]:
        values = [row.get(key) for row in successful]
        tokens[label] = sum(values) if all(finite(value) and value >= 0 for value in values) else None
    e2e_key = "e2e_ms" if reference else "e2e_latency_ms"
    # The published InferenceX latency aggregation excludes non-positive values;
    # the native report retains a valid zero-ms observation.
    e2es = [
        row[e2e_key]
        for row in successful
        if finite(row.get(e2e_key)) and (row[e2e_key] > 0 if reference else row[e2e_key] >= 0)
    ]
    total = tokens["input"] + tokens["output"] if all(value is not None for value in tokens.values()) else None
    result = {
        "successful_requests": len(successful),
        "e2e_sample_count": len(e2es),
        "mean_e2e_ms": mean(e2es),
        "first_success_start": start,
        "last_success_end": end,
        "timestamp_unit": "ns" if reference else "ms",
        "successful_span_seconds": span,
        "input_tokens": tokens["input"],
        "output_tokens": tokens["output"],
        "total_tokens": total,
    }
    for label, count in [*tokens.items(), ("total", total)]:
        result[f"{label}_throughput_tok_s"] = (
            count / span if count is not None and span is not None and span > 0 else None
        )
    return result


def paired_record(
    ref: dict | None,
    sim: dict | None,
    key: tuple | None,
    namespaces: dict,
    match_status: str,
    candidates: list[str] | None = None,
) -> dict:
    ref = ref or {}
    sim = sim or {}
    row = {
        **key_fields(key, namespaces),
        "phase": phase(ref or sim),
        "match_status": match_status,
        "reference_request_id": ref.get("request_id"),
        "simulated_uuid": sim.get("uuid"),
        "simulated_request_id": sim.get("request_id"),
        "simulated_play_id": sim.get("play_id"),
        "simulated_terminal_status": sim.get("terminal_status"),
        "reference_error": ref.get("error"),
        "candidate_simulated_uuids": candidates or [],
        "reference_e2e_ms": ref.get("e2e_ms"),
        "simulated_e2e_ms": sim.get("e2e_latency_ms"),
        "reference_ttft_ms": ref.get("ttft_ms"),
        "simulated_ttft_ms": sim.get("ttft_ms"),
        "reference_tpot_ms": ref.get("tpot_ms"),
        "simulated_tpot_ms": sim.get("itl_ms"),
        "reference_input_tokens": ref.get("input_tokens"),
        "simulated_input_tokens": sim.get("input_length"),
        "reference_output_tokens": ref.get("output_tokens"),
        "simulated_output_tokens": sim.get("output_length"),
        "simulated_requested_output_tokens": sim.get("requested_output_length"),
        "reference_start_ns": ref.get("start_ns"),
        "reference_end_ns": ref.get("end_ns"),
        "simulated_arrival_time_ms": sim.get("arrival_time_ms"),
        "simulated_terminal_time_ms": sim.get("terminal_time_ms"),
    }
    for label in ["input", "output"]:
        left, right = row[f"reference_{label}_tokens"], row[f"simulated_{label}_tokens"]
        available = finite(left) and finite(right)
        row[f"{label}_length_delta_tokens"] = right - left if available else None
        row[f"{label}_length_matches"] = left == right if available else None
    reasons = []
    if row["phase"] != "profile":
        reasons.append("outside_profile")
    if match_status != "matched_unique_source":
        reasons.append(match_status)
    if ref.get("error"):
        reasons.append("reference_error")
    if sim and sim.get("terminal_status") != "completed":
        reasons.append(f"simulated_{sim.get('terminal_status', 'unknown_status')}")
    if sim and not finite(sim.get("first_admit_ms")):
        reasons.append("simulated_not_admitted")
    if not finite(row["reference_e2e_ms"]):
        reasons.append("reference_e2e_unavailable")
    elif row["reference_e2e_ms"] <= 0:
        reasons.append("reference_e2e_nonpositive")
    if not finite(row["simulated_e2e_ms"]) or row["simulated_e2e_ms"] < 0:
        reasons.append("simulated_e2e_unavailable")
    row["e2e_score_exclusions"] = reasons
    row["e2e_ape_percent"] = ape(row["reference_e2e_ms"], row["simulated_e2e_ms"]) if not reasons else None
    row["strict_output_e2e_eligible"] = row["e2e_ape_percent"] is not None and row["output_length_matches"] is True
    return row


def match_requests(reference: list[dict], native: list[dict], namespaces: dict) -> list[dict]:
    reverse = {value: key for key, value in namespaces.items()}
    if len(reverse) != len(namespaces):
        raise ValueError("Namespace map must be one-to-one")
    by_native = defaultdict(list)
    by_reference = Counter((phase(row), reference_key(row)) for row in reference)
    for row in native:
        key = native_key(row, reverse)
        if key is not None:
            by_native[phase(row), key].append(row)
    used = set()
    results = []
    for ref in reference:
        key = reference_key(ref)
        match_key = phase(ref), key
        candidates = by_native.get(match_key, [])
        if by_reference[match_key] > 1:
            status = "ambiguous_reference_source"
        elif len(candidates) > 1:
            status = "ambiguous_simulated_source"
        elif not candidates:
            status = "unmatched_reference"
        else:
            status = "matched_unique_source"
        sim = candidates[0] if status == "matched_unique_source" else None
        if sim is not None:
            used.add(sim["uuid"])
        results.append(paired_record(ref, sim, key, namespaces, status, [row["uuid"] for row in candidates]))
    for sim in native:
        if sim["uuid"] not in used:
            results.append(paired_record(None, sim, native_key(sim, reverse), namespaces, "unmatched_simulated"))
    return results


def score_subset(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["e2e_ape_percent"] is not None]
    return {
        "count": len(eligible),
        "mape_percent": mean([row["e2e_ape_percent"] for row in eligible]),
        "reference_mean_e2e_ms": mean([row["reference_e2e_ms"] for row in eligible]),
        "simulated_mean_e2e_ms": mean([row["simulated_e2e_ms"] for row in eligible]),
    }


def check_native_summary(prediction: dict, computed: dict) -> list[dict]:
    fields = {
        "completed_requests": "successful_requests",
        "total_input_tokens": "input_tokens",
        "total_output_tokens": "output_tokens",
        "mean_e2e_latency_ms": "mean_e2e_ms",
        "input_throughput_tok_s": "input_throughput_tok_s",
        "output_throughput_tok_s": "output_throughput_tok_s",
        "total_throughput_tok_s": "total_throughput_tok_s",
    }
    checks = []
    for published, recomputed in fields.items():
        left, right = prediction.get(published), computed.get(recomputed)
        comparable = finite(left) and finite(right)
        matches = math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7) if comparable else None
        checks.append({"field": published, "prediction": left, "recomputed": right, "matches": matches})
    if prediction.get("agentic_profile") is not None:
        span = computed["successful_span_seconds"]
        duration = prediction.get("duration_ms")
        matches = (
            math.isclose(duration, span * 1000, rel_tol=1e-9, abs_tol=1e-7)
            if finite(duration) and finite(span)
            else None
        )
        checks.append(
            {
                "field": "duration_ms",
                "prediction": duration,
                "recomputed": span * 1000 if span is not None else None,
                "matches": matches,
            }
        )
    failed = [row for row in checks if row["matches"] is False]
    if failed:
        raise ValueError(f"Native successful-cohort summary differs from terminal telemetry: {failed}")
    return checks


def analyze(
    prediction: dict, reference: list[dict], native: list[dict], namespaces: dict, execution: dict, variant: str
) -> tuple[dict, list[dict], dict]:
    wall = execution.get("wall_time_seconds")
    if not finite(wall) or wall < 0:
        raise ValueError("execution.json must contain non-negative wall_time_seconds measured around the CLI")
    if execution.get("returncode", execution.get("exit_code", 0)) != 0:
        raise ValueError("Cannot score an execution whose recorded exit code is nonzero")
    if len({row["uuid"] for row in native}) != len(native):
        raise ValueError("Native request UUIDs must be unique")
    paired = match_requests(reference, native, namespaces)
    profile_reference = [row for row in reference if phase(row) == "profile"]
    successful_reference = [row for row in profile_reference if not row.get("error")]
    scoreable_reference = [row for row in successful_reference if finite(row.get("e2e_ms")) and row["e2e_ms"] > 0]
    matched = [row for row in paired if row["phase"] == "profile" and row["match_status"] == "matched_unique_source"]
    successful_matched = [
        row for row in matched if not row["reference_error"] and row["simulated_terminal_status"] == "completed"
    ]
    all_score = score_subset(matched)
    strict_score = score_subset([row for row in matched if row["output_length_matches"] is True])
    exact_lengths_score = score_subset(
        [row for row in matched if row["output_length_matches"] is True and row["input_length_matches"] is True]
    )
    ref_cohort, sim_cohort = cohort(reference, True), cohort(native, False)
    summary_checks = check_native_summary(prediction, sim_cohort)
    coverage = {
        "reference_profile_requests": len(profile_reference),
        "reference_successes": len(successful_reference),
        "reference_errors_excluded": len(profile_reference) - len(successful_reference),
        "reference_positive_e2e_requests": len(scoreable_reference),
        "uniquely_source_matched_profile_requests": len(matched),
        "successful_source_matched_requests": len(successful_matched),
        "source_match_coverage_percent": percentage(
            len([row for row in matched if not row["reference_error"]]), len(successful_reference)
        ),
        "scored_e2e_coverage_percent": percentage(all_score["count"], len(scoreable_reference)),
        "strict_output_e2e_coverage_percent": percentage(strict_score["count"], len(scoreable_reference)),
        "paired_input_length_mismatches": sum(row["input_length_matches"] is False for row in matched),
        "paired_output_length_mismatches": sum(row["output_length_matches"] is False for row in matched),
        "native_profile_terminal_statuses": dict(
            Counter(row.get("terminal_status") for row in native if phase(row) == "profile")
        ),
        "matching_statuses_all_phases": dict(Counter(row["match_status"] for row in paired)),
        "scoring_exclusions_all_phases": dict(
            Counter(reason for row in paired for reason in row["e2e_score_exclusions"])
        ),
    }
    comparisons = []
    for metric, result, scope in [
        (
            "E2E · source-matched",
            all_score,
            "Request-level MAPE: unique source IDs; both successful; positive reference E2E; output lengths may differ",
        ),
        (
            "E2E · output-length matched",
            strict_score,
            (
                "Request-level MAPE: source-matched successful pairs with equal actual output length; input"
                " lengths may differ"
            ),
        ),
    ]:
        comparisons.append(
            {
                "metric": metric,
                "unit": "ms",
                "reference": result["reference_mean_e2e_ms"],
                "simulated": result["simulated_mean_e2e_ms"],
                "mape_percent": result["mape_percent"],
                "count": result["count"],
                "scope": scope,
            }
        )
    for metric, key, unit in [
        ("Mean E2E · full profile", "mean_e2e_ms", "ms"),
        ("Output throughput · full profile", "output_throughput_tok_s", "tokens/s"),
        ("Input throughput · full profile", "input_throughput_tok_s", "tokens/s"),
        ("Total throughput · full profile", "total_throughput_tok_s", "tokens/s"),
    ]:
        left, right = ref_cohort[key], sim_cohort[key]
        comparisons.append(
            {
                "metric": metric,
                "unit": unit,
                "reference": left,
                "simulated": right,
                "mape_percent": ape(left, right),
                "count": 1,
                "scope": (
                    "One aggregate data point (N=1): APE, not request-level MAPE; each full successful profile "
                    "cohort uses its own realized requests and span"
                ),
            }
        )
    limitations = [
        (
            "Source identity alone does not establish equal payloads, scheduling, or cache history. "
            "Input and output token mismatches are exported for every pair."
        ),
        (
            "Strict output-length matching does not require equal input length; the additional "
            "exact-input-and-output subset is in accuracy-summary.json."
        ),
        (
            "Reference errors, canceled/failed simulation outcomes, missing latencies, and non-positive"
            " reference E2E are retained but excluded from request-level MAPE."
        ),
        (
            "Repeated source IDs without a unique occurrence match are reported as ambiguous and "
            "excluded; no nearest-time or nearest-length matching is used."
        ),
        (
            "Full-profile aggregate comparisons have N=1 and can compare different realized request "
            "cohorts; their errors are APE, not population MAPE."
        ),
        (
            "Unfinished work at a simulation time cap has no terminal request row; telemetry completion"
            " status and gauges retain that limitation."
        ),
    ]
    methodology = [
        (
            "Match normalized profile phase and exact (source trace namespace, outer request index, "
            "optional inner request index). Namespace mapping follows native Weka identities."
        ),
        (
            "Request-level MAPE = mean(100 * abs(simulated E2E - reference E2E) / reference E2E) over "
            "eligible successful pairs."
        ),
        (
            "Profile throughput = successful profile token count / (last successful terminal time - "
            "first successful arrival time). Reference nanoseconds and simulation milliseconds are "
            "converted to seconds."
        ),
        (
            "Reference latency means use positive successful latency observations; native means follow "
            "prediction.json and retain valid zero-ms observations."
        ),
        (
            "CLI wall time is read from execution.json and includes its declared process scope; "
            "download, build, and plotting time are not silently substituted."
        ),
        "No timing, cache-hit, or error correction factor is fitted or applied.",
    ]
    measurement = {
        "schema": "aisimulate.agentx.comparison.v1",
        "title": f"AgentX 446063 · {variant}",
        "benchmark_url": BENCHMARK_URL,
        "variant": variant,
        "wall_time_seconds": wall,
        "wall_time_scope": execution.get("wall_time_scope"),
        "execution": execution,
        "comparisons": comparisons,
        "coverage": coverage,
        "methodology": methodology,
        "limitations": limitations,
        "request_pairs": [
            {
                "metric": "E2E",
                "reference": row["reference_e2e_ms"],
                "simulated": row["simulated_e2e_ms"],
                "phase": row["phase"],
                "request_id": row["reference_request_id"],
                "output_length_matches": row["output_length_matches"],
            }
            for row in paired
            if row["e2e_ape_percent"] is not None
        ],
    }
    summary = {
        "schema": "aisimulate.agentx.accuracy.v1",
        "point_id": 446063,
        "variant": variant,
        "coverage": coverage,
        "request_e2e_source_matched": all_score,
        "request_e2e_output_length_matched": strict_score,
        "request_e2e_input_and_output_length_matched": exact_lengths_score,
        "reference_successful_profile": ref_cohort,
        "native_successful_profile": sim_cohort,
        "native_summary_checks": summary_checks,
        "aggregate_comparisons": comparisons[2:],
        "wall_time_seconds": wall,
        "wall_time_scope": execution.get("wall_time_scope"),
        "telemetry_completion_status": prediction.get("telemetry", {}).get("completion_status"),
        "methodology": methodology,
        "limitations": limitations,
    }
    return summary, paired, measurement


def self_test() -> None:
    namespaces = {"trace-a": "weka:0000000000000001", "trace-b": "weka:0000000000000002"}

    def ref(outer, e2e, start, output, inner=None, error=None):
        return {
            "source_trace_id": "trace-a",
            "source_outer_idx": outer,
            "source_inner_idx": inner,
            "request_id": f"ref-{outer}",
            "phase": "profiling",
            "start_ns": int(start * 1e6),
            "end_ns": int((start + (e2e or 0)) * 1e6),
            "e2e_ms": e2e,
            "input_tokens": 100,
            "output_tokens": output,
            "error": error,
        }

    def sim(outer, e2e, start, output, inner=None, status="completed", inputs=100):
        suffix = f":inner:{inner}" if inner is not None else ""
        return {
            "request_id": f"weka:0000000000000001:request:outer:{outer}{suffix}",
            "uuid": f"sim-{outer}",
            "phase": "profile",
            "first_admit_ms": start,
            "arrival_time_ms": start,
            "terminal_time_ms": start + e2e,
            "e2e_latency_ms": e2e,
            "input_length": inputs,
            "output_length": output,
            "terminal_status": status,
        }

    refs = [
        ref(1, 100, 0, 10),
        ref(2, 200, 200, 20, inner=0),
        ref(3, None, 450, None, error={"type": "NoContent"}),
        ref(4, 50, 500, 5),
        ref(5, 0, 600, 1),
    ]
    native = [
        sim(1, 110, 0, 10, inputs=99),
        sim(2, 180, 200, 21, inner=0),
        sim(3, 40, 450, 3),
        sim(4, 9499, 500, 2, status="canceled"),
        sim(5, 0, 600, 1),
    ]
    execution = {"wall_time_seconds": 2, "returncode": 0}
    summary, pairs, measurement = analyze({}, refs, native, namespaces, execution, "synthetic self-test")
    assert summary["request_e2e_source_matched"]["count"] == 2
    assert summary["request_e2e_source_matched"]["mape_percent"] == 10
    assert summary["request_e2e_output_length_matched"]["count"] == 1
    assert summary["request_e2e_input_and_output_length_matched"]["count"] == 0
    assert summary["request_e2e_input_and_output_length_matched"]["mape_percent"] is None
    assert summary["coverage"]["paired_input_length_mismatches"] == 1
    assert summary["coverage"]["paired_output_length_mismatches"] == 2  # mismatched success + retained cancellation
    assert summary["native_successful_profile"]["output_tokens"] == 35
    assert summary["native_successful_profile"]["successful_span_seconds"] == 0.6
    assert math.isclose(summary["native_successful_profile"]["output_throughput_tok_s"], 35 / 0.6)
    assert any(row["simulated_terminal_status"] == "canceled" for row in pairs)
    assert any(row["reference_error"] for row in pairs)
    assert all(row["count"] == 1 for row in measurement["comparisons"][2:])
    assert ape(0, 0) is None and ape(0, 1) is None and ape(1, None) is None and ape(1, 0) == 100
    assert cohort([], False)["output_throughput_tok_s"] is None
    zero = sim(5, 0, 0, 1)
    assert cohort([zero], False)["output_throughput_tok_s"] is None
    assert native_key(sim(2, 1, 0, 1), {v: k for k, v in namespaces.items()}) != native_key(
        native[1], {v: k for k, v in namespaces.items()}
    )
    ambiguous = [*native, {**native[0], "uuid": "duplicate-source"}]
    duplicate_summary, duplicate_pairs, _ = analyze({}, refs, ambiguous, namespaces, execution, "duplicate test")
    assert duplicate_summary["request_e2e_source_matched"]["count"] == 1
    assert any(row["match_status"] == "ambiguous_simulated_source" for row in duplicate_pairs)
    assert sum(row["match_status"] == "unmatched_simulated" for row in duplicate_pairs) == 2
    foreign = {**native[0], "request_id": "weka:0000000000000002:request:outer:1", "uuid": "other-trace"}
    assert native_key(foreign, {v: k for k, v in namespaces.items()}) != reference_key(refs[0])
    try:
        check_native_summary({"total_output_tokens": 100}, cohort(native, False))
    except ValueError:
        pass
    else:
        raise AssertionError("Inconsistent published native numerator must be rejected")
    print(
        "Self-test passed: exact namespace/outer/inner joins, ambiguity, errors/cancellation, "
        "length filters, cohort span, zeros, summary consistency"
    )


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def provenance(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run-dir", "reference", "namespace-map", "execution", "output-dir"]:
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ["run_dir", "reference", "namespace_map", "execution", "output_dir"]:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required unless --self-test is used")
    prediction_path = args.run_dir / "prediction.json"
    requests_path = args.run_dir / "telemetry-requests.jsonl"
    prediction = json.loads(prediction_path.read_text())
    reference = json.loads(args.reference.read_text())
    namespaces = json.loads(args.namespace_map.read_text())
    execution = json.loads(args.execution.read_text())
    native = read_rows(requests_path)
    if prediction.get("telemetry", {}).get("clock") != "absolute_virtual_ms":
        raise ValueError("Expected native all-phase telemetry on the absolute virtual clock")
    variant = args.variant or execution.get("variant", "unspecified replay variant")
    summary, pairs, measurement = analyze(prediction, reference, native, namespaces, execution, variant)
    sources = {
        name: provenance(path)
        for name, path in [
            ("prediction", prediction_path),
            ("requests", requests_path),
            ("reference", args.reference),
            ("namespace_map", args.namespace_map),
            ("execution", args.execution),
        ]
    }
    summary["provenance"] = sources
    measurement["provenance"] = sources
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("accuracy-summary.json", summary),
        ("paired-requests.json", pairs),
        ("measurement.json", measurement),
    ]:
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "paired-requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]) if pairs else [])
        writer.writeheader()
        for row in pairs:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )
    print(
        json.dumps(
            {
                "output_directory": str(args.output_dir),
                "coverage": summary["coverage"],
                "request_e2e_source_matched": summary["request_e2e_source_matched"],
                "request_e2e_output_length_matched": summary["request_e2e_output_length_matched"],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
