#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantify delivery-only blocking on frozen E034 output; no replay feedback or qualification."""

import argparse
import bisect
import hashlib
import json
import math
import statistics
from pathlib import Path

POINTS = {1: "agentx-446063", 8: "agentx-446069", 14: "agentx-446062"}
PROFILE_SHA = "741c9b07352c57c267d0890403a0950435a048e32926c77456018d61025c8158"


def read(path):
    return json.loads(path.read_bytes())


def gate_time(ready_ms, arrivals, booked_until):
    """Zero-cost responses join behind input work already queued when ready."""
    index = bisect.bisect_right(arrivals, ready_ms)
    return max(ready_ms, booked_until[index - 1]) if index else ready_ms


def summary(pairs):
    if not pairs:
        return {"n": 0, "mape": None}
    if any(not math.isfinite(p) or p < 0 or not math.isfinite(r) or r <= 0 for p, r in pairs):
        raise ValueError("Invalid qualified latency")
    values = sorted(100 * abs(p - r) / r for p, r in pairs)

    def percentile(q):
        position = (len(values) - 1) * q
        lo = math.floor(position)
        hi = math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {
        "n": len(pairs),
        "mape": statistics.mean(values),
        "sum_ape": sum(values),
        "median_ape": statistics.median(values),
        "p95_ape": percentile(0.95),
        "p99_ape": percentile(0.99),
        "signed_percent": statistics.mean(100 * (p - r) / r for p, r in pairs),
        "mean_absolute_ms": statistics.mean(abs(p - r) for p, r in pairs),
    }


def bookings(requests, rate):
    jobs = []
    seen = set()
    for row in requests:
        if row["uuid"] in seen:
            raise ValueError("Duplicate native request UUID")
        seen.add(row["uuid"])
        arrival, end = row.get("dispatched_at_ms"), row.get("tokenized_ms")
        if arrival is None and end is None:
            continue
        if arrival is None or end is None:
            raise ValueError("Incomplete CPU booking; cannot reconstruct the shared queue")
        service = row["input_length"] * rate / 1000
        start = end - service
        if start + 1e-6 < arrival:
            raise ValueError("CPU booking violates measured service units")
        jobs.append((arrival, end, start))
    jobs.sort()
    previous = -math.inf
    arrivals, available = [], []
    for arrival, end, start in jobs:
        if start + 1e-6 < previous:
            raise ValueError("One-worker CPU bookings overlap")
        previous = end
        arrivals.append(arrival)
        available.append(max(end, available[-1] if available else end))
    return arrivals, available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.input_manifest)
    for name, expected in manifest["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe input manifest path")
        if hashlib.sha256((args.campaign / path).read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("Frozen experiment input changed")
    profile_path = args.campaign / "E034-component-tokenization-profile.json"
    if hashlib.sha256(profile_path.read_bytes()).hexdigest() != PROFILE_SHA:
        raise ValueError("Keep the original component profile; do not tune service cost")
    profile = read(profile_path)["tokenization"]
    if profile["workers"] != 1 or profile["base_ms"] != 0:
        raise ValueError("Expected the predeclared one-worker, zero-base profile")
    frozen = read(args.campaign / "frozen/frozen-cohorts.json")
    pooled = {
        metric: {variant: [] for variant in ("baseline_frozen", "baseline_retained", "source", "gated")}
        for metric in ("ttft_ms", "tpot_ms")
    }
    lost = dict.fromkeys(pooled, 0)
    results = []
    for lanes, point in POINTS.items():
        root = args.campaign / "E034-cpu-admission-dev" / point
        native = read(root / "native.json")
        source_score = read(root / "score.json")
        baseline = {
            row["request_id"]: row for row in read(args.campaign / "E003-baseline" / point / "score.json")["pairs"]
        }
        execution = read(root / "execution.json")
        if execution["tokenization_profile_sha256"] != PROFILE_SHA or execution["status"] != "completed":
            raise ValueError("Source replay differs from the frozen E034 experiment")
        requests = native["telemetry"]["requests"]
        arrivals, free_at = bookings(requests, profile["per_input_token_us"])
        predictions = {}
        for row in requests:
            if row.get("phase") != "profile":
                continue
            key = hashlib.sha256(row["request_id"].encode()).hexdigest()[:16]
            predictions.setdefault(key, []).append(row)
        transformed = {}
        after_grace = 0
        for row in source_score["pairs"]:
            if row["status"] != "matched":
                continue
            candidates = predictions.get(row["request_id"], [])
            if len(candidates) != 1:
                raise ValueError("Source matched identity is ambiguous or absent")
            pred = candidates[0]
            first, last = pred["first_token_ms"], pred["last_token_ms"]
            if abs(first - pred["arrival_time_ms"] - pred["ttft_ms"]) > 1e-6 or last < first:
                raise ValueError("Native timestamp units differ")
            for metric, key in (("ttft_ms", "ttft_ms"), ("tpot_ms", "itl_ms")):
                value = row["predicted_" + metric]
                if value is not None and (pred[key] is None or abs(value - pred[key]) > 1e-6):
                    raise ValueError("Source score does not reconcile with native request")
            first_received, last_received = (gate_time(t, arrivals, free_at) for t in (first, last))
            count = pred["output_length"]
            if count > 1 and abs((last - first) / (count - 1) - pred["itl_ms"]) > 1e-6:
                raise ValueError("Native TPOT arithmetic differs")
            transformed[row["request_id"]] = {
                "ttft_ms": pred["ttft_ms"] + first_received - first,
                "tpot_ms": (last_received - first_received) / (count - 1) if count > 1 else None,
            }
            if (
                gate_time(pred["terminal_time_ms"], arrivals, free_at)
                > native["agentic_profile"]["response_grace_deadline_ms"]
            ):
                after_grace += 1
        source_pairs = {row["request_id"]: row for row in source_score["pairs"]}
        per_metric = {}
        for metric in pooled:
            values = {variant: [] for variant in pooled[metric]}
            missing, changed, zeroed = [], 0, 0
            details = []
            for reference in frozen["cohorts"][point]["dashboard"]:
                if reference.get(metric + "_ape") is None:
                    continue
                key, actual = reference["request_id"], reference["silicon_" + metric]
                base = baseline[key]
                if base.get(metric + "_ape") is None or base["silicon_" + metric] != actual:
                    raise ValueError("Frozen research baseline identity changed")
                values["baseline_frozen"].append((base["predicted_" + metric], actual))
                source = source_pairs.get(key)
                if source is None or source.get(metric + "_ape") is None:
                    missing.append(key)
                    continue
                old, new = source["predicted_" + metric], transformed[key][metric]
                if new is None:
                    raise ValueError("Delivery diagnostic loses an eligible metric")
                values["baseline_retained"].append((base["predicted_" + metric], actual))
                values["source"].append((old, actual))
                values["gated"].append((new, actual))
                changed += abs(new - old) > 1e-9
                zeroed += new == 0
                details.append({"request_id": key, "reference_ms": actual, "source_ms": old, "gated_ms": new})
            for variant in values:
                pooled[metric][variant].extend(values[variant])
            lost[metric] += len(missing)
            tiny = [row for row in details if metric == "tpot_ms" and row["reference_ms"] < 0.1]
            sensitivity = {
                "n": len(tiny),
                "threshold_ms": 0.1,
                "source_contribution_pp": sum(
                    100 * abs(row["source_ms"] - row["reference_ms"]) / row["reference_ms"] for row in tiny
                )
                / len(details)
                if details
                else None,
                "gated_contribution_pp": sum(
                    100 * abs(row["gated_ms"] - row["reference_ms"]) / row["reference_ms"] for row in tiny
                )
                / len(details)
                if details
                else None,
            }
            per_metric[metric] = {
                "summaries": {v: summary(p) for v, p in values.items()},
                "lost_frozen": missing,
                "changed": changed,
                "zero_predictions": zeroed,
                "pairs": details,
                "small_reference_sensitivity": sensitivity if metric == "tpot_ms" else None,
            }
        results.append(
            {
                "point": point,
                "lanes": lanes,
                "cpu_bookings": len(arrivals),
                "metrics": per_metric,
                "source_coverage_unchanged": source_score["coverage"],
                "delivery_after_original_grace": after_grace,
            }
        )
    result = {
        "schema": "aisimulate.glm-rsi.delivery-only-diagnostic.v1",
        "qualified_model_result": False,
        "limitations": [
            "No closed-loop arrival or cache feedback",
            "Original missing observations retained",
            "FCFS input-first ties; zero-cost response jobs",
            "No point/throughput or publication artifact",
        ],
        "gpu_seconds": 0,
        "input_manifest_sha256": hashlib.sha256(args.input_manifest.read_bytes()).hexdigest(),
        "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "points": results,
        "pooled": {
            metric: {"summaries": {v: summary(p) for v, p in values.items()}, "lost_frozen": lost[metric]}
            for metric, values in pooled.items()
        },
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
