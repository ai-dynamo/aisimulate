# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build collection summaries and request charts from selected, pinned evidence."""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict

from scripts.fpm_accuracy.hf import parquet


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def quantile(values, percentile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def histogram(values):
    values = [v for v in values if number(v)]
    if not values:
        return {"count": 0, "bins": [], "p50": None, "p90": None}
    positive = [v for v in values if v > 0]
    bins = []
    zeros = len(values) - len(positive)
    if zeros:
        bins.append({"lower": 0, "upper": 0, "count": zeros})
    if positive:
        lo, hi = min(positive), max(positive)
        if lo == hi:
            bins.append({"lower": lo, "upper": hi, "count": len(positive)})
        else:
            count = min(32 - bool(zeros), len(set(positive)))
            edges = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / count) for i in range(count + 1)]
            edges[0], edges[-1] = lo, hi
            counts = [0] * count
            for value in positive:
                counts[min(count - 1, max(0, bisect.bisect_right(edges, value) - 1))] += 1
            bins.extend({"lower": edges[i], "upper": edges[i + 1], "count": counts[i]} for i in range(count))
    return {"count": len(values), "bins": bins, "p50": quantile(values, 0.5), "p90": quantile(values, 0.9)}


def charts(records):
    records = list(records)
    selected = [r for r in records if r["stage"] == "profiling"]
    ttft, interactivity = [], []
    for r in sorted(selected, key=lambda r: (r["start_offset_ms"] is None, r["start_offset_ms"] or 0, r["request_id"])):
        start = r["start_offset_ms"]
        if not number(start):
            continue
        if number(r["ttft_ms"]):
            ttft.append([start / 1000, r["ttft_ms"] / 1000])
        tpot = r["tpot_ms"]
        if (
            tpot is None
            and r.get("timing_boundaries_compatible") is True
            and number(r["e2e_ms"])
            and number(r["ttft_ms"])
            and number(r["output_tokens"])
            and r["output_tokens"] > 1
        ):
            tpot = (r["e2e_ms"] - r["ttft_ms"]) / (r["output_tokens"] - 1)
        if number(tpot) and tpot > 0:
            interactivity.append([start / 1000, 1000 / tpot])

    def series(points):
        return {
            "count": len(points),
            "excluded": len(selected) - len(points),
            "points": points,
            "rolling_p90": [
                [p[0], quantile([x[1] for x in points[max(0, i - 49) : i + 1]], 0.9)] for i, p in enumerate(points)
            ],
        }

    return {
        "request_count": len(selected),
        "stage_counts": dict(Counter(r["stage"] for r in records)),
        "outcome_counts": dict(Counter(r.get("outcome", "unknown") for r in selected)),
        "input": histogram(r["input_tokens"] for r in selected),
        "output": histogram(r["output_tokens"] for r in selected),
        "ttft": series(ttft),
        "interactivity": series(interactivity),
    }


def workload_details(case):
    selected_rows = defaultdict(list)
    for observation in case.observations:
        selected_rows[observation.source_file_id].append(observation.source_row)
    for rows in selected_rows.values():
        rows.sort()
    files = {f.measurement_file_id: f for f in case.helper_files}
    runs = []
    for run in case.collection_runs:
        contributing = 0
        for binding in run["truth_bindings"]:
            rows = selected_rows[binding["file_id"]]
            ranges = binding["record_ranges"]
            contributing += (
                len(rows)
                if ranges is None
                else sum(bisect.bisect_left(rows, hi) - bisect.bisect_left(rows, lo) for lo, hi in ranges)
            )
        if not contributing:
            continue
        # Original internal source references remain in the pinned HF manifest.
        value = {
            k: run[k]
            for k in (
                "id",
                "collection_type",
                "benchmark_preset",
                "benchmark_id",
                "started_at",
                "collector",
                "replay_mode",
                "dataset",
                "workload",
                "serving",
            )
        }
        value["measurement_count"] = contributing
        value["availability"] = run["request_metrics"]["status"]
        value["reason"] = run["request_metrics"]["reason"]
        value["charts"] = None
        requested = run["request_metrics"]["file_ids"]
        if value["availability"] == "available":
            if not requested or any(fid not in files for fid in requested):
                value.update(
                    availability="unavailable", reason="Request evidence is not part of this selected evaluation."
                )
            else:
                records = [row for fid in requested for row in parquet.rows(files[fid]) if row["run_id"] == run["id"]]
                value["charts"] = charts(records)
        runs.append(value)
    runs.sort(key=lambda r: (r["started_at"] is None, r["started_at"] or "", r["id"]))
    return {
        "runs": runs,
        "unattributed_measurements": len(case.observations) - sum(r["measurement_count"] for r in runs),
    }


def workload_summary(details):
    runs = details["runs"]
    return {
        "types": sorted({r["collection_type"] for r in runs}),
        "datasets": sorted({r["dataset"]["name"] for r in runs if r["dataset"]["name"]}),
        "run_count": len(runs),
        "unattributed_measurements": details["unattributed_measurements"],
    }
