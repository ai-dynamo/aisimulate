# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501 -- Chart captions and score definitions retain readable literal prose.
"""Compare native replay curves with measured silicon, without fitting clocks.

Counters are distributed uniformly within their recorded intervals when rebinned.
Native gauges use a left-held snapshot; silicon gauges are exported interval
averages. Missing coverage never becomes a measured zero.
"""

import bisect
import math
import re
import statistics
from collections import Counter, defaultdict

SCHEMA = "aisimulate.agentx.chart-comparisons.v1"
REQUEST_ID = re.compile(r"(weka:[0-9a-f]{16}):request:outer:(\d+)(?::inner:(\d+))?$")
SMOOTH_SECONDS = 50


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stage(row):
    return "profile" if row.get("phase", "profile") in ("profile", "profiling") else "preparation"


def integer(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def quantile(values, percentile):
    values = sorted(value for value in values if finite(value))
    if not values:
        return None
    index = (len(values) - 1) * percentile / 100
    low = math.floor(index)
    return values[low] + (values[math.ceil(index)] - values[low]) * (index - low)


def grid(duration):
    if not finite(duration) or duration <= 0:
        return []
    result = list(range(1, math.floor(duration) + 1))
    if not result or duration - result[-1] > 1e-8:
        result.append(duration)
    return result


def rebin(intervals, ticks, counter=False):
    """Return full-coverage bucket values; a counter bucket contains a count."""
    totals = [0.0] * len(ticks)
    coverage = [0.0] * len(ticks)
    starts = [0, *ticks[:-1]]
    for row in intervals:
        a, b, value = row.get("start"), row.get("end"), row.get("value")
        if not (finite(a) and finite(b) and b > a and finite(value)) or row.get("coverage", 1) < 1 - 1e-8:
            continue
        index = bisect.bisect_right(ticks, a)
        while index < len(ticks) and starts[index] < b:
            overlap = max(0.0, min(b, ticks[index]) - max(a, starts[index]))
            totals[index] += value * overlap / (b - a) if counter else value * overlap
            coverage[index] += overlap
            index += 1
    result = []
    for index, end in enumerate(ticks):
        width = end - starts[index]
        valid = abs(coverage[index] - width) <= 1e-7
        result.append((totals[index] if counter else totals[index] / width) if valid else None)
    return result


def smooth(values, ticks, counter=False, start=0):
    """Trailing 50 seconds, with no propagation across missing observations."""
    result = []
    starts = [0, *ticks[:-1]]
    for index, end in enumerate(ticks):
        begin = max(start, end - SMOOTH_SECONDS)
        first = bisect.bisect_right(ticks, begin)
        total = 0.0
        valid = True
        for other in range(first, index + 1):
            overlap = ticks[other] - max(begin, starts[other])
            if not finite(values[other]):
                valid = False
                break
            total += values[other] * overlap / (ticks[other] - starts[other]) if counter else values[other] * overlap
        result.append(total / (end - begin) if valid and end > begin else None)
    return result


def smooth_pair(reference, simulated, ref_ticks, sim_ticks, counter=False):
    """Start paired windows at the first fully covered common bucket boundary."""
    ref = dict(zip(ref_ticks, reference, strict=True))
    sim = dict(zip(sim_ticks, simulated, strict=True))
    common = next(
        (tick for tick in ref_ticks if tick == int(tick) and finite(ref[tick]) and finite(sim.get(tick))), None
    )

    def initial(values, ticks):
        if common is not None:
            return max(0, common - 1)
        index = next((i for i, value in enumerate(values) if finite(value)), len(ticks))
        return ticks[index - 1] if index else 0

    return (
        smooth(reference, ref_ticks, counter, initial(reference, ref_ticks)),
        smooth(simulated, sim_ticks, counter, initial(simulated, sim_ticks)),
    )


def combine(*arrays):
    return [sum(items) if all(finite(value) for value in items) else None for items in zip(*arrays, strict=True)]


def ratio(numerator, denominator, scale=100):
    return [
        scale * left / right if finite(left) and finite(right) and right > 0 else None
        for left, right in zip(numerator, denominator, strict=True)
    ]


def cumulative(values, start_index=0):
    total = 0.0
    complete = True
    result = []
    for index, value in enumerate(values):
        if index < start_index:
            result.append(None)
            continue
        complete = complete and finite(value)
        if complete:
            total += value
        result.append(total if complete else None)
    return result


def series(metric, label, origin, ticks, values, color="blue", kind="line", step=False):
    return {
        "id": f"{metric}:{origin}",
        "label": label,
        "origin": origin,
        "metric": metric,
        "color": color,
        "x": ticks,
        "y": values,
        "kind": kind,
        "step": step,
    }


def curve_score(metric, label, reference, simulated, reference_duration, simulated_duration, reason=None, start=0):
    ref = dict(zip(reference["x"], reference["y"], strict=True)) if reference else {}
    sim = dict(zip(simulated["x"], simulated["y"], strict=True)) if simulated else {}
    duration = min(reference_duration or 0, simulated_duration or 0)
    last_tick = math.floor(duration + 1e-8)
    first_tick = math.floor(start + 1e-8) + 1
    denominator = max(0, last_tick - first_tick + 1)
    aligned = []
    zeros = missing = false_positive = 0
    absolute = []
    zero_errors = []
    for tick in range(first_tick, last_tick + 1):
        left, right = ref.get(tick), sim.get(tick)
        if not finite(left) or not finite(right):
            missing += 1
            continue
        absolute.append(abs(right - left))
        if left == 0:
            zeros += 1
            false_positive += right != 0
            zero_errors.append(abs(right))
        elif left > 0:
            aligned.append(
                {"t": tick, "reference": left, "simulated": right, "ape_percent": abs(right - left) / left * 100}
            )
    count = len(aligned)
    result = {
        "label": label,
        "metric": metric,
        "reference_series_id": f"{metric}:silicon",
        "simulated_series_id": f"{metric}:simulated",
        "mape_percent": statistics.mean(row["ape_percent"] for row in aligned) if count and not reason else None,
        "count": count if not reason else 0,
        "coverage": count / denominator if denominator and not reason else 0,
        "denominator_count": denominator,
        "coverage_definition": "Scored positive-silicon bins / all integer-second ticks in the common phase interval",
        "zero_reference_count": zeros,
        "zero_reference_simulated_nonzero_count": false_positive,
        "zero_reference_mae": statistics.mean(zero_errors) if zero_errors else None,
        "zero_reference_simulated_activity_total": sum(zero_errors),
        "missing_count": missing,
        "mae": statistics.mean(absolute) if absolute and not reason else None,
        "scope": "MAPE on the displayed curve, evaluated on a 1-second phase-relative grid; no clock fitting",
        "window_seconds": [start, duration],
        "aligned": aligned if not reason else [],
        "reason": reason or (None if count else "No covered bins with a positive silicon denominator"),
    }
    return result


def phase_bounds(dataset, selected):
    manifest = dataset.get("telemetry") or {}
    bound = next((row for row in manifest.get("phase_boundaries", []) if row.get("phase") == selected), {})
    origin = bound.get("start_ms", manifest.get("profile_start_ms", 0) if selected == "profile" else 0)
    origin = origin if finite(origin) else 0
    end = bound.get("end_ms")
    if not finite(end):
        ends = [row.get("terminal_time_ms") for row in dataset.get("requests", []) if stage(row) == selected]
        ends += [row.get("t", 0) * 1000 for row in dataset.get("samples", []) if row.get("phase") == selected]
        end = max([origin, *[value for value in ends if finite(value)]])
    return origin, max(0.0, (end - origin) / 1000)


def native_intervals(samples, key, origin, duration, gauge=False, max_gap=1.5):
    result = []
    if gauge:
        points = {row["t"] - origin: row.get(key) for row in samples if finite(row.get("t"))}
        times = sorted(points)
        for index, start in enumerate(times):
            end = min(times[index + 1] if index + 1 < len(times) else duration, duration, start + max_gap)
            if end > start:
                result.append({"start": start, "end": end, "value": points[start]})
    else:
        for row in samples:
            if finite(row.get("start")) and finite(row.get("t")):
                result.append({"start": row["start"] - origin, "end": row["t"] - origin, "value": row.get(key)})
    return result


def request_clock(rows, origin_ms, silicon_phase=None):
    result = []
    epoch = integer((silicon_phase or {}).get("start_ns"))
    for row in rows:
        if silicon_phase is not None:
            start, end = integer(row.get("start_ns")), integer(row.get("end_ns"))
            if epoch is None or start is None or end is None:
                continue
            start, end = (start - epoch) / 1e9, (end - epoch) / 1e9
        else:
            start, end = row.get("arrival_time_ms"), row.get("terminal_time_ms")
            if not finite(start) or not finite(end):
                continue
            start, end = (start - origin_ms) / 1000, (end - origin_ms) / 1000
        result.append({**row, "_start": start, "_end": end})
    return result


def successful(row, silicon=False):
    return not row.get("error") if silicon else row.get("terminal_status") == "completed"


def event_counts(rows, ticks, field=None, silicon=False):
    """Terminal-event counts, never approximated as token-generation timestamps."""
    counts = [0.0] * len(ticks)
    for row in rows:
        if not successful(row, silicon) or row["_end"] < 0:
            continue
        value = 1 if field is None else row.get(field)
        if not finite(value):
            continue
        index = bisect.bisect_left(ticks, row["_end"])
        if index < len(ticks):
            counts[index] += value
    return counts


def conversation(row, silicon=False):
    if row.get("_conversation_proxy"):
        return row["_conversation_proxy"]
    return (
        row.get("conversation_id", row.get("source_trace_id", "Unassigned"))
        if silicon
        else (row.get("agentic") or {}).get("conversation_id") or row.get("session_id") or "Unassigned"
    )


def inflight(rows, ticks, field, silicon=False, unique=False):
    values = []
    for tick in ticks:
        active = [row for row in rows if row["_start"] <= tick < row["_end"]]
        if any(not finite(row.get(field)) for row in active):
            values.append(None)
        elif unique:
            groups = defaultdict(list)
            for row in active:
                groups[conversation(row, silicon)].append(row[field])
            values.append(sum(max(items) for items in groups.values()))
        else:
            values.append(statistics.mean(row[field] for row in active) if active else None)
    return values


def request_curve(rows, ticks, key, percentile, silicon=False, inverse=False, running=False):
    valid = sorted(
        (row for row in rows if successful(row, silicon) and finite(row.get(key)) and (not inverse or row[key] > 0)),
        key=lambda row: row["_start"],
    )
    arrivals = [row["_start"] for row in valid]
    measurements = [row[key] for row in valid]
    estimates = [
        quantile(measurements[0 if running else max(0, index - 49) : index + 1], percentile)
        for index in range(len(valid))
    ]
    result = []
    for tick in ticks:
        index = bisect.bisect_right(arrivals, tick) - 1
        value = estimates[index] if index >= 0 else None
        result.append((1000 / value if inverse else value / 1000) if finite(value) else None)
    return result


def reference_key(row):
    if row.get("source_trace_id") is None or integer(row.get("source_outer_idx")) is None:
        return None
    return row["source_trace_id"], integer(row["source_outer_idx"]), integer(row.get("source_inner_idx"))


def match_rows(dataset, native, reference, namespace_map):
    reverse = {value: key for key, value in (namespace_map or {}).items()}
    if len(reverse) != len(namespace_map or {}):
        raise ValueError("Source namespaces must be one-to-one")
    fallback = {
        row["simulated_uuid"]: reference_key(row)
        for row in dataset.get("pairing", [])
        if row.get("simulated_uuid") and row.get("match_status") == "matched_unique_source"
    }
    by_native = defaultdict(list)
    by_reference = defaultdict(list)
    for row in native:
        match = REQUEST_ID.search(row.get("request_id") or (row.get("agentic") or {}).get("request_id", ""))
        key = (
            (reverse[match[1]], int(match[2]), int(match[3]) if match[3] else None)
            if match and match[1] in reverse
            else fallback.get(row.get("uuid"))
        )
        if key:
            by_native[key].append(row)
    for row in reference:
        key = reference_key(row)
        if key:
            by_reference[key].append(row)
    pairs = []
    for row in reference:
        key = reference_key(row)
        candidates = by_native.get(key, [])
        reason = (
            "missing_source_identity"
            if key is None
            else "ambiguous_reference_source"
            if len(by_reference[key]) != 1
            else "ambiguous_simulated_source"
            if len(candidates) > 1
            else "unmatched_reference"
            if not candidates
            else None
        )
        pairs.append((row, candidates[0] if reason is None else None, reason))
    return pairs


def matched_score(pairs, metric):
    ref_key = {"ttft_ms": "ttft_ms", "e2e_latency_ms": "e2e_ms", "interactivity": "tpot_ms"}[metric]
    sim_key = "itl_ms" if metric == "interactivity" else metric
    records = []
    valid = []
    eligible_reference = zeros = missing = 0
    for ref, sim, match_reason in pairs:
        left, right = ref.get(ref_key), (sim or {}).get(sim_key)
        reasons = [match_reason] if match_reason else []
        if not successful(ref, True):
            reasons.append("reference_error")
        if sim and not successful(sim):
            reasons.append("simulated_not_successful")
        if sim and "first_admit_ms" in sim and not finite(sim["first_admit_ms"]):
            reasons.append("simulated_not_admitted")
        if not finite(left):
            reasons.append("reference_metric_unavailable")
            missing += 1
        elif left <= 0:
            reasons.append("reference_metric_nonpositive")
            zeros += left == 0
        elif successful(ref, True):
            eligible_reference += 1
        if not finite(right) or right < 0 or (metric == "interactivity" and right <= 0):
            reasons.append("simulated_metric_unavailable")
        if sim and (not finite(ref.get("output_tokens")) or ref["output_tokens"] != sim.get("output_length")):
            reasons.append("output_length_mismatch_or_unavailable")
        if metric == "interactivity":
            left = 1000 / left if finite(left) and left > 0 else None
            right = 1000 / right if finite(right) and right > 0 else None
        error = abs(right - left) / left * 100 if not reasons else None
        record = {
            "reference_request_id": ref.get("request_id"),
            "simulated_uuid": (sim or {}).get("uuid"),
            "source_trace_id": ref.get("source_trace_id"),
            "source_outer_idx": ref.get("source_outer_idx"),
            "source_inner_idx": ref.get("source_inner_idx"),
            "reference": left,
            "simulated": right,
            "ape_percent": error,
            "exclusions": reasons,
            "input_length_matches": ref.get("input_tokens") == (sim or {}).get("input_length") if sim else None,
        }
        records.append(record)
        if error is not None:
            valid.append(error)
    return {
        "metric": metric,
        "label": {"ttft_ms": "TTFT", "e2e_latency_ms": "E2E latency", "interactivity": "Interactivity"}[metric],
        "mape_percent": statistics.mean(valid) if valid else None,
        "count": len(valid),
        "coverage": len(valid) / eligible_reference if eligible_reference else 0,
        "denominator_count": eligible_reference,
        "coverage_definition": "Successful equal-output-length matched pairs / successful silicon requests with positive metric",
        "zero_reference_count": zeros,
        "missing_count": missing,
        "scope": "Exact source lineage, both successful, equal actual output lengths; input lengths may differ",
        "reason": None if valid else "No eligible successful equal-output-length pairs",
        "pairs": records,
    }


def timeline_data(native, reference, pairs, origin_ms, silicon_phase):
    matches = {ref.get("request_id"): sim for ref, sim, reason in pairs if not reason}
    conversation_matches = defaultdict(set)
    for ref, sim, reason in pairs:
        if not reason:
            conversation_matches[conversation(sim)].add("silicon:" + conversation(ref, True))
    native_groups = {}
    for row in native:
        names = conversation_matches[conversation(row)]
        native_groups[row["uuid"]] = next(iter(names)) if len(names) == 1 else "simulated:" + conversation(row)
    rows = []
    native_matches = {}
    for index, ref in enumerate(reference):
        matched = matches.get(ref.get("request_id"))
        uuid = "silicon:" + str(ref.get("request_id", index))
        if matched:
            native_matches[matched["uuid"]] = uuid
        ttft, e2e = ref.get("ttft_ms"), ref.get("e2e_ms")
        start = ref["_start"] * 1000
        rows.append(
            {
                "uuid": uuid,
                "request_id": ref.get("request_id"),
                "phase": "profile" if stage(ref) == "profile" else "warmup",
                "origin": "silicon",
                "timestamps_aligned": True,
                "origin_ns": (silicon_phase or {}).get("start_ns"),
                "start_ns": str(ref.get("start_ns")),
                "end_ns": str(ref.get("end_ns")),
                "arrival_phase_ms": start,
                "terminal_phase_ms": ref["_end"] * 1000,
                "arrival_time_ms": origin_ms + start,
                "terminal_time_ms": origin_ms + ref["_end"] * 1000,
                "first_token_ms": origin_ms + start + ttft if finite(ttft) else None,
                "last_token_ms": origin_ms + start + e2e if finite(e2e) else None,
                "input_length": ref.get("input_tokens"),
                "output_length": ref.get("output_tokens"),
                "requested_output_length": ref.get("source_output_tokens"),
                "ttft_ms": ttft,
                "e2e_latency_ms": e2e,
                "itl_ms": ref.get("tpot_ms"),
                "terminal_status": "failed" if ref.get("error") else "completed",
                "error": ref.get("error"),
                "worker_id": ref.get("worker_id"),
                "decode_worker_idx": None,
                "prefill_worker_idx": None,
                "matched_simulated_uuid": matched.get("uuid") if matched else None,
                "canonical_conversation": "silicon:" + conversation(ref, True),
                "source": {key: ref.get(key) for key in ("source_trace_id", "source_outer_idx", "source_inner_idx")},
            }
        )
    return {"reference_requests": rows, "native_groups": native_groups, "native_matches": native_matches}


def build_phase(dataset, silicon, namespace_map, selected):
    origin_ms, native_duration = phase_bounds(dataset, selected)
    measured = ((silicon or {}).get("phases") or {}).get(selected)
    has_clock = isinstance(measured, dict) and integer(measured.get("start_ns")) is not None
    measured = measured or {}
    ref_duration = measured.get("duration_s", 0) if has_clock else 0
    nt, rt = grid(native_duration), grid(ref_duration)
    raw_native = [row for row in dataset.get("requests", []) if stage(row) == selected]
    raw_reference = [row for row in dataset.get("reference", []) if stage(row) == selected]
    native = request_clock(raw_native, origin_ms)
    reference = request_clock(raw_reference, origin_ms, measured) if has_clock else []
    pairs = match_rows(dataset, raw_native, raw_reference, namespace_map)
    timeline = timeline_data(raw_native, reference, pairs, origin_ms, measured)
    native_proxy = [{**row, "_conversation_proxy": timeline["native_groups"].get(row.get("uuid"))} for row in native]
    samples = [row for row in dataset.get("samples", []) if row.get("phase") == selected]
    pass_supported = any(
        finite(row.get("scheduler_queries")) and row["scheduler_queries"] > 0 for row in dataset.get("samples", [])
    )
    if pass_supported:
        # Native serialization uses both-null for zero considered tokens, too.
        # Establish support elsewhere first; never manufacture an absent interval.
        samples = [
            {**row, "scheduler_hits": 0, "scheduler_queries": 0}
            if row.get("scheduler_hits") is None and row.get("scheduler_queries") is None
            else row
            for row in samples
        ]
    max_gap = max(1e-6, (dataset.get("telemetry", {}).get("sample_interval_ms", 1000) or 1000) / 1000 * 1.5)
    native_bins, ref_bins = {}, {}
    gauges = {"running", "waiting", "cache_pct"}
    for key in (
        "running",
        "waiting",
        "cache_pct",
        "prefill_tokens",
        "output_tokens",
        "scheduler_hits",
        "scheduler_queries",
        "lookup_hits",
        "lookup_queries",
        "hbm",
        "host",
        "miss",
        "admission_input",
        "accept",
        "accept_count",
    ):
        intervals = native_intervals(samples, key, origin_ms / 1000, native_duration, key in gauges, max_gap)
        native_bins[key] = rebin(intervals, nt, key not in gauges)
    acceptance_samples = [
        {
            **row,
            "accept_tokens": row["accept"] * row["accept_count"]
            if finite(row.get("accept")) and finite(row.get("accept_count"))
            else 0
            if row.get("accept_count") == 0
            else None,
        }
        for row in samples
    ]
    native_bins["accept_tokens"] = rebin(
        native_intervals(acceptance_samples, "accept_tokens", origin_ms / 1000, native_duration), nt, True
    )
    raw_series = measured.get("series") or {}
    for key in (
        "cache_used",
        "cache_evictable",
        "cache_capacity",
        "running",
        "waiting",
        "acceptance",
        "prefill_tokens",
        "output_processed_tokens",
        "scheduler_hits",
        "scheduler_queries",
        "hbm",
        "host",
        "miss",
        "admission_input",
    ):
        ref_bins[key] = rebin(
            raw_series.get(key, []),
            rt,
            key not in {"cache_used", "cache_evictable", "cache_capacity", "running", "waiting", "acceptance"},
        )
    charts = {}
    clock_reason = (
        None if has_clock else "Actual silicon phase origin is unavailable; no elapsed-time alignment inferred"
    )

    score_end = min(ref_duration, native_duration, 3600 if selected == "profile" else math.inf)

    def chart(key, title, unit, metrics, note="", foot="", reason=None, kind="line", step=False, score_start=0):
        lines, scores = [], []
        for metric, label, ref_values, sim_values, color in metrics:
            truth = (
                series(
                    metric,
                    label,
                    "silicon",
                    rt if kind == "line" else ref_values[0],
                    ref_values if kind == "line" else ref_values[1],
                    color,
                    kind,
                    step,
                )
                if (has_clock and any(finite(value) for value in ref_values)) or kind == "bar"
                else None
            )
            simulation = series(
                metric,
                label,
                "simulated",
                nt if kind == "line" else sim_values[0],
                sim_values if kind == "line" else sim_values[1],
                color,
                kind,
                step,
            )
            if truth:
                lines.append(truth)
            lines.append(simulation)
            scores.append(
                curve_score(metric, label, truth, simulation, score_end, score_end, reason or clock_reason, score_start)
            )
        charts[key] = {
            "title": title,
            "unit": unit,
            "note": note,
            "foot": foot,
            "series": lines,
            "scores": scores,
            "score_window_start_s": score_start,
            "score_window_end_s": score_end if has_clock and not reason else None,
        }

    time_note = "1-second phase grid; trailing 50-second smoothing from common covered startup; no clock fitting"
    gap_foot = "Missing coverage stays unavailable. MAPE excludes zero-silicon bins; their counts and MAE are reported separately."
    for name, native_key, ref_key in (
        ("input", "input_length", "input_tokens"),
        ("output", "output_length", "output_tokens"),
    ):
        chart(
            name + "_inflight",
            f"In-flight {name} sequence length",
            "tokens",
            [
                (
                    name,
                    "In-flight mean",
                    inflight(reference, rt, ref_key, True),
                    inflight(native, nt, native_key),
                    "blue",
                )
            ],
            "Mean final request length among active requests at each second",
            "Output length is the final delivered length, not tokens generated so far; no active requests is unavailable.",
        )
    for percentile in (75, 90):
        for name, native_key, ref_key, title in (
            ("interactivity", "itl_ms", "tpot_ms", "Interactivity"),
            ("latency_ttft_ms", "ttft_ms", "ttft_ms", "TTFT"),
            ("latency_e2e_latency_ms", "e2e_latency_ms", "e2e_ms", "End-to-end latency"),
        ):
            inverse = name == "interactivity"
            metrics = []
            for running, color in ((False, "cyan" if inverse else "accent"), (True, "red")):
                label = (
                    ("1 / " if inverse else "")
                    + ("Cumulative" if running else "Rolling 50-request")
                    + f" P{percentile}"
                    + (" TPOT" if inverse else "")
                )
                metrics.append(
                    (
                        name + ("_cumulative" if running else "_rolling"),
                        label,
                        request_curve(reference, rt, ref_key, percentile, True, inverse, running),
                        request_curve(native, nt, native_key, percentile, False, inverse, running),
                        color,
                    )
                )
            chart(
                f"{name}_p{percentile}",
                f"{title} over time",
                "tokens/s/user" if inverse else "s",
                metrics,
                "Successful request observations at arrival; last observed percentile held on the 1-second grid",
                "Interactivity takes the TPOT percentile before inversion (1000 / Pxx TPOT). Curve MAPE includes workload timing; matched-request MAPE compares raw equal-output-length source pairs."
                if inverse
                else "Curve MAPE includes workload timing differences; separate matched-request MAPE compares raw equal-output-length source pairs.",
                step=True,
            )
    cache = ratio(combine(ref_bins["cache_used"], ref_bins["cache_evictable"]), ref_bins["cache_capacity"])
    chart(
        "cache",
        "Resident KV cache utilization",
        "%",
        [("resident_cache", "Resident cache", *smooth_pair(cache, native_bins["cache_pct"], rt, nt), "blue")],
        time_note,
        "Resident = used + evictable; capacity is counted once per worker pool, not once per TP replica. " + gap_foot,
    )
    queue = [
        (key, label, *smooth_pair(ref_bins[key], native_bins[key], rt, nt), color)
        for key, label, color in (("running", "Running", "green"), ("waiting", "Waiting", "red"))
    ]
    queue.append(
        (
            "total",
            "Running + waiting",
            *smooth_pair(
                combine(ref_bins["running"], ref_bins["waiting"]),
                combine(native_bins["running"], native_bins["waiting"]),
                rt,
                nt,
            ),
            "blue",
        )
    )
    chart(
        "queue",
        "Request queue depth",
        "requests",
        queue,
        time_note,
        "Scheduler queues; router pending state is not added. " + gap_foot,
    )
    completed_ref = event_counts(reference, rt, silicon=True)
    completed_sim = event_counts(native, nt)
    chart(
        "completed",
        "Successful request completions",
        "requests",
        [("completed", "Completed cumulative", cumulative(completed_ref), cumulative(completed_sim), "green")],
        "Successful terminal records, including zero-output native completions",
        gap_foot,
        step=True,
    )
    hit_pair = smooth_pair(ref_bins["scheduler_hits"], native_bins["scheduler_hits"], rt, nt, True)
    query_pair = smooth_pair(ref_bins["scheduler_queries"], native_bins["scheduler_queries"], rt, nt, True)
    ref_hits, sim_hits = ratio(hit_pair[0], query_pair[0]), ratio(hit_pair[1], query_pair[1])
    lookup_only = not any(finite(value) for value in native_bins["scheduler_queries"]) and any(
        finite(row.get("lookup_queries")) for row in samples
    )
    chart(
        "hits",
        "Scheduler prefix reuse",
        "%",
        [("scheduler_reuse", "Pass token reuse", ref_hits, sim_hits, "purple")],
        "Ratio of summed hit / considered token counts in the trailing 50 seconds",
        "Pass-local reuse differs from lookup retries and first-admission attribution. After support is established, native captured intervals with both counters null contribute zero queries; missing intervals stay unavailable. "
        + gap_foot,
        reason="Native lookup counters have no equivalent silicon observation" if lookup_only else None,
    )
    if any(finite(value) for value in native_bins["lookup_queries"]):
        lookup = series(
            "native_lookup",
            "Native lookup hit rate (retries included)",
            "simulated",
            nt,
            ratio(smooth(native_bins["lookup_hits"], nt, True), smooth(native_bins["lookup_queries"], nt, True)),
            "cyan",
        )
        charts["hits"]["series"].append(lookup)
        charts["hits"]["scores"].append(
            curve_score(
                "native_lookup",
                "Native lookup",
                None,
                lookup,
                score_end,
                score_end,
                "Native lookup counters have no equivalent silicon observation",
            )
        )
    ref_output = event_counts(reference, rt, "output_tokens", True)
    sim_output = event_counts(native, nt, "output_length")
    prefill = (
        "prefill",
        "Executed prefill",
        *smooth_pair(ref_bins["prefill_tokens"], native_bins["prefill_tokens"], rt, nt, True),
        "blue",
    )
    output = (
        "output_completed",
        "Successful completion-accounted output",
        *smooth_pair(ref_output, sim_output, rt, nt, True),
        "accent",
    )
    for suffix, metrics in (("both", [prefill, output]), ("prefill", [prefill]), ("decode", [output])):
        chart(
            "throughput_" + suffix,
            "Token throughput",
            "tokens/s",
            metrics,
            time_note,
            "Output is counted at successful request completion on both sides; it does not reconstruct streaming token cadence. Prefill counts executed work, including recomputation. "
            + gap_foot,
        )
    chart(
        "throughput_delivered",
        "Live delivered output throughput",
        "tokens/s",
        [
            (
                "output_delivered",
                "Client-delivered output",
                [None] * len(rt),
                smooth(native_bins["output_tokens"], nt, True),
                "accent",
            )
        ],
        time_note,
        "Silicon has server-processed decode counts but no exact client-delivered token timestamps.",
        reason="Client-delivered and server-processed output have different semantics",
    )
    source_metrics = []
    source_keys = ("miss", "hbm", "host", "admission_input")
    source_start = next(
        (
            index
            for index in range(min(len(nt), len(rt)))
            if all(finite(ref_bins[key][index]) and finite(native_bins[key][index]) for key in source_keys)
        ),
        None,
    )
    ref_source_start = source_start if source_start is not None else len(rt)
    sim_source_start = source_start if source_start is not None else (0 if not has_clock else len(nt))
    for key, label, color in (
        ("miss", "Compute (miss)", "accent"),
        ("hbm", "HBM reuse", "blue"),
        ("host", "Host reuse", "purple"),
    ):
        source_metrics.append(
            (
                key,
                label,
                ratio(
                    cumulative(ref_bins[key], ref_source_start),
                    cumulative(ref_bins["admission_input"], ref_source_start),
                ),
                ratio(
                    cumulative(native_bins[key], sim_source_start),
                    cumulative(native_bins["admission_input"], sim_source_start),
                ),
                color,
            )
        )
    source_begin = 0 if source_start == 0 or source_start is None else nt[source_start - 1]
    chart(
        "source",
        "Cumulative first-admission prompt source",
        "%",
        source_metrics,
        f"Observed cumulative counts from {source_begin:g}s, the first common fully covered bucket boundary",
        "Silicon effective-prefill excludes retractions; native counts first admissions. Initial uncovered fractions are omitted from BOTH curves; internal gaps remain unavailable. Ratios use summed token counts. "
        + gap_foot,
        score_start=source_begin,
    )
    chart(
        "unique",
        "Conversation-level input footprint proxy",
        "tokens",
        [
            (
                "active_input_proxy",
                "Sum of maximum active ISL per conversation",
                inflight(reference, rt, "input_tokens", True, True),
                inflight(native_proxy, nt, "input_length", False, True),
                "blue",
            )
        ],
        "Same request-lifetime proxy on each side; all outcomes retained",
        "Exact-source matches align conversation groups; unmatched groups retain native identity. This is not exact unique-prefix or physical KV occupancy. Terminal-only captures can omit unfinished work. "
        + gap_foot,
        step=True,
    )
    native_acceptance = ratio(
        smooth(native_bins["accept_tokens"], nt, True), smooth(native_bins["accept_count"], nt, True), 1
    )
    chart(
        "acceptance",
        "Speculative acceptance length",
        "tokens/forward",
        [
            (
                "acceptance",
                "Accepted drafts + base token",
                smooth_pair(ref_bins["acceptance"], native_acceptance, rt, nt)[0],
                native_acceptance,
                "purple",
            )
        ],
        "Silicon time-averages a retained gauge; native weights by decode request-forwards over 50 seconds",
        "Both include the base token before output truncation. Their temporal weights differ; exact curve MAPE is unavailable.",
        reason="Retained silicon gauge versus native forward-weighted interval mean",
    )
    statuses = sorted(
        {row.get("terminal_status", "unknown") for row in raw_native}
        | {"failed" if row.get("error") else "completed" for row in raw_reference}
    )
    counts_ref = Counter("failed" if row.get("error") else "completed" for row in raw_reference)
    counts_sim = Counter(row.get("terminal_status", "unknown") for row in raw_native)
    chart(
        "completion",
        "Terminal request outcomes",
        "requests",
        [
            (
                "terminal_status",
                "Terminal outcomes",
                (statuses, [counts_ref[key] for key in statuses]),
                (statuses, [counts_sim[key] for key in statuses]),
                "green",
            )
        ],
        "All terminal outcomes, including failed and zero-output requests",
        "Categorical counts have no time-curve MAPE; no failures are removed from this chart.",
        reason="Categorical outcome counts are not a time curve",
        kind="bar",
    )
    return {
        "charts": charts,
        "matched": {metric: matched_score(pairs, metric) for metric in ("ttft_ms", "e2e_latency_ms", "interactivity")},
        "timeline": timeline,
        "alignment": {
            "clock": "seconds_since_own_phase_start",
            "silicon_origin_ns": measured.get("start_ns"),
            "native_origin_ms": origin_ms,
            "silicon_duration_s": ref_duration,
            "native_duration_s": native_duration,
            "grid_seconds": 1,
            "smoothing_seconds": SMOOTH_SECONDS,
            "clock_reason": clock_reason,
        },
        "counts": {
            "silicon": len(raw_reference),
            "simulated": len(raw_native),
            "silicon_failed": sum(bool(row.get("error")) for row in raw_reference),
            "simulated_not_completed": sum(row.get("terminal_status") != "completed" for row in raw_native),
        },
    }


def build_comparisons(dataset, silicon=None, namespace_map=None):
    """Produce display curves, reproducible aligned scores, and timeline joins."""
    if silicon and dataset.get("point_id") is not None and silicon.get("point_id") != dataset["point_id"]:
        raise ValueError("Silicon and simulated benchmark point IDs differ")
    return {
        "schema": SCHEMA,
        "phases": {
            selected: build_phase(dataset, silicon, namespace_map, selected) for selected in ("profile", "preparation")
        },
        "methodology": [
            "Solid silicon and dotted simulated curves use their own actual phase start; no lag fitting or time warping.",
            "Curve MAPE = mean(100 * abs(simulated - silicon) / silicon) on displayed values at covered integer-second ticks in the common phase span.",
            "Zero silicon denominators are excluded, counted, and accompanied by MAE and nonzero-simulation counts; no epsilon denominator is invented.",
            "Counter rebinning assumes a uniform rate within each recorded interval and conserves its count. Gauges use interval averages; native snapshots are left-held.",
            "Request MAPE uses unique source lineage and equal actual output lengths; preparation folds primer and warmup only for matching.",
        ],
    }
