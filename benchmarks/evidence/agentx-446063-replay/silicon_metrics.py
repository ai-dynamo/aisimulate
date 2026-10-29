# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select comparable silicon intervals from the pinned AIPerf metrics export."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

SCHEMA = "aisimulate.agentx.silicon.v1"

METRIC_DEFINITIONS = {
    "cache_used": ("gauge", "tokens", "Actively used KV slots; excludes evictable radix-cache slots."),
    "cache_evictable": ("gauge", "tokens", "Evictable resident KV slots; add to used slots for physical residency."),
    "cache_capacity": ("gauge", "tokens", "Logical KV pool capacity, counted once per TP ownership domain."),
    "running": ("gauge", "requests", "Scrape-averaged server running-request gauge."),
    "waiting": ("gauge", "requests", "Scrape-averaged server waiting-request gauge."),
    "acceptance": (
        "gauge",
        "tokens/forward",
        "Retained speculative acceptance gauge: accepted drafts plus bonus token; "
        "not an interval forward-weighted mean. Zero can be an initialization value.",
    ),
    "prefill_tokens": ("counter", "tokens", "Executed prefill tokens, updated at server log intervals."),
    "output_processed_tokens": (
        "counter",
        "tokens",
        "Server processed-decode tokens, updated at log intervals; not client-delivered output tokens.",
    ),
    "scheduler_hits": ("counter", "tokens", "Pass-local prefix-cache tokens from realtime prefill_cache."),
    "scheduler_queries": ("counter", "tokens", "Pass-local prefill_cache plus prefill_compute tokens."),
    "hbm": ("counter", "tokens", "Effective device-cache hits, excluding retraction recounts."),
    "host": ("counter", "tokens", "Effective host plus storage hits; unavailable unless both modes are observed."),
    "miss": ("counter", "tokens", "Effective computed prompt tokens, excluding retraction recounts."),
    "admission_input": ("counter", "tokens", "Sum of all four effective-prefill modes, excluding retraction recounts."),
}

GAUGES = {
    "cache_used": "sglang:kv_used_tokens",
    "cache_evictable": "sglang:kv_evictable_tokens",
    "cache_capacity": "sglang:max_total_num_tokens",
    "running": "sglang:num_running_reqs",
    "waiting": "sglang:num_queue_reqs",
    "acceptance": "sglang:spec_accept_length",
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nanoseconds(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError("Nanosecond timestamps must be integers or decimal strings")


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _intervals(series, kind, diagnostics, context):
    """AIPerf counter totals are already interval deltas, not cumulative values."""
    result = {}
    for row in series.get("timeslices", []):
        try:
            start, end = _nanoseconds(row.get("start_ns")), _nanoseconds(row.get("end_ns"))
        except (ValueError, AttributeError):
            diagnostics[(context, "invalid_timestamp")] += 1
            continue
        value = row.get("avg" if kind == "gauge" else "total")
        if end <= start or not _finite(value) or value < 0:
            diagnostics[(context, "invalid_interval_or_value")] += 1
            continue
        if kind == "counter" and any(row.get(key) for key in ("reset", "counter_reset", "reset_count", "resets")):
            diagnostics[(context, "counter_reset")] += 1
            continue
        key = start, end
        if key in result and result[key] != value:
            raise ValueError(f"Conflicting duplicate interval in {context}")
        result[key] = value
    keys = sorted(result)
    if any(keys[i][0] < keys[i - 1][1] for i in range(1, len(keys))):
        raise ValueError(f"Overlapping intervals in {context}")
    return result


def _sum_intervals(parts, kind="counter"):
    """Sum only common coverage, splitting counters uniformly at source edges."""
    if not parts or any(not part for part in parts):
        return {}
    if len(parts) == 1:
        return parts[0]
    boundaries = sorted({edge for part in parts for interval in part for edge in interval})
    ordered = [sorted(part.items()) for part in parts]
    indices, result = [0] * len(parts), {}
    for left, right in pairwise(boundaries):
        total = 0.0
        for index, items in enumerate(ordered):
            while indices[index] < len(items) and items[indices[index]][0][1] <= left:
                indices[index] += 1
            if indices[index] == len(items):
                break
            (start, end), value = items[indices[index]]
            if start > left or end < right:
                break
            total += value * (right - left) / (end - start) if kind == "counter" else value
        else:
            result[(left, right)] = total
    return result


def _rank(series):
    value = str(series.get("labels", {}).get("tp_rank", "0"))
    if not value.isdecimal():
        raise ValueError(f"Invalid tp_rank: {value}")
    return int(value)


def _select_metric(metrics, name, kind, phase, diagnostics, ownership, mode=None):
    metric = metrics.get(name)
    if metric is None:
        return {}
    if metric.get("type") != kind:
        raise ValueError(f"{name} must be a {kind}")
    domains = defaultdict(list)
    for series in metric.get("series", []):
        labels = series.get("labels", {})
        if mode is not None and labels.get("mode") != mode:
            continue
        pool_labels = {key: value for key, value in labels.items() if key not in ("tp_rank", "mode")}
        domain = json.dumps({"endpoint": series.get("endpoint_url"), "labels": pool_labels}, sort_keys=True)
        context = f"{phase}:{name}" + (f":{mode}" if mode else "")
        domains[domain].append((series, _intervals(series, kind, diagnostics, context)))
    pools = []
    for domain, replicas in sorted(domains.items()):
        replicas.sort(key=lambda replica: _rank(replica[0]))
        if len({_rank(series) for series, _ in replicas}) != len(replicas):
            raise ValueError(f"Duplicate TP ownership series for {name}")
        # Effective-prefill counters expose zero placeholders on non-reporting TP ranks.
        active = [(series, rows) for series, rows in replicas if any(value > 0 for value in rows.values())]
        selected, rows = (active if kind == "counter" and active else replicas)[0]
        for other, other_rows in replicas:
            if other is selected or (kind == "counter" and not any(value > 0 for value in other_rows.values())):
                continue
            for key in rows.keys() & other_rows.keys():
                if not math.isclose(rows[key], other_rows[key], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Conflicting TP replicas for {name}; cannot deduplicate this pool")
        ownership.append(
            {
                "phase": phase,
                "metric": name,
                "mode": mode,
                "domain": json.loads(domain),
                "selected_tp_rank": _rank(selected),
                "replica_tp_ranks": [_rank(series) for series, _ in replicas],
                "policy": "one reporting TP rank per logical pool; do not sum TP replicas",
            }
        )
        pools.append(rows)
    return _sum_intervals(pools, kind)


def _validate_compact(data, point_id):
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported silicon schema: {data.get('schema')!r}")
    if data.get("point_id") != point_id:
        raise ValueError(f"Silicon point_id {data.get('point_id')!r} does not match {point_id}")
    phases = data.get("phases")
    if not isinstance(phases, dict) or not phases or not set(phases).issubset({"profile", "preparation"}):
        raise ValueError("Silicon phases must contain profile and/or preparation")
    for phase, values in phases.items():
        start, end = _nanoseconds(values.get("start_ns")), _nanoseconds(values.get("end_ns"))
        duration = values.get("duration_s")
        if end <= start or not _finite(duration) or not math.isclose(duration, (end - start) / 1e9, abs_tol=1e-9):
            raise ValueError(f"Invalid phase boundaries for {phase}")
        series = values.get("series")
        if not isinstance(series, dict) or set(series) != set(METRIC_DEFINITIONS):
            raise ValueError(f"Missing or unknown silicon metric series for {phase}")
        for metric, rows in series.items():
            previous_end = -math.inf
            if not isinstance(rows, list):
                raise ValueError(f"Intervals must be a list for {phase}:{metric}")
            for row in rows:
                if not isinstance(row, dict) or any(
                    not _finite(row.get(key)) for key in ("start", "end", "value", "coverage")
                ):
                    raise ValueError(f"Invalid compact interval for {phase}:{metric}")
                if row["start"] < 0 or row["end"] <= row["start"] or row["value"] < 0 or not 0 < row["coverage"] <= 1:
                    raise ValueError(f"Invalid compact interval bounds for {phase}:{metric}")
                if row["start"] < previous_end:
                    raise ValueError(f"Overlapping compact intervals for {phase}:{metric}")
                previous_end = row["end"]
    return data


def load_silicon_metrics(path: Path | None, point_id: int) -> dict | None:
    """Load raw AIPerf 1.1 metrics or the portable selected-interval schema.

    Valid rows have coverage=1; absent/invalid intervals have no row. Counter
    values are counts and gauge values are scrape averages. Raw final buckets
    can extend past the phase boundary; the comparison layer clips/rebins them.
    """
    if path is None:
        return None
    if not isinstance(point_id, int) or isinstance(point_id, bool) or point_id <= 0:
        raise ValueError("point_id must be a positive integer")
    path = Path(path)
    with path.open() as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Silicon metrics must be a JSON object")
    if "schema" in data:
        return _validate_compact(data, point_id)
    if str(data.get("schema_version")) != "1.1" or not isinstance(data.get("metrics"), dict):
        raise ValueError("Expected AIPerf server_metrics_export.json schema_version 1.1")
    if data.get("metrics_phase") not in ("profiling", "profile"):
        raise ValueError("Raw metrics must describe the profiling phase")
    ranges = data.get("summary", {}).get("phase_time_ranges", {})
    if not isinstance(ranges, dict) or "profiling" not in ranges:
        raise ValueError("Raw metrics require exact profiling phase_time_ranges")
    diagnostics, ownership, phases = Counter(), [], {}
    for raw_phase, phase, metric_key in (
        ("profiling", "profile", "metrics"),
        ("warmup", "preparation", "warmup_metrics"),
    ):
        if raw_phase not in ranges:
            continue
        start = _nanoseconds(ranges[raw_phase].get("start_ns"))
        end = _nanoseconds(ranges[raw_phase].get("end_ns"))
        if end <= start:
            raise ValueError(f"Invalid raw {raw_phase} phase boundaries")
        metrics = data.get(metric_key, {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid raw metrics for {raw_phase}")

        def select(name, kind, mode=None):
            return _select_metric(metrics, name, kind, phase, diagnostics, ownership, mode)

        result = {key: select(name, "gauge") for key, name in GAUGES.items()}
        realtime = {
            mode: select("sglang:realtime_tokens", "counter", mode)
            for mode in ("prefill_compute", "prefill_cache", "decode")
        }
        effective = {
            mode: select("sglang:prefill_effective_tokens", "counter", mode)
            for mode in ("device_hit", "host_hit", "storage_hit", "input")
        }
        result.update(
            prefill_tokens=realtime["prefill_compute"],
            output_processed_tokens=realtime["decode"],
            scheduler_hits=realtime["prefill_cache"],
            scheduler_queries=_sum_intervals([realtime["prefill_cache"], realtime["prefill_compute"]]),
            hbm=effective["device_hit"],
            host=_sum_intervals([effective["host_hit"], effective["storage_hit"]]),
            miss=effective["input"],
            admission_input=_sum_intervals(list(effective.values())),
        )
        phases[phase] = {
            "start_ns": str(start),
            "end_ns": str(end),
            "duration_s": (end - start) / 1e9,
            "series": {
                name: [
                    {"start": (left - start) / 1e9, "end": (right - start) / 1e9, "value": value, "coverage": 1.0}
                    for (left, right), value in sorted(rows.items())
                    if left >= start
                ]
                for name, rows in result.items()
            },
        }
    compact = {
        "schema": SCHEMA,
        "point_id": point_id,
        "phases": phases,
        "metric_definitions": {
            key: {"kind": kind, "unit": unit, "description": description}
            for key, (kind, unit, description) in METRIC_DEFINITIONS.items()
        },
        "ownership": ownership,
        "provenance": {
            "source_path": str(path.resolve()),
            "source_sha256": _hash_file(path),
            "source_bytes": path.stat().st_size,
            "format": "AIPerf server_metrics_export.json",
            "schema_version": str(data["schema_version"]),
            "aiperf_version": data.get("aiperf_version"),
            "benchmark_id": data.get("benchmark_id"),
            "phase_origin": "summary.phase_time_ranges; integer epoch nanoseconds subtracted before conversion",
        },
        "semantics": {
            "intervals": "start/end are seconds relative to the actual phase start; "
            "counters are interval totals, gauges are scrape averages",
            "coverage": "Fraction of the interval represented by a valid exported row; "
            "raw rows use 1, gaps and invalid rows are omitted",
            "missing": "Empty series means unavailable; an observed zero remains a row with value 0",
            "aggregation": "Combine only common observed coverage across required components and ownership domains; "
            "split misaligned counters proportionally by duration at source interval boundaries",
            "counter_resets": "AIPerf exports interval deltas, not cumulative readings; "
            "explicit reset flags and negative deltas are omitted, hidden upstream resets cannot be reconstructed",
            "phase_end": "Final source intervals may extend beyond the phase boundary; "
            "consumers must clip/rebin without changing the original count",
        },
        "diagnostics": [
            {"context": context, "reason": reason, "count": count}
            for (context, reason), count in sorted(diagnostics.items())
        ],
    }
    return _validate_compact(compact, point_id)
