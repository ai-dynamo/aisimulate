# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render native replay telemetry as a self-contained, offline HTML dashboard."""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import plotly
from plotly.offline import get_plotlyjs

_STREAMS = ("requests", "samples", "lifecycle")
_UNITS = {
    "timestamps": "ms",
    "durations": "ms",
    "traffic_duration": "s",
    "tokens": "tokens",
    "cache_blocks": "blocks",
    "cache_bytes": "bytes",
    "cache_usage": "ratio",
}
_REQUEST_NUMBERS = {
    "input_length",
    "output_length",
    "requested_output_length",
    "turn_index",
    "prefill_worker_idx",
    "decode_worker_idx",
    "admission_count",
    "readmission_count",
    "reused_input_tokens",
    "first_admission_reused_input_tokens",
    "first_admission_g1_reused_input_tokens",
    "first_admission_host_reused_input_tokens",
    "decode_reused_input_tokens",
    "prefill_route_overlap_tokens",
    "decode_route_overlap_tokens",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON mapping")
    return value


def _rows(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{name} must be a list of JSON mappings")
    return value


def _number(value: Any, name: str, *, required: bool = False) -> int | float | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number{' or null' if not required else ''}")
    return value


def _metric(row: Mapping[str, Any], key: str) -> int | float | None:
    return _number(row.get(key), key)


def _observed_sum(values: list[int | float | None]) -> int | float | None:
    observed = [value for value in values if value is not None]
    return sum(observed) if observed else None


def _gauge_sum(values: list[int | float | None]) -> int | float | None:
    return sum(values) if values and all(value is not None for value in values) else None


def _cache_percent(workers: list[dict[str, Any]]) -> float | None:
    if not workers:
        return None
    capacities = [worker.get("kv_cache_capacity_bytes") for worker in workers]
    if all(value is not None and value > 0 for value in capacities):
        used = [worker.get("kv_cache_used_bytes") for worker in workers]
        return 100 * sum(used) / sum(capacities) if all(value is not None for value in used) else None
    capacities = [worker.get("total_blocks") for worker in workers]
    usage = [worker.get("physical_cache_usage") for worker in workers]
    if all(value is not None and value > 0 for value in capacities) and all(value is not None for value in usage):
        return 100 * sum(size * ratio for size, ratio in zip(capacities, usage, strict=True)) / sum(capacities)
    # Grouped byte capacities and linear block capacities cannot be added.
    return None


def _sample(row: Mapping[str, Any], *, aggregated: bool) -> dict[str, Any]:
    start = _number(row.get("interval_start_ms"), "sample.interval_start_ms", required=True)
    end = _number(row.get("sampled_at_ms"), "sample.sampled_at_ms", required=True)
    if end < start:
        raise ValueError("sample.sampled_at_ms precedes interval_start_ms")
    if row.get("phase") not in {"preparation", "profile"}:
        raise ValueError("sample.phase must be preparation or profile")
    if row.get("kind") not in {"baseline", "periodic", "final"}:
        raise ValueError("sample.kind must be baseline, periodic or final")
    traffic = _mapping(row.get("traffic", {}), "sample.traffic")
    admission = _mapping(traffic.get("first_admission", {}), "sample.traffic.first_admission")
    duration = _metric(traffic, "duration_s")
    if duration is not None and not math.isclose(duration * 1000, end - start, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("sample.traffic.duration_s does not match its native interval")
    workers = []
    intervals = []
    for role in ("prefill", "decode"):
        intervals.append(_mapping(row.get(f"{role}_interval_metrics", {}), f"sample.{role}_interval_metrics"))
        for worker in _rows(row.get(f"{role}_scheduler_metrics", []), f"sample.{role}_scheduler_metrics"):
            normalized = {**worker, "metric_role": role, "role": "agg" if aggregated and role == "decode" else role}
            for key in (
                "worker_id",
                "dp_rank",
                "active_blocks",
                "inactive_blocks",
                "total_blocks",
                "kv_cache_used_bytes",
                "kv_cache_capacity_bytes",
                "active_cache_usage",
                "physical_cache_usage",
                "running_requests",
                "waiting_requests",
            ):
                normalized[key] = _metric(worker, key)
            workers.append(normalized)
    lookups = [
        _mapping(interval["native_prefix_cache_lookup"], "native_prefix_cache_lookup")
        if interval.get("native_prefix_cache_lookup") is not None
        else {}
        for interval in intervals
    ]
    host_pools = []
    for pool in _rows(row.get("host_caches", []), "sample.host_caches"):
        normalized = dict(pool)
        for key in (
            "capacity_blocks",
            "resident_blocks",
            "used_blocks",
            "capacity_bytes",
            "resident_bytes",
            "used_bytes",
        ):
            normalized[key] = _metric(pool, key)
        host_pools.append(normalized)
    sample = {
        "t": end / 1000,
        "start": start / 1000,
        "phase": row["phase"],
        "kind": row["kind"],
        "ordinal": _metric(row, "sample_ordinal"),
        "duration": duration,
        "running": _gauge_sum([worker["running_requests"] for worker in workers]),
        "waiting": _gauge_sum([worker["waiting_requests"] for worker in workers]),
        "router_pending": _gauge_sum(
            [_metric(row, f"router_pending_{role}_requests") for role in ("prefill", "decode")]
        ),
        "cache_pct": _cache_percent(workers),
        "workers": workers,
        "host_pools": host_pools,
    }
    for name, key in (
        ("output_tokens", "delivered_output_tokens"),
        ("prefill_tokens", "committed_prefill_tokens"),
        ("arrivals", "arriving_requests"),
        ("completed", "completed_requests"),
        ("accept", "avg_accept_length"),
        ("accept_count", "accept_length_forward_count"),
    ):
        sample[name] = _metric(traffic, key)
    for name, key in (
        ("admission_input", "input_tokens"),
        ("admission_requests", "requests"),
        ("hbm", "hbm_reused_tokens"),
        ("host", "host_reused_tokens"),
        ("miss", "uncached_tokens"),
    ):
        sample[name] = _metric(admission, key)
    for name, key, sources in (
        ("lookup_hits", "matched_tokens", lookups),
        ("lookup_queries", "queried_tokens", lookups),
        ("lookup_count", "query_count", lookups),
        ("scheduler_hits", "cache_hit_tokens", intervals),
        ("scheduler_queries", "cache_total_tokens", intervals),
        ("preemptions", "preemptions", intervals),
    ):
        sample[name] = _observed_sum([_metric(source, key) for source in sources])
    return sample


def _request(row: Mapping[str, Any]) -> dict[str, Any]:
    # Keep scalar evidence and correlation history, omitting large token arrays.
    compact = {key: value for key, value in row.items() if value is None or isinstance(value, (str, int, float, bool))}
    for key in _REQUEST_NUMBERS | {key for key in row if key.endswith("_ms")}:
        if key in row:
            compact[key] = _metric(row, key)
    for key in ("arrival_time_ms", "terminal_time_ms"):
        compact[key] = _number(row.get(key), f"request.{key}", required=True)
    if compact["terminal_time_ms"] < compact["arrival_time_ms"]:
        raise ValueError("request.terminal_time_ms precedes arrival_time_ms")
    if row.get("phase") not in {"primer", "warmup", "aborted", "profile"}:
        raise ValueError("request.phase must be primer, warmup, aborted or profile")
    for key in ("admission_history", "routing_history"):
        compact[key] = [dict(item) for item in _rows(row.get(key, []), f"request.{key}")]
        for item in compact[key]:
            for field in item:
                if field.endswith("_ms"):
                    _number(item[field], f"request.{key}.{field}")
    if row.get("agentic") is not None:
        compact["agentic"] = dict(_mapping(row["agentic"], "request.agentic"))
    return compact


def build_replay_dashboard_data(
    prediction: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt native observations without inventing missing metrics or durations."""
    prediction = _mapping(prediction, "prediction")
    prediction = {**prediction, **_mapping(prediction.get("summary", {}), "prediction.summary")}
    telemetry = _mapping(telemetry, "telemetry")
    if telemetry.get("schema") != "aisimulate.replay.telemetry.v1":
        raise ValueError("dashboard requires telemetry schema aisimulate.replay.telemetry.v1")
    if telemetry.get("clock") != "absolute_virtual_ms":
        raise ValueError("dashboard requires the native absolute_virtual_ms clock")
    units = _mapping(telemetry.get("units"), "telemetry.units")
    if any(units.get(key) != value for key, value in _UNITS.items()):
        raise ValueError("dashboard requires the native telemetry units")
    interval = _number(telemetry.get("sample_interval_ms"), "telemetry.sample_interval_ms", required=True)
    if interval == 0:
        raise ValueError("telemetry.sample_interval_ms must be positive")
    _number(telemetry.get("clock_origin_ms"), "telemetry.clock_origin_ms", required=True)
    _number(telemetry.get("profile_start_ms"), "telemetry.profile_start_ms")
    streams = {key: _rows(telemetry.get(key), f"telemetry.{key}") for key in _STREAMS}
    phases = {}
    for boundary in _rows(telemetry.get("phase_boundaries"), "telemetry.phase_boundaries"):
        phase = boundary.get("phase")
        if phase not in {"preparation", "profile"} or phase in phases:
            raise ValueError("telemetry.phase_boundaries must contain unique preparation/profile phases")
        start = _number(boundary.get("start_ms"), "phase.start_ms", required=True)
        end = _number(boundary.get("end_ms"), "phase.end_ms", required=True)
        if end < start:
            raise ValueError("phase.end_ms precedes start_ms")
        phases[phase] = dict(boundary)
    if not phases:
        raise ValueError("telemetry.phase_boundaries must contain at least one phase")
    profile = _mapping(prediction.get("agentic_profile", {}), "prediction.agentic_profile")
    cutoff = _metric(profile, "admission_cutoff_ms")
    if cutoff is not None and "profile" in phases:
        if cutoff < phases["profile"]["start_ms"]:
            raise ValueError("agentic_profile.admission_cutoff_ms precedes profile start")
        phases["profile"]["admission_cutoff_ms"] = cutoff
    engine = _mapping(_mapping(config or {}, "config").get("engine", {}), "config.engine")
    requests = [_request(row) for row in streams["requests"]]
    pools = {item.get("pool") for row in requests for item in row["routing_history"]}
    topology = engine.get("mode", prediction.get("deployment_mode"))
    if topology is None and pools:
        topology = "aggregated" if pools == {"agg"} else "disaggregated" if pools <= {"prefill", "decode"} else None
    lifecycle = []
    for row in streams["lifecycle"]:
        _number(row.get("at_ms"), "lifecycle.at_ms", required=True)
        if row.get("phase") not in {"primer", "warmup", "aborted", "profile"}:
            raise ValueError("lifecycle.phase must be primer, warmup, aborted or profile")
        lifecycle.append(dict(row))
    summary = {
        key: value
        for key, value in prediction.items()
        if value is None
        or isinstance(value, (str, int, float, bool))
        or key in {"agentic_graph", "agentic_model_projection", "agentic_profile"}
    }
    if prediction.get("agentic_phases") is not None:
        summary["agentic_phases"] = {
            key: value
            for key, value in _mapping(prediction["agentic_phases"], "prediction.agentic_phases").items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
    manifest = {
        key: value
        for source in (_mapping(prediction.get("telemetry", {}), "prediction.telemetry"), telemetry)
        for key, value in source.items()
        if key not in _STREAMS
    }
    return {
        "schema": "aisimulate.replay.dashboard.v1",
        "metadata": {
            "title": "AISimulate replay",
            "model": engine.get("model", prediction.get("model")),
            "hardware": engine.get("hardware", prediction.get("hardware")),
            "framework": engine.get("backend", prediction.get("backend")),
            "backend_version": engine.get("backend_version", prediction.get("backend_version")),
            "topology": topology,
            "plotly_python_version": plotly.__version__,
        },
        "prediction": summary,
        "telemetry": manifest,
        "phases": phases,
        "requests": requests,
        "samples": [_sample(row, aggregated=topology in {"agg", "aggregated"}) for row in streams["samples"]],
        "lifecycle": lifecycle,
    }


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_replay_dashboard(
    root: Path,
    prediction: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Write report.html atomically; no network or additional data files are needed."""
    payload = _script_json(build_replay_dashboard_data(prediction, telemetry, config=config))
    assets = Path(__file__).parent
    template = (assets / "dashboard.html").read_text(encoding="utf-8")
    replacements = {
        "__REPORT_DATA_JSON__": payload,
        "__PLOTLY_JS__": get_plotlyjs(),
        "__PLOTLY_LICENSE__": html.escape((assets / "plotly-js-LICENSE.txt").read_text(encoding="utf-8")),
    }
    for token in replacements:
        if template.count(token) != 1:
            raise ValueError(f"Expected exactly one {token} in dashboard template")
    # Replace template tokens in one pass, never tokens inside request text.
    document = re.sub(
        "|".join(re.escape(token) for token in replacements), lambda match: replacements[match[0]], template
    )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".report-", suffix=".tmp", delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(document)
        target = root / "report.html"
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_replay_dashboard(directory: Path, *, config: Mapping[str, Any] | None = None) -> Path:
    """Regenerate report.html from prediction.json and its native telemetry artifacts."""
    directory = Path(directory)
    prediction = _mapping(json.loads((directory / "prediction.json").read_text(encoding="utf-8")), "prediction")
    telemetry = dict(_mapping(prediction.get("telemetry"), "prediction.telemetry"))
    artifacts = _mapping(telemetry.get("artifacts"), "telemetry.artifacts")
    counts = _mapping(telemetry.get("record_counts"), "telemetry.record_counts")
    for stream in _STREAMS:
        filename = artifacts.get(stream)
        if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
            raise ValueError(f"telemetry.artifacts.{stream} must name a file inside the replay directory")
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f"telemetry.artifacts.{stream} is outside the replay directory")
        with path.open(encoding="utf-8") as source:
            telemetry[stream] = [json.loads(line) for line in source if line.strip()]
        if counts.get(stream) != len(telemetry[stream]):
            raise ValueError(f"telemetry.record_counts.{stream} does not match its artifact")
    return write_replay_dashboard(directory, prediction, telemetry, config=config)
