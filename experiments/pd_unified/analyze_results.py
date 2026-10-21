#!/usr/bin/env python3
"""Summarize a DynoSim baseline run using only the Python standard library.

All percentiles are equally weighted per-request percentiles, computed with
linear interpolation at (n - 1) * q. Scheduler cache occupancy is a KV-cache
metric; this analyzer does not infer GPU compute utilization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _nonnegative_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _difference(later: Any, earlier: Any) -> float | None:
    later_number, earlier_number = _number(later), _number(earlier)
    if later_number is None or earlier_number is None:
        return None
    difference = later_number - earlier_number
    # Suppress only sub-nanosecond numerical noise, not real ordering errors.
    if difference < -1e-6:
        return None
    return max(0.0, difference)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values) if values else None,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values) if values else None,
    }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]], leading: list[str]) -> None:
    fields = list(dict.fromkeys(leading + sorted({key for row in rows for key in row})))
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _request_mean_itl(record: dict[str, Any], output_length: int | None) -> float | None:
    if output_length is None or output_length <= 1:
        return None
    reported = _nonnegative_number(record.get("itl_ms"))
    if reported is not None:
        return reported
    span = _difference(record.get("last_token_ms"), record.get("first_token_ms"))
    if span is None:
        span = _difference(record.get("e2e_latency_ms"), record.get("ttft_ms"))
    return span / (output_length - 1) if span is not None else None


def _telemetry_tables(
    samples: list[dict[str, Any]], engine_mode: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scheduler_rows, traffic_rows = [], []
    is_aggregated = engine_mode in {"agg", "aggregated"}
    for sample in samples:
        envelope = {
            key: sample.get(key)
            for key in (
                "sample_ordinal",
                "kind",
                "interval_start_ms",
                "sampled_at_ms",
                "router_pending_prefill_requests",
                "router_pending_decode_requests",
            )
        }
        traffic = sample.get("traffic", {})
        if not isinstance(traffic, dict):
            raise ValueError("telemetry traffic must be an object")
        traffic_row = {**envelope, **traffic}
        for role in ("prefill", "decode"):
            output_role = "aggregated" if role == "decode" and is_aggregated else role
            interval = sample.get(f"{role}_interval_metrics", {})
            if not isinstance(interval, dict):
                raise ValueError(f"telemetry {role}_interval_metrics must be an object")
            for name, value in interval.items():
                traffic_row[f"{output_role}_{name}"] = value
            hits = _nonnegative_number(interval.get("cache_hit_tokens"))
            total = _nonnegative_number(interval.get("cache_total_tokens"))
            traffic_row[f"{output_role}_scheduler_cache_hit_ratio"] = (
                hits / total if hits is not None and total is not None and total > 0 else None
            )
            for lifecycle in ("active", "starting", "draining"):
                ids = sample.get(f"{lifecycle}_{role}_ids", [])
                traffic_row[f"{lifecycle}_{output_role}_ids"] = ids
                traffic_row[f"{lifecycle}_{output_role}_count"] = (
                    len(ids) if isinstance(ids, list) else None
                )
            gauges = sample.get(f"{role}_scheduler_metrics", [])
            if not isinstance(gauges, list):
                raise ValueError(f"telemetry {role}_scheduler_metrics must be a list")
            for gauge in gauges:
                if not isinstance(gauge, dict):
                    raise ValueError("scheduler gauge rows must be objects")
                scheduler_rows.append(
                    {**envelope, "role": output_role, "native_role": role, **gauge}
                )
        traffic_rows.append(traffic_row)
    return scheduler_rows, traffic_rows


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Write analysis JSON/CSVs and return the analysis, without running replay.

    A valid baseline requires all input requests to reach completed status and
    emit exactly their requested output length. Invalid runs retain observed
    diagnostics, but their published ``goodput_request_rps`` is null.
    """
    run_dir = Path(run_dir)
    native = _load_object(run_dir / "prediction.json")
    metadata = _load_object(run_dir / "run-metadata.json")
    summary = native.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("prediction.json must contain a summary object")
    records = native.get("per_request")
    if records is None and (run_dir / "requests.jsonl").is_file():
        records = _load_jsonl(run_dir / "requests.jsonl")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("per_request must be a list of JSON objects")

    input_requests = _count(metadata.get("input_requests"))
    num_reported = _count(summary.get("num_requests"))
    summary_completed = _count(summary.get("completed_requests"))
    issues: list[str] = []
    if input_requests is None or input_requests == 0:
        issues.append("input_requests must be a positive integer")
    if num_reported is None:
        issues.append("summary.num_requests is missing or invalid")
    if num_reported != input_requests:
        issues.append("summary.num_requests differs from input_requests")

    sla = metadata.get("sla", {"ttft_ms": 5000.0, "itl_ms": 50.0})
    if not isinstance(sla, dict):
        raise ValueError("run-metadata.json sla must be an object")
    ttft_limit = _nonnegative_number(sla.get("ttft_ms", 5000.0))
    itl_limit = _nonnegative_number(sla.get("itl_ms", 50.0))
    if ttft_limit is None or ttft_limit <= 0 or itl_limit is None or itl_limit <= 0:
        raise ValueError("sla.ttft_ms and sla.itl_ms must be positive finite numbers")

    statuses: Counter[str] = Counter()
    distributions: dict[str, list[float]] = {
        "ttft": [],
        "request_mean_itl": [],
        "first_admit_wait": [],
        "post_prefill_wait": [],
    }
    request_rows = []
    full_output_count = good_count = completed_output_mismatch_count = 0
    invalid_length_count = missing_latency_count = 0
    uuids: list[str] = []
    for record in records:
        status = record.get("terminal_status")
        status = status if isinstance(status, str) else "missing"
        statuses[status] += 1
        if record.get("uuid") is not None:
            uuids.append(str(record["uuid"]))
        requested = _count(record.get("requested_output_length"))
        actual = _count(record.get("output_length"))
        lengths_valid = requested is not None and actual is not None
        if not lengths_valid:
            invalid_length_count += 1
        full_output = status == "completed" and lengths_valid and actual == requested
        if status == "completed" and lengths_valid and actual != requested:
            completed_output_mismatch_count += 1
        ttft = _nonnegative_number(record.get("ttft_ms"))
        if ttft is None:
            ttft = _difference(record.get("first_token_ms"), record.get("arrival_time_ms"))
        mean_itl = _request_mean_itl(record, actual)
        admit_wait = _difference(record.get("first_admit_ms"), record.get("arrival_time_ms"))
        post_prefill_wait = _difference(record.get("decode_admit_ms"), record.get("source_held_ms"))
        good = bool(
            full_output
            and ttft is not None
            and ttft <= ttft_limit
            and (actual <= 1 or (mean_itl is not None and mean_itl <= itl_limit))
        )
        if full_output:
            full_output_count += 1
            good_count += int(good)
            if (actual > 0 and ttft is None) or (actual > 1 and mean_itl is None):
                missing_latency_count += 1
            for key, value in (
                ("ttft", ttft),
                ("request_mean_itl", mean_itl),
                ("first_admit_wait", admit_wait),
                ("post_prefill_wait", post_prefill_wait),
            ):
                if value is not None:
                    distributions[key].append(value)
        request_rows.append(
            {
                **record,
                "completed_full_output": full_output,
                "meets_sla_full_output": good,
                "analysis_ttft_ms": ttft,
                "request_mean_itl_ms": mean_itl,
                "first_admit_wait_ms": admit_wait,
                "post_prefill_wait_ms": post_prefill_wait,
            }
        )

    if len(records) != input_requests:
        issues.append("terminal record count differs from input_requests")
    if summary_completed != statuses["completed"]:
        issues.append("summary.completed_requests differs from completed terminal records")
    if full_output_count != input_requests:
        issues.append("not all input requests completed with their full requested output")
    if any(status != "completed" and count for status, count in statuses.items()):
        issues.append("non-completed or unknown terminal statuses are present")
    if completed_output_mismatch_count:
        issues.append("completed requests emitted a different output length than requested")
    if invalid_length_count:
        issues.append("request output lengths are missing or invalid")
    if missing_latency_count:
        issues.append("completed requests are missing latency evidence")
    if len(uuids) != len(set(uuids)):
        issues.append("duplicate request UUIDs are present")
    duration_ms = _nonnegative_number(summary.get("duration_ms"))
    if duration_ms is None or duration_ms <= 0:
        issues.append("summary.duration_ms must be positive for throughput accounting")
    valid = not issues
    observed_goodput = (
        good_count * 1000.0 / duration_ms if duration_ms is not None and duration_ms > 0 else None
    )

    telemetry_path = run_dir / "telemetry.jsonl"
    telemetry_source = "telemetry.jsonl"
    if telemetry_path.is_file():
        samples = _load_jsonl(telemetry_path)
    else:
        telemetry_source = "prediction.json"
        envelope = native.get("telemetry")
        samples = envelope.get("samples", []) if isinstance(envelope, dict) else []
        if not isinstance(samples, list) or any(not isinstance(row, dict) for row in samples):
            raise ValueError("telemetry.samples must be a list of objects")
    scheduler_rows, traffic_rows = _telemetry_tables(
        samples, str(metadata.get("engine_mode", ""))
    )
    stats = {key: _distribution(values) for key, values in distributions.items()}
    analysis: dict[str, Any] = {
        "schema_version": 1,
        "valid": valid,
        "invalid_reasons": issues,
        "case": metadata.get("case"),
        "engine_mode": metadata.get("engine_mode"),
        "num_gpus": metadata.get("num_gpus"),
        "speedup": metadata.get("speedup"),
        "sla": {"ttft_ms": ttft_limit, "itl_ms": itl_limit},
        "num_input_requests": input_requests,
        "num_reported_requests": num_reported,
        "num_terminal_records": len(records),
        "num_reported_without_terminal_record": (
            num_reported - len(records) if num_reported is not None else None
        ),
        "num_input_minus_reported": (
            input_requests - num_reported
            if input_requests is not None and num_reported is not None
            else None
        ),
        "num_completed_full_output": full_output_count,
        "num_completed_output_mismatch": completed_output_mismatch_count,
        "num_sla_good_full_output": good_count,
        "terminal_status_counts": dict(sorted(statuses.items())),
        "duration_ms": duration_ms,
        "goodput_request_rps": observed_goodput if valid else None,
        "observed_goodput_request_rps": observed_goodput,
        "p50_ttft_ms": stats["ttft"]["p50_ms"],
        "p95_ttft_ms": stats["ttft"]["p95_ms"],
        "p99_ttft_ms": stats["ttft"]["p99_ms"],
        "p50_request_mean_itl_ms": stats["request_mean_itl"]["p50_ms"],
        "p95_request_mean_itl_ms": stats["request_mean_itl"]["p95_ms"],
        "p99_request_mean_itl_ms": stats["request_mean_itl"]["p99_ms"],
        "p99_first_admit_wait_ms": stats["first_admit_wait"]["p99_ms"],
        "p99_post_prefill_wait_ms": stats["post_prefill_wait"]["p99_ms"],
        "distributions": stats,
        "telemetry_source": telemetry_source,
        "num_telemetry_samples": len(samples),
        "metric_semantics": {
            "percentiles": "Equal weight per full-output completed request; linear interpolation at (n-1)*q.",
            "request_mean_itl": "Distribution of per-request mean ITL; excludes OSL <= 1; not a pooled token-gap percentile.",
            "first_admit_wait": "first_admit_ms - arrival_time_ms; may include routing and scheduler admission.",
            "post_prefill_wait": "decode_admit_ms - source_held_ms; combined post-prefill wait, not isolated transfer time.",
            "goodput": "SLA-good full-output completed requests / native simulated duration; official value withheld for invalid runs.",
            "observed_goodput": "Diagnostic survivor numerator over native duration; never substitutes for a valid complete baseline.",
            "cache_usage": "KV-cache occupancy only; does not measure GPU compute utilization.",
            "telemetry": "Scheduler rows are point-in-time gauges; traffic and interval metrics are interval observations; router queues are separate.",
            "native_summary": "Preserved unchanged; native summary cohort or quantile semantics may differ from the analysis.",
        },
        "native_summary": summary,
    }
    _write_csv(
        run_dir / "requests.csv",
        request_rows,
        [
            "uuid",
            "arrival_time_ms",
            "terminal_status",
            "input_length",
            "requested_output_length",
            "output_length",
            "completed_full_output",
            "meets_sla_full_output",
            "analysis_ttft_ms",
            "request_mean_itl_ms",
            "first_admit_wait_ms",
            "post_prefill_wait_ms",
        ],
    )
    _write_csv(
        run_dir / "scheduler_samples.csv",
        scheduler_rows,
        [
            "sample_ordinal",
            "kind",
            "sampled_at_ms",
            "role",
            "worker_id",
            "dp_rank",
            "running_requests",
            "waiting_requests",
            "active_cache_usage",
            "physical_cache_usage",
            "router_pending_prefill_requests",
            "router_pending_decode_requests",
        ],
    )
    _write_csv(
        run_dir / "traffic_samples.csv",
        traffic_rows,
        [
            "sample_ordinal",
            "kind",
            "interval_start_ms",
            "sampled_at_ms",
            "arriving_requests",
            "completed_requests",
            "router_pending_prefill_requests",
            "router_pending_decode_requests",
        ],
    )
    (run_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze_run(args.run_dir)
    print(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if analysis["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
