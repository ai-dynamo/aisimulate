# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the offline interactive report from native replay and scoring artifacts.

The original UI template is adjacent. Plotly's unmodified installed JavaScript
bundle is embedded with its upstream license; see THIRD_PARTY_NOTICES.md.
"""

import html
import json
import math
from pathlib import Path

import plotly
from chart_comparisons import build_comparisons
from plotly.offline import get_plotlyjs

REQUEST_FIELDS = (
    "uuid",
    "request_id",
    "phase",
    "agentic_phase",
    "arrival_time_ms",
    "dispatched_at_ms",
    "first_admit_ms",
    "first_token_ms",
    "last_token_ms",
    "terminal_time_ms",
    "input_length",
    "output_length",
    "requested_output_length",
    "ttft_ms",
    "ttst_ms",
    "e2e_latency_ms",
    "itl_ms",
    "terminal_status",
    "first_admission_reused_input_tokens",
    "first_admission_g1_reused_input_tokens",
    "first_admission_host_reused_input_tokens",
    "reused_input_tokens",
    "agentic",
    "play_id",
    "session_id",
    "turn_index",
    "prefill_worker_idx",
    "decode_worker_idx",
    "admission_count",
    "readmission_count",
    "admission_history",
    "routing_history",
    "prefill_admit_ms",
    "decode_admit_ms",
    "prefill_complete_ms",
    "tokenized_ms",
    "encoder_ready_ms",
    "frontend_ready_ms",
    "scheduler_received_ms",
    "selected_ms",
)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def observed_sum(values):
    """Absent counters are unavailable; observed zero remains a measured zero."""
    observed = [value for value in values if finite(value)]
    return sum(observed) if observed else None


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_sample(row, aggregated):
    traffic = row.get("traffic") or {}
    admission = traffic.get("first_admission") or {}
    workers = [
        {**worker, "metric_role": role, "role": "agg" if aggregated and role == "decode" else role}
        for role in ("prefill", "decode")
        for worker in row.get(f"{role}_scheduler_metrics", [])
    ]
    intervals = [row.get(f"{role}_interval_metrics") or {} for role in ("prefill", "decode")]
    lookups = [interval.get("native_prefix_cache_lookup") or {} for interval in intervals]
    capacities = [worker.get("total_blocks") for worker in workers]
    capacity = sum(capacities) if workers and all(finite(value) and value > 0 for value in capacities) else None
    cache_pct = (
        100 * sum(worker["physical_cache_usage"] * worker["total_blocks"] for worker in workers) / capacity
        if capacity and all(finite(worker.get("physical_cache_usage")) for worker in workers)
        else None
    )
    return {
        "t": row["sampled_at_ms"] / 1000,
        "start": row["interval_start_ms"] / 1000,
        "phase": row.get("phase"),
        "kind": row.get("kind"),
        "ordinal": row.get("sample_ordinal"),
        "duration": traffic.get("duration_s"),
        "output_tokens": traffic.get("delivered_output_tokens"),
        "prefill_tokens": traffic.get("committed_prefill_tokens"),
        "arrivals": traffic.get("arriving_requests"),
        "completed": traffic.get("completed_requests"),
        "running": observed_sum(worker.get("running_requests") for worker in workers),
        "waiting": observed_sum(worker.get("waiting_requests") for worker in workers),
        "router_pending": observed_sum(row.get(f"router_pending_{role}_requests") for role in ("prefill", "decode")),
        "cache_pct": cache_pct,
        "lookup_hits": observed_sum(lookup.get("matched_tokens") for lookup in lookups),
        "lookup_queries": observed_sum(lookup.get("queried_tokens") for lookup in lookups),
        "lookup_count": observed_sum(lookup.get("query_count") for lookup in lookups),
        "scheduler_hits": observed_sum(interval.get("cache_hit_tokens") for interval in intervals),
        "scheduler_queries": observed_sum(interval.get("cache_total_tokens") for interval in intervals),
        "preemptions": observed_sum(interval.get("preemptions") for interval in intervals),
        "admission_input": admission.get("input_tokens"),
        "admission_requests": admission.get("requests"),
        "hbm": admission.get("hbm_reused_tokens"),
        "host": admission.get("host_reused_tokens"),
        "miss": admission.get("uncached_tokens"),
        "accept": traffic.get("avg_accept_length"),
        "accept_count": traffic.get("accept_length_forward_count"),
        "capacity_tokens": None,
        "workers": workers,
        "host_pools": row.get("host_caches", []),
    }


def compact_reference(reference):
    """Subtract integer epoch nanoseconds before converting to browser-safe time."""
    origins = {}
    for row in reference:
        phase = "profile" if row.get("phase") == "profiling" else row.get("phase")
        if isinstance(row.get("start_ns"), int):
            origins[phase] = min(origins.get(phase, row["start_ns"]), row["start_ns"])
    result = []
    for row in reference:
        phase = "profile" if row.get("phase") == "profiling" else row.get("phase")
        copy = dict(row)
        copy["phase"] = phase
        copy["phase_origin_ns"] = str(origins[phase]) if phase in origins else None
        for key in ("start_ns", "end_ns", "credit_ns"):
            value = row.get(key)
            copy[key] = str(value) if isinstance(value, int) else None
            copy[key.removesuffix("_ns") + "_ms"] = (
                (value - origins[phase]) / 1e6 if isinstance(value, int) and phase in origins else None
            )
        result.append(copy)
    return result


def browser_values(value):
    if isinstance(value, dict):
        return {
            key: str(item) if key.endswith("_ns") and isinstance(item, int) else browser_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [browser_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Report data contains a non-finite measurement")
    return value


def build_dataset(identifier, prediction, measurement, requests, reference, lifecycle, samples, report_dir):
    manifest = prediction.get("telemetry") or {}
    if manifest.get("clock") != "absolute_virtual_ms":
        raise ValueError("Interactive report requires the native absolute_virtual_ms clock")
    point_id = measurement.get("point_id", 446063)
    if not isinstance(point_id, int) or isinstance(point_id, bool) or point_id <= 0:
        raise ValueError("measurement.point_id must be a positive integer")
    pools = {item.get("pool") for row in requests for item in row.get("routing_history", [])}
    pairs_path = report_dir / "paired-requests.json"
    pairs = read_json(pairs_path) if pairs_path.exists() else []
    pairs_by_uuid = {row["simulated_uuid"]: row for row in pairs if row.get("simulated_uuid")}
    request_rows = []
    for row in requests:
        compact = {key: row[key] for key in REQUEST_FIELDS if key in row}
        compact["comparison"] = pairs_by_uuid.get(row.get("uuid"))
        request_rows.append(compact)
    summary = {
        key: value for key, value in prediction.items() if isinstance(value, (str, int, float, bool)) or value is None
    }
    summary.update({key: prediction[key] for key in ("agentic_model_projection", "agentic_graph") if key in prediction})
    return {
        "id": identifier,
        "point_id": point_id,
        "label": {"observed": "Recorded output lengths", "raw": "Original trace"}.get(
            identifier, measurement.get("variant", identifier)
        ),
        "prediction": summary,
        "measurement": measurement,
        "telemetry": manifest,
        "requests": request_rows,
        "samples": [normalize_sample(row, pools == {"agg"}) for row in samples],
        "reference": compact_reference(reference),
        "pairing": pairs,
        "lifecycle": [{key: value for key, value in row.items() if key != "agentic"} for row in lifecycle],
    }


def with_reference_workers(reference, reference_path):
    """Enrich a copy only when source lineage uniquely identifies a worker."""
    if not reference_path:
        return reference
    reference_path = Path(reference_path)
    timeline_path = reference_path.parent / "request-timeline.json"
    if timeline_path.is_file():
        timeline = read_json(timeline_path)
        workers = {}
        for row in timeline.get("requests", []):
            key = (
                row.get("phase"),
                row.get("srcTrace"),
                row.get("srcOuter"),
                row.get("srcInner"),
                row.get("cid"),
                row.get("ti"),
            )
            workers.setdefault(key, []).append(row.get("wid"))
        reference = [dict(row) for row in reference]
        for row in reference:
            key = (
                row.get("phase"),
                row.get("source_trace_id"),
                row.get("source_outer_idx"),
                row.get("source_inner_idx"),
                row.get("conversation_id"),
                row.get("turn_index"),
            )
            matches = workers.get(key, [])
            if len(matches) == 1 and matches[0] is not None:
                row["worker_id"] = matches[0]
    return reference


def build_report_data(
    output,
    prediction,
    measurement,
    requests,
    reference,
    lifecycle,
    samples,
    provenance,
    *,
    silicon=None,
    namespace_map=None,
):
    output = Path(output)
    variant = measurement.get("variant", "").lower()
    identifier = "observed" if "observed" in variant else "raw" if variant == "unmodified_full_corpus" else output.name
    reference = with_reference_workers(reference, provenance.get("reference_requests"))
    datasets = [build_dataset(identifier, prediction, measurement, requests, reference, lifecycle, samples, output)]
    current_prediction = measurement.get("provenance", {}).get("prediction", {}).get("path")
    for sibling in ("observed", "raw"):
        run_dir = output.parent / f"{sibling}-replay"
        report_dir = output.parent / f"{sibling}-report"
        prediction_path = run_dir / "prediction.json"
        measurement_path = report_dir / "measurement.json"
        if (
            sibling == identifier
            or not prediction_path.exists()
            or not measurement_path.exists()
            or (current_prediction and prediction_path.resolve() == Path(current_prediction).resolve())
        ):
            continue
        sibling_measurement = read_json(measurement_path)
        if sibling_measurement.get("point_id", 446063) != measurement.get("point_id", 446063):
            continue
        reference_path = sibling_measurement.get("provenance", {}).get("reference", {}).get("path")
        sibling_reference = (
            with_reference_workers(read_json(Path(reference_path)), reference_path)
            if reference_path and Path(reference_path).exists()
            else reference
        )
        datasets.append(
            build_dataset(
                sibling,
                read_json(prediction_path),
                sibling_measurement,
                read_rows(run_dir / "telemetry-requests.jsonl"),
                sibling_reference,
                read_rows(run_dir / "telemetry-lifecycle.jsonl"),
                read_rows(run_dir / "telemetry-samples.jsonl"),
                report_dir,
            )
        )
    for dataset in datasets:
        dataset["comparison"] = build_comparisons(dataset, silicon, namespace_map)
    return browser_values(
        {
            "schema": "aisimulate.agentx.interactive-report.v1",
            "activeVariant": identifier,
            "datasets": datasets,
            "provenance": provenance,
            "silicon": silicon,
            "plotly_python_version": plotly.__version__,
        }
    )


def script_json(value):
    return (
        json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_interactive_report(
    output,
    prediction,
    measurement,
    requests,
    reference,
    lifecycle,
    samples,
    provenance,
    *,
    silicon=None,
    namespace_map=None,
):
    data = build_report_data(
        output,
        prediction,
        measurement,
        requests,
        reference,
        lifecycle,
        samples,
        provenance,
        silicon=silicon,
        namespace_map=namespace_map,
    )
    directory = Path(__file__).resolve().parent
    template = (directory / "report_template.html").read_text()
    substitutions = {
        "__REPORT_DATA_JSON__": script_json(data),
        "__PLOTLY_JS__": get_plotlyjs(),
        "__PLOTLY_LICENSE__": html.escape((directory / "plotly-js-LICENSE.txt").read_text()),
    }
    for placeholder in substitutions:
        if template.count(placeholder) != 1:
            raise ValueError(f"Expected exactly one {placeholder} in report template")
    # Substitute only template tokens, never matching a token inside user data.
    chunks = template.split("__REPORT_DATA_JSON__")
    for placeholder in ("__PLOTLY_JS__", "__PLOTLY_LICENSE__"):
        chunks = [chunk.replace(placeholder, substitutions[placeholder]) for chunk in chunks]
    document = substitutions["__REPORT_DATA_JSON__"].join(chunks)
    path = Path(output) / "report.html"
    path.write_text(document)
    comparisons = {
        "schema": "aisimulate.agentx.chart-comparisons.v1",
        "point_id": measurement.get("point_id", 446063),
        "activeVariant": data["activeVariant"],
        "silicon_provenance": silicon.get("provenance") if silicon else None,
        "datasets": [{"id": item["id"], "comparison": item["comparison"]} for item in data["datasets"]],
    }
    (Path(output) / "chart-comparisons.json").write_text(script_json(comparisons) + "\n")
    if silicon is not None:
        (Path(output) / "silicon-telemetry.json").write_text(script_json(silicon) + "\n")
    else:
        (Path(output) / "silicon-telemetry.json").unlink(missing_ok=True)
    return path


def self_check():
    sample = {
        "sampled_at_ms": 1000,
        "interval_start_ms": 1000,
        "phase": "profile",
        "traffic": {"duration_s": 0, "delivered_output_tokens": 3},
        "decode_scheduler_metrics": [
            {"worker_id": 0, "dp_rank": 0, "physical_cache_usage": 0.5, "total_blocks": 10},
            {"worker_id": 0, "dp_rank": 1, "physical_cache_usage": 0.25, "total_blocks": 30},
        ],
        "decode_interval_metrics": {"cache_hit_tokens": 0, "cache_total_tokens": 0},
    }
    row = normalize_sample(sample, True)
    assert row["cache_pct"] == 31.25 and row["lookup_hits"] is None and row["scheduler_hits"] == 0
    assert row["duration"] == 0 and row["output_tokens"] == 3 and row["accept"] is None
    assert row["workers"][1]["role"] == "agg" and row["workers"][1]["dp_rank"] == 1
    sample["decode_scheduler_metrics"][1]["total_blocks"] = None
    assert normalize_sample(sample, False)["cache_pct"] is None
    epoch = 1792599226807345062
    reference = compact_reference([{"phase": "profiling", "start_ns": epoch, "end_ns": epoch + 1234567}])
    assert reference[0]["end_ms"] == 1.234567 and reference[0]["start_ns"] == str(epoch)
    injected = {"text": "</script><script>alert(1)</script>\u2028\u2029"}
    encoded = script_json(injected)
    assert "<" not in encoded and "\u2028" not in encoded and json.loads(encoded) == injected
    prediction = {"telemetry": {"clock": "absolute_virtual_ms"}, "agentic_lanes": 4}
    measurement = {"point_id": 446068, "variant": "observed_profile_output_lengths"}
    report = build_report_data(Path(__file__).parent, prediction, measurement, [], [], [], [], {})
    assert len(report["datasets"]) == 1 and report["activeVariant"] == "observed"
    assert report["datasets"][0]["prediction"]["agentic_lanes"] == 4
    assert report["datasets"][0]["measurement"]["point_id"] == 446068
    legacy = build_dataset("observed", prediction, {}, [], [], [], [], Path(__file__).parent)
    assert legacy["point_id"] == 446063 and legacy["measurement"] == {}


if __name__ == "__main__":
    self_check()
    print("Interactive report data self-check passed")
