# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode versioned columnar measurements into the existing semantic validators."""

import json
import math
from itertools import pairwise

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.fpm_accuracy.exceptions import DataError

SCHEMAS = {
    "fpm-iterations-parquet-v1",
    "forward-pass-records-parquet-v1",
    "iteration-index-parquet-v1",
    "request-metrics-parquet-v1",
}


def rows(file):
    reader = pq.ParquetFile(file.local_path)
    metadata = reader.schema_arrow.metadata or {}
    if file.storage_schema not in SCHEMAS or metadata.get(b"schema_id") != file.storage_schema.encode():
        raise DataError(f"Parquet schema mismatch: {file.path}")
    if metadata.get(b"schema_version") != b"1" or reader.metadata.num_rows != file.logical_row_count:
        raise DataError(f"Parquet version or row count mismatch: {file.path}")
    for name in ("source_row", "received_at_ns", "input_tokens", "output_tokens", "source_record"):
        if name in reader.schema_arrow.names and reader.schema_arrow.field(name).type != pa.int64():
            raise DataError(f"Parquet integer column type mismatch: {name}")
    for name in ("truth_latency_ms", "ttft_ms", "tpot_ms", "e2e_ms", "start_offset_ms"):
        if name in reader.schema_arrow.names and reader.schema_arrow.field(name).type != pa.float64():
            raise DataError(f"Parquet timing column type mismatch: {name}")
    if "rank_measurements" in reader.schema_arrow.names:
        rank_type = reader.schema_arrow.field("rank_measurements").type
        if not pa.types.is_list(rank_type) or not pa.types.is_struct(rank_type.value_type):
            raise DataError("Parquet rank measurements must be a typed list of structs")
        if rank_type.value_type.field("received_at_ns").type != pa.int64():
            raise DataError("Parquet rank timestamps must be int64 nanoseconds")
    previous = 0
    for batch in reader.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            if "source_row" in row:
                current = row["source_row"]
                if type(current) is not int or current <= previous:
                    raise DataError(f"Non-monotonic logical record order: {file.path}")
                previous = current
            yield row


def file_metadata(file):
    return json.loads((pq.ParquetFile(file.local_path).schema_arrow.metadata or {}).get(b"file_metadata", b"{}"))


def rank(record):
    value = {k: v for k, v in record.items() if k != "extensions_json" and v is not None}
    for key in ("scheduled_requests", "queued_requests"):
        if key in value:
            value[key] = {k: v for k, v in value[key].items() if v is not None}
    extras = json.loads(record["extensions_json"])
    for key in ("scheduled_requests", "queued_requests"):
        remaining = extras.pop("_metric_extensions_" + key, {})
        if remaining:
            value[key].update(remaining)
    value.update(extras)
    return value


def validate_collection_manifest(manifest):
    """Fail closed on corrupt run references before applying evaluation overrides."""
    files = {f["measurement_file_id"]: f for f in manifest["files"]}
    runs = manifest.get("collection_runs")
    if not isinstance(runs, list):
        raise DataError("Measurement v5 requires collection_runs")
    ids, ownership = set(), {}
    for file in files.values():
        if file.get("format") not in ("parquet", "json", "tsv", "raw"):
            raise DataError("Measurement v5 requires an explicit format")
        if file["format"] == "parquet" and (
            file.get("storage_schema") not in SCHEMAS
            or type(file.get("logical_row_count")) is not int
            or file["logical_row_count"] < 0
        ):
            raise DataError("Invalid Parquet representation or logical row count")
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("id"), str) or run["id"] in ids:
            raise DataError("Invalid or duplicate collection run identity")
        ids.add(run["id"])
        if run.get("collection_type") not in ("unknown", "self_benchmark", "static_serving", "trace_replay"):
            raise DataError("Unknown collection type")
        if any(
            not isinstance(run.get(k), dict) for k in ("dataset", "collector", "workload", "serving", "request_metrics")
        ):
            raise DataError("Incomplete collection metadata")
        for name in (
            "concurrency",
            "duration_s",
            "requested_requests",
            "completed_requests",
            "failed_requests",
            "cancelled_requests",
        ):
            value = run["workload"].get(name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise DataError("Invalid workload numeric value")
        if not isinstance(run.get("truth_bindings"), list):
            raise DataError("Missing collection truth bindings")
        for binding in run["truth_bindings"]:
            file = files.get(binding.get("file_id"))
            if file is None or file["role"] not in ("truth", "derived_truth"):
                raise DataError("Collection binding must reference declared truth")
            spans = binding.get("record_ranges")
            if spans is None:
                spans = [(1, math.inf)]
            if not isinstance(spans, (list, tuple)):
                raise DataError("Invalid logical record ranges")
            for span in spans:
                if len(span) != 2 or type(span[0]) is not int or span[0] < 1 or span[1] <= span[0]:
                    raise DataError("Invalid logical record range")
                ownership.setdefault(file["measurement_file_id"], []).append(span)
        metrics = run["request_metrics"]
        if metrics.get("status") not in ("available", "unavailable", "not_applicable"):
            raise DataError("Invalid request metric availability")
        if metrics["status"] == "available" and not metrics.get("file_ids"):
            raise DataError("Available request metrics require an asset")
        for fid in metrics.get("file_ids", []):
            if files.get(fid, {}).get("storage_schema") != "request-metrics-parquet-v1":
                raise DataError("Invalid request metric reference")
    for spans in ownership.values():
        ordered = sorted(spans)
        if any(a[1] > b[0] for a, b in pairwise(ordered)):
            raise DataError("Collection truth bindings overlap")


def iteration(row):
    ranks = [rank(item) for item in row["rank_measurements"]]
    if row["layout"] == "pre_grouped_rank_lists":
        return ranks
    if row["layout"] == "rank_observation":
        return ranks[0]
    if row["layout"] not in ("benchmark", "canonical_iteration"):
        raise DataError("Unknown Parquet iteration layout")
    value = json.loads(row["metadata_json"])
    for key in value.pop("_present_columns", []):
        value[key] = row[key]
    if row["layout"] == "benchmark":
        metadata = value.pop("_rank_result_metadata")
        value["rank_results"] = [{**meta, "fpms": [item]} for meta, item in zip(metadata, ranks, strict=True)]
    else:
        value["rank_measurements"] = ranks
    return value
