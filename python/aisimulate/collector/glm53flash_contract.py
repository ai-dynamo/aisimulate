# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured GLM-5.3-Flash operation contract (CPU-only).

Measurements use the production graph's complete serialized geometry. Checkpoint
format remains a physical dimension even where two checkpoints have BF16 local
projections: their native dispatch equivalence has not been established.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

COMPONENTS = {"Glm53Attention": "attention", "Glm53Mhc": "mhc", "Glm53Router": "router"}
BACKENDS = {
    "vllm": ("0.30.0", "ced6857afa0ea7b2e3f0846a62e1394e90f15607"),
    "sglang": ("0.5.20", "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"),
}
CHECKPOINTS = {
    "fp8": ("zai-org/GLM-5.3-Flash", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"),
    "nvfp4": ("nvidia/GLM-5.3-Flash-NVFP4", "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"),
}
KEY_COLUMNS = ("component", "geometry", "batch_size", "prefix", "x")
INTEGER_COLUMNS = ("batch_size", "prefix", "x", "sample_count")
PROVENANCE_COLUMNS = (
    "backend",
    "backend_version",
    "backend_revision",
    "checkpoint_revision",
    "source_sha256",
    "config_sha256",
    "runtime_digest",
    "used_cuda_graph",
    "kernel_source",
    "state_mode",
)
ROW_COLUMNS = (
    *KEY_COLUMNS,
    "latency",
    "sample_count",
    "measurement_scope",
    "kv_seed_regime",
    *PROVENANCE_COLUMNS,
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def operation_geometry(body: dict) -> str:
    return canonical_json({key: value for key, value in body.items() if key != "name"})


def build_manifest(model) -> dict:
    """Take identities from a configured production model, never reconstruct them."""
    phases = {}
    for phase, ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        entries = []

        def visit(spec):
            kind, body = next(iter(spec.items()))
            if kind in COMPONENTS:
                entries.append(
                    {"component": COMPONENTS[kind], "name": body["name"], "geometry": operation_geometry(body)}
                )
            elif kind == "Overlap":
                for child in (*body["group_a"], *body["group_b"]):
                    visit(child)
            elif "children" in body:
                for child in body["children"]:
                    visit(child)

        for op in ops:
            visit(json.loads(op._spec_json()))
        if sum(entry["component"] == "attention" for entry in entries) != 45:
            raise ValueError("GLM-5.3-Flash manifest must cover all 45 text attention layers")
        if len({entry["name"] for entry in entries}) != len(entries):
            raise ValueError("native operation display names must uniquely identify graph occurrences")
        phases[phase] = entries
    return {"schema_version": 1, "phases": phases}


def _uint32(value, label: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not (int(positive) <= value <= 2**32 - 1):
        raise ValueError(f"{label} must be an exact {'positive ' if positive else ''}uint32")


def validate_row(row: dict) -> None:
    """Reject unverifiable identities before they can enter a measured table."""
    if row.get("component") not in COMPONENTS.values():
        raise ValueError("unknown GLM-5.3-Flash component")
    shape = json.loads(row["geometry"])
    if not isinstance(shape, dict) or "name" in shape or canonical_json(shape) != row["geometry"]:
        raise ValueError("geometry must be canonical JSON excluding the display name")
    checkpoint_format = shape.get("checkpoint_format")
    if checkpoint_format not in CHECKPOINTS:
        raise ValueError("geometry must preserve the exact checkpoint format")
    if row["checkpoint_revision"] != CHECKPOINTS[checkpoint_format][1]:
        raise ValueError("unqualified checkpoint revision")
    if row["backend"] not in BACKENDS or (row["backend_version"], row["backend_revision"]) != BACKENDS[row["backend"]]:
        raise ValueError("unqualified backend revision")
    if shape.get("backend") != row["backend"]:
        raise ValueError("operation geometry and observed backend disagree")
    for key in INTEGER_COLUMNS:
        _uint32(row[key], key, positive=key != "prefix")
    if isinstance(row["latency"], bool) or not math.isfinite(row["latency"]) or row["latency"] <= 0:
        raise ValueError("latency must be finite positive milliseconds")
    for key in ("source_sha256", "config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
            raise ValueError(f"invalid {key}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", row["runtime_digest"]):
        raise ValueError("immutable platform image digest required")
    if row["measurement_scope"] != "local_compute" or not row["kernel_source"].strip():
        raise ValueError("observed local compute dispatch is required")
    if not isinstance(row["used_cuda_graph"], bool):
        raise ValueError("used_cuda_graph must be boolean")
    if row["component"] == "attention":
        _uint32(shape.get("tp_size"), "tp_size", positive=True)
        if shape["tp_size"] not in (1, 2, 4) or (shape["tp_size"] == 1 and checkpoint_format != "nvfp4"):
            raise ValueError("unqualified TP/checkpoint combination")
        if not isinstance(shape.get("is_context"), bool):
            raise ValueError("attention is_context must be boolean")
        if shape["is_context"]:
            expected_modes = ("cached_prefill", "chunked_prefill") if row["prefix"] else ("full_prefill",)
            if row["state_mode"] not in expected_modes:
                raise ValueError("prefill state mode disagrees with its measured prefix")
            if row["prefix"] + row["x"] > 131072:
                raise ValueError("prefill exceeds the qualified 128K context")
        elif row["prefix"] or row["state_mode"] != "decode" or row["x"] > 131072:
            raise ValueError("decode requires absolute past-KV x, prefix=0, and decode state mode")
        if row["kv_seed_regime"] != ("real_kv" if row["prefix"] or not shape["is_context"] else "empty"):
            raise ValueError("cached prefill/decode requires native real-prefix state")
    elif (row["batch_size"], row["prefix"], row["kv_seed_regime"], row["state_mode"]) != (1, 0, "n/a", "token_only"):
        raise ValueError("token-only components require batch=1, prefix=0 and no state label")


def aggregate_rank_records(paths: list[Path], tp_size: int, manifest: dict) -> list[dict]:
    """Median of per-invocation rank maxima, after complete graph/rank coverage.

    The raw records retain layer occurrence, workload and sample identities.
    Identical shapes in different layers may reduce together only after every
    occurrence in each observed phase has been observed. Failed/incomplete attempts
    must remain on disk and cannot be repaired by merging attempts.
    """
    if {path.name for path in paths} != {f"rank-{rank}.jsonl" for rank in range(tp_size)}:
        raise ValueError("missing or unexpected TP rank files")
    groups = defaultdict(list)
    coverage = defaultdict(set)
    expected = {
        phase: {(entry["name"], entry["component"], entry["geometry"]) for entry in entries}
        for phase, entries in manifest["phases"].items()
    }
    for path in paths:
        expected_rank = int(path.stem.split("-")[1])
        for line in path.read_text().splitlines():
            row = json.loads(line)
            validate_row(row)
            if row["tp_rank"] != expected_rank or isinstance(row["tp_rank"], bool):
                raise ValueError("rank record does not belong to its evidence file")
            for key in ("sample", "invocation"):
                _uint32(row[key], key)
            phase = row["phase"]
            identity = (row["name"], row["component"], row["geometry"])
            if identity not in expected.get(phase, set()):
                raise ValueError("observed native operation is absent from the production graph")
            invocation_key = (expected_rank, phase, row["sample"], row["invocation"])
            if identity in coverage[invocation_key]:
                raise ValueError("native graph occurrence was observed more than once")
            coverage[invocation_key].add(identity)
            groups[tuple(row[key] for key in KEY_COLUMNS)].append(row)
    if not groups:
        raise ValueError("no native measurements")
    for (_, phase, _, _), observed in coverage.items():
        if observed != expected[phase]:
            raise ValueError(f"incomplete native {phase} graph coverage")
    output = []
    for rows in groups.values():
        if len({tuple(row[key] for key in PROVENANCE_COLUMNS) for row in rows}) != 1:
            raise ValueError("incompatible native invocations collide on one physical key")
        samples = defaultdict(dict)
        for row in rows:
            sample = samples[(row["phase"], row["sample"], row["invocation"], row["name"])]
            if row["tp_rank"] in sample:
                raise ValueError("duplicate rank within one native invocation")
            sample[row["tp_rank"]] = row["latency"]
        if any(set(sample) != set(range(tp_size)) for sample in samples.values()):
            raise ValueError("incomplete TP rank set within a native invocation")
        result = {key: rows[0][key] for key in ROW_COLUMNS}
        result["latency"] = statistics.median(max(sample.values()) for sample in samples.values())
        result["sample_count"] = len(samples)
        output.append(result)
    return output


def write_parquet(rows: list[dict], destination: Path) -> None:
    """Write one complete table; never silently replace duplicate physical keys."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError("cannot publish an empty measured table")
    keys = set()
    identities = {}
    for row in rows:
        validate_row(row)
        key = tuple(row[column] for column in KEY_COLUMNS)
        if key in keys:
            raise ValueError("duplicate GLM-5.3-Flash physical key")
        keys.add(key)
        checkpoint_format = json.loads(row["geometry"])["checkpoint_format"]
        identity = tuple(
            row[key] for key in PROVENANCE_COLUMNS if key not in ("used_cuda_graph", "kernel_source", "state_mode")
        )
        if identities.setdefault(checkpoint_format, identity) != identity:
            raise ValueError("table mixes runtime/config/source identities within a checkpoint format")
    schema = pa.schema(
        [
            (
                column,
                pa.int64()
                if column in INTEGER_COLUMNS
                else pa.float64()
                if column == "latency"
                else pa.bool_()
                if column == "used_cuda_graph"
                else pa.string(),
            )
            for column in ROW_COLUMNS
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination)
