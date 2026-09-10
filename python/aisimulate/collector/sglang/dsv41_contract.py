# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact native-operation identities for V4.1 measurements.

This module is CPU-only. Geometry comes from the production model graph;
the runtime collector never reconstructs cache or attention metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

COMPONENTS = {"Dsv41Attention": "attention", "Dsv41Mhc": "mhc", "Dsv41Engram": "engram", "Dsv41Linear": "linear"}
INTEGER_COLUMNS = ("batch_size", "prefix", "x", "sample_count")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def operation_geometry(body: dict) -> str:
    return canonical_json({key: value for key, value in body.items() if key != "name"})


def build_manifest(tp_size: int, decoder_replay: bool) -> dict:
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.deepseek_v41 import MODEL_PATH
    from aiconfigurator_core.sdk.models import get_model

    model = get_model(
        MODEL_PATH,
        ModelConfig(
            tp_size=tp_size,
            pp_size=1,
            attention_dp_size=1,
            moe_tp_size=tp_size,
            moe_ep_size=1,
            decoder_replay=decoder_replay,
        ),
        "sglang",
    )
    phases = {}
    for phase, ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        rows = []
        for op in ops:
            spec = json.loads(op._spec_json())
            if "Dsv41Stage" not in spec:
                continue
            stage = spec["Dsv41Stage"]
            layer = int(stage["name"].rsplit("_", 1)[1])

            def visit(children):
                for child in children:
                    kind, body = next(iter(child.items()))
                    if kind in COMPONENTS:
                        rows.append(
                            {
                                "layer": layer,
                                "component": COMPONENTS[kind],
                                "geometry": operation_geometry(body),
                                "name": body["name"],
                            }
                        )
                    elif kind == "Overlap":
                        visit(body["group_a"])
                        visit(body["group_b"])

            visit(stage["children"])
        phases[phase] = rows
    return {
        "config_sha256": sha256_json(model.raw_config),
        "tp_size": tp_size,
        "execution_profile": model.execution_profile,
        "phases": phases,
    }


def validate_row(row: dict) -> None:
    if row.get("component") not in COMPONENTS.values():
        raise ValueError("unknown V4.1 component")
    geometry = json.loads(row["geometry"])
    if "name" in geometry or canonical_json(geometry) != row["geometry"]:
        raise ValueError("geometry must be canonical native body excluding name")
    for key in INTEGER_COLUMNS:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**32 - 1:
            raise ValueError(f"{key} must be an exact uint32")
    if not row["x"] or not row["batch_size"] or not row["sample_count"]:
        raise ValueError("empty measurement")
    if not math.isfinite(row["latency"]) or row["latency"] <= 0:
        raise ValueError("latency must be positive finite milliseconds")
    for key in ("source_sha256", "config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
            raise ValueError(f"invalid {key}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", row["runtime_digest"]):
        raise ValueError("immutable runtime digest required")
    if row["measurement_scope"] != "local_compute" or not row["kernel_source"]:
        raise ValueError("local compute dispatch witness required")
    if not isinstance(row["used_cuda_graph"], bool):
        raise ValueError("used_cuda_graph must be boolean")
    if row["execution_profile"] not in ("full", "decoder_bounded"):
        raise ValueError("invalid execution profile")
    needs_kv = row["component"] == "attention" and (not geometry["is_context"] or row["prefix"] > 0)
    if needs_kv and row["kv_seed_regime"] != "real_kv":
        raise ValueError("decode and cached prefill require real KV")
    if row["component"] != "attention" and (row["batch_size"], row["prefix"]) != (1, 0):
        raise ValueError("token-only component keys must use batch=1, prefix=0")
    if row["component"] != "attention" and row["kv_seed_regime"] != "n/a":
        raise ValueError("token-only components have no KV seed regime")
    if row["component"] == "attention":
        if not geometry["is_context"] and row["prefix"]:
            raise ValueError("decode uses absolute sequence length x and prefix=0")
        if geometry["bounded_prefill"] and (
            not geometry["is_context"]
            or row["execution_profile"] != "decoder_bounded"
            or row["x"] > geometry["window_size"]
        ):
            raise ValueError("bounded geometry requires bounded context within its window")


def write_parquet(rows: list[dict], path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys = set()
    provenance = set()
    for row in rows:
        validate_row(row)
        key = tuple(row[k] for k in ("component", "geometry", "batch_size", "prefix", "x"))
        if key in keys:
            raise ValueError(f"duplicate physical V4.1 key: {key}")
        keys.add(key)
        provenance.add(tuple(row[k] for k in ("source_sha256", "config_sha256", "runtime_digest", "used_cuda_graph")))
    if len(provenance) != 1:
        raise ValueError("a table needs one immutable runtime/config/source/measurement method")
    fields = [
        (
            k,
            pa.int64()
            if k in INTEGER_COLUMNS
            else pa.float64()
            if k == "latency"
            else pa.bool_()
            if k == "used_cuda_graph"
            else pa.string(),
        )
        for k in rows[0]
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema(fields)), path)
