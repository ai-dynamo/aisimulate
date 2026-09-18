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


def validate_attention_geometry(attention, geometry: dict, layer_id: int) -> None:
    """Reject a planned identity that differs from the loaded serving module.

    SGLang@1aa0e962 dsv41_sparse.py:203-228 replicates all index heads and
    both projections across TP. The descriptor labels must reflect those
    actual dimensions before any timing or model-method instrumentation.
    """
    for field, attribute in (
        ("num_heads", "n_local_heads"),
        ("o_groups", "n_local_groups"),
        ("head_dim", "head_dim"),
        ("q_lora_rank", "q_lora_rank"),
        ("o_lora_rank", "o_lora_rank"),
        ("compress_ratio", "compress_ratio"),
    ):
        if int(getattr(attention, attribute)) != geometry[field]:
            raise RuntimeError(f"native attention {attribute} differs from graph at layer {layer_id}")
    indexer = getattr(attention, "indexer", None)
    owns_indexer = geometry["role"] in ("full", "reindex")
    if owns_indexer != (indexer is not None):
        raise RuntimeError(f"native indexer ownership differs from graph at layer {layer_id}")
    if indexer is None:
        return
    for field, attribute in (
        ("index_n_heads", "n_heads"),
        ("index_n_heads", "n_local_heads"),
        ("index_head_dim", "index_head_dim"),
        ("index_topk", "index_topk"),
    ):
        if int(getattr(indexer, attribute)) != geometry[field]:
            raise RuntimeError(f"native indexer {attribute} differs from graph at layer {layer_id}")
    expected = {
        "wq_b": (geometry["index_n_heads"] * geometry["index_head_dim"], geometry["q_lora_rank"]),
        "weights_proj": (geometry["index_n_heads"], geometry["hidden_size"]),
    }
    for name, shape in expected.items():
        if tuple(getattr(indexer, name).weight.shape) != shape:
            raise RuntimeError(f"native indexer {name} weight shape differs from graph at layer {layer_id}")


def validate_attention_manifest(layers, manifest: dict) -> None:
    """Validate both phase identities before installing any timing wrapper."""
    indexer = next((layer.self_attn.indexer for layer in layers if getattr(layer.self_attn, "indexer", None)), None)
    if indexer is None:
        raise RuntimeError("V4.1 collection requires an actual indexer owner")
    for phase in ("context", "generation"):
        entries = manifest["phases"][phase]
        for layer_id, layer in enumerate(layers):
            shapes = [e["geometry"] for e in entries if e["component"] == "attention" and e["layer"] == layer_id]
            if len(shapes) != 1:
                raise RuntimeError(f"expected one {phase} attention geometry at layer {layer_id}")
            geometry = json.loads(shapes[0])
            validate_attention_geometry(layer.self_attn, geometry, layer_id)
            # These global fields also label SWA/reuse rows, whose modules do
            # not own an indexer. Bind them to an actual owner rather than
            # allowing inconsistent unused dimensions into physical keys.
            for field, attribute in (
                ("index_n_heads", "n_heads"),
                ("index_head_dim", "index_head_dim"),
                ("index_topk", "index_topk"),
            ):
                if geometry[field] != int(getattr(indexer, attribute)):
                    raise RuntimeError(f"native indexer {attribute} differs from graph at layer {layer_id}")


def build_manifest(tp_size: int, decoder_replay: bool) -> dict:
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.deepseek_v41 import MODEL_PATH
    from aisimulate_core.sdk.models import get_model
    from aisimulate_core.sdk.utils import _load_pre_downloaded_hf_config

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
                        if kind == "Dsv41Attention":
                            # Match the measured-module key in the native reader:
                            # SOL's payload layout is not a measurement dimension.
                            body = {key: value for key, value in body.items() if key != "kv_cache_layout"}
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
        "config_sha256": sha256_json(_load_pre_downloaded_hf_config(MODEL_PATH)),
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
