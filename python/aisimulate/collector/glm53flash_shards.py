# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen physical ownership for complete native Ops calibration shards.

This module does not schedule requests or combine failed runs. Every child must
first pass its original native evidence adapter. Ownership depends only on the
production graph and pre-collection point map, never measured latency.
"""

from __future__ import annotations

import json

from collector.glm53flash_contract import (
    KEY_COLUMNS,
    PROVENANCE_COLUMNS,
    sha256_json,
    validate_calibration_row,
    validate_row,
)


def physical_ownership(model_manifest: dict, shard_manifest: dict, parent_cell_id: str) -> dict:
    if (shard_manifest.get("schema_name"), shard_manifest.get("schema_version")) != ("aic_fpm_shard_manifest", 1):
        raise ValueError("unknown frozen shard manifest")
    shards = [row for row in shard_manifest["shards"] if row["parent_cell_id"] == parent_cell_id]
    if not shards or len({row["shard_id"] for row in shards}) != len(shards):
        raise ValueError("missing or duplicate native shard declarations")
    seen_points, owners, local_owners = set(), {}, {}
    for shard in shards:
        local, native_ids = {}, set()
        phase = "context" if shard["phase"] == "prefill" else "generation"
        if shard["phase"] not in ("prefill", "decode") or phase not in model_manifest["phases"]:
            raise ValueError("frozen shard phase differs from model manifest")
        for mapped in shard["point_map"]:
            original, native = mapped["original_point_id"], mapped["native_benchmark_id"]
            if any(type(value) is not int or value < 1 for value in (original, native)):
                raise ValueError("shard point identities must be positive integers")
            if original in seen_points or native in native_ids:
                raise ValueError("shard points overlap or alias native benchmark IDs")
            seen_points.add(original)
            native_ids.add(native)
            point = mapped["point"]
            batch, past = point["batch_size"], point["total_kv_read_tokens"]
            new = point["total_prefill_tokens"] if phase == "context" else batch
            if any(type(value) is not int for value in (batch, past, new)) or batch < 1 or past < 0 or new < 1:
                raise ValueError("invalid frozen Ops workload")
            if past % batch or new % batch or (past + new) // batch > 131072:
                raise ValueError("frozen Ops workload is heterogeneous or exceeds context")
            for op in model_manifest["phases"][phase]:
                shape = json.loads(op["geometry"])
                if op["component"] == "attention":
                    coordinates = (
                        (batch, past // batch, new // batch) if phase == "context" else (batch, 0, past // batch)
                    )
                else:
                    coordinates = (1, 0, batch if shape.get("token_selection") == "last_per_request" else new)
                key = op["component"], op["geometry"], *coordinates
                candidate = {"original_point_id": original, "owner_benchmark_id": native, "shard_id": shard["shard_id"]}
                if key not in local or original < local[key]["original_point_id"]:
                    local[key] = candidate
                if key not in owners or original < owners[key]["original_point_id"]:
                    owners[key] = candidate
        if native_ids != set(range(1, len(native_ids) + 1)):
            raise ValueError("native shard benchmark IDs must cover its complete point array")
        local_owners[shard["shard_id"]] = local
    rows = [{**dict(zip(KEY_COLUMNS, key, strict=True)), **owner} for key, owner in sorted(owners.items())]
    frozen = {
        "schema": "glm53flash_ops_physical_ownership_v1",
        "policy": "lowest_frozen_original_point_id",
        "parent_cell_id": parent_cell_id,
        "shard_manifest_sha256": sha256_json(shard_manifest),
        "model_manifest_sha256": sha256_json(model_manifest),
        "rows": rows,
    }
    return {"frozen": frozen, "owners": owners, "local_owners": local_owners}


def merge_shard_rows(
    model_manifest: dict, shard_manifest: dict, parent_cell_id: str, rows_by_shard: dict
) -> tuple[list, dict]:
    ownership = physical_ownership(model_manifest, shard_manifest, parent_cell_id)
    if set(rows_by_shard) != set(ownership["local_owners"]):
        raise ValueError("Ops publication requires the exact complete frozen shard set")
    selected, baseline = {}, {}
    for shard_id, rows in rows_by_shard.items():
        expected, seen = ownership["local_owners"][shard_id], set()
        for row in rows:
            validate_row(row)
            validate_calibration_row(row)
            key = tuple(row[column] for column in KEY_COLUMNS)
            if key in seen or key not in expected:
                raise ValueError("Ops shard contains duplicate or undeclared physical keys")
            seen.add(key)
            if any(row.get(field) != expected[key][field] for field in ("owner_benchmark_id", "original_point_id")):
                raise ValueError("Ops shard row does not use its frozen physical owner")
            signature = tuple(
                row[field]
                for field in (
                    *PROVENANCE_COLUMNS,
                    "dispatch_fingerprint",
                    "corpus_sha256",
                    "measurement_scope",
                    "kv_seed_regime",
                )
            )
            if baseline.setdefault(key, signature) != signature:
                raise ValueError("Ops shards disagree on native runtime, dispatch, state or calibration corpus")
            if ownership["owners"][key]["shard_id"] == shard_id:
                selected[key] = row
        if seen != expected.keys():
            raise ValueError("Ops shard omits frozen physical measurements")
    if selected.keys() != ownership["owners"].keys():
        raise ValueError("Ops physical ownership is incomplete")
    rows = [selected[key] for key in sorted(selected)]
    return rows, ownership["frozen"]
