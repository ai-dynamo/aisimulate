# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared immutable shard identity checks, independent of either producer."""

from __future__ import annotations

import hashlib
import json

SCHEMA = "aic_fpm_shard_manifest"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_point_union(
    parent: dict,
    manifest: dict,
    child_plans: dict[str, dict],
    *,
    parent_cell_id: str | None = None,
) -> None:
    """Verify original IDs and bytes before admitting any child results."""
    if (
        manifest.get("schema_name") != SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("parent_plan_sha256") != parent["sha256"]
        or manifest.get("parent_points_sha256") != parent["options"]["benchmark_points"]["sha256"]
    ):
        raise ValueError("FPM shard manifest is not bound to its parent plan")
    if manifest != parent.get("sharding"):
        raise ValueError("FPM shard partition differs from the frozen parent inventory")
    points = parent["options"]["benchmark_points"]["payload"]
    if digest(points) != manifest["parent_points_sha256"]:
        raise ValueError("parent frozen point payload digest differs")
    cells = {cell["cell_id"]: cell for cell in parent["cells"]}
    if parent_cell_id is not None:
        if parent_cell_id not in cells:
            raise ValueError("FPM shard acceptance names an unknown parent cell")
        cells = {parent_cell_id: cells[parent_cell_id]}
    expected = {
        (cell_id, index)
        for cell_id, cell in cells.items()
        for index in range(1, len(points[cell["workload_kind"]]) + 1)
    }
    seen, child_ids = set(), set()
    for shard in manifest["shards"]:
        if parent_cell_id is not None and shard["parent_cell_id"] != parent_cell_id:
            continue
        child_id = shard["child_cell_id"]
        if child_id in child_ids or child_id not in child_plans:
            raise ValueError("FPM shard child identity is duplicated or missing")
        child_ids.add(child_id)
        child = child_plans[child_id]
        identity = {
            key: value for key, value in shard.items() if key not in {"shard_id", "child_cell_id", "child_plan_sha256"}
        }
        if (
            shard["shard_id"] != digest(identity)[:20]
            or child_id != f"fpm-shard-{shard['shard_id']}"
            or digest({"shard": identity, "child_plan": {**child, "sha256": ""}}) != child["sha256"]
        ):
            raise ValueError("FPM shard child content does not match its frozen digest")
        parent_cell = cells.get(shard["parent_cell_id"])
        if parent_cell is None or shard["phase"] != parent_cell["workload_kind"]:
            raise ValueError("FPM shard references another parent cell or phase")
        if child["sha256"] != shard["child_plan_sha256"] or len(child["cells"]) != 1:
            raise ValueError("FPM shard child plan identity differs")
        if child["cells"][0] != {**parent_cell, "cell_id": child_id}:
            raise ValueError("FPM shard altered the parent execution cell")
        for key in (
            "backend",
            "model_path",
            "system",
            "aic_revision",
            "generator_config_sha256",
            "capability",
            "dtype_profile",
            "topologies",
            "topology_memory_admission",
            "backend_policies",
        ):
            if child[key] != parent[key]:
                raise ValueError(f"FPM shard changed parent {key}")
        child_points = child["options"]["benchmark_points"]["payload"]
        expected_options = {key: value for key, value in parent["options"].items() if key != "shard_token_budget"}
        expected_options["benchmark_points"] = {"payload": child_points, "sha256": digest(child_points)}
        if child["options"] != expected_options:
            raise ValueError("FPM shard altered frozen parent execution or corpus options")
        phase = shard["phase"]
        mapping = shard["point_map"]
        if len(mapping) != len(child_points[phase]) or child_points["decode" if phase == "prefill" else "prefill"]:
            raise ValueError("FPM shard point coverage differs from its child plan")
        for local_id, (entry, point) in enumerate(zip(mapping, child_points[phase], strict=True), 1):
            key = (shard["parent_cell_id"], entry["original_point_id"])
            if key not in expected or key in seen or entry["native_benchmark_id"] != local_id:
                raise ValueError("FPM shard original point IDs overlap, are missing, or were renumbered")
            if point != entry["point"] or point != points[phase][entry["original_point_id"] - 1]:
                raise ValueError("FPM shard changed an original frozen point")
            seen.add(key)
    if seen != expected or child_ids != set(child_plans):
        raise ValueError("FPM shards do not form the exact complete parent point union")
