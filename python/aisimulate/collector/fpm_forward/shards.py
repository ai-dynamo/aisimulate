# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable bounded partitions executed by the existing FPM campaign runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from .planner import FPMCollectionPlan

SCHEMA = "aic_fpm_shard_manifest"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class FPMShard:
    plan: FPMCollectionPlan
    identity: dict


def make_shards(plan: FPMCollectionPlan) -> tuple[FPMShard, ...]:
    """Keep each original point's full five warmups and ten samples together."""
    budget = plan.options.shard_token_budget
    if type(budget) is not int or budget < 1:
        raise ValueError("FPM shard token budget must be a positive integer")
    if (
        plan.capability.architecture != "Glm5NextForConditionalGeneration"
        or plan.options.benchmark_points_json is None
        or any(not cell.state_protocol for cell in plan.cells)
    ):
        raise ValueError("bounded FPM shards require a frozen explicit GLM native plan")
    original = json.loads(plan.options.benchmark_points_json)
    if digest(original) != plan.options.benchmark_points_sha256:
        raise ValueError("parent FPM point payload changed after freezing")
    shards = []
    for cell in plan.cells:
        phase = cell.workload_kind
        groups, current, current_tokens = [], [], 0
        for original_id, point in enumerate(original[phase], 1):
            query = point["total_prefill_tokens"] if phase == "prefill" else point["batch_size"]
            tokens = 15 * (point["total_kv_read_tokens"] + query)
            if current and current_tokens + tokens > budget:
                groups.append((current, current_tokens))
                current, current_tokens = [], 0
            current.append((original_id, point))
            current_tokens += tokens
        if current:
            groups.append((current, current_tokens))
        if not groups:
            raise ValueError("cannot shard an empty frozen phase")
        for group, tokens in groups:
            point_map = [
                {"native_benchmark_id": local_id, "original_point_id": original_id, "point": point}
                for local_id, (original_id, point) in enumerate(group, 1)
            ]
            identity = {
                "parent_plan_sha256": plan.sha256,
                "parent_points_sha256": plan.options.benchmark_points_sha256,
                "parent_cell_id": cell.cell_id,
                "phase": phase,
                "point_map": point_map,
                "requested_real_tokens": tokens,
                "oversized_single_point": len(group) == 1 and tokens > budget,
            }
            shard_id = digest(identity)[:20]
            child_id = f"fpm-shard-{shard_id}"
            payload = {"schema_version": original["schema_version"], "prefill": [], "decode": []}
            payload[phase] = [point for _, point in group]
            options = replace(
                plan.options,
                benchmark_points_json=canonical(payload),
                benchmark_points_sha256=digest(payload),
                shard_token_budget=None,
            )
            child = replace(plan, options=options, cells=(replace(cell, cell_id=child_id),), sha256="")
            child = replace(child, sha256=digest({"shard": identity, "child_plan": child.to_dict()}))
            shards.append(
                FPMShard(
                    child,
                    {
                        **identity,
                        "shard_id": shard_id,
                        "child_cell_id": child_id,
                        "child_plan_sha256": child.sha256,
                    },
                )
            )
    return tuple(shards)


def shard_manifest(plan: FPMCollectionPlan, shards: tuple[FPMShard, ...] | None = None) -> dict:
    return {
        "schema_name": SCHEMA,
        "schema_version": 1,
        "parent_plan_sha256": plan.sha256,
        "parent_points_sha256": plan.options.benchmark_points_sha256,
        "token_budget": plan.options.shard_token_budget,
        "token_budget_scope": "all requested seed and target tokens, five warmups plus ten measurements",
        "engine_lifecycle": "one existing runner/native Engine per child; weights reload between children",
        "shards": [shard.identity for shard in (make_shards(plan) if shards is None else shards)],
    }


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


def _immutable_json(path: Path, value) -> None:
    raw = canonical(value) + "\n"
    if path.exists():
        if path.read_text() != raw:
            raise ValueError(f"frozen shard identity changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(raw)


def run_sharded_collection(
    plan: FPMCollectionPlan,
    *,
    generator_overrides: dict,
    checkpoint_dir: str,
    artifact_root: str,
    resume: bool,
    retry_failed: bool,
    smoke: bool = False,
    cell_limit: int | None = None,
    database_root: str | None = None,
    publish_partial: bool = False,
    collect_only: bool = False,
) -> list[dict[str, object]]:
    """Reuse normal child execution and publish only a verified complete union."""
    from .database import aggregate_cell, validate_formal_database_commit, write_formal_database
    from .runner import _atomic_json, _file_manifest, _file_metadata, run_collection

    if smoke or cell_limit is not None or publish_partial:
        raise ValueError("frozen FPM shards forbid smoke, cell limits, and partial publication")
    shards = make_shards(plan)
    manifest = shard_manifest(plan, shards)
    validate_point_union(
        plan.to_dict(), manifest, {shard.identity["child_cell_id"]: shard.plan.to_dict() for shard in shards}
    )
    root = Path(artifact_root).expanduser().resolve() / plan.sha256[:16]
    root.mkdir(parents=True, exist_ok=True)
    _immutable_json(root / "collection-plan.json", plan.to_dict())
    _immutable_json(root / "shard-manifest.json", manifest)
    checkpoints = Path(checkpoint_dir).expanduser().resolve()
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints / "fpm_forward_sharded.json"
    checkpoint = {
        "schema": "aic_fpm_sharded_checkpoint_v1",
        "parent_plan_sha256": plan.sha256,
        "shard_manifest_sha256": digest(manifest),
        "shards": {},
        "status": "incomplete",
        "accuracy_acceptance": "NOT_EVALUATED",
    }
    if checkpoint_path.exists():
        previous = json.loads(checkpoint_path.read_text())
        if not resume:
            raise ValueError("existing bounded campaign requires --resume; no attempt evidence is overwritten")
        if any(
            previous.get(key) != checkpoint[key]
            for key in (
                "schema",
                "parent_plan_sha256",
                "shard_manifest_sha256",
            )
        ):
            raise ValueError("bounded checkpoint differs from its immutable parent and shard partition")
        checkpoint = previous
    _atomic_json(checkpoint_path, checkpoint)
    errors, completed = [], []
    for shard in shards:
        child_id = shard.identity["child_cell_id"]
        child_checkpoint_dir = checkpoints / "shards" / child_id
        child_checkpoint_path = child_checkpoint_dir / "fpm_forward.json"
        child_root = root / "shards" / shard.plan.sha256[:16]
        _immutable_json(root / "plans" / f"{child_id}.json", shard.plan.to_dict())
        try:
            child_errors = run_collection(
                shard.plan,
                generator_overrides=generator_overrides,
                checkpoint_dir=str(child_checkpoint_dir),
                artifact_root=str(root / "shards"),
                resume=resume,
                retry_failed=retry_failed,
                collect_only=True,
            )
            errors.extend({**error, "shard_id": shard.identity["shard_id"]} for error in child_errors)
            child_checkpoint = json.loads(child_checkpoint_path.read_text())
            entry = child_checkpoint["cells"].get(child_id, {})
            checkpoint["shards"][child_id] = {
                **shard.identity,
                "status": entry.get("status", "incomplete"),
                "attempt_id": entry.get("attempt_id"),
                "checkpoint": str(child_checkpoint_path),
                "artifact_root": str(child_root),
            }
            if entry.get("status") == "passed":
                completed.append((shard, child_root / "cells" / child_id, entry["attempt_id"]))
        except (KeyboardInterrupt, SystemExit):
            checkpoint["shards"][child_id] = {**shard.identity, "status": "interrupted"}
            _atomic_json(checkpoint_path, checkpoint)
            raise
        except Exception as error:
            failure = {"shard_id": shard.identity["shard_id"], "error_type": type(error).__name__, "error": str(error)}
            errors.append(failure)
            checkpoint["shards"][child_id] = {**shard.identity, "status": "failed", **failure}
        _atomic_json(checkpoint_path, checkpoint)
    if errors or len(completed) != len(shards):
        checkpoint["status"] = "incomplete"
        checkpoint["missing_children"] = [
            shard.identity["child_cell_id"]
            for shard in shards
            if checkpoint["shards"].get(shard.identity["child_cell_id"], {}).get("status") != "passed"
        ]
        _atomic_json(checkpoint_path, checkpoint)
        return errors or [{"error_type": "campaign_incomplete", "missing_children": checkpoint["missing_children"]}]
    rows, receipts = [], []
    try:
        for shard, child_dir, attempt_id in completed:
            child_rows = aggregate_cell(shard.plan, shard.plan.cells[0], child_dir, expected_attempt_id=attempt_id)

            def coordinates(point):
                return point["batch_size"], point.get("total_prefill_tokens", 0), point["total_kv_read_tokens"]

            expected = {coordinates(entry["point"]) for entry in shard.identity["point_map"]}
            actual = {coordinates(row) for row in child_rows}
            if actual != expected or len(actual) != len(child_rows):
                raise ValueError("native shard rows differ from the complete original point subset")
            rows.extend(child_rows)
            receipts.append(
                {
                    **shard.identity,
                    "collector_attempt_id": attempt_id,
                    "artifact_root": str(child_dir),
                    "files": _file_manifest(child_dir / "raw"),
                    "archived_attempt_receipts": {
                        str(path.relative_to(child_dir)): _file_metadata(path)
                        for path in sorted((child_dir / "attempts").glob("*/file-receipts.json"))
                    },
                    "rows_sha256": digest(child_rows),
                }
            )
        verified = {
            "schema_name": "aic_fpm_verified_shard_union",
            "schema_version": 1,
            "parent_plan_sha256": plan.sha256,
            "shard_manifest_sha256": digest(manifest),
            "shards": receipts,
            "row_count": len(rows),
            "accuracy_acceptance": "NOT_EVALUATED",
        }
        # Failed attempts remain in each child's attempts/ subtree; this receipt
        # names the independently verified attempt selected for publication.
        _atomic_json(root / "verified-union.json", verified)
        verified_file = _file_metadata(root / "verified-union.json")
        checkpoint["verified_union_sha256"] = verified_file["sha256"]
        if plan.options.dataset_role == "holdout" or collect_only:
            checkpoint.update(status="passed", formal_database_written=False)
        else:
            parquet, metadata, skipped = write_formal_database(
                plan,
                rows,
                systems_root=Path(database_root).expanduser().resolve() if database_root else None,
                reject_replaced_cells=True,
            )
            if skipped:
                raise ValueError("verified union publication unexpectedly omitted child cells")
            cell_rows = {}
            for shard, _, attempt_id in completed:
                committed = validate_formal_database_commit(
                    parquet,
                    metadata,
                    shard.plan,
                    expected_attempt_ids={shard.identity["child_cell_id"]: attempt_id},
                )
                cell_rows.update(committed["cell_rows"])
            checkpoint.update(
                status="passed",
                formal_database_written=True,
                database={
                    "parquet": str(parquet),
                    "metadata": str(metadata),
                    "cell_rows": cell_rows,
                    "verified_union_sha256": verified_file["sha256"],
                },
            )
        _atomic_json(checkpoint_path, checkpoint)
    except Exception as error:
        failure = {"error_type": type(error).__name__, "error": str(error)}
        checkpoint.update(status="failed", publication_error=failure)
        _atomic_json(checkpoint_path, checkpoint)
        errors.append(failure)
    return errors
