# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit pre-freeze Ops leaves, preserving the complete original FPM union.

The observation family is a declaration, never a native dispatch override.
Source: vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607,
v1/worker/gpu/cudagraph_utils.py (_init_candidates/dispatch), Apache-2.0.
New local benchmark IDs change token offsets under the unchanged scheduler;
only geometry and corpus identity are inherited, not historical token bytes.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from types import SimpleNamespace

from collector.glm53flash_contract import CHECKPOINTS, canonical_json, sha256_json
from collector.glm53flash_jsonl import file_sha256
from collector.glm53flash_shard_contract import validate_point_union
from collector.glm53flash_vllm_graph_policy import SOURCE_PINS

CONTRACT = "glm53flash_ops_observation_partition_v1"
LEAF_SCHEMA = "glm53flash_ops_observation_leaf_v1"
RUNTIME_CONTRACT = "glm53flash_formal_observation_runtime_v1"
OWNERSHIP_SCHEMA = "glm53flash_serving_observation_ownership_v1"
FAMILIES = ("NONE", "PIECEWISE", "FULL")
RULE = {
    "native_source_sha256": SOURCE_PINS["v1/worker/gpu/cudagraph_utils.py"],
    "max_num_seqs": 32,
    "max_num_batched_tokens": 8192,
    "max_capture_tokens": 2048,
    "selection": "decode FULL; prefill BQ<=2048 PIECEWISE; prefill BQ>2048 NONE",
    "actual_dispatch_required": True,
}


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("observation identity requires lowercase SHA256")
    return value


def family_for(phase, point):
    """Declare an observation family for this source/config, then verify reality."""
    batch, past = point.get("batch_size"), point.get("total_kv_read_tokens")
    new = point.get("total_prefill_tokens", 0)
    if (
        type(batch) is not int
        or not 1 <= batch <= RULE["max_num_seqs"]
        or type(past) is not int
        or past < 0
        or past % batch
        or type(new) is not int
        or phase not in ("prefill", "decode")
        or (phase == "decode" and new != 0)
        or (phase == "prefill" and (not 1 <= new <= RULE["max_num_batched_tokens"] or new % batch))
        or (past + (batch if phase == "decode" else new)) // batch > 131072
    ):
        raise ValueError("observation partition requires bounded homogeneous original geometry")
    return "FULL" if phase == "decode" else "PIECEWISE" if new <= RULE["max_capture_tokens"] else "NONE"


def validate_runtime_identity(identity):
    """Validate a new common source declaration, not an execution qualification."""
    if not isinstance(identity, dict) or identity.get("schema") != RUNTIME_CONTRACT:
        raise ValueError("observation partition requires an explicit common runtime identity")
    producer = identity.get("producer", {})
    if not isinstance(producer.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", producer["commit"]):
        raise ValueError("observation producer commit is missing")
    for name in ("arm_wheel_sha256", "source_resource_map_sha256"):
        _hash(producer.get(name))
    for name in ("shared_cache_sha256", "shared_overlay_sha256"):
        _hash(identity.get(name))
    files, mechanisms = identity.get("entry_files", {}), identity.get("entry_mechanisms", {})
    if not files or set(mechanisms) != set(FAMILIES) or not isinstance(identity.get("native_runtime"), dict):
        raise ValueError("common runtime lacks both original entry mechanisms and native identity")
    for name, entry in files.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("runtime entry source name must be relative")
        _hash(entry.get("sha256"))
        if type(entry.get("bytes")) is not int or entry["bytes"] < 1:
            raise ValueError("runtime entry source size is missing")
    for family, mechanism in mechanisms.items():
        flag = "AISIM_GLM53_SERVING_NONE_MEASURED" if family == "NONE" else "AISIM_GLM53_PIECEWISE_REPLAY"
        if (
            mechanism.get("entry") not in files
            or mechanism.get("observer_flag") != flag
            or mechanism.get("control_holdout_addon") is not False
            or not isinstance(mechanism.get("calibration_addon"), list)
            or any(name not in files for name in mechanism["calibration_addon"])
        ):
            raise ValueError("runtime observation entry mechanism differs")
    if mechanisms["FULL"] != mechanisms["PIECEWISE"]:
        raise ValueError("FULL and PIECEWISE must share the native graph entry mechanism")
    return "sha256:" + sha256_json(identity)


def build_partition(parent, child_plans, runtime_identity, *, parent_cell_id, role, campaign_id):
    """Derive all leaves from complete original bytes; no latency/result input."""
    if (
        parent.get("schema_name") != "aic_fpm_collection_plan"
        or parent.get("backend") != "vllm"
        or role not in ("calibration", "control", "holdout")
        or not isinstance(campaign_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id)
    ):
        raise ValueError("unsupported observation source, role or campaign identity")
    dataset_role = "calibration" if role == "control" else role
    if parent["options"].get("dataset_role", "calibration") != dataset_role:
        raise ValueError("observation role differs from original dataset")
    if any(
        type(parent["options"].get(key)) is not int or parent["options"][key] != value
        for key, value in (("warmup_repeats", 5), ("measurement_repeats", 10))
    ):
        raise ValueError("observation partition must preserve original five warmup/ten retained repeats")
    validate_point_union(parent, parent["sharding"], child_plans)
    cells = [cell for cell in parent["cells"] if cell["cell_id"] == parent_cell_id]
    if len(cells) != 1:
        raise ValueError("observation partition references unknown original parent cell")
    cell = cells[0]
    phase = cell["workload_kind"]
    corpus = _hash(parent["options"]["input_text_sha256"])
    if cell.get("input_text_sha256") != corpus:
        raise ValueError("observation original parent corpus differs")
    runtime_digest = validate_runtime_identity(runtime_identity)
    leaves = []
    for shard in parent["sharding"]["shards"]:
        if shard["parent_cell_id"] != parent_cell_id:
            continue
        if any(
            type(item.get(key)) is not int
            for item in shard["point_map"]
            for key in ("native_benchmark_id", "original_point_id")
        ):
            raise ValueError("original observation point identities must be exact integers")
        for family in FAMILIES:
            selected = [item for item in shard["point_map"] if family_for(phase, item["point"]) == family]
            if not selected:
                continue
            mapping = [
                dict(
                    native_benchmark_id=i,
                    original_child_benchmark_id=item["native_benchmark_id"],
                    original_point_id=item["original_point_id"],
                    point=copy.deepcopy(item["point"]),
                )
                for i, item in enumerate(selected, 1)
            ]
            payload = {
                "schema_version": parent["options"]["benchmark_points"]["payload"]["schema_version"],
                "prefill": [],
                "decode": [],
            }
            payload[phase] = [item["point"] for item in mapping]
            body = {
                "campaign_id": campaign_id,
                "role": role,
                "phase": phase,
                "observation_family": family,
                "original_parent_cell_id": parent_cell_id,
                "original_parent_plan_sha256": parent["sha256"],
                "original_child_cell_id": shard["child_cell_id"],
                "original_child_plan_sha256": shard["child_plan_sha256"],
                "original_shard_id": shard["shard_id"],
                "corpus_sha256": corpus,
                "point_map": mapping,
                "points_payload": payload,
                "runtime_digest": runtime_digest,
                "warmup_repeats": 5,
                "measurement_repeats": 10,
            }
            leaf_id = "ops-observation-" + sha256_json(body)[:24]
            leaves.append({"leaf_id": leaf_id, "run_id": campaign_id + "-" + leaf_id, **body})
    if not leaves:
        raise ValueError("observation partition has no original points")
    return {
        "schema": CONTRACT,
        "campaign_id": campaign_id,
        "role": role,
        "parent_cell_id": parent_cell_id,
        "source_parent_sha256": sha256_json(parent),
        "source_shard_manifest_sha256": sha256_json(parent["sharding"]),
        "source_children": copy.deepcopy(child_plans),
        "runtime_identity": copy.deepcopy(runtime_identity),
        "runtime_digest": runtime_digest,
        "family_rule": copy.deepcopy(RULE),
        "leaves": leaves,
        "token_identity": "NEW_LEAF_REQUESTS_NOT_HISTORICAL_TOKEN_REPLAY",
    }


def validate_partition(partition, parent):
    if partition.get("schema") != CONTRACT:
        raise ValueError("unsupported observation partition contract")
    expected = build_partition(
        parent,
        partition["source_children"],
        partition["runtime_identity"],
        parent_cell_id=partition["parent_cell_id"],
        role=partition["role"],
        campaign_id=partition["campaign_id"],
    )
    if canonical_json(partition) != canonical_json(expected):
        raise ValueError("observation partition differs from complete original source/point union")
    return {leaf["leaf_id"]: leaf for leaf in expected["leaves"]}


def leaf_plan(partition, leaf_id):
    """A new Ops plan, deliberately not an FPMCollectionPlan."""
    leaf = next(item for item in partition["leaves"] if item["leaf_id"] == leaf_id)
    source = partition["source_children"][leaf["original_child_cell_id"]]
    options = copy.deepcopy(source["options"])
    options["benchmark_points"] = {
        "payload": copy.deepcopy(leaf["points_payload"]),
        "sha256": sha256_json(leaf["points_payload"]),
    }
    body = {
        "schema_name": LEAF_SCHEMA,
        "schema_version": 1,
        **{key: copy.deepcopy(source[key]) for key in ("backend", "model_path", "system", "capability")},
        "cells": [{**source["cells"][0], "cell_id": leaf_id}],
        "options": options,
        "observation": copy.deepcopy(leaf),
        "partition_sha256": sha256_json(partition),
    }
    return {**body, "sha256": sha256_json(body)}


def load_partition_run(parent, base):
    """Explicit public plan route; original plans and new leaves remain separate."""
    from collector.fpm_forward.glm53flash_validation import _geometry, _read_json_receipt

    spec = parent["spec"]
    if (
        spec.get("ops_execution_mode") != "native_serving"
        or parent["key"][0] != "vllm"
        or "shards" in spec
        or "shard_manifest" in spec
        or "raw_root" in spec
    ):
        raise ValueError("observation partition requires a distinct serving parent, not an aggregate native run")
    partition = _read_json_receipt(spec["ops_observation_partition"], base)
    declared = validate_partition(partition, parent["plan"])
    if partition["parent_cell_id"] != parent["cell"]["cell_id"] or partition["role"] != parent["role"]:
        raise ValueError("observation partition changed selected parent/role")
    source_refs = spec.get("observation_runtime_sources", {})
    required = partition["runtime_identity"]["entry_files"]
    if source_refs.keys() != required.keys():
        raise ValueError("observation entry source closure is incomplete")
    for name, ref in source_refs.items():
        path = base / ref["path"]
        if (
            path.is_symlink()
            or file_sha256(path) != required[name]["sha256"]
            or ref["sha256"] != required[name]["sha256"]
            or path.stat().st_size != required[name]["bytes"]
        ):
            raise ValueError("observation entry source bytes changed")
    supplied = spec.get("observation_children", [])
    if len(supplied) != len(declared) or {item.get("cell_id") for item in supplied} != declared.keys():
        raise ValueError("observation children do not cover the complete original union")
    children = []
    for child_spec in supplied:
        cid = child_spec["cell_id"]
        leaf = declared[cid]
        plan = _read_json_receipt(child_spec["plan"], base)
        if (
            plan != leaf_plan(partition, cid)
            or child_spec.get("ops_execution_mode") != "native_serving"
            or not child_spec.get("raw_root")
            or any(key in child_spec for key in ("shards", "ops_observation_partition", "observation_children"))
        ):
            raise ValueError("observation child differs from its custom frozen Ops plan")
        points = [
            dict(
                item["point"],
                benchmark_id=item["native_benchmark_id"],
                point_type=leaf["phase"],
                total_prefill_tokens=item["point"].get("total_prefill_tokens", 0),
            )
            for item in leaf["point_map"]
        ]
        children.append(
            {
                **{key: value for key, value in parent.items() if key not in ("children", "shard_manifest")},
                "plan": plan,
                "cell": plan["cells"][0],
                "spec": child_spec,
                "points": points,
                "geometries": {_geometry(point) for point in points},
                "runtime_cell": SimpleNamespace(**{**vars(parent["runtime_cell"]), "cell_id": cid}),
                "original_point_ids": {
                    item["native_benchmark_id"]: item["original_point_id"] for item in leaf["point_map"]
                },
                "observation_partition": partition,
                "observation_leaf": leaf,
            }
        )
    parent.update(
        children=children,
        shard_manifest=partition,
        observation_partition=partition,
        parent_cell_id=parent["cell"]["cell_id"],
    )
    validate_children(parent, children)
    return parent


def validate_children(parent, children):
    partition = parent["observation_partition"]
    declared = validate_partition(partition, parent["plan"])
    phase = parent["cell"]["workload_kind"]
    fmt = {"fp8": "fp8", "fp8_block": "fp8", "nvfp4": "nvfp4"}.get(parent["cell"]["weight_quantization"])
    expected_parent = [
        dict(point, benchmark_id=i, point_type=phase, total_prefill_tokens=point.get("total_prefill_tokens", 0))
        for i, point in enumerate(parent["plan"]["options"]["benchmark_points"]["payload"][phase], 1)
    ]
    if (
        parent["spec"].get("ops_execution_mode") != "native_serving"
        or fmt not in CHECKPOINTS
        or parent["plan"]["model_path"] != CHECKPOINTS[fmt][0]
        or tuple(parent["key"]) != ("vllm", fmt, parent["cell"]["topology"]["tp"], phase)
        or parent["points"] != expected_parent
        or parent["cell"] not in parent["plan"]["cells"]
        or parent.get("shard_manifest") != partition
        or partition["role"] != parent["role"]
        or partition["parent_cell_id"] != parent["cell"]["cell_id"]
        or len(children) != len(declared)
        or {run["cell"]["cell_id"] for run in children} != declared.keys()
    ):
        raise ValueError("observation group differs from complete parent/role/leaf union")
    result = {}
    for run in children:
        cid = run["cell"]["cell_id"]
        leaf = declared[cid]
        expected_points = [
            dict(
                item["point"],
                benchmark_id=item["native_benchmark_id"],
                point_type=leaf["phase"],
                total_prefill_tokens=item["point"].get("total_prefill_tokens", 0),
            )
            for item in leaf["point_map"]
        ]
        if (
            run.get("observation_partition") != partition
            or run.get("observation_leaf") != leaf
            or run["plan"] != leaf_plan(partition, cid)
            or run["cell"] != run["plan"]["cells"][0]
            or tuple(run["key"]) != tuple(parent["key"])
            or run["role"] != parent["role"]
            or run["corpus"] != parent["corpus"]
            or run["corpus"] != leaf["corpus_sha256"]
            or run["points"] != expected_points
            or "children" in run
            or run["spec"].get("ops_execution_mode") != "native_serving"
            or run.get("original_point_ids")
            != {item["native_benchmark_id"]: item["original_point_id"] for item in leaf["point_map"]}
        ):
            raise ValueError("observation leaf changed original geometry/source/role/runtime identity")
        result[cid] = {**leaf, "shard_id": cid}
    return result


def check_native_proof(run, proof):
    """Additional family/run gate AFTER the unchanged strict serving reader."""
    if "observation_leaf" not in run:
        if run.get("plan", {}).get("schema_name") == LEAF_SCHEMA or "observation_partition" in run:
            raise ValueError("custom observation leaf cannot bypass its explicit native contract")
        return
    leaf = run["observation_leaf"]
    partition = run["observation_partition"]
    if (
        run["plan"] != leaf_plan(partition, leaf["leaf_id"])
        or leaf not in partition["leaves"]
        or proof["provenance"].get("run_id") != leaf["run_id"]
        or proof["provenance"].get("runtime_digest") != partition["runtime_digest"]
    ):
        raise ValueError("observation native provenance differs from its new leaf identity")
    expected = {(item["native_benchmark_id"], rep) for item in leaf["point_map"] for rep in range(15)}
    if proof["forwards"].keys() != expected:
        raise ValueError("observation native targets omit original five warmup/ten retained repetitions")
    if set(proof["snapshots"]) != set(range(run["key"][2])) or any(
        snapshot.get("max_num_reqs") != RULE["max_num_seqs"]
        or snapshot.get("max_capture_tokens") != RULE["max_capture_tokens"]
        or snapshot.get("resolved_mode") != "FULL_AND_PIECEWISE"
        for snapshot in proof["snapshots"].values()
    ):
        raise ValueError("observation native capture policy differs from the predeclared family rule")
    for (bid, rep), ranks in proof["forwards"].items():
        if ranks.keys() != set(range(run["key"][2])):
            raise ValueError("observation native target omits a worker")
        for rank, row in ranks.items():
            if (
                row.get("runtime_mode") != leaf["observation_family"]
                or row.get("tp_rank") != rank
                or row.get("benchmark_id") != bid
                or row.get("repetition") != rep
            ):
                raise ValueError("actual native dispatch differs from frozen observation family")


def check_control_pair(calibration, control):
    """Never substitute a historical token replay or a differently partitioned control."""
    left, right = calibration.get("observation_leaf"), control.get("observation_leaf")
    if left is None and right is None:
        return
    if left is None or right is None or left["role"] != "calibration" or right["role"] != "control":
        raise ValueError("observation calibration/control contract is missing or mixed")
    ignored = {"leaf_id", "run_id", "role"}
    if {key: value for key, value in left.items() if key not in ignored} != {
        key: value for key, value in right.items() if key not in ignored
    } or left["run_id"] == right["run_id"]:
        raise ValueError("observation control is not the independent paired new leaf")


def freeze_leaf_run(run):
    """Keep every native-reader input while omitting planner-only Python objects."""
    if "observation_leaf" not in run or "children" in run:
        raise ValueError("only a real observation leaf can be frozen as a native run")
    return copy.deepcopy(
        {
            key: run[key]
            for key in (
                "key",
                "plan",
                "cell",
                "points",
                "corpus",
                "spec",
                "role",
                "observation_partition",
                "observation_leaf",
                "original_point_ids",
            )
        }
    )


def origin(run, native_id):
    item = next(item for item in run["observation_leaf"]["point_map"] if item["native_benchmark_id"] == native_id)
    return {
        "observation_leaf_id": run["cell"]["cell_id"],
        "original_child_cell_id": run["observation_leaf"]["original_child_cell_id"],
        "original_child_benchmark_id": item["original_child_benchmark_id"],
        "native_benchmark_id": native_id,
        "original_point_id": item["original_point_id"],
    }


def validate_ownership_rows(ownership):
    """A rehashed sidecar cannot change the original/native point correspondence."""
    from collector.glm53flash_serving_shards import _coordinates
    from collector.glm53flash_vllm_serving_export import KEYS

    leaves = {leaf["leaf_id"]: leaf for leaf in ownership["observation_partition"]["leaves"]}
    seen, covered = set(), set()
    expected = {(cid, item["native_benchmark_id"]) for cid, leaf in leaves.items() for item in leaf["point_map"]}
    for row in ownership["rows"]:
        cid = row.get("observation_leaf_id")
        leaf = leaves.get(cid)
        if leaf is None:
            raise ValueError("observation ownership names an unknown leaf")
        items = [item for item in leaf["point_map"] if item["native_benchmark_id"] == row.get("native_benchmark_id")]
        if len(items) != 1:
            raise ValueError("observation ownership names an unknown native point")
        item = items[0]
        coordinates = tuple(row[key] for key in ("phase", "batch_size", "query_length", "prefix"))
        key = tuple(row[name] for name in KEYS)
        if (
            key in seen
            or row.get("shard_id") != cid
            or row.get("original_child_cell_id") != leaf["original_child_cell_id"]
            or row.get("original_child_benchmark_id") != item["original_child_benchmark_id"]
            or row.get("original_point_id") != item["original_point_id"]
            or coordinates != _coordinates(item["point"], leaf["phase"])
        ):
            raise ValueError("observation ownership changed original/native identity or geometry")
        seen.add(key)
        covered.add((cid, item["native_benchmark_id"]))
    if covered != expected:
        raise ValueError("observation ownership omits an original point")
