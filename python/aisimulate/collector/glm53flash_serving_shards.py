# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve complete frozen point ownership across native serving attempts.

Rows retain their original named unit, policy, rank selection and raw evidence.
No repeated geometry is averaged and no failed child is replaced by a donor.
"""

from __future__ import annotations

import json
from pathlib import Path

from collector.glm53flash_contract import CHECKPOINTS, canonical_json, sha256_json
from collector.glm53flash_jsonl import file_sha256
from collector.glm53flash_shard_contract import validate_point_union


def validate_children(parent, children):
    """Recheck the original parent bytes and every child/map before any merge."""
    if (
        parent["spec"].get("ops_execution_mode") != "native_serving"
        or parent["key"][0] != "vllm"
        or parent["key"][1] not in CHECKPOINTS
        or parent["plan"]["backend"] != parent["key"][0]
        or parent["plan"]["model_path"] != CHECKPOINTS[parent["key"][1]][0]
        or parent["cell"]["workload_kind"] != parent["key"][3]
        or parent["cell"] not in parent["plan"]["cells"]
        or not children
        or any(
            run["spec"].get("ops_execution_mode") != "native_serving"
            or tuple(run["key"]) != tuple(parent["key"])
            or run["role"] != parent["role"]
            or run["corpus"] != parent["corpus"]
            or run["plan"]["cells"] != [run["cell"]]
            or "children" in run
            for run in children
        )
    ):
        raise ValueError("serving shards changed their parent mode/phase/role/corpus")
    by_id = {run["cell"]["cell_id"]: run for run in children}
    if len(by_id) != len(children):
        raise ValueError("duplicate serving shard child")
    manifest = parent["shard_manifest"]
    parent_id = parent["cell"]["cell_id"]
    validate_point_union(
        parent["plan"], manifest, {cid: run["plan"] for cid, run in by_id.items()}, parent_cell_id=parent_id
    )
    expected_parent = parent["plan"]["options"]["benchmark_points"]["payload"][parent["key"][3]]
    expected_parent = [
        dict(
            point,
            benchmark_id=i,
            point_type=parent["key"][3],
            total_prefill_tokens=point.get("total_prefill_tokens", 0),
        )
        for i, point in enumerate(expected_parent, 1)
    ]
    if parent["points"] != expected_parent:
        raise ValueError("serving parent requested points differ from its original plan")
    result = {}
    for shard in manifest["shards"]:
        if shard["parent_cell_id"] != parent_id:
            continue
        child = by_id[shard["child_cell_id"]]
        mapping = {item["native_benchmark_id"]: item["original_point_id"] for item in shard["point_map"]}
        points = [
            dict(
                item["point"],
                benchmark_id=item["native_benchmark_id"],
                point_type=shard["phase"],
                total_prefill_tokens=item["point"].get("total_prefill_tokens", 0),
            )
            for item in shard["point_map"]
        ]
        if child.get("original_point_ids") != mapping or child["points"] != points:
            raise ValueError("serving child points or original IDs differ from the frozen map")
        result[shard["child_cell_id"]] = shard
    return result


def same_native_policy(left, right):
    from collector.glm53flash_vllm_graph_export import same_execution_policy

    same_execution_policy(left, right)
    policy = left.get("graph_policy")
    if not isinstance(policy, dict) or policy.get("schema_version") != 3 or right.get("graph_policy") != policy:
        raise ValueError("serving shards changed their complete initialized native policy")


def _coordinates(point, phase):
    batch, past = point["batch_size"], point["total_kv_read_tokens"]
    new = point["total_prefill_tokens"] if phase == "prefill" else batch
    if (
        any(type(value) is not int for value in (batch, past, new))
        or not 1 <= batch <= 32
        or past < 0
        or new < 1
        or past % batch
        or new % batch
        or (past + new) // batch > 131072
    ):
        raise ValueError("serving frozen point is not bounded homogeneous B/Q/P")
    return ("context" if phase == "prefill" else "generation", batch, new // batch, past // batch)


def calibration_rows(parent, children, *, lookup_contract=None):
    """Reproduce every child from its native source, control and trace evidence."""
    from collector.glm53flash_vllm_serving_export import (
        KEYS,
        aggregate_serving,
        analysis_rows,
        read_serving_run,
        verify_evidence,
    )

    if parent["role"] != "calibration":
        raise ValueError("serving shard publication requires calibration, never holdout")
    declared = validate_children(parent, [run for run, _ in children])
    baseline = children[0][1]
    rows, owners, receipts, ids, requests = [], [], [], set(), set()
    physical = set()
    for run, native in children:
        same_native_policy(baseline, native)
        cid = run["cell"]["cell_id"]
        shard = declared[cid]
        if native["runtime_run_id"] in ids or requests & native["request_ids"]:
            raise ValueError("serving shards reused original native runs or requests")
        ids.add(native["runtime_run_id"])
        requests.update(native["request_ids"])
        root = Path(native["evidence_root"])
        proof = read_serving_run(root, run)
        evidence = verify_evidence(root, proof)
        if (
            proof["policy"] != native["graph_policy"]
            or evidence["request_set"] != native["runtime_run_id"]
            or evidence["source_plan_sha256"] != run["plan"]["sha256"]
            or evidence["corpus_sha256"] != run["corpus"]
        ):
            raise ValueError("serving shard evidence differs from its admitted native run")
        evidence_sha = file_sha256(root / "serving-calibration-evidence.json")
        actual, _ = aggregate_serving(proof, evidence_sha256=evidence_sha)
        actual = analysis_rows(proof, actual, lookup_contract)
        coordinate_owners = {_coordinates(item["point"], shard["phase"]): item for item in shard["point_map"]}
        if len(coordinate_owners) != len(shard["point_map"]):
            raise ValueError("serving frozen shard repeats physical point geometry")
        observed = set()
        for row in actual:
            coordinates = tuple(row[key] for key in ("phase", "batch_size", "query_length", "prefix"))
            key = tuple(row[name] for name in KEYS)
            if coordinates not in coordinate_owners or key in physical:
                raise ValueError("serving physical row is duplicated or outside frozen point ownership")
            physical.add(key)
            observed.add(coordinates)
            owner = coordinate_owners[coordinates]
            rows.append(row)
            owners.append(
                {
                    **{name: row[name] for name in KEYS},
                    "shard_id": shard["shard_id"],
                    "original_point_id": owner["original_point_id"],
                    "native_benchmark_id": owner["native_benchmark_id"],
                }
            )
        if observed != coordinate_owners.keys():
            raise ValueError("serving shard omits one or more original frozen points")
        receipts.append(
            {
                "child_cell_id": cid,
                "shard_id": shard["shard_id"],
                "source_plan_sha256": run["plan"]["sha256"],
                "evidence_sha256": evidence_sha,
                "native_runtime_run_id": native["runtime_run_id"],
                "policy_evidence_sha256": proof["policy_evidence_sha256"],
                "rows": len(actual),
            }
        )
    ownership = {
        "schema": "glm53flash_serving_shard_ownership_v1",
        "parent_cell_id": parent["cell"]["cell_id"],
        "source_plan_sha256": parent["plan"]["sha256"],
        "shard_manifest_sha256": sha256_json(parent["shard_manifest"]),
        "graph_policy_sha256": sha256_json(baseline["graph_policy"]),
        "corpus_sha256": parent["corpus"],
        "rows": sorted(owners, key=canonical_json),
        "children": sorted(receipts, key=canonical_json),
        **({"lookup_contract": lookup_contract} if lookup_contract else {}),
    }
    return sorted(rows, key=canonical_json), ownership


def publish_calibration(parent, children, destination, *, lookup_contract=None):
    """Write a new complete table; preserve per-attempt provenance verbatim."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from collector.glm53flash_vllm_serving_export import BASENAME

    destination = Path(destination)
    sidecar = destination.with_suffix(".evidence.json")
    if destination.name != BASENAME or destination.exists() or sidecar.exists():
        raise ValueError("serving shard publication requires new canonical table and evidence paths")
    rows, ownership = calibration_rows(parent, children, lookup_contract=lookup_contract)
    pq.write_table(pa.Table.from_pylist(rows), destination)
    receipt = {"ownership": ownership, "table_sha256": file_sha256(destination), "accuracy_acceptance": "NOT_EVALUATED"}
    with sidecar.open("x") as stream:
        stream.write(canonical_json(receipt) + "\n")
    return receipt


def bind_calibration(paths, parent, children):
    """Compare the complete final table to all original child observations."""
    import pyarrow.parquet as pq

    from collector.glm53flash_vllm_serving_export import BASENAME, table_lookup_contract

    backend, fmt, tp, phase = parent["key"]
    selected, tables = [], []
    for path in paths:
        if path.name != BASENAME:
            continue
        tables.append({"path": str(path), "sha256": file_sha256(path)})
        for row in pq.read_table(path).to_pylist():
            policy = json.loads(row["graph_policy"])
            if (policy["backend"], policy["checkpoint_format"], policy["tp_size"]) == (backend, fmt, tp):
                if policy.get("schema_version") != 3:
                    raise ValueError("serving shard table mixes a legacy policy for the same deployment")
                if row["phase"] == ("context" if phase == "prefill" else "generation"):
                    selected.append(row)
    lookup_contract = table_lookup_contract(selected)
    expected, ownership = calibration_rows(parent, children, lookup_contract=lookup_contract)
    if sorted(selected, key=canonical_json) != expected:
        raise ValueError("serving shard table differs from complete original calibration observations")
    return {
        "rows": len(selected),
        "tables": tables,
        "graph_policy_sha256": ownership["graph_policy_sha256"],
        "calibration_group_sha256": sha256_json(ownership),
        "ownership": ownership,
        "children": ownership["children"],
        "source_plan_sha256": parent["plan"]["sha256"],
        **({"lookup_contract": lookup_contract} if lookup_contract else {}),
    }


def validate_prediction_binding(native, binding):
    """Keep actual run identities for a group; never invent one aggregate run."""
    from collector.glm53flash_vllm_serving_export import _hash

    if "_children" not in native:
        if binding.get("native_runtime_run_id") != native["runtime_run_id"] or not _hash(
            binding.get("evidence_sha256")
        ):
            raise ValueError("serving prediction lacks its original calibration run binding")
        return
    ownership = binding.get("ownership", {})
    receipts = binding.get("children", [])
    by_id = {row["child_cell_id"]: row for row in receipts}
    if (
        not receipts
        or len(by_id) != len(receipts)
        or by_id.keys() != native["_children"].keys()
        or ownership.get("children") != sorted(receipts, key=canonical_json)
        or ownership.get("schema") != "glm53flash_serving_shard_ownership_v1"
        or ownership.get("graph_policy_sha256") != sha256_json(native["graph_policy"])
        or ownership.get("source_plan_sha256") != binding.get("source_plan_sha256")
        or binding.get("calibration_group_sha256") != sha256_json(ownership)
    ):
        raise ValueError("serving prediction lacks its complete original calibration shard group")
    for cid, child in native["_children"].items():
        same_native_policy(native, child)
        receipt = by_id[cid]
        evidence = Path(child["evidence_root"]) / "serving-calibration-evidence.json"
        if (
            receipt["native_runtime_run_id"] != child["runtime_run_id"]
            or receipt["evidence_sha256"] != file_sha256(evidence)
            or json.loads(evidence.read_bytes())["source_plan_sha256"] != receipt["source_plan_sha256"]
        ):
            raise ValueError("serving calibration shard provenance changed before prediction")


def predict_homogeneous(run, base, config, calibration_native, calibration_binding):
    """Predict each original holdout point once, preserving missing/error rows."""
    from collector.glm53flash_vllm_serving_export import predict_homogeneous as predict_leaf

    validate_prediction_binding(calibration_native, calibration_binding)
    if "children" not in run:
        return predict_leaf(run, base, config, calibration_native, calibration_binding)
    validate_children(run, run["children"])
    if run["role"] != "holdout":
        raise ValueError("serving shard prediction requires independent holdout")
    rows, diagnostics, prediction_evidence, evidence_origins = {}, [], {}, {}
    for child in run["children"]:
        result = predict_leaf(child, base, config, calibration_native, calibration_binding)
        mapping = child["original_point_ids"]
        if result["rows"].keys() != mapping.keys():
            raise ValueError("serving prediction omitted a frozen child point or failure")
        mapped = {mapping[key]: value for key, value in result["rows"].items()}
        if len(mapped) != len(mapping) or rows.keys() & mapped.keys():
            raise ValueError("serving predictions overlap original holdout point identities")
        rows.update(mapped)
        audits = result.get("prediction_evidence", {})
        successful = {key for key, row in result["rows"].items() if "prediction_ms" in row and "error" not in row}
        if (
            not isinstance(audits, dict)
            or (calibration_binding.get("lookup_contract") and audits.keys() != successful)
            or (not calibration_binding.get("lookup_contract") and audits)
        ):
            raise ValueError("serving endpoint evidence must cover exactly the successful opt-in predictions")
        for native_id, evidence in audits.items():
            if type(native_id) is not int or native_id not in mapping or native_id not in successful:
                raise ValueError("serving endpoint evidence lacks an original successful holdout point")
            original_id = mapping[native_id]
            if original_id in prediction_evidence:
                raise ValueError("serving endpoint evidence overlaps original holdout point identities")
            prediction_evidence[original_id] = evidence
            evidence_origins[original_id] = {
                "child_cell_id": child["cell"]["cell_id"],
                "native_benchmark_id": native_id,
                "original_point_id": original_id,
            }
        diagnostics.append({"child_cell_id": child["cell"]["cell_id"], **result["diagnostics"]})
    if rows.keys() != {point["benchmark_id"] for point in run["points"]}:
        raise ValueError("serving predictions omit original frozen holdout coverage")
    return {
        "rows": rows,
        "calibration_binding": calibration_binding,
        "diagnostics": {"shards": diagnostics},
        **(
            {"prediction_evidence": prediction_evidence, "prediction_evidence_origins": evidence_origins}
            if calibration_binding.get("lookup_contract")
            else {}
        ),
    }
