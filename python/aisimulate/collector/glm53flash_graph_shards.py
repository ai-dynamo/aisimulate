# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve complete frozen point ownership across native SGLang FULL decode attempts.

Rows retain their original named unit, policy, rank selection and raw evidence.
No repeated geometry is averaged and no failed child is replaced by a donor.
The complete capture policy must match across calibration children, including
its source/evidence identities. Different policies are rejected, not normalized.
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
        parent["spec"].get("ops_execution_mode") != "native_full_graph"
        or parent["key"][0] != "sglang"
        or parent["key"][3] != "decode"
        or parent["role"] not in ("calibration", "holdout")
        or parent["corpus"] != parent["plan"]["options"]["input_text_sha256"]
        or parent["role"] != parent["plan"]["options"].get("dataset_role", "calibration")
        or parent["key"][1] not in CHECKPOINTS
        or parent["plan"]["backend"] != parent["key"][0]
        or parent["plan"]["model_path"] != CHECKPOINTS[parent["key"][1]][0]
        or parent["cell"]["workload_kind"] != parent["key"][3]
        or parent["cell"] not in parent["plan"]["cells"]
        or not children
        or any(
            run["spec"].get("ops_execution_mode") != "native_full_graph"
            or tuple(run["key"]) != tuple(parent["key"])
            or run["role"] != parent["role"]
            or run["corpus"] != parent["corpus"]
            or run["plan"]["cells"] != [run["cell"]]
            or "children" in run
            for run in children
        )
    ):
        raise ValueError("graph shards changed their parent mode/phase/role/corpus")
    by_id = {run["cell"]["cell_id"]: run for run in children}
    if len(by_id) != len(children):
        raise ValueError("duplicate graph shard child")
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
        raise ValueError("graph parent requested points differ from its original plan")
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
            raise ValueError("graph child points or original IDs differ from the frozen map")
        result[shard["child_cell_id"]] = shard
    return result


def same_native_policy(left, right):
    from collector.fpm_forward.glm53flash_validation import _same_sglang_policy

    _same_sglang_policy(left, right, "native graph shards")
    policy = left.get("graph_policy")
    if (
        not isinstance(policy, dict)
        or (policy.get("schema_version") != 1 and set(policy) != {"native_snapshot", "provenance"})
        or right.get("graph_policy") != policy
    ):
        raise ValueError("graph shards changed their complete native execution/capture policy")


def _coordinates(point, phase):
    batch, past = point["batch_size"], point["total_kv_read_tokens"]
    if (
        phase != "decode"
        or any(type(value) is not int for value in (batch, past))
        or not 1 <= batch <= 32
        or past < 0
        or past % batch
        or past // batch + 1 > 131072
        or point.get("total_prefill_tokens", 0) != 0
    ):
        raise ValueError("graph frozen point is not bounded homogeneous decode B/P")
    return batch, past // batch


def _control_identity(root):
    """Keep each real control's original run, requests and file hashes."""
    from collector.glm53flash_graph_export import _local
    from collector.glm53flash_validation import _load_native

    path = _local(root, "graph-profile-control.json")
    receipt = json.loads(path.read_bytes())
    control_root = Path(receipt["evidence_root"]).resolve()
    run = receipt["frozen_run"]
    if run["role"] != "control" or control_root == root.resolve():
        raise ValueError("graph shard requires its independent original control")
    native = _load_native(run, control_root, calibration_evidence=False)
    return {
        "evidence_root": str(control_root),
        "native_runtime_run_id": native["runtime_run_id"],
        "request_ids": sorted(native["request_ids"]),
        "profile_control_sha256": file_sha256(path),
        "receipts": native["receipts"],
    }


def _claim_identity(roots, runs, requests, *, root, run_id, request_ids):
    root, request_ids = Path(root).resolve(), set(request_ids)
    if root in roots or run_id in runs or requests & request_ids:
        raise ValueError("graph shards reused original native calibration/control roots, runs or requests")
    roots.add(root)
    runs.add(run_id)
    requests.update(request_ids)


def calibration_rows(parent, children, *, lookup_contract=None):
    """Reproduce every child from its native source, control and trace evidence."""
    from collector.glm53flash_graph_export import (
        KEYS,
        NAMED_CONTRACT,
        aggregate_graph,
        read_graph_run,
        verify_evidence,
    )

    if lookup_contract != NAMED_CONTRACT:
        raise ValueError("graph shard publication requires the explicit named operation contract")
    if parent["role"] != "calibration":
        raise ValueError("graph shard publication requires calibration, never holdout")
    declared = validate_children(parent, [run for run, _ in children])
    baseline = children[0][1]
    rows, owners, receipts, roots, ids, requests = [], [], [], set(), set(), set()
    physical = set()
    for run, native in children:
        same_native_policy(baseline, native)
        cid = run["cell"]["cell_id"]
        shard = declared[cid]
        root = Path(native["evidence_root"])
        _claim_identity(
            roots, ids, requests, root=root, run_id=native["runtime_run_id"], request_ids=native["request_ids"]
        )
        proof = read_graph_run(root, run)
        same_native_policy(native, {**proof, "graph_policy": proof["policy"]})
        evidence = verify_evidence(root, proof)
        actual_requests = {
            request for ranks in proof["forwards"].values() for row in ranks.values() for request in row["request_ids"]
        }
        if (
            proof["policy"] != native["graph_policy"]
            or actual_requests != native["request_ids"]
            or evidence["request_set"] != native["runtime_run_id"]
            or evidence["source_plan_sha256"] != run["plan"]["sha256"]
            or evidence["corpus_sha256"] != run["corpus"]
        ):
            raise ValueError("graph shard evidence differs from its admitted native run")
        control = _control_identity(root)
        _claim_identity(
            roots,
            ids,
            requests,
            root=control["evidence_root"],
            run_id=control["native_runtime_run_id"],
            request_ids=control["request_ids"],
        )
        evidence_sha = file_sha256(root / "graph-calibration-evidence.json")
        actual, _ = aggregate_graph(proof, evidence_sha256=evidence_sha, lookup_contract=lookup_contract)
        coordinate_owners = {_coordinates(item["point"], shard["phase"]): item for item in shard["point_map"]}
        if len(coordinate_owners) != len(shard["point_map"]):
            raise ValueError("graph frozen shard repeats physical point geometry")
        observed = set()
        for row in actual:
            coordinates = row["batch_size"], row["prefix"]
            key = tuple(row[name] for name in (*KEYS, "operation_name"))
            if coordinates not in coordinate_owners or key in physical:
                raise ValueError("graph physical row is duplicated or outside frozen point ownership")
            physical.add(key)
            observed.add(coordinates)
            owner = coordinate_owners[coordinates]
            rows.append(row)
            owners.append(
                {
                    **{name: row[name] for name in (*KEYS, "operation_name")},
                    "shard_id": shard["shard_id"],
                    "original_point_id": owner["original_point_id"],
                    "native_benchmark_id": owner["native_benchmark_id"],
                }
            )
        if observed != coordinate_owners.keys():
            raise ValueError("graph shard omits one or more original frozen points")
        receipts.append(
            {
                "child_cell_id": cid,
                "shard_id": shard["shard_id"],
                "source_plan_sha256": run["plan"]["sha256"],
                "evidence_sha256": evidence_sha,
                "native_runtime_run_id": native["runtime_run_id"],
                "graph_policy_sha256": sha256_json(proof["policy"]),
                "rows": len(actual),
                "control": control,
            }
        )
    ownership = {
        "schema": "glm53flash_sglang_graph_shard_ownership_v1",
        "parent_cell_id": parent["cell"]["cell_id"],
        "source_plan_sha256": parent["plan"]["sha256"],
        "shard_manifest_sha256": sha256_json(parent["shard_manifest"]),
        "graph_policy_sha256": sha256_json(baseline["graph_policy"]),
        "corpus_sha256": parent["corpus"],
        "rows": sorted(owners, key=canonical_json),
        "children": sorted(receipts, key=canonical_json),
    }
    return sorted(rows, key=canonical_json), ownership


def publish_calibration(parent, children, destination, *, lookup_contract=None):
    """Write a new complete table; preserve per-attempt provenance verbatim."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from collector.glm53flash_graph_export import BASENAME

    destination = Path(destination)
    sidecar = destination.with_suffix(".evidence.json")
    if destination.name != BASENAME or destination.exists() or sidecar.exists():
        raise ValueError("graph shard publication requires new canonical table and evidence paths")
    rows, ownership = calibration_rows(parent, children, lookup_contract=lookup_contract)
    pq.write_table(pa.Table.from_pylist(rows), destination)
    receipt = {
        "ownership": ownership,
        "table_sha256": file_sha256(destination),
        "accuracy_acceptance": "NOT_EVALUATED",
        **({"lookup_contract": lookup_contract} if lookup_contract else {}),
    }
    with sidecar.open("x") as stream:
        stream.write(canonical_json(receipt) + "\n")
    return receipt


def bind_calibration(paths, parent, children):
    """Compare the complete final table to all original child observations."""
    import pyarrow.parquet as pq

    from collector.glm53flash_graph_export import BASENAME

    backend, fmt, tp = parent["key"][:3]
    selected, tables = [], []
    for path in paths:
        if path.name != BASENAME:
            continue
        tables.append({"path": str(path), "sha256": file_sha256(path)})
        for row in pq.read_table(path).to_pylist():
            policy = json.loads(row["graph_policy"])
            if (policy["backend"], policy["checkpoint_format"], policy["tp_size"]) == (backend, fmt, tp):
                if policy.get("schema_version") != 1:
                    raise ValueError("graph shard table mixes a legacy policy for the same deployment")
                selected.append(row)
    from collector.glm53flash_graph_export import NAMED_CONTRACT

    if not selected or any(row.get("graph_lookup_contract") != NAMED_CONTRACT for row in selected):
        raise ValueError("graph shards require complete named operation rows")
    lookup_contract = NAMED_CONTRACT
    expected, ownership = calibration_rows(parent, children, lookup_contract=lookup_contract)
    if sorted(selected, key=canonical_json) != expected:
        raise ValueError("graph shard table differs from complete original calibration observations")
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
    if "_children" not in native:
        if binding.get("native_runtime_run_id") != native["runtime_run_id"] or not binding.get("evidence_sha256"):
            raise ValueError("graph prediction lacks its original calibration run binding")
        evidence = Path(native["evidence_root"]) / "graph-calibration-evidence.json"
        if (
            binding["evidence_sha256"] != file_sha256(evidence)
            or binding.get("source_plan_sha256") != json.loads(evidence.read_bytes())["source_plan_sha256"]
        ):
            raise ValueError("native graph prediction calibration binding evidence changed")
        return
    ownership = binding.get("ownership", {})
    receipts = binding.get("children", [])
    by_id = {row["child_cell_id"]: row for row in receipts}
    if (
        not receipts
        or len(by_id) != len(receipts)
        or by_id.keys() != native["_children"].keys()
        or ownership.get("children") != sorted(receipts, key=canonical_json)
        or ownership.get("schema") != "glm53flash_sglang_graph_shard_ownership_v1"
        or ownership.get("graph_policy_sha256") != sha256_json(native["graph_policy"])
        or ownership.get("source_plan_sha256") != binding.get("source_plan_sha256")
        or binding.get("calibration_group_sha256") != sha256_json(ownership)
    ):
        raise ValueError("graph prediction lacks its complete original calibration shard group")
    for cid, child in native["_children"].items():
        same_native_policy(native, child)
        receipt = by_id[cid]
        evidence = Path(child["evidence_root"]) / "graph-calibration-evidence.json"
        if (
            receipt["native_runtime_run_id"] != child["runtime_run_id"]
            or receipt["evidence_sha256"] != file_sha256(evidence)
            or json.loads(evidence.read_bytes())["source_plan_sha256"] != receipt["source_plan_sha256"]
            or receipt.get("control") != _control_identity(Path(child["evidence_root"]))
        ):
            raise ValueError("graph calibration shard provenance changed before prediction")


def predict_homogeneous(run, base, config, calibration_native, calibration_binding):
    """Predict each original holdout point once, preserving missing/error rows."""
    from collector.glm53flash_graph_export import predict_homogeneous as predict_leaf

    validate_prediction_binding(calibration_native, calibration_binding)
    if "children" not in run:
        return predict_leaf(run, base, config, calibration_native, calibration_binding=calibration_binding)
    validate_children(run, run["children"])
    if run["role"] != "holdout":
        raise ValueError("graph shard prediction requires independent holdout")
    # Recheck the whole holdout's independent native attempt identities before
    # calling the existing leaf predictor. No aggregate native run is created.
    from collector.fpm_forward.glm53flash_validation import _load_native

    holdout = _load_native(run, base, "ops")
    from collector.glm53flash_graph_export import _same_execution_policy

    _same_execution_policy(calibration_native, holdout, "graph calibration/holdout shards")
    if calibration_native["request_ids"] & holdout["request_ids"]:
        raise ValueError("graph holdout reused calibration request identities")
    rows, diagnostics, prediction_evidence, prediction_evidence_origins = {}, [], {}, {}
    for child in run["children"]:
        result = predict_leaf(child, base, config, calibration_native, calibration_binding=calibration_binding)
        mapping = child["original_point_ids"]
        if result["rows"].keys() != mapping.keys():
            raise ValueError("graph prediction omitted a frozen child point or failure")
        mapped = {mapping[key]: value for key, value in result["rows"].items()}
        if rows.keys() & mapped.keys():
            raise ValueError("graph predictions overlap original holdout point identities")
        rows.update(mapped)
        audits = result.get("prediction_evidence", {})
        successful = {key for key, row in result["rows"].items() if "prediction_ms" in row and "error" not in row}
        if calibration_binding.get("lookup_contract") and (
            not isinstance(audits, dict) or any(type(key) is not int for key in audits) or audits.keys() != successful
        ):
            raise ValueError("graph opt-in prediction requires exactly one audit per successful original point")
        for native_id, evidence in audits.items():
            if native_id not in mapping or native_id not in successful or mapping[native_id] in prediction_evidence:
                raise ValueError("graph endpoint evidence lacks an original successful holdout point")
            original_id = mapping[native_id]
            prediction_evidence[original_id] = evidence
            prediction_evidence_origins[original_id] = {
                "child_cell_id": child["cell"]["cell_id"],
                "native_benchmark_id": native_id,
                "original_point_id": original_id,
            }
        diagnostics.append({"child_cell_id": child["cell"]["cell_id"], **result["diagnostics"]})
    if rows.keys() != {point["benchmark_id"] for point in run["points"]}:
        raise ValueError("graph predictions omit original frozen holdout coverage")
    return {
        "rows": rows,
        "calibration_binding": calibration_binding,
        "diagnostics": {"shards": diagnostics},
        **(
            {"prediction_evidence": prediction_evidence, "prediction_evidence_origins": prediction_evidence_origins}
            if calibration_binding.get("lookup_contract")
            else {}
        ),
    }
