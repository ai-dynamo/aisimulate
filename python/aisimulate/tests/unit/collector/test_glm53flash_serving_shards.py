# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY frozen shard contracts, not model or performance qualification."""

import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from collector import glm53flash_serving_shards as shards
from collector import glm53flash_vllm_serving_export as serving
from collector.glm53flash_contract import canonical_json, sha256_json

from .test_glm53flash_vllm_serving_export import fixture as serving_fixture

pytestmark = pytest.mark.unit


def frozen_parent(role="calibration"):
    points = [{"batch_size": 1, "total_kv_read_tokens": n, "total_prefill_tokens": 0} for n in (128, 256)]
    payload = {"schema_version": 3, "prefill": [], "decode": points}
    corpus = ("c" if role == "calibration" else "d") * 64
    cell = {"cell_id": "TEST_ONLY-parent", "workload_kind": "decode"}
    plan = {
        "sha256": "a" * 64,
        "backend": "vllm",
        "model_path": "zai-org/GLM-5.3-Flash",
        "system": "gb300",
        "cells": [cell],
        "options": {
            "dataset_role": role,
            "input_text_sha256": corpus,
            "benchmark_points": {"payload": payload, "sha256": sha256_json(payload)},
        },
        **dict.fromkeys(
            (
                "aic_revision",
                "generator_config_sha256",
                "capability",
                "dtype_profile",
                "topologies",
                "topology_memory_admission",
                "backend_policies",
            ),
            "TEST_ONLY",
        ),
    }
    manifest = {
        "schema_name": "aic_fpm_shard_manifest",
        "schema_version": 1,
        "parent_plan_sha256": plan["sha256"],
        "parent_points_sha256": sha256_json(payload),
        "shards": [],
    }
    runs = []
    for index, point in enumerate(points, 1):
        identity = {
            "parent_cell_id": cell["cell_id"],
            "phase": "decode",
            "point_map": [{"native_benchmark_id": 1, "original_point_id": index, "point": point}],
        }
        sid = sha256_json(identity)[:20]
        child = copy.deepcopy(plan)
        child["cells"][0]["cell_id"] = cid = f"fpm-shard-{sid}"
        child_payload = {"schema_version": 3, "prefill": [], "decode": [point]}
        child["options"]["benchmark_points"] = {"payload": child_payload, "sha256": sha256_json(child_payload)}
        child["sha256"] = sha256_json({"shard": identity, "child_plan": {**child, "sha256": ""}})
        manifest["shards"].append(
            {**identity, "shard_id": sid, "child_cell_id": cid, "child_plan_sha256": child["sha256"]}
        )
        runs.append(
            {
                "key": ("vllm", "fp8", 2, "decode"),
                "plan": child,
                "cell": child["cells"][0],
                "points": [dict(point, benchmark_id=1, point_type="decode")],
                "corpus": corpus,
                "spec": {"ops_execution_mode": "native_serving"},
                "role": role,
                "original_point_ids": {1: index},
            }
        )
    plan["sharding"] = manifest
    return {
        "key": runs[0]["key"],
        "plan": plan,
        "cell": cell,
        "points": [dict(p, benchmark_id=i, point_type="decode") for i, p in enumerate(points, 1)],
        "corpus": corpus,
        "spec": {"ops_execution_mode": "native_serving"},
        "role": role,
        "children": runs,
        "shard_manifest": manifest,
    }


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    parent = frozen_parent()
    children, proofs, evidence = [], {}, {}
    for index, run in enumerate(parent["children"]):
        root = tmp_path / str(index)
        root.mkdir()
        proof = serving_fixture(batch=1, prefix=run["points"][0]["total_kv_read_tokens"])
        rid = f"TEST_ONLY-attempt-{index}"
        for ranks in proof["forwards"].values():
            for row in ranks.values():
                row["request_set"] = rid
                row["request_ids"] = [rid + str(row["repetition"])]
        receipt = {"request_set": rid, "source_plan_sha256": run["plan"]["sha256"], "corpus_sha256": run["corpus"]}
        (root / "serving-calibration-evidence.json").write_text(canonical_json(receipt))
        native = {
            "evidence_root": str(root),
            "runtime_run_id": rid,
            "request_ids": {rid},
            "graph_policy": proof["policy"],
            "backend_version": "0.30.0",
            "values": {1: 100.0},
            "timing_boundary": serving.BOUNDARY,
            **{key: proof[key] for key in ("execution_policy", "_execution_policy")},
        }
        children.append((run, native))
        proofs[root], evidence[root] = proof, receipt
    # Native source/trace/hardware prerequisites have their own tests. These
    # substitutions isolate complete point ownership and use the real named
    # 277-unit reducer; they never claim that authored timings are GPU data.
    monkeypatch.setattr(serving, "read_serving_run", lambda root, run: proofs[root])
    monkeypatch.setattr(serving, "verify_evidence", lambda root, proof: evidence[root])
    return parent, children, proofs


def test_complete_original_union_preserves_every_named_row_and_evidence(campaign, tmp_path):
    parent, children, _ = campaign
    path = tmp_path / serving.BASENAME
    publication = shards.publish_calibration(parent, children, path)
    bound = shards.bind_calibration([path], parent, children)
    assert bound["rows"] == 556
    assert {row["original_point_id"] for row in publication["ownership"]["rows"]} == {1, 2}
    rows = pq.read_table(path).to_pylist()
    assert len({row["operation_name"] for row in rows}) == 278
    assert len({row["evidence_sha256"] for row in rows}) == 2
    assert publication["accuracy_acceptance"] == "NOT_EVALUATED"
    with pytest.raises(ValueError, match="new canonical"):
        shards.publish_calibration(parent, children, path)
    rows[0]["latency"] *= 0.5
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="differs from complete"):
        shards.bind_calibration([path], parent, children)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "map", "parent_point", "child_point", "role", "corpus"])
def test_frozen_union_rejects_omissions_and_replaced_point_identity(defect):
    parent = frozen_parent()
    children = parent["children"]
    if defect == "missing":
        children.pop()
    elif defect == "duplicate":
        children.append(copy.deepcopy(children[0]))
    elif defect == "map":
        children[0]["original_point_ids"][1] = 2
    elif defect == "parent_point":
        parent["points"][0]["total_kv_read_tokens"] += 1
    elif defect == "child_point":
        children[0]["points"][0]["total_kv_read_tokens"] += 1
    elif defect == "role":
        children[0]["role"] = "holdout"
    else:
        children[0]["corpus"] = "e" * 64
    with pytest.raises(ValueError):
        shards.validate_children(parent, children)


@pytest.mark.parametrize("defect", ["policy", "args", "request", "run_id", "point", "missing_unit"])
def test_native_shards_cannot_borrow_another_run_or_silently_fill_gaps(campaign, defect):
    parent, children, proofs = campaign
    native = children[1][1]
    if defect == "policy":
        native["graph_policy"] = {**native["graph_policy"], "config_sha256": "f" * 64}
    elif defect == "args":
        native["_execution_policy"] = {**native["_execution_policy"], "max_num_seqs": 8}
        native["execution_policy"] = {**native["execution_policy"], "sha256": sha256_json(native["_execution_policy"])}
    elif defect == "request":
        native["request_ids"] = children[0][1]["request_ids"]
    elif defect == "run_id":
        native["runtime_run_id"] = children[0][1]["runtime_run_id"]
    else:
        proof = proofs[next(root for root in proofs if str(root) == native["evidence_root"])]
        for ranks in proof["forwards"].values():
            for row in ranks.values():
                if defect == "point":
                    row["prefix_lengths"] = [512]
                else:
                    row["binding"].pop("native_graph_setup")
    with pytest.raises(ValueError):
        shards.calibration_rows(parent, children)


def test_prediction_keeps_real_child_identities_and_original_error_rows(campaign, tmp_path, monkeypatch):
    parent, children, _ = campaign
    path = tmp_path / serving.BASENAME
    shards.publish_calibration(parent, children, path)
    bound = shards.bind_calibration([path], parent, children)
    native = {**children[0][1], "_children": {run["cell"]["cell_id"]: value for run, value in children}}
    del native["runtime_run_id"]
    shards.validate_prediction_binding(native, bound)
    holdout = frozen_parent("holdout")

    def predict(run, *args):
        value = {"error": "TEST_ONLY missing bracket"} if run["original_point_ids"][1] == 2 else {"prediction_ms": 7.0}
        return {"rows": {1: value}, "diagnostics": {"TEST_ONLY": True}}

    monkeypatch.setattr(serving, "predict_homogeneous", predict)
    result = shards.predict_homogeneous(holdout, tmp_path, {}, native, bound)
    assert result["rows"] == {1: {"prediction_ms": 7.0}, 2: {"error": "TEST_ONLY missing bracket"}}
    changed = copy.deepcopy(bound)
    changed["children"].pop()
    with pytest.raises(ValueError, match="complete original"):
        shards.validate_prediction_binding(native, changed)
    evidence = tmp_path / "1/serving-calibration-evidence.json"
    evidence.write_text(json.dumps({"source_plan_sha256": "f" * 64}))
    with pytest.raises(ValueError, match="provenance changed"):
        shards.validate_prediction_binding(native, bound)


def test_parent_native_loader_keeps_actual_shared_policy_and_rejects_changed_child(campaign, tmp_path, monkeypatch):
    from collector import glm53flash_validation as native_reader
    from collector.fpm_forward import glm53flash_validation as acceptance

    parent, children, _ = campaign
    admitted = {run["cell"]["cell_id"]: native for run, native in children}
    monkeypatch.setattr(native_reader, "load_native", lambda run, base: admitted[run["cell"]["cell_id"]])
    result = acceptance._load_native(parent, tmp_path, "ops")
    assert result["graph_policy"] == children[0][1]["graph_policy"]
    assert result["values"] == {1: 100.0, 2: 100.0}
    assert "runtime_run_id" not in result
    assert result["_children"].keys() == admitted.keys()
    children[1][1]["graph_policy"] = {**children[1][1]["graph_policy"], "runtime_digest": "sha256:" + "e" * 64}
    with pytest.raises(ValueError, match="initialized native policy"):
        acceptance._load_native(parent, tmp_path, "ops")


def test_existing_publication_binding_entry_requires_exact_parent_for_serving(campaign, tmp_path):
    from collector import glm53flash_validation as native_reader

    parent, children, _ = campaign
    path = tmp_path / serving.BASENAME
    with pytest.raises(ValueError, match="original complete parent"):
        native_reader.publish_sharded_calibration(children, parent["shard_manifest"], path)
    native_reader.publish_sharded_calibration(children, parent["shard_manifest"], path, parent_run=parent)
    with pytest.raises(ValueError, match="original complete parent"):
        native_reader.bind_sharded_calibration([path], children, parent["shard_manifest"])
    result = native_reader.bind_sharded_calibration([path], children, parent["shard_manifest"], parent_run=parent)
    assert result["rows"] == 556


def test_actual_public_rust_uses_merged_named_rows_without_losing_attempt_provenance(campaign, tmp_path):
    import shutil
    from importlib.resources import files

    from aisimulate_core.sdk.engine import EngineHandle

    parent, children, _ = campaign
    systems = tmp_path / "systems"
    data = systems / "data/gb300/vllm/0.30.0"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    table = data / serving.BASENAME
    shards.publish_calibration(parent, children, table)
    binding = shards.bind_calibration([table], parent, children)
    engine = EngineHandle.compile(
        "zai-org/GLM-5.3-Flash",
        "gb300",
        "vllm",
        backend_version="0.30.0",
        tp_size=2,
        moe_tp_size=2,
        moe_ep_size=1,
        systems_path=str(systems),
        database_mode="SILICON",
        strict_provenance=True,
    )
    rows = pq.read_table(table).to_pylist()
    for prefix in (128, 192, 256):
        # TEST_ONLY fixtures give equal per-unit endpoint timings. Actual Rust
        # still must use all named units/setup and the compatible P bracket.
        expected = sum(row["latency"] for row in rows if row["prefix"] == 128)
        assert engine.predict_decode_latency(1, prefix, 2) == pytest.approx(expected)
        assert engine.last_provenance() is None
    assert len(binding["children"]) == 2
    with pytest.raises(ValueError):
        engine.predict_decode_latency(1, 512, 2)
