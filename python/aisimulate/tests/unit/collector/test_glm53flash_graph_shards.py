# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY complete named FULL decode shards; real reducer/Parquet/consumer."""

import copy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from collector import glm53flash_graph_export as export
from collector import glm53flash_graph_shards as shards
from collector.fpm_forward.glm53flash_validation import _sglang_execution_policy
from collector.glm53flash_contract import canonical_json, sha256_json

from ..sdk.test_glm53flash_graph_named_consumer import authored_proof

pytestmark = pytest.mark.unit


def frozen_parent(role="calibration"):
    points = [{"batch_size": 1, "total_kv_read_tokens": n, "total_prefill_tokens": 0} for n in (128, 136)]
    payload = {"schema_version": 3, "decode": points, "prefill": []}
    corpus = ("c" if role == "calibration" else "d") * 64
    cell = {"cell_id": "TEST_ONLY-parent", "workload_kind": "decode"}
    plan = {
        "sha256": "a" * 64,
        "backend": "sglang",
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
        child_payload = {"schema_version": 3, "decode": [point], "prefill": []}
        child["options"]["benchmark_points"] = {"payload": child_payload, "sha256": sha256_json(child_payload)}
        child["sha256"] = sha256_json({"shard": identity, "child_plan": {**child, "sha256": ""}})
        manifest["shards"].append(
            {**identity, "shard_id": sid, "child_cell_id": cid, "child_plan_sha256": child["sha256"]}
        )
        runs.append(
            {
                "key": ("sglang", "fp8", 2, "decode"),
                "plan": child,
                "cell": child["cells"][0],
                "points": [dict(point, benchmark_id=1, point_type="decode")],
                "corpus": corpus,
                "spec": {"ops_execution_mode": "native_full_graph"},
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
        "spec": {"ops_execution_mode": "native_full_graph"},
        "role": role,
        "children": runs,
        "shard_manifest": manifest,
    }


def proof_fixture(run, index):
    proof = authored_proof()
    forwards = {}
    for (point, repetition), ranks in proof["forwards"].items():
        if point != index + 1:
            continue
        for row in ranks.values():
            row["request_ids"] = [f"TEST_ONLY-cal-{index}-{repetition}"]
        forwards[1, repetition] = ranks
    proof["forwards"] = forwards
    proof.update(
        _sglang_execution_policy(
            {
                "cuda_graph_config": {"prefill": {"backend": "disabled"}, "decode": {"backend": "full"}},
                "mem_fraction_static": 0.82,
            }
        )
    )
    return proof


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    from collector import glm53flash_validation as reader

    parent = frozen_parent()
    children, proofs, evidence, native_by_root = [], {}, {}, {}
    for index, run in enumerate(parent["children"]):
        root, control_root = tmp_path / f"cal-{index}", tmp_path / f"control-{index}"
        root.mkdir()
        control_root.mkdir()
        proof = proof_fixture(run, index)
        rid = f"TEST_ONLY-cal-{index}"
        receipt = {"request_set": rid, "source_plan_sha256": run["plan"]["sha256"], "corpus_sha256": run["corpus"]}
        (root / "graph-calibration-evidence.json").write_text(canonical_json(receipt))
        native = {
            "evidence_root": str(root),
            "runtime_run_id": rid,
            "request_ids": {row[0]["request_ids"][0] for row in proof["forwards"].values()},
            "graph_policy": proof["policy"],
            "backend_version": "0.5.20",
            "values": {1: 0.5},
            "timing_boundary": export.BOUNDARY,
            **{key: proof[key] for key in ("execution_policy", "_execution_policy")},
        }
        control_run = {**run, "role": "control"}
        control = {
            **copy.deepcopy(native),
            "evidence_root": str(control_root),
            "runtime_run_id": f"TEST_ONLY-control-{index}",
            "request_ids": {f"TEST_ONLY-control-{index}-{r}" for r in range(15)},
            "receipts": [{"path": "TEST_ONLY-original.json", "sha256": "d" * 64}],
        }
        (root / "graph-profile-control.json").write_text(
            canonical_json(
                {
                    "evidence_root": str(control_root),
                    "frozen_run": control_run,
                }
            )
        )
        native_by_root[root], native_by_root[control_root] = native, control
        children.append((run, native))
        proofs[root], evidence[root] = proof, receipt
    # Source/trace/whole-forward/control rederivation is covered by the original
    # graph reader tests. These substitutions isolate ownership using the
    # actual complete 366+1 named reducer and public Rust consumer.
    monkeypatch.setattr(export, "read_graph_run", lambda root, run: proofs[Path(root)])
    monkeypatch.setattr(export, "verify_evidence", lambda root, proof: evidence[Path(root)])
    monkeypatch.setattr(reader, "_load_native", lambda run, root, **kw: native_by_root[Path(root)])
    return parent, children, proofs, native_by_root


def test_complete_publication_preserves_all_units_and_independent_attempts(campaign, tmp_path):
    parent, children, _, _ = campaign
    path = tmp_path / export.BASENAME
    publication = shards.publish_calibration(parent, children, path, lookup_contract=export.NAMED_CONTRACT)
    bound = shards.bind_calibration([path], parent, children)
    rows = pq.read_table(path).to_pylist()
    assert bound["rows"] == 734
    assert len({row["operation_name"] for row in rows}) == 367
    assert len({row["evidence_sha256"] for row in rows}) == 2
    assert {row["original_point_id"] for row in publication["ownership"]["rows"]} == {1, 2}
    assert len({child["control"]["native_runtime_run_id"] for child in bound["children"]}) == 2
    assert publication["accuracy_acceptance"] == "NOT_EVALUATED"
    with pytest.raises(ValueError, match="new canonical"):
        shards.publish_calibration(parent, children, path, lookup_contract=export.NAMED_CONTRACT)
    rows[0]["latency"] *= 0.5
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="differs from complete"):
        shards.bind_calibration([path], parent, children)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "map", "parent_point", "child_point", "role", "corpus", "phase", "mode"]
)
def test_frozen_union_rejects_omitted_or_replaced_original_points(defect):
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
    elif defect == "corpus":
        children[0]["corpus"] = "e" * 64
    elif defect == "phase":
        parent["key"] = ("sglang", "fp8", 2, "prefill")
    else:
        children[0]["spec"]["ops_execution_mode"] = "eager"
    with pytest.raises(ValueError):
        shards.validate_children(parent, children)


@pytest.mark.parametrize("field,value", [("corpus", "e" * 64), ("role", "holdout")])
def test_consistent_run_headers_cannot_relabel_the_original_plan(field, value):
    parent = frozen_parent()
    for run in [parent, *parent["children"]]:
        run[field] = value
    with pytest.raises(ValueError, match="parent mode/phase/role/corpus"):
        shards.validate_children(parent, parent["children"])


@pytest.mark.parametrize(
    "defect",
    [
        "policy",
        "args",
        "request",
        "run_id",
        "root",
        "point",
        "missing_unit",
        "control_run",
        "control_request",
        "cross_control_request",
    ],
)
def test_calibration_or_control_shards_cannot_borrow_other_attempts(campaign, defect):
    parent, children, proofs, natives = campaign
    native = children[1][1]
    if defect == "policy":
        native["graph_policy"] = {**native["graph_policy"], "config_sha256": "f" * 64}
    elif defect == "args":
        native.update(_sglang_execution_policy({**native["_execution_policy"], "mem_fraction_static": 0.7}))
    elif defect == "request":
        native["request_ids"] = children[0][1]["request_ids"]
    elif defect == "run_id":
        native["runtime_run_id"] = children[0][1]["runtime_run_id"]
    elif defect == "root":
        native["evidence_root"] = children[0][1]["evidence_root"]
    elif defect.startswith("control") or defect == "cross_control_request":
        controls = [n for path, n in natives.items() if path.name.startswith("control")]
        if defect == "control_run":
            controls[1]["runtime_run_id"] = controls[0]["runtime_run_id"]
        elif defect == "control_request":
            controls[1]["request_ids"] = controls[0]["request_ids"]
        else:
            controls[1]["request_ids"] = children[0][1]["request_ids"]
    else:
        for ranks in proofs[Path(native["evidence_root"])]["forwards"].values():
            for row in ranks.values():
                if defect == "point":
                    row["prefix_lengths"] = [512]
                else:
                    row["binding"].pop("native_graph_setup")
    with pytest.raises(ValueError):
        shards.calibration_rows(parent, children, lookup_contract=export.NAMED_CONTRACT)


def test_parent_loader_preserves_policy_and_does_not_invent_aggregate_run(campaign, tmp_path, monkeypatch):
    from collector import glm53flash_validation as reader
    from collector.fpm_forward import glm53flash_validation as acceptance

    parent, children, _, _ = campaign
    admitted = {run["cell"]["cell_id"]: native for run, native in children}
    monkeypatch.setattr(reader, "load_native", lambda run, base: admitted[run["cell"]["cell_id"]])
    native = acceptance._load_native(parent, tmp_path, "ops")
    assert native["graph_policy"] == children[0][1]["graph_policy"]
    assert native["values"] == {1: 0.5, 2: 0.5}
    assert "runtime_run_id" not in native and "evidence_root" not in native
    path = tmp_path / export.BASENAME
    shards.publish_calibration(parent, children, path, lookup_contract=export.NAMED_CONTRACT)
    binding = shards.bind_calibration([path], parent, children)
    shards.validate_prediction_binding(native, binding)
    changed = copy.deepcopy(binding)
    changed["children"].pop()
    with pytest.raises(ValueError, match="complete original"):
        shards.validate_prediction_binding(native, changed)
    children[1][1]["graph_policy"] = {**children[1][1]["graph_policy"], "runtime_digest": "sha256:" + "e" * 64}
    with pytest.raises(ValueError, match="complete native"):
        acceptance._load_native(parent, tmp_path, "ops")


def test_public_routes_require_complete_parent_and_explicit_named_contract(campaign, tmp_path):
    from collector import glm53flash_validation as reader

    parent, children, _, _ = campaign
    path = tmp_path / export.BASENAME
    with pytest.raises(ValueError, match="original complete parent"):
        reader.publish_sharded_calibration(children, parent["shard_manifest"], path)
    with pytest.raises(ValueError, match="explicit named"):
        reader.publish_sharded_calibration(children, parent["shard_manifest"], path, parent_run=parent)
    reader.publish_sharded_calibration(
        children, parent["shard_manifest"], path, parent_run=parent, lookup_contract=export.NAMED_CONTRACT
    )
    with pytest.raises(ValueError, match="original complete parent"):
        reader.bind_sharded_calibration([path], children, parent["shard_manifest"])
    assert reader.bind_sharded_calibration([path], children, parent["shard_manifest"], parent_run=parent)["rows"] == 734


@pytest.mark.parametrize("field", ["capture_registry_sha256", "state_layout_sha256", "resolved_config_sha256"])
def test_distinct_capture_evidence_is_rejected_not_normalized(campaign, field):
    parent, children, _, _ = campaign
    policy = copy.deepcopy(children[1][1]["graph_policy"])
    if isinstance(policy[field], dict):
        policy[field]["0"] = "f" * 64
    else:
        policy[field] = "f" * 64
    children[1][1]["graph_policy"] = policy
    with pytest.raises(ValueError, match="execution/capture policy"):
        shards.calibration_rows(parent, children, lookup_contract=export.NAMED_CONTRACT)


def test_real_rust_queries_preserve_each_named_shard_endpoint(campaign, tmp_path):
    import shutil
    from importlib.resources import files

    from aisimulate_core.sdk.engine import EngineHandle

    parent, children, _, _ = campaign
    systems = tmp_path / "systems"
    data = systems / "data/gb300/sglang/0.5.20"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    shards.publish_calibration(parent, children, data / export.BASENAME, lookup_contract=export.NAMED_CONTRACT)
    binding = shards.bind_calibration([data / export.BASENAME], parent, children)
    engine = EngineHandle.compile(
        "zai-org/GLM-5.3-Flash",
        "gb300",
        "sglang",
        backend_version="0.5.20",
        tp_size=2,
        moe_tp_size=2,
        moe_ep_size=1,
        systems_path=str(systems),
        database_mode="SILICON",
        strict_provenance=True,
    )
    # Independent authored fixture: one .8ms embedding collective,90 other
    # .004ms collectives,275 .001ms units and one .002ms setup unit.
    expected = 0.8 + 90 * 0.004 + 275 * 0.001 + 0.002
    for prefix, factor in [(128, 1), (132, 1.5), (136, 2)]:
        assert engine.predict_decode_latency(1, prefix, 2) == pytest.approx(expected * factor)
        assert engine.last_provenance() is None
    audit = engine.glm53flash_lookup_audit("generation", 1, 1, 132)
    assert len(audit["operations"]) == 367
    evidence = {child["evidence_sha256"] for child in binding["children"]}
    assert all({e["evidence_sha256"] for e in row["endpoints"]} == evidence for row in audit["operations"])
    with pytest.raises(ValueError):
        engine.predict_decode_latency(1, 140, 2)


@pytest.mark.parametrize("defect", [None, "missing_audit", "missing_point", "control_changed"])
def test_group_holdout_keeps_original_ids_errors_and_endpoint_audits(campaign, tmp_path, monkeypatch, defect):
    from collector import glm53flash_validation as reader

    parent, children, _, natives = campaign
    path = tmp_path / export.BASENAME
    shards.publish_calibration(parent, children, path, lookup_contract=export.NAMED_CONTRACT)
    binding = shards.bind_calibration([path], parent, children)
    calibration = {
        **children[0][1],
        "_children": {run["cell"]["cell_id"]: native for run, native in children},
        "request_ids": set.union(*(native["request_ids"] for _, native in children)),
    }
    del calibration["runtime_run_id"]
    del calibration["evidence_root"]
    holdout = frozen_parent("holdout")
    admitted = {
        run["cell"]["cell_id"]: {
            **children[index][1],
            "runtime_run_id": f"TEST_ONLY-holdout-{index}",
            "evidence_root": str(tmp_path / f"holdout-{index}"),
            "request_ids": {f"TEST_ONLY-holdout-{index}"},
        }
        for index, run in enumerate(holdout["children"])
    }
    monkeypatch.setattr(reader, "load_native", lambda run, base: admitted[run["cell"]["cell_id"]])

    def predict(run, *args, **kwargs):
        if defect == "missing_point":
            return {"rows": {}, "diagnostics": {}}
        if run["original_point_ids"][1] == 2:
            return {
                "rows": {1: {"error": "TEST_ONLY missing measured endpoint"}},
                "prediction_evidence": {},
                "diagnostics": {},
            }
        return {
            "rows": {1: {"prediction_ms": 1.437}},
            "prediction_evidence": {} if defect == "missing_audit" else {1: {"TEST_ONLY": "endpoint audit"}},
            "diagnostics": {},
        }

    monkeypatch.setattr(export, "predict_homogeneous", predict)
    if defect == "control_changed":
        control = next(native for root, native in natives.items() if root.name == "control-1")
        control["receipts"][0]["sha256"] = "f" * 64
    if defect:
        with pytest.raises(ValueError):
            shards.predict_homogeneous(holdout, tmp_path, {}, calibration, binding)
    else:
        result = shards.predict_homogeneous(holdout, tmp_path, {}, calibration, binding)
        assert set(result["rows"]) == {1, 2}
        assert "error" in result["rows"][2]
        assert set(result["prediction_evidence"]) == {1}
        assert result["prediction_evidence_origins"][1] == {
            "child_cell_id": holdout["children"][0]["cell"]["cell_id"],
            "native_benchmark_id": 1,
            "original_point_id": 1,
        }
