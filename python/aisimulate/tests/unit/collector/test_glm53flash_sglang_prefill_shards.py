# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY complete schema4 point ownership; no measured GPU data."""

import copy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from collector import glm53flash_sglang_prefill_export as export
from collector import glm53flash_sglang_prefill_shards as shards
from collector.fpm_forward.glm53flash_validation import _sglang_execution_policy
from collector.glm53flash_contract import build_model_manifest, canonical_json, sha256_json
from collector.glm53flash_graph_policy import NATIVE_SOURCE_SHA256

pytestmark = pytest.mark.unit


def frozen_parent(role="calibration"):
    points = [{"batch_size": 1, "total_kv_read_tokens": n, "total_prefill_tokens": 32} for n in (118, 126)]
    payload = {"schema_version": 3, "decode": [], "prefill": points}
    corpus = ("c" if role == "calibration" else "d") * 64
    cell = {"cell_id": "TEST_ONLY-parent", "workload_kind": "prefill"}
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
            "phase": "prefill",
            "point_map": [{"native_benchmark_id": 1, "original_point_id": index, "point": point}],
        }
        sid = sha256_json(identity)[:20]
        child = copy.deepcopy(plan)
        child["cells"][0]["cell_id"] = cid = f"fpm-shard-{sid}"
        child_payload = {"schema_version": 3, "decode": [], "prefill": [point]}
        child["options"]["benchmark_points"] = {"payload": child_payload, "sha256": sha256_json(child_payload)}
        child["sha256"] = sha256_json({"shard": identity, "child_plan": {**child, "sha256": ""}})
        manifest["shards"].append(
            {**identity, "shard_id": sid, "child_cell_id": cid, "child_plan_sha256": child["sha256"]}
        )
        runs.append(
            {
                "key": ("sglang", "fp8", 2, "prefill"),
                "plan": child,
                "cell": child["cells"][0],
                "points": [dict(point, benchmark_id=1, point_type="prefill")],
                "corpus": corpus,
                "spec": {"ops_execution_mode": "native_eager_prefill"},
                "role": role,
                "original_point_ids": {1: index},
            }
        )
    plan["sharding"] = manifest
    return {
        "key": runs[0]["key"],
        "plan": plan,
        "cell": cell,
        "points": [dict(p, benchmark_id=i, point_type="prefill") for i, p in enumerate(points, 1)],
        "corpus": corpus,
        "spec": {"ops_execution_mode": "native_eager_prefill"},
        "role": role,
        "children": runs,
        "shard_manifest": manifest,
    }


def proof_fixture(run, index):
    manifest = build_model_manifest("sglang", "fp8", 2)
    execution = {
        "cuda_graph_config": {"prefill": {"backend": "disabled"}, "decode": {"backend": "full"}},
        "mem_fraction_static": 0.82,
    }
    provenance = {**manifest, "source_sha256": NATIVE_SOURCE_SHA256, "runtime_digest": "sha256:" + "a" * 64}
    point = run["points"][0]
    forwards = {}
    entries = manifest["phases"]["context"] + manifest["runtime_operations"]["context"]
    for repetition in range(15):
        row = {
            "phase": "context",
            "runtime_mode": "NONE",
            "used_cuda_graph": False,
            "request_ids": [f"TEST_ONLY-cal-{index}-{repetition}"],
            "benchmark_id": 1,
            "repetition": repetition,
            "sampling_role": "warmup" if repetition < 5 else "measurement",
            "dataset_role": "calibration",
            "request_set": f"TEST_ONLY-cal-{index}",
            "corpus_sha256": run["corpus"],
            "batch_size": 1,
            "prefix_lengths": [point["total_kv_read_tokens"]],
            "query_lengths": [32],
            "num_padded_tokens": 32,
            "token_witness": [{"TEST_ONLY": True}],
            "whole_forward_gpu_ms": 0.5,
            "forward_id": f"TEST_ONLY-{index}-{repetition}",
            "invocation": repetition,
            "native_prefill_setup": {
                "source": "sglang.srt.utils.common.BumpAllocator.__init__",
                "buffer_size": 90,
                "dtype": "torch.float32",
            },
            "native_prefill_calls": [
                {
                    "operation": entry["name"],
                    "source": "TEST_ONLY.native." + entry["name"],
                    "included_sources": [],
                    "excluded_collective_sources": [],
                    "parent_operation": None,
                }
                for entry in entries
                if entry["component"] != "runtime"
            ],
            "binding": {
                entry["name"]: {
                    "dispatch": sha256_json({"TEST_ONLY": entry["name"]}),
                    "activity_count": 1,
                    "contribution_count": 1,
                    "latency": 0.002 if entry["component"] == "runtime" else 0.001,
                }
                for entry in entries
            },
        }
        forwards[1, repetition] = {rank: {**copy.deepcopy(row), "tp_rank": rank} for rank in range(2)}
    return {
        "policy": export.build_policy(manifest, provenance, execution),
        "manifest": manifest,
        "policy_evidence_sha256": str(index + 1) * 64,
        "forwards": forwards,
        **_sglang_execution_policy(execution),
    }


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
        (root / "prefill-calibration-evidence.json").write_text(canonical_json(receipt))
        native = {
            "evidence_root": str(root),
            "runtime_run_id": rid,
            "request_ids": {row[0]["request_ids"][0] for row in proof["forwards"].values()},
            "prefill_policy": proof["policy"],
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
        (root / "prefill-profile-control.json").write_text(
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
    # prefill reader tests. These substitutions isolate ownership using the
    # actual complete 366+1 named reducer and public Rust consumer.
    monkeypatch.setattr(export, "read_prefill_run", lambda root, run: proofs[Path(root)])
    monkeypatch.setattr(export, "verify_evidence", lambda root, proof: evidence[Path(root)])
    monkeypatch.setattr(reader, "_load_native", lambda run, root, **kw: native_by_root[Path(root)])
    return parent, children, proofs, native_by_root


def test_complete_publication_preserves_all_units_and_independent_attempts(campaign, tmp_path):
    parent, children, _, _ = campaign
    path = tmp_path / export.BASENAME
    publication = shards.publish_calibration(parent, children, path)
    bound = shards.bind_calibration([path], parent, children)
    rows = pq.read_table(path).to_pylist()
    assert bound["rows"] == 734
    assert len({row["operation_name"] for row in rows}) == 367
    assert len({row["evidence_sha256"] for row in rows}) == 2
    assert {row["original_point_id"] for row in publication["ownership"]["rows"]} == {1, 2}
    assert len({child["control"]["native_runtime_run_id"] for child in bound["children"]}) == 2
    assert publication["accuracy_acceptance"] == "NOT_EVALUATED"
    with pytest.raises(ValueError, match="new canonical"):
        shards.publish_calibration(parent, children, path)
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
        parent["key"] = ("sglang", "fp8", 2, "decode")
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
        native["prefill_policy"] = {**native["prefill_policy"], "config_sha256": "f" * 64}
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
        shards.calibration_rows(parent, children)


def test_parent_loader_preserves_policy_and_does_not_invent_aggregate_run(campaign, tmp_path, monkeypatch):
    from collector import glm53flash_validation as reader
    from collector.fpm_forward import glm53flash_validation as acceptance

    parent, children, _, _ = campaign
    admitted = {run["cell"]["cell_id"]: native for run, native in children}
    monkeypatch.setattr(reader, "load_native", lambda run, base: admitted[run["cell"]["cell_id"]])
    native = acceptance._load_native(parent, tmp_path, "ops")
    assert native["prefill_policy"] == children[0][1]["prefill_policy"]
    assert native["values"] == {1: 0.5, 2: 0.5}
    assert "runtime_run_id" not in native and "evidence_root" not in native
    path = tmp_path / export.BASENAME
    shards.publish_calibration(parent, children, path)
    binding = shards.bind_calibration([path], parent, children)
    shards.validate_prediction_binding(native, binding)
    changed = copy.deepcopy(binding)
    changed["children"].pop()
    with pytest.raises(ValueError, match="complete original"):
        shards.validate_prediction_binding(native, changed)
    children[1][1]["prefill_policy"] = {**children[1][1]["prefill_policy"], "runtime_digest": "sha256:" + "e" * 64}
    with pytest.raises(ValueError, match="complete native"):
        acceptance._load_native(parent, tmp_path, "ops")


@pytest.mark.parametrize("defect", ["known_unknown", "different_actual"])
def test_allocator_identity_is_not_lost_between_original_shards(campaign, defect):
    from collector.fpm_forward import sglang_allocator as allocator

    from .test_glm53flash_sglang_allocator import worker

    parent, children, proofs, _ = campaign
    request = allocator.request_policy(16384)
    observed = worker(0, request, {}, {})
    actual = allocator.validate_worker(observed, rank=0, run_id="run", execution_identity={}, policy=request)
    for index, (_, native) in enumerate(children):
        if index == 1 and defect == "known_unknown":
            continue
        policy = copy.deepcopy(actual)
        if index == 1:
            policy["torch"]["files"]["lib/libtorch_cuda.so"]["sha256"] = "f" * 64
        update = _sglang_execution_policy(native["_execution_policy"], policy)
        native.update(update)
        native["prefill_policy"] = {
            **native["prefill_policy"],
            "execution_policy_sha256": update["execution_policy"]["sha256"],
        }
        proof = proofs[Path(native["evidence_root"])]
        proof.update(update)
        proof["policy"] = native["prefill_policy"]
    with pytest.raises(ValueError, match="execution policies differ"):
        shards.calibration_rows(parent, children)


@pytest.mark.parametrize("defect", ["duplicate", "missing", "wrong_phase"])
def test_final_table_cannot_add_or_drop_a_named_unit(campaign, tmp_path, defect):
    parent, children, _, _ = campaign
    path = tmp_path / export.BASENAME
    shards.publish_calibration(parent, children, path)
    rows = pq.read_table(path).to_pylist()
    if defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "missing":
        rows.pop()
    else:
        rows.append({**rows[0], "phase": "generation"})
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="differs from complete"):
        shards.bind_calibration([path], parent, children)


def test_original_control_change_invalidates_prediction_binding(campaign, tmp_path):
    parent, children, _, natives = campaign
    path = tmp_path / export.BASENAME
    shards.publish_calibration(parent, children, path)
    binding = copy.deepcopy(shards.bind_calibration([path], parent, children))
    native = {**children[0][1], "_children": {run["cell"]["cell_id"]: child for run, child in children}}
    del native["runtime_run_id"]
    del native["evidence_root"]
    shards.validate_prediction_binding(native, binding)
    control = next(value for root, value in natives.items() if root.name == "control-1")
    control["receipts"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="provenance changed"):
        shards.validate_prediction_binding(native, binding)


def test_holdout_preserves_original_ids_and_exact_only_failures(campaign, tmp_path, monkeypatch):
    from collector import glm53flash_validation as reader

    parent, children, _, _ = campaign
    path = tmp_path / export.BASENAME
    shards.publish_calibration(parent, children, path)
    binding = shards.bind_calibration([path], parent, children)
    native = {
        **children[0][1],
        "_children": {run["cell"]["cell_id"]: value for run, value in children},
        "request_ids": set.union(*(value["request_ids"] for _, value in children)),
    }
    del native["runtime_run_id"]
    del native["evidence_root"]
    holdout = frozen_parent("holdout")
    admitted = {
        run["cell"]["cell_id"]: {
            **children[index][1],
            "runtime_run_id": f"TEST_ONLY-holdout-{index}",
            "request_ids": {f"TEST_ONLY-holdout-{index}"},
            "evidence_root": str(tmp_path / f"holdout-{index}"),
        }
        for index, run in enumerate(holdout["children"])
    }
    monkeypatch.setattr(reader, "load_native", lambda run, base: admitted[run["cell"]["cell_id"]])

    def predict(run, *args):
        value = (
            {"error": "TEST_ONLY missing exact measurement"}
            if run["original_point_ids"][1] == 2
            else {"prediction_ms": 0.368}
        )
        return {"rows": {1: value}, "diagnostics": {"interpolation": "EXACT_ONLY"}}

    monkeypatch.setattr(export, "predict_homogeneous", predict)
    result = shards.predict_homogeneous(holdout, tmp_path, {}, native, binding)
    assert result["rows"] == {1: {"prediction_ms": 0.368}, 2: {"error": "TEST_ONLY missing exact measurement"}}
    second = admitted[holdout["children"][1]["cell"]["cell_id"]]
    second["runtime_run_id"] = next(iter(admitted.values()))["runtime_run_id"]
    with pytest.raises(ValueError, match="reused an original native run"):
        shards.predict_homogeneous(holdout, tmp_path, {}, native, binding)
    second["runtime_run_id"] = "TEST_ONLY-holdout-1"
    monkeypatch.setattr(export, "predict_homogeneous", lambda *args: {"rows": {}, "diagnostics": {}})
    with pytest.raises(ValueError, match="omitted a frozen child point or failure"):
        shards.predict_homogeneous(holdout, tmp_path, {}, native, binding)


def test_publication_entry_requires_original_parent_and_uses_schema4(campaign, tmp_path):
    from collector import glm53flash_validation as reader

    parent, children, _, _ = campaign
    path = tmp_path / export.BASENAME
    with pytest.raises(ValueError, match="original complete parent"):
        reader.publish_sharded_calibration(children, parent["shard_manifest"], path)
    reader.publish_sharded_calibration(children, parent["shard_manifest"], path, parent_run=parent)
    with pytest.raises(ValueError, match="original complete parent"):
        reader.bind_sharded_calibration([path], children, parent["shard_manifest"])
    assert reader.bind_sharded_calibration([path], children, parent["shard_manifest"], parent_run=parent)["rows"] == 734


@pytest.mark.parametrize("bounded", [False, True])
def test_actual_public_rust_queries_complete_merged_points_and_rejects_missing_exact(campaign, tmp_path, bounded):
    import shutil
    from importlib.resources import files

    from aisimulate_core.sdk.engine import EngineHandle

    parent, children, _, _ = campaign
    systems = tmp_path / "systems"
    data = systems / "data/gb300/sglang/0.5.20"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    shards.publish_calibration(
        parent, children, data / export.BASENAME, lookup_contract=export.LOOKUP_CONTRACT if bounded else None
    )
    binding = shards.bind_calibration([data / export.BASENAME], parent, children)
    assert binding.get("lookup_contract") == (export.LOOKUP_CONTRACT if bounded else None)
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
    for prefix in (118, 126):
        assert engine.predict_prefill_latency(1, prefix + 32, prefix) == pytest.approx(0.368)
        assert engine.last_provenance() is None
    if bounded:
        assert engine.predict_prefill_latency(1, 154, 122) == pytest.approx(0.368)
        audit = engine.glm53flash_lookup_audit("context", 1, 32, 122)
        assert len(audit["operations"]) == 367
        assert all([p["point"]["prefix"] for p in row["endpoints"]] == [118, 126] for row in audit["operations"])
        evidence = {child["evidence_sha256"] for child in binding["children"]}
        assert all(
            {p["measurement"]["evidence_sha256"] for p in row["endpoints"]} == evidence for row in audit["operations"]
        )
    else:
        with pytest.raises(ValueError):
            engine.predict_prefill_latency(1, 154, 122)


@pytest.mark.parametrize("defect", [None, "missing", "orphan", "failed", "both_fields"])
def test_optin_holdout_audit_is_complete_and_preserves_original_child_origin(campaign, tmp_path, monkeypatch, defect):
    from collector import glm53flash_validation as reader

    parent, children, _, _ = campaign
    path = tmp_path / export.BASENAME
    shards.publish_calibration(parent, children, path, lookup_contract=export.LOOKUP_CONTRACT)
    binding = shards.bind_calibration([path], parent, children)
    native = {
        **children[0][1],
        "_children": {run["cell"]["cell_id"]: value for run, value in children},
        "request_ids": set.union(*(value["request_ids"] for _, value in children)),
    }
    del native["runtime_run_id"]
    del native["evidence_root"]
    holdout = frozen_parent("holdout")
    admitted = {
        run["cell"]["cell_id"]: {
            **children[index][1],
            "runtime_run_id": f"TEST_ONLY-independent-holdout-{index}",
            "request_ids": {f"TEST_ONLY-independent-holdout-{index}"},
            "evidence_root": str(tmp_path / f"holdout-{index}"),
        }
        for index, run in enumerate(holdout["children"])
    }
    monkeypatch.setattr(reader, "load_native", lambda run, base: admitted[run["cell"]["cell_id"]])
    original_audit = {"TEST_ONLY": "native audit preserved without rewriting"}

    def predict(run, *args):
        index = run["original_point_ids"][1]
        row = {"prediction_ms": 0.368} if index == 1 else {"error": "TEST_ONLY unsupported geometry"}
        audits = {1: original_audit} if index == 1 else {}
        if index == 1 and defect == "missing":
            audits = {}
        elif index == 1 and defect == "orphan":
            audits[99] = original_audit
        elif index == 2 and defect == "failed":
            audits[1] = original_audit
        elif index == 1 and defect == "both_fields":
            row["error"] = "TEST_ONLY conflicting row"
        return {"rows": {1: row}, "prediction_evidence": audits, "diagnostics": {}}

    monkeypatch.setattr(export, "predict_homogeneous", predict)
    if defect:
        with pytest.raises(ValueError, match="exactly one audit"):
            shards.predict_homogeneous(holdout, tmp_path, {}, native, binding)
    else:
        result = shards.predict_homogeneous(holdout, tmp_path, {}, native, binding)
        assert result["prediction_evidence"] == {1: original_audit}
        assert result["prediction_evidence"][1] is original_audit
        assert result["prediction_evidence_origins"] == {
            1: {
                "child_cell_id": holdout["children"][0]["cell"]["cell_id"],
                "native_benchmark_id": 1,
                "original_point_id": 1,
            }
        }
        assert set(result["rows"]) == {1, 2} and "error" in result["rows"][2]
