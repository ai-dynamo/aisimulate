# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY original/derived identities; authored times never qualify a GPU."""

import copy
import json
from pathlib import Path

import pytest
from collector import glm53flash_observation_partition as partition
from collector import glm53flash_serving_shards as shards
from collector import glm53flash_vllm_serving_export as serving
from collector.fpm_forward import glm53flash_validation as validation
from collector.glm53flash_contract import canonical_json, sha256_json
from collector.glm53flash_jsonl import file_sha256

from .test_glm53flash_validation import write_plan
from .test_glm53flash_vllm_serving_export import fixture as serving_fixture

pytestmark = pytest.mark.unit


def write_ref(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value))
    return {"path": name, "sha256": file_sha256(path)}


def runtime(root):
    sources = {}
    for name in ("none.py", "graph.py"):
        path = root / name
        path.write_text("# TEST_ONLY entry source, never executed\n")
        sources[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    graph = {
        "entry": "graph.py",
        "observer_flag": "AISIM_GLM53_PIECEWISE_REPLAY",
        "calibration_addon": [],
        "control_holdout_addon": False,
    }
    return {
        "schema": partition.RUNTIME_CONTRACT,
        "producer": {"commit": "1" * 40, "arm_wheel_sha256": "2" * 64, "source_resource_map_sha256": "3" * 64},
        "shared_cache_sha256": "4" * 64,
        "shared_overlay_sha256": "5" * 64,
        "native_runtime": {"TEST_ONLY": True},
        "entry_files": sources,
        "entry_mechanisms": {
            "NONE": {
                "entry": "none.py",
                "observer_flag": "AISIM_GLM53_SERVING_NONE_MEASURED",
                "calibration_addon": [],
                "control_holdout_addon": False,
            },
            "FULL": graph,
            "PIECEWISE": copy.deepcopy(graph),
        },
    }


def original(root, role="calibration", phase="prefill"):
    spec = write_plan(root, ("vllm", "fp8", 2, phase), "calibration" if role == "control" else role)
    parent = json.loads((root / spec["plan"]["path"]).read_bytes())
    offset = 16 if role == "holdout" else 0
    points = [
        {
            "batch_size": 1,
            "total_kv_read_tokens": 128 + offset + i * 128,
            "total_prefill_tokens": q if phase == "prefill" else 0,
        }
        for i, q in enumerate((4, 4096, 2) if phase == "prefill" else (0, 0))
    ]
    payload = {"schema_version": 3, "prefill": [], "decode": []}
    payload[phase] = points
    parent["options"].update(
        benchmark_points={"payload": payload, "sha256": sha256_json(payload)}, warmup_repeats=5, measurement_repeats=10
    )
    parent.update(
        dict.fromkeys(
            (
                "aic_revision",
                "generator_config_sha256",
                "dtype_profile",
                "topologies",
                "topology_memory_admission",
                "backend_policies",
            ),
            "TEST_ONLY",
        )
    )
    parent["capability"] = {"aic_database_version": "0.30.0"}
    identity = {
        "parent_cell_id": spec["cell_id"],
        "phase": phase,
        "point_map": [{"native_benchmark_id": i, "original_point_id": i, "point": p} for i, p in enumerate(points, 1)],
    }
    sid = sha256_json(identity)[:20]
    cid = "fpm-shard-" + sid
    child = copy.deepcopy(parent)
    child["cells"][0]["cell_id"] = cid
    child["sha256"] = sha256_json({"shard": identity, "child_plan": {**child, "sha256": ""}})
    parent["sharding"] = {
        "schema_name": "aic_fpm_shard_manifest",
        "schema_version": 1,
        "parent_plan_sha256": parent["sha256"],
        "parent_points_sha256": sha256_json(payload),
        "shards": [{**identity, "shard_id": sid, "child_cell_id": cid, "child_plan_sha256": child["sha256"]}],
    }
    return parent, {cid: child}


def setup(root, role="calibration", phase="prefill"):
    root.mkdir(parents=True, exist_ok=True)
    parent, children = original(root, role, phase)
    identity = runtime(root)
    receipt = partition.build_partition(
        parent,
        children,
        identity,
        parent_cell_id=parent["cells"][0]["cell_id"],
        role=role,
        campaign_id="TEST_ONLY-campaign",
    )
    spec = {
        "plan": write_ref(root, "parent.json", parent),
        "cell_id": parent["cells"][0]["cell_id"],
        "ops_execution_mode": "native_serving",
        "ops_observation_partition": write_ref(root, "partition.json", receipt),
        "observation_runtime_sources": {
            name: {"path": name, "sha256": item["sha256"]} for name, item in identity["entry_files"].items()
        },
        "observation_children": [],
    }
    for leaf in receipt["leaves"]:
        cid = leaf["leaf_id"]
        spec["observation_children"].append(
            {
                "plan": write_ref(root, cid + ".json", partition.leaf_plan(receipt, cid)),
                "cell_id": cid,
                "ops_execution_mode": "native_serving",
                "raw_root": str(root / cid / "native"),
            }
        )
    return spec, validation._plan_run(spec, root, role)


@pytest.mark.parametrize(
    "role,phase", [(r, p) for r in ("calibration", "control", "holdout") for p in ("prefill", "decode")]
)
def test_public_partition_route_keeps_full_original_union_and_new_native_ids(tmp_path, role, phase):
    _, run = setup(tmp_path, role, phase)
    declared = shards.validate_children(run, run["children"])
    assert len(declared) == (2 if phase == "prefill" else 1)
    assert sorted(i for child in run["children"] for i in child["original_point_ids"].values()) == list(
        range(1, len(run["points"]) + 1)
    )
    assert all(child["plan"]["schema_name"] == partition.LEAF_SCHEMA for child in run["children"])
    if phase == "prefill":
        pw = next(c for c in run["children"] if c["observation_leaf"]["observation_family"] == "PIECEWISE")
        assert partition.origin(pw, 2)["original_child_benchmark_id"] == 3
        assert partition.origin(pw, 2)["original_point_id"] == 3
        assert pw["points"][1]["benchmark_id"] == 2
    assert "raw_root" not in run["spec"] and "runtime_run_id" not in run


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "extra",
        "family",
        "old_local",
        "parent_id",
        "point",
        "phase",
        "corpus",
        "runtime",
        "source",
        "rule",
        "unused_phase",
    ],
)
def test_consistently_hashed_wrong_partition_still_rejects_original_identity(tmp_path, defect):
    spec, run = setup(tmp_path)
    value = copy.deepcopy(run["observation_partition"])
    leaf = value["leaves"][0]
    if defect == "missing":
        value["leaves"].pop()
    elif defect == "duplicate":
        value["leaves"].append(copy.deepcopy(leaf))
    elif defect == "extra":
        leaf["unexpected"] = True
    elif defect == "family":
        leaf["observation_family"] = "FULL"
    elif defect == "old_local":
        leaf["point_map"][0]["original_child_benchmark_id"] += 1
    elif defect == "parent_id":
        leaf["point_map"][0]["original_point_id"] += 1
    elif defect == "point":
        leaf["point_map"][0]["point"]["total_kv_read_tokens"] += 1
    elif defect == "phase":
        leaf["phase"] = "decode"
    elif defect == "corpus":
        leaf["corpus_sha256"] = "f" * 64
    elif defect == "runtime":
        value["runtime_identity"]["producer"]["commit"] = "f" * 40
    elif defect == "source":
        next(iter(value["source_children"].values()))["options"]["measurement_repeats"] = 9
    elif defect == "rule":
        value["family_rule"]["max_capture_tokens"] = 4096
    else:
        del leaf["points_payload"]["decode"]
    spec["ops_observation_partition"] = write_ref(tmp_path, "changed-partition.json", value)
    with pytest.raises(ValueError):
        validation._plan_run(spec, tmp_path, "calibration")


@pytest.mark.parametrize("defect", ["missing", "duplicate", "plan", "mode", "source", "aggregate", "route"])
def test_public_leaf_route_rejects_incomplete_source_or_custom_plan(tmp_path, defect):
    spec, _ = setup(tmp_path)
    if defect == "missing":
        spec["observation_children"].pop()
    elif defect == "duplicate":
        spec["observation_children"][-1] = copy.deepcopy(spec["observation_children"][0])
    elif defect == "mode":
        spec["observation_children"][0]["ops_execution_mode"] = "eager"
    elif defect == "source":
        (tmp_path / "none.py").write_text("# changed entry\n")
    elif defect == "aggregate":
        spec["raw_root"] = "made-up-native"
    elif defect == "route":
        del spec["ops_observation_partition"]
    else:
        ref = spec["observation_children"][0]["plan"]
        plan = json.loads((tmp_path / ref["path"]).read_bytes())
        plan["schema_name"] = "aic_fpm_collection_plan"
        spec["observation_children"][0]["plan"] = write_ref(tmp_path, "wrong-plan.json", plan)
    with pytest.raises(ValueError):
        validation._plan_run(spec, tmp_path, "calibration")


def proof_for(run):
    leaf = run["observation_leaf"]
    proof = {
        "provenance": {"run_id": leaf["run_id"], "runtime_digest": leaf["runtime_digest"]},
        "snapshots": {
            i: {"max_num_reqs": 32, "max_capture_tokens": 2048, "resolved_mode": "FULL_AND_PIECEWISE"} for i in range(2)
        },
        "forwards": {},
    }
    for point in run["points"]:
        for rep in range(15):
            proof["forwards"][(point["benchmark_id"], rep)] = {
                rank: {
                    "runtime_mode": leaf["observation_family"],
                    "tp_rank": rank,
                    "benchmark_id": point["benchmark_id"],
                    "repetition": rep,
                }
                for rank in range(2)
            }
    return proof


@pytest.mark.parametrize("defect", [None, "mode", "rank", "repetition", "run", "runtime", "policy"])
def test_actual_family_gate_is_additional_and_requires_each_completed_target(tmp_path, defect):
    _, parent = setup(tmp_path)
    run = parent["children"][0]
    proof = proof_for(run)
    if defect == "mode":
        proof["forwards"][(1, 0)][0]["runtime_mode"] = "FULL"
    elif defect == "rank":
        del proof["forwards"][(1, 0)][1]
    elif defect == "repetition":
        del proof["forwards"][(1, 14)]
    elif defect == "run":
        proof["provenance"]["run_id"] = "historical-pilot"
    elif defect == "runtime":
        proof["provenance"]["runtime_digest"] = "sha256:" + "a" * 64
    elif defect == "policy":
        proof["snapshots"][0]["max_capture_tokens"] = 4
    if defect:
        with pytest.raises(ValueError):
            partition.check_native_proof(run, proof)
    else:
        partition.check_native_proof(run, proof)


@pytest.mark.parametrize("defect", [None, "missing", "mixed", "point", "campaign"])
def test_control_pair_has_same_new_inputs_but_distinct_execution_identity(tmp_path, defect):
    _, cal = setup(tmp_path / "cal")
    _, ctrl = setup(tmp_path / "ctrl", "control")
    left, right = cal["children"][0], ctrl["children"][0]
    if defect == "missing":
        right = {}
    elif defect == "mixed":
        right["observation_leaf"]["role"] = "holdout"
    elif defect == "point":
        right["observation_leaf"]["point_map"][0]["point"]["total_kv_read_tokens"] += 1
    elif defect == "campaign":
        right["observation_leaf"]["campaign_id"] = "other"
    if defect:
        with pytest.raises(ValueError):
            partition.check_control_pair(left, right)
    else:
        partition.check_control_pair(left, right)
        assert left["observation_leaf"]["run_id"] != right["observation_leaf"]["run_id"]


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    _, parent = setup(tmp_path / "cal", phase="decode")
    run = parent["children"][0]
    root = Path(run["spec"]["raw_root"])
    root.mkdir(parents=True)
    proof = serving_fixture(batch=1)
    first = copy.deepcopy(proof["forwards"])
    for index, point in enumerate(run["points"], 1):
        for (_, rep), ranks in copy.deepcopy(first).items():
            for row in ranks.values():
                row["benchmark_id"] = index
                row["invocation"] = index * 15 + rep + 1
                row["forward_id"] = f"rank-{row['tp_rank']}/forward-{row['invocation']}"
                row["prefix_lengths"] = [point["total_kv_read_tokens"]]
                row["request_set"] = "TEST_ONLY-actual-request-set"
                row["request_ids"] = [f"TEST_ONLY-{index}-{rep}"]
                for name, unit in row["binding"].items():
                    unit["source_ownership_sha256"] = sha256_json({"TEST_ONLY_owner": name})
            proof["forwards"][(index, rep)] = ranks
    receipt = {
        "request_set": "TEST_ONLY-actual-request-set",
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
    }
    (root / "serving-calibration-evidence.json").write_text(canonical_json(receipt))
    native = {
        "evidence_root": str(root),
        "runtime_run_id": receipt["request_set"],
        "request_ids": {"TEST_ONLY-request"},
        "graph_policy": proof["policy"],
        "backend_version": "0.30.0",
        "values": {},
        "timing_boundary": serving.BOUNDARY,
        **{key: proof[key] for key in ("execution_policy", "_execution_policy")},
        "observation_identity": {
            "leaf": run["observation_leaf"],
            "partition_sha256": sha256_json(run["observation_partition"]),
            "source_plan_sha256": run["plan"]["sha256"],
        },
    }
    # Raw source/trace/hardware gates have separate fixtures. Only their I/O is
    # replaced here; the original real named 277+setup reducer remains in use.
    monkeypatch.setattr(serving, "read_serving_run", lambda root, run: proof)
    monkeypatch.setattr(serving, "verify_evidence", lambda root, proof: receipt)
    return parent, [(run, native)], proof


def test_observation_publication_preserves_named_rows_and_origin_without_fake_fpm_child(calibration, tmp_path):
    parent, children, _ = calibration
    path = tmp_path / serving.BASENAME
    result = shards.publish_calibration(parent, children, path, lookup_contract=serving.LOOKUP_CONTRACT)
    binding = shards.bind_calibration([path], parent, children)
    assert binding["rows"] == 556
    owner = result["ownership"]
    assert owner["schema"] == partition.OWNERSHIP_SCHEMA
    assert {row["original_point_id"] for row in owner["rows"]} == {1, 2}
    assert all(row["observation_leaf_id"] != row["original_child_cell_id"] for row in owner["rows"])
    native = {**children[0][1], "_children": {run["cell"]["cell_id"]: n for run, n in children}}
    del native["runtime_run_id"]
    shards.validate_prediction_binding(native, binding)
    changed = copy.deepcopy(binding)
    changed["ownership"]["observation_partition"]["leaves"][0]["point_map"][0]["original_point_id"] = 99
    changed["ownership"]["shard_manifest_sha256"] = sha256_json(changed["ownership"]["observation_partition"])
    changed["calibration_group_sha256"] = sha256_json(changed["ownership"])
    with pytest.raises(ValueError, match="actual native leaf"):
        shards.validate_prediction_binding(native, changed)


def test_holdout_origins_preserve_both_local_ids_and_errors(calibration, tmp_path, monkeypatch):
    parent, children, _ = calibration
    path = tmp_path / serving.BASENAME
    shards.publish_calibration(parent, children, path, lookup_contract=serving.LOOKUP_CONTRACT)
    bound = shards.bind_calibration([path], parent, children)
    native = {**children[0][1], "_children": {run["cell"]["cell_id"]: n for run, n in children}}
    _, holdout = setup(tmp_path / "hold", "holdout", "decode")
    audit = {"TEST_ONLY": "unchanged Rust-shaped audit object"}
    monkeypatch.setattr(
        serving,
        "predict_homogeneous",
        lambda *args: {
            "rows": {1: {"prediction_ms": 7.0}, 2: {"error": "TEST_ONLY missing bracket"}},
            "prediction_evidence": {1: audit},
            "diagnostics": {},
        },
    )
    result = shards.predict_homogeneous(holdout, tmp_path, {}, native, bound)
    assert set(result["rows"]) == {1, 2} and "error" in result["rows"][2]
    assert result["prediction_evidence"][1] is audit
    assert result["prediction_evidence_origins"][1]["original_child_cell_id"].startswith("fpm-shard-")
    assert result["prediction_evidence_origins"][1]["observation_leaf_id"].startswith("ops-observation-")


@pytest.mark.parametrize("defect", ["owner", "downgrade", "actual_source", "missing_point"])
def test_rehashed_ownership_cannot_change_origins_or_downgrade(calibration, tmp_path, defect):
    parent, children, _ = calibration
    path = tmp_path / serving.BASENAME
    shards.publish_calibration(parent, children, path)
    bound = shards.bind_calibration([path], parent, children)
    native = {**children[0][1], "_children": {run["cell"]["cell_id"]: n for run, n in children}}
    if defect == "owner":
        bound["ownership"]["rows"][0]["original_child_benchmark_id"] = 999
    elif defect == "downgrade":
        bound["ownership"]["schema"] = "glm53flash_serving_shard_ownership_v1"
        del bound["ownership"]["observation_partition"]
    elif defect == "actual_source":
        next(iter(native["_children"].values()))["observation_identity"]["source_plan_sha256"] = "f" * 64
    else:
        bound["ownership"]["rows"] = [r for r in bound["ownership"]["rows"] if r["original_point_id"] != 2]
    bound["calibration_group_sha256"] = sha256_json(bound["ownership"])
    with pytest.raises(ValueError):
        shards.validate_prediction_binding(native, bound)


@pytest.mark.parametrize("defect", [None, "tokens"])
def test_paired_control_is_json_portable_and_preserves_actual_token_gate(calibration, tmp_path, monkeypatch, defect):
    from collector import glm53flash_validation as reader

    cal, children, proof = calibration
    _, ctrl = setup(tmp_path / "ctrl", "control", "decode")
    control_run = ctrl["children"][0]
    control_root = Path(control_run["spec"]["raw_root"])
    control_root.mkdir(parents=True)
    proof["observation_leaf"] = cal["children"][0]["observation_leaf"]
    other = copy.deepcopy(proof)
    other["observation_leaf"] = control_run["observation_leaf"]
    for ranks in other["forwards"].values():
        for row in ranks.values():
            row["request_set"] = "TEST_ONLY-control"
            row["request_ids"] = ["control-" + rid for rid in row["request_ids"]]
            if defect == "tokens":
                row["token_witness"] = [{"TEST_ONLY": "changed actual input"}]
    monkeypatch.setattr(serving, "read_serving_run", lambda root, run: other)
    monkeypatch.setattr(
        reader, "load_native", lambda run, base: {"evidence_root": str(control_root.resolve()), "receipts": []}
    )
    if defect:
        with pytest.raises(ValueError, match="actual cohort/tokens/dispatch"):
            serving.profile_control(Path(children[0][1]["evidence_root"]), proof, control_root, control_run)
    else:
        value = serving.profile_control(Path(children[0][1]["evidence_root"]), proof, control_root, control_run)
        frozen = json.loads(canonical_json(value))
        assert "runtime_cell" not in frozen["frozen_run"]
        assert frozen["frozen_run"]["plan"]["schema_name"] == partition.LEAF_SCHEMA
        assert len(frozen["results"]) == 2
        reread = serving.profile_control(
            Path(children[0][1]["evidence_root"]), proof, control_root, frozen["frozen_run"]
        )
        assert canonical_json(reread) == canonical_json(value)


@pytest.mark.parametrize("defect", [None, "missing", "same_run", "same_root", "policy"])
def test_parent_loader_keeps_all_original_ids_and_real_leaf_attempts(tmp_path, monkeypatch, defect):
    from collector import glm53flash_validation as reader

    _, run = setup(tmp_path, "holdout")
    natives = {}
    for index, child in enumerate(run["children"]):
        proof = serving_fixture()
        natives[child["cell"]["cell_id"]] = {
            "runtime_run_id": f"TEST_ONLY-runtime-{index}",
            "evidence_root": str(tmp_path / f"raw-{index}"),
            "request_ids": {f"TEST_ONLY-request-{index}"},
            "backend_version": "0.30.0",
            "timing_boundary": serving.BOUNDARY,
            "graph_policy": proof["policy"],
            **{key: proof[key] for key in ("execution_policy", "_execution_policy")},
            "values": {p["benchmark_id"]: 100.0 for p in child["points"]},
        }
    a, b = list(natives.values())
    if defect == "missing":
        b["values"].pop(1)
    elif defect == "same_run":
        b["runtime_run_id"] = a["runtime_run_id"]
    elif defect == "same_root":
        b["evidence_root"] = a["evidence_root"]
    elif defect == "policy":
        b["graph_policy"]["config_sha256"] = "f" * 64
    monkeypatch.setattr(reader, "load_native", lambda child, base: natives[child["cell"]["cell_id"]])
    if defect:
        with pytest.raises(ValueError):
            validation._load_native(run, tmp_path, "ops")
    else:
        result = validation._load_native(run, tmp_path, "ops")
        assert set(result["values"]) == {1, 2, 3}
        assert len(result["_children"]) == 2 and "runtime_run_id" not in result
    with pytest.raises(ValueError, match="explicit Ops"):
        validation._load_native(run, tmp_path, "fpm")


def test_public_rust_query_preserves_real_named_rows_and_original_leaf_binding(calibration, tmp_path):
    import shutil
    from importlib.resources import files

    import pyarrow.parquet as pq

    from aisimulate.sdk.engine import EngineHandle

    parent, children, _ = calibration
    systems = tmp_path / "systems"
    data = systems / "data/gb300/vllm/0.30.0"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    table = data / serving.BASENAME
    shards.publish_calibration(parent, children, table, lookup_contract=serving.LOOKUP_CONTRACT)
    bound = shards.bind_calibration([table], parent, children)
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
    expected = sum(r["latency"] for r in rows if r["prefix"] == 128)
    for prefix in (128, 192, 256):
        assert engine.predict_decode_latency(1, prefix, 2) == pytest.approx(expected)
        assert engine.last_provenance() is None
    audit = engine.glm53flash_lookup_audit("generation", 1, 1, 192)
    assert len(audit["operations"]) == 278
    evidence = {r["evidence_sha256"] for r in bound["children"]}
    assert all(
        {e["measurement"]["evidence_sha256"] for e in row["endpoints"]} == evidence for row in audit["operations"]
    )
    with pytest.raises(ValueError):
        engine.predict_decode_latency(1, 512, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", True),
        ("batch_size", 33),
        ("total_prefill_tokens", 8193),
        ("total_prefill_tokens", 0.0),
        ("total_kv_read_tokens", -1),
    ],
)
def test_family_declaration_rejects_malformed_or_out_of_bound_geometry(field, value):
    point = {"batch_size": 1, "total_prefill_tokens": 4, "total_kv_read_tokens": 128}
    point[field] = value
    with pytest.raises(ValueError):
        partition.family_for("prefill", point)


def test_original_repeat_contract_is_not_overwritten(tmp_path):
    parent, children = original(tmp_path)
    parent["options"]["measurement_repeats"] = 9
    with pytest.raises(ValueError, match="five warmup/ten retained"):
        partition.build_partition(
            parent,
            children,
            runtime(tmp_path),
            parent_cell_id=parent["cells"][0]["cell_id"],
            role="calibration",
            campaign_id="TEST_ONLY",
        )


def test_real_serving_reader_does_not_let_custom_leaf_silently_use_legacy_contract(tmp_path, monkeypatch):
    from .test_glm53flash_vllm_serving_export import complete_full_files

    run, root = complete_full_files(tmp_path, monkeypatch)
    assert serving.read_serving_run(root, run)["policy"]["schema_version"] == 3
    run["plan"]["schema_name"] = partition.LEAF_SCHEMA
    with pytest.raises(ValueError, match="cannot bypass"):
        serving.read_serving_run(root, run)


def test_returned_partition_cannot_mutate_the_source_bound_family_rule(tmp_path):
    _, run = setup(tmp_path)
    changed = run["observation_partition"]
    changed["family_rule"]["max_capture_tokens"] = 8192
    assert (
        partition.family_for("prefill", {"batch_size": 1, "total_prefill_tokens": 4096, "total_kv_read_tokens": 0})
        == "NONE"
    )
    with pytest.raises(ValueError, match="complete original"):
        partition.validate_partition(changed, run["plan"])
