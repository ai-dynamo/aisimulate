# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY independent named graph members; real reducer and Rust consumer."""

import copy
import json
import shutil
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from collector import glm53flash_graph_export as export
from collector import glm53flash_graph_group as group
from collector import glm53flash_graph_shards as shards
from collector.glm53flash_contract import canonical_json, sha256_json

from .test_glm53flash_graph_shards import campaign  # noqa: F401

pytestmark = pytest.mark.unit
SHA = "a" * 64


def layout(rank=0):
    dtypes = {
        "kda_conv": "torch.bfloat16",
        "kda_temporal": "torch.float32",
        "mla_latent": "torch.uint8",
        "pooled_index_packed": "torch.uint8",
        "index_tail_key": "torch.bfloat16",
        "index_tail_score": "torch.bfloat16",
    }
    return {
        "admitted": True,
        "tp_rank": rank,
        "logical_kv_dtype": "torch.float8_e4m3fn",
        "physical_kv_dtype": "torch.uint8",
        "pooled_index_layout": "packed_fp8_keys_and_fp32_scales",
        "pool_class": "TEST_ONLY_HybridPool",
        "full_pool_class": "TEST_ONLY_DSACache",
        "groups": {
            name: [{"dtype": dtype, "shape": [8, 4], "stride": [4, 1], "device": f"cuda:{rank}", "nbytes": 32}]
            for name, dtype in dtypes.items()
        },
        "hardware": {
            "schema": "glm53flash_gpu_identity_v1",
            "name": "TEST_ONLY NVIDIA GB300",
            "compute_capability": [10, 3],
            "cuda_device_index": rank,
            "total_memory_bytes": 200_000_000_000,
            "uuid": f"TEST_ONLY-device-{rank}",
        },
    }


def compatibility_inputs():
    return (
        {"capture_sizes": [1, 2, 4]},
        {"normalization": "resolved_server_args_except_random_seed_v1", "sha256": SHA},
        {rank: layout(rank) for rank in range(2)},
        {
            rank: {
                "TEST_ONLY": {
                    "native_shape_key": {"size": 1},
                    "capture_scope": "model_with_logits",
                    "operations": [{"name": "TEST_ONLY"}],
                    "native_api_libraries": {
                        "cudart": {"sha256": SHA, "runtime_version": 13000, "abi": "TEST_ONLY"},
                        "cupti": {"sha256": SHA},
                    },
                    "calls": [
                        {
                            "name": "TEST_ONLY",
                            "source": "TEST_ONLY.forward",
                            "index": 0,
                            "completed": True,
                            "owned_node_ids": [100 + rank],
                        }
                    ],
                }
            }
            for rank in range(2)
        },
    )


def test_compatibility_keeps_layout_and_source_but_preserves_per_process_provenance():
    inputs = compatibility_inputs()
    before = copy.deepcopy(inputs)
    expected = group.compatibility(*inputs)
    changed = copy.deepcopy(inputs)
    changed[2][0]["hardware"]["uuid"] = "TEST_ONLY-other-allocation"
    changed[3][0]["TEST_ONLY"]["calls"][0]["owned_node_ids"] = [999]
    assert group.compatibility(*changed) == expected
    assert inputs == before
    changed[2][0]["groups"]["kda_conv"][0]["stride"] = [8, 1]
    assert group.compatibility(*changed)["state_layout_sha256"] != expected["state_layout_sha256"]
    changed = copy.deepcopy(inputs)
    changed[3][0]["TEST_ONLY"]["calls"][0]["source"] = "TEST_ONLY.different_native_forward"
    assert group.compatibility(*changed)["source_ownership_sha256"] != expected["source_ownership_sha256"]
    changed[2][0]["groups"]["kda_conv"][0]["dtype"] = "torch.float32"
    with pytest.raises(ValueError, match="dtype"):
        group.compatibility(*changed)


@pytest.fixture
def grouped(request):
    parent, children, proofs, natives = request.getfixturevalue("campaign")
    parent["spec"]["ops_graph_group_contract"] = group.CONTRACT
    for index, (_, native) in enumerate(children):
        proof = proofs[Path(native["evidence_root"])]
        # Distinct original receipts, without changing source/runtime declarations.
        policy = proof["policy"]
        policy["resolved_config_sha256"] = sha256_json(["TEST_ONLY seed", index])
        policy["capture_registry_sha256"] = {
            str(r): [sha256_json(["TEST_ONLY capture", index, r, b]) for b in (1, 2, 4)] for r in range(2)
        }
        policy["state_layout_sha256"] = {str(r): sha256_json(["TEST_ONLY original GPU", index, r]) for r in range(2)}
        proof["graph_group_compatibility"] = {
            "execution_policy": proof["execution_policy"],
            "native_snapshot_sha256": SHA,
            "state_layout_sha256": "b" * 64,
            "source_ownership_sha256": "c" * 64,
        }
        native["graph_group_compatibility"] = proof["graph_group_compatibility"]
    return parent, children, proofs, natives


def publish(grouped, tmp_path):
    parent, children, _, _ = grouped
    systems = tmp_path / "systems"
    path = systems / "data/gb300/sglang/0.5.20" / export.BASENAME
    path.parent.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    shards.publish_calibration(parent, children, path, lookup_contract=group.CONTRACT)
    return systems, path, shards.bind_calibration([path], parent, children)


def engine(systems):
    from aisimulate_core.sdk.engine import EngineHandle

    return EngineHandle.compile(
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


def test_group_publication_preserves_every_original_policy_and_binding(grouped, tmp_path):
    parent, children, _, _ = grouped
    originals = {
        native["evidence_root"]: (Path(native["evidence_root"]) / "graph-calibration-evidence.json").read_bytes()
        for _, native in children
    }
    _, path, binding = publish(grouped, tmp_path)
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 734
    assert len({row["graph_policy_sha256"] for row in rows}) == 2
    assert len({row["graph_group_sha256"] for row in rows}) == 1
    assert binding["group_contract"] == group.CONTRACT and "graph_policy_sha256" not in binding
    assert set(binding["graph_group"]["members"]) == {run["cell"]["cell_id"] for run, _ in children}
    for run, native in children:
        member = binding["graph_group"]["members"][run["cell"]["cell_id"]]
        assert member["graph_policy"] == native["graph_policy"]
        assert (Path(native["evidence_root"]) / "graph-calibration-evidence.json").read_bytes() == originals[
            native["evidence_root"]
        ]
    native_group = {**children[0][1], "_children": {run["cell"]["cell_id"]: native for run, native in children}}
    shards.validate_prediction_binding(native_group, binding)
    damaged = copy.deepcopy(binding)
    cid = next(iter(damaged["graph_group"]["members"]))
    damaged["graph_group"]["members"][cid]["graph_policy"]["resolved_config_sha256"] = "f" * 64
    damaged["graph_group_sha256"] = sha256_json(damaged["graph_group"])
    damaged["calibration_group_sha256"] = sha256_json(damaged["ownership"])
    with pytest.raises(ValueError, match="member policy"):
        shards.validate_prediction_binding(native_group, damaged)


@pytest.mark.parametrize("field", ["native_snapshot_sha256", "state_layout_sha256", "source_ownership_sha256"])
def test_group_rejects_changed_actual_compatibility(grouped, field):
    parent, children, proofs, _ = grouped
    proof = proofs[Path(children[1][1]["evidence_root"])]
    proof["graph_group_compatibility"][field] = "e" * 64
    with pytest.raises(ValueError, match="actual native snapshot"):
        shards.calibration_rows(parent, children, lookup_contract=group.CONTRACT)


def test_group_requires_analysis_opt_in_and_old_exact_policy_guard(grouped):
    parent, children, _, _ = grouped
    with pytest.raises(ValueError, match="matching explicit"):
        shards.calibration_rows(parent, children, lookup_contract=export.NAMED_CONTRACT)
    del parent["spec"]["ops_graph_group_contract"]
    with pytest.raises(ValueError, match="execution/capture policy"):
        shards.calibration_rows(parent, children, lookup_contract=export.NAMED_CONTRACT)
    with pytest.raises(ValueError, match="matching explicit"):
        shards.calibration_rows(parent, children, lookup_contract=group.CONTRACT)


def test_real_rust_group_exact_cross_member_bracket_and_original_audit(grouped, tmp_path):
    systems, _, binding = publish(grouped, tmp_path)
    predictor = engine(systems)
    expected = 0.8 + 90 * 0.004 + 275 * 0.001 + 0.002
    for prefix, factor in ((128, 1), (132, 1.5), (136, 2)):
        assert predictor.predict_decode_latency(1, prefix, 2) == pytest.approx(expected * factor)
        audit = predictor.glm53flash_lookup_audit("generation", 1, 1, prefix)
        group.validate_audit(audit, binding["graph_group"])
        assert len(audit["operations"]) == 367
        assert sum(op["latency_ms"] for op in audit["operations"]) == pytest.approx(expected * factor)
        assert "graph_policy_sha256" not in audit
        assert predictor.last_provenance() is None
        for operation in audit["operations"]:
            assert len(operation["endpoints"]) == (2 if prefix == 132 else 1)
        if prefix == 132:
            expected_members = set(binding["graph_group"]["members"])
            assert all(
                {e["graph_member_id"] for e in op["endpoints"]} == expected_members for op in audit["operations"]
            )
    for prefix in (0, 124, 140):
        with pytest.raises(ValueError):
            predictor.predict_decode_latency(1, prefix, 2)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_unit",
        "missing_point",
        "duplicate",
        "unknown_member",
        "wrong_evidence",
        "mixed_contract",
        "changed_policy",
        "changed_group",
        "missing_metadata",
        "declared_extra_member",
        "wrong_original_id",
        "changed_runtime",
    ],
)
def test_real_rust_group_rejects_incomplete_or_changed_originals(grouped, tmp_path, defect):
    systems, path, _ = publish(grouped, tmp_path)
    rows = pq.read_table(path).to_pylist()
    if defect == "missing_unit":
        rows.pop()
    elif defect == "missing_point":
        rows = [row for row in rows if row["prefix"] == 128]
    elif defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "unknown_member":
        rows[0]["graph_member_id"] = "TEST_ONLY unknown"
    elif defect == "wrong_evidence":
        rows[0]["evidence_sha256"] = "f" * 64
    elif defect == "mixed_contract":
        for key in group.COLUMNS:
            rows[0][key] = None
    elif defect == "changed_policy":
        policy = json.loads(rows[0]["graph_policy"])
        policy["resolved_config_sha256"] = "f" * 64
        rows[0]["graph_policy"], rows[0]["graph_policy_sha256"] = canonical_json(policy), sha256_json(policy)
    elif defect == "changed_group":
        rows[0]["graph_group_sha256"] = "f" * 64
    elif defect == "missing_metadata":
        for row in rows:
            del row["graph_member_id"]
    else:
        value = json.loads(rows[0]["graph_group"])
        member = next(iter(value["members"].values()))
        if defect == "declared_extra_member":
            value["members"]["TEST_ONLY extra"] = copy.deepcopy(member)
        elif defect == "wrong_original_id":
            member["points"][0]["original_point_id"] = 99
        else:
            member["graph_policy"]["runtime_digest"] = "sha256:" + "f" * 64
            member["graph_policy_sha256"] = sha256_json(member["graph_policy"])
        group.annotate(rows, value)
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError):
        engine(systems).predict_decode_latency(1, 132, 2)


def test_group_never_borrows_a_different_endpoint_for_one_missing_dispatch(grouped, tmp_path):
    systems, path, _ = publish(grouped, tmp_path)
    rows = pq.read_table(path).to_pylist()
    changed = next(row for row in rows if row["prefix"] == 136 and row["operation_name"] == "embedding_allreduce")
    changed["dispatch_fingerprint"] = "e" * 64
    pq.write_table(pa.Table.from_pylist(rows), path)
    predictor = engine(systems)
    assert predictor.predict_decode_latency(1, 128, 2) > 0
    with pytest.raises(ValueError, match="complete same-pad"):
        predictor.predict_decode_latency(1, 132, 2)


def test_parent_group_loader_keeps_real_members_without_aggregate_policy(grouped, tmp_path, monkeypatch):
    from collector import glm53flash_validation as reader
    from collector.fpm_forward import glm53flash_validation as acceptance

    parent, children, _, _ = grouped
    admitted = {run["cell"]["cell_id"]: native for run, native in children}
    seen = []

    def load(run, base):
        assert run["spec"]["ops_graph_group_contract"] == group.CONTRACT
        seen.append(run["cell"]["cell_id"])
        return admitted[run["cell"]["cell_id"]]

    monkeypatch.setattr(reader, "load_native", load)
    native = acceptance._load_native(parent, tmp_path, "ops")
    assert set(seen) == set(admitted)
    assert "graph_policy" not in native and "runtime_run_id" not in native and "evidence_root" not in native
    assert native["graph_group_contract"] == group.CONTRACT
    assert native["_children"] == admitted
    _, _, binding = publish(grouped, tmp_path)
    shards.validate_prediction_binding(native, binding)


def test_group_actual_public_leaf_prediction_and_tampered_endpoint_audit(grouped, tmp_path, monkeypatch):
    from collector import glm53flash_validation as reader

    from aisimulate_core.sdk.engine import EngineHandle

    parent, children, _, _ = grouped
    systems, _, binding = publish(grouped, tmp_path)
    native = {
        **children[0][1],
        "_children": {run["cell"]["cell_id"]: value for run, value in children},
        "graph_group_contract": group.CONTRACT,
    }
    del native["graph_policy"]
    policy = children[0][1]["graph_policy"]
    fields = (
        "backend",
        "backend_version",
        "backend_revision",
        "capture_sizes",
        "disable_padding",
        "captured_req_width",
        "native_flags",
        "source_pins",
    )
    holdout_native = {
        **children[0][1],
        "runtime_run_id": "TEST_ONLY independent holdout",
        "evidence_root": str(tmp_path / "independent-holdout"),
        "request_ids": {"TEST_ONLY holdout request"},
        "graph_policy": {
            "native_snapshot": {key: policy[key] for key in fields},
            "provenance": {
                key: policy[key] for key in ("source_sha256", "config_sha256", "runtime_digest", "checkpoint_revision")
            },
        },
        "graph_group_compatibility": {**native["graph_group_compatibility"], "source_ownership_sha256": None},
    }
    monkeypatch.setattr(reader, "load_native", lambda run, base: holdout_native)
    holdout = {
        **children[0][0],
        "role": "holdout",
        "points": [
            {
                "point_type": "decode",
                "benchmark_id": 1,
                "batch_size": 1,
                "total_prefill_tokens": 0,
                "total_kv_read_tokens": 132,
            }
        ],
    }
    config = {
        "model": "zai-org/GLM-5.3-Flash",
        "system": "gb300",
        "backend": "sglang",
        "backend_version": "0.5.20",
        "worker_type": "decode",
        "tp": 2,
        "moe_tp_size": 2,
        "moe_ep_size": 1,
        "systems_paths": [str(systems)],
        "database_mode": "SILICON",
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "strict_provenance": True,
    }
    result = export.predict_homogeneous(holdout, tmp_path, config, native, calibration_binding=binding)
    assert result["rows"][1]["prediction_ms"] == pytest.approx(1.437 * 1.5)
    group.validate_audit(result["prediction_evidence"][1], binding["graph_group"])
    original = EngineHandle.glm53flash_lookup_audit

    def changed(self, *args):
        audit = original(self, *args)
        audit["operations"][0]["endpoints"][0]["graph_policy_sha256"] = "f" * 64
        return audit

    monkeypatch.setattr(EngineHandle, "glm53flash_lookup_audit", changed)
    result = export.predict_homogeneous(holdout, tmp_path, config, native, calibration_binding=binding)
    assert "error" in result["rows"][1] and not result["prediction_evidence"]
    holdout_native["request_ids"] = children[0][1]["request_ids"]
    with pytest.raises(ValueError, match="reused"):
        export.predict_homogeneous(holdout, tmp_path, config, native, calibration_binding=binding)
    holdout_native["request_ids"] = {"TEST_ONLY holdout request"}
    holdout_native["graph_group_compatibility"]["state_layout_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="execution/state/padding"):
        export.predict_homogeneous(holdout, tmp_path, config, native, calibration_binding=binding)


def test_real_rust_group_uses_nearest_complete_pair(grouped, tmp_path):
    systems, path, _ = publish(grouped, tmp_path)
    rows = pq.read_table(path).to_pylist()
    manifest = json.loads(rows[0]["graph_group"])
    for member in manifest["members"].values():
        member["points"] = []
    original = copy.deepcopy(rows)
    rows = []
    for original_id, (prefix, source_prefix, factor) in enumerate(
        ((120, 128, 0.5), (128, 128, 1), (136, 136, 2), (144, 136, 3)), 1
    ):
        selected = [copy.deepcopy(row) for row in original if row["prefix"] == source_prefix]
        cid = selected[0]["graph_member_id"]
        points = manifest["members"][cid]["points"]
        points.append(
            {
                "batch_size": 1,
                "prefix": prefix,
                "padded_batch_size": 1,
                "native_benchmark_id": len(points) + 1,
                "original_point_id": original_id,
            }
        )
        for row in selected:
            row["prefix"] = prefix
            row["latency"] *= factor / (1 if source_prefix == 128 else 2)
        rows.extend(selected)
    group.annotate(rows, manifest)
    pq.write_table(pa.Table.from_pylist(rows), path)
    predictor = engine(systems)
    assert predictor.predict_decode_latency(1, 132, 2) == pytest.approx(1.437 * 1.5)
    audit = predictor.glm53flash_lookup_audit("generation", 1, 1, 132)
    assert all({p["prefix"] for p in op["endpoints"]} == {128, 136} for op in audit["operations"])


def test_real_rust_group_and_different_legacy_deployment_coexist(grouped, tmp_path):
    from aisimulate_core.sdk.engine import EngineHandle

    from ..sdk.test_glm53flash_graph_named_consumer import authored_proof

    systems, path, _ = publish(grouped, tmp_path)
    rows = pq.read_table(path).to_pylist()
    legacy, _ = export.aggregate_graph(authored_proof("sglang", "nvfp4", 2), evidence_sha256="d" * 64)
    for row in legacy:
        row.update(dict.fromkeys((*group.COLUMNS, "operation_name", "graph_lookup_contract")))
    pq.write_table(pa.Table.from_pylist(rows + legacy), path)
    assert engine(systems).predict_decode_latency(1, 132, 2) == pytest.approx(1.437 * 1.5)
    predictor = EngineHandle.compile(
        "nvidia/GLM-5.3-Flash-NVFP4",
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
    assert predictor.predict_decode_latency(1, 128, 2) > 0
    with pytest.raises(ValueError):
        predictor.glm53flash_lookup_audit("generation", 1, 1, 128)
