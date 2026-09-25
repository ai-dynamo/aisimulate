# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY original named call identities through export and real Rust queries."""

import copy
import json
import shutil
from importlib.resources import files

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.engine import EngineHandle, build_engine_spec_json
from aisimulate_core.sdk.models import get_model
from collector import glm53flash_graph_export as graph
from collector import glm53flash_graph_policy as sg
from collector import glm53flash_vllm_graph_policy as vllm
from collector.glm53flash_contract import (
    BACKENDS,
    CHECKPOINTS,
    build_model_manifest,
    canonical_json,
    runtime_source_pins,
    sha256_json,
)

pytestmark = pytest.mark.unit
SHA = "a" * 64


def authored_proof(backend="sglang", checkpoint="fp8", tp=2):
    version, revision = BACKENDS[backend]
    policy = dict(
        schema_version=1 if backend == "sglang" else 2,
        backend=backend,
        backend_version=version,
        backend_revision=revision,
        checkpoint_format=checkpoint,
        checkpoint_revision=CHECKPOINTS[checkpoint][1],
        tp_size=tp,
        phase="generation",
        runtime_mode="FULL",
        capture_sizes=[1, 2, 4],
        disable_padding=False,
        captured_req_width=1,
        native_flags=dict.fromkeys(sg.DIRECT_FLAGS + sg.EXTRA_FLAGS if backend == "sglang" else vllm.FLAGS, False),
        source_pins=sg.SOURCE_PINS if backend == "sglang" else vllm.SOURCE_PINS,
        source_sha256=sha256_json(runtime_source_pins(backend, version)),
        config_sha256=SHA,
        runtime_digest="sha256:" + SHA,
        resolved_config_sha256=SHA,
        native_policy_receipt_sha256=SHA,
        state_layout_sha256={str(r): SHA for r in range(tp)},
        capture_registry_sha256={str(r): [SHA] * 3 for r in range(tp)},
    )
    manifest = build_model_manifest(backend, checkpoint, tp)
    entries = manifest["phases"]["generation"] + manifest["runtime_operations"]["generation"]
    forwards = {}
    for point, prefix in enumerate((128, 136), 1):
        for rep in range(15):
            ranks = {}
            for rank in range(tp):
                binding = {}
                for entry in entries:
                    shape = json.loads(entry["geometry"])
                    latency = (
                        0.8
                        if entry["name"] == "embedding_allreduce"
                        else 0.004
                        if shape.get("role") == "allreduce"
                        else 0.002
                        if entry["component"] == "runtime"
                        else 0.001
                    )
                    binding[entry["name"]] = dict(latency=latency * point, dispatch=SHA, activity_count=1)
                ranks[rank] = dict(
                    whole_forward_gpu_ms=10.0 + rank,
                    forward_id=f"TEST_ONLY-{point}-{rep}-{rank}",
                    invocation=point * 15 + rep,
                    sampling_role="warmup" if rep < 5 else "measurement",
                    batch_size=1,
                    prefix_lengths=[prefix],
                    num_padded_tokens=1,
                    binding=binding,
                )
            forwards[point, rep] = ranks
    return dict(policy=policy, manifest=manifest, forwards=forwards)


@pytest.fixture
def named(tmp_path, request):
    backend, checkpoint, tp = getattr(request, "param", ("sglang", "fp8", 2))
    proof = authored_proof(backend, checkpoint, tp)
    systems = tmp_path / "systems"
    data = systems / "data" / "gb300" / backend / BACKENDS[backend][0]
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    model_path = CHECKPOINTS[checkpoint][0]
    model = get_model(model_path, ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1), backend)
    rows, _ = graph.aggregate_graph(proof, evidence_sha256="b" * 64, lookup_contract=graph.NAMED_CONTRACT)
    return dict(
        rows=rows,
        path=data / graph.BASENAME,
        systems=systems,
        proof=proof,
        model=model,
        model_path=model_path,
        backend=backend,
        tp=tp,
    )


def engine(case, mutate_spec=None):
    import aisimulate_core

    pq.write_table(pa.Table.from_pylist(case["rows"]), case["path"])
    spec = json.loads(
        build_engine_spec_json(
            case["model"],
            model_path=case["model_path"],
            system="gb300",
            backend=case["backend"],
            backend_version=BACKENDS[case["backend"]][0],
            kv_block_size=None,
            systems_path=str(case["systems"]),
            nextn=0,
            database_mode="SILICON",
            shared_layer=False,
            strict_provenance=True,
        )
    )
    if mutate_spec:
        mutate_spec(spec)
    return EngineHandle(
        aisimulate_core.engine_spec_bincode_from_json(json.dumps(spec)), systems_path=str(case["systems"])
    )


@pytest.mark.parametrize(
    "named", [(b, f, t) for b in ("sglang", "vllm") for f, t in (("fp8", 2), ("nvfp4", 4))], indirect=True
)
def test_real_public_exact_and_bracket_keep_embedding_and90_layers(named):
    count = 367 if named["backend"] == "sglang" else 278
    expected = 0.8 + 90 * 0.004 + (count - 92) * 0.001 + 0.002
    predictor = engine(named)
    assert predictor.predict_decode_latency(1, 128, 2) == pytest.approx(expected)
    assert predictor.predict_decode_latency(1, 132, 2) == pytest.approx(1.5 * expected)
    audit = predictor.glm53flash_lookup_audit("generation", 1, 1, 132)
    assert audit["lookup_contract"] == graph.NAMED_CONTRACT
    assert len(audit["operations"]) == count
    assert sum(row["latency_ms"] for row in audit["operations"]) == pytest.approx(1.5 * expected)
    by_name = {r["operation_name"]: r for r in audit["operations"]}
    assert by_name["embedding_allreduce"]["latency_ms"] == pytest.approx(1.2)
    for i in range(45):
        for label in ("attention", "ffn"):
            assert by_name[f"{label}_allreduce_{i}"]["latency_ms"] == pytest.approx(0.006)
    for row in audit["operations"]:
        assert [(e["prefix"], e["weight"]) for e in row["endpoints"]] == [(128, 0.5), (136, 0.5)]
        assert {e["evidence_sha256"] for e in row["endpoints"]} == {"b" * 64}
    assert predictor.last_provenance() is None
    for batch, past in [(2, 132), (1, 124), (1, 140)]:
        with pytest.raises(ValueError):
            predictor.predict_decode_latency(batch, past, 2)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_embedding",
        "missing_layer",
        "duplicate",
        "name",
        "role",
        "layer_kind",
        "sample",
        "contract",
        "unpaired",
        "mixed",
        "policy",
        "partial_point",
    ],
)
def test_named_malformed_tables_never_fall_back(named, defect):
    rows = named["rows"]
    target = next(r for r in rows if r["operation_name"] == "embedding_allreduce")
    if defect == "missing_embedding":
        rows.remove(target)
    elif defect == "missing_layer":
        rows[:] = [r for r in rows if r["operation_name"] != "ffn_allreduce_44"]
    elif defect == "duplicate":
        rows.append(copy.deepcopy(target))
    elif defect == "name":
        target["operation_name"] = "unobserved_collective"
    elif defect == "role":
        target["operation_name"] = "logits"
    elif defect == "layer_kind":
        a = next(r for r in rows if r["operation_name"] == "attention_0")
        a["operation_name"] = "attention_3"
    elif defect == "sample":
        target["sample_count"] = 11
    elif defect == "contract":
        target["graph_lookup_contract"] = "unknown"
    elif defect == "unpaired":
        for row in rows:
            row.pop("graph_lookup_contract")
    elif defect == "mixed":
        target["graph_lookup_contract"] = target["operation_name"] = None
    elif defect == "policy":
        policy = json.loads(target["graph_policy"])
        policy["capture_sizes"] = [1, 2, 4, 8]
        target["graph_policy"] = canonical_json(policy)
        target["graph_policy_sha256"] = sha256_json(policy)
    elif defect == "partial_point":
        target["prefix"] = 144
    with pytest.raises(ValueError):
        engine(named).predict_decode_latency(1, 128, 2)


@pytest.mark.parametrize("defect", ["empty", "missing", "duplicate", "rename", "unknown_model_partial"])
def test_named_compiled_generation_must_remain_complete(named, defect):
    def mutate(spec):
        if defect == "empty":
            spec["generation_ops"] = []
        elif defect == "missing":
            spec["generation_ops"].pop(0)
        elif defect == "duplicate":
            spec["generation_ops"].append(copy.deepcopy(spec["generation_ops"][0]))
        elif defect == "unknown_model_partial":
            spec["engine"]["model_name"] = "TEST_ONLY_alias"
            spec["generation_ops"].pop(0)
        else:
            next(iter(spec["generation_ops"][1].values()))["name"] = "missing"

    with pytest.raises(ValueError):
        engine(named, mutate)


@pytest.mark.parametrize("named", [("sglang", "fp8", 2), ("vllm", "fp8", 2)], indirect=True)
def test_unopted_graph_table_retains_original_pooling(named):
    named["rows"], _ = graph.aggregate_graph(named["proof"], evidence_sha256="b" * 64)
    predictor = engine(named)
    count = 367 if named["backend"] == "sglang" else 278
    expected = 91 * 0.004 + (count - 92) * 0.001 + 0.002
    assert predictor.predict_decode_latency(1, 132, 2) == pytest.approx(1.5 * expected)
    with pytest.raises(ValueError):
        predictor.glm53flash_lookup_audit("generation", 1, 1, 132)


def test_named_and_legacy_deployments_select_their_own_contract(named):
    other = authored_proof("sglang", "nvfp4", 2)
    legacy, _ = graph.aggregate_graph(other, evidence_sha256="c" * 64)
    # Arrow takes columns from the first rows: nullable metadata identifies legacy rows.
    for row in legacy:
        row.update(graph_lookup_contract=None, operation_name=None)
    named["rows"] += legacy
    first = engine(named)
    assert first.glm53flash_lookup_audit("generation", 1, 1, 128)["lookup_contract"] == graph.NAMED_CONTRACT
    named["model_path"] = CHECKPOINTS["nvfp4"][0]
    named["model"] = get_model(named["model_path"], ModelConfig(tp_size=2, moe_tp_size=2, moe_ep_size=1), "sglang")
    second = engine(named)
    assert second.predict_decode_latency(1, 128, 2) == pytest.approx(0.641)
    with pytest.raises(ValueError, match="serving"):
        second.glm53flash_lookup_audit("generation", 1, 1, 128)


@pytest.mark.parametrize("defect", [None, "missing", "contract", "run", "source", "table"])
def test_public_prediction_requires_bound_named_table_and_retains_rust_audit(named, monkeypatch, defect):
    from collector import glm53flash_validation as native
    from collector.glm53flash_jsonl import file_sha256

    engine(named)
    policy = named["proof"]["policy"]
    root = named["path"].parent
    evidence = root / "graph-calibration-evidence.json"
    evidence.write_text(canonical_json({"source_plan_sha256": SHA}))
    calibration = dict(graph_policy=policy, evidence_root=str(root), runtime_run_id="TEST_ONLY-native-calibration")
    frozen = dict(
        key=["sglang", "fp8", 2, "decode"],
        role="holdout",
        spec=dict(ops_execution_mode="native_full_graph"),
        points=[
            dict(benchmark_id=1, point_type="decode", batch_size=1, total_prefill_tokens=0, total_kv_read_tokens=132)
        ],
    )
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
    holdout = {
        "graph_policy": {
            "native_snapshot": {k: policy[k] for k in fields},
            "provenance": {
                k: policy[k] for k in ("source_sha256", "config_sha256", "runtime_digest", "checkpoint_revision")
            },
        }
    }
    # This test isolates the installed public analysis adapter; the separate original-trace
    # tests verify native loading/policy/control. These are authored TEST_ONLY receipts.
    monkeypatch.setattr(native, "load_native", lambda *_: holdout)
    monkeypatch.setattr(graph, "_same_execution_policy", lambda *_: None)
    binding = dict(
        lookup_contract=graph.NAMED_CONTRACT,
        graph_policy_sha256=sha256_json(policy),
        native_runtime_run_id=calibration["runtime_run_id"],
        source_plan_sha256=SHA,
        evidence_sha256=file_sha256(evidence),
        tables=[dict(path=str(named["path"]), sha256=file_sha256(named["path"]))],
    )
    if defect == "missing":
        binding = None
    elif defect == "contract":
        binding["lookup_contract"] = "wrong"
    elif defect == "run":
        binding["native_runtime_run_id"] = "another-run"
    elif defect == "source":
        binding["source_plan_sha256"] = "e" * 64
    elif defect == "table":
        binding["tables"][0]["sha256"] = "f" * 64
    config = dict(
        model=named["model_path"],
        system="gb300",
        backend="sglang",
        backend_version="0.5.20",
        worker_type="aggregated",
        tp=2,
        pp=1,
        attention_dp=1,
        moe_tp_size=2,
        moe_ep_size=1,
        kvcache_quant_mode="fp8",
        estimation_mode="op_level",
        database_mode="SILICON",
        systems_paths=[str(named["systems"])],
        fallback_policy="deny",
        strict_provenance=True,
        enable_shared_layer=False,
    )
    if defect:
        with pytest.raises(ValueError, match="binding"):
            graph.predict_homogeneous(frozen, root, config, calibration, calibration_binding=binding)
    else:
        result = graph.predict_homogeneous(frozen, root, config, calibration, calibration_binding=binding)
        assert result["rows"][1]["prediction_ms"] == pytest.approx(1.5 * 1.437)
        assert len(result["prediction_evidence"][1]["operations"]) == 367
