# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY authored graph fixtures; no GPU observations or qualification."""

import copy
import hashlib
import json

import pytest

from collector import glm53flash_graph_export as common
from collector import glm53flash_vllm_graph_export as graph
from collector.glm53flash_contract import BACKENDS, CHECKPOINTS, _runtime_contract, canonical_json, sha256_json
from collector.glm53flash_graph_callbacks import resolve_registry
from collector.glm53flash_graph_nodes import bind_replay_kernels, bind_vllm_execution_activity, trace_forward_identity
from collector.glm53flash_jsonl import file_sha256, iter_records
from collector.glm53flash_vllm_graph_policy import full_policy_fields, select_descriptor

from .test_glm53flash_ops_evidence import put, put_lines
from .test_glm53flash_vllm_graph_execution import trace
from .test_glm53flash_vllm_graph_policy import snapshot

pytestmark = pytest.mark.unit


def dispatch_row(policy, *, context=False, query=1, batch=1, stage="measure"):
    shape = select_descriptor(policy, batch=batch, query=query, is_context=context)
    return {
        "batch_size": batch,
        "query_lengths": [query] * batch,
        "prefix_lengths": [128] * batch,
        "phase": "context" if context else "generation",
        "stage": stage,
        "native_dispatch": {
            "descriptor": shape,
            "policy_sha256": sha256_json(policy),
            "physical_tokens": shape["num_tokens"],
            "physical_requests": shape["num_reqs"] or batch,
        },
        "runtime_mode": shape["cg_mode"],
        "used_cuda_graph": shape["cg_mode"] != "NONE",
        "num_padded_tokens": shape["num_tokens"],
        "native_graph_replay_completed": shape["cg_mode"] == "FULL",
    }


def fixture(tmp_path, monkeypatch, role="calibration"):
    root = tmp_path
    version = BACKENDS["vllm"][0]
    _, pins = _runtime_contract("vllm", version)
    provenance = dict(
        backend="vllm",
        backend_version=version,
        backend_revision=BACKENDS["vllm"][1],
        checkpoint_revision=CHECKPOINTS["fp8"][1],
        config_sha256="c" * 64,
        source_sha256=hashlib.sha256(pins.encode()).hexdigest(),
        runtime_digest="sha256:" + "d" * 64,
    )
    ops = [
        {"name": name, "component": component, "geometry": canonical_json({"backend": "vllm", "role": name})}
        for name, component in [("attention_0", "attention"), ("zero", "mhc"), ("logits", "primitive")]
    ]
    setup = {"name": "native_graph_setup", "component": "runtime", "geometry": "{}"}
    manifest = {"phases": {"generation": ops}, "runtime_operations": {"generation": [setup]}}
    monkeypatch.setattr(graph, "build_model_manifest", lambda *_: copy.deepcopy(manifest))
    put(root / "manifest.json", manifest)
    put(root / "provenance.json", provenance)
    put(
        root / "resolved-config-node0.json",
        {"config": {"engine_args": {"enforce_eager": False, "seed": 7, "max_num_seqs": 32}}},
    )
    run = {
        "key": ("vllm", "fp8", 2, "decode"),
        "role": role,
        "corpus": "a" * 64,
        "plan": {"sha256": "b" * 64},
        "spec": {"ops_execution_mode": "native_full_graph"},
    }
    for rank in range(2):
        policy = snapshot()
        policy.update(tp_size=2, tp_rank=rank)
        put(root / f"vllm-graph-policy-rank-{rank}.json", policy)
        put(root / f"state-layout-rank-{rank}.json", {"TEST_ONLY": rank})
        put(root / f"vllm-graph-inventory-rank-{rank}.json", {"source_pins": graph.CAPTURE_PINS})
        captures = {}
        for index, shape in enumerate(policy["full_graphs"]):
            source = {
                "tp_rank": rank,
                "provenance": provenance,
                "native_shape_key": shape,
                "capture_scope": "vllm_hidden_states",
                "uncaptured_operations": ["logits"],
                "graph_mutations": False,
                "operations": ops[:-1],
                "physical_padded_tokens": shape["num_tokens"],
                "native_api_libraries": {
                    "cupti": {"path": "/TEST_ONLY/libcupti.so", "sha256": graph.QUALIFIED_CUPTI_SHA256},
                    "cudart": {
                        "path": "/TEST_ONLY/libcudart.so",
                        "sha256": "a" * 64,
                        "runtime_version": 13000,
                        "abi": "CUDA13_capture7_edges5",
                    },
                },
                "graph_id": 4,
                "nodes": [{"node_id": 17, "node_type": 0, "name": "attention_0"}],
                "edges": [],
                "calls": [
                    {
                        "name": "attention_0",
                        "source": "TEST_ONLY.native_attention",
                        "completed": True,
                        "owned_node_ids": [17],
                    },
                    {"name": "zero", "source": "TEST_ONLY.native_empty", "completed": True, "owned_node_ids": []},
                ],
            }
            callbacks = {
                "callback_errors": [],
                "graph_mutations": False,
                "actual_graph_exec_id": 7,
                "callbacks": [
                    {
                        "kind": "graph_exec_created",
                        "graph_id": 4,
                        "graph_exec_id": 7,
                        "raw_fields": {"graph": 12, "graphExec": 39},
                    },
                    {
                        "kind": "node_cloned",
                        "original_node_id": 17,
                        "node_id": 70,
                        "node_type": 0,
                        "raw_fields": {"graph": 39, "originalGraph": 12, "node": 72, "originalNode": 73, "nodeType": 0},
                    },
                ],
            }
            source["graph_id"] += index * 10
            source["nodes"][0]["node_id"] += index * 100
            source["calls"][0]["owned_node_ids"] = [source["nodes"][0]["node_id"]]
            callbacks["actual_graph_exec_id"] += index * 10
            create, clone = callbacks["callbacks"]
            create["graph_id"] = source["graph_id"]
            create["graph_exec_id"] = callbacks["actual_graph_exec_id"]
            for field in ("graph", "graphExec"):
                create["raw_fields"][field] += index * 1000
            clone["original_node_id"] = source["nodes"][0]["node_id"]
            clone["node_id"] += index * 100
            for field in ("graph", "originalGraph", "node", "originalNode"):
                clone["raw_fields"][field] += index * 1000
            path = root / f"vllm-graph-clones-rank-{rank}-capture-0-{index}.json"
            put(path, callbacks)
            put(path.with_name(path.stem + "-source.json"), source)
            registry = resolve_registry(source, callbacks)
            registry["instantiation_receipt"] = {"file": path.name, "sha256": file_sha256(path)}
            cap_path = root / f"vllm-capture-rank-{rank}-{index}.json"
            put(cap_path, registry)
            captures[canonical_json(shape)] = (registry, cap_path)
        forwards = []
        graph_rows = []
        for repetition in range(15):
            row = {
                **provenance,
                **dispatch_row(policy),
                "tp_rank": rank,
                "invocation": repetition + 1,
                "forward_id": f"rank-{rank}/forward-{repetition + 1}",
                "benchmark_id": 1,
                "repetition": repetition,
                "sampling_role": "warmup" if repetition < 5 else "measurement",
                "dataset_role": "calibration",
                "request_set": "TEST_ONLY-run",
                "run_id": "TEST_ONLY-run-id",
                "corpus_sha256": "a" * 64,
                "request_ids": [f"request-{repetition}"],
                "gpu_completed": True,
                "ops_instrumented": role == "calibration",
                "whole_forward_gpu_ms": 2 if rank == 0 else 3,
                "whole_forward_boundary": graph.BOUNDARY,
                "requests": [{"input_tokens_sha256": "e" * 64, "prompt_token_ids": [1, 2], "sampled_token_id": 3}],
            }
            forwards.append(row)
            if role != "calibration":
                continue
            registry, cap_path = captures[canonical_json(row["native_dispatch"]["descriptor"])]
            _, events = trace()
            path = root / f"graph-profile-rank-{rank}-forward-{row['invocation']}.json"
            put(
                path,
                {
                    "traceEvents": events,
                    "aisim_native_forward": trace_forward_identity(row),
                    "aisim_native_execution": {
                        "backend": "vllm",
                        "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
                        "logits_source_sha256": graph.LOGITS_SOURCE_PIN,
                        "failed": False,
                    },
                },
            )
            binding = bind_vllm_execution_activity(bind_replay_kernels(registry, events, correlation=2), events)
            binding.update(trace_file=path.name, trace_sha256=file_sha256(path))
            graph_rows.append(
                {
                    **row,
                    "replay_nodes": binding,
                    "capture_registry_file": cap_path.name,
                    "capture_registry_sha256": file_sha256(cap_path),
                    "logits_source_sha256": graph.LOGITS_SOURCE_PIN,
                    "profiled": True,
                    "measurement_method": "native_cupti_graph_nodes_and_external_logits",
                }
            )
        put_lines(root / f"forward-rank-{rank}.jsonl", forwards)
        if role == "calibration":
            put_lines(root / f"graph-forward-rank-{rank}.jsonl", graph_rows)
    return run, root


def test_full_reader_joins_external_logits_and_real_zero_boundary(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    proof = common.read_graph_run(root, run)
    rows, selection = common.aggregate_graph(proof, evidence_sha256="f" * 64)
    assert proof["policy"]["schema_version"] == 2
    assert proof["policy"]["backend"] == "vllm"
    assert len(rows) == 4 and {row["sample_count"] for row in rows} == {10}
    assert {row["selected_rank"] for row in selection["forwards"]} == {1}
    assert sum(row["latency"] for row in rows) == pytest.approx(0.025)
    assert next(row for row in rows if row["component"] == "mhc")["activity_count"] == 0


@pytest.mark.parametrize("role", ["control", "holdout"])
def test_unprofiled_reader_does_not_require_calibration_graph_forward_file(tmp_path, monkeypatch, role):
    run, root = fixture(tmp_path, monkeypatch, role)
    proof = common.read_graph_run(root, run)
    assert len(proof["forwards"]) == 15
    assert all(row["binding"] is None for rows in proof["forwards"].values() for row in rows.values())


@pytest.mark.parametrize(
    "defect",
    [
        "reuse_trace",
        "copy_trace",
        "missing_setup",
        "wrong_logits",
        "wrong_padding",
        "partial_forward",
        "zero_not_called",
    ],
)
def test_false_or_incomplete_measurements_reject(tmp_path, monkeypatch, defect):
    run, root = fixture(tmp_path, monkeypatch)
    path = root / "graph-forward-rank-0.jsonl"
    rows = list(iter_records(path))
    if defect == "reuse_trace":
        rows[5]["replay_nodes"] = copy.deepcopy(rows[0]["replay_nodes"])
    elif defect == "copy_trace":
        target = root / rows[5]["replay_nodes"]["trace_file"]
        target.write_bytes((root / rows[0]["replay_nodes"]["trace_file"]).read_bytes())
        rows[5]["replay_nodes"]["trace_sha256"] = file_sha256(target)
    elif defect == "wrong_logits":
        rows[5]["logits_source_sha256"] = "0" * 64
    elif defect == "wrong_padding":
        rows[5]["num_padded_tokens"] = 2
    elif defect == "partial_forward":
        rows.pop()
    elif defect == "missing_setup":
        target = root / rows[5]["replay_nodes"]["trace_file"]
        value = json.loads(target.read_bytes())
        value["traceEvents"] = [event for event in value["traceEvents"] if event.get("cat") != "gpu_memset"]
        put(target, value)
        rows[5]["replay_nodes"]["trace_sha256"] = file_sha256(target)
    else:
        target = root / "vllm-graph-clones-rank-0-capture-0-0-source.json"
        value = json.loads(target.read_bytes())
        value["calls"].pop()
        put(target, value)
    put_lines(path, rows)
    with pytest.raises(ValueError):
        common.read_graph_run(root, run)


@pytest.mark.parametrize(
    "context,query,stage,mode",
    [(True, 8192, "seed", "NONE"), (True, 1, "seed", "PIECEWISE"), (False, 1, "measure", "FULL")],
)
def test_actual_seed_modes_are_checked_from_initialized_native_policy(context, query, stage, mode):
    policy = snapshot()
    row = dispatch_row(policy, context=context, query=query, stage=stage)
    assert graph.check_dispatch(row, policy)["cg_mode"] == mode
    row["num_padded_tokens"] += 1
    with pytest.raises(ValueError, match="dispatch"):
        graph.check_dispatch(row, policy)


def test_config_comparison_excludes_only_seed(tmp_path):
    path = tmp_path / "resolved-config-node0.json"
    put(path, {"config": {"engine_args": {"enforce_eager": False, "seed": 1, "memory": 0.8}}})
    a = graph.execution_policy(tmp_path)
    put(path, {"config": {"engine_args": {"enforce_eager": False, "seed": 2, "memory": 0.8}}})
    b = graph.execution_policy(tmp_path)
    graph.same_execution_policy(a, b)
    put(path, {"config": {"engine_args": {"enforce_eager": False, "seed": 2, "memory": 0.9}}})
    with pytest.raises(ValueError, match="changed"):
        graph.same_execution_policy(a, graph.execution_policy(tmp_path))


def test_full_region_excludes_piecewise_only_buckets():
    value = snapshot()
    value["max_num_reqs"] = 2
    value["full_graphs"] = value["full_graphs"][:2]
    value["capture_descriptors"]["FULL"] = list(reversed(value["full_graphs"]))
    for candidate in value["candidates"]:
        if candidate["num_tokens"] > 2:
            candidate["descriptors"] = candidate["descriptors"][1:]
    assert full_policy_fields(value)["capture_sizes"] == [1, 2]
    assert select_descriptor(value, batch=1, query=4, is_context=True)["cg_mode"] == "PIECEWISE"


def test_prediction_uses_calibration_full_policy_and_keeps_native_past_axis(tmp_path, monkeypatch):
    """The source inventory includes PW, but the public query uses only FULL."""
    from types import SimpleNamespace

    from aisimulate_core.sdk.engine import EngineHandle
    from collector import glm53flash_validation as native

    run, root = fixture(tmp_path, monkeypatch)
    calibration = common.read_graph_run(root, run)
    policy = calibration["policy"]
    holdout = {
        "graph_policy": {"native_snapshot": calibration["native_snapshot"], "provenance": calibration["provenance"]},
        **{key: calibration[key] for key in ("execution_policy", "_execution_policy")},
    }
    monkeypatch.setattr(native, "load_native", lambda *_: holdout)
    calls = []
    engine = SimpleNamespace(
        predict_decode_latency=lambda *args: calls.append(args) or 12.0, last_provenance=lambda: None
    )
    monkeypatch.setattr(EngineHandle, "compile", lambda *args, **kwargs: engine)
    config = {
        "model": CHECKPOINTS["fp8"][0],
        "system": "gb300",
        "backend": "vllm",
        "backend_version": BACKENDS["vllm"][0],
        "worker_type": "aggregated",
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp_size": 2,
        "moe_ep_size": 1,
        "kvcache_quant_mode": "fp8",
        "estimation_mode": "op_level",
        "database_mode": "SILICON",
        "systems_paths": [str(root)],
        "fallback_policy": "deny",
        "strict_provenance": True,
        "enable_shared_layer": False,
    }
    run["role"] = "holdout"
    run["points"] = [
        {
            "benchmark_id": 1,
            "point_type": "decode",
            "batch_size": 3,
            "total_prefill_tokens": 0,
            "total_kv_read_tokens": 399,
        }
    ]
    cal_native = {**holdout, "graph_policy": policy}
    result = common.predict_homogeneous(run, root, config, cal_native)
    assert calls == [(3, 133, 2)] and result["rows"] == {1: {"prediction_ms": 12.0}}
    changed = copy.deepcopy(cal_native)
    changed["graph_policy"]["capture_sizes"] = [1, 2]
    with pytest.raises(ValueError, match="changed its frozen"):
        common.predict_homogeneous(run, root, config, changed)


def test_real_model_graph_public_consumer_includes_external_logits_and_one_setup(tmp_path, monkeypatch):
    """Actual native public API plus complete model with explicitly TEST_ONLY costs."""
    import shutil
    from importlib.resources import files

    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models import get_model
    from collector import glm53flash_validation as native
    from collector.glm53flash_contract import operation_geometry

    run, root = fixture(tmp_path, monkeypatch)
    proof = common.read_graph_run(root, run)
    policy = proof["policy"]
    holdout = {
        "graph_policy": {"native_snapshot": proof["native_snapshot"], "provenance": proof["provenance"]},
        **{key: proof[key] for key in ("execution_policy", "_execution_policy")},
    }
    monkeypatch.setattr(native, "load_native", lambda *_: holdout)
    systems = tmp_path / "systems"
    data = systems / "data/gb300/vllm/0.30.0"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    model = get_model(CHECKPOINTS["fp8"][0], ModelConfig(tp_size=2, moe_tp_size=2, moe_ep_size=1), "vllm")
    components = {
        "Glm53Attention": "attention",
        "Glm53Mhc": "mhc",
        "Glm53Ffn": "ffn",
        "Glm53Primitive": "primitive",
        "Glm53Runtime": "runtime",
    }
    rows = {}
    for op in model.generation_ops:
        kind, shape = next(iter(json.loads(op._spec_json()).items()))
        geometry = operation_geometry(shape)
        rows[(kind, geometry)] = {
            "component": components[kind],
            "geometry": geometry,
            "batch_size": 1,
            "prefix": 128,
            "padded_batch_size": 1,
            "latency": 10.0 if kind == "Glm53Runtime" else 1.0,
            "activity_count": 1,
            "sample_count": 10,
            "dispatch_fingerprint": "a" * 64,
            "graph_policy": canonical_json(policy),
            "graph_policy_sha256": sha256_json(policy),
            "dataset_role": "calibration",
            "aggregation_policy": common.WHOLE_FORWARD_RANK,
            "rank_selection_sha256": "b" * 64,
            "evidence_sha256": "c" * 64,
            "measurement_scope": common.SCOPE,
        }
    pq.write_table(pa.Table.from_pylist(list(rows.values())), data / common.BASENAME)
    config = {
        "model": CHECKPOINTS["fp8"][0],
        "system": "gb300",
        "backend": "vllm",
        "backend_version": "0.30.0",
        "worker_type": "aggregated",
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp_size": 2,
        "moe_ep_size": 1,
        "kvcache_quant_mode": "fp8",
        "estimation_mode": "op_level",
        "database_mode": "SILICON",
        "systems_paths": [str(systems)],
        "fallback_policy": "deny",
        "strict_provenance": True,
        "enable_shared_layer": False,
    }
    run["role"] = "holdout"
    run["points"] = [
        {
            "benchmark_id": 1,
            "point_type": "decode",
            "batch_size": 1,
            "total_prefill_tokens": 0,
            "total_kv_read_tokens": 128,
        }
    ]
    result = common.predict_homogeneous(run, root, config, {**holdout, "graph_policy": policy})
    assert result["rows"] == {1: {"prediction_ms": len(model.generation_ops) + 9.0}}


@pytest.mark.parametrize("identity", ["source", "exec_id", "exec_handle"])
def test_descriptors_cannot_reuse_native_source_or_live_executable(tmp_path, monkeypatch, identity):
    run, root = fixture(tmp_path, monkeypatch)
    callback_path = root / "vllm-graph-clones-rank-0-capture-0-1.json"
    source_path = callback_path.with_name(callback_path.stem + "-source.json")
    source = json.loads(source_path.read_bytes())
    callback = json.loads(callback_path.read_bytes())
    create, clone = callback["callbacks"]
    if identity == "source":
        source["graph_id"] = create["graph_id"] = 4
    elif identity == "exec_id":
        callback["actual_graph_exec_id"] = create["graph_exec_id"] = 7
    else:
        create["raw_fields"]["graphExec"] = clone["raw_fields"]["graph"] = 39
    put(source_path, source)
    put(callback_path, callback)
    registry = resolve_registry(source, callback)
    registry["instantiation_receipt"] = {"file": callback_path.name, "sha256": file_sha256(callback_path)}
    put(root / "vllm-capture-rank-0-1.json", registry)
    with pytest.raises(ValueError, match="reuse a source or live executable"):
        common.read_graph_run(root, run)
