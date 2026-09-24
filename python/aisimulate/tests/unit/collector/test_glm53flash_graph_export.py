# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY authored traces exercise the real export and native-state readers."""

import copy
import json
import shutil

import pyarrow.parquet as pq
import pytest
from collector import glm53flash_graph_export as graph
from collector import glm53flash_validation as native
from collector.glm53flash_contract import canonical_json, operation_geometry, sha256_json
from collector.glm53flash_graph_callbacks import EVENT_RECORD_CUDART_SHA256, resolve_registry
from collector.glm53flash_graph_nodes import EXECUTION_RANGE, bind_execution_activity, bind_replay_kernels
from collector.glm53flash_jsonl import file_sha256, iter_records

from .test_glm53flash_graph_policy import snapshot
from .test_glm53flash_ops_evidence import native_fixture, put, put_lines

pytestmark = pytest.mark.unit


def fixture(tmp_path, monkeypatch, role="calibration"):
    run, root, records, _, manifest = native_fixture(tmp_path, role)
    run["spec"]["ops_execution_mode"] = "native_full_graph"
    zero = {"name": "zero", "component": "mhc", "geometry": canonical_json({"role": "pre"})}
    setup = {"name": "native_graph_setup", "component": "runtime", "geometry": canonical_json({"is_context": False})}
    manifest["phases"]["generation"].append(zero)
    manifest["runtime_operations"] = {"generation": [setup]}
    put(root / "manifest.json", manifest)
    monkeypatch.setattr(graph, "build_model_manifest", lambda *_: copy.deepcopy(manifest))
    config = json.loads((root / "sglang-resolved-config.json").read_text())
    config["cuda_graph_config"]["decode"]["backend"] = "full"
    put(root / "sglang-resolved-config.json", config)
    provenance = json.loads((root / "provenance.json").read_text())
    # Match actual producer layering: the input identity is stable across
    # independent processes, while capture receives the complete driver run.
    put(
        root / "provenance.json",
        {
            key: provenance[key]
            for key in (
                "backend",
                "backend_version",
                "backend_revision",
                "checkpoint_revision",
                "config_sha256",
                "source_sha256",
                "runtime_digest",
            )
        },
    )
    for rank in range(2):
        policy = snapshot(rank, [1])
        put(root / f"graph-policy-rank-{rank}.json", policy)
        shape = policy["captured_keys"][0]
        source = {
            "tp_rank": rank,
            "provenance": {**provenance, "native_projection_source_sha256": graph.SGLANG_PROJECTION_SOURCE_PINS},
            "capture_scope": "model_with_logits",
            "uncaptured_operations": [],
            "graph_mutations": False,
            "operations": manifest["phases"]["generation"],
            "physical_padded_tokens": 1,
            "native_shape_key": shape,
            "native_api_libraries": {
                "cupti": {"sha256": graph.QUALIFIED_CUPTI_SHA256, "path": "/TEST_ONLY/libcupti.so"},
                "cudart": {
                    "sha256": "a" * 64,
                    "path": "/TEST_ONLY/libcudart.so",
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
                {"name": "zero", "source": "TEST_ONLY.native_empty_call", "completed": True, "owned_node_ids": []},
            ],
        }
        callback = {
            "callback_errors": [],
            "graph_mutations": False,
            "actual_graph_exec_id": 71,
            "callbacks": [
                {
                    "kind": "graph_exec_created",
                    "graph_id": 4,
                    "graph_exec_id": 71,
                    "raw_fields": {"graph": 12, "graphExec": 39},
                },
                {
                    "kind": "node_cloned",
                    "original_node_id": 17,
                    "node_id": 53,
                    "node_type": 0,
                    "raw_fields": {"graph": 39, "originalGraph": 12, "node": 72, "originalNode": 73, "nodeType": 0},
                },
            ],
        }
        callback_path = root / f"graph-clones-rank-{rank}-capture-0.json"
        put(callback_path, callback)
        registry = resolve_registry(source, callback)
        registry["instantiation_receipt"] = {"file": callback_path.name, "sha256": file_sha256(callback_path)}
        put_lines(root / f"capture-source-nodes-rank-{rank}.jsonl", [source])
        put_lines(root / f"capture-nodes-rank-{rank}.jsonl", [registry])
        graph_rows = []
        for forward in records[rank]:
            if forward["stage"] != "measure":
                continue
            forward.update(
                used_cuda_graph=True, runtime_mode="FULL", num_padded_tokens=1, whole_forward_boundary=graph.BOUNDARY
            )
            events = [
                {
                    "cat": "user_annotation",
                    "ph": "X",
                    "name": EXECUTION_RANGE,
                    "pid": 1,
                    "tid": 2,
                    "ts": 0,
                    "dur": 10000,
                },
                {
                    "cat": "cuda_runtime",
                    "name": "cudaMemsetAsync",
                    "pid": 1,
                    "tid": 2,
                    "ts": 1,
                    "dur": 1,
                    "args": {"correlation": 8},
                },
                {
                    "cat": "cuda_runtime",
                    "name": "cudaGraphLaunch",
                    "pid": 1,
                    "tid": 2,
                    "ts": 4,
                    "dur": 1,
                    "args": {"correlation": 9},
                },
                {
                    "cat": "gpu_memset",
                    "name": "TEST_ONLY metadata",
                    "ts": 2,
                    "dur": 4,
                    "args": {"bytes": 1024, "stream": 1, "correlation": 8, "graph id": 0, "graph node id": 0},
                },
                {
                    "cat": "kernel",
                    "name": "TEST_ONLY attention",
                    "ts": 5,
                    "dur": 3 if rank == 1 else 9,
                    "args": {
                        "stream": 1,
                        "grid": [1, 1, 1],
                        "block": [32, 1, 1],
                        "shared memory": 0,
                        "correlation": 9,
                        "graph id": 71,
                        "graph node id": 53,
                    },
                },
            ]
            trace_path = root / f"graph-profile-rank-{rank}-forward-{forward['invocation']}.json"
            put(trace_path, {"traceEvents": events, "aisim_native_forward": graph.trace_forward_identity(forward)})
            binding = bind_execution_activity(bind_replay_kernels(registry, events, correlation=9), events)
            binding.update(trace_file=trace_path.name, trace_sha256=file_sha256(trace_path))
            graph_rows.append(
                {
                    **forward,
                    "native_shape_key": shape,
                    "native_execute_source_sha256": graph.SOURCE_PINS[
                        "srt/model_executor/runner/decode_cuda_graph_runner.py"
                    ],
                    "native_dispatch_policy_receipt": {
                        "file": f"graph-policy-rank-{rank}.json",
                        "sha256": file_sha256(root / f"graph-policy-rank-{rank}.json"),
                    },
                    "capture_registry_file": f"capture-nodes-rank-{rank}.jsonl" if role == "calibration" else None,
                    "capture_registry_sha256": graph._semantic_sha(registry) if role == "calibration" else None,
                    "replay_nodes": binding if role == "calibration" else None,
                    "profiled": role == "calibration",
                }
            )
        put_lines(root / f"forward-rank-{rank}.jsonl", records[rank])
        put_lines(root / f"graph-forward-rank-{rank}.jsonl", graph_rows)
    return run, root


def control_fixture(tmp_path, monkeypatch):
    directory = tmp_path / "control"
    directory.mkdir()
    run, root = fixture(directory, monkeypatch, "holdout")
    for path in root.iterdir():
        if path.suffix in (".json", ".jsonl"):
            text = path.read_text().replace("authored-cpu-fixture", "authored-cpu-control")
            text = text.replace("authored-unit-run", "authored-independent-control-run")
            text = text.replace("request-", "control-request-")
            text = text.replace('"dataset_role": "holdout"', '"dataset_role": "calibration"')
            text = text.replace('"dataset_role":"holdout"', '"dataset_role":"calibration"')
            path.write_text(text)
    for name in ("sglang-declared-config.json", "sglang-resolved-config.json"):
        path = root / name
        config = json.loads(path.read_text())
        config["model_path"] = "/models/authored-cpu-fixture"
        put(path, config)
    run["role"] = "control"
    return {"control_run": run, "control_root": root}


@pytest.mark.parametrize(
    "field,value", [("mem_fraction_static", 0.82), ("max_running_requests", 16), ("moe_runner_backend", "changed")]
)
def test_graph_profile_control_rejects_different_actual_native_settings(tmp_path, monkeypatch, field, value):
    run, root = fixture(tmp_path, monkeypatch)
    control = control_fixture(tmp_path, monkeypatch)
    for name in ("sglang-declared-config.json", "sglang-resolved-config.json"):
        path = control["control_root"] / name
        config = json.loads(path.read_text())
        config[field] = value
        put(path, config)
    with pytest.raises(ValueError, match="execution policies differ across graph calibration/profile control"):
        graph.export_graph(root, run, root / graph.BASENAME, **control)


def test_graph_profile_control_normalizes_only_random_seed(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    control = control_fixture(tmp_path, monkeypatch)
    for directory, seed in ((root, 123), (control["control_root"], 456)):
        for name in ("sglang-declared-config.json", "sglang-resolved-config.json"):
            path = directory / name
            config = json.loads(path.read_text())
            config.update(random_seed=seed, api_key="TEST_ONLY_PRIVATE")
            put(path, config)
    graph.export_graph(root, run, root / graph.BASENAME, **control)
    receipt = json.loads((root / "graph-profile-control.json").read_text())
    assert receipt["execution_policy"]["normalization"] == "resolved_server_args_except_random_seed_v1"
    assert "TEST_ONLY_PRIVATE" not in json.dumps(receipt)


def test_original_node_activity_export_and_public_binding(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    table = root / graph.BASENAME
    result = graph.export_graph(root, run, table, **control_fixture(tmp_path, monkeypatch))
    assert result["rows"] == 3 and result["accuracy_acceptance"] == "NOT_EVALUATED"
    rows = {row["component"]: row for row in pq.read_table(table).to_pylist()}
    assert rows["attention"]["latency"] == pytest.approx(0.006)
    assert rows["runtime"]["latency"] == pytest.approx(0.004)
    assert rows["mhc"]["latency"] == 0 and rows["mhc"]["activity_count"] == 0
    assert all(row["sample_count"] == 10 for row in rows.values())
    receipt = native.load_native(run, root)
    binding = native.bind_calibration([table], run, receipt)
    assert binding["rows"] == 3
    assert binding["tables"] == [{"path": str(table), "sha256": file_sha256(table)}]
    assert binding["graph_policy_sha256"] == sha256_json(receipt["graph_policy"])
    selection = json.loads((root / "graph-rank-selection.json").read_text())
    assert [row["selected_rank"] for row in selection["forwards"]] == [0] * 5 + [1] * 5 + [0] * 5


def test_independent_graph_truth_preserves_actual_whole_boundary(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch, "holdout")
    result = native.load_native(run, root)
    assert result["values"] == {1: 5.0}
    assert result["timing_boundary"] == graph.BOUNDARY
    with pytest.raises(ValueError, match="calibration"):
        graph.export_graph(root, run, root / graph.BASENAME, **control_fixture(tmp_path, monkeypatch))


@pytest.mark.parametrize("defect", ["run_id", "projection_source", "missing_projection", "stable_identity"])
def test_capture_requires_actual_driver_run_and_verified_projection_sources(tmp_path, monkeypatch, defect):
    run, root = fixture(tmp_path, monkeypatch)
    path = root / "capture-source-nodes-rank-0.jsonl"
    rows = list(iter_records(path))
    if defect == "run_id":
        rows[0]["provenance"]["run_id"] = "another-native-run"
    elif defect == "projection_source":
        rows[0]["provenance"]["native_projection_source_sha256"] = {"unverified.py": "a" * 64}
    elif defect == "missing_projection":
        del rows[0]["provenance"]["native_projection_source_sha256"]
    else:
        provenance = json.loads((root / "provenance.json").read_bytes())
        provenance["runtime_digest"] = "sha256:" + "f" * 64
        put(root / "provenance.json", provenance)
    put_lines(path, rows)
    with pytest.raises(ValueError, match="provenance|ownership"):
        graph.read_graph_run(root, run)


@pytest.mark.parametrize(
    "defect", ["zero_missing_call", "zero_unfinished", "source_nodes", "clone", "trace", "target", "policy", "padding"]
)
def test_graph_export_cannot_borrow_or_invent_raw_measurements(tmp_path, monkeypatch, defect):
    run, root = fixture(tmp_path, monkeypatch)
    if defect.startswith("zero") or defect == "source_nodes":
        path = root / "capture-source-nodes-rank-0.jsonl"
        row = json.loads(path.read_text())
        if defect == "zero_missing_call":
            row["calls"].pop()
        elif defect == "zero_unfinished":
            row["calls"][-1]["completed"] = False
        else:
            row["calls"][0]["owned_node_ids"] = []
        put_lines(path, [row])
    elif defect == "clone":
        path = root / "graph-clones-rank-0-capture-0.json"
        row = json.loads(path.read_text())
        row["callbacks"][-1]["original_node_id"] = 77
        put(path, row)
    else:
        path = root / "graph-forward-rank-0.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if defect == "trace":
            rows[0]["replay_nodes"]["operation_activity_unions"][0]["active_union_us"] = 10000
        elif defect == "target":
            rows[0]["request_ids"] = ["another-request"]
        elif defect == "policy":
            rows[0]["native_dispatch_policy_receipt"]["sha256"] = "a" * 64
        else:
            rows[0]["native_shape_key"]["size"] = 2
        put_lines(path, rows)
    with pytest.raises(ValueError):
        graph.export_graph(root, run, root / graph.BASENAME, **control_fixture(tmp_path, monkeypatch))


def test_exported_table_cannot_replace_activity_with_whole_forward_residual(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    path = root / graph.BASENAME
    graph.export_graph(root, run, path, **control_fixture(tmp_path, monkeypatch))
    receipt = native.load_native(run, root)
    import pyarrow as pa

    rows = pq.read_table(path).to_pylist()
    rows[0]["latency"] += 1
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="original native activity"):
        native.bind_calibration([path], run, receipt)


def test_profile_control_reports_perturbation_without_scaling_measured_units(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    control = control_fixture(tmp_path, monkeypatch)
    for rank in range(2):
        for stem in ("forward", "graph-forward"):
            path = control["control_root"] / f"{stem}-rank-{rank}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row["whole_forward_gpu_ms"] *= 2
            put_lines(path, rows)
    graph.export_graph(root, run, root / graph.BASENAME, **control)
    report = json.loads((root / "graph-profile-control.json").read_text())
    assert report["results"][0]["profiled_to_control_ratio"] == 0.5
    assert report["timing_equivalence"] == "REPORTED_NOT_ASSUMED"
    rows = {row["component"]: row for row in pq.read_table(root / graph.BASENAME).to_pylist()}
    assert rows["attention"]["latency"] == pytest.approx(0.006)
    assert rows["runtime"]["latency"] == pytest.approx(0.004)


@pytest.mark.parametrize("defect", ["instrumented", "missing_bucket", "request_set", "changed_after_export"])
def test_profile_control_and_unused_capture_policy_remain_required(tmp_path, monkeypatch, defect):
    run, root = fixture(tmp_path, monkeypatch)
    control = control_fixture(tmp_path, monkeypatch)
    if defect == "changed_after_export":
        graph.export_graph(root, run, root / graph.BASENAME, **control)
        path = control["control_root"] / "retained-rank-0.jsonl"
        path.write_text(path.read_text().replace("{", "{ ", 1))
        with pytest.raises(ValueError, match="profiler control differs"):
            native.load_native(run, root)
        return
    if defect == "missing_bucket":
        put(root / "graph-policy-rank-0.json", snapshot(0, [1, 2]))
    else:
        path = control["control_root"] / "graph-forward-rank-0.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["profiled" if defect == "instrumented" else "request_set"] = (
            True if defect == "instrumented" else "borrowed-calibration"
        )
        put_lines(path, rows)
    with pytest.raises(ValueError):
        graph.export_graph(root, run, root / graph.BASENAME, **control)


def test_public_homogeneous_rust_prediction_uses_frozen_past_coordinate(tmp_path, monkeypatch):
    """Full real model graph with TEST_ONLY rows; this checks API composition."""
    from importlib.resources import files

    import pyarrow as pa

    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models import get_model

    calibration_run, calibration_root = fixture(tmp_path, monkeypatch)
    proof = graph.read_graph_run(calibration_root, calibration_run)
    policy = proof["policy"]
    calibration_native = {
        "graph_policy": policy,
        **{key: proof[key] for key in ("execution_policy", "_execution_policy")},
    }
    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    run, root = fixture(holdout_dir, monkeypatch, "holdout")
    systems = tmp_path / "systems"
    data = systems / "data/gb300/sglang/0.5.20"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    model = get_model("zai-org/GLM-5.3-Flash", ModelConfig(tp_size=2, moe_tp_size=2, moe_ep_size=1), "sglang")
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
            "prefix": 2,
            "padded_batch_size": 1,
            "latency": 10.0 if kind == "Glm53Runtime" else 1.0,
            "activity_count": 1,
            "sample_count": 10,
            "dispatch_fingerprint": "a" * 64,
            "graph_policy": canonical_json(policy),
            "graph_policy_sha256": sha256_json(policy),
            "dataset_role": "calibration",
            "aggregation_policy": graph.WHOLE_FORWARD_RANK,
            "rank_selection_sha256": "b" * 64,
            "evidence_sha256": "c" * 64,
            "measurement_scope": graph.SCOPE,
        }
    pq.write_table(pa.Table.from_pylist(list(rows.values())), data / graph.BASENAME)
    config = {
        "model": "zai-org/GLM-5.3-Flash",
        "system": "gb300",
        "backend": "sglang",
        "backend_version": "0.5.20",
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
    result = graph.predict_homogeneous(run, root, config, calibration_native)
    assert result["rows"] == {1: {"prediction_ms": 376.0}}
    assert result["diagnostics"]["consumer"] == "public_EngineHandle_predict_decode_latency"
    changed = copy.deepcopy(policy)
    changed["capture_sizes"] = [1, 2, 4]
    with pytest.raises(ValueError, match="changed its frozen"):
        graph.predict_homogeneous(run, root, config, {**calibration_native, "graph_policy": changed})
    resolved = root / "sglang-resolved-config.json"
    changed_config = json.loads(resolved.read_text())
    changed_config["mem_fraction_static"] = 0.82
    put(resolved, changed_config)
    with pytest.raises(ValueError, match="execution policies differ across graph calibration/holdout"):
        graph.predict_homogeneous(run, root, config, calibration_native)


@pytest.mark.parametrize("defect", ["filename", "copied_bytes", "rank", "invocation", "run", "role", "missing"])
def test_trace_cannot_be_reused_for_another_native_forward(tmp_path, monkeypatch, defect):
    run, root = fixture(tmp_path, monkeypatch)
    path = root / "graph-forward-rank-0.jsonl"
    records = list(iter_records(path))
    first, second = records[0]["replay_nodes"], records[1]["replay_nodes"]
    trace_path = root / second["trace_file"]
    if defect == "filename":
        records[1]["replay_nodes"] = first
    elif defect == "copied_bytes":
        trace_path.write_bytes((root / first["trace_file"]).read_bytes())
    else:
        trace = json.loads(trace_path.read_bytes())
        if defect == "missing":
            trace.pop("aisim_native_forward")
        else:
            key = {"rank": "tp_rank", "invocation": "invocation", "run": "run_id", "role": "sampling_role"}[defect]
            trace["aisim_native_forward"][key] = {"rank": 1, "invocation": 900, "run": "other", "role": "measurement"}[
                defect
            ]
        put(trace_path, trace)
    if defect != "filename":
        second["trace_sha256"] = file_sha256(trace_path)
    put_lines(path, records)
    with pytest.raises(ValueError, match="trace"):
        graph.read_graph_run(root, run)


def test_missing_setup_activity_cannot_be_exported_as_zero(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    path = root / "graph-forward-rank-0.jsonl"
    rows = list(iter_records(path))
    for row in rows:
        replay = row["replay_nodes"]
        trace_path = root / replay["trace_file"]
        trace = json.loads(trace_path.read_bytes())
        trace["traceEvents"] = [event for event in trace["traceEvents"] if event.get("cat") != "gpu_memset"]
        put(trace_path, trace)
        replay["trace_sha256"] = file_sha256(trace_path)
    put_lines(path, rows)
    with pytest.raises(ValueError, match="device-work call lacks"):
        graph.export_graph(root, run, root / graph.BASENAME, **control_fixture(tmp_path, monkeypatch))


def test_exporter_rechecks_the_hashed_event_record_query_receipt(tmp_path, monkeypatch):
    _, root = fixture(tmp_path, monkeypatch)
    source_path = root / "capture-source-nodes-rank-0.jsonl"
    source = next(iter_records(source_path))
    source["native_api_libraries"]["cudart"]["sha256"] = EVENT_RECORD_CUDART_SHA256
    source["nodes"].append({"node_id": 18, "node_type": 7, "name": "native_graph_setup"})
    callback_path = root / "graph-clones-rank-0-capture-0.json"
    callback = json.loads(callback_path.read_text())
    callback["callback_subscription_closed"] = True
    callback["callbacks"].append(
        {
            "kind": "node_cloned",
            "original_node_id": 18,
            "node_id": 54,
            "node_type": 0,
            "raw_fields": {"graph": 39, "originalGraph": 12, "node": 74, "originalNode": 75, "nodeType": 0},
        }
    )
    proof = {
        "method": "CUDA13_EVENT_RECORD_CLONE_QUERY_V1",
        "native_api_libraries": source["native_api_libraries"],
        "graph_exec_handle": 39,
        "graph_exec_id": 71,
        "callback_subscription_closed": True,
        "completed": True,
        "queries": [
            {
                "original_node_id": 18,
                "node_id": 54,
                "source_node_handle": 75,
                "clone_node_handle": 74,
                "source_node_type": 7,
                "callback_node_type": 0,
                "api": "cudaGraphNodeGetType",
                "rc": 0,
                "native_node_type": 7,
            }
        ],
    }
    proof_path = root / "TEST_ONLY-event-record-query.json"
    put(callback_path, callback)
    put(proof_path, proof)
    put_lines(source_path, [source])
    registry = resolve_registry(source, callback, proof)
    registry["instantiation_receipt"] = {"file": callback_path.name, "sha256": file_sha256(callback_path)}
    registry["node_type_receipt"] = {"file": proof_path.name, "sha256": file_sha256(proof_path)}
    target = root / "capture-nodes-rank-0.jsonl"
    put_lines(target, [registry])
    files = set()
    args = [
        root,
        0,
        json.loads((root / "graph-policy-rank-0.json").read_text()),
        json.loads((root / "manifest.json").read_text()),
        json.loads((root / "provenance.json").read_text()),
    ]
    captures = graph._captures(*args, files)
    assert proof_path.name in files
    assert list(captures.values())[0]["nodes"][-1]["node_type"] == 7
    proof["queries"][0]["rc"] = 1
    put(proof_path, proof)
    with pytest.raises(ValueError):
        graph._captures(*args, set())  # Original file hash rejects replacement.
    registry["node_type_receipt"]["sha256"] = file_sha256(proof_path)
    put_lines(target, [registry])
    with pytest.raises(ValueError, match="exact deferred native type proof"):
        graph._captures(*args, set())  # Rehashing cannot hide the actual native rc.


@pytest.mark.parametrize("defect", [None, "direction", "bytes", "missing"])
def test_pending_source_memcpy_requires_actual_forward_activity_before_export(tmp_path, monkeypatch, defect):
    from collector.glm53flash_graph_nodes import MEMCPY_TRACE_NAME

    from .test_glm53flash_graph_memcpy import pending_copy

    run, root = fixture(tmp_path, monkeypatch)
    for rank in range(2):
        source_path = root / f"capture-source-nodes-rank-{rank}.jsonl"
        source = next(iter_records(source_path))
        source["native_api_libraries"]["cudart"]["sha256"] = EVENT_RECORD_CUDART_SHA256
        copy_node = pending_copy()[0]["nodes"][1]
        copy_node.update(node_id=18, name="attention_0")
        copy_node["memcpy_params"]["source_node_handle"] = 74
        source["nodes"].append(copy_node)
        source["calls"][0]["owned_node_ids"].append(18)
        callback_path = root / f"graph-clones-rank-{rank}-capture-0.json"
        callback = json.loads(callback_path.read_text())
        callback["callback_subscription_closed"] = True
        copy_callback = copy.deepcopy(callback["callbacks"][-1])
        copy_callback.update(original_node_id=18, node_id=54)
        copy_callback["raw_fields"].update(originalNode=74, node=75)
        callback["callbacks"].append(copy_callback)
        put(callback_path, callback)
        registry = resolve_registry(source, callback, allow_pending_memcpy=True)
        registry["instantiation_receipt"] = {"file": callback_path.name, "sha256": file_sha256(callback_path)}
        put_lines(source_path, [source])
        put_lines(root / f"capture-nodes-rank-{rank}.jsonl", [registry])
        forwards_path = root / f"graph-forward-rank-{rank}.jsonl"
        forwards = list(iter_records(forwards_path))
        for row in forwards:
            path = root / row["replay_nodes"]["trace_file"]
            trace = json.loads(path.read_text())
            trace["traceEvents"].append(
                {
                    "cat": "gpu_memcpy",
                    "name": MEMCPY_TRACE_NAME,
                    "ts": 6,
                    "dur": 1,
                    "args": {"bytes": 256, "stream": 1, "correlation": 9, "graph id": 71, "graph node id": 54},
                }
            )
            binding = bind_execution_activity(
                bind_replay_kernels(registry, trace["traceEvents"], correlation=9), trace["traceEvents"]
            )
            if rank == 0 and row is forwards[-1]:
                if defect == "direction":
                    trace["traceEvents"][-1]["name"] = "Memcpy HtoD (Pinned -> Device)"
                elif defect == "bytes":
                    trace["traceEvents"][-1]["args"]["bytes"] = 128
                elif defect == "missing":
                    trace["traceEvents"].pop()
            put(path, trace)
            binding.update(trace_file=path.name, trace_sha256=file_sha256(path))
            row.update(replay_nodes=binding, capture_registry_sha256=graph._semantic_sha(registry))
        put_lines(forwards_path, forwards)
    if defect is None:
        result = graph.read_graph_run(root, run)
        assert result["forwards"]
    else:
        with pytest.raises(ValueError, match="pending native memcpy|omits captured"):
            graph.read_graph_run(root, run)
