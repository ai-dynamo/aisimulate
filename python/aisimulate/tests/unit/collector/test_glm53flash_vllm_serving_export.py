# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY native-shaped rows; no GPU, runtime or performance qualification."""

import copy
import hashlib

import pytest
from collector import glm53flash_vllm_serving_export as serving
from collector.glm53flash_contract import BACKENDS, _runtime_contract, build_model_manifest, sha256_json
from collector.glm53flash_vllm_graph_policy import SOURCE_PINS, select_descriptor

from .test_glm53flash_vllm_graph_policy import snapshot

pytestmark = pytest.mark.unit


def fixture(*, phase="generation", query=1, batch=3, prefix=128):
    manifest = build_model_manifest("vllm", "fp8", 2)
    version = BACKENDS["vllm"][0]
    _, pins = _runtime_contract("vllm", version)
    provenance = {
        **{
            key: manifest[key]
            for key in ("backend", "backend_version", "backend_revision", "checkpoint_revision", "config_sha256")
        },
        "source_sha256": hashlib.sha256(pins.encode()).hexdigest(),
        "runtime_digest": "sha256:" + "a" * 64,
    }
    args = {"enforce_eager": False, "max_num_seqs": 4, "compile_mode": 0}
    proof = {
        "manifest": manifest,
        "provenance": provenance,
        "snapshots": {},
        "forwards": {},
        "policy_evidence_sha256": "b" * 64,
        "execution_policy": {"normalization": "native_vllm_engine_args_except_seed_v1", "sha256": sha256_json(args)},
        "_execution_policy": args,
    }
    for rank in range(2):
        native = snapshot()
        native.update(tp_size=2, tp_rank=rank)
        proof["snapshots"][rank] = native
        descriptor = select_descriptor(native, batch=batch, query=query, is_context=phase == "context")
        mode = descriptor["cg_mode"]
        entries = manifest["phases"][phase] + manifest["runtime_operations"][phase]
        for repetition in range(15):
            # The whole-forward slowest rank alternates. Opposite per-unit
            # extrema deliberately test that no per-operation TP max is used.
            row = {
                "phase": phase,
                "runtime_mode": mode,
                "stage": "measure",
                "tp_rank": rank,
                "batch_size": batch,
                "query_lengths": [query] * batch,
                "prefix_lengths": [prefix] * batch,
                "num_padded_tokens": descriptor["num_tokens"],
                "used_cuda_graph": mode != "NONE",
                "native_dispatch": {
                    "descriptor": descriptor,
                    "policy_sha256": sha256_json(native),
                    "physical_tokens": descriptor["num_tokens"],
                    "physical_requests": descriptor["num_reqs"] or batch,
                },
                "native_graph_replay_completed": mode != "NONE",
                "request_ids": [f"TEST_ONLY-{repetition}-{i}" for i in range(batch)],
                "benchmark_id": 1,
                "repetition": repetition,
                "sampling_role": "warmup" if repetition < 5 else "measurement",
                "dataset_role": "calibration",
                "request_set": "TEST_ONLY-run",
                "corpus_sha256": "c" * 64,
                "token_witness": [{"TEST_ONLY_token_sha256": "d" * 64}] * batch,
                "gpu_completed": True,
                "whole_forward_boundary": serving.BOUNDARY,
                "whole_forward_gpu_ms": 100.0 if rank == repetition % 2 else 99.0,
                "forward_id": f"rank-{rank}/forward-{repetition + 1}",
                "invocation": repetition + 1,
                "binding": {
                    entry["name"]: {
                        "latency": float(1 + rank + i % 3),
                        "dispatch": sha256_json({"TEST_ONLY": entry["name"]}),
                        "activity_count": 1,
                    }
                    for i, entry in enumerate(entries)
                },
            }
            if mode == "PIECEWISE":
                entry = next(
                    item for item in native["piecewise_entries"] if item["num_tokens"] == descriptor["num_tokens"]
                )
                row.update(
                    native_piecewise_replay_completed=True,
                    native_piecewise_replay={
                        "source_sha256": SOURCE_PINS["compilation/breakable_cudagraph.py"],
                        "entry_descriptor": {
                            key: entry[key]
                            for key in ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras")
                        },
                        "segment_count": entry["num_graphs"] + entry["num_eager_breaks"],
                    },
                )
            proof["forwards"].setdefault((1, repetition), {})[rank] = row
    proof["policy"] = serving.build_serving_policy(proof["snapshots"], manifest, provenance, proof)
    return proof


def test_policy_has_only_stable_complete_native_identity():
    proof = fixture()
    policy = proof["policy"]
    assert policy["schema_version"] == 3
    assert "tp_rank" not in policy["native_policy"]
    assert "schema_version" not in policy["native_policy"]
    assert policy["native_policy_sha256"] == sha256_json(policy["native_policy"])
    changed_attempt = copy.deepcopy(proof)
    changed_attempt["policy_evidence_sha256"] = "9" * 64
    assert (
        serving.build_serving_policy(changed_attempt["snapshots"], proof["manifest"], proof["provenance"], proof)
        == policy
    )


@pytest.mark.parametrize(
    "phase,query,batch,mode,physical",
    [
        ("generation", 1, 3, "FULL", 4),
        ("context", 1, 3, "PIECEWISE", 4),
        ("context", 2, 1, "PIECEWISE", 2),
    ],
)
def test_named_units_and_coherent_rank_preserve_actual_dispatch(phase, query, batch, mode, physical):
    proof = fixture(phase=phase, query=query, batch=batch)
    rows, selection = serving.aggregate_serving(proof, evidence_sha256="e" * 64)
    assert len(rows) == 278
    assert {row["sample_count"] for row in rows} == {10}
    assert len({row["operation_name"] for row in rows}) == 278
    assert len({row["geometry"] for row in rows}) < 278  # layers were not collapsed
    assert {row["runtime_mode"] for row in rows} == {mode}
    assert {row["physical_num_tokens"] for row in rows} == {physical}
    assert {row["physical_num_requests"] for row in rows} == ({physical} if mode == "FULL" else {batch})
    assert [row["selected_rank"] for row in selection["forwards"]] == [i % 2 for i in range(15)]
    by_name = {row["operation_name"]: row for row in rows}
    for i, entry in enumerate(proof["manifest"]["phases"][phase] + proof["manifest"]["runtime_operations"][phase]):
        assert by_name[entry["name"]]["latency"] == 1.5 + i % 3
    assert set(rows[0]) == set(serving.COLUMNS)


def test_equal_whole_intervals_use_lowest_actual_rank():
    proof = fixture()
    for group in proof["forwards"].values():
        for row in group.values():
            row["whole_forward_gpu_ms"] = 100.0
    _, selection = serving.aggregate_serving(proof, evidence_sha256="e" * 64)
    assert {row["selected_rank"] for row in selection["forwards"]} == {0}


@pytest.mark.parametrize(
    "defect",
    [
        "missing_rank",
        "different_requests",
        "wrong_role",
        "missing_warmup",
        "reused_invocation",
        "missing_unit",
        "extra_unit",
        "missing_setup",
        "nonfinite",
        "dispatch_change",
        "padding",
        "incomplete_replay",
        "diagnostic",
    ],
)
def test_incomplete_or_crossbound_native_forward_rejects(defect):
    proof = fixture()
    row = proof["forwards"][(1, 6)][0]
    name = next(iter(row["binding"]))
    if defect == "missing_rank":
        del proof["forwards"][(1, 6)][1]
    elif defect == "different_requests":
        row["request_ids"][0] = "OTHER"
    elif defect == "wrong_role":
        row["sampling_role"] = "warmup"
    elif defect == "missing_warmup":
        del proof["forwards"][(1, 0)]
    elif defect == "reused_invocation":
        row.update(invocation=1, forward_id="rank-0/forward-1")
    elif defect == "missing_unit":
        del row["binding"][name]
    elif defect == "extra_unit":
        row["binding"]["unknown"] = row["binding"][name]
    elif defect == "missing_setup":
        del row["binding"]["native_graph_setup"]
    elif defect == "nonfinite":
        row["whole_forward_gpu_ms"] = float("nan")
    elif defect == "dispatch_change":
        row["binding"][name]["dispatch"] = "f" * 64
    elif defect == "padding":
        row["num_padded_tokens"] = 3
    elif defect == "incomplete_replay":
        row["native_graph_replay_completed"] = False
    else:
        row["measurement_admission"] = "DIAGNOSTIC_ONLY_NO_TABLE_EXPORT"
    with pytest.raises(ValueError):
        serving.aggregate_serving(proof, evidence_sha256="e" * 64)


@pytest.mark.parametrize("defect", ["rank_policy", "invented_schema", "source", "config", "eager", "unknown_runtime"])
def test_policy_rejects_cross_runtime_config_or_invented_native_fields(defect):
    proof = fixture()
    if defect == "rank_policy":
        proof["snapshots"][1]["max_num_reqs"] = 8
    elif defect == "invented_schema":
        proof["snapshots"][0]["schema_version"] = 1
    elif defect == "source":
        proof["provenance"]["source_sha256"] = "0" * 64
    elif defect == "config":
        proof["provenance"]["config_sha256"] = "0" * 64
    elif defect == "unknown_runtime":
        proof["snapshots"][0]["backend_version"] += "+UNQUALIFIED"
    else:
        proof["_execution_policy"]["enforce_eager"] = True
        proof["execution_policy"]["sha256"] = sha256_json(proof["_execution_policy"])
    with pytest.raises(ValueError):
        serving.build_serving_policy(proof["snapshots"], proof["manifest"], proof["provenance"], proof)


def test_native_none_seed_is_valid_but_diagnostic_measurement_cannot_export():
    proof = fixture(phase="context", query=8, batch=1, prefix=0)
    row = proof["forwards"][(1, 0)][0]
    row["stage"] = "seed"
    assert serving.check_serving_dispatch(row, proof["snapshots"][0])["cg_mode"] == "NONE"
    row["stage"] = "measure"
    row["measurement_admission"] = "DIAGNOSTIC_ONLY_NO_TABLE_EXPORT"
    with pytest.raises(ValueError, match="diagnostic"):
        serving.aggregate_serving(proof, evidence_sha256="e" * 64)
    del row["measurement_admission"]
    with pytest.raises(ValueError, match="independent native qualification"):
        serving.aggregate_serving(proof, evidence_sha256="e" * 64)


def test_piecewise_actual_native_entry_identity_required():
    proof = fixture(phase="context", query=1)
    proof["forwards"][(1, 5)][0]["native_piecewise_replay"]["segment_count"] -= 1
    with pytest.raises(ValueError, match="entry/segments"):
        serving.aggregate_serving(proof, evidence_sha256="e" * 64)


def piecewise_files(tmp_path):
    """Write a full named model with TEST_ONLY multi-segment ownership."""
    import json

    from collector.glm53flash_graph_callbacks import QUALIFIED_CUPTI_SHA256, resolve_registry
    from collector.glm53flash_jsonl import file_sha256
    from collector.glm53flash_vllm_piecewise import BREAKABLE_SOURCE_PIN, EAGER_RANGE_PREFIX

    from .test_glm53flash_ops_evidence import put

    proof = fixture(phase="context", query=1, batch=3)
    policy = proof["snapshots"][0]
    for entry in policy["piecewise_entries"]:
        entry.update(num_graphs=2, num_eager_breaks=1)
    entries = proof["manifest"]["phases"]["context"]
    calls = [
        {"index": i, "name": entry["name"], "source": "TEST_ONLY.native", "completed": True}
        for i, entry in enumerate(item for item in entries if item["name"] != "logits")
    ]
    call = next(row for row in calls if row["name"] == "attention_3")
    libraries = {
        "cupti": {"path": "/TEST_ONLY/libcupti.so", "sha256": QUALIFIED_CUPTI_SHA256},
        "cudart": {
            "path": "/TEST_ONLY/libcudart.so",
            "sha256": "a" * 64,
            "runtime_version": 13000,
            "abi": "CUDA13_capture7_edges5",
        },
    }
    stem = "vllm-graph-clones-rank-0-capture-0"
    callbacks, observed, sources = [], [], []
    for index, entry in enumerate(policy["piecewise_entries"]):
        segments = []
        for position in (0, 2):
            ident = index * 100 + (10 if position == 0 else 20)
            source_id, exec_id, source_handle, exec_handle = ident + 1000, ident, ident + 10000, ident + 20000
            source_node, clone_node = ident + 3000, ident + 4000
            callbacks.extend(
                [
                    {
                        "kind": "graph_exec_created",
                        "graph_id": source_id,
                        "graph_exec_id": exec_id,
                        "raw_fields": {"graph": source_handle, "graphExec": exec_handle},
                    },
                    {
                        "kind": "node_cloned",
                        "node_id": clone_node,
                        "original_node_id": source_node,
                        "node_type": 0,
                        "raw_fields": {
                            "nodeType": 0,
                            "node": ident + 50000,
                            "originalNode": ident + 60000,
                            "graph": exec_handle,
                            "originalGraph": source_handle,
                        },
                    },
                ]
            )
            observed.append(
                {
                    "capture_index": index,
                    "position": position,
                    "actual_graph_exec_id": exec_id,
                    "actual_graph_exec_handle": exec_handle,
                }
            )
            segments.append(
                {
                    "kind": "graph",
                    "position": position,
                    "graph_id": source_id,
                    "capture_id": ident + 2000,
                    "nodes": [
                        {"node_id": source_node, "node_type": 0, "name": call["name"], "call_index": call["index"]}
                    ],
                    "edges": [],
                }
            )
        segments.insert(
            1,
            {
                "kind": "eager",
                "position": 1,
                "name": call["name"],
                "call_index": call["index"],
                "eager_id": 0,
                "range": EAGER_RANGE_PREFIX + "0",
                "source_sha256": BREAKABLE_SOURCE_PIN,
                "source_file": "/TEST_ONLY/breakable_cudagraph.py",
                "qualname": "TEST_ONLY.native_break",
            },
        )
        source = {
            "segments": segments,
            "calls": copy.deepcopy(calls),
            "graph_mutations": False,
            "native_shape_key": {
                key: entry[key] for key in ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras")
            },
            "tp_rank": 0,
            "physical_padded_tokens": entry["num_tokens"],
            "capture_scope": "vllm_piecewise_hidden_states",
            "uncaptured_operations": ["logits"],
            "provenance": proof["provenance"],
            "native_api_libraries": libraries,
            "operations": entries,
            "measurement_method": "native_piecewise_capture_ownership",
        }
        path = tmp_path / f"vllm-piecewise-source-rank-0-capture-0-{index}.json"
        put(path, source)
        sources.append((source, {"file": path.name, "sha256": file_sha256(path)}))
    shared_path = tmp_path / f"{stem}-piecewise-callbacks.json"
    shared = {
        "schema": "glm53flash_piecewise_callbacks_v1",
        "callbacks": callbacks,
        "callback_errors": [],
        "callback_subscription_closed": True,
        "observed_executables": observed,
        "graph_mutations": False,
    }
    put(shared_path, shared)
    shared_ref = {"file": shared_path.name, "sha256": file_sha256(shared_path)}
    for index, (source, source_ref) in enumerate(sources):
        segments = []
        for segment in source["segments"]:
            if segment["kind"] == "eager":
                segments.append(segment)
                continue
            position = segment["position"]
            original = {key: value for key, value in segment.items() if key not in ("position", "kind")}
            original["native_api_libraries"] = libraries
            executable = next(row for row in observed if row["capture_index"] == index and row["position"] == position)
            receipt = {
                "callbacks": callbacks,
                "callback_errors": [],
                "graph_mutations": False,
                "callback_subscription_closed": True,
                "actual_graph_exec_id": executable["actual_graph_exec_id"],
            }
            bound = resolve_registry(original, receipt)
            bound.update(shared_callback_receipt=shared_ref, observed_executable=executable)
            segments.append({"position": position, "kind": "graph", **bound})
        put(
            tmp_path / f"{stem}-piecewise-{index}-bound.json",
            {**source, "segments": segments, "source_receipt": source_ref},
        )
    # JSON roundtrips prevent accidentally depending on shared Python objects.
    return json.loads(json.dumps(policy)), proof["manifest"], proof["provenance"]


def test_piecewise_original_source_shared_callbacks_and_zero_calls_rederive(tmp_path):
    policy, manifest, provenance = piecewise_files(tmp_path)
    files = set()
    captures = serving._piecewise_captures(tmp_path, 0, policy, manifest, provenance, files, {})
    assert set(captures) == {1, 2, 4}
    assert len(files) == 7  # 3 original + 3 bound + one shared raw callback stream
    assert len(captures[4][0]["calls"]) == 276
    assert len(captures[4][0]["segments"]) == 3


@pytest.mark.parametrize(
    "defect",
    [
        "omitted_bucket",
        "omitted_zero_call",
        "wrong_owner",
        "extra_observed_exec",
        "aliased_exec",
        "wrong_source",
        "changed_native_segment_count",
    ],
)
def test_piecewise_copied_or_incomplete_capture_cannot_supply_measurement(tmp_path, defect):
    import json

    from collector.glm53flash_jsonl import file_sha256

    from .test_glm53flash_ops_evidence import put

    policy, manifest, provenance = piecewise_files(tmp_path)
    bound_path = tmp_path / "vllm-graph-clones-rank-0-capture-0-piecewise-2-bound.json"
    bound = json.loads(bound_path.read_bytes())
    if defect == "omitted_bucket":
        bound_path.unlink()
    elif defect == "changed_native_segment_count":
        policy["piecewise_entries"][2]["num_graphs"] += 1
    elif defect == "aliased_exec":
        bound["segments"][0]["observed_executable"]["actual_graph_exec_id"] = 10
        put(bound_path, bound)
    elif defect == "extra_observed_exec":
        path = tmp_path / "vllm-graph-clones-rank-0-capture-0-piecewise-callbacks.json"
        value = json.loads(path.read_bytes())
        value["observed_executables"].append(copy.deepcopy(value["observed_executables"][0]))
        put(path, value)
        for path2 in tmp_path.glob("*-bound.json"):
            value2 = json.loads(path2.read_bytes())
            for segment in value2["segments"]:
                if segment["kind"] == "graph":
                    segment["shared_callback_receipt"]["sha256"] = file_sha256(path)
            put(path2, value2)
    else:
        path = tmp_path / bound["source_receipt"]["file"]
        value = json.loads(path.read_bytes())
        if defect == "omitted_zero_call":
            value["calls"].pop()
        elif defect == "wrong_owner":
            value["segments"][1]["call_index"] = 999
        else:
            value["provenance"]["source_sha256"] = "0" * 64
        put(path, value)
        bound["source_receipt"]["sha256"] = file_sha256(path)
        put(bound_path, bound)
    with pytest.raises(ValueError):
        serving._piecewise_captures(tmp_path, 0, policy, manifest, provenance, set(), {})


def complete_full_files(root, monkeypatch, role="calibration"):
    """Original TEST_ONLY graphs with the real 277-unit model manifest."""
    import json

    from collector import glm53flash_validation as native
    from collector.glm53flash_graph_callbacks import resolve_registry
    from collector.glm53flash_graph_nodes import trace_forward_identity
    from collector.glm53flash_jsonl import file_sha256, iter_records

    from .test_glm53flash_ops_evidence import put, put_lines
    from .test_glm53flash_vllm_graph_export import fixture as old_fixture

    root.mkdir(exist_ok=True)
    run, _ = old_fixture(root, monkeypatch, role)
    run["spec"] = {"ops_execution_mode": "native_serving", "raw_root": "."}
    run["points"] = [
        {
            "benchmark_id": 1,
            "point_type": "decode",
            "batch_size": 1,
            "total_prefill_tokens": 0,
            "total_kv_read_tokens": 128,
        }
    ]
    proof = fixture(batch=1)
    manifest, provenance = proof["manifest"], proof["provenance"]
    put(root / "manifest.json", manifest)
    put(root / "provenance.json", provenance)
    for rank in range(2):
        path = root / f"vllm-graph-policy-rank-{rank}.json"
        policy = json.loads(path.read_bytes())
        policy.update(resolved_mode="FULL_DECODE_ONLY", use_breakable_cg=False, piecewise_entries=[])
        del policy["capture_descriptors"]["PIECEWISE"]
        for candidate in policy["candidates"]:
            candidate["descriptors"] = candidate["descriptors"][:1]
        put(path, policy)
        for path in root.glob(f"vllm-capture-rank-{rank}-*.json"):
            original = json.loads(path.read_bytes())
            callback_path = root / original["instantiation_receipt"]["file"]
            source_path = callback_path.with_name(callback_path.stem + "-source.json")
            source = json.loads(source_path.read_bytes())
            source["provenance"] = provenance
            source["operations"] = [entry for entry in manifest["phases"]["generation"] if entry["name"] != "logits"]
            source["calls"] = [
                {
                    "name": entry["name"],
                    "source": "TEST_ONLY.native_boundary",
                    "completed": True,
                    "owned_node_ids": [source["nodes"][0]["node_id"]] if entry["name"] == "attention_0" else [],
                }
                for entry in source["operations"]
            ]
            put(source_path, source)
            bound = resolve_registry(source, json.loads(callback_path.read_bytes()))
            bound["instantiation_receipt"] = original["instantiation_receipt"]
            put(path, bound)
        for prefix in ("forward", "graph-forward"):
            path = root / f"{prefix}-rank-{rank}.jsonl"
            if not path.exists():
                continue
            rows = list(iter_records(path))
            for row in rows:
                row.update(provenance)
                row["native_dispatch"]["policy_sha256"] = sha256_json(policy)
                row["request_set"] = f"TEST_ONLY-{role}"
                row["request_ids"] = [f"TEST_ONLY-{role}-{row['repetition']}"]
                if role == "holdout":
                    row["dataset_role"] = "holdout"
                if prefix == "graph-forward":
                    row["capture_registry_sha256"] = file_sha256(root / row["capture_registry_file"])
                    trace_path = root / row["replay_nodes"]["trace_file"]
                    trace = json.loads(trace_path.read_bytes())
                    trace["aisim_native_forward"] = trace_forward_identity(row)
                    put(trace_path, trace)
                    row["replay_nodes"]["trace_sha256"] = file_sha256(trace_path)
            put_lines(path, rows)
    for name in native._required_files(2, "vllm") - {"rank-0.jsonl", "rank-1.jsonl"}:
        if not (root / name).exists():
            put(root / name, {"TEST_ONLY_reader_fixture": True})
    return run, root


def test_complete_original_full_reader_exports_schema3_names_and_attempt_closure(tmp_path, monkeypatch):
    run, root = complete_full_files(tmp_path, monkeypatch)
    proof = serving.read_serving_run(root, run)
    rows, selection = serving.aggregate_serving(proof, evidence_sha256="a" * 64)
    assert len(rows) == 278
    assert len(selection["forwards"]) == 15
    assert proof["policy_evidence_sha256"] == sha256_json(proof["policy_evidence"])
    assert sum(row["latency"] for row in rows) == pytest.approx(0.025)
    assert len([row for row in rows if row["contribution_count"] == 0]) == 275


@pytest.mark.parametrize(
    "defect", ["reuse_trace", "copy_trace", "unrelated_config", "missing_target", "unobserved_zero"]
)
def test_serving_original_raw_rechecked_before_named_reduction(tmp_path, monkeypatch, defect):
    import json

    from collector.glm53flash_jsonl import file_sha256, iter_records

    from .test_glm53flash_ops_evidence import put, put_lines

    run, root = complete_full_files(tmp_path, monkeypatch)
    path = root / "graph-forward-rank-0.jsonl"
    rows = list(iter_records(path))
    if defect == "reuse_trace":
        rows[5]["replay_nodes"] = copy.deepcopy(rows[0]["replay_nodes"])
    elif defect == "copy_trace":
        trace = root / rows[5]["replay_nodes"]["trace_file"]
        trace.write_bytes((root / rows[0]["replay_nodes"]["trace_file"]).read_bytes())
        rows[5]["replay_nodes"]["trace_sha256"] = file_sha256(trace)
    elif defect == "unrelated_config":
        value = json.loads((root / "provenance.json").read_bytes())
        value["config_sha256"] = "f" * 64
        put(root / "provenance.json", value)
    elif defect == "missing_target":
        rows.pop()
    else:
        source = root / "vllm-graph-clones-rank-0-capture-0-0-source.json"
        value = json.loads(source.read_bytes())
        value["calls"].pop()
        put(source, value)
    put_lines(path, rows)
    with pytest.raises(ValueError):
        serving.read_serving_run(root, run)


def test_export_bind_preserves_control_and_original_table_provenance(tmp_path, monkeypatch):
    import json

    from collector import glm53flash_validation as native
    from collector.glm53flash_jsonl import file_sha256

    cal_run, cal = complete_full_files(tmp_path / "cal", monkeypatch)
    control_run, control = complete_full_files(tmp_path / "control", monkeypatch, "control")

    def truth(run, root, **kwargs):
        # Native history/hardware has separate common-loader tests. This test
        # only substitutes that prerequisite, never capture/trace/control data.
        return {
            "evidence_root": str(root.resolve()),
            "runtime_run_id": f"TEST_ONLY-{run['role']}",
            "receipts": [{"path": path.name, "sha256": file_sha256(path)} for path in root.iterdir() if path.is_file()],
        }

    monkeypatch.setattr(native, "load_native", truth)
    monkeypatch.setattr(native, "_load_native", truth)
    output = tmp_path / serving.BASENAME
    result = serving.export_serving(cal, cal_run, output, control_root=control, control_run=control_run)
    bound = serving.bind_calibration([output], cal_run, truth(cal_run, cal))
    assert result["rows"] == bound["rows"] == 278
    assert result["table_sha256"] == bound["tables"][0]["sha256"] == file_sha256(output)
    receipt = json.loads((cal / "serving-calibration-evidence.json").read_bytes())
    assert receipt["policy_evidence_sha256"] == serving.read_serving_run(cal, cal_run)["policy_evidence_sha256"]
    changed = json.loads((control / "resolved-config-node0.json").read_bytes())
    changed["config"]["engine_args"]["max_num_seqs"] = 8
    from .test_glm53flash_ops_evidence import put

    put(control / "resolved-config-node0.json", changed)
    with pytest.raises(ValueError, match="EngineArgs"):
        serving.bind_calibration([output], cal_run, truth(cal_run, cal))


def test_piecewise_trace_rebinds_exact_native_segment_activity_and_each_forward(tmp_path):
    from collector.glm53flash_graph_nodes import trace_forward_identity
    from collector.glm53flash_jsonl import file_sha256
    from collector.glm53flash_vllm_graph_ops import LOGITS_SOURCE_PIN
    from collector.glm53flash_vllm_piecewise_activity import bind_piecewise_execution

    from .test_glm53flash_ops_evidence import put
    from .test_glm53flash_vllm_piecewise_activity import fixture as activity_fixture

    policy, manifest, provenance = piecewise_files(tmp_path)
    registry, _ = serving._piecewise_captures(tmp_path, 0, policy, manifest, provenance, set(), {})[4]
    _, events = activity_fixture()
    for event in events:
        args = event.get("args", {})
        if args.get("graph id") in (10, 20):
            args["graph id"] += 200
            args["graph node id"] = args["graph id"] + 4000
    row = fixture(phase="context")["forwards"][(1, 5)][0]
    row["run_id"] = "TEST_ONLY-run-id"
    path = tmp_path / f"graph-profile-rank-0-forward-{row['invocation']}.json"
    put(
        path,
        {
            "traceEvents": events,
            "aisim_native_forward": trace_forward_identity(row),
            "aisim_native_execution": {
                "backend": "vllm",
                "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
                "logits_source_sha256": LOGITS_SOURCE_PIN,
                "failed": False,
                "runtime_mode": "PIECEWISE",
            },
        },
    )
    binding = bind_piecewise_execution(registry, events)
    binding.update(trace_file=path.name, trace_sha256=file_sha256(path))
    row["replay_nodes"] = binding
    rederived = serving._replay_binding(tmp_path, row, registry, set())
    units = {unit["operation"]: unit for unit in rederived["operation_activity_unions"]}
    assert units["attention_3"]["active_union_us"] == 40  # union of graph+eager, not three charges
    assert units["native_graph_setup"]["active_union_us"] == 2
    with pytest.raises(ValueError, match="uniquely"):
        serving._replay_binding(tmp_path, row, registry, {path.name})
    copied = copy.deepcopy(row)
    copied["invocation"] += 1
    copied["forward_id"] = f"rank-0/forward-{copied['invocation']}"
    copied_path = tmp_path / f"graph-profile-rank-0-forward-{copied['invocation']}.json"
    copied_path.write_bytes(path.read_bytes())
    copied["replay_nodes"]["trace_file"] = copied_path.name
    with pytest.raises(ValueError, match="another native forward"):
        serving._replay_binding(tmp_path, copied, registry, set())


def test_exported_schema3_table_uses_actual_public_rust_and_returns_binding(tmp_path, monkeypatch):
    import shutil
    from importlib.resources import files

    from collector import glm53flash_validation as native
    from collector.glm53flash_contract import CHECKPOINTS
    from collector.glm53flash_jsonl import file_sha256

    cal_run, cal = complete_full_files(tmp_path / "cal", monkeypatch)
    control_run, control = complete_full_files(tmp_path / "control", monkeypatch, "control")
    holdout_run, holdout = complete_full_files(tmp_path / "holdout", monkeypatch, "holdout")

    def truth(run, root, **kwargs):
        proof = serving.read_serving_run(root, run)
        return {
            **{key: proof[key] for key in ("execution_policy", "_execution_policy")},
            "graph_policy": proof["policy"],
            "evidence_root": str(root.resolve()),
            "runtime_run_id": f"TEST_ONLY-{run['role']}",
            "receipts": [{"path": path.name, "sha256": file_sha256(path)} for path in root.iterdir() if path.is_file()],
        }

    monkeypatch.setattr(native, "load_native", truth)
    monkeypatch.setattr(native, "_load_native", truth)
    systems = tmp_path / "systems"
    data = systems / "data/gb300/vllm/0.30.0"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    output = data / serving.BASENAME
    serving.export_serving(cal, cal_run, output, control_root=control, control_run=control_run)
    native_cal = truth(cal_run, cal)
    bound = serving.bind_calibration([output], cal_run, native_cal)
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
    result = serving.predict_homogeneous(holdout_run, holdout, config, native_cal, bound)
    assert result["rows"] == {1: {"prediction_ms": pytest.approx(0.025)}}
    assert result["calibration_binding"] == bound and bound["tables"][0]["sha256"] == file_sha256(output)
    bad = copy.deepcopy(bound)
    bad["tables"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound actual calibration"):
        serving.predict_homogeneous(holdout_run, holdout, config, native_cal, bad)
