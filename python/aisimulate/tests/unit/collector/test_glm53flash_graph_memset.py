# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY pending ownership and replay contracts, never native qualification.

The rc0/source2/clone0 case was observed in our failed935 jobs627422/627423.
All live parameters and replay activities here are synthetic: those failed runs
have neither. No external implementation or historical measurements are copied.
"""

import copy
import ctypes
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from collector import glm53flash_graph_callbacks as callbacks
from collector import glm53flash_graph_nodes as nodes

from .test_glm53flash_graph_nodes import native_abi_fixture
from .test_glm53flash_native_classification import graph_fixture, replay

pytestmark = pytest.mark.unit


def parameters(handle, libraries):
    return {
        "contract": nodes.MEMSET_PENDING_CONTRACT,
        "api": "cudaGraphMemsetNodeGetParams",
        "rc": 0,
        "source_node_handle": handle,
        "parameters": {"dst": 4096, "pitch": 0, "value": 0, "elementSize": 4, "width": 32, "height": 1},
        "native_api_libraries": copy.deepcopy(libraries),
        "trace_producer": {
            "torch_git_version": nodes.MEMCPY_TORCH_REVISION,
            "kineto_gitlink": nodes.MEMCPY_KINETO_REVISION,
        },
    }


def source_parameters(source, receipt):
    for node in source["nodes"]:
        if node["node_type"] == 2:
            clone = next(row for row in receipt["callbacks"] if row.get("original_node_id") == node["node_id"])
            node["memset_params"] = parameters(clone["raw_fields"]["originalNode"], source["native_api_libraries"])


def pending(tmp_path, *, rc=0, kind=0):
    source, receipt, api, graph, path, calls = graph_fixture(tmp_path, rc=rc, queried_type=kind)
    source_parameters(source, receipt)
    return source, receipt, api, graph, path, calls


def record(source, receipt, api, graph, path):
    return callbacks.record_event_record_types(
        api, graph, source, receipt, path, allow_pending_memcpy=True, allow_pending_memset=True
    )


def resolve(source, receipt, proof):
    return callbacks.resolve_registry(source, receipt, proof, allow_pending_memcpy=True, allow_pending_memset=True)


def events(registry):
    rows = replay(registry)
    rows[-1]["name"] = nodes.MEMSET_TRACE_NAME
    return rows


@pytest.mark.parametrize("kind", [0, 2])
def test_pending_observation_preserves_actual_type_and_needs_positive_replay(tmp_path, kind):
    source, receipt, api, graph, path, calls = pending(tmp_path, kind=kind)
    originals = copy.deepcopy((source, receipt))
    proof = record(source, receipt, api, graph, path)
    assert proof["method"] == "CUDA13_LIVE_SOURCE_MEMSET_PENDING_REPLAY_V1"
    assert proof["queries"][0]["native_node_type"] == kind
    assert proof["queries"][0]["source_node_type"] == 2
    registry = resolve(source, receipt, proof)
    assert registry["memset_pending_contract"] == nodes.MEMSET_PENDING_CONTRACT
    assert registry["nodes"][-1]["memset_activity_requirement"]["bytes"] == 128
    assert registry["native_instantiation"]["event_record_type_proof"] == proof
    assert (source, receipt) == originals and len(calls) == 1
    with pytest.raises(ValueError, match="omits captured"):
        nodes.bind_replay_kernels(registry, [], correlation=12)
    bound = nodes.bind_replay_kernels(registry, events(registry), correlation=12)
    assert bound["activities"][-1]["activity"] == "gpu_memset"
    assert bound["activities"][-1]["fingerprint"]["bytes"] == 128
    assert bound["activities"][-1]["operation"] == source["nodes"][-1]["name"]
    assert bound["formal_admission"] is False
    # A new proof cannot silently become a historical strict proof.
    with pytest.raises(ValueError, match="exact deferred"):
        callbacks.resolve_registry(source, receipt, proof, allow_pending_memcpy=True, allow_memset_query=True)


@pytest.mark.parametrize("rc,kind", [(1, 0), (0, 7), (0, 5)])
def test_new_contract_preserves_failed_native_queries(tmp_path, rc, kind):
    source, receipt, api, graph, path, _ = pending(tmp_path, rc=rc, kind=kind)
    with pytest.raises(RuntimeError, match="failed or changed"):
        record(source, receipt, api, graph, path)
    proof = json.loads(path.read_text())
    assert proof["completed"] is False and proof["queries"][0]["rc"] == rc
    assert proof["queries"][0]["native_node_type"] == kind
    with pytest.raises(ValueError):
        resolve(source, receipt, proof)


@pytest.mark.parametrize(
    "defect",
    [
        "rc",
        "handle",
        "provider",
        "trace_producer",
        "missing",
        "contract",
        "dst",
        "width",
        "height",
        "elementSize",
        "bool",
        "overflow",
        "value",
        "pitch",
        "open",
        "graph",
        "missing_clone",
        "duplicate_clone",
    ],
)
def test_pending_source_requires_exact_live_parameters_and_closed_bijection(tmp_path, defect):
    source, receipt, api, graph, path, calls = pending(tmp_path)
    proof = source["nodes"][-1]["memset_params"]
    if defect == "rc":
        proof["rc"] = 1
    elif defect == "handle":
        proof["source_node_handle"] += 1
    elif defect == "provider":
        proof["native_api_libraries"]["cudart"]["sha256"] = "0" * 64
    elif defect == "trace_producer":
        proof["trace_producer"]["torch_git_version"] = "unknown"
    elif defect == "missing":
        source["nodes"][-1].pop("memset_params")
    elif defect == "contract":
        proof["contract"] += "unknown"
    elif defect == "open":
        receipt["callback_subscription_closed"] = False
    elif defect == "graph":
        graph.raw_cuda_graph_exec = lambda: 123456
    elif defect == "missing_clone":
        receipt["callbacks"].pop()
    elif defect == "duplicate_clone":
        receipt["callbacks"].append(copy.deepcopy(receipt["callbacks"][-1]))
    else:
        key, value = {
            "dst": ("dst", 0),
            "width": ("width", 0),
            "height": ("height", 2),
            "elementSize": ("elementSize", 8),
            "bool": ("width", True),
            "overflow": ("width", 2**64),
            "value": ("value", -1),
            "pitch": ("pitch", -1),
        }[defect]
        proof["parameters"][key] = value
    with pytest.raises((ValueError, RuntimeError)):
        record(source, receipt, api, graph, path)
    assert not calls and not path.exists()


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "kernel",
        "wrong_bytes",
        "wrong_name",
        "graph",
        "node",
        "launch",
        "zero",
        "negative",
        "nan",
        "missing_requirement",
        "missing_contract",
    ],
)
def test_actual_memset_activity_cannot_be_inferred_or_substituted(tmp_path, defect):
    source, receipt, api, graph, path, _ = pending(tmp_path)
    registry = resolve(source, receipt, record(source, receipt, api, graph, path))
    rows = events(registry)
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif defect == "kernel":
        rows[-1]["cat"] = "kernel"
    elif defect == "wrong_bytes":
        rows[-1]["args"]["bytes"] += 1
    elif defect == "wrong_name":
        rows[-1]["name"] = "Memset (Managed)"
    elif defect in ("graph", "node", "launch"):
        rows[-1]["args"][{"graph": "graph id", "node": "graph node id", "launch": "correlation"}[defect]] += 1
    elif defect == "missing_requirement":
        registry["nodes"][-1].pop("memset_activity_requirement")
    elif defect == "missing_contract":
        registry.pop("memset_pending_contract")
    else:
        rows[-1]["dur"] = {"zero": 0, "negative": -1, "nan": float("nan")}[defect]
    with pytest.raises(ValueError):
        nodes.bind_replay_kernels(registry, rows, correlation=12)


def test_native_memset_parameter_abi_queries_only_live_source_when_opted_in(monkeypatch):
    native_abi_fixture(monkeypatch, node_count=1)
    runtime, _ = nodes._library("cudart")
    calls = []

    def query(handle, out):
        calls.append(handle)
        params = ctypes.cast(out, ctypes.POINTER(nodes.GraphMemsetParams)).contents
        params.dst, params.elementSize, params.width, params.height = 4096, 4, 32, 1
        return 0

    runtime.cudaGraphMemsetNodeGetParams = query
    runtime.cudaGraphNodeGetType = (
        lambda handle, out: setattr(ctypes.cast(out, ctypes.POINTER(ctypes.c_int)).contents, "value", 2) or 0
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(git_version=nodes.MEMCPY_TORCH_REVISION))
    )
    assert nodes.NativeGraphAPI().snapshot(77)["nodes"][1101] == {"node_type": 2}
    assert calls == []
    api = nodes.NativeGraphAPI(capture_memset_parameters=True)
    api.libraries["cudart"]["sha256"] = callbacks.EVENT_RECORD_CUDART_SHA256
    api.libraries["cupti"]["sha256"] = callbacks.QUALIFIED_CUPTI_SHA256
    observed = api.snapshot(77)["nodes"][1101]
    assert calls == [101] and observed["memset_params"]["source_node_handle"] == 101
    assert nodes.memset_activity_requirement(observed)["bytes"] == 128
    assert ctypes.sizeof(nodes.GraphMemsetParams) == 40
    assert [getattr(nodes.GraphMemsetParams, name).offset for name, _ in nodes.GraphMemsetParams._fields_] == [
        0,
        8,
        16,
        20,
        24,
        32,
    ]


@pytest.mark.parametrize("defect", ["source_type", "query_type", "query_rc", "handle", "method", "incomplete"])
def test_saved_observation_cannot_be_relabelled(tmp_path, defect):
    source, receipt, api, graph, path, _ = pending(tmp_path)
    proof = record(source, receipt, api, graph, path)
    if defect == "method":
        proof["method"] = "CUDA13_MEMSET_EVENT_RECORD_CLONE_QUERY_V1"
    elif defect == "incomplete":
        proof["completed"] = False
    else:
        key, value = {
            "source_type": ("source_node_type", 0),
            "query_type": ("native_node_type", 7),
            "query_rc": ("rc", False),
            "handle": ("clone_node_handle", 123),
        }[defect]
        proof["queries"][0][key] = value
    with pytest.raises(ValueError):
        resolve(source, receipt, proof)


def reference(path):
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def put(path, value):
    path.write_text(json.dumps(value))


def add_memset(source, receipt):
    """Add a TEST_ONLY memory node owned by the same real fixture call."""
    node = copy.deepcopy(source["nodes"][0])
    original = node["node_id"]
    node.update(node_id=original + 1, node_type=2)
    clone = copy.deepcopy(next(row for row in receipt["callbacks"] if row.get("original_node_id") == original))
    clone.update(original_node_id=node["node_id"], node_id=clone["node_id"] + 1)
    clone["raw_fields"]["node"] += 1
    clone["raw_fields"]["originalNode"] += 1
    source["nodes"].append(node)
    receipt["callbacks"].append(clone)
    source_parameters(source, receipt)
    return node


def libraries():
    return {
        "cupti": {"path": "/TEST_ONLY/libcupti.so", "sha256": callbacks.QUALIFIED_CUPTI_SHA256},
        "cudart": {
            "path": "/TEST_ONLY/libcudart.so",
            "sha256": callbacks.EVENT_RECORD_CUDART_SHA256,
            "runtime_version": 13000,
            "abi": "CUDA13_capture7_edges5",
        },
    }


def observed_query(source, receipt, path):
    def query(handle, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int)).contents.value = 0
        return 0

    api = SimpleNamespace(libraries=source["native_api_libraries"], runtime=SimpleNamespace(cudaGraphNodeGetType=query))
    create = next(row for row in receipt["callbacks"] if row.get("graph_exec_id") == receipt["actual_graph_exec_id"])
    graph = SimpleNamespace(raw_cuda_graph_exec=lambda: create["raw_fields"]["graphExec"])
    return record(source, receipt, api, graph, path)


@pytest.mark.parametrize(
    "defect", [None, "missing_contract", "unknown_contract", "missing_params", "source_bytes", "missing_query"]
)
@pytest.mark.parametrize("contract", nodes.MEMSET_CONTRACTS)
def test_full_exporter_rederives_new_source_query_contract(tmp_path, monkeypatch, defect, contract):
    from collector import glm53flash_vllm_graph_export as export

    from .test_glm53flash_vllm_graph_export import fixture

    fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "vllm-graph-clones-rank-0-capture-0-0-source.json"
    callback_path = tmp_path / "vllm-graph-clones-rank-0-capture-0-0.json"
    source, receipt = json.loads(source_path.read_text()), json.loads(callback_path.read_text())
    source["native_api_libraries"] = libraries()
    receipt["callback_subscription_closed"] = True
    node = add_memset(source, receipt)
    node["memset_params"]["contract"] = contract
    source["calls"][0]["owned_node_ids"].append(node["node_id"])
    path = tmp_path / "TEST_ONLY-pending-memset.json"
    bound = resolve(source, receipt, observed_query(source, receipt, path))
    put(source_path, source)
    put(callback_path, receipt)
    bound.update(instantiation_receipt=reference(callback_path), node_type_receipt=reference(path))
    if defect == "missing_contract":
        bound.pop("memset_pending_contract")
    elif defect == "unknown_contract":
        bound["memset_pending_contract"] = "unknown"
    elif defect == "missing_params":
        node.pop("memset_params")
        put(source_path, source)
    elif defect == "source_bytes":
        node["memset_params"]["parameters"]["width"] += 1
        put(source_path, source)
    elif defect == "missing_query":
        bound.pop("node_type_receipt")
    put(tmp_path / "vllm-capture-rank-0-0.json", bound)
    args = (
        tmp_path,
        0,
        json.loads((tmp_path / "vllm-graph-policy-rank-0.json").read_text()),
        json.loads((tmp_path / "manifest.json").read_text()),
        json.loads((tmp_path / "provenance.json").read_text()),
        set(),
    )
    if defect:
        with pytest.raises(ValueError):
            export._captures(*args)
    else:
        captures = export._captures(*args)
        assert next(value[0] for value in captures.values() if value[0]["graph_id"] == bound["graph_id"]) == bound
        row = {
            "tp_rank": 0,
            "invocation": 1,
            "replay_nodes": {"trace_file": "graph-profile-rank-0-forward-1.json", "trace_sha256": "a" * 64},
        }
        with pytest.raises((ValueError, FileNotFoundError)):
            export._binding(tmp_path, row, bound, set())
        row = json.loads((tmp_path / "graph-forward-rank-0.jsonl").read_text().splitlines()[0])
        trace_path = tmp_path / row["replay_nodes"]["trace_file"]
        trace = json.loads(trace_path.read_text())
        memory = {
            "cat": "gpu_memset",
            "name": nodes.MEMSET_TRACE_NAME if contract == nodes.MEMSET_PENDING_CONTRACT else "Memset (Unknown)",
            "ts": 11,
            "dur": 2,
            "args": {
                "graph id": bound["graph_id"],
                "graph node id": bound["nodes"][-1]["node_id"],
                "correlation": 2,
                "stream": 1,
                "bytes": 128,
            },
        }
        trace["traceEvents"].append(memory)
        put(trace_path, trace)
        binding = nodes.bind_vllm_execution_activity(
            nodes.bind_replay_kernels(bound, trace["traceEvents"], correlation=2), trace["traceEvents"]
        )
        binding.update(trace_file=trace_path.name, trace_sha256=reference(trace_path)["sha256"])
        row["replay_nodes"] = binding
        assert export._binding(tmp_path, row, bound, set()) == binding
        memory["args"]["bytes"] += 1
        put(trace_path, trace)
        row["replay_nodes"]["trace_sha256"] = reference(trace_path)["sha256"]
        with pytest.raises(ValueError, match="source bytes"):
            export._binding(tmp_path, row, bound, set())


def piecewise_pending_files(tmp_path, contract=nodes.MEMSET_PENDING_CONTRACT):
    from .test_glm53flash_vllm_serving_export import piecewise_files

    policy, manifest, provenance = piecewise_files(tmp_path)
    stem = "vllm-graph-clones-rank-0-capture-0"
    shared_path = tmp_path / f"{stem}-piecewise-callbacks.json"
    shared = json.loads(shared_path.read_text())
    sources = []
    for index in range(3):
        source_path = tmp_path / f"vllm-piecewise-source-rank-0-capture-0-{index}.json"
        source = json.loads(source_path.read_text())
        source["native_api_libraries"] = libraries()
        segment = source["segments"][0]
        segment["native_api_libraries"] = source["native_api_libraries"]
        add_memset(segment, shared)["memset_params"]["contract"] = contract
        segment.pop("native_api_libraries")
        put(source_path, source)
        sources.append((source, source_path))
    put(shared_path, shared)
    for index, (source, source_path) in enumerate(sources):
        segments = []
        for segment in source["segments"]:
            if segment["kind"] == "eager":
                segments.append(segment)
                continue
            original = {key: value for key, value in segment.items() if key not in ("position", "kind")}
            original["native_api_libraries"] = source["native_api_libraries"]
            observed = next(
                row
                for row in shared["observed_executables"]
                if row["capture_index"] == index and row["position"] == segment["position"]
            )
            receipt = {
                key: shared[key]
                for key in ("callbacks", "callback_errors", "callback_subscription_closed", "graph_mutations")
            }
            receipt["actual_graph_exec_id"] = observed["actual_graph_exec_id"]
            path = tmp_path / f"{stem}-piecewise-{index}-{segment['position']}-event-types.json"
            proof = observed_query(original, receipt, path)
            bound = resolve(original, receipt, proof)
            bound.update(shared_callback_receipt=reference(shared_path), observed_executable=observed)
            if proof is not None:
                bound["node_type_receipt"] = reference(path)
            segments.append({"position": segment["position"], "kind": "graph", **bound})
        put(
            tmp_path / f"{stem}-piecewise-{index}-bound.json",
            {**source, "segments": segments, "source_receipt": reference(source_path)},
        )
    return policy, manifest, provenance


@pytest.mark.parametrize(
    "defect", [None, "missing_contract", "unknown_contract", "missing_source", "wrong_bytes", "missing_query"]
)
@pytest.mark.parametrize("contract", nodes.MEMSET_CONTRACTS)
def test_piecewise_exporter_rederives_new_source_query_contract(tmp_path, defect, contract):
    from collector import glm53flash_vllm_serving_export as export

    policy, manifest, provenance = piecewise_pending_files(tmp_path, contract)
    path = tmp_path / "vllm-graph-clones-rank-0-capture-0-piecewise-0-bound.json"
    bound = json.loads(path.read_text())
    segment = bound["segments"][0]
    if defect == "missing_contract":
        segment.pop("memset_pending_contract")
    elif defect == "unknown_contract":
        segment["memset_pending_contract"] = "unknown"
    elif defect == "missing_source":
        segment["nodes"][-1].pop("memset_params")
    elif defect == "wrong_bytes":
        segment["nodes"][-1]["memset_activity_requirement"]["bytes"] += 1
    elif defect == "missing_query":
        segment.pop("node_type_receipt")
    put(path, bound)
    args = (tmp_path, 0, policy, manifest, provenance, set(), {})
    if defect:
        with pytest.raises(ValueError):
            export._piecewise_captures(*args)
    else:
        captures = export._piecewise_captures(*args)
        assert set(captures) == {1, 2, 4}
        assert all(value[0]["segments"][0]["memset_pending_contract"] == contract for value in captures.values())
        row = {
            "runtime_mode": "PIECEWISE",
            "tp_rank": 0,
            "invocation": 1,
            "replay_nodes": {"trace_file": "graph-profile-rank-0-forward-1.json", "trace_sha256": "a" * 64},
        }
        with pytest.raises((ValueError, FileNotFoundError)):
            export._replay_binding(tmp_path, row, captures[4][0], set())
        from collector.glm53flash_vllm_graph_ops import LOGITS_SOURCE_PIN
        from collector.glm53flash_vllm_piecewise_activity import bind_piecewise_execution

        from .test_glm53flash_vllm_piecewise_activity import fixture as activity_fixture
        from .test_glm53flash_vllm_serving_export import fixture as serving_fixture

        registry = captures[4][0]
        _, activity = activity_fixture()
        for event in activity:
            args = event.get("args", {})
            if args.get("graph id") in (10, 20):
                args["graph id"] += 200
                args["graph node id"] = args["graph id"] + 4000
        segment = registry["segments"][0]
        memory = {
            "cat": "gpu_memset",
            "name": nodes.MEMSET_TRACE_NAME if contract == nodes.MEMSET_PENDING_CONTRACT else "Memset (Unknown)",
            "ts": 16,
            "dur": 2,
            "args": {
                "graph id": segment["graph_id"],
                "graph node id": segment["nodes"][-1]["node_id"],
                "correlation": 2,
                "stream": 1,
                "bytes": 128,
            },
        }
        activity.append(memory)
        row = serving_fixture(phase="context")["forwards"][(1, 5)][0]
        row["run_id"] = "TEST_ONLY-run-id"
        trace_path = tmp_path / f"graph-profile-rank-0-forward-{row['invocation']}.json"
        trace = {
            "traceEvents": activity,
            "aisim_native_forward": nodes.trace_forward_identity(row),
            "aisim_native_execution": {
                "backend": "vllm",
                "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
                "logits_source_sha256": LOGITS_SOURCE_PIN,
                "failed": False,
                "runtime_mode": "PIECEWISE",
            },
        }
        put(trace_path, trace)
        binding = bind_piecewise_execution(registry, activity)
        binding.update(trace_file=trace_path.name, trace_sha256=reference(trace_path)["sha256"])
        row["replay_nodes"] = binding
        assert export._replay_binding(tmp_path, row, registry, set()) == binding
        memory["args"]["bytes"] += 1
        put(trace_path, trace)
        row["replay_nodes"]["trace_sha256"] = reference(trace_path)["sha256"]
        with pytest.raises(ValueError, match="source bytes"):
            export._replay_binding(tmp_path, row, registry, set())


def test_live_memset_parameter_mutation_is_not_silently_reowned():
    libs = libraries()
    state = {
        "capture_id": 1,
        "graph_id": 2,
        "nodes": {9: {"node_type": 2, "memset_params": parameters(99, libs)}},
        "edges": [],
    }
    registry = nodes.CaptureNodeRegistry(lambda: copy.deepcopy(state))
    token = registry.enter("TEST_ONLY", "TEST_ONLY.native")
    state["nodes"][9]["memset_params"]["parameters"]["width"] += 1
    with pytest.raises(RuntimeError, match="parameters changed"):
        registry.leave(token)


def test_source_query_never_runs_outside_active_capture(monkeypatch):
    native_abi_fixture(monkeypatch, node_count=1)
    runtime, _ = nodes._library("cudart")
    calls = []
    runtime.cudaGraphMemsetNodeGetParams = lambda *_: calls.append("unexpected") or 0
    original = runtime.cudaStreamGetCaptureInfo

    def inactive(*args):
        result = original(*args)
        ctypes.cast(args[1], ctypes.POINTER(ctypes.c_int)).contents.value = 0
        return result

    runtime.cudaStreamGetCaptureInfo = inactive
    with pytest.raises(RuntimeError, match="active native capture"):
        nodes.NativeGraphAPI(capture_memset_parameters=True).snapshot(77)
    assert not calls
