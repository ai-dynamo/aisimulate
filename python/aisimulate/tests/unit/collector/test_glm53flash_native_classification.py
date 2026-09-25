# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY reductions of our observed APIs/nodes; no model qualification.

See fixtures/glm53flash_native_classification_observed.README.md for original
artifact hashes and reduction scope. Native query/replay doubles below are
explicitly synthetic and never appended to the original failed runs.
"""

import copy
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector import glm53flash_graph_callbacks as callbacks
from collector import glm53flash_graph_nodes as nodes

pytestmark = pytest.mark.unit


def observed():
    return json.loads((Path(__file__).parent / "fixtures/glm53flash_native_classification_observed.json").read_text())


def eager_events():
    fixture = observed()["none"]
    return [
        dict(fixture["region"], name=nodes.EXECUTION_RANGE),
        *fixture["control_calls"],
        fixture["launch"],
        fixture["activity"],
    ]


def test_observed_exact_lookup_and_configuration_apis_preserve_only_measured_work():
    events = eager_events()
    original = copy.deepcopy(events)
    result = nodes.bind_native_eager_activity(events)
    assert {call["name"] for call in events[1:4]} == nodes.FUNCTION_CONTROL_APIS
    assert len(result["activities"]) == 1
    assert result["activities"][0]["launch_correlation"] == events[-2]["args"]["correlation"]
    baseline = nodes.bind_native_eager_activity([events[0], *events[-2:]])
    assert result["activity_union_us"] == baseline["activity_union_us"] > 0
    assert result["formal_admission"] is False
    assert events == original


@pytest.mark.parametrize("index", range(1, 4))
@pytest.mark.parametrize("defect", ["unknown_suffix", "unexpected_activity"])
def test_lookup_names_cannot_hide_unknown_dispatch_or_device_work(index, defect):
    events = eager_events()
    if defect == "unknown_suffix":
        events[index]["name"] += "UnknownDispatch"
        match = "unknown native CUDA dispatch"
    else:
        activity = copy.deepcopy(events[-1])
        activity["args"]["correlation"] = events[index]["args"]["correlation"]
        events.append(activity)
        match = "unexpectedly owns device activity"
    with pytest.raises(ValueError, match=match):
        nodes.bind_native_eager_activity(events)


def graph_fixture(tmp_path, *, rc=0, queried_type=2):
    fixture = observed()["graph"]
    source, receipt = fixture["source"], fixture["callbacks"]
    calls = []
    path = tmp_path / "TEST_ONLY-deferred-types.json"
    clone = next(
        row
        for row in receipt["callbacks"]
        if row["kind"] == "node_cloned" and row["original_node_id"] == source["nodes"][2]["node_id"]
    )

    def query(handle, out):
        assert handle == clone["raw_fields"]["node"]
        assert receipt["callback_subscription_closed"] is True
        pending = json.loads(path.read_text())
        assert pending["queries"][-1]["rc"] is None and pending["completed"] is False
        calls.append(handle)
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int)).contents.value = queried_type
        return rc

    api = SimpleNamespace(libraries=source["native_api_libraries"], runtime=SimpleNamespace(cudaGraphNodeGetType=query))
    graph = SimpleNamespace(raw_cuda_graph_exec=lambda: receipt["callbacks"][0]["raw_fields"]["graphExec"])
    return source, receipt, api, graph, path, calls


def resolve(source, receipt, proof=None):
    return callbacks.resolve_registry(source, receipt, proof, allow_pending_memcpy=True, allow_memset_query=True)


def query_proof(source, receipt, api, graph, path):
    return callbacks.record_event_record_types(
        api, graph, source, receipt, path, allow_pending_memcpy=True, allow_memset_query=True
    )


def replay(registry):
    # TEST_ONLY intervals and kernel name; only node/category identities come
    # from our original source/callback records. No real replay was completed.
    events = []
    for index, node in enumerate(registry["nodes"]):
        kind = node["node_type"]
        args = {"correlation": 12, "graph id": registry["graph_id"], "graph node id": node["node_id"], "stream": 1}
        if kind == 0:
            args.update(grid=[1, 1, 1], block=[32, 1, 1], **{"shared memory": 0})
        else:
            args["bytes"] = node["memcpy_activity_requirement"]["bytes"] if kind == 1 else 128
        events.append(
            {
                "cat": {0: "kernel", 1: "gpu_memcpy", 2: "gpu_memset"}[kind],
                "name": nodes.MEMCPY_TRACE_NAME if kind == 1 else "TEST_ONLY",
                "ts": index + 1,
                "dur": 1,
                "args": args,
            }
        )
    return events


def test_observed_memset_copy_mapping_requires_deferred_query_and_replay(tmp_path):
    source, receipt, api, graph, path, calls = graph_fixture(tmp_path)
    original = copy.deepcopy((source, receipt))
    with pytest.raises(ValueError, match="unqualified changed node type"):
        callbacks.resolve_registry(source, receipt, allow_pending_memcpy=True)
    with pytest.raises(ValueError, match="exact deferred native type proof"):
        resolve(source, receipt)
    proof = query_proof(source, receipt, api, graph, path)
    assert len(calls) == 1
    assert proof["method"] == "CUDA13_MEMSET_EVENT_RECORD_CLONE_QUERY_V1"
    assert proof["queries"][0]["source_node_type"] == proof["queries"][0]["native_node_type"] == 2
    assert proof["queries"][0]["callback_node_type"] == 0
    registry = resolve(source, receipt, proof)
    assert [n["node_type"] for n in registry["nodes"]] == [0, 1, 2]
    assert (source, receipt) == original
    events = replay(registry)
    bound = nodes.bind_replay_kernels(registry, events, correlation=12)
    assert [x["activity"] for x in bound["activities"]] == ["kernel", "gpu_memcpy", "gpu_memset"]
    assert bound["unmeasured_structural_nodes"] == []
    assert bound["formal_admission"] is False


@pytest.mark.parametrize("rc,kind", [(1, 2), (0, 0), (0, 7)])
def test_memset_query_failure_is_preserved_not_normalized(tmp_path, rc, kind):
    source, receipt, api, graph, path, calls = graph_fixture(tmp_path, rc=rc, queried_type=kind)
    with pytest.raises(RuntimeError, match="failed or changed type"):
        query_proof(source, receipt, api, graph, path)
    proof = json.loads(path.read_text())
    assert len(calls) == 1 and proof["completed"] is False
    assert proof["queries"][0]["rc"] == rc and proof["queries"][0]["native_node_type"] == kind
    with pytest.raises(ValueError):
        resolve(source, receipt, proof)


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_provider",
        "missing_clone",
        "duplicate_clone",
        "foreign_source",
        "open_subscription",
        "empty_node",
        "wait_node",
        "host_node",
    ],
)
def test_observed_mapping_does_not_relax_native_identity_or_other_types(tmp_path, defect):
    source, receipt, api, graph, path, calls = graph_fixture(tmp_path)
    if defect == "wrong_provider":
        source["native_api_libraries"]["cudart"]["sha256"] = "0" * 64
    elif defect == "missing_clone":
        receipt["callbacks"].pop()
    elif defect == "duplicate_clone":
        receipt["callbacks"].append(copy.deepcopy(receipt["callbacks"][-1]))
    elif defect == "foreign_source":
        receipt["callbacks"][-1]["raw_fields"]["originalGraph"] += 1
    elif defect == "open_subscription":
        receipt["callback_subscription_closed"] = False
    else:
        source["nodes"][2]["node_type"] = {"empty_node": 5, "wait_node": 6, "host_node": 3}[defect]
    with pytest.raises((ValueError, RuntimeError)):
        query_proof(source, receipt, api, graph, path)
    assert not calls and not path.exists()


@pytest.mark.parametrize("defect", ["method", "source_type", "bool_rc", "wrong_handle", "missing_query", "extra_query"])
def test_memset_proof_cannot_be_relabelled_or_crossbound(tmp_path, defect):
    source, receipt, api, graph, path, _ = graph_fixture(tmp_path)
    proof = query_proof(source, receipt, api, graph, path)
    if defect == "method":
        proof["method"] = "CUDA13_EVENT_RECORD_CLONE_QUERY_V1"
    elif defect == "source_type":
        proof["queries"][0]["source_node_type"] = 7
    elif defect == "bool_rc":
        proof["queries"][0]["rc"] = False
    elif defect == "wrong_handle":
        proof["queries"][0]["clone_node_handle"] += 1
    elif defect == "missing_query":
        proof["queries"].clear()
    else:
        proof["queries"].append(copy.deepcopy(proof["queries"][0]))
    with pytest.raises(ValueError, match="exact deferred native type proof"):
        resolve(source, receipt, proof)


@pytest.mark.parametrize(
    "defect", ["missing_copy", "missing_memset", "duplicate", "category", "foreign_graph", "zero_bytes"]
)
def test_capture_proof_cannot_replace_each_replay_memory_activity(tmp_path, defect):
    source, receipt, api, graph, path, _ = graph_fixture(tmp_path)
    registry = resolve(source, receipt, query_proof(source, receipt, api, graph, path))
    events = replay(registry)
    if defect == "missing_copy":
        events.pop(1)
    elif defect == "missing_memset":
        events.pop(2)
    elif defect == "duplicate":
        events.append(copy.deepcopy(events[-1]))
    elif defect == "category":
        events[-1]["cat"] = "kernel"
    elif defect == "foreign_graph":
        events[-1]["args"]["graph id"] += 1
    else:
        events[-1]["args"]["bytes"] = 0
    with pytest.raises(ValueError):
        nodes.bind_replay_kernels(registry, events, correlation=12)


def test_piecewise_producer_retains_original_copy_and_memset_evidence(tmp_path):
    from collector.glm53flash_vllm_piecewise import bind_piecewise_instantiations

    from .test_glm53flash_graph_memset import source_parameters

    source, receipt, api, graph, _, calls = graph_fixture(tmp_path)
    source_parameters(source, receipt)
    # The native query remains a CPU double; this exercises the real producer
    # seam, serialization and independently rederived mapping.
    queried = []

    def query(handle, out):
        queried.append(handle)
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int)).contents.value = 2
        return 0

    api.runtime.cudaGraphNodeGetType = query
    original_source = copy.deepcopy(source)
    segment = {key: value for key, value in source.items() if key != "native_api_libraries"}
    source = {
        "native_api_libraries": source["native_api_libraries"],
        "segments": [{"kind": "graph", "position": 0, **segment}],
    }
    capture = object()

    def validate(value):
        assert value is capture

    registry = SimpleNamespace(validate_replay=validate, graphs=[{"position": 0, "graph": graph}])
    subscriber = SimpleNamespace(
        subscription_closed=True,
        rows=receipt["callbacks"],
        errors=[],
        receipt=lambda actual: receipt if actual is graph else None,
    )
    item = {
        "registry": registry,
        "source": source,
        "entry": SimpleNamespace(capture=capture),
        "source_receipt": {"file": "TEST_ONLY-source.json", "sha256": "a" * 64},
    }
    bind_piecewise_instantiations([item], subscriber, api, tmp_path, "TEST_ONLY")
    bound = registry.bound_capture["segments"][0]
    proof = json.loads((tmp_path / bound["node_type_receipt"]["file"]).read_text())
    derived = callbacks.resolve_registry(
        original_source, receipt, proof, allow_pending_memcpy=True, allow_pending_memset=True
    )
    assert bound["nodes"] == derived["nodes"]
    assert [n["node_type"] for n in bound["nodes"]] == [0, 1, 2]
    assert len(queried) == 1 and not calls
    assert (
        json.loads((tmp_path / bound["shared_callback_receipt"]["file"]).read_text())["callbacks"]
        == receipt["callbacks"]
    )


@pytest.mark.parametrize("defect", [None, "missing_proof", "changed_proof"])
def test_full_original_evidence_reader_rederives_explicit_memory_mapping(tmp_path, monkeypatch, defect):
    import hashlib

    from collector import glm53flash_vllm_graph_export as export

    from .test_glm53flash_vllm_graph_export import fixture

    fixture(tmp_path, monkeypatch)
    source_path = tmp_path / "vllm-graph-clones-rank-0-capture-0-0-source.json"
    callback_path = tmp_path / "vllm-graph-clones-rank-0-capture-0-0.json"
    source = json.loads(source_path.read_text())
    receipt = json.loads(callback_path.read_text())
    observed_source = observed()["graph"]["source"]
    source["native_api_libraries"] = observed_source["native_api_libraries"]
    receipt["callback_subscription_closed"] = True
    original_clone = receipt["callbacks"][1]
    for offset, native in enumerate(observed_source["nodes"][1:], 1):
        node = copy.deepcopy(native)
        node.update(node_id=source["nodes"][0]["node_id"] + offset, name="attention_0")
        clone = copy.deepcopy(original_clone)
        clone.update(original_node_id=node["node_id"], node_id=original_clone["node_id"] + offset)
        clone["raw_fields"]["node"] += offset * 10
        clone["raw_fields"]["originalNode"] += offset * 10
        if node["node_type"] == 1:
            node["memcpy_params"]["source_node_handle"] = clone["raw_fields"]["originalNode"]
        receipt["callbacks"].append(clone)
        source["nodes"].append(node)
        source["calls"][0]["owned_node_ids"].append(node["node_id"])
    proof_path = tmp_path / "TEST_ONLY-reader-types.json"

    def query(handle, out):
        assert handle == receipt["callbacks"][-1]["raw_fields"]["node"]
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int)).contents.value = 2
        return 0

    api = SimpleNamespace(libraries=source["native_api_libraries"], runtime=SimpleNamespace(cudaGraphNodeGetType=query))
    graph = SimpleNamespace(raw_cuda_graph_exec=lambda: receipt["callbacks"][0]["raw_fields"]["graphExec"])
    proof = query_proof(source, receipt, api, graph, proof_path)
    bound = resolve(source, receipt, proof)
    source_path.write_text(json.dumps(source))
    callback_path.write_text(json.dumps(receipt))

    def reference(path):
        return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    bound["instantiation_receipt"] = reference(callback_path)
    bound["node_type_receipt"] = reference(proof_path)
    if defect == "missing_proof":
        bound.pop("node_type_receipt")
    elif defect == "changed_proof":
        proof["queries"][0]["rc"] = 1
        proof_path.write_text(json.dumps(proof))
        bound["node_type_receipt"] = reference(proof_path)
    (tmp_path / "vllm-capture-rank-0-0.json").write_text(json.dumps(bound))
    args = (
        tmp_path,
        0,
        json.loads((tmp_path / "vllm-graph-policy-rank-0.json").read_text()),
        json.loads((tmp_path / "manifest.json").read_text()),
        json.loads((tmp_path / "provenance.json").read_text()),
        set(),
    )
    if defect:
        with pytest.raises(ValueError, match="exact deferred native type proof"):
            export._captures(*args)
    else:
        captures = export._captures(*args)
        selected = next(value[0] for value in captures.values() if value[0]["graph_id"] == bound["graph_id"])
        assert selected == bound
        assert [n["node_type"] for n in selected["nodes"]] == [0, 1, 2]
