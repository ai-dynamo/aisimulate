# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU integrity tests; synthetic graph nodes never qualify GPU observations."""

import ctypes
from types import SimpleNamespace

import pytest
from collector import glm53flash_graph_nodes as graph_nodes
from collector.glm53flash_graph_nodes import CaptureNodeRegistry, bind_replay_kernels

pytestmark = pytest.mark.unit


def native_abi_fixture(monkeypatch, version=13000, node_count=2, changed_count=None):
    """Exercise real ctypes pointer writes; these are never GPU evidence."""
    pointer = ctypes.c_void_p
    pointer_out = ctypes.POINTER(pointer)
    size_out = ctypes.POINTER(ctypes.c_size_t)
    edge_type = graph_nodes.GraphEdgeData
    runtime, cupti = SimpleNamespace(), SimpleNamespace()
    calls = []

    def attach(library, name, arguments, implementation):
        function = ctypes.CFUNCTYPE(ctypes.c_int, *arguments)(implementation)
        function.__name__ = name
        setattr(library, name, function)

    def write_version(result):
        result[0] = version
        return 0

    def capture(stream, status, capture_id, graph, dependencies, edges, count):
        assert stream == 77 and not dependencies and not edges and not count
        status[0], capture_id[0], graph[0] = 1, 3, 100
        return 0

    def nodes(graph, handles, count):
        assert graph == 100
        calls.append("nodes-fill" if handles else "nodes-count")
        if handles:
            if node_count == 0:
                return 1
            for index in range(node_count):
                handles[index] = 101 + index
        count[0] = node_count + (
            1
            if (handles and changed_count == "nodes")
            or (not handles and changed_count == "nodes_added" and calls.count("nodes-count") > 1)
            else 0
        )
        return 0

    def edges(graph, sources, targets, data, count):
        assert graph == 100
        calls.append("edges-fill" if sources else "edges-count")
        edge_count = max(0, node_count - 1)
        if sources:
            if edge_count == 0:
                return 1
            assert data
            sources[0], targets[0] = 101, 102
            data[0].from_port, data[0].to_port, data[0].type = 2, 0, 1
            data[0].reserved[:] = [3, 4, 5, 6, 7]
        count[0] = edge_count + (
            1
            if (sources and changed_count == "edges")
            or (not sources and changed_count == "edges_added" and calls.count("edges-count") > 1)
            else 0
        )
        return 0

    def graph_id(graph, result):
        result[0] = 4
        return 0

    def node_id(node, result):
        result[0] = node + 1000
        return 0

    def node_type(node, result):
        result[0] = 0
        return 0

    attach(runtime, "cudaRuntimeGetVersion", [ctypes.POINTER(ctypes.c_int)], write_version)
    attach(
        runtime,
        "cudaStreamGetCaptureInfo",
        [
            pointer,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulonglong),
            pointer_out,
            ctypes.POINTER(pointer_out),
            ctypes.POINTER(ctypes.POINTER(edge_type)),
            size_out,
        ],
        capture,
    )
    attach(runtime, "cudaGraphGetNodes", [pointer, pointer_out, size_out], nodes)
    attach(
        runtime, "cudaGraphGetEdges", [pointer, pointer_out, pointer_out, ctypes.POINTER(edge_type), size_out], edges
    )
    attach(runtime, "cudaGraphNodeGetType", [pointer, ctypes.POINTER(ctypes.c_int)], node_type)
    attach(cupti, "cuptiGetGraphId", [pointer, ctypes.POINTER(ctypes.c_uint32)], graph_id)
    attach(cupti, "cuptiGetGraphNodeId", [pointer, ctypes.POINTER(ctypes.c_uint64)], node_id)
    monkeypatch.setattr(graph_nodes, "_library", lambda name: ({"cudart": runtime, "cupti": cupti}[name], {}))
    return calls


def test_cuda13_pointer_abi_retains_nondefault_dependency_metadata(monkeypatch):
    native_abi_fixture(monkeypatch)
    api = graph_nodes.NativeGraphAPI()
    result = api.snapshot(77)
    assert result["graph_id"] == 4 and result["capture_id"] == 3
    assert result["nodes"] == {1101: {"node_type": 0}, 1102: {"node_type": 0}}
    assert result["edges"] == [
        {"from": 1101, "to": 1102, "from_port": 2, "to_port": 0, "type": 1, "reserved": [3, 4, 5, 6, 7]}
    ]
    assert ctypes.sizeof(graph_nodes.GraphEdgeData) == 8
    assert [
        getattr(graph_nodes.GraphEdgeData, name).offset for name in ("from_port", "to_port", "type", "reserved")
    ] == [
        0,
        1,
        2,
        3,
    ]
    assert api.libraries["cudart"] == {"runtime_version": 13000, "abi": "CUDA13_capture7_edges5"}


@pytest.mark.parametrize("version", [12080, 14000])
def test_unverified_cuda_major_cannot_use_cuda13_capture_abi(monkeypatch, version):
    native_abi_fixture(monkeypatch, version)
    with pytest.raises(RuntimeError, match="verified CUDA13"):
        graph_nodes.NativeGraphAPI()


@pytest.mark.parametrize("node_count", [0, 1])
def test_empty_capture_enumerates_native_counts_without_nonnull_zero_capacity_arrays(monkeypatch, node_count):
    calls = native_abi_fixture(monkeypatch, node_count=node_count)
    api = graph_nodes.NativeGraphAPI()
    snapshot = api.snapshot(77)
    assert len(snapshot["nodes"]) == node_count and snapshot["edges"] == []
    expected = (
        ["nodes-count", "nodes-fill", "nodes-count", "edges-count"] if node_count else ["nodes-count", "edges-count"]
    )
    assert calls == expected
    api.snapshot(77)
    assert calls == expected + expected  # Each boundary obtains new actual counts.


@pytest.mark.parametrize("changed_count", ["nodes", "edges", "nodes_added", "edges_added"])
def test_changed_native_count_between_queries_fails_instead_of_truncating(monkeypatch, changed_count):
    native_abi_fixture(monkeypatch, changed_count=changed_count)
    with pytest.raises(RuntimeError, match="count changed"):
        graph_nodes.NativeGraphAPI().snapshot(77)


def test_nested_collective_nodes_have_one_owner_without_interval_subtraction():
    nodes = {1: {"node_type": 2}}
    registry = CaptureNodeRegistry(lambda: {"capture_id": 3, "graph_id": 4, "nodes": dict(nodes), "edges": []})
    compute = registry.enter("attention_0", "native.attention")
    nodes[2] = {"node_type": 0}
    collective = registry.enter("attention_allreduce_0", "native.all_reduce")
    nodes[3] = {"node_type": 0}
    registry.leave(collective)
    nodes[4] = {"node_type": 0}
    registry.leave(compute)
    result = registry.finish()
    assert {row["node_id"]: row["name"] for row in result["nodes"]} == {
        1: "native_graph_setup",
        2: "attention_0",
        3: "attention_allreduce_0",
        4: "attention_0",
    }
    assert result["graph_mutations"] is False


@pytest.mark.parametrize("defect", ["changed_capture", "removed_node", "open_scope", "wrong_nesting"])
def test_capture_identity_and_boundary_fail_closed(defect):
    state = {"capture_id": 1, "graph_id": 2, "nodes": {3: {"node_type": 0}}, "edges": []}
    registry = CaptureNodeRegistry(lambda: state)
    token = registry.enter("attention", "native")
    with pytest.raises(RuntimeError):
        if defect == "changed_capture":
            state["capture_id"] = 2
            registry.leave(token)
        elif defect == "removed_node":
            state["nodes"] = {}
            registry.leave(token)
        elif defect == "open_scope":
            registry.finish()
        else:
            registry.leave(dict(token))


def replay_fixture():
    registry = {
        "graph_id": 4,
        "nodes": [{"node_id": n, "node_type": 0, "name": name} for n, name in ((11, "attention"), (12, "allreduce"))],
    }
    events = [
        {
            "cat": "kernel",
            "name": name,
            "ts": start,
            "dur": 4,
            "args": {
                "graph id": 4,
                "graph node id": node,
                "correlation": 7,
                "stream": stream,
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "shared memory": 0,
            },
        }
        for node, name, start, stream in ((11, "compute", 0, 1), (12, "nccl", 2, 2))
    ]
    return registry, events


def test_replay_joins_cupti_node_ids_and_retains_cross_stream_overlap():
    registry, events = replay_fixture()
    result = bind_replay_kernels(registry, events, correlation=7)
    assert result["overlapping_node_pairs"] == [[11, 12]]
    assert result["kernel_interval_sum_us"] == 8
    assert result["kernel_envelope_us"] == 6
    assert result["formal_admission"] is False
    assert [row["operation"] for row in result["kernels"]] == ["attention", "allreduce"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "graph", "launch", "duration"])
def test_replay_requires_complete_exact_native_node_and_launch_identity(defect):
    registry, events = replay_fixture()
    if defect == "missing":
        events.pop()
    elif defect == "duplicate":
        events.append(events[0])
    elif defect == "unknown":
        events[0]["args"]["graph node id"] = 55
    elif defect == "graph":
        events[0]["args"]["graph id"] = 5
    elif defect == "launch":
        events[0]["args"]["correlation"] = 8
    else:
        events[0]["dur"] = float("nan")
    with pytest.raises(ValueError):
        bind_replay_kernels(registry, events, correlation=7)


@pytest.mark.parametrize("category,node_type", [("gpu_memcpy", 1), ("gpu_memset", 2)])
def test_actual_graph_memory_activity_is_required_and_retained(category, node_type):
    registry, events = replay_fixture()
    registry["nodes"] += [
        {"node_id": 13, "node_type": node_type, "name": "attention"},
        {"node_id": 14, "node_type": 6, "name": "attention"},
    ]
    with pytest.raises(ValueError, match="omits"):
        bind_replay_kernels(registry, events, correlation=7)
    events.append(
        {
            "cat": category,
            "name": "native graph memory operation",
            "ts": 1,
            "dur": 2,
            "args": {"graph id": 4, "graph node id": 13, "correlation": 7, "stream": 3, "bytes": 3088},
        }
    )
    result = bind_replay_kernels(registry, events, correlation=7)
    assert len(result["kernels"]) == 2 and len(result["activities"]) == 3
    assert result["kernel_interval_sum_us"] == 8 and result["activity_interval_sum_us"] == 10
    assert result["unmeasured_structural_nodes"] == [{"node_id": 14, "node_type": 6, "name": "attention"}]
    assert result["formal_admission"] is False
    events[-1]["args"]["bytes"] = True
    with pytest.raises(ValueError, match="byte count"):
        bind_replay_kernels(registry, events, correlation=7)
    events[-1]["args"]["bytes"] = 3088
    events[-1]["cat"] = "gpu_memset" if category == "gpu_memcpy" else "gpu_memcpy"
    with pytest.raises(ValueError, match="unique captured"):
        bind_replay_kernels(registry, events, correlation=7)
