# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU integrity tests; synthetic graph nodes never qualify GPU observations."""

import pytest

from collector.glm53flash_graph_nodes import CaptureNodeRegistry, bind_replay_kernels

pytestmark = pytest.mark.unit


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
