# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY CUPTI fixtures, never measured performance or native qualification."""

import copy

import pytest
from collector.glm53flash_graph_nodes import VLLM_EXECUTION_RANGE, VLLM_LOGITS_RANGE
from collector.glm53flash_vllm_piecewise import BREAKABLE_SOURCE_PIN, EAGER_RANGE_PREFIX
from collector.glm53flash_vllm_piecewise_activity import bind_piecewise_execution

pytestmark = pytest.mark.unit


def fixture():
    def region(name, ts, dur):
        return {"ph": "X", "name": name, "ts": ts, "dur": dur, "pid": 1, "tid": 2}

    def call(name, correlation, ts):
        return {"cat": "cuda_runtime", "args": {"correlation": correlation}, **region(name, ts, 1)}

    def kernel(name, correlation, ts, dur, graph=0, node=0):
        return {
            "cat": "kernel",
            "name": name,
            "ts": ts,
            "dur": dur,
            "args": {
                "correlation": correlation,
                "graph id": graph,
                "graph node id": node,
                "stream": 1,
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "shared memory": 0,
            },
        }

    def graph(position, graph_id, node):
        return {
            "position": position,
            "kind": "graph",
            "graph_id": graph_id,
            "capture_graph_id": graph_id + 100,
            "capture_id": graph_id + 200,
            "nodes": [{"node_id": node, "node_type": 0, "name": "attention_3", "call_index": 0}],
        }

    capture = {
        "capture_scope": "vllm_piecewise_hidden_states",
        "uncaptured_operations": ["logits"],
        "graph_mutations": False,
        "calls": [{"index": 0, "completed": True, "name": "attention_3"}],
        "segments": [
            graph(0, 10, 101),
            {
                "position": 1,
                "kind": "eager",
                "name": "attention_3",
                "call_index": 0,
                "eager_id": 0,
                "range": EAGER_RANGE_PREFIX + "0",
                "source_sha256": BREAKABLE_SOURCE_PIN,
            },
            graph(2, 20, 201),
        ],
    }
    events = [
        region(VLLM_EXECUTION_RANGE, 0, 100),
        region(VLLM_LOGITS_RANGE, 60, 10),
        region(EAGER_RANGE_PREFIX + "0", 20, 10),
        call("cudaLaunchKernel", 1, 1),
        call("cudaGraphLaunch", 2, 10),
        call("cudaLaunchKernel", 3, 22),
        call("cudaGraphLaunch", 4, 40),
        call("cudaLaunchKernel", 5, 62),
        kernel("setup", 1, 3, 2),
        kernel("graph_attention_a", 2, 15, 20, 10, 101),
        kernel("eager_attention", 3, 25, 20),
        kernel("graph_attention_b", 4, 45, 10, 20, 201),
        kernel("logits", 5, 65, 10),
    ]
    return capture, events


def test_one_physical_operation_spans_graphs_and_eager_without_repeated_charge():
    capture, events = fixture()
    originals = copy.deepcopy((capture, events))
    result = bind_piecewise_execution(capture, events)
    units = {row["operation"]: row for row in result["operation_activity_unions"]}
    assert units["attention_3"]["active_union_us"] == 40
    assert units["attention_3"]["activity_interval_sum_us"] == 50
    assert units["attention_3"]["captured_node_ids"] == [101, 201]
    assert units["attention_3"]["outside_graph_activity_indices"] == [10]
    assert units["native_graph_setup"]["active_union_us"] == 2
    assert units["logits"]["active_union_us"] == 10
    assert result["approximate_additive_operation_union_us"] == 52
    assert result["execution_range"]["dur"] == 100  # No residual/CPU-gap redistribution.
    assert result["formal_admission"] is False
    assert result["whole_forward_accuracy"] == "NOT_EVALUATED"
    assert (capture, events) == originals


@pytest.mark.parametrize(
    "defect",
    [
        "missing_graph",
        "duplicate_graph",
        "wrong_graph_id",
        "missing_node",
        "extra_node",
        "missing_eager_range",
        "duplicate_eager_range",
        "wrong_eager_source",
        "wrong_eager_thread",
        "reordered_segments",
        "aliased_graph",
        "unfinished_call",
        "wrong_node_owner",
        "wrong_eager_owner",
        "graph_inside_eager",
        "straddling_eager",
        "missing_eager_activity",
        "missing_setup_activity",
        "unknown_cuda_dispatch",
        "missing_logits",
        "graph_outside_execution",
        "mutated_capture",
    ],
)
def test_incomplete_or_ambiguous_piecewise_receipts_cannot_supply_costs(defect):
    capture, events = fixture()
    if defect == "missing_graph":
        events.pop(6)
    elif defect == "duplicate_graph":
        events.append(copy.deepcopy(events[6]))
    elif defect == "wrong_graph_id":
        events[11]["args"]["graph id"] = 99
    elif defect == "missing_node":
        events.pop(11)
    elif defect == "extra_node":
        extra = copy.deepcopy(events[11])
        extra["args"]["graph node id"] = 202
        events.append(extra)
    elif defect == "missing_eager_range":
        events.pop(2)
    elif defect == "duplicate_eager_range":
        events.append(copy.deepcopy(events[2]))
    elif defect == "wrong_eager_source":
        capture["segments"][1]["source_sha256"] = "0" * 64
    elif defect == "wrong_eager_thread":
        events[2]["tid"] = 3
    elif defect == "reordered_segments":
        events[4]["ts"], events[6]["ts"] = 40, 10
    elif defect == "aliased_graph":
        capture["segments"][2]["capture_graph_id"] = 110
    elif defect == "unfinished_call":
        capture["calls"][0]["completed"] = False
    elif defect == "wrong_node_owner":
        capture["segments"][0]["nodes"][0]["name"] = "ffn_3"
    elif defect == "wrong_eager_owner":
        capture["segments"][1]["call_index"] = 1
    elif defect == "graph_inside_eager":
        events[6]["ts"] = 25
    elif defect == "straddling_eager":
        events[5]["ts"] = 19.5
    elif defect == "missing_eager_activity":
        events.pop(10)
    elif defect == "missing_setup_activity":
        events.pop(8)
    elif defect == "unknown_cuda_dispatch":
        events[3]["name"] = "cudaUnknownKernelWork"
    elif defect == "missing_logits":
        events.pop(12)
    elif defect == "graph_outside_execution":
        events[6]["tid"] = 9
    elif defect == "mutated_capture":
        capture["graph_mutations"] = True
    with pytest.raises(ValueError):
        bind_piecewise_execution(capture, events)


def test_structural_only_graph_does_not_silently_supply_zero_latency():
    capture, events = fixture()
    capture["segments"][0]["nodes"][0]["node_type"] = 7
    events.pop(9)
    with pytest.raises(ValueError, match="omits captured"):
        bind_piecewise_execution(capture, events)


@pytest.mark.parametrize("boundary", [20, 30, 60, 70])
def test_zero_duration_launch_on_operation_boundary_is_ambiguous(boundary):
    capture, events = fixture()
    events[5].update(ts=boundary, dur=0)
    with pytest.raises(ValueError, match="ambiguous ownership"):
        bind_piecewise_execution(capture, events)


def test_zero_duration_launch_strictly_inside_eager_keeps_its_physical_owner():
    capture, events = fixture()
    events[5]["dur"] = 0
    result = bind_piecewise_execution(capture, events)
    assert {row["operation"]: row["active_union_us"] for row in result["operation_activity_unions"]} == {
        "attention_3": 40,
        "native_graph_setup": 2,
        "logits": 10,
    }
