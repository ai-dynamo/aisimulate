# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic source/replay contracts; no failed native artifact is upgraded."""

import copy

import pytest
from collector import glm53flash_graph_callbacks as callbacks
from collector import glm53flash_graph_nodes as nodes
from collector.glm53flash_vllm_piecewise_activity import bind_piecewise_execution

from .test_glm53flash_graph_memset import events, pending, record, resolve
from .test_glm53flash_native_classification import eager_events
from .test_glm53flash_vllm_piecewise_activity import fixture

pytestmark = pytest.mark.unit


def reported_registry(tmp_path):
    source, receipt, api, graph, path, _ = pending(tmp_path)
    source["nodes"][-1]["memset_params"]["contract"] = nodes.MEMSET_REPORTED_KIND_CONTRACT
    proof = record(source, receipt, api, graph, path)
    return resolve(source, receipt, proof)


@pytest.mark.parametrize("name,kind", [("Memset (Unknown)", "Unknown"), ("Memset (Device)", "Device")])
def test_reported_kind_is_retained_without_destination_type_inference(tmp_path, name, kind):
    registry = reported_registry(tmp_path)
    trace = events(registry)
    trace[-1]["name"] = name
    original = copy.deepcopy((registry, trace))
    result = nodes.bind_replay_kernels(registry, trace, correlation=12)
    fingerprint = result["activities"][-1]["fingerprint"]
    assert fingerprint == {
        "name": name,
        "bytes": 128,
        "reported_memory_kind": kind,
        "memory_kind_contract": nodes.MEMSET_REPORTED_KIND_CONTRACT,
    }
    assert result["formal_admission"] is False
    assert (registry, trace) == original
    assert callbacks.vllm_memset_contract_options(registry) == {"allow_pending_memset": True}


def test_original_device_only_contract_still_rejects_unknown(tmp_path):
    source, receipt, api, graph, path, _ = pending(tmp_path)
    registry = resolve(source, receipt, record(source, receipt, api, graph, path))
    trace = events(registry)
    trace[-1]["name"] = "Memset (Unknown)"
    with pytest.raises(ValueError, match="exact replay category"):
        nodes.bind_replay_kernels(registry, trace, correlation=12)


@pytest.mark.parametrize(
    "defect",
    [
        "name",
        "managed",
        "bytes",
        "category",
        "graph",
        "node",
        "correlation",
        "duration",
        "provider",
        "source_bytes",
        "registry_contract",
        "source_contract",
        "requirement",
        "duplicate",
        "missing",
    ],
)
def test_reported_kind_never_weakens_source_or_replay_identity(tmp_path, defect):
    registry = reported_registry(tmp_path)
    trace = events(registry)
    trace[-1]["name"] = "Memset (Unknown)"
    event, node = trace[-1], registry["nodes"][-1]
    if defect in ("name", "managed"):
        event["name"] = "Memset (Unknown)Extra" if defect == "name" else "Memset (Managed)"
    elif defect == "bytes":
        event["args"]["bytes"] += 1
    elif defect == "category":
        event["cat"] = "kernel"
    elif defect in ("graph", "node", "correlation"):
        event["args"][{"graph": "graph id", "node": "graph node id", "correlation": "correlation"}[defect]] += 1
    elif defect == "duration":
        event["dur"] = 0
    elif defect == "provider":
        node["memset_params"]["native_api_libraries"]["cupti"]["sha256"] = "0" * 64
    elif defect == "source_bytes":
        node["memset_params"]["parameters"]["width"] += 1
    elif defect == "registry_contract":
        registry["memset_pending_contract"] = nodes.MEMSET_PENDING_CONTRACT
    elif defect == "source_contract":
        node["memset_params"]["contract"] = nodes.MEMSET_PENDING_CONTRACT
    elif defect == "requirement":
        node["memset_activity_requirement"]["reported_names"]["Memset (Unknown)"] = "Device"
    elif defect == "duplicate":
        trace.append(copy.deepcopy(event))
    elif defect == "missing":
        trace.pop()
    with pytest.raises(ValueError):
        nodes.bind_replay_kernels(registry, trace, correlation=12)


def host_call(template):
    return {**copy.deepcopy(template), "name": "cudaHostAlloc", "args": {"correlation": 99999}}


def test_host_allocation_retains_cpu_duration_without_manufactured_gpu_cost():
    trace = eager_events()
    original = copy.deepcopy(trace)
    baseline = nodes.bind_native_eager_activity(trace)
    call = host_call(trace[1])
    trace.append(call)
    result = nodes.bind_native_eager_activity(trace)
    assert result["activities"] == baseline["activities"]
    assert result["activity_union_us"] == baseline["activity_union_us"]
    assert result["host_memory_api_observations"] == [
        {
            "name": "cudaHostAlloc",
            "correlation": 99999,
            "start_us": call["ts"],
            "duration_us": call["dur"],
            "pid": call["pid"],
            "tid": call["tid"],
            "timing_domain": "host",
            "included_in_device_activity_sum": False,
        }
    ]
    assert trace[:-1] == original


@pytest.mark.parametrize("defect", ["unknown_api", "device_work"])
def test_host_allocation_cannot_hide_device_activity_or_unknown_api(defect):
    trace = eager_events()
    call = host_call(trace[1])
    trace.append(call)
    if defect == "unknown_api":
        call["name"] += "Async"
    else:
        activity = copy.deepcopy(trace[-2])
        activity["args"]["correlation"] = 99999
        trace.append(activity)
    with pytest.raises(ValueError, match="unknown native CUDA dispatch|host-memory API unexpectedly owns"):
        nodes.bind_native_eager_activity(trace)


def test_piecewise_composition_keeps_host_observations_separate():
    capture, trace = fixture()
    baseline = bind_piecewise_execution(capture, trace)
    call = host_call(trace[3])
    trace.append(call)
    result = bind_piecewise_execution(capture, trace)
    assert result["activities"] == baseline["activities"]
    assert result["approximate_additive_operation_union_us"] == baseline["approximate_additive_operation_union_us"]
    assert result["host_memory_api_observations"][0]["duration_us"] == call["dur"]


def test_full_graph_composition_preserves_host_interval_without_charging_it():
    from .test_glm53flash_vllm_graph_execution import trace

    binding, events = trace()
    baseline = nodes.bind_vllm_execution_activity(binding, events)
    call = host_call(events[2])
    events.append(call)
    result = nodes.bind_vllm_execution_activity(binding, events)
    assert result["activities"] == baseline["activities"]
    assert result["host_memory_api_observations"][0]["duration_us"] == call["dur"]


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_eager_native_composition_retains_host_memory_observations(backend):
    from collector.glm53flash_sglang_prefill_activity import bind_prefill_activity
    from collector.glm53flash_vllm_none_activity import bind_none_execution

    from .test_glm53flash_sglang_prefill_activity import fixture as sglang_fixture
    from .test_glm53flash_vllm_none_activity import trace_fixture

    factory, bind = (
        (trace_fixture, bind_none_execution) if backend == "vllm" else (sglang_fixture, bind_prefill_activity)
    )
    events, calls, names = factory()
    baseline = bind(events, calls, names)
    call = host_call(next(row for row in events if row.get("cat") == "cuda_runtime"))
    events.append(call)
    result = bind(events, calls, names)
    assert result["activities"] == baseline["activities"]
    assert result["host_memory_api_observations"][0]["duration_us"] == call["dur"]
