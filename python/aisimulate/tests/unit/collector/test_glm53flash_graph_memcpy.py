# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY source-copy/replay joins; synthetic activities never qualify native data."""

import copy
import ctypes
import sys
from types import SimpleNamespace

import pytest
from collector import glm53flash_graph_nodes as nodes
from collector.glm53flash_graph_callbacks import (
    EVENT_RECORD_CUDART_SHA256,
    QUALIFIED_CUPTI_SHA256,
    resolve_registry,
)

from .test_glm53flash_graph_callbacks import CloneIntegrity
from .test_glm53flash_graph_nodes import native_abi_fixture

pytestmark = pytest.mark.unit


def pending_copy():
    fixture = CloneIntegrity()
    fixture.setUp()
    source, receipt = fixture.registry, fixture.receipt
    source["nodes"][1].update(
        node_type=1,
        memcpy_params={
            "api": "cudaGraphMemcpyNodeGetParams",
            "rc": 0,
            "source_node_handle": 33,
            "trace_producer": {
                "torch_git_version": nodes.MEMCPY_TORCH_REVISION,
                "kineto_gitlink": nodes.MEMCPY_KINETO_REVISION,
            },
            "parameters": {
                "srcArray": 0,
                "dstArray": 0,
                "srcPos": {"x": 0, "y": 0, "z": 0},
                "dstPos": {"x": 0, "y": 0, "z": 0},
                "srcPtr": {"ptr": 1000, "pitch": 256, "xsize": 256, "ysize": 1},
                "dstPtr": {"ptr": 2000, "pitch": 256, "xsize": 256, "ysize": 1},
                "extent": {"width": 256, "height": 1, "depth": 1},
                "kind": 3,
            },
        },
    )
    source["native_api_libraries"] = {
        "cudart": {"sha256": EVENT_RECORD_CUDART_SHA256},
        "cupti": {"sha256": QUALIFIED_CUPTI_SHA256},
    }
    receipt["callbacks"][1]["node_type"] = 0
    receipt["callbacks"][1]["raw_fields"]["nodeType"] = 0
    receipt["callback_subscription_closed"] = True
    return source, receipt


def events():
    return [
        {
            "cat": "kernel",
            "name": "TEST_ONLY_kernel",
            "ts": 1,
            "dur": 2,
            "args": {
                "correlation": 12,
                "graph id": 19,
                "graph node id": 8,
                "stream": 2,
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "shared memory": 0,
            },
        },
        {
            "cat": "gpu_memcpy",
            "name": nodes.MEMCPY_TRACE_NAME,
            "ts": 2,
            "dur": 1,
            "args": {"correlation": 12, "graph id": 19, "graph node id": 808, "stream": 2, "bytes": 256},
        },
    ]


def test_source_copy_mapping_is_pending_and_default_remains_strict():
    source, callback = pending_copy()
    original = copy.deepcopy((source, callback))
    with pytest.raises(ValueError, match="unqualified changed node type"):
        resolve_registry(source, callback)
    result = resolve_registry(source, callback, allow_pending_memcpy=True)
    assert (source, callback) == original
    assert result["nodes"][1]["node_type"] == 1
    assert result["native_instantiation"]["node_clones"][0]["node_type"] == 0
    assert result["nodes"][1]["memcpy_activity_requirement"] == {
        "category": "gpu_memcpy",
        "name": nodes.MEMCPY_TRACE_NAME,
        "bytes": 256,
    }
    bound = nodes.bind_replay_kernels(result, events(), correlation=12)
    assert bound["activities"][1]["fingerprint"]["copy_direction"] == "D2D"
    assert bound["activities"][1]["operation"] == "b"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s, c: s["nodes"][1]["memcpy_params"].update(rc=1),
        lambda s, c: s["nodes"][1]["memcpy_params"].update(source_node_handle=34),
        lambda s, c: s["nodes"][1]["memcpy_params"]["parameters"].update(kind=4),
        lambda s, c: s["nodes"][1]["memcpy_params"]["parameters"]["extent"].update(height=2),
        lambda s, c: s["nodes"][1]["memcpy_params"]["parameters"].update(srcArray=5),
        lambda s, c: s["nodes"][1]["memcpy_params"]["parameters"]["srcPos"].update(y=1),
        lambda s, c: s["nodes"][1]["memcpy_params"]["trace_producer"].update(torch_git_version="other"),
        lambda s, c: s["native_api_libraries"]["cudart"].update(sha256="other"),
        lambda s, c: c.update(callback_subscription_closed=False),
        lambda s, c: s["nodes"][1].update(node_type=5),
    ],
)
def test_pending_copy_cannot_weaken_other_guards(mutation):
    source, callback = pending_copy()
    mutation(source, callback)
    with pytest.raises(ValueError):
        resolve_registry(source, callback, allow_pending_memcpy=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda e: e.pop(),
        lambda e: e.append(copy.deepcopy(e[-1])),
        lambda e: e[1].update(cat="kernel"),
        lambda e: e[1].update(name="Memcpy HtoD (Pinned -> Device)"),
        lambda e: e[1]["args"].update(bytes=128),
        lambda e: e[1]["args"].update(correlation=99),
        lambda e: e[1]["args"].update(**{"graph id": 99}),
        lambda e: e[1]["args"].update(**{"graph node id": 99}),
        lambda e: e[1].update(dur=0),
    ],
)
def test_each_replay_requires_original_copy_activity(mutation):
    source, callback = pending_copy()
    registry = resolve_registry(source, callback, allow_pending_memcpy=True)
    replay = events()
    mutation(replay)
    with pytest.raises(ValueError):
        nodes.bind_replay_kernels(registry, replay, correlation=12)


def test_native_copy_parameter_pointer_abi_and_actual_source_handle(monkeypatch):
    native_abi_fixture(monkeypatch, node_count=1)
    api = nodes.NativeGraphAPI()
    api.runtime.cudaGraphNodeGetType = (
        lambda _handle, out: setattr(ctypes.cast(out, ctypes.POINTER(ctypes.c_int)).contents, "value", 1) or 0
    )
    calls = []

    def query(handle, out):
        calls.append(handle)
        p = ctypes.cast(out, ctypes.POINTER(nodes.GraphCopyParams)).contents
        p.srcPtr.ptr, p.dstPtr.ptr = 1000, 2000
        p.extent.width, p.extent.height, p.extent.depth, p.kind = 256, 1, 1, 3
        return 0

    api.runtime.cudaGraphMemcpyNodeGetParams = query
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(git_version=nodes.MEMCPY_TORCH_REVISION))
    )
    result = api.snapshot(77)
    proof = result["nodes"][1101]["memcpy_params"]
    assert calls == [101] and proof["source_node_handle"] == 101 and proof["rc"] == 0
    assert nodes.memcpy_activity_requirement(result["nodes"][1101])["bytes"] == 256
    assert ctypes.sizeof(nodes.GraphCopyParams) == 160
    assert [getattr(nodes.GraphCopyParams, name).offset for name, _ in nodes.GraphCopyParams._fields_] == [
        0,
        8,
        32,
        64,
        72,
        96,
        128,
        152,
    ]
