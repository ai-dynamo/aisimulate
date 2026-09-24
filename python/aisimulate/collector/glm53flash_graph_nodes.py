# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only CUDA capture-node ownership, joined to actual CUPTI replay data.

Original bindings to documented NVIDIA CUDA/CUPTI APIs; no CUDA source copied.
See README.glm53flash.md for API references and qualification limits. No event,
kernel, dependency, stream wait or other graph node is inserted by this module.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math
from pathlib import Path


class GraphEdgeData(ctypes.Structure):
    """CUDA13 documented eight-byte edge ABI; preserve nondefault PDL data."""

    _fields_ = (
        ("from_port", ctypes.c_ubyte),
        ("to_port", ctypes.c_ubyte),
        ("type", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte * 5),
    )


def _library(stem):
    mapped = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split()) >= 6 and Path(line.split()[-1]).name.startswith(f"lib{stem}.so")
    }
    if len(mapped) > 1:
        raise RuntimeError(f"multiple loaded {stem} libraries cannot form one capture identity")
    name = next(iter(mapped), None) or ctypes.util.find_library(stem)
    if name is None:
        for directory in ("/usr/local/cuda/extras/CUPTI/lib64", "/usr/local/cuda/lib64"):
            found = sorted(Path(directory).glob(f"lib{stem}.so*"))
            if found:
                name = str(found[0])
                break
    if name is None:
        raise RuntimeError(f"native {stem} library is unavailable")
    library = ctypes.CDLL(name)
    mapped = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split()) >= 6 and Path(line.split()[-1]).name.startswith(f"lib{stem}.so")
    }
    if len(mapped) != 1:
        raise RuntimeError(f"loaded {stem} library identity is ambiguous")
    path = Path(mapped.pop()).resolve()
    with path.open("rb") as stream:
        receipt = {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    return library, receipt


class NativeGraphAPI:
    """Minimal read-only API surface, resolved without initializing CUDA."""

    def __init__(self):
        self.runtime, runtime = _library("cudart")
        self.cupti, cupti = _library("cupti")
        self.libraries = {"cudart": runtime, "cupti": cupti}
        pointer = ctypes.c_void_p
        pointer_out = ctypes.POINTER(pointer)
        size_out = ctypes.POINTER(ctypes.c_size_t)
        self._bind(self.runtime, "cudaRuntimeGetVersion", [ctypes.POINTER(ctypes.c_int)])
        version = ctypes.c_int()
        self._call(self.runtime.cudaRuntimeGetVersion, ctypes.byref(version))
        if version.value // 1000 != 13:
            raise RuntimeError("native graph query ABI currently requires verified CUDA13 runtime")
        self.libraries["cudart"].update(runtime_version=version.value, abi="CUDA13_capture7_edges5")
        self._bind(
            self.runtime,
            "cudaStreamGetCaptureInfo",
            [
                pointer,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_ulonglong),
                pointer_out,
                ctypes.POINTER(pointer_out),
                ctypes.POINTER(ctypes.POINTER(GraphEdgeData)),
                size_out,
            ],
        )
        self._bind(self.runtime, "cudaGraphGetNodes", [pointer, pointer_out, size_out])
        self._bind(
            self.runtime,
            "cudaGraphGetEdges",
            [pointer, pointer_out, pointer_out, ctypes.POINTER(GraphEdgeData), size_out],
        )
        self._bind(self.runtime, "cudaGraphNodeGetType", [pointer, ctypes.POINTER(ctypes.c_int)])
        self._bind(self.cupti, "cuptiGetGraphId", [pointer, ctypes.POINTER(ctypes.c_uint32)])
        self._bind(self.cupti, "cuptiGetGraphNodeId", [pointer, ctypes.POINTER(ctypes.c_uint64)])

    @staticmethod
    def _bind(library, name, arguments):
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int

    @staticmethod
    def _call(function, *args):
        status = function(*args)
        if status != 0:
            raise RuntimeError(f"read-only native graph query {function.__name__} failed with status {status}")

    def snapshot(self, stream: int) -> dict:
        status, capture_id, graph = ctypes.c_int(), ctypes.c_ulonglong(), ctypes.c_void_p()
        self._call(
            self.runtime.cudaStreamGetCaptureInfo,
            stream,
            ctypes.byref(status),
            ctypes.byref(capture_id),
            ctypes.byref(graph),
            None,
            None,
            None,
        )
        if status.value != 1 or not graph.value:
            raise RuntimeError("node ownership requires an actual active native capture")
        graph_id = ctypes.c_uint32()
        self._call(self.cupti.cuptiGetGraphId, graph, ctypes.byref(graph_id))
        count = ctypes.c_size_t()
        self._call(self.runtime.cudaGraphGetNodes, graph, None, ctypes.byref(count))
        handles = (ctypes.c_void_p * count.value)()
        self._call(self.runtime.cudaGraphGetNodes, graph, handles, ctypes.byref(count))
        nodes, by_handle = {}, {}
        for handle in handles[: count.value]:
            node_id, node_type = ctypes.c_uint64(), ctypes.c_int()
            self._call(self.cupti.cuptiGetGraphNodeId, handle, ctypes.byref(node_id))
            self._call(self.runtime.cudaGraphNodeGetType, handle, ctypes.byref(node_type))
            if node_id.value in nodes:
                raise RuntimeError("native graph query returned duplicate node identity")
            if node_type.value == 4:
                raise RuntimeError("child graphs require recursive native ownership before admission")
            nodes[node_id.value] = {"node_type": node_type.value}
            by_handle[handle] = node_id.value
        count = ctypes.c_size_t()
        self._call(self.runtime.cudaGraphGetEdges, graph, None, None, None, ctypes.byref(count))
        sources, targets = (ctypes.c_void_p * count.value)(), (ctypes.c_void_p * count.value)()
        metadata = (GraphEdgeData * count.value)()
        self._call(self.runtime.cudaGraphGetEdges, graph, sources, targets, metadata, ctypes.byref(count))
        edges = [
            {
                "from": by_handle[source],
                "to": by_handle[target],
                "from_port": data.from_port,
                "to_port": data.to_port,
                "type": data.type,
                "reserved": list(data.reserved),
            }
            for source, target, data in zip(
                sources[: count.value], targets[: count.value], metadata[: count.value], strict=True
            )
        ]
        return {"capture_id": capture_id.value, "graph_id": graph_id.value, "nodes": nodes, "edges": edges}


class CaptureNodeRegistry:
    """Assign each captured node to exactly one actual native call boundary."""

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.identity = None
        self.seen = set()
        self.owners = {}
        self.stack = []
        self.calls = []
        self.last = None

    def _flush(self):
        value = self.snapshot()
        identity = (value["capture_id"], value["graph_id"])
        if self.identity is None:
            self.identity = identity
        if identity != self.identity or not self.seen.issubset(value["nodes"]):
            raise RuntimeError("native capture identity changed or removed existing graph nodes")
        owner = self.stack[-1]["name"] if self.stack else "native_graph_setup"
        for node in value["nodes"].keys() - self.seen:
            self.owners[node] = {"name": owner, **value["nodes"][node]}
        self.seen = set(value["nodes"])
        self.last = value

    def enter(self, name: str, source: str):
        self._flush()
        token = {"name": name, "source": source, "index": len(self.calls)}
        self.calls.append(token)
        self.stack.append(token)
        return token

    def leave(self, token):
        if not self.stack or self.stack[-1] is not token:
            raise RuntimeError("native operation capture boundaries are not properly nested")
        self._flush()
        self.stack.pop()

    def finish(self):
        if self.stack:
            raise RuntimeError("native operation capture boundary remains open")
        self._flush()
        return {
            "capture_id": self.identity[0],
            "graph_id": self.identity[1],
            "nodes": [{"node_id": node, **value} for node, value in sorted(self.owners.items())],
            "edges": self.last["edges"],
            "calls": self.calls,
            "graph_mutations": False,
            "measurement_method": "native_capture_node_ownership",
        }


def bind_replay_kernels(registry: dict, events: list[dict], *, correlation: int) -> dict:
    """Join one real graph launch by CUPTI IDs, retaining overlap explicitly.

    This returns measured node intervals and per-operation active-time unions.
    It does not turn sums into a whole-forward prediction or certify accuracy.
    """
    expected = {row["node_id"]: row for row in registry["nodes"] if row["node_type"] == 0}
    actual, streams = {}, set()
    for event in events:
        args = event.get("args", {})
        if event.get("cat") != "kernel" or args.get("correlation") != correlation:
            continue
        node = args.get("graph node id")
        if args.get("graph id") != registry["graph_id"] or node not in expected or node in actual:
            raise ValueError("CUPTI replay kernel does not match a unique captured native node")
        if (
            type(args.get("stream")) is not int
            or any(not isinstance(args.get(key), list) or len(args[key]) != 3 for key in ("grid", "block"))
            or type(args.get("shared memory")) is not int
        ):
            raise ValueError("CUPTI replay kernel lacks actual stream/launch geometry")
        start, duration = event.get("ts"), event.get("dur")
        if (
            any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, duration))
            or duration <= 0
        ):
            raise ValueError("CUPTI replay lacks a positive native kernel interval")
        streams.add(args.get("stream"))
        actual[node] = {
            "node_id": node,
            "operation": expected[node]["name"],
            "start_us": start,
            "end_us": start + duration,
            "stream": args.get("stream"),
            "fingerprint": {
                "name": event["name"],
                **{key: args.get(key) for key in ("grid", "block", "shared memory")},
            },
        }
    if not expected or actual.keys() != expected.keys():
        raise ValueError("CUPTI launch omits captured native kernel nodes")
    rows = sorted(actual.values(), key=lambda row: (row["start_us"], row["node_id"]))
    overlaps = [
        [left["node_id"], right["node_id"]]
        for index, left in enumerate(rows)
        for right in rows[index + 1 :]
        if right["start_us"] < left["end_us"]
    ]
    return {
        "graph_id": registry["graph_id"],
        "correlation": correlation,
        "kernels": rows,
        "overlapping_node_pairs": overlaps,
        "streams": sorted(streams),
        "kernel_interval_sum_us": sum(row["end_us"] - row["start_us"] for row in rows),
        "kernel_envelope_us": max(row["end_us"] for row in rows) - min(row["start_us"] for row in rows),
        "whole_forward_accuracy": "NOT_EVALUATED",
        "formal_admission": False,
    }
