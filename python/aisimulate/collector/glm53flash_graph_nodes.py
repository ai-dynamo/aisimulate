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
    binding = None
    if len(mapped) > 1:
        if stem != "cudart":
            raise RuntimeError(f"multiple loaded {stem} libraries cannot form one capture identity")
        from .glm53flash_graph_libraries import resolve_torch_cudart

        name, binding = resolve_torch_cudart(mapped)
    else:
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
    if binding is None and len(mapped) != 1:
        raise RuntimeError(f"loaded {stem} library identity is ambiguous")
    path = Path(name).resolve() if binding is not None else Path(mapped.pop()).resolve()
    if binding is not None and mapped != {item["path"] for item in binding["candidates"]}:
        raise RuntimeError("CUDA runtime mappings changed after actual provider selection")
    with path.open("rb") as stream:
        receipt = {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    if binding is not None:
        receipt["provider_binding"] = binding
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
        capacity = count.value
        handles = (ctypes.c_void_p * capacity)()
        if capacity:
            self._call(self.runtime.cudaGraphGetNodes, graph, handles, ctypes.byref(count))
            confirmed = ctypes.c_size_t()
            self._call(self.runtime.cudaGraphGetNodes, graph, None, ctypes.byref(confirmed))
            if count.value != capacity or confirmed.value != capacity:
                raise RuntimeError("native capture node count changed during its read-only enumeration")
        # CUDA13 rejects a nonnull zero-capacity output array. An empty result
        # is based on the actual native count, and every boundary queries again.
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
        capacity = count.value
        sources, targets = (ctypes.c_void_p * capacity)(), (ctypes.c_void_p * capacity)()
        metadata = (GraphEdgeData * capacity)()
        if capacity:
            self._call(self.runtime.cudaGraphGetEdges, graph, sources, targets, metadata, ctypes.byref(count))
            confirmed = ctypes.c_size_t()
            self._call(self.runtime.cudaGraphGetEdges, graph, None, None, None, ctypes.byref(confirmed))
            if count.value != capacity or confirmed.value != capacity:
                raise RuntimeError("native capture edge count changed during its read-only enumeration")
        if any(
            source not in by_handle or target not in by_handle for source, target in zip(sources, targets, strict=True)
        ):
            raise RuntimeError("native capture edge references a node absent from the actual node snapshot")
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
        token = {"name": name, "source": source, "index": len(self.calls), "completed": False}
        self.calls.append(token)
        self.stack.append(token)
        return token

    def leave(self, token):
        if not self.stack or self.stack[-1] is not token:
            raise RuntimeError("native operation capture boundaries are not properly nested")
        self._flush()
        token["owned_node_ids"] = sorted(node for node, owner in self.owners.items() if owner["name"] == token["name"])
        token["completed"] = True
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


def _active_union_us(rows: list[dict]) -> float:
    """Union of measured device activity intervals, excluding idle gaps."""
    total, start, end = 0.0, None, None
    for row in sorted(rows, key=lambda row: row["start_us"]):
        if start is None:
            start, end = row["start_us"], row["end_us"]
        elif row["start_us"] > end:
            total += end - start
            start, end = row["start_us"], row["end_us"]
        else:
            end = max(end, row["end_us"])
    return total + (end - start if start is not None else 0.0)


def bind_replay_kernels(registry: dict, events: list[dict], *, correlation: int) -> dict:
    """Join one real graph launch by CUPTI IDs, retaining overlap explicitly.

    Kernels and graph memcpy/memset nodes require complete activity evidence.
    Structural nodes remain explicit, without assigning them invented costs.
    It does not turn sums into a whole-forward prediction or certify accuracy.
    """
    node_types = {"kernel": 0, "gpu_memcpy": 1, "gpu_memset": 2}
    expected = {row["node_id"]: row for row in registry["nodes"] if row["node_type"] in node_types.values()}
    structural = [row for row in registry["nodes"] if row["node_type"] not in node_types.values()]
    if any(row["node_type"] not in (5, 6, 7) for row in structural):
        raise ValueError("native graph has an unqualified host/control/memory-allocation node")
    actual, streams = {}, set()
    for event in events:
        args = event.get("args", {})
        category = event.get("cat")
        if category not in node_types or args.get("correlation") != correlation:
            continue
        node = args.get("graph node id")
        if (
            args.get("graph id") != registry["graph_id"]
            or node not in expected
            or node in actual
            or expected[node]["node_type"] != node_types[category]
        ):
            raise ValueError("CUPTI replay kernel does not match a unique captured native node")
        if type(args.get("stream")) is not int:
            raise ValueError("CUPTI replay kernel lacks actual stream/launch geometry")
        if category == "kernel":
            if (
                any(not isinstance(args.get(key), list) or len(args[key]) != 3 for key in ("grid", "block"))
                or type(args.get("shared memory")) is not int
            ):
                raise ValueError("CUPTI replay kernel lacks actual stream/launch geometry")
            fingerprint = {key: args[key] for key in ("grid", "block", "shared memory")}
        else:
            if type(args.get("bytes")) is not int or args["bytes"] <= 0:
                raise ValueError("CUPTI replay memory node lacks positive actual byte count")
            fingerprint = {"bytes": args["bytes"]}
        start, duration = event.get("ts"), event.get("dur")
        if (
            any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, duration))
            or duration <= 0
        ):
            raise ValueError("CUPTI replay lacks a positive native kernel interval")
        streams.add(args.get("stream"))
        actual[node] = {
            "node_id": node,
            "activity": category,
            "operation": expected[node]["name"],
            "start_us": start,
            "end_us": start + duration,
            "stream": args.get("stream"),
            "fingerprint": {"name": event["name"], **fingerprint},
        }
    if not expected or actual.keys() != expected.keys():
        raise ValueError("CUPTI launch omits captured native kernel or memory nodes")
    rows = sorted(actual.values(), key=lambda row: (row["start_us"], row["node_id"]))
    kernels = [row for row in rows if row["activity"] == "kernel"]
    if not kernels:
        raise ValueError("native model graph replay has no measured kernel nodes")
    overlaps = [
        [left["node_id"], right["node_id"]]
        for index, left in enumerate(rows)
        for right in rows[index + 1 :]
        if right["start_us"] < left["end_us"]
    ]
    groups = {}
    for row in rows:
        groups.setdefault(row["operation"], []).append(row)
    units = [
        {
            "operation": operation,
            "node_ids": [row["node_id"] for row in owned],
            "active_union_us": _active_union_us(owned),
            "activity_interval_sum_us": sum(row["end_us"] - row["start_us"] for row in owned),
            "activity_envelope_us": max(row["end_us"] for row in owned) - min(row["start_us"] for row in owned),
        }
        for operation, owned in groups.items()
    ]
    return {
        "graph_id": registry["graph_id"],
        "correlation": correlation,
        "kernels": kernels,
        "activities": rows,
        "unmeasured_structural_nodes": structural,
        "overlapping_node_pairs": overlaps,
        "streams": sorted(streams),
        "kernel_interval_sum_us": sum(row["end_us"] - row["start_us"] for row in kernels),
        "kernel_envelope_us": max(row["end_us"] for row in kernels) - min(row["start_us"] for row in kernels),
        "activity_interval_sum_us": sum(row["end_us"] - row["start_us"] for row in rows),
        "activity_envelope_us": max(row["end_us"] for row in rows) - min(row["start_us"] for row in rows),
        "activity_union_us": _active_union_us(rows),
        "operation_activity_unions": units,
        "approximate_additive_operation_union_us": sum(row["active_union_us"] for row in units),
        "composition": "disjoint_node_ownership_additive_active_unions_not_critical_path",
        "whole_forward_accuracy": "NOT_EVALUATED",
        "formal_admission": False,
    }


EXECUTION_RANGE = "aisim.glm53/native_decode_execute"


def trace_forward_identity(record: dict) -> dict:
    """Original profiler document's one native forward, before serialization."""
    fields = (
        "run_id",
        "request_set",
        "tp_rank",
        "invocation",
        "forward_id",
        "phase",
        "benchmark_id",
        "repetition",
        "sampling_role",
        "dataset_role",
        "corpus_sha256",
        "request_ids",
    )
    identity = {key: record[key] for key in fields}
    if (
        any(
            type(identity[key]) is not int or identity[key] < 0
            for key in ("tp_rank", "invocation", "benchmark_id", "repetition")
        )
        or any(
            not isinstance(identity[key], str) or not identity[key]
            for key in ("run_id", "request_set", "forward_id", "corpus_sha256")
        )
        or identity["sampling_role"] not in ("warmup", "measurement")
    ):
        raise ValueError("native graph trace lacks an exact forward identity")
    return {"schema": "glm53flash_graph_trace_forward_v1", **identity}


def bind_execution_activity(binding: dict, events: list[dict]) -> dict:
    """Account for native GPU preparation outside the captured model graph.

    Each extra activity must correlate to a CUDA launch inside the one actual
    source-bound DecodeCudaGraphRunner.execute CPU range. An additional graph
    needs its own capture registry; it cannot be relabeled as ordinary setup.
    CPU gaps remain diagnostic elapsed time, never allocated back into units.
    """
    ranges = [row for row in events if row.get("name") == EXECUTION_RANGE and row.get("ph") == "X"]
    if len(ranges) != 1:
        raise ValueError("native execution lacks one complete source-bound profiler range")
    region = ranges[0]
    begin, duration = region.get("ts"), region.get("dur")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (begin, duration)) or duration <= 0:
        raise ValueError("native execution profiler range lacks a positive interval")
    if any(type(region.get(key)) is not int for key in ("pid", "tid")):
        raise ValueError("native execution profiler range lacks its actual process/thread")
    calls = {}
    for row in events:
        if row.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        start, length = row.get("ts"), row.get("dur")
        if (
            row.get("pid") != region.get("pid")
            or row.get("tid") != region.get("tid")
            or type(start) not in (int, float)
            or type(length) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(length)
            or length < 0
            or start < begin
            or start + length > begin + duration
        ):
            continue
        correlation = row.get("args", {}).get("correlation")
        if type(correlation) is not int or correlation < 0:
            raise ValueError("native execution CUDA call lacks its actual launch correlation")
        if correlation in calls:
            raise ValueError("native execution CUDA call correlation is ambiguous")
        calls[correlation] = row
    launches = [row for row in calls.values() if row.get("name") in ("cudaGraphLaunch", "cuGraphLaunch")]
    if len(launches) != 1 or launches[0]["args"]["correlation"] != binding["correlation"]:
        raise ValueError("native metadata/model execution needs one registered graph launch")
    setup = []
    actual_graph = []
    for index, event in enumerate(events):
        category, args = event.get("cat"), event.get("args", {})
        if category not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        correlation = args.get("correlation")
        if correlation not in calls:
            raise ValueError("GPU activity is outside the source-bound native execution scope")
        if correlation == binding["correlation"]:
            actual_graph.append(event)
            continue
        if args.get("graph id", 0) not in (0, None) or args.get("graph node id", 0) not in (0, None):
            raise ValueError("native preparation contains an unregistered additional CUDA graph")
        start, length = event.get("ts"), event.get("dur")
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, length))
            or length <= 0
            or type(args.get("stream")) is not int
        ):
            raise ValueError("native preparation activity lacks a positive actual device interval")
        if category == "kernel":
            if (
                any(not isinstance(args.get(k), list) or len(args[k]) != 3 for k in ("grid", "block"))
                or type(args.get("shared memory")) is not int
            ):
                raise ValueError("native preparation kernel lacks actual launch geometry")
            fingerprint = {k: args[k] for k in ("grid", "block", "shared memory")}
        else:
            if type(args.get("bytes")) is not int or args["bytes"] <= 0:
                raise ValueError("native preparation memory activity lacks actual byte count")
            fingerprint = {"bytes": args["bytes"]}
        setup.append(
            {
                "activity_index": index,
                "launch_correlation": correlation,
                "activity": category,
                "operation": "native_graph_setup",
                "start_us": start,
                "end_us": start + length,
                "stream": args["stream"],
                "fingerprint": {"name": event["name"], **fingerprint},
                "source_boundary": "sglang.DecodeCudaGraphRunner.execute",
                "native_launch": calls[correlation]["name"],
            }
        )
    # Prove both directions: absence of CUPTI activity is not a zero-cost
    # memory/kernel launch. A zero-byte call also needs explicit native proof;
    # profiler API names alone cannot establish zero work.
    for correlation, call in calls.items():
        name = call["name"]
        expected = (
            "gpu_memcpy"
            if name.startswith(("cudaMemcpy", "cuMemcpy"))
            else "gpu_memset"
            if name.startswith(("cudaMemset", "cuMemset"))
            else "kernel"
            if name.startswith(
                ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchCooperativeKernel", "cuLaunchCooperativeKernel")
            )
            else None
        )
        # Only known host/control APIs may lack a GPU activity. Unknown CUDA
        # dispatch is an unsupported scope, never an implicit zero-cost setup.
        control_prefixes = (
            "cudaEvent",
            "cuEvent",
            "cudaStreamSynchronize",
            "cuStreamSynchronize",
            "cudaStreamWaitEvent",
            "cuStreamWaitEvent",
            "cudaStreamIsCapturing",
            "cuStreamIsCapturing",
            "cudaStreamGetCaptureInfo",
            "cuStreamGetCaptureInfo",
            "cudaGetDevice",
            "cuDeviceGet",
            "cudaSetDevice",
            "cudaDeviceGet",
            "cudaGetLastError",
            "cudaPeekAtLastError",
            "cudaPointerGetAttributes",
            "cuPointerGetAttribute",
            "cudaFuncGetAttributes",
            "cuFuncGetAttribute",
            "cudaOccupancy",
            "cuOccupancy",
            "cudaMemGetInfo",
            "cuMemGetInfo",
            "cuCtxGetCurrent",
            "cuCtxSetCurrent",
            "cudaDeviceSynchronize",
            "cuCtxSynchronize",
        )
        if (
            expected is None
            and name not in ("cudaGraphLaunch", "cuGraphLaunch")
            and not name.startswith(control_prefixes)
        ):
            raise ValueError("unknown native CUDA dispatch cannot be admitted as zero-cost setup")
        if expected is not None and not any(
            row["launch_correlation"] == correlation and row["activity"] == expected for row in setup
        ):
            raise ValueError("native device-work call lacks its matching GPU activity; zero work is unproved")
    if len(actual_graph) != len(binding["activities"]):
        raise ValueError("native graph activity differs between node and execution scope proofs")
    combined = binding["activities"] + setup
    groups = {}
    for row in combined:
        groups.setdefault(row["operation"], []).append(row)
    units = [
        {
            "operation": operation,
            "active_union_us": _active_union_us(rows),
            "activity_interval_sum_us": sum(row["end_us"] - row["start_us"] for row in rows),
            "activity_envelope_us": max(row["end_us"] for row in rows) - min(row["start_us"] for row in rows),
            "captured_node_ids": [row["node_id"] for row in rows if "node_id" in row],
            "outside_graph_activity_indices": [row["activity_index"] for row in rows if "activity_index" in row],
        }
        for operation, rows in groups.items()
    ]
    return {
        "graph": binding,
        "outside_graph_setup": setup,
        "execution_range": region,
        "activities": combined,
        "operation_activity_unions": units,
        "activity_union_us": _active_union_us(combined),
        "activity_interval_sum_us": sum(row["end_us"] - row["start_us"] for row in combined),
        "activity_envelope_us": max(row["end_us"] for row in combined) - min(row["start_us"] for row in combined),
        "approximate_additive_operation_union_us": sum(row["active_union_us"] for row in units),
        "composition": binding["composition"],
        "formal_admission": False,
        "whole_forward_accuracy": "NOT_EVALUATED",
    }
