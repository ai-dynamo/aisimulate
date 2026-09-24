# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only CUDA capture-node ownership, joined to actual CUPTI replay data.

Original bindings to documented NVIDIA CUDA/CUPTI APIs; no CUDA source copied.
See README.glm53flash.md for API references and qualification limits. No event,
kernel, dependency, stream wait or other graph node is inserted by this module.

The memcpy trace contract references Kineto 094d3c1d072362d0a919a77299459eee94f97931,
libkineto/src/{CuptiActivity.h,cupti_strings.cpp} and include/ActivityType.h;
independently authored strict
parser, not copied C++ code. BSD attribution is in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math
from pathlib import Path

MEMCPY_TORCH_REVISION = "cf30153c4c131c8164ee7798e5022d810682e2cb"
MEMCPY_KINETO_REVISION = "094d3c1d072362d0a919a77299459eee94f97931"
MEMCPY_TRACE_NAME = "Memcpy DtoD (Device -> Device)"


class GraphCopyPosition(ctypes.Structure):
    _fields_ = tuple((name, ctypes.c_size_t) for name in ("x", "y", "z"))


class GraphCopyExtent(ctypes.Structure):
    _fields_ = tuple((name, ctypes.c_size_t) for name in ("width", "height", "depth"))


class GraphCopyPointer(ctypes.Structure):
    _fields_ = (("ptr", ctypes.c_void_p),) + tuple((name, ctypes.c_size_t) for name in ("pitch", "xsize", "ysize"))


class GraphCopyParams(ctypes.Structure):
    """Documented cudaMemcpy3DParms ABI, checked against the native header."""

    _fields_ = (
        ("srcArray", ctypes.c_void_p),
        ("srcPos", GraphCopyPosition),
        ("srcPtr", GraphCopyPointer),
        ("dstArray", ctypes.c_void_p),
        ("dstPos", GraphCopyPosition),
        ("dstPtr", GraphCopyPointer),
        ("extent", GraphCopyExtent),
        ("kind", ctypes.c_int),
    )


def memcpy_activity_requirement(node):
    """A pending source copy identity, never evidence that replay took place."""
    proof = node.get("memcpy_params", {})
    params = proof.get("parameters", {})
    extent = params.get("extent", {})
    if (
        node.get("node_type") != 1
        or proof.get("api") != "cudaGraphMemcpyNodeGetParams"
        or type(proof.get("rc")) is not int
        or proof["rc"] != 0
        or type(proof.get("source_node_handle")) is not int
        or proof["source_node_handle"] <= 0
        or proof.get("trace_producer")
        != {"torch_git_version": MEMCPY_TORCH_REVISION, "kineto_gitlink": MEMCPY_KINETO_REVISION}
        or type(params.get("kind")) is not int
        or params["kind"] != 3
        or any(type(params.get(key)) is not int or params[key] != 0 for key in ("srcArray", "dstArray"))
        or any(params.get(key) != {"x": 0, "y": 0, "z": 0} for key in ("srcPos", "dstPos"))
        or any(type(params[key][axis]) is not int for key in ("srcPos", "dstPos") for axis in ("x", "y", "z"))
        or any(type(extent.get(key)) is not int for key in ("width", "height", "depth"))
        or extent.get("width", 0) <= 0
        or extent.get("height") != 1
        or extent.get("depth") != 1
        or any(
            type(params.get(key, {}).get("ptr")) is not int or params[key]["ptr"] <= 0 for key in ("srcPtr", "dstPtr")
        )
    ):
        raise ValueError("native memcpy requires checked source parameters for a one-dimensional D2D copy")
    return {"category": "gpu_memcpy", "name": MEMCPY_TRACE_NAME, "bytes": extent["width"]}


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
        self._bind(self.runtime, "cudaGraphMemcpyNodeGetParams", [pointer, ctypes.POINTER(GraphCopyParams)])
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
            if node_type.value == 1:
                import torch

                params = GraphCopyParams()
                rc = self.runtime.cudaGraphMemcpyNodeGetParams(handle, ctypes.byref(params))

                def fields(value):
                    return {name: int(getattr(value, name) or 0) for name, _ in value._fields_}

                nodes[node_id.value]["memcpy_params"] = {
                    "api": "cudaGraphMemcpyNodeGetParams",
                    "source_node_handle": handle,
                    "rc": rc,
                    "parameters": {
                        name: fields(getattr(params, name))
                        if isinstance(getattr(params, name), ctypes.Structure)
                        else int(getattr(params, name) or 0)
                        for name, _ in params._fields_
                    },
                    "trace_producer": {
                        "torch_git_version": torch.version.git_version,
                        "kineto_gitlink": MEMCPY_KINETO_REVISION,
                    },
                }
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
        if any(
            {key: item for key, item in self.owners[node].items() if key != "name"} != value["nodes"][node]
            for node in self.seen
        ):
            raise RuntimeError("native capture node type or copy parameters changed after ownership")
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
            requirement = expected[node].get("memcpy_activity_requirement")
            if requirement is not None:
                if (
                    requirement != memcpy_activity_requirement(expected[node])
                    or category != requirement["category"]
                    or event.get("name") != requirement["name"]
                    or args["bytes"] != requirement["bytes"]
                ):
                    raise ValueError("pending native memcpy lacks its exact replay category, direction, and bytes")
                fingerprint["copy_direction"] = "D2D"
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
VLLM_EXECUTION_RANGE = "aisim.glm53/vllm_metadata_to_logits"
VLLM_LOGITS_RANGE = "aisim.glm53/vllm_logits_processor"


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


def bind_execution_activity(binding: dict, events: list[dict], *, additional_bindings=()) -> dict:
    """Account for native GPU preparation outside the captured model graph.

    Each extra activity must correlate to a CUDA launch inside the one actual
    source-bound DecodeCudaGraphRunner.execute CPU range. An additional graph
    needs its own capture registry; it cannot be relabeled as ordinary setup.
    CPU gaps remain diagnostic elapsed time, never allocated back into units.
    Explicit additional bindings require one distinct observed graph/launch
    each; the default retains the single-graph contract. No extra graph can
    enter through ordinary setup activity.
    """
    bindings = [binding, *additional_bindings]
    correlations = {item["correlation"]: item for item in bindings}
    if len(correlations) != len(bindings) or len({item["graph_id"] for item in bindings}) != len(bindings):
        raise ValueError("native execution contains aliased graph or launch identities")
    # Kineto emits same-named GPU annotations for each stream. Only its CPU
    # user_annotation owns runtime launch calls; GPU ranges remain diagnostic.
    ranges = [
        row
        for row in events
        if row.get("name") == EXECUTION_RANGE and row.get("ph") == "X" and row.get("cat") == "user_annotation"
    ]
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
    if len(launches) != len(bindings) or {row["args"]["correlation"] for row in launches} != correlations.keys():
        raise ValueError("native metadata/model execution needs its exact registered graph launches")
    setup = []
    actual_graph = {correlation: [] for correlation in correlations}
    for index, event in enumerate(events):
        category, args = event.get("cat"), event.get("args", {})
        if category not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        correlation = args.get("correlation")
        if correlation not in calls:
            raise ValueError("GPU activity is outside the source-bound native execution scope")
        if correlation in correlations:
            actual_graph[correlation].append(event)
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
    if any(len(actual_graph[item["correlation"]]) != len(item["activities"]) for item in bindings):
        raise ValueError("native graph activity differs between node and execution scope proofs")
    if additional_bindings:
        binding = {
            "graphs": bindings,
            "activities": [row for item in bindings for row in item["activities"]],
            "composition": "disjoint_node_ownership_additive_active_unions_not_critical_path",
        }
    return _compose_execution(binding, region, setup)


def _compose_execution(binding, region, outside):
    combined = binding["activities"] + outside
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
        "outside_graph_setup": outside,
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


def bind_vllm_execution_activity(binding: dict, events: list[dict]) -> dict:
    """Bind V2 metadata→FULL hidden states→separate native logits activity.

    The frozen V2 boundary supplies one complete outer range and the actual
    LogitsProcessor.forward range. Ownership follows the CPU launch correlation,
    never GPU order or a whole-forward residual. The same bidirectional device
    activity checks used by SGLang remain mandatory. Compiled or graph-backed
    logits require a separate registry and therefore remain unsupported here.
    """
    ranges = [
        row
        for row in events
        if row.get("name") == VLLM_EXECUTION_RANGE and row.get("ph") == "X" and row.get("cat") == "user_annotation"
    ]
    logits = [
        row
        for row in events
        if row.get("name") == VLLM_LOGITS_RANGE and row.get("ph") == "X" and row.get("cat") == "user_annotation"
    ]
    if len(ranges) != 1 or len(logits) != 1 or any(row.get("name") == EXECUTION_RANGE for row in events):
        raise ValueError("native V2 execution requires one independent outer and logits boundary")
    region, unit = ranges[0], logits[0]
    for row in (region, unit):
        if (
            any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) for key in ("ts", "dur"))
            or row["dur"] <= 0
            or any(type(row.get(key)) is not int for key in ("pid", "tid"))
        ):
            raise ValueError("native V2 profiler boundary lacks a complete actual interval")
    if (
        any(unit[key] != region[key] for key in ("pid", "tid"))
        or unit["ts"] < region["ts"]
        or unit["ts"] + unit["dur"] > region["ts"] + region["dur"]
    ):
        raise ValueError("native logits boundary is outside the same-thread V2 execution")
    correlations = set()
    for row in events:
        if row.get("cat") not in ("cuda_runtime", "cuda_driver") or any(
            row.get(key) != region[key] for key in ("pid", "tid")
        ):
            continue
        begin, duration = row.get("ts"), row.get("dur")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (begin, duration)) or duration < 0:
            raise ValueError("native V2 CUDA call lacks its complete source-bound interval")
        left, right = unit["ts"], unit["ts"] + unit["dur"]
        if duration == 0 and begin in (left, right):
            raise ValueError("zero-duration CUDA call at logits boundary has ambiguous ownership")
        if begin < right and begin + duration > left:
            if begin < left or begin + duration > right:
                raise ValueError("native CUDA call straddles logits ownership boundaries")
            if row.get("name") in ("cudaGraphLaunch", "cuGraphLaunch"):
                raise ValueError("native logits contains an unregistered graph")
            correlations.add(row.get("args", {}).get("correlation"))
    # Reuse the complete existing API↔GPU activity proof, without mutating the
    # original trace. This name substitution carries no timing or cost values.
    scoped = [dict(row, name=EXECUTION_RANGE) if row is region else row for row in events]
    checked = bind_execution_activity(binding, scoped)
    outside = checked["outside_graph_setup"]
    for row in outside:
        is_logits = row["launch_correlation"] in correlations
        row["operation"] = "logits" if is_logits else "native_graph_setup"
        row["source_boundary"] = (
            "vllm.LogitsProcessor.forward" if is_logits else "vllm.GPUModelRunner.metadata_to_logits"
        )
    if not any(row["operation"] == "logits" and row["activity"] == "kernel" for row in outside):
        raise ValueError("native logits lacks actual projected GPU activity")
    if any(row["operation"] == "logits" for row in binding["activities"]):
        raise ValueError("native logits cannot be charged inside and outside the same graph")
    result = _compose_execution(binding, region, outside)
    result["outside_graph_setup"] = [row for row in outside if row["operation"] == "native_graph_setup"]
    result["outside_graph_operations"] = [row for row in outside if row["operation"] != "native_graph_setup"]
    result["logits_range"] = unit
    return result
