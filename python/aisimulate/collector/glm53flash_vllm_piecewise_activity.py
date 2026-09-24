# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join native breakable graph launches and eager activity without repeated costs.

Original observation of vllm-project/vllm@ced6857afa0ea7b2e3f0846a62e1394e90f15607,
vllm/compilation/breakable_cudagraph.py (Apache-2.0). This does not reconstruct
native execution or provide serving dispatch, table admission or accuracy.
"""

from __future__ import annotations

import math
from itertools import pairwise

from collector.glm53flash_graph_nodes import (
    EXECUTION_RANGE,
    VLLM_EXECUTION_RANGE,
    VLLM_LOGITS_RANGE,
    _compose_execution,
    bind_execution_activity,
    bind_replay_kernels,
)
from collector.glm53flash_vllm_piecewise import BREAKABLE_SOURCE_PIN, EAGER_RANGE_PREFIX


def _interval(row, *, positive=True):
    start, length = row.get("ts"), row.get("dur")
    if (
        any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, length))
        or length < 0
        or (positive and length == 0)
        or any(type(row.get(key)) is not int for key in ("pid", "tid"))
    ):
        raise ValueError("piecewise CPU boundary lacks a complete actual interval")
    return start, start + length


def _inside(row, region):
    left, right = _interval(row, positive=False)
    begin, end = _interval(region)
    return all(row[key] == region[key] for key in ("pid", "tid")) and begin <= left <= right <= end


def _owners(capture):
    calls = capture["calls"]
    if any(row.get("index") != i or row.get("completed") is not True for i, row in enumerate(calls)):
        raise ValueError("piecewise ownership lacks complete original operation calls")

    def check(row):
        index, name = row.get("call_index"), row.get("name")
        if index is None and name == "native_graph_setup":
            return
        if type(index) is not int or not 0 <= index < len(calls) or calls[index]["name"] != name:
            raise ValueError("piecewise contribution differs from its original operation call")
        if name == "logits":
            raise ValueError("piecewise hidden states cannot own separate native logits")

    return check


def bind_piecewise_execution(capture, events):
    """Recompute contribution ownership from every graph and eager CPU range.

    Every captured segment currently requires kernel activity, as required by
    the shared node validator. Structural-only segments need separate native
    launch-identity evidence before support; they cannot silently cost zero.
    No trace event, graph node, or whole-forward residual is manufactured.
    """
    if capture.get("capture_scope") != "vllm_piecewise_hidden_states" or capture.get("uncaptured_operations") != [
        "logits"
    ]:
        raise ValueError("piecewise capture differs from the native hidden-state boundary")
    if capture.get("graph_mutations") is not False:
        raise ValueError("piecewise capture mutated the original native graph")
    segments = capture["segments"]
    if not segments or any(row.get("position") != i for i, row in enumerate(segments)):
        raise ValueError("piecewise segment positions are not the original complete ordering")
    if any(row.get("kind") not in ("graph", "eager") for row in segments):
        raise ValueError("piecewise capture contains an unknown native segment kind")
    check_owner = _owners(capture)
    graphs = [row for row in segments if row["kind"] == "graph"]
    if not graphs or any(
        len({row[key] for row in graphs}) != len(graphs) for key in ("graph_id", "capture_graph_id", "capture_id")
    ):
        raise ValueError("piecewise graph segments have missing or aliased native identities")
    for graph in graphs:
        for node in graph["nodes"]:
            check_owner(node)
    ranges = [row for row in events if row.get("name") == VLLM_EXECUTION_RANGE and row.get("ph") == "X"]
    logits = [row for row in events if row.get("name") == VLLM_LOGITS_RANGE and row.get("ph") == "X"]
    if len(ranges) != 1 or len(logits) != 1 or any(row.get("name") == EXECUTION_RANGE for row in events):
        raise ValueError("piecewise execution lacks one outer and external logits range")
    region, logits_range = ranges[0], logits[0]
    _interval(region)
    if not _inside(logits_range, region):
        raise ValueError("piecewise logits is outside the original execution thread/range")
    eager = [row for row in segments if row["kind"] == "eager"]
    expected_ranges = {row["range"]: row for row in eager}
    actual_ranges = [row for row in events if row.get("name", "").startswith(EAGER_RANGE_PREFIX)]
    if len(expected_ranges) != len(eager) or len(actual_ranges) != len(eager):
        raise ValueError("piecewise eager range count is incomplete or duplicated")
    scopes = [(logits_range, "logits", "vllm.LogitsProcessor.forward")]
    positions = {}
    for row in actual_ranges:
        owner = expected_ranges.get(row["name"])
        if (
            owner is None
            or row.get("ph") != "X"
            or not _inside(row, region)
            or owner["position"] in positions
            or owner.get("source_sha256") != BREAKABLE_SOURCE_PIN
            or owner["range"] != EAGER_RANGE_PREFIX + str(owner["eager_id"])
        ):
            raise ValueError("piecewise eager range lacks its unique source-bound native callable")
        check_owner(owner)
        positions[owner["position"]] = row
        scopes.append((row, owner["name"], "vllm.BreakableCUDAGraphCapture.eager_callable"))
    launches = [
        row
        for row in events
        if row.get("cat") in ("cuda_runtime", "cuda_driver") and row.get("name") in ("cudaGraphLaunch", "cuGraphLaunch")
    ]
    if len(launches) != len(graphs) or any(not _inside(row, region) for row in launches):
        raise ValueError("piecewise replay omits or repeats original graph launches")
    bindings = []
    for graph, launch in zip(graphs, sorted(launches, key=lambda row: _interval(row)[0]), strict=True):
        correlation = launch.get("args", {}).get("correlation")
        if type(correlation) is not int or correlation < 0:
            raise ValueError("piecewise graph launch lacks its actual correlation")
        binding = bind_replay_kernels(graph, events, correlation=correlation)
        for row in binding["activities"]:
            row["graph_id"] = graph["graph_id"]
            row["segment_position"] = graph["position"]
        bindings.append(binding)
        positions[graph["position"]] = launch
    ordered = [positions[i] for i in range(len(segments))] + [logits_range]
    if any(
        _interval(left, positive=False)[1] > _interval(right, positive=False)[0] for left, right in pairwise(ordered)
    ):
        raise ValueError("piecewise replay changed graph/eager/logits native execution order")
    ownership = {}
    for row in events:
        if row.get("cat") not in ("cuda_runtime", "cuda_driver") or not _inside(row, region):
            continue
        start, end = _interval(row, positive=False)
        for scope, operation, source in scopes:
            left, right = _interval(scope)
            if start == end and start in (left, right):
                raise ValueError("zero-duration CUDA call at piecewise boundary has ambiguous ownership")
            if start < right and end > left:
                if start < left or end > right:
                    raise ValueError("piecewise CUDA call straddles source ownership boundaries")
                if row["name"] in ("cudaGraphLaunch", "cuGraphLaunch"):
                    raise ValueError("piecewise eager/logits callable launches an unregistered graph")
                correlation = row.get("args", {}).get("correlation")
                if correlation in ownership:
                    raise ValueError("piecewise CUDA call has ambiguous source ownership")
                ownership[correlation] = operation, source
    scoped = [dict(row, name=EXECUTION_RANGE) if row is region else row for row in events]
    checked = bind_execution_activity(bindings[0], scoped, additional_bindings=bindings[1:])
    outside = checked["outside_graph_setup"]
    for row in outside:
        row["operation"], row["source_boundary"] = ownership.get(
            row["launch_correlation"], ("native_graph_setup", "vllm.GPUModelRunner.metadata_to_logits")
        )
    if not any(row["operation"] == "logits" and row["activity"] == "kernel" for row in outside):
        raise ValueError("piecewise logits lacks actual projected GPU activity")
    result = _compose_execution(checked["graph"], region, outside)
    result["outside_graph_setup"] = [row for row in outside if row["operation"] == "native_graph_setup"]
    result["outside_graph_operations"] = [row for row in outside if row["operation"] != "native_graph_setup"]
    result["logits_range"] = logits_range
    result["native_segments"] = [
        {"position": row["position"], "kind": row["kind"], "cpu_interval": positions[row["position"]]}
        for row in segments
    ]
    return result
