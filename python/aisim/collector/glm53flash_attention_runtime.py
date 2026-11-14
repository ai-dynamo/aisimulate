# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework-neutral GPU helpers shared by the GLM attention runners.

Both runners replay the loaded attention module under the framework's own
serving graph mechanism (prefill: breakable/piecewise prefill graph with eager
breaks; decode: a full CUDA graph built from the framework's decode-graph
capture) and time it with ``KernelTimer``: GPU kernel time only, the union of
the GPU-busy intervals of the CUPTI kernel/memcpy/memset activities that
belong to each repetition.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

from collector.glm53flash_attention_contract import (
    KERNEL_DECODE,
    KERNEL_PREFILL,
    TIMING_METHODS,
    attention_body,
    geometry_key,
    sha256_json,
)


def package_source_sha256(package_root: Path) -> tuple[str, dict]:
    """Hash every Python source of the imported framework package."""
    sources = {
        str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
    }
    return sha256_json(sources), sources


def config_sha256(checkpoint: Path) -> str:
    return sha256_json(json.loads((Path(checkpoint) / "config.json").read_text()))


def request_tokens(tokens: list[int], request: int, length: int) -> list[int]:
    """Distinct window of the generated ids per request (deterministic offset)."""
    start = (request * 4099) % (len(tokens) - length + 1)
    return tokens[start : start + length]


# Chrome-trace categories of the torch profiler (kineto) export.
GPU_ACTIVITY_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")
REPETITION_RANGE = "glm53_attention_repetition_"


class TimingAttributionError(RuntimeError):
    """The profiler trace cannot be attributed to repetitions unambiguously."""


def _union_us(intervals: list[tuple[float, float]]) -> float:
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + (current_end - current_start if current_end is not None else 0.0)


def _containing(ranges: list[tuple[float, float, int]], start: float, end: float) -> int | None:
    for low, high, repetition in ranges:
        if low <= start and end <= high:
            return repetition
    return None


def attribute_repetitions(trace_events: list[dict], repetitions: int, device: int | None) -> list[float]:
    """Assign the GPU activities of a profiler trace to timed repetitions.

    Pure function over chrome-trace events (``ts``/``dur`` in microseconds).
    Repetition ``i`` is the host range ``REPETITION_RANGE + str(i)``
    (``user_annotation``) that encloses its launches and the device
    synchronization that ends it, so repetitions never overlap on the GPU.
    A GPU activity (kernel, memcpy, memset) belongs to the repetition whose
    range contains the runtime/driver call with the same correlation id. An
    activity without a recorded launch call is attributed by time only if it
    lies entirely inside one range; one whose launch lies outside every range
    but whose execution lies inside a range is foreign work and fails, as does
    a correlation/time contradiction or a repetition without GPU work.

    Returns the GPU-busy union (microseconds) of each repetition.
    """
    ranges: dict[int, tuple[float, float]] = {}
    for event in trace_events:
        name = event.get("name", "")
        if event.get("cat") == "user_annotation" and name.startswith(REPETITION_RANGE):
            index = int(name[len(REPETITION_RANGE) :])
            if index in ranges:
                raise TimingAttributionError(f"repetition range {index} recorded twice")
            ranges[index] = (float(event["ts"]), float(event["ts"]) + float(event.get("dur", 0.0)))
    if sorted(ranges) != list(range(repetitions)):
        raise TimingAttributionError(f"trace has repetition ranges {sorted(ranges)}, expected {repetitions}")
    ordered = sorted((low, high, index) for index, (low, high) in ranges.items())
    for (_, high, _), (low, _, _) in itertools.pairwise(ordered):
        if low < high:
            raise TimingAttributionError("repetition ranges overlap")
    launch_repetition: dict[int, int | None] = {}
    for event in trace_events:
        if event.get("cat") not in LAUNCH_CATEGORIES:
            continue
        correlation = (event.get("args") or {}).get("correlation")
        if correlation is None:
            continue
        start = float(event["ts"])
        launch_repetition[int(correlation)] = _containing(ordered, start, start)
    intervals: list[list[tuple[float, float]]] = [[] for _ in range(repetitions)]
    kernels = [0] * repetitions
    for event in trace_events:
        category = event.get("cat")
        if category not in GPU_ACTIVITY_CATEGORIES:
            continue
        args = event.get("args") or {}
        if device is not None and args.get("device") is not None and int(args["device"]) != device:
            continue
        start = float(event["ts"])
        end = start + float(event.get("dur", 0.0))
        by_time = _containing(ordered, start, end)
        correlation = args.get("correlation")
        known = correlation is not None and int(correlation) in launch_repetition
        by_launch = launch_repetition.get(int(correlation)) if known else None
        if known and by_launch is None:
            if by_time is not None:
                raise TimingAttributionError(f"{event.get('name')} launched outside every repetition ran inside one")
            continue
        if known:
            if by_time is not None and by_time != by_launch:
                raise TimingAttributionError(
                    f"{event.get('name')} launched in repetition {by_launch} but ran in repetition {by_time}"
                )
            repetition = by_launch
        elif by_time is not None:
            repetition = by_time
        else:
            continue
        intervals[repetition].append((start, end))
        kernels[repetition] += category == "kernel"
    for index, count in enumerate(kernels):
        if not count:
            raise TimingAttributionError(f"repetition {index} has no attributed GPU kernel")
    return [_union_us(busy) for busy in intervals]


class KernelTimer:
    """GPU kernel time of ``fn`` per repetition from CUPTI activity records.

    The torch profiler (kineto/CUPTI) records CUDA activities while ``fn`` runs
    ``warmup + iterations`` times; each repetition is enclosed by a
    ``record_function`` range and ends with a device synchronization, so GPU
    work never spans two repetitions and the profiler's own start/stop work
    lies outside every range. Latency per repetition is the GPU-busy union of
    its activities, which excludes every host launch gap by construction.
    """

    def __init__(self, torch, scratch: Path):
        self.torch = torch
        self.scratch = Path(scratch)

    def measure(self, fn, repetitions: int) -> list[float]:
        """GPU-busy latency (ms) of each of ``repetitions`` calls of ``fn``."""
        torch = self.torch
        from torch.profiler import ProfilerActivity, profile, record_function

        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            for index in range(repetitions):
                with record_function(f"{REPETITION_RANGE}{index}"):
                    fn()
                    torch.cuda.synchronize()
        self.scratch.mkdir(parents=True, exist_ok=True)
        path = self.scratch / f"trace-{os.getpid()}.json"
        try:
            profiler.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
        finally:
            path.unlink(missing_ok=True)
        busy_us = attribute_repetitions(trace["traceEvents"], repetitions, torch.cuda.current_device())
        return [value / 1e3 for value in busy_us]


def time_replays(torch, replay, warmup: int, iterations: int) -> list[float]:
    """Kernel-only latencies (ms) of ``warmup + iterations`` replays of ``replay``."""
    import tempfile

    timer = KernelTimer(torch, Path(tempfile.gettempdir()) / "glm53-attention-traces")
    return timer.measure(replay, warmup + iterations)


class RawWriter:
    """Per-rank raw JSONL stream of repetition samples."""

    def __init__(self, output: Path, rank: int, key_base: dict, provenance: dict):
        self.path = Path(output) / f"rank-{rank}.jsonl"
        self.rank = rank
        self.key_base = key_base
        self.provenance = provenance

    def samples(
        self,
        target: dict,
        latencies: list[float],
        warmup: int,
        kernel_source: str,
        extra: dict,
        timing_method: str | None = None,
    ) -> None:
        phase = target["phase"]
        if timing_method is None:
            timing_method = {"context": KERNEL_PREFILL, "generation": KERNEL_DECODE}[phase]
        used_graph = TIMING_METHODS[phase][timing_method]
        key = {
            "geometry": geometry_key(attention_body(self.key_base, phase == "context")),
            "batch_size": target["batch_size"],
            "prefix": target["prefix"],
            "x": target["x"],
        }
        with self.path.open("a") as stream:
            for repetition, latency in enumerate(latencies):
                if repetition < warmup:
                    continue
                stream.write(
                    json.dumps(
                        {
                            "record": "sample",
                            "target_id": target["target_id"],
                            "tp_rank": self.rank,
                            "repetition": repetition - warmup,
                            "latency_ms": latency,
                            "key": key,
                            "provenance": self.provenance,
                            "kernel_source": kernel_source,
                            "timing_method": timing_method,
                            "used_cuda_graph": used_graph,
                            "extra": extra,
                        }
                    )
                    + "\n"
                )


def target_id(phase: str, batch: int, prefix: int, x: int) -> str:
    return f"{phase}-b{batch}-p{prefix}-x{x}"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
