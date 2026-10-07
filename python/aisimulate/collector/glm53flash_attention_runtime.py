# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework-neutral GPU helpers shared by the GLM attention runners.

Both runners replay the loaded attention module under the framework's own
serving graph mechanism (prefill: breakable/piecewise prefill graph with eager
breaks; decode: a full CUDA graph built from the framework's decode-graph
capture) and time it with ``KernelTimer``: GPU kernel time only, the union of
the GPU-busy intervals of the CUPTI kernel/memcpy/memset activities that
belong to each repetition. ``EventTimer`` keeps the previous host-inclusive
CUDA-event interval as a diagnostic.
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
    indexer_regime,
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


class EventTimer:
    """Enqueue every repetition back to back, then read all intervals."""

    def __init__(self, torch):
        self.torch = torch
        self.pairs = []
        self.host_ms = []

    def __call__(self, fn):
        import time

        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        # Host enqueue time of the same call: diagnoses launch-bound timings.
        self.host_ms.append((time.perf_counter() - begin) * 1e3)
        self.pairs.append((start, end))
        return result

    def read(self) -> list[float]:
        self.torch.cuda.synchronize()
        values = [float(start.elapsed_time(end)) for start, end in self.pairs]
        self.pairs.clear()
        return values


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


def attribute_repetitions(trace_events: list[dict], repetitions: int, device: int | None) -> dict:
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

    Returns ``{"repetitions": [per-repetition stats], "diagnostics": {...}}``;
    ``busy_us`` is the union of the repetition's GPU-busy intervals.
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
    per = [
        {"intervals": [], "kernel_us": 0.0, "kernels": 0, "memcpy": 0, "memset": 0, "time_only": 0}
        for _ in range(repetitions)
    ]
    diagnostics = {"devices": set(), "outside": 0, "time_only": 0}
    for event in trace_events:
        category = event.get("cat")
        if category not in GPU_ACTIVITY_CATEGORIES:
            continue
        args = event.get("args") or {}
        if device is not None and args.get("device") is not None and int(args["device"]) != device:
            continue
        diagnostics["devices"].add(args.get("device"))
        start = float(event["ts"])
        end = start + float(event.get("dur", 0.0))
        by_time = _containing(ordered, start, end)
        correlation = args.get("correlation")
        known = correlation is not None and int(correlation) in launch_repetition
        by_launch = launch_repetition.get(int(correlation)) if known else None
        if known and by_launch is None:
            if by_time is not None:
                raise TimingAttributionError(f"{event.get('name')} launched outside every repetition ran inside one")
            diagnostics["outside"] += 1
            continue
        if known:
            if by_time is not None and by_time != by_launch:
                raise TimingAttributionError(
                    f"{event.get('name')} launched in repetition {by_launch} but ran in repetition {by_time}"
                )
            repetition = by_launch
        elif by_time is not None:
            repetition = by_time
            per[repetition]["time_only"] += 1
            diagnostics["time_only"] += 1
        else:
            diagnostics["outside"] += 1
            continue
        stats = per[repetition]
        stats["intervals"].append((start, end))
        if category == "kernel":
            stats["kernels"] += 1
            stats["kernel_us"] += end - start
        elif category == "gpu_memcpy":
            stats["memcpy"] += 1
        else:
            stats["memset"] += 1
    results = []
    for index, stats in enumerate(per):
        if not stats["kernels"]:
            raise TimingAttributionError(f"repetition {index} has no attributed GPU kernel")
        intervals = stats.pop("intervals")
        low, high = ranges[index]
        results.append(
            {
                **stats,
                "busy_us": _union_us(intervals),
                "activity_sum_us": sum(end - start for start, end in intervals),
                "gpu_span_us": max(end for _, end in intervals) - min(start for start, _ in intervals),
                "host_range_us": high - low,
            }
        )
    diagnostics["devices"] = sorted(d for d in diagnostics["devices"] if d is not None)
    return {"repetitions": results, "diagnostics": diagnostics}


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

    def measure(self, fn, repetitions: int) -> dict:
        torch = self.torch
        from torch.profiler import ProfilerActivity, profile, record_function

        events = []
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            for index in range(repetitions):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with record_function(f"{REPETITION_RANGE}{index}"):
                    start.record()
                    result = fn()
                    end.record()
                    torch.cuda.synchronize()
                events.append((start, end))
        self.scratch.mkdir(parents=True, exist_ok=True)
        path = self.scratch / f"trace-{os.getpid()}.json"
        try:
            profiler.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
        finally:
            path.unlink(missing_ok=True)
        attributed = attribute_repetitions(trace["traceEvents"], repetitions, torch.cuda.current_device())
        for stats, (start, end) in zip(attributed["repetitions"], events, strict=True):
            # Diagnostic only: the host-inclusive interval under the profiler.
            stats["event_ms_profiled"] = float(start.elapsed_time(end))
        attributed["result"] = result
        return attributed


def kernel_samples(attributed: dict, event_ms: list[float], warmup: int) -> tuple[list[float], dict]:
    """Per-repetition kernel-only latencies (ms) and their timing diagnostics.

    ``event_ms`` are the host-inclusive CUDA-event intervals of the same call
    replayed back to back without the profiler (the previous timing method),
    kept for comparison only. Diagnostics cover the published repetitions.
    """
    reps = attributed["repetitions"]
    if len(event_ms) != len(reps):
        raise ValueError("event and kernel timings cover different repetition counts")
    latencies = [stats["busy_us"] / 1e3 for stats in reps]
    timed = reps[warmup:]

    def column(name, scale=None, digits=5):
        if scale is None:
            return [int(stats[name]) for stats in timed]
        return [round(stats[name] * scale, digits) for stats in timed]

    return latencies, {
        "kernel_busy_ms": column("busy_us", 1e-3),
        "kernel_sum_ms": column("kernel_us", 1e-3),
        "activity_sum_ms": column("activity_sum_us", 1e-3),
        "gpu_span_ms": column("gpu_span_us", 1e-3),
        "kernel_count": column("kernels"),
        "memcpy_count": column("memcpy"),
        "memset_count": column("memset"),
        "time_only_count": column("time_only"),
        "event_ms_profiled": column("event_ms_profiled", 1.0),
        "event_ms_unprofiled": [round(v, 5) for v in event_ms[warmup:]],
        "attribution": attributed["diagnostics"],
    }


def time_replays(torch, replay, warmup: int, iterations: int) -> tuple[list[float], dict]:
    """Kernel-only latencies of ``replay`` and their diagnostics.

    First the call is replayed back to back under CUDA events without the
    profiler (the previous host-inclusive method, kept as a diagnostic), then
    timed again under ``KernelTimer``; published latencies are the latter.
    """
    import tempfile

    repetitions = warmup + iterations
    timer = EventTimer(torch)
    for _ in range(repetitions):
        timer(replay)
    event_ms = timer.read()
    host = [round(v, 4) for v in timer.host_ms[warmup:]]
    attributed = KernelTimer(torch, Path(tempfile.gettempdir()) / "glm53-attention-traces").measure(replay, repetitions)
    latencies, timing = kernel_samples(attributed, event_ms, warmup)
    return latencies, {**timing, "host_enqueue_ms": host}


class RawWriter:
    """Per-rank raw JSONL stream of repetition samples and diagnostics."""

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
            "indexer_regime": indexer_regime(phase, target["prefix"], target["x"], self.key_base["index_topk"]),
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

    def diagnostic(self, payload: dict) -> None:
        with self.path.open("a") as stream:
            stream.write(json.dumps({"record": "diagnostic", "tp_rank": self.rank, **payload}) + "\n")


def target_id(phase: str, batch: int, prefix: int, x: int) -> str:
    return f"{phase}-b{batch}-p{prefix}-x{x}"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
