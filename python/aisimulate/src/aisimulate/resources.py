# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host resource accounting before expensive simulation materialization.

Estimates describe host allocations, never simulated model weights or GPU KV
capacity. They are conservative planning estimates, not promises about peak RSS.
"""

from __future__ import annotations

import bisect
import gzip
import heapq
import itertools
import json
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import ijson
import psutil

from .config.common import ResourceConfig

GB = 1_000_000_000
MIB = 1024**2
WORKER_BASELINE_BYTES = 512 * MIB
COORDINATOR_RESERVE_BYTES = 256 * MIB
_POLICY: ContextVar[ResourceConfig | None] = ContextVar("aisimulate_resource_policy", default=None)


@dataclass(frozen=True)
class HostResources:
    total_memory_bytes: int
    available_memory_bytes: int
    cpu_count: float
    process_memory_bytes: int = 0


@dataclass(frozen=True)
class ResourceEstimate:
    allocation_model: str
    request_count: int | None
    input_token_bytes: int
    lower_bound_bytes: int
    estimated_peak_bytes: int | None
    reason: str = ""
    api_version: int = 1


class ResourceLimitError(RuntimeError):
    """A host cannot safely admit this execution; it is not model infeasibility."""

    def __init__(self, message: str, *, plan: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.plan = plan or {"status": "resource_limited", "reason": message}


class _SerialAdmissionRequired(ResourceLimitError):
    """An unknown peak prevents a multi-candidate wave, regardless of headroom."""


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ResourceLimitError(f"cannot inspect container resource file {path}: {exc}") from exc


def _cgroup_directories(proc: Path, root: Path) -> list[tuple[Path, str]]:
    """Resolve membership against mounts, including a container's mount root."""
    memberships = _read_text(proc / "self/cgroup")
    mounts = _read_text(proc / "self/mountinfo")
    if memberships is None or mounts is None:
        raise ResourceLimitError("cannot inspect container resource membership and mounts")
    result: list[tuple[Path, str]] = []
    for line in memberships.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, member = parts
        for mount in mounts.splitlines():
            fields = mount.split()
            if "-" not in fields or len(fields) < 7:
                continue
            sep = fields.index("-")
            if len(fields) <= sep + 3:
                continue
            kind = fields[sep + 1]
            if kind not in {"cgroup", "cgroup2"}:
                continue
            options = set(fields[sep + 3].split(","))
            if kind == "cgroup" and not options.intersection(controllers.split(",")):
                continue
            mount_root = Path(fields[3].replace("\\040", " "))
            mount_point = root / fields[4].replace("\\040", " ").lstrip("/")
            try:
                relative = Path(member).relative_to(mount_root)
            except ValueError:
                # Cgroup namespaces expose membership relative to the mounted root.
                relative = Path(member.lstrip("/"))
                if ".." in relative.parts:
                    raise ResourceLimitError("container resource membership is outside its visible namespace") from None
            if ".." in relative.parts:
                raise ResourceLimitError("container resource membership is outside its visible namespace")
            directory = mount_point / relative
            while directory.is_relative_to(mount_point):
                result.append((directory, kind))
                if directory == mount_point:
                    break
                directory = directory.parent
    if memberships.strip() and not result:
        raise ResourceLimitError("cannot resolve container resource membership against its mounts")
    return list(dict.fromkeys(result))


def constrain_to_cgroups(host: HostResources, *, proc: Path = Path("/proc"), root: Path = Path("/")) -> HostResources:
    total, available, cpus = host.total_memory_bytes, host.available_memory_bytes, host.cpu_count
    for directory, kind in _cgroup_directories(proc, root):
        if kind == "cgroup2":
            limit = _read_text(directory / "memory.max")
            usage = _read_text(directory / "memory.current")
            cpu = (_read_text(directory / "cpu.max") or "").split()
        else:
            limit = _read_text(directory / "memory.limit_in_bytes")
            usage = _read_text(directory / "memory.usage_in_bytes")
            cpu = [_read_text(directory / "cpu.cfs_quota_us"), _read_text(directory / "cpu.cfs_period_us")]
        if limit and limit.isdecimal():
            total = min(total, int(limit))
            # Missing usage must not make a finite cgroup appear entirely free.
            available = min(available, max(0, int(limit) - int(usage)) if usage and usage.isdecimal() else 0)
        if len(cpu) == 2 and cpu[0] and cpu[1]:
            try:
                quota, period = int(cpu[0]), int(cpu[1])
            except ValueError:
                continue
            if quota > 0 and period > 0:
                cpus = min(cpus, quota / period)
    return HostResources(total, min(available, total), cpus, host.process_memory_bytes)


def discover_host() -> HostResources:
    try:
        memory = psutil.virtual_memory()
        cpus = float(os.cpu_count() or 1)
        if hasattr(os, "sched_getaffinity"):
            cpus = min(cpus, float(len(os.sched_getaffinity(0))))
        host = HostResources(memory.total, memory.available, cpus, psutil.Process().memory_info().rss)
    except (OSError, RuntimeError, psutil.Error) as exc:
        raise ResourceLimitError(f"cannot discover execution-host resources: {exc}") from exc
    return constrain_to_cgroups(host) if Path("/proc/self/cgroup").exists() else host


def resolve_budget(policy: ResourceConfig, host: HostResources) -> dict[str, Any]:
    inherited = os.environ.get("_AISIMULATE_SUPERVISED_BUDGET")
    if inherited:
        budget = json.loads(inherited)
        try:
            supervisor_rss = psutil.Process(budget["supervisor_pid"]).memory_info().rss
        except psutil.Error as exc:
            raise ResourceLimitError(f"cannot inspect execution supervisor: {exc}") from exc
        return {key: budget[key] for key in ("memory_limit_bytes", "cpu_limit", "reserved_host_memory_bytes")} | {
            "coordinator_memory_bytes": host.process_memory_bytes + supervisor_rss + COORDINATOR_RESERVE_BYTES
        }
    reserve = max(int(policy.reserve_memory_gb * GB), int(policy.reserve_memory_fraction * host.total_memory_bytes))
    headroom = max(0, host.available_memory_bytes - reserve)
    if policy.memory_limit_gb == "auto":
        budget = min(int(policy.available_memory_fraction * host.available_memory_bytes), headroom)
    else:
        budget = int(policy.memory_limit_gb * GB)
        if budget > headroom:
            raise ResourceLimitError(
                f"requested host memory budget {budget / GB:.2f} GB exceeds available headroom {headroom / GB:.2f} GB"
            )
    cpus = max(1, math.floor(host.cpu_count) - (1 if host.cpu_count > 1 else 0))
    if policy.cpu_limit != "auto":
        if policy.cpu_limit > max(1, math.floor(host.cpu_count)):
            raise ResourceLimitError("requested CPU budget exceeds the host/container CPU allowance")
        cpus = policy.cpu_limit
    return {
        "memory_limit_bytes": budget,
        "cpu_limit": cpus,
        "reserved_host_memory_bytes": reserve,
        "coordinator_memory_bytes": host.process_memory_bytes + COORDINATOR_RESERVE_BYTES,
    }


def _upper(value: Any, default: int | float) -> int | float:
    if value is None:
        return default
    if isinstance(value, Mapping):
        if "choices" in value:
            return max(value["choices"])
        if "range" in value:
            return value["range"]["max"]
    return value


def workload_bounds(config: Any) -> dict[str, Any]:
    """Bound a validated public domain without enumerating its cross product."""
    traffic = config.traffic
    if traffic is None:
        return {"isl": 1024, "osl": 128, "concurrency": 10, "request_count": 100, "source_type": "synthetic"}
    raw = traffic.model_dump(mode="python", exclude_none=True)
    source, load, stop = raw["source"], raw["load"], raw.get("stop", {})
    if source["type"] == "trace":
        return {
            "source_type": "trace",
            "trace_paths": source["paths"],
            "trace_format": source["format"],
            "trace_block_size": source.get("block_size"),
            "agentic_lanes": load.get("agentic_lanes", 1),
            **{key: load[key] for key in ("agentic_snapshot", "agentic_warmup", "agentic_profile") if key in load},
        }
    session = source["type"] == "synthetic-session"
    count = stop.get("sessions" if session else "requests")
    ratio = stop.get("sessions_per_load_unit" if session else "requests_per_load_unit", 0)
    concurrency = _upper(load.get("concurrency"), 0)
    rate = _upper(load.get("requests_per_second", load.get("sessions_per_second")), 0)
    if count is None and load["type"] == "kv_capacity_fraction":
        engine = config.engine.model_dump(mode="python", exclude_none=True)
        capacities = []
        for role in engine.get("workers", {}).values():
            cache = role.get("kv_cache") or {}
            capacity = cache.get("capacity") or {}
            if capacity.get("type") != "fixed" or cache.get("block_size") is None:
                return {"source_type": source["type"], "unresolved_resource_count": True}
            capacities.append(int(_upper(capacity["blocks"], 0)) * int(_upper(cache["block_size"], 0)))
        gpus = getattr(getattr(config, "optimization", None), "constraints", None)
        if not capacities or gpus is None:
            return {"source_type": source["type"], "unresolved_resource_count": True}
        # Each physical GPU can contribute at most one replica's fixed token
        # capacity. Ignoring prompt length deliberately overbounds concurrency.
        concurrency = max(1, math.ceil(max(capacities) * gpus.max_candidate_gpus * _upper(load["fraction"], 1)))
    count = count if count is not None else max(1, round(ratio * (concurrency or rate)))
    return {
        "source_type": source["type"],
        "isl": source.get("new_input_tokens_per_turn" if session else "input_tokens", 1024),
        "osl": source.get("output_tokens_per_turn" if session else "output_tokens", 128),
        "turns_per_session": source.get("session", {}).get("turns", 1),
        "concurrency": int(concurrency) if concurrency else None,
        "request_count": count,
    }


def _weka_active_inputs(intervals: list[tuple[float, float, int, int | None]]) -> int:
    """Bound the input tokens in a scope's causally concurrent requests.

    Sequence edges and completion frontiers normally order disjoint intervals.
    Hashless requests and epsilon joins can break a chain's recorded-end order;
    reserve all scope inputs in those cases. A detached preamble can also hide
    one main-stream request behind its later recorded end. Different scopes
    are summed, and equal starts remain concurrent even with zero API duration.
    """
    total = sum(item[2] for item in intervals)
    if any(item[3] is None for item in intervals):
        return total
    ordered = sorted(intervals, key=lambda item: item[0])
    ends = sorted(item[1] for item in ordered)
    for start, end, _, _ in ordered:
        # Match JOIN_EPSILON_SECONDS in the native Weka importer. If a
        # near-zero-duration request can extend a not-quite-finished chain,
        # its end may precede the predecessor selected by another frontier.
        next_end = bisect.bisect_right(ends, end)
        if next_end < len(ends) and ends[next_end] <= start + 1e-6:
            return total
    pending: list[tuple[float, int]] = []
    active = peak = 0
    for start, group in itertools.groupby(ordered, key=lambda item: item[0]):
        while pending and pending[0][0] <= start:
            active -= heapq.heappop(pending)[1]
        for _, end, tokens, _ in group:
            heapq.heappush(pending, (end, tokens))
            active += tokens
        peak = max(peak, active)
    if ordered and all(item[3] != ordered[0][3] for item in ordered[1:]):
        # split_preamble may prepend a disjoint first request to the main
        # stream regardless of its recorded end. A frontier can then select
        # that preamble while a later main request is still active. Only one
        # such main request can run; other streams retain the interval bound.
        peak += max((item[2] for item in ordered[1:] if item[1] < ordered[0][1]), default=0)
    return min(total, peak)


def _estimate_weka_trace(workload: Mapping[str, Any], *, inspection_budget_bytes: int | None) -> ResourceEstimate:
    """Account for the native Weka import and replay phases, without token arrays.

    These are planning allowances, not a hard RSS guarantee. Import materializes
    one play at a time before building the cached corpus graph. Replay retains
    compact hashes and planned outputs; full prompts exist for active requests.
    See docs/reference/local-resources.md for the consumer contracts and calibration.
    """
    baseline = 256 * MIB
    buffer_size = 64 * 1024
    unqualified = lambda reason: ResourceEstimate("weka-unqualified-v1", None, 0, 0, None, reason)
    expected_block_size = workload.get("trace_block_size")
    corpus_block_size = None
    plays = records = hashes = outputs = active_inputs = import_peak = 0
    paths = workload.get("trace_paths") or [workload["trace_path"]]
    try:
        for raw_path in paths:
            source = Path(raw_path)
            files = source.rglob("*") if source.is_dir() else (source,)
            for path in files:
                if path.is_symlink():
                    return unqualified("trace symlinks have no stable resource identity")
                if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                    continue
                # ijson streams arrays, but a single string scalar can still
                # expand while decoding. Bound that inspection before parsing;
                # the native per-play storage estimate is computed separately.
                inspection_peak = baseline + 4 * path.stat().st_size
                if inspection_budget_bytes is not None and inspection_peak > inspection_budget_bytes:
                    return ResourceEstimate(
                        "weka-materialized-v1",
                        None,
                        0,
                        0,
                        inspection_peak,
                        "trace scalar inspection estimate exceeds live headroom before metadata parsing",
                    )
                frames: list[tuple[str, dict[str, Any] | None]] = []
                scopes: list[tuple[str, int]] = []
                requests: list[dict[str, Any]] = []
                intervals: list[list[tuple[float, float, int, int | None]]] = []
                document_start = 0
                with path.open("rb") as stream:
                    for prefix, event, value in ijson.parse(stream, multiple_values=True, buf_size=buffer_size):
                        if event == "start_map":
                            if prefix == "":
                                document_start = max(0, stream.tell() - buffer_size)
                                requests, intervals = [], []
                                frame = {}
                            elif scopes and prefix == scopes[-1][0] + ".item":
                                frame = {"scope": scopes[-1][1], "hash_count": 0, "first_hash": None}
                            else:
                                frame = None
                            frames.append((prefix, frame))
                        elif event == "end_map":
                            map_prefix, frame = frames.pop()
                            if frame is None:
                                continue
                            if map_prefix:
                                if frame.get("type") == "subagent":
                                    continue
                                if frame.get("type") not in {"n", "s"}:
                                    raise ValueError("Weka request requires type n or s")
                                for key in ("in", "out"):
                                    length = frame.get(key)
                                    if type(length) is not int or length < (1 if key == "in" else 0):
                                        raise ValueError(
                                            "Weka request lengths must be nonnegative integers with positive in"
                                        )
                                start = float(frame["t"])
                                duration = float(frame.get("api_time") or 0)
                                end = start + duration
                                if min(start, duration) < 0 or not all(map(math.isfinite, (start, duration, end))):
                                    raise ValueError("Weka request times must be finite and nonnegative")
                                intervals[frame["scope"]].append((start, end, frame["in"], frame["first_hash"]))
                                requests.append(frame)
                                continue
                            block_size = frame.get("block_size")
                            if type(block_size) is not int or block_size <= 0:
                                raise ValueError("Weka play requires a positive integer block_size")
                            if expected_block_size is not None and expected_block_size != block_size:
                                raise ValueError("Weka source block size does not match configured block size")
                            if corpus_block_size is not None and corpus_block_size != block_size:
                                raise ValueError("Weka corpus mixes block sizes")
                            corpus_block_size = block_size
                            if not requests:
                                raise ValueError("Weka play contains no requests")
                            normalized = sum((r["in"] + block_size - 1) // block_size for r in requests)
                            source_hashes = sum(r["hash_count"] for r in requests)
                            # Source JSON, temporary normalized hashes/identities,
                            # and lowering/validation rows overlap for one play.
                            # tell() includes at most a parser buffer of lookahead.
                            document_bytes = stream.tell() - document_start
                            import_peak = max(
                                import_peak,
                                32 * document_bytes + 128 * max(normalized, source_hashes) + 32768 * len(requests),
                            )
                            plays += 1
                            records += len(requests)
                            hashes += normalized
                            outputs += sum(r["out"] for r in requests)
                            active_inputs += sum(_weka_active_inputs(scope) for scope in intervals)
                        elif event == "start_array" and frames and frames[-1][1] is not None:
                            map_prefix = frames[-1][0]
                            if prefix == (map_prefix + "." if map_prefix else "") + "requests":
                                scopes.append((prefix, len(intervals)))
                                intervals.append([])
                        elif event == "end_array" and scopes and prefix == scopes[-1][0]:
                            scopes.pop()
                        elif frames and frames[-1][1] is not None and event not in {"map_key", "end_array"}:
                            map_prefix, frame = frames[-1]
                            member = prefix.removeprefix(map_prefix + ".") if map_prefix else prefix
                            if member in {"in", "out", "t", "api_time", "type", "block_size"}:
                                frame[member] = value
                            elif member == "hash_ids.item":
                                if event != "number" or type(value) is not int or not 0 <= value < 2**64:
                                    raise ValueError("Weka hash_ids must contain unsigned 64-bit integers")
                                if not frame["hash_count"]:
                                    frame["first_hash"] = value
                                frame["hash_count"] += 1
    except (OSError, ValueError, TypeError, KeyError, OverflowError, ijson.JSONError) as exc:
        return unqualified(f"cannot inspect Weka metadata: {exc}")
    if not plays:
        return unqualified("Weka metadata contains no recognized plays")

    lanes = int(workload.get("agentic_lanes") or 1)
    snapshot = workload.get("agentic_snapshot") is not None
    # Finite ordinary lanes partition plays. Initial snapshots cycle through
    # source plays, so each corpus play is copied at most ceil(lanes / plays).
    copies = (lanes + plays - 1) // plays if snapshot else 1
    replay = copies * (32 * hashes + 32 * outputs + 32768 * records + 16 * active_inputs)
    if snapshot:
        # Immutable snapshot context owns the graph, whose hash IDs are play-local
        # u32 ranks, and the output plans.
        replay += 4 * hashes + 4 * outputs + 2048 * records
    if workload.get("agentic_warmup"):
        # Preparation and profile token payloads do not overlap across the
        # quiescent barrier; retain their extra evidence/identity allowance.
        replay += 32768 * (copies * records + 10 * lanes)
    # The complete graph and output plans are materialized even when a snapshot
    # skips history. Recorded overlap is only an upper allowance for active
    # inputs, not an unavoidable allocation or an admission lower bound.
    lower = max(8 * hashes, 4 * outputs)
    peak = max(lower, baseline + max(import_peak, replay))
    if workload.get("agentic_profile") is not None:
        if inspection_budget_bytes is not None and peak > inspection_budget_bytes:
            return ResourceEstimate(
                "agentic-profile-materialization-v1",
                None,
                0,
                lower,
                peak,
                "initial profile trace materialization estimate exceeds live headroom",
            )
        return ResourceEstimate(
            "agentic-profile-unqualified-v1",
            None,
            0,
            lower,
            None,
            "agentic profile retains evidence beyond the initial corpus; total memory has no qualified static bound "
            "and requires supervised serial execution",
        )
    return ResourceEstimate(
        "weka-materialized-v1",
        None,
        4 * active_inputs,
        lower,
        peak,
        "Weka per-play import and native replay estimate; normalized hashes, planned outputs and active prompts; "
        "runtime trace validation still required",
    )


OBSERVED_TRACE_MODEL = "trace-observed-peak-v1"
# Reserve this multiple of the largest peak a completed replay of the same trace
# reached. It is admission policy, not a model of native allocations.
OBSERVED_PEAK_MARGIN = 1.5
_TRACE_FORMATS = frozenset(
    {"mooncake", "mooncake-delta", "agentic_mooncake", "applied_compute_agentic", "dynamo", "weka"}
)


def _token_length(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("trace token lengths must be nonnegative integers")
    return value


@dataclass
class _TraceCounters:
    """Keep scalar counters only; never retain request or token arrays."""

    # Dynamo, Agentic Mooncake and Weka loaders hold every u64 source hash
    # array at once. Mooncake-family loaders convert each row as it is read.
    retains_source_hashes: bool
    # Mooncake-delta setup replaces each turn's hashes with its synthesized prompt.
    materializes_inputs: bool = False
    hashes: int = 0
    authored_outputs: int = 0
    outputs: int = 0
    largest_input: int = 0
    total_input: int = 0
    requests: int = 0

    @property
    def lower_bound(self) -> int:
        loaded = 8 * self.hashes + 4 * self.authored_outputs if self.retains_source_hashes else 0
        if self.materializes_inputs:
            prepared = max(4 * self.hashes, 4 * (self.total_input + self.outputs))
        else:
            prepared = 4 * (self.hashes + self.outputs)
        return max(loaded, prepared, 4 * self.largest_input)


class _InspectionStopped(Exception):
    """The running lower bound already exceeds the supplied headroom."""

    def __init__(self, counters: _TraceCounters) -> None:
        super().__init__()
        self.counters = counters


def _observed_peak_estimate(counters: _TraceCounters) -> ResourceEstimate:
    return ResourceEstimate(
        OBSERVED_TRACE_MODEL,
        counters.requests or None,
        0,
        counters.lower_bound,
        None,
        "trace arrays and the largest prompt provide only a lower bound; the peak is measured from completed "
        "replays of the same trace and requires supervised serial execution until observed",
    )


def _check_headroom(counters: _TraceCounters, limit: int | None) -> None:
    if limit is not None and counters.lower_bound > limit:
        raise _InspectionStopped(counters)


_MOONCAKE_INPUTS = frozenset({"input_length", "input_tokens"})
_MOONCAKE_OUTPUTS = frozenset({"output_length", "output_tokens"})


def _scan_mooncake(events: Any, counters: _TraceCounters, state: dict[str, Any], limit: int | None) -> None:
    """Count Mooncake, Mooncake-delta and Agentic Mooncake rows by their top-level fields."""
    # Mooncake loaders size an omitted prompt from its hashes; Agentic Mooncake requires it.
    block_size = state["omitted_input_block_size"]
    hashes = authored = 0
    row = False
    row_input = None
    for prefix, event, value in events:
        if prefix == "hash_ids.item":
            if event == "number":
                hashes += 1
        elif prefix == "hash_ids":
            row = True
        elif prefix == "output_token_ids.item":
            if event == "number":
                authored += 1
        elif prefix == "":
            if event == "start_map":
                row = False
                row_input = None
            elif event == "end_map":
                length = hashes * block_size if row_input is None else row_input
                counters.largest_input = max(counters.largest_input, length)
                counters.total_input += length
                counters.hashes += hashes
                counters.authored_outputs += authored
                hashes = authored = 0
                counters.requests += int(row)
                _check_headroom(counters, limit)
        elif prefix in _MOONCAKE_INPUTS:
            row_input = _token_length(value)
            row = True
        elif prefix in _MOONCAKE_OUTPUTS:
            counters.outputs += _token_length(value)
            row = True


def _scan_applied_compute(events: Any, counters: _TraceCounters, state: dict[str, Any], limit: int | None) -> None:
    """Count sessions without the synthetic hashes the loader derives from lengths.

    Each turn's prompt adds the previous response and tool output, so the final
    turn carries the session's largest prompt.
    """
    prompt = 0
    for prefix, event, value in events:
        if prefix == "assistant_response_length.item":
            length = _token_length(value)
            counters.outputs += length
            prompt += length
        elif prefix == "tool_call_output_length.item" or prefix == "input_prompt_length":
            prompt += _token_length(value)
        elif prefix == "final_assistant_response_length":
            counters.outputs += _token_length(value)
        elif prefix == "num_turns":
            counters.requests += _token_length(value) + 1
        elif prefix == "":
            if event == "start_map":
                prompt = 0
            elif event == "end_map":
                counters.largest_input = max(counters.largest_input, prompt)
                _check_headroom(counters, limit)


_DYNAMO_HASHES = "request.replay.input_sequence_hashes.item"
_WRAPPED_DYNAMO_HASHES = "event." + _DYNAMO_HASHES
_DYNAMO_FIELDS = frozenset({"event_type", "request.output_tokens", "request.replay.input_length"})


def _scan_dynamo(events: Any, counters: _TraceCounters, state: dict[str, Any], limit: int | None) -> None:
    """Count Dynamo request_end events, accepting a direct event or an outer event wrapper.

    The first request event selects standard or agentic mode for every file.
    Later requests must keep that mode.
    """
    agentic = state.get("agentic")
    direct: dict[str, Any] | None = None
    wrapper: dict[str, Any] = {}
    wrapped = False
    direct_hashes = wrapper_hashes = 0
    for prefix, event, value in events:
        if prefix == _DYNAMO_HASHES:
            if event == "number":
                direct_hashes += 1
            continue
        if prefix == _WRAPPED_DYNAMO_HASHES:
            if event == "number":
                wrapper_hashes += 1
            continue
        if prefix == "":
            if event == "start_map":
                direct, wrapper, wrapped = {}, {}, False
                direct_hashes = wrapper_hashes = 0
            elif event == "end_map" and direct is not None:
                row, row_hashes = (wrapper, wrapper_hashes) if wrapped else (direct, direct_hashes)
                if row.get("event_type") != "request_end":
                    continue
                contextual = row.get("agent_context", False)
                if agentic is not None and contextual != agentic:
                    raise ResourceLimitError("Dynamo request trace cannot mix requests with and without agent_context")
                agentic = state["agentic"] = contextual
                counters.hashes += row_hashes
                counters.outputs += _token_length(row.get("request.output_tokens"))
                counters.largest_input = max(
                    counters.largest_input, _token_length(row.get("request.replay.input_length"))
                )
                counters.requests += 1
                _check_headroom(counters, limit)
            continue
        if prefix == "event":
            wrapped = True
            continue
        if direct is None:
            continue
        if prefix.startswith("event."):
            row, key = wrapper, prefix[6:]
        else:
            row, key = direct, prefix
        if key in _DYNAMO_FIELDS:
            row[key] = value
        elif key == "agent_context" and event in {"start_map", "null"}:
            row[key] = event == "start_map"


def _scan_weka(events: Any, counters: _TraceCounters, state: dict[str, Any], limit: int | None) -> None:
    """Count WekaEntry and WekaInnerEntry requests with generated hashes for each prompt."""
    block_size = state["block_size"]
    rows: dict[str, dict[str, Any]] = {}
    for prefix, event, value in events:
        if prefix.endswith(".hash_ids.item"):
            continue
        if prefix == "block_size" and event == "number" and value != block_size:
            raise ValueError("Weka corpus mixes block sizes")
        # Totals, provenance, and the subagent marker itself are not requests.
        if prefix in {"requests.item", "requests.item.requests.item"}:
            if event == "start_map":
                rows[prefix] = {}
            elif event == "end_map":
                row = rows.pop(prefix)
                if row.get("type") not in {"n", "s"}:
                    continue
                length = _token_length(row.get("in"))
                counters.hashes += (length + block_size - 1) // block_size
                counters.outputs += _token_length(row.get("out"))
                counters.largest_input = max(counters.largest_input, length)
                counters.requests += 1
                _check_headroom(counters, limit)
        else:
            base, _, key = prefix.rpartition(".")
            if key in {"type", "in", "out"} and base in rows:
                rows[base][key] = value


def _weka_block_size(path: Path) -> int:
    # serde accepts any field order. Read the first play's block size before
    # counting requests, without retaining preceding input lengths or arrays.
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, multiple_values=True, buf_size=64 * 1024):
            if prefix == "block_size":
                block_size = _token_length(value)
                if block_size > 0:
                    return block_size
                break
    raise ValueError("Weka trace requires a positive block_size")


_TRACE_SCANNERS = {
    "mooncake": _scan_mooncake,
    "mooncake-delta": _scan_mooncake,
    "agentic_mooncake": _scan_mooncake,
    "applied_compute_agentic": _scan_applied_compute,
    "dynamo": _scan_dynamo,
    "weka": _scan_weka,
}


def _trace_files(workload: Mapping[str, Any], format_name: str) -> list[Path] | None:
    """Return inspected files in scan order, or None when a symlink hides the resource identity."""
    files = []
    for raw_path in workload.get("trace_paths") or [workload["trace_path"]]:
        source = Path(raw_path)
        for path in source.rglob("*") if source.is_dir() else (source,):
            if path.is_symlink():
                return None
            if not path.is_file() or (format_name == "weka" and path.suffix.lower() not in {".json", ".jsonl"}):
                continue
            files.append(path)
    return files


def _scan_finite_trace(
    files: list[Path], format_name: str, configured_block_size: int, inspection_budget_bytes: int | None
) -> _TraceCounters:
    counters = _TraceCounters(
        retains_source_hashes=format_name in {"agentic_mooncake", "dynamo", "weka"},
        materializes_inputs=format_name == "mooncake-delta",
    )
    limit = None if inspection_budget_bytes is None else inspection_budget_bytes - WORKER_BASELINE_BYTES
    scan = _TRACE_SCANNERS[format_name]
    state: dict[str, Any] = {
        "omitted_input_block_size": configured_block_size if format_name in {"mooncake", "mooncake-delta"} else 0
    }
    for path in files:
        if format_name == "weka" and "block_size" not in state:
            state["block_size"] = _weka_block_size(path)
        # Streaming holds individual scalars, never request or token arrays.
        with gzip.open(path, "rb") if format_name == "dynamo" and path.suffix == ".gz" else path.open("rb") as stream:
            scan(ijson.parse(stream, multiple_values=True, buf_size=64 * 1024), counters, state, limit)
    return counters


_TRACE_SCAN_CACHE_SIZE = 16
_TRACE_SCAN_CACHE: OrderedDict[tuple[Any, ...], _TraceCounters] = OrderedDict()
_TRACE_SCAN_CACHE_LOCK = threading.Lock()


def _clear_trace_scan_cache() -> None:
    with _TRACE_SCAN_CACHE_LOCK:
        _TRACE_SCAN_CACHE.clear()


def _estimate_finite_trace(
    workload: Mapping[str, Any], *, format_name: str, inspection_budget_bytes: int | None
) -> ResourceEstimate:
    """Bound a finite replay from one cached streaming pass; its peak is observed at runtime."""
    unqualified = lambda reason: ResourceEstimate("trace-unqualified-v1", None, 0, 0, None, reason)
    try:
        files = _trace_files(workload, format_name)
        if files is None:
            return unqualified("trace symlinks have no stable resource identity")
        # Completed counters depend only on file contents, so admission and the
        # per-run guard share one pass. Headroom is compared live on every call.
        key = (
            "trace-scan-v1",
            format_name,
            workload.get("trace_block_size"),
            tuple(
                (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                for path, stat in ((path, path.stat()) for path in files)
            ),
        )
        with _TRACE_SCAN_CACHE_LOCK:
            counters = _TRACE_SCAN_CACHE.get(key)
            if counters is not None:
                _TRACE_SCAN_CACHE.move_to_end(key)
        if counters is None:
            block_size = int(workload.get("trace_block_size") or 512)
            counters = _scan_finite_trace(files, format_name, block_size, inspection_budget_bytes)
            if counters.requests:
                with _TRACE_SCAN_CACHE_LOCK:
                    _TRACE_SCAN_CACHE[key] = counters
                    while len(_TRACE_SCAN_CACHE) > _TRACE_SCAN_CACHE_SIZE:
                        _TRACE_SCAN_CACHE.popitem(last=False)
    except _InspectionStopped as stopped:
        # Remaining rows were not inspected; never cache this partial bound.
        return _observed_peak_estimate(stopped.counters)
    except (OSError, EOFError, ValueError, ijson.JSONError) as exc:
        return unqualified(f"cannot inspect trace metadata: {exc}")
    if not counters.requests:
        return unqualified("trace metadata contains no recognized request token lengths")
    return _observed_peak_estimate(counters)


def _estimate_trace(
    workload: Mapping[str, Any], *, stack: str, inspection_budget_bytes: int | None = None
) -> ResourceEstimate:
    """Stream JSON/JSONL metadata without materializing request or token arrays."""
    format_name = workload.get("trace_format", "mooncake")
    if format_name == "weka" and stack == "engine":
        # Native Weka has a peak model, including snapshot and warmup storage.
        return _estimate_weka_trace(workload, inspection_budget_bytes=inspection_budget_bytes)
    if stack not in {"engine", "dynamo"} or format_name not in _TRACE_FORMATS:
        return ResourceEstimate(
            "trace-unqualified-v1", None, 0, 0, None, "trace format or runner has no qualified allocation model"
        )
    if (
        workload.get("agentic_profile") is not None
        or workload.get("agentic_snapshot") is not None
        or workload.get("agentic_warmup")
    ):
        return _estimate_trace_legacy(
            workload, stack=stack, format_name=format_name, inspection_budget_bytes=inspection_budget_bytes
        )
    return _estimate_finite_trace(workload, format_name=format_name, inspection_budget_bytes=inspection_budget_bytes)


def _estimate_trace_legacy(
    workload: Mapping[str, Any], *, stack: str, format_name: str, inspection_budget_bytes: int | None
) -> ResourceEstimate:
    """Keep the conservative storage estimate for snapshot, warmup and profile replays."""
    unqualified = lambda reason: ResourceEstimate("trace-unqualified-v1", None, 0, 0, None, reason)
    length_keys = {
        "in",
        "out",
        "input_length",
        "output_length",
        "input_tokens",
        "output_tokens",
        "input_prompt_length",
        "assistant_response_length",
        "tool_call_output_length",
        "final_assistant_response_length",
        "max_output_tokens",
        "tool_tokens",
        "system_tokens",
    }
    token_keys = {"input_token_ids", "output_token_ids", "prompt_token_ids"}
    record_keys = length_keys | token_keys
    hash_keys = {"hash_ids", "input_sequence_hashes"}
    total_bytes = tokens = hashes = records = turns = 0
    block_size = int(workload.get("trace_block_size") or 512)

    paths = workload.get("trace_paths") or [workload["trace_path"]]
    try:
        for raw_path in paths:
            source = Path(raw_path)
            files = source.rglob("*") if source.is_dir() else (source,)
            for path in files:
                if path.is_symlink():
                    return unqualified("trace symlinks have no stable resource identity")
                if not path.is_file() or (format_name == "weka" and path.suffix.lower() not in {".json", ".jsonl"}):
                    continue
                storage_estimate = WORKER_BASELINE_BYTES + 128 * (total_bytes + path.stat().st_size)
                if inspection_budget_bytes is not None and storage_estimate > inspection_budget_bytes:
                    return ResourceEstimate(
                        "trace-json-metadata-v1",
                        None,
                        0,
                        0,
                        storage_estimate,
                        "trace storage estimate exceeds live headroom before metadata parsing",
                    )
                # Stream documents and arrays. The parser still holds individual
                # scalar values in memory.
                maps: list[bool] = []
                with (
                    gzip.open(path, "rb") if format_name == "dynamo" and path.suffix == ".gz" else path.open("rb")
                ) as stream:
                    for prefix, event, value in ijson.parse(stream, multiple_values=True, buf_size=64 * 1024):
                        if event == "start_map":
                            maps.append(False)
                        elif event == "end_map":
                            records += int(maps.pop())
                        elif event == "map_key" and value in record_keys:
                            maps[-1] = True
                        else:
                            parts = prefix.rsplit(".", 2)
                            key = parts[-2] if parts[-1] == "item" and len(parts) > 1 else parts[-1]
                            if key in length_keys and event not in {"start_array", "end_array"}:
                                if event != "number" or type(value) is not int or value < 0:
                                    raise ValueError("trace token lengths must be nonnegative integers")
                                tokens += value
                            elif key in token_keys and parts[-1] == "item" and event == "number":
                                tokens += 1
                            elif key in hash_keys and parts[-1] == "item" and event in {"number", "string"}:
                                hashes += 1
                            elif key in {"block_size", "trace_block_size"} and event == "number":
                                block_size = max(block_size, int(value))
                            elif key == "num_turns" and event == "number":
                                turns += int(value) + 1
                    total_bytes += stream.tell()
    except (OSError, EOFError, ValueError, ijson.JSONError) as exc:
        return unqualified(f"cannot inspect trace metadata: {exc}")
    if records == 0:
        return unqualified("trace metadata contains no recognized request token lengths")
    tokens += hashes * block_size
    count = max(records, turns)
    # Delta and tool-turn sources can accumulate every preceding turn's tokens.
    cumulative = count if format_name in {"mooncake-delta", "applied_compute_agentic"} else 1
    lanes = int(workload.get("agentic_lanes") or 1)
    peak = WORKER_BASELINE_BYTES + 128 * total_bytes + lanes * (32 * tokens * cumulative + 65536 * count)
    profile = workload.get("agentic_profile") is not None
    if profile or stack == "dynamo":
        # Native routing retains additional state outside the engine estimate.
        # Profiles also retain retired identities and lifecycle rows, whose
        # count depends on simulated completion times rather than corpus size.
        # Keep the initial-materialization refusal before admitting unknown peaks.
        model = "agentic-profile" if profile else "dynamo"
        if inspection_budget_bytes is not None and peak > inspection_budget_bytes:
            return ResourceEstimate(
                f"{model}-materialization-v1",
                None,
                0,
                0,
                peak,
                "initial trace materialization estimate exceeds live headroom",
            )
        return ResourceEstimate(
            f"{model}-unqualified-v1",
            None,
            0,
            0,
            None,
            (
                "agentic profile retains evidence beyond the initial corpus"
                if profile
                else "Dynamo policy retains native router state beyond the engine trace estimate"
            )
            + "; total memory has no qualified static bound and requires supervised serial execution",
        )
    return ResourceEstimate(
        "trace-json-metadata-v1",
        None,
        0,
        0,
        peak,
        "streamed metadata estimate; runtime trace validation still required",
    )


def estimate_workload(
    workload: Mapping[str, Any],
    *,
    stack: str,
    concurrency: int | None = None,
    inspection_budget_bytes: int | None = None,
) -> ResourceEstimate:
    if workload.get("trace_paths") or workload.get("trace_path"):
        return _estimate_trace(workload, stack=stack, inspection_budget_bytes=inspection_budget_bytes)
    load = concurrency or workload.get("concurrency") or workload.get("request_rate")
    count = workload.get("request_count")
    if count is None:
        if not load or workload.get("unresolved_resource_count"):
            return ResourceEstimate("unresolved-v1", None, 0, 0, None, "candidate-specific request count is unresolved")
        count = max(1, round(float(workload.get("num_request_ratio") or 0) * load))
    count = int(count)
    isl, osl = int(workload.get("isl", 1024)), int(workload.get("osl", 128))
    turns = int(workload.get("turns_per_session", 1))
    if count < 1 or min(isl, osl, turns) < 1:
        raise ValueError("resource estimates require positive request counts and token lengths")
    active = min(count, int(concurrency or workload.get("concurrency") or count))
    if stack == "dynamo" and turns == 1:
        tokens = count * isl * 4
        lower = tokens
        model = "dynamo-eager-u32-v1"
        peak = WORKER_BASELINE_BYTES + 2 * tokens + count * (4096 + 16 * osl)
    elif stack == "engine":
        # A block size of one bounds all supported hash arrays. Sessions may
        # retain cumulative prompts and planned output IDs until they complete.
        tokens = active * isl * turns * 4
        lower = count * turns * osl * 4
        model = "engine-session-metadata-v1"
        peak = WORKER_BASELINE_BYTES + count * turns * (4096 + 32 * (isl + osl) * turns) + tokens
    else:
        return ResourceEstimate("runner-unqualified-v1", count, 0, 0, None, "runner allocation model is unqualified")
    return ResourceEstimate(model, count, tokens, lower, peak)


def _reserved_bytes(peak: int | None, lower: int, *, external: bool) -> int:
    if peak is not None:
        return peak
    # Adapter v1 lower bounds may already contain process memory. Built-in
    # unknown bounds describe arrays separately from the worker baseline.
    return max(WORKER_BASELINE_BYTES, lower) if external else WORKER_BASELINE_BYTES + lower


def build_plan(
    workload: Mapping[str, Any],
    *,
    stack: str,
    policy: ResourceConfig | None = None,
    requested_parallelism: int = 1,
    host: HostResources | None = None,
    factory: Any = None,
    concurrency: int | None = None,
    native_router: bool = False,
) -> dict[str, Any]:
    policy = policy or _POLICY.get() or ResourceConfig()
    host = host or discover_host()
    budget = resolve_budget(policy, host)
    free = max(
        0,
        min(
            budget["memory_limit_bytes"] - budget["coordinator_memory_bytes"],
            host.available_memory_bytes - budget["reserved_host_memory_bytes"] - COORDINATOR_RESERVE_BYTES,
        ),
    )
    estimator = getattr(factory, "estimate_host_resources", None)
    estimate = (
        estimator(workload, concurrency=concurrency)
        if callable(estimator)
        else estimate_workload(workload, stack=stack, concurrency=concurrency, inspection_budget_bytes=free)
    )
    if not isinstance(estimate, ResourceEstimate) or type(estimate.api_version) is not int or estimate.api_version != 1:
        raise ResourceLimitError("runner returned an incompatible host resource estimate")
    if estimate.request_count is not None and (type(estimate.request_count) is not int or estimate.request_count < 1):
        raise ResourceLimitError("runner returned an invalid request count")
    peak = estimate.estimated_peak_bytes
    if any(type(value) is not int or value < 0 for value in (estimate.input_token_bytes, estimate.lower_bound_bytes)):
        raise ResourceLimitError("runner returned an invalid host resource estimate")
    if peak is not None and (type(peak) is not int or peak <= 0 or peak < estimate.lower_bound_bytes):
        raise ResourceLimitError("runner returned an invalid host resource estimate")
    if native_router and ((peak is not None and peak <= free) or estimate.allocation_model == OBSERVED_TRACE_MODEL):
        # Preserve initial-materialization refusals and lower bounds, but neither
        # an engine-only estimate nor an observed trace peak qualifies router state.
        estimate = replace(
            estimate,
            allocation_model="dynamo-unqualified-v1",
            estimated_peak_bytes=None,
            reason="Dynamo policy retains native router state beyond the engine estimate; "
            "total memory requires supervised serial execution",
        )
        peak = None
    workers = min(requested_parallelism, budget["cpu_limit"], free // peak) if peak else 0
    reason = "" if workers else "candidate exceeds the host memory budget"
    if peak is None:
        if free < _reserved_bytes(peak, estimate.lower_bound_bytes, external=callable(estimator)):
            reason = (
                "candidate estimate exceeds available host memory"
                if callable(estimator)
                else "candidate lower bound plus worker baseline exceeds available host memory"
            )
        elif not os.environ.get("_AISIMULATE_SUPERVISED_BUDGET"):
            reason = f"unknown peak requires supervised serial execution: {estimate.reason}"
        else:
            workers = 1  # Unqualified estimates require serial, continuously monitored execution.
            reason = estimate.reason
    return {
        "schema_version": 1,
        "status": "admitted" if workers else "resource_limited",
        "stack": stack,
        "host": asdict(host),
        "budget": budget,
        "estimate": asdict(estimate),
        "requested_parallelism": requested_parallelism,
        "effective_parallelism": workers,
        "reason": reason,
    }


def require_plan(plan: dict[str, Any]) -> None:
    if plan["status"] == "resource_limited":
        estimate = plan["estimate"]
        peak = estimate["estimated_peak_bytes"]
        peak_description = "unknown" if peak is None else f"{peak / GB:.2f} GB"
        raise ResourceLimitError(
            f"resource_limited: {plan['reason']}; allocation model={estimate['allocation_model']}, "
            f"requests={estimate['request_count']}, estimated peak={peak_description}, "
            f"lower bound={estimate['lower_bound_bytes'] / GB:.2f} GB, "
            f"host budget={plan['budget']['memory_limit_bytes'] / GB:.2f} GB. "
            "Choose an explicit smaller workload or an execution host with sufficient resources.",
            plan=plan,
        )


@contextmanager
def resource_policy(policy: ResourceConfig):
    token = _POLICY.set(policy)
    try:
        yield
    finally:
        _POLICY.reset(token)


def _uses_native_router(spec: Any) -> bool:
    return any(
        hook.provider == "engine.router"
        and hook.kind == "placement_policy"
        and hook.config.get("provider") == "dynamo_kv_router"
        for hook in getattr(spec, "runtime_hooks", ())
    )


def guard_replay(spec: Any, *, stack: str, factory: Any = None) -> dict[str, Any]:
    plan = build_plan(
        spec.workload,
        stack=stack,
        concurrency=spec.concurrency,
        factory=factory,
        native_router=_uses_native_router(spec),
    )
    require_plan(plan)
    return plan


@dataclass
class GuardedRunner:
    runner: Any
    stack: str
    policy: ResourceConfig
    factory: Any

    def run(self, spec, *, output_requirements=None):
        with resource_policy(self.policy):
            guard_replay(spec, stack=self.stack, factory=self.factory)
            if output_requirements is None:
                return self.runner.run(spec)
            return self.runner.run(spec, output_requirements=output_requirements)

    def close(self):
        self.runner.close()


def _child_memory_bytes() -> int:
    try:
        children = psutil.Process().children(recursive=True)
        total = 0
        for child in children:
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total
    except psutil.Error as exc:
        raise ResourceLimitError(f"cannot inspect owned execution processes: {exc}") from exc


def _trace_identity(workload: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Identify a finite trace replay whose peak a completed replay can measure."""
    paths = workload.get("trace_paths") or ([workload["trace_path"]] if workload.get("trace_path") else None)
    # Snapshots, warmup and profiles retain state beyond the finite corpus.
    if (
        not paths
        or workload.get("agentic_profile") is not None
        or workload.get("agentic_snapshot") is not None
        or workload.get("agentic_warmup")
    ):
        return None
    return workload.get("trace_format", "mooncake"), workload.get("trace_block_size"), tuple(map(str, paths))


@dataclass(frozen=True)
class GuardedRunnerFactory:
    factory: Any
    stack: str
    policy: ResourceConfig
    # Coordinator-local peaks from completed replays, keyed by trace identity.
    # Workers receive a pickled copy and never consult it.
    observed_peaks: dict[tuple[Any, ...], int] = field(default_factory=dict, compare=False, repr=False)

    def capabilities(self):
        return self.factory.capabilities()

    def create(self, worker_id):
        return GuardedRunner(self.factory.create(worker_id), self.stack, self.policy, self.factory)

    def record_peak(self, spec: Any, peak_bytes: int) -> None:
        """Remember the largest RSS that a completed replay of this trace reached."""
        key = _trace_identity(spec.workload)
        if key is not None and peak_bytes > 0:
            self.observed_peaks[key] = max(peak_bytes, self.observed_peaks.get(key, 0))

    def observed_peak(self, spec: Any) -> int | None:
        key = _trace_identity(spec.workload)
        return None if key is None else self.observed_peaks.get(key)

    def _observed_reservation(self, spec: Any, estimate: Mapping[str, Any]) -> int | None:
        # Candidates differ in engine shape, not in their trace; the margin and
        # live-pressure halving cover a later candidate that needs more.
        observed = self.observed_peak(spec) if estimate["allocation_model"] == OBSERVED_TRACE_MODEL else None
        if observed is None:
            return None
        return max(WORKER_BASELINE_BYTES + estimate["lower_bound_bytes"], math.ceil(observed * OBSERVED_PEAK_MARGIN))

    def admit_wave(self, specs: list[Any]) -> dict[str, Any]:
        """Reserve the sum of a whole wave before creating any of its workers."""
        host = discover_host()
        external = callable(getattr(self.factory, "estimate_host_resources", None))
        plans = []
        reservations = []
        for spec in specs:
            plan = build_plan(
                spec.workload,
                stack=self.stack,
                policy=self.policy,
                host=host,
                factory=self.factory,
                concurrency=spec.concurrency,
                native_router=_uses_native_router(spec),
            )
            plans.append(plan)
            estimate = plan["estimate"]
            observed = self._observed_reservation(spec, estimate)
            if estimate["estimated_peak_bytes"] is None and observed is None and len(specs) > 1:
                raise _SerialAdmissionRequired(
                    "candidate wave requires supervised serial execution",
                    plan={"status": "resource_limited", "workers": len(specs), "candidates": plans},
                )
            require_plan(plan)
            # The observed margin only sizes parallel waves. A lone candidate is
            # admitted like the unobserved probe, so it is never refused for it.
            reservations.append(
                observed
                if observed is not None and len(specs) > 1
                else _reserved_bytes(estimate["estimated_peak_bytes"], estimate["lower_bound_bytes"], external=external)
            )
        budget = resolve_budget(self.policy, host)
        budget["coordinator_memory_bytes"] += _child_memory_bytes()
        available = max(
            0,
            min(
                budget["memory_limit_bytes"] - budget["coordinator_memory_bytes"],
                host.available_memory_bytes - budget["reserved_host_memory_bytes"] - COORDINATOR_RESERVE_BYTES,
            ),
        )
        required = sum(reservations)
        admitted = required <= available and len(specs) <= budget["cpu_limit"]
        plan = {
            "status": "admitted" if admitted else "resource_limited",
            "reserved_bytes": reservations,
            "observed_peak_bytes": [self.observed_peak(spec) for spec in specs],
            "required_bytes": required,
            "available_bytes": available,
            "workers": len(specs),
            "candidates": plans,
        }
        if not admitted:
            raise ResourceLimitError("candidate wave exceeds current host headroom", plan=plan)
        return plan

    def live_pressure(self) -> dict[str, Any] | None:
        host = discover_host()
        budget = resolve_budget(self.policy, host)
        worker_rss = _child_memory_bytes()
        used = budget["coordinator_memory_bytes"] - COORDINATOR_RESERVE_BYTES + worker_rss
        if used >= budget["memory_limit_bytes"] * 0.9 or host.available_memory_bytes < (
            budget["reserved_host_memory_bytes"] + COORDINATOR_RESERVE_BYTES
        ):
            return {
                "status": "resource_limited",
                "reason": "live memory pressure",
                "observed_rss_bytes": used,
                "memory_limit_bytes": budget["memory_limit_bytes"],
            }
        return None
