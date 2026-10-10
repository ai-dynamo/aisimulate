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
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
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
        # Immutable snapshot context owns a graph, u32 hash ranks and outputs.
        replay += 12 * hashes + 4 * outputs + 2048 * records
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


def _finite_agentic_estimate(lower: int) -> ResourceEstimate:
    return ResourceEstimate(
        "agentic-trace-unqualified-v1",
        None,
        0,
        lower,
        None,
        "finite agentic array storage and largest prompt provide only a lower bound; "
        "the full-run peak is unknown and requires supervised serial execution",
    )


def _token_length(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("trace token lengths must be nonnegative integers")
    return value


@dataclass
class _AgenticTraceMetadata:
    """Keep scalar counters only; never retain request or token arrays."""

    format: str
    block_size: int = 0
    hashes: int = 0
    authored_outputs: int = 0
    outputs: int = 0
    largest_input: int = 0
    requests: int = 0
    agentic: bool | None = None
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    wrapped: bool = False

    @property
    def lower_bound(self) -> int:
        return max(
            8 * self.hashes + 4 * self.authored_outputs, 4 * (self.hashes + self.outputs), 4 * self.largest_input
        )

    def consume(self, prefix: str, event: str, value: Any) -> bool:
        """Return whether the running array bound changed."""
        if self.format == "agentic_mooncake":
            if prefix in {"input_length", "input_tokens", "output_length", "output_tokens"}:
                length = _token_length(value)
                if prefix in {"input_length", "input_tokens"}:
                    self.largest_input = max(self.largest_input, length)
                else:
                    self.outputs += length
                self.requests += 1
            elif prefix == "hash_ids.item" and event == "number":
                self.hashes += 1
            elif prefix == "output_token_ids.item" and event == "number":
                self.authored_outputs += 1
            else:
                return False
            return True
        if self.format == "weka":
            if prefix == "block_size" and event == "number" and value != self.block_size:
                raise ValueError("Weka corpus mixes block sizes")
            # Only request objects used by WekaEntry and WekaInnerEntry count.
            # Totals, provenance, and the subagent marker itself are not requests.
            if prefix in {"requests.item", "requests.item.requests.item"}:
                if event == "start_map":
                    self.rows[prefix] = {}
                elif event == "end_map":
                    row = self.rows.pop(prefix)
                    if row.get("type") not in {"n", "s"}:
                        return False
                    length = _token_length(row.get("in"))
                    self.hashes += (length + self.block_size - 1) // self.block_size
                    self.outputs += _token_length(row.get("out"))
                    self.largest_input = max(self.largest_input, length)
                    self.requests += 1
                    return True
            else:
                base, _, key = prefix.rpartition(".")
                if key in {"type", "in", "out"} and base in self.rows:
                    self.rows[base][key] = value
            return False

        # Dynamo JSONL accepts either a direct event or an outer event wrapper.
        if prefix == "" and event == "start_map":
            self.rows = {"": {}, "event": {}}
            self.wrapped = False
        if prefix == "event":
            self.wrapped = True
        base = "event" if prefix.startswith("event.") else ""
        key = prefix.removeprefix("event.") if base else prefix
        row = self.rows.get(base)
        if row is not None:
            if key in {"event_type", "request.output_tokens", "request.replay.input_length"}:
                row[key] = value
            elif key == "agent_context" and event in {"start_map", "null"}:
                row[key] = event == "start_map"
            elif key == "request.replay.input_sequence_hashes.item" and event == "number":
                row["hashes"] = row.get("hashes", 0) + 1
        if prefix != "" or event != "end_map":
            return False
        row = self.rows["event" if self.wrapped else ""]
        if row.get("event_type") != "request_end":
            return False
        contextual = row.get("agent_context", False)
        if self.agentic is not None and contextual != self.agentic:
            raise ResourceLimitError("Dynamo request trace cannot mix requests with and without agent_context")
        self.agentic = contextual
        self.hashes += row.get("hashes", 0)
        self.outputs += _token_length(row.get("request.output_tokens"))
        self.largest_input = max(self.largest_input, _token_length(row.get("request.replay.input_length")))
        self.requests += 1
        return True


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


def _estimate_trace(
    workload: Mapping[str, Any], *, stack: str, inspection_budget_bytes: int | None = None
) -> ResourceEstimate:
    """Stream JSON/JSONL metadata without materializing request or token arrays."""
    unqualified = lambda reason: ResourceEstimate("trace-unqualified-v1", None, 0, 0, None, reason)
    format_name = workload.get("trace_format", "mooncake")
    if format_name == "weka" and stack == "engine":
        # Native Weka has a peak model, including snapshot and warmup storage.
        return _estimate_weka_trace(workload, inspection_budget_bytes=inspection_budget_bytes)
    if stack not in {"engine", "dynamo"} or format_name not in {
        "mooncake",
        "mooncake-delta",
        "agentic_mooncake",
        "applied_compute_agentic",
        "dynamo",
        "weka",
    }:
        return unqualified("trace format or runner has no qualified allocation model")
    finite_agentic = (
        format_name in {"agentic_mooncake", "weka", "dynamo"}
        and workload.get("agentic_profile") is None
        and workload.get("agentic_snapshot") is None
        and not workload.get("agentic_warmup")
    )
    metadata = (
        _AgenticTraceMetadata(format_name, agentic=None if format_name == "dynamo" else True)
        if finite_agentic
        else None
    )
    legacy_error = None
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
                if metadata and format_name == "weka" and not metadata.block_size:
                    metadata.block_size = _weka_block_size(path)
                if not metadata or metadata.agentic is False:
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
                # Stream documents and arrays. Finite AgentX file size does not
                # bound resident data; use array counts and runtime supervision.
                # The parser still holds individual scalar values in memory.
                maps: list[bool] = []
                with (
                    gzip.open(path, "rb") if format_name == "dynamo" and path.suffix == ".gz" else path.open("rb")
                ) as stream:
                    for prefix, event, value in ijson.parse(stream, multiple_values=True, buf_size=64 * 1024):
                        if metadata:
                            previous_mode = metadata.agentic
                            changed = metadata.consume(prefix, event, value)
                            if (
                                changed
                                and metadata.agentic
                                and inspection_budget_bytes is not None
                                and WORKER_BASELINE_BYTES + metadata.lower_bound > inspection_budget_bytes
                            ):
                                return _finite_agentic_estimate(metadata.lower_bound)
                            if format_name != "dynamo" or metadata.agentic:
                                continue
                            # Dynamo mode is data-driven. Preserve legacy counters
                            # in this same pass until request events select a mode.
                            if (
                                previous_mode is None
                                and metadata.agentic is False
                                and inspection_budget_bytes is not None
                            ):
                                storage_estimate = WORKER_BASELINE_BYTES + 128 * (total_bytes + path.stat().st_size)
                                if storage_estimate > inspection_budget_bytes:
                                    return ResourceEstimate(
                                        "trace-json-metadata-v1",
                                        None,
                                        0,
                                        0,
                                        storage_estimate,
                                        "trace storage estimate exceeds live headroom during Dynamo mode inspection",
                                    )
                        if event == "start_map":
                            maps.append(False)
                        elif event == "end_map":
                            records += int(maps.pop())
                        elif event == "map_key" and value in length_keys | token_keys:
                            maps[-1] = True
                        else:
                            parts = prefix.rsplit(".", 2)
                            key = parts[-2] if parts[-1] == "item" and len(parts) > 1 else parts[-1]
                            if key in length_keys and event not in {"start_array", "end_array"}:
                                if event != "number" or type(value) is not int or value < 0:
                                    if metadata:
                                        legacy_error = "trace token lengths must be nonnegative integers"
                                        continue
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
    if metadata and metadata.agentic and metadata.requests:
        return _finite_agentic_estimate(metadata.lower_bound)
    if legacy_error:
        return unqualified(f"cannot inspect trace metadata: {legacy_error}")
    if records == 0:
        return unqualified("trace metadata contains no recognized request token lengths")
    tokens += hashes * block_size
    count = max(records, turns)
    # Delta and tool-turn sources can accumulate every preceding turn's tokens.
    cumulative = count if format_name in {"mooncake-delta", "applied_compute_agentic"} else 1
    lanes = int(workload.get("agentic_lanes") or 1)
    peak = WORKER_BASELINE_BYTES + 128 * total_bytes + lanes * (32 * tokens * cumulative + 65536 * count)
    if workload.get("agentic_profile") is not None:
        # Retired plays release their large payloads, but retain identities and
        # lifecycle rows. Their count depends on simulated completion times, so
        # neither the finite corpus nor duration alone bounds the full profile.
        # Keep the initial-materialization refusal before admitting unknown peaks.
        if inspection_budget_bytes is not None and peak > inspection_budget_bytes:
            return ResourceEstimate(
                "agentic-profile-materialization-v1",
                None,
                0,
                0,
                peak,
                "initial profile trace materialization estimate exceeds live headroom",
            )
        return ResourceEstimate(
            "agentic-profile-unqualified-v1",
            None,
            0,
            0,
            None,
            "agentic profile retains evidence beyond the initial corpus; total memory has no qualified static bound "
            "and requires supervised serial execution",
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


def guard_replay(spec: Any, *, stack: str, factory: Any = None) -> dict[str, Any]:
    plan = build_plan(spec.workload, stack=stack, concurrency=spec.concurrency, factory=factory)
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


@dataclass(frozen=True)
class GuardedRunnerFactory:
    factory: Any
    stack: str
    policy: ResourceConfig

    def capabilities(self):
        return self.factory.capabilities()

    def create(self, worker_id):
        return GuardedRunner(self.factory.create(worker_id), self.stack, self.policy, self.factory)

    def admit_wave(self, specs: list[Any]) -> dict[str, Any]:
        """Reserve the sum of a whole wave before creating any of its workers."""
        host = discover_host()
        plans = []
        for spec in specs:
            plan = build_plan(
                spec.workload,
                stack=self.stack,
                policy=self.policy,
                host=host,
                factory=self.factory,
                concurrency=spec.concurrency,
            )
            plans.append(plan)
            if plan["estimate"]["estimated_peak_bytes"] is None and len(specs) > 1:
                raise _SerialAdmissionRequired(
                    "candidate wave requires supervised serial execution",
                    plan={"status": "resource_limited", "workers": len(specs), "candidates": plans},
                )
            require_plan(plan)
        budget = resolve_budget(self.policy, host)
        budget["coordinator_memory_bytes"] += _child_memory_bytes()
        available = max(
            0,
            min(
                budget["memory_limit_bytes"] - budget["coordinator_memory_bytes"],
                host.available_memory_bytes - budget["reserved_host_memory_bytes"] - COORDINATOR_RESERVE_BYTES,
            ),
        )
        external = callable(getattr(self.factory, "estimate_host_resources", None))
        required = sum(
            _reserved_bytes(
                candidate["estimate"]["estimated_peak_bytes"],
                candidate["estimate"]["lower_bound_bytes"],
                external=external,
            )
            for candidate in plans
        )
        admitted = required <= available and len(specs) <= budget["cpu_limit"]
        plan = {
            "status": "admitted" if admitted else "resource_limited",
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
