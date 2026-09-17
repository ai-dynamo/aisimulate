# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host resource accounting before expensive simulation materialization.

Estimates describe host allocations, never simulated model weights or GPU KV
capacity. They are conservative planning estimates, not promises about peak RSS.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
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
            "trace_block_size": source.get("block_size", 512),
            "agentic_lanes": load.get("agentic_lanes", 1),
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


def _estimate_trace(workload: Mapping[str, Any], *, stack: str) -> ResourceEstimate:
    """Stream JSON/JSONL metadata without materializing request or token arrays."""
    unqualified = lambda reason: ResourceEstimate("trace-unqualified-v1", None, 0, 0, None, reason)
    format_name = workload.get("trace_format", "mooncake")
    if stack not in {"engine", "dynamo"} or format_name not in {
        "mooncake",
        "mooncake-delta",
        "agentic_mooncake",
        "applied_compute_agentic",
        "dynamo",
        "weka",
    }:
        return unqualified("trace format or runner has no qualified allocation model")
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
                # Parse events rather than loading a document or an entire JSONL
                # record. Metadata memory depends on nesting, not trace length.
                maps: list[bool] = []
                with path.open("rb") as stream:
                    for prefix, event, value in ijson.parse(stream, multiple_values=True, buf_size=64 * 1024):
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
    except (OSError, ValueError, ijson.JSONError) as exc:
        return unqualified(f"cannot inspect trace metadata: {exc}")
    if records == 0:
        return unqualified("trace metadata contains no recognized request token lengths")
    tokens += hashes * block_size
    count = max(records, turns)
    # Delta and tool-turn sources can accumulate every preceding turn's tokens.
    cumulative = count if format_name in {"mooncake-delta", "applied_compute_agentic"} else 1
    lanes = int(workload.get("agentic_lanes") or 1)
    peak = WORKER_BASELINE_BYTES + 128 * total_bytes + lanes * (32 * tokens * cumulative + 65536 * count)
    return ResourceEstimate(
        "trace-json-metadata-v1",
        None,
        0,
        0,
        peak,
        "streamed metadata estimate; runtime trace validation still required",
    )


def estimate_workload(workload: Mapping[str, Any], *, stack: str, concurrency: int | None = None) -> ResourceEstimate:
    if workload.get("trace_paths") or workload.get("trace_path"):
        return _estimate_trace(workload, stack=stack)
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
    estimator = getattr(factory, "estimate_host_resources", None)
    estimate = (
        estimator(workload, concurrency=concurrency)
        if callable(estimator)
        else estimate_workload(workload, stack=stack, concurrency=concurrency)
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
    free = max(
        0,
        min(
            budget["memory_limit_bytes"] - budget["coordinator_memory_bytes"],
            host.available_memory_bytes - budget["reserved_host_memory_bytes"] - COORDINATOR_RESERVE_BYTES,
        ),
    )
    workers = min(requested_parallelism, budget["cpu_limit"], free // peak) if peak else 0
    if (
        peak is None
        and os.environ.get("_AISIMULATE_SUPERVISED_BUDGET")
        and free >= max(WORKER_BASELINE_BYTES, estimate.lower_bound_bytes)
    ):
        workers = 1  # Unqualified estimates require serial, continuously monitored execution.
    return {
        "schema_version": 1,
        "status": "admitted" if workers else "resource_limited",
        "stack": stack,
        "host": asdict(host),
        "budget": budget,
        "estimate": asdict(estimate),
        "requested_parallelism": requested_parallelism,
        "effective_parallelism": workers,
        "reason": estimate.reason if peak is None else ("" if workers else "candidate exceeds the host memory budget"),
    }


def require_plan(plan: dict[str, Any]) -> None:
    if plan["status"] == "resource_limited":
        estimate = plan["estimate"]
        raise ResourceLimitError(
            f"resource_limited: {plan['reason']}; allocation model={estimate['allocation_model']}, "
            f"requests={estimate['request_count']}, lower bound={estimate['lower_bound_bytes'] / GB:.2f} GB, "
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
        plans = [
            build_plan(
                spec.workload,
                stack=self.stack,
                policy=self.policy,
                host=host,
                factory=self.factory,
                concurrency=spec.concurrency,
            )
            for spec in specs
        ]
        for plan in plans:
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
        peaks = [plan["estimate"]["estimated_peak_bytes"] for plan in plans]
        required = sum(peak or WORKER_BASELINE_BYTES for peak in peaks)
        admitted = (
            required <= available
            and len(specs) <= budget["cpu_limit"]
            and (all(peak is not None for peak in peaks) or len(specs) == 1)
        )
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
