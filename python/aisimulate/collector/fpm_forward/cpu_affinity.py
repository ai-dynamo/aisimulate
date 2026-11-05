# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate initialization-time CPU evidence separately from memory and timings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .runtime_instrumentation import read_json


def _integers(value: Any, label: str, *, minimum: int, empty: bool = False) -> list[int]:
    if (
        not isinstance(value, list)
        or (not value and not empty)
        or any(type(item) is not int or item < minimum for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{label} must contain sorted, unique integers >= {minimum}")
    return value


def _snapshot(record: dict[str, Any]) -> None:
    if not isinstance(record.get("hostname"), str) or not record["hostname"].strip():
        raise ValueError("CPU observation has no hostname")
    for name in ("pid", "observer_thread_id"):
        if type(record.get(name)) is not int or record[name] < 1:
            raise ValueError(f"CPU observation has invalid {name}")
    status = record.get("status")
    if status not in {"observed", "partial", "unavailable"}:
        raise ValueError("CPU observation has invalid status")
    main = _integers(
        record.get("main_thread_allowed_cpus"), "main thread CPU mask", minimum=0, empty=status == "unavailable"
    )
    before = _integers(record.get("thread_ids_before"), "thread IDs before observation", minimum=1, empty=True)
    after = _integers(record.get("thread_ids_after"), "thread IDs after observation", minimum=1, empty=True)
    threads = record.get("threads")
    if not isinstance(threads, list):
        raise ValueError("CPU observation lacks thread masks")
    seen = {}
    for thread in threads:
        if not isinstance(thread, dict):
            raise ValueError("CPU thread observation must be an object")
        tid = thread.get("tid")
        if type(tid) is not int or tid < 1 or tid in seen or tid not in before:
            raise ValueError("CPU thread observation has an invalid, duplicate or unlisted thread ID")
        seen[tid] = _integers(thread.get("allowed_cpus"), "thread CPU mask", minimum=0)
    errors, thread_errors = record.get("errors"), record.get("thread_errors")
    if not isinstance(errors, list) or any(not isinstance(error, str) or not error for error in errors):
        raise ValueError("CPU observation errors must be a list of messages")
    if not isinstance(thread_errors, list) or any(
        not isinstance(item, dict)
        or type(item.get("tid")) is not int
        or item["tid"] not in before
        or not isinstance(item.get("error"), str)
        or not item["error"]
        for item in thread_errors
    ):
        raise ValueError("CPU thread errors must identify an observed thread and error")
    if status == "observed" and (
        before != after
        or set(seen) != set(before)
        or seen.get(record["pid"]) != main
        or record["observer_thread_id"] not in seen
        or errors
        or thread_errors
    ):
        raise ValueError("CPU observation claims complete thread evidence with missing or inconsistent masks")


def _can_schedule_distinct(masks: list[list[int]]) -> bool:
    """Check main-thread capacity without requiring disjoint affinity masks."""
    assigned: dict[int, int] = {}

    def assign(index: int, seen: set[int]) -> bool:
        for cpu in masks[index]:
            if cpu in seen:
                continue
            seen.add(cpu)
            previous = assigned.get(cpu)
            if previous is None or assign(previous, seen):
                assigned[cpu] = index
                return True
        return False

    return all(assign(index, set()) for index in range(len(masks)))


def inspect_cpu_affinity(
    cell, raw_root: Path, collection, *, plan, expected_nodes: int | None = None
) -> dict[str, Any]:
    """Inspect only this cell's attempt; old artifacts remain CPU-unqualified.

    Main-thread masks and thread snapshots describe specific initialization
    hooks. They do not prove CPU utilization, exclusive cores, NUMA locality or
    affinity throughout later timed forwards. Shared multi-core pools are valid.
    """
    requested = getattr(plan.options, "slurm_cpus_per_task", None)
    binding = getattr(plan.options, "slurm_cpu_bind", None)
    required = plan.options.executor == "slurm" and (requested is not None or binding is not None)
    missing, failures, observations = [], [], []
    if required and (type(requested) is not int or requested < 1 or binding not in {"cores", "none"}):
        failures.append("frozen CPU policy requires positive cpus_per_task and cpu_bind=cores|none")
    elif not required:
        missing.append("frozen CPU policy is unavailable; historical observations do not qualify CPU allocation")
    if required and expected_nodes is None:
        missing.append("generated launch node count is unavailable")
    elif expected_nodes is not None and (type(expected_nodes) is not int or expected_nodes < 1):
        failures.append("generated launch node count must be a positive integer")
        expected_nodes = None
    expected_workers = {
        (dp, tp, pp)
        for dp in range(cell.topology.dp)
        for tp in range(cell.topology.tp)
        for pp in range(cell.topology.pp)
    }
    workers, schedulers, launchers = {}, {}, {}
    processes = set()
    measurements = {
        "launcher": "before_engine_start",
        "worker": "after_warmup",
        "scheduler": "scheduler_initialized",
    }
    for path in sorted(raw_root.glob("**/fpm-cpu-*.json")):
        try:
            raw = path.read_bytes()
            record = read_json(path)
            source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
            item = {**record, "source": source}
            observations.append(item)
            if (
                record.get("schema_name") != "aisimulate_fpm_cpu_affinity"
                or type(record.get("schema_version")) is not int
                or record["schema_version"] != 1
            ):
                raise ValueError("CPU observation has an unsupported schema")
            provenance = record.get("collector_provenance")
            if not isinstance(provenance, dict) or any(
                provenance.get(key) != value
                for key, value in {
                    "plan_sha256": plan.sha256,
                    "cell_id": cell.cell_id,
                    "attempt_id": collection.collector_attempt_id,
                }.items()
            ):
                raise ValueError("CPU evidence has a different plan, cell or attempt")
            kind = record.get("kind")
            if kind not in measurements or record.get("measurement") != measurements[kind]:
                raise ValueError("CPU observation has an unsupported role or lifecycle hook")
            _snapshot(record)
            process = (kind, record["hostname"], record["pid"])
            if process in processes:
                raise ValueError("CPU observation has a duplicate process identity for its role")
            processes.add(process)
            rank = tuple(record.get(name) for name in ("dp_rank", "tp_rank", "pp_rank"))
            if kind == "worker":
                if any(type(value) is not int for value in rank) or rank not in expected_workers or rank in workers:
                    raise ValueError("CPU worker rank is unexpected or duplicate")
                workers[rank] = item
                suffix = f"-dp{rank[0]}-tp{rank[1]}-pp{rank[2]}"
            elif kind == "scheduler":
                dp, tp, pp = rank
                if (
                    type(dp) is not int
                    or not 0 <= dp < cell.topology.dp
                    or tp is not None
                    or pp is not None
                    or dp in schedulers
                ):
                    raise ValueError("CPU scheduler rank is unexpected or duplicate")
                schedulers[dp] = item
                suffix = f"-dp{dp}"
            else:
                if any(value is not None for value in rank) or record["hostname"] in launchers:
                    raise ValueError("CPU launcher has rank coordinates or a duplicate hostname")
                launchers[record["hostname"]] = item
                suffix = ""
            if path.name != f"fpm-cpu-{kind}{suffix}.json":
                raise ValueError("CPU observation filename disagrees with its role/rank")
            if record["status"] != "observed":
                missing.append(f"{kind} {rank} on {record['hostname']}: {record['status']} CPU affinity snapshot")
        except (OSError, TypeError, ValueError, KeyError) as error:
            failures.append(f"{path.name}: {error}")
    if workers.keys() != expected_workers:
        missing.append(f"worker CPU observations missing for ranks {sorted(expected_workers - workers.keys())}")
    if schedulers.keys() != set(range(cell.topology.dp)):
        missing.append(
            "scheduler CPU observations missing for DP ranks "
            f"{sorted(set(range(cell.topology.dp)) - schedulers.keys())}"
        )
    for dp, scheduler in schedulers.items():
        worker = workers.get((dp, 0, 0))
        if worker is not None and worker["hostname"] != scheduler["hostname"]:
            failures.append(f"DP{dp} scheduler host differs from its TP0/PP0 worker")
    observed_hosts = {item["hostname"] for item in [*workers.values(), *schedulers.values()]}
    if expected_nodes is not None and len(workers) == len(expected_workers):
        worker_hosts = {item["hostname"] for item in workers.values()}
        if len(worker_hosts) != expected_nodes or (
            len(launchers) == len(worker_hosts) and launchers.keys() != worker_hosts
        ):
            failures.append("observed worker/launcher hosts differ from the generated launch node count")
        if len(launchers) > expected_nodes:
            failures.append("launcher observations exceed the generated node count")
    nodes = []
    for hostname in sorted(observed_hosts | launchers.keys()):
        local_workers = [item for item in workers.values() if item["hostname"] == hostname]
        local_schedulers = [item for item in schedulers.values() if item["hostname"] == hostname]
        launcher = launchers.get(hostname)
        node = {
            "hostname": hostname,
            "worker_ranks": [
                {name: item[name] for name in ("dp_rank", "tp_rank", "pp_rank")} for item in local_workers
            ],
            "scheduler_dp_ranks": [item["dp_rank"] for item in local_schedulers],
            "launcher_allowed_cpus": launcher["main_thread_allowed_cpus"] if launcher else None,
        }
        nodes.append(node)
        if launcher is None:
            missing.append(f"launcher CPU pool observation missing for {hostname}")
            # A rank on a host that has no launcher while all expected pools are
            # present is also caught by the empty worker pool check below.
            continue
        if not local_workers and len(workers) == len(expected_workers):
            failures.append(f"launcher {hostname} has no worker CPU observations on its host")
        pool = set(launcher["main_thread_allowed_cpus"])
        if required:
            if (
                type(launcher.get("requested_cpus_per_task")) is not int
                or launcher["requested_cpus_per_task"] != requested
                or launcher.get("cpu_bind") != binding
            ):
                failures.append(f"launcher {hostname} CPU policy differs from the frozen requested policy")
            if pool and type(requested) is int and len(pool) < requested:
                failures.append(
                    f"launcher {hostname} allows fewer logical CPUs than requested cpus_per_task={requested}"
                )
            if type(requested) is int and requested < len(local_schedulers):
                failures.append(f"launcher {hostname} requested CPU count is below its local DP scheduler count")
        local_gpus = launcher.get("local_gpu_count")
        if type(local_gpus) is not int or local_gpus < 1:
            failures.append(f"launcher {hostname} has no valid local GPU count")
        elif local_gpus != len(local_workers) and len(workers) == len(expected_workers):
            failures.append(f"launcher {hostname} local GPU count differs from observed worker rank count")
        if expected_nodes is not None and local_gpus != len(expected_workers) / expected_nodes:
            failures.append(f"launcher {hostname} local GPU count differs from the generated launch geometry")
        for item in [launcher, *local_workers, *local_schedulers]:
            if pool and not set(item["main_thread_allowed_cpus"]) <= pool:
                failures.append(f"{item['kind']} on {hostname} has a main-thread mask outside the launcher CPU pool")
            for thread in item["threads"]:
                if pool and not set(thread["allowed_cpus"]) <= pool:
                    failures.append(
                        f"{item['kind']} on {hostname} has thread {thread['tid']} outside the launcher CPU pool"
                    )
        masks = [item["main_thread_allowed_cpus"] for item in local_schedulers]
        if masks and all(masks):
            node["scheduler_main_threads_have_distinct_cpu_capacity"] = _can_schedule_distinct(masks)
            if not node["scheduler_main_threads_have_distinct_cpu_capacity"]:
                failures.append(f"DP scheduler main threads on {hostname} cannot run on distinct logical CPUs")
    if not launchers:
        missing.append("launcher CPU pool observations are missing")
    return {
        "status": "failed" if failures else "incomplete" if missing else "qualified",
        "policy_required": required,
        "policy": {"executor": plan.options.executor, "cpus_per_task": requested, "cpu_bind": binding},
        "expected_nodes": expected_nodes,
        "missing_evidence": sorted(set(missing)),
        "failures": sorted(set(failures)),
        "observations": observations,
        "nodes": nodes,
        "scope": "Linux logical CPU masks at launcher startup, worker warm-up completion and scheduler initialization; "
        "per-thread snapshots, not utilization, exclusive-core, NUMA or timed-forward affinity qualification",
    }
