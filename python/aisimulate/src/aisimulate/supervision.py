# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervise an owned execution process before importing simulator runtimes."""

from __future__ import annotations

import json
import multiprocessing.spawn
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psutil

from .config.common import ResourceConfig
from .resources import ResourceLimitError, discover_host, resolve_budget

_CONTEXT = "_AISIMULATE_SUPERVISED_BUDGET"
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
_POLL_SECONDS = 0.05
_GRACE_SECONDS = 1.0


def in_supervised_process() -> bool:
    return bool(os.environ.get(_CONTEXT))


def runtime_budget() -> dict[str, Any] | None:
    raw = os.environ.get(_CONTEXT)
    return json.loads(raw) if raw else None


def checkpoint(event: str, value: dict[str, Any]) -> None:
    budget = runtime_budget()
    if budget is None:
        return
    record = json.dumps({"event": event, "value": value}, allow_nan=False) + "\n"
    path = Path(budget["events_path"])
    if len(record) > 1024 * 1024 or (path.exists() and path.stat().st_size + len(record) > 64 * 1024 * 1024):
        raise ResourceLimitError("execution evidence exceeds the bounded checkpoint budget")
    with path.open("a", encoding="utf-8") as target:
        target.write(record)


def mark_execution_ready() -> None:
    budget = runtime_budget()
    if budget is not None:
        Path(budget["ready_path"]).write_text("running")


def mark_shutdown() -> None:
    budget = runtime_budget()
    if budget is not None:
        Path(budget["ready_path"]).write_text("shutdown")


def _live(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def terminate_processes(processes: list[psutil.Process], *, grace: float = _GRACE_SECONDS) -> None:
    """Escalate only owned process identities; wait before returning."""
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=grace)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=grace)
    if any(_live(process) for process in alive):
        raise RuntimeError("owned execution processes survived termination; refusing replacement")


def terminate_pool(pool, *, workers=None) -> None:
    """Reap every pool worker before a timeout can create a replacement pool."""
    workers = workers if workers is not None else list((getattr(pool, "_processes", None) or {}).values())
    descendants: list[psutil.Process] = []
    for worker in workers:
        try:
            descendants.extend(psutil.Process(worker.pid).children(recursive=True))
        except psutil.NoSuchProcess:
            pass
    terminate_processes(descendants)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    deadline = time.monotonic() + _GRACE_SECONDS
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    for worker in workers:
        if worker.is_alive():
            worker.kill()
    for worker in workers:
        worker.join(_GRACE_SECONDS)
        if worker.is_alive():
            raise RuntimeError("worker survived termination; refusing replacement")
    pool.shutdown(wait=True, cancel_futures=True)


def close_pool(pool) -> None:
    workers = list((getattr(pool, "_processes", None) or {}).values())
    if not workers:
        pool.shutdown(wait=True, cancel_futures=True)
        return
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + _GRACE_SECONDS
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    if any(worker.is_alive() for worker in workers):
        terminate_pool(pool, workers=workers)


class OwnedTree:
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.root = psutil.Process(process.pid)
        self.known: dict[tuple[int, float], psutil.Process] = {}

    def sample(self) -> int:
        try:
            processes = [self.root, *self.root.children(recursive=True)]
        except psutil.NoSuchProcess:
            processes = []
        for process in processes:
            try:
                self.known[(process.pid, process.create_time())] = process
            except psutil.NoSuchProcess:
                pass
        rss = 0
        for identity, process in list(self.known.items()):
            try:
                if not _live(process):
                    self.known.pop(identity)
                    continue
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                self.known.pop(identity, None)
        return rss

    def close(self) -> None:
        # The new session also catches descendants spawned between samples.
        # Escaped descendants observed earlier are tracked by PID + creation time.
        self.sample()
        if os.name == "posix":
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            terminate_processes([p for p in self.known.values() if p.pid != self.process.pid])
            try:
                self.process.wait(timeout=_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        finally:
            if os.name == "posix":
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=2 * _GRACE_SECONDS)


def run_process(
    command: Sequence[str],
    *,
    policy: ResourceConfig,
    timeout: float | None = None,
    events_output: str | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aisimulate-monitor-") as directory:
        return _run_process(
            command, policy=policy, timeout=timeout, events_output=events_output, directory=Path(directory)
        )


def _run_process(
    command: Sequence[str],
    *,
    policy: ResourceConfig,
    timeout: float | None,
    events_output: str | None,
    directory: Path,
) -> dict[str, Any]:
    """Run an owned process tree under one fixed budget and live host headroom.

    RSS polling is best effort on macOS. Preallocation guards remain mandatory.
    stdout/stderr are inherited rather than accumulated in parent memory.
    """
    host = discover_host()
    budget = resolve_budget(policy, host)
    if budget["memory_limit_bytes"] <= host.process_memory_bytes:
        raise ResourceLimitError("no host memory remains for an execution process")
    env = dict(os.environ)
    env.update(dict.fromkeys(_THREAD_VARIABLES, "1"))
    env["TOKENIZERS_PARALLELISM"] = "false"
    events_path = directory / "events.jsonl"
    ready_path = directory / "ready"
    env[_CONTEXT] = json.dumps(
        {**budget, "supervisor_pid": os.getpid(), "events_path": str(events_path), "ready_path": str(ready_path)}
    )
    started = time.monotonic()
    process = subprocess.Popen(list(command), env=env, start_new_session=os.name == "posix")
    tree = OwnedTree(process)
    peak = 0
    reason = ""
    status = "completed"
    shutdown_started: float | None = None
    try:
        while True:
            current = discover_host()
            rss = current.process_memory_bytes + tree.sample()
            peak = max(peak, rss)
            if rss > budget["memory_limit_bytes"]:
                reason, status = "owned process tree exceeded the memory budget", "resource_limited"
                break
            if current.available_memory_bytes < budget["reserved_host_memory_bytes"]:
                reason, status = "available host memory fell below reserved headroom", "resource_limited"
                break
            if not ready_path.exists() and time.monotonic() - started > policy.initialization_timeout_seconds:
                reason, status = "execution initialization timed out", "timed_out"
                break
            if ready_path.exists() and ready_path.read_text() == "shutdown":
                shutdown_started = shutdown_started or time.monotonic()
                if time.monotonic() - shutdown_started > policy.shutdown_timeout_seconds:
                    reason, status = "execution shutdown timed out", "timed_out"
                    break
            else:
                shutdown_started = None
            if timeout is not None and time.monotonic() - started > timeout:
                reason, status = "execution initialization or runtime timed out", "timed_out"
                break
            if process.poll() is not None:
                break
            time.sleep(_POLL_SECONDS)
    except KeyboardInterrupt:
        reason, status = "execution cancelled", "cancelled"
    except (psutil.Error, OSError, ResourceLimitError) as exc:
        reason, status = f"cannot monitor execution resources: {exc}", "resource_limited"
    finally:
        tree.close()
    if events_output is not None and events_path.exists():
        # Copy only complete records after workers have stopped. A final partial
        # write interrupted by termination must not masquerade as a finished event.
        target_path = Path(events_output)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("x", encoding="utf-8") as target, events_path.open(encoding="utf-8") as source:
            while line := source.readline(1024 * 1024 + 1):
                if not line.endswith("\n") or len(line) > 1024 * 1024:
                    break
                target.write(line)
    code = process.returncode
    if status == "completed" and code:
        status = "resource_limited" if code == 3 else "failed"
        reason = f"execution process exited with status {code}"
    return {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "exit_code": code,
        "budget": budget,
        "requested_resources": policy.model_dump(mode="json"),
        "peak_observed_rss_bytes": peak,
        "wall_seconds": time.monotonic() - started,
        "thread_limit_per_runtime": 1,
        "termination_complete": True,
    }


def _save_runtime_report(output: str, report: dict[str, Any], *, overwrite: bool) -> None:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "resource-runtime.json"
    if overwrite and (path.is_file() or path.is_symlink()):
        path.unlink()
    # Exclusive creation prevents a diagnostic from overwriting unrelated evidence.
    with path.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    from .cli_args import _apply_overrides, _load_mapping, build_parser
    from .output import prepare_output_directory

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    raw = None
    try:
        raw = _load_mapping(args.config)
        _apply_overrides(raw, args.overrides, command=args.command)
        policy = ResourceConfig.model_validate(raw.get("execution", {}).get("resources", {}))
    except (OSError, ValueError, AttributeError):
        # The child retains stack/schema error ordering under conservative limits.
        policy = ResourceConfig()
    if raw is not None:
        # Validate the lightweight core envelope before --overwrite removes outputs.
        # Native imports, adapter preparation, and replay stay inside supervision.
        try:
            from .config.cli import CorePredictionConfig, CoreRecommendationConfig
            from .config.common import split_config_sections

            core_raw, adapter_raw = split_config_sections(raw, command=args.command)
            config_type = CorePredictionConfig if args.command == "predict" else CoreRecommendationConfig
            config = config_type.model_validate(core_raw)
            if config.engine.workers.encoder is not None:
                if args.command == "predict" and (
                    args.stack != "engine" or args.online or args.capture_per_request or adapter_raw
                ):
                    raise ValueError(
                        "analytical EPD requires offline --stack engine without adapters or per-request capture"
                    )
                if args.command == "recommend" and (args.stack != "engine" or adapter_raw):
                    raise ValueError("analytical EPD requires --stack engine without adapters")
        except (ValueError, TypeError, AttributeError) as exc:
            parser.error(f"{args.config}: {exc}")
    try:
        output = prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    event_output = output / "execution-events.jsonl"
    try:
        report = run_process(
            [sys.executable, "-m", "aisimulate.resource_worker", "cli", *arguments],
            policy=policy,
            events_output=str(event_output),
        )
    except ResourceLimitError as exc:
        report = exc.plan
    if report["status"] != "completed":
        sys.stderr.write(f"aisimulate: {report.get('reason', report['status'])}\n")
    try:
        _save_runtime_report(args.output_dir, report, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"could not save resource runtime report: {exc}\n")
    if report["status"] == "resource_limited":
        return 3
    if report["status"] == "cancelled":
        return 130
    if report["status"] == "timed_out":
        return 124
    return int(report.get("exit_code") or 0)


def supervised_recommendation(config, kwargs):
    from .sweeper.result import SweepResult

    with tempfile.TemporaryDirectory(prefix="aisimulate-execution-") as directory:
        root = Path(directory)
        payload = root / "input.pickle"
        with payload.open("wb") as target:
            try:
                preparation = multiprocessing.spawn.get_preparation_data("aisimulate-execution")
                preparation.pop("authkey", None)
                main_path = preparation.get("init_main_from_path")
                if main_path and not Path(main_path).is_file():
                    preparation.pop("init_main_from_path")
                pickle.dump(preparation, target)
                pickle.dump((config.model_dump(mode="json"), kwargs), target)
            except (TypeError, AttributeError, pickle.PicklingError) as exc:
                raise ValueError("supervised recommendation requires pickleable factories and providers") from exc
        report = run_process(
            [sys.executable, "-m", "aisimulate.resource_worker", "recommend", str(root)],
            policy=config.execution.resources,
            events_output=str(root / "events.jsonl"),
        )
        result = root / "result.json"
        error_path = root / "error.json"
        if report["status"] != "completed":
            message = report["reason"]
            if error_path.exists() and error_path.stat().st_size < 1024 * 1024:
                error = json.loads(error_path.read_text())
                message = error["message"]
            if report["status"] == "resource_limited":
                events = root / "events.jsonl"
                report["partial_events"] = []
                if events.exists():
                    with events.open() as source:
                        retained_bytes = 0
                        for line in source:
                            retained_bytes += len(line)
                            if retained_bytes > 1024 * 1024:
                                report["partial_evidence_truncated"] = True
                                break
                            report["partial_events"].append(json.loads(line))
                raise ResourceLimitError(message, plan=report)
            if report["status"] == "cancelled":
                raise KeyboardInterrupt
            raise RuntimeError(message)
        # Bound parent-side JSON expansion before loading any result bytes.
        host = discover_host()
        safe_bytes = max(
            0,
            min(
                host.available_memory_bytes - report["budget"]["reserved_host_memory_bytes"],
                report["budget"]["memory_limit_bytes"] - host.process_memory_bytes,
            ),
        )
        if result.stat().st_size * 64 > safe_bytes:
            raise ResourceLimitError(
                "recommendation result exceeds the bounded parent transfer budget",
                plan={
                    **report,
                    "status": "resource_limited",
                    "reason": "result transfer budget exceeded",
                },
            )
        parsed = SweepResult.model_validate_json(result.read_text())
        parsed._execution_resources = report
        return parsed
