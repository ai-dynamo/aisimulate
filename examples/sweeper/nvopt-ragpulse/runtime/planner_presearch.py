# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel execution of independent native Planner forecast evaluations.

All traffic aggregation, warmup sizing, predictor construction, forecasting and
loss computations are delegated to the installed Dynamo implementation. The
compatibility target is Dynamo c7241c2f153efba10b57c38c2144b70d82194a4d,
components/src/dynamo/planner/simulation/load_predictor.py. This module does not
replay requests, select engine configurations, or alter native predictor knobs.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import traceback
from typing import Any

_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "STAN_NUM_THREADS",
)
_THREAD_LIMITER = None


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict[str, Any]):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _worker_init():
    global _THREAD_LIMITER
    for key in _THREAD_ENV:
        os.environ[key] = "1"
    # Spawn may import NumPy through the caller's __main__ before this runs.
    # Limit already-loaded BLAS pools as well as future subprocesses/imports.
    from threadpoolctl import threadpool_limits
    _THREAD_LIMITER = threadpool_limits(limits=1)


def _evaluate(task):
    from dynamo.planner.simulation import load_predictor as native

    interval, index, entry, windows, warmup, destination = task
    started = time.perf_counter()
    record = {
        "status": "running", "interval_seconds": interval,
        "candidate_index": index, "candidate": entry,
        "label": native._entry_label(entry, index), "pid": os.getpid(),
        "started_utc": _utc(), "window_count": len(windows),
        "common_warmup": warmup, "engine_replays": 0,
    }
    path = Path(destination) if destination else None
    if path is not None:
        _write(path, record)
    try:
        loss = native.evaluate_preset(
            windows, native._internal_preset(entry), interval, warmup
        )
        record.update(status="completed", loss=loss if math.isfinite(loss) else None,
                      native_loss_repr=repr(loss))
        return interval, index, loss, record
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_utc=_utc(), wall_seconds=time.perf_counter() - started)
        if path is not None:
            _write(path, record)


def parallel_sweep_load_predictor(
    *,
    policies: list[str | dict[str, Any]],
    candidates: list[str | dict[str, Any]],
    trace_path: str | None,
    show_progress: bool = False,
    trace_paths: list[str] | None = None,
    trace_format: str | None = None,
    max_workers: int = 8,
    output_dir: str | Path | None = None,
):
    """Return native LoadPredictorResult, retaining full histories and tie order.

    ``output_dir`` belongs to one preparation attempt. Existing manifests are
    refused so interrupted work is not silently treated as a resumed presearch.
    Checkpoints contain task timing/losses; aggregate wall time is separate from
    summed parallel task durations. Native-handled nonfinite losses remain
    native fallback candidates; unexpected worker errors propagate.
    """
    from dynamo.planner.simulation import load_predictor as native
    from dynamo.planner.simulation.presets import throughput_intervals

    if type(max_workers) is not int or not 1 <= max_workers <= 16:
        raise ValueError("Planner presearch max_workers must be an integer in 1..16")
    started = time.perf_counter()
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "presearch-manifest.json").exists():
            raise FileExistsError("Presearch attempt already exists; preserve it and choose a new output directory")
        (output / "tasks").mkdir(exist_ok=True)
    intervals = throughput_intervals(policies)
    paths = list(trace_paths or ([trace_path] if trace_path is not None else []))
    fmt = trace_format or "mooncake"
    history_files = []
    reads_history = bool(intervals and candidates and paths and fmt in {"mooncake", "dynamo"})
    for name in paths if reads_history else []:
        path = Path(name)
        before = path.stat()
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Predictor history changed while being fingerprinted")
        history_files.append({"path": str(path.resolve()), "sha256": checksum, "bytes": after.st_size})
    from dynamo.planner.core.load import predictors
    from dynamo.planner.offline import trace_data
    source_paths = [Path(__file__), Path(native.__file__), Path(predictors.__file__), Path(trace_data.__file__)]
    identity = {
        "algorithm": "nvopt.native-planner-presearch.spawn.v1",
        "history_files": history_files, "requested_trace_paths": paths, "policies": deepcopy(policies),
        "candidates": deepcopy(candidates), "trace_format": fmt,
        "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "package_versions": {name: version(name) for name in ("ai-dynamo", "aisimulate", "numpy", "pandas", "prophet", "pmdarima")},
    }
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    state = {
        "schema": "nvopt.planner-presearch.v1", "status": "preparing",
        "started_utc": _utc(), "max_workers": max_workers,
        "intervals_seconds": intervals, "candidates": deepcopy(candidates),
        "trace_paths": paths, "trace_format": fmt, "policies": deepcopy(policies),
        "cache_key": cache_key, "semantic_identity": identity,
        "result_type": "dynamo.planner.simulation.load_predictor.LoadPredictorResult",
        "tasks_total": 0, "tasks_completed": 0, "cadences": [],
        "completed_tasks": [], "engine_replays": 0,
        "semantics": "Native complete windows; common warmup over all candidates; native evaluate_preset; original candidate order and strict loss comparison.",
    }
    if output is not None:
        _write(output / "presearch-manifest.json", state)

    def checkpoint(event_name):
        state["updated_utc"] = _utc()
        state["wall_seconds"] = time.perf_counter() - started
        if output is not None:
            _write(output / "presearch-progress.json", state)
            with (output / "presearch-events.jsonl").open("a") as stream:
                stream.write(json.dumps({"event": event_name, "observed_utc": state["updated_utc"],
                    "wall_seconds": state["wall_seconds"], "tasks_completed": state["tasks_completed"],
                    "tasks_total": state["tasks_total"]}, allow_nan=False) + "\n")
        if show_progress:
            print(f"Planner presearch: {event_name}; {state['tasks_completed']}/{state['tasks_total']} tasks; {state['wall_seconds']:.3f}s", flush=True)

    prior_env = {key: os.environ.get(key) for key in _THREAD_ENV}
    try:
        if not intervals or not candidates or not paths or fmt not in {"mooncake", "dynamo"}:
            # Preserve native early returns, error ordering, and configured
            # static fallback without inventing parallel tasks.
            result = native.sweep_load_predictor(
                policies=policies, candidates=candidates, trace_path=trace_path,
                show_progress=show_progress, trace_paths=trace_paths, trace_format=trace_format,
            )
        else:
            tasks = []
            for interval in intervals:
                build_started = time.perf_counter()
                windows = native.build_windows_from_trace_paths(paths, fmt, interval)
                warmup = native._common_warmup(candidates, interval)
                state["cadences"].append({"interval_seconds": interval, "window_count": len(windows),
                    "common_warmup": warmup, "preparation_wall_seconds": time.perf_counter() - build_started})
                for index, entry in enumerate(candidates):
                    destination = str(output / "tasks" / f"interval-{interval}-candidate-{index:02d}.json") if output is not None else None
                    tasks.append((interval, index, deepcopy(entry), windows, warmup, destination))
                checkpoint("cadence_prepared")
            state.update(status="evaluating", tasks_total=len(tasks))
            checkpoint("evaluations_started")
            losses = {}
            for key in _THREAD_ENV:
                os.environ[key] = "1"
            with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks)),
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init) as executor:
                futures = [executor.submit(_evaluate, task) for task in tasks]
                for future in as_completed(futures):
                    interval, index, loss, record = future.result()
                    losses[interval, index] = loss
                    state["completed_tasks"].append(record)
                    state["tasks_completed"] += 1
                    checkpoint("evaluation_completed")
            # Only assembly happens here. Forecasting and its loss values are
            # unchanged; completion order cannot influence equal-loss winners.
            result = native.LoadPredictorResult(reason="swept")
            fallback_intervals = []
            for interval in intervals:
                best_index = None
                best_loss = math.inf
                result.losses[interval] = {}
                for index, entry in enumerate(candidates):
                    loss = losses[interval, index]
                    result.losses[interval][native._entry_label(entry, index)] = loss
                    if loss < best_loss:
                        best_loss, best_index = loss, index
                if best_index is None:
                    best_index = 0
                    fallback_intervals.append(interval)
                result.best_by_interval[interval] = deepcopy(candidates[best_index])
            if fallback_intervals:
                result.reason = f"swept; no_winner_configured_fallback@{fallback_intervals}"
        state.update(status="completed", result=result.to_state(),
                     summed_evaluation_wall_seconds=sum(row["wall_seconds"] for row in state["completed_tasks"]))
        checkpoint("completed")
        return result
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        checkpoint("failed")
        raise
    finally:
        for key, value in prior_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        state["finished_utc"] = _utc()
        state["wall_seconds"] = time.perf_counter() - started
        if output is not None:
            _write(output / "presearch-completion.json", state)
