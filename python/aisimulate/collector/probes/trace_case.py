# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: the CUDA kernels ONE planned case launches, run the way the collector runs it.

Arguments (``--case-filter key=value,...``): ``backend`` (collect.py's ``--backend``),
``op``, and op-plan case fields naming exactly one planned case on this SM, e.g.
``op=dsa_context_module,model_path=deepseek-ai/DeepSeek-V3.2,seq_len=4096,batch_size=1,...``.
A sweep field takes a ``+``-joined subset of its planned values (``num_tokens=1+4096``), and
``env.<NAME>=value`` sets a variable the collector reads for this run only (recorded in the
result).

``planned_row`` takes the case from the op plan (the shared cases minus collector/skips.py
for this backend and SM) and builds its row from the plan's own rows / batch_rows,
restricted to that case, so a batch row times exactly that point and everything the plan
decides per SM (a pinned dispatch, a KV pool) is what a collection would run. The row runs
once through the backend registry's run function under ``torch.profiler``, binding what
collect.py binds, with its perf files in a discarded scratch directory. The result lists
the CUDA kernels the row times (``timed_only``: inside benchmark_with_power; otherwise the
whole run): the collector side of an opharness path gate. A collector that times its
points in a child process (``helper.trace_child_argv``) runs that child under the profiler
too, and its kernels are included (``measurement_children`` counts them).
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import shutil
import tempfile
from inspect import signature
from pathlib import Path

from collector.probes._traced_child import kernel_names

_RESTART_CODE = 10  # collector.helper.EXIT_CODE_RESTART: the worker recycles after the case ran


def _sweep_subset(value) -> list:
    """A selection for a sweep (list) field: a list, or ``1+4096`` from a probe argument."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [int(v) if v.lstrip("-").isdigit() else v for v in str(value).split("+")]


def _matches(case, select: dict) -> bool:
    for field, value in select.items():
        planned = getattr(case, field)
        if isinstance(planned, (list, tuple)):  # a sweep field: the selection names a subset of the planned sweep
            sub = _sweep_subset(value)
            if not sub or not set(sub) <= set(planned):
                return False
        elif str(planned) != str(value):
            return False
    return True


def planned_row(backend: str, op: str, sm_version: int | None, **select):
    """The run arguments the collector queues for the ONE planned case whose fields equal
    ``select``. A sweep field (a list in the case, such as moe ``num_tokens`` or gdn
    ``seq_lens``) may be given a subset of its planned values; the case then sweeps only
    those. A selection matching no planned case, or several, raises."""
    from collector.op_plans import plan

    op_plan = plan(op, backend, sm_version)
    picked = [i for i, case in enumerate(op_plan.cases) if _matches(case, select)]
    if len(picked) != 1:
        raise ValueError(f"{backend} {op} sm{sm_version}: select {select} matches {len(picked)} planned cases, want 1")
    i = picked[0]
    case = op_plan.cases[i]
    narrowed = {}
    for field in select:
        planned = getattr(case, field)
        if isinstance(planned, (list, tuple)):
            keep = set(_sweep_subset(select[field]))
            narrowed[field] = type(planned)(v for v in planned if v in keep)
    if narrowed:
        case = case._replace(**narrowed)
    one = dataclasses.replace(
        op_plan, cases=[case], dispatches=op_plan.dispatches[i : i + 1] if op_plan.dispatches else []
    )
    rows = one.rows
    if len(rows) != 1:
        raise ValueError(f"{backend} {op}: the planned case {case} runs as {len(rows)} rows, want 1")
    return rows[0]


def _resolve(backend: str, op: str):
    """(registry entry, run function) for the op."""
    registry = importlib.import_module(f"collector.{backend}.registry")
    entries = [e for e in registry.REGISTRY if e.op == op]
    if len(entries) != 1:
        raise ValueError(f"op {op!r}: {len(entries)} registry entries in collector.{backend}.registry, want 1")
    entry = entries[0]
    return entry, getattr(importlib.import_module(entry.module), entry.run_func)


def _bound_kwargs(entry, run_func, out_dir: Path) -> dict:
    """The keyword arguments collect.py binds besides perf_filename and device."""
    extra = {}
    if getattr(entry, "extra_perf_filenames", None) and "extra_perf_filenames" in signature(run_func).parameters:
        extra["extra_perf_filenames"] = tuple(str(out_dir / str(p)) for p in entry.extra_perf_filenames)
    if getattr(entry, "run_kwargs", None):
        extra.update(dict(entry.run_kwargs))
    return extra


def _run_row(run_func, row, perf_path: str, device: str, extra: dict) -> None:
    """Call the run function with the row the way collect.py's worker does (tuple, {id, params} or dict)."""
    if isinstance(row, dict) and "params" in row:
        params = row["params"]
        if isinstance(params, dict):
            run_func(**params, perf_filename=perf_path, device=device, **extra)
        else:
            run_func(*params, perf_filename=perf_path, device=device, **extra)
    elif isinstance(row, dict):
        run_func(**{k: v for k, v in row.items() if k != "id"}, perf_filename=perf_path, device=device, **extra)
    else:
        run_func(*row, perf_filename=perf_path, device=device, **extra)


def run(args: dict[str, str], out_dir: Path) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from collector.helper import TRACE_CHILD_DIR_ENV, get_sm_version

    select = {k: v for k, v in args.items() if not k.startswith("env.")}
    env = {k[len("env.") :]: v for k, v in args.items() if k.startswith("env.")}
    backend, op = select.pop("backend"), select.pop("op")
    sm = get_sm_version()
    row = planned_row(backend, op, sm, **select)
    entry, run_func = _resolve(backend, op)
    scratch = Path(tempfile.mkdtemp(prefix="trace_case_"))
    children = scratch / "children"
    saved = {name: os.environ.get(name) for name in (TRACE_CHILD_DIR_ENV, *env)}
    os.environ[TRACE_CHILD_DIR_ENV] = str(children)  # a measurement child records its own kernels there
    os.environ.update(env)  # env.<NAME>=value: a capture-only switch the collector reads (it is in the result)
    error = None
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            try:
                perf = str(scratch / str(entry.perf_filename))
                _run_row(run_func, row, perf, "cuda:0", _bound_kwargs(entry, run_func, scratch))
            except SystemExit as exc:
                if exc.code not in (0, None, _RESTART_CODE):
                    error = f"exit={exc.code}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        own, own_timed = kernel_names(prof)
        child_records = [json.loads(f.read_text()) for f in sorted(children.glob("child_*.json"))]
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(scratch, ignore_errors=True)
    # a process that timed through benchmark_with_power gave its timed kernels; when any did,
    # the processes that timed nothing (a parent that only spawned the measurement) are setup
    traced = [(own, own_timed), *((r["kernels"], r.get("timed", False)) for r in child_records)]
    timed = any(t for _, t in traced)
    kernels = sorted(set().union(*(set(k) for k, t in traced if t or not timed)))
    return {
        "backend": backend,
        "op": op,
        "sm": sm,
        "select": select,
        "env": env,
        "row": str(row),
        "kernels": kernels,
        "timed_only": timed,
        "measurement_children": len(child_records),
        "error": error,
        "failed": error is not None or not kernels,
    }
