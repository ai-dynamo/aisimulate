# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: the CUDA kernels ONE planned case launches, run the way the collector runs it.

Arguments (``--case-filter key=value,...``): ``backend`` (collect.py's ``--backend``),
``op``, and op-plan case fields naming exactly one planned case on this SM, e.g.
``op=dsa_context_module,model_path=deepseek-ai/DeepSeek-V3.2,seq_len=4096,batch_size=1,...``.

``planned_row`` takes the case from the op plan (the shared cases minus collector/skips.py
for this backend and SM) and builds its row from the plan's own rows / batch_rows,
restricted to that case, so a batch row times exactly that point and everything the plan
decides per SM (a pinned dispatch, a KV pool) is what a collection would run. The row runs
once through the backend registry's run function under ``torch.profiler``, binding what
collect.py binds, with its perf files in a discarded scratch directory. The result lists
the kernels with device time, minus aten/memcpy/memset/runtime entries: the collector side
of an opharness path gate. A worker that spawns a subprocess shows no kernels here.
"""

from __future__ import annotations

import dataclasses
import importlib
import shutil
import tempfile
from inspect import signature
from pathlib import Path

_NOT_KERNELS = ("aten::", "Memcpy", "Memset", "cuda", "Cuda")
_RESTART_CODE = 10  # collector.helper.EXIT_CODE_RESTART: the worker recycles after the case ran


def planned_row(backend: str, op: str, sm_version: int | None, **select):
    """The run arguments the collector queues for the ONE planned case whose fields equal
    ``select``. A selection matching no planned case, or several, raises."""
    from collector.op_plans import plan

    op_plan = plan(op, backend, sm_version)
    picked = [
        i
        for i, case in enumerate(op_plan.cases)
        if all(str(getattr(case, field)) == str(value) for field, value in select.items())
    ]
    if len(picked) != 1:
        raise ValueError(f"{backend} {op} sm{sm_version}: select {select} matches {len(picked)} planned cases, want 1")
    i = picked[0]
    one = dataclasses.replace(
        op_plan, cases=[op_plan.cases[i]], dispatches=op_plan.dispatches[i : i + 1] if op_plan.dispatches else []
    )
    rows = one.rows
    if len(rows) != 1:
        raise ValueError(f"{backend} {op}: the planned case {op_plan.cases[i]} runs as {len(rows)} rows, want 1")
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


def kernel_names(prof) -> list[str]:
    return sorted(
        {
            ev.key
            for ev in prof.key_averages()
            if (getattr(ev, "self_device_time_total", 0) or getattr(ev, "self_cuda_time_total", 0)) > 0
            and not ev.key.startswith(_NOT_KERNELS)
        }
    )


def run(args: dict[str, str], out_dir: Path) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from collector.helper import get_sm_version

    select = dict(args)
    backend, op = select.pop("backend"), select.pop("op")
    sm = get_sm_version()
    row = planned_row(backend, op, sm, **select)
    entry, run_func = _resolve(backend, op)
    scratch = Path(tempfile.mkdtemp(prefix="trace_case_"))
    error = None
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            try:
                _run_row(
                    run_func,
                    row,
                    str(scratch / str(entry.perf_filename)),
                    "cuda:0",
                    _bound_kwargs(entry, run_func, scratch),
                )
            except SystemExit as exc:
                if exc.code not in (0, None, _RESTART_CODE):
                    error = f"exit={exc.code}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        kernels = kernel_names(prof)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "backend": backend,
        "op": op,
        "sm": sm,
        "select": select,
        "row": str(row),
        "kernels": kernels,
        "error": error,
        "failed": error is not None or not kernels,
    }
