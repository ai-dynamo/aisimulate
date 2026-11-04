#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: run ONE planned case of a registered op in this process, as the collector runs it.

planned_row() picks the case of the op plan (collector/op_plans.py) whose fields equal a
selection and returns the row the collector queues for it on this SM; run_planned() runs
that row through the backend registry's run function and perf file, binding what
collect.py binds. A capture calls run_planned(backend, op, **fields), so its cell is a
planned case by construction. Smoking many cases is collect.py's own job
(`--smoke`, `--limit`, `--case-filter`).

The one law applies: a case runs or the component raises; no skip, no substitute.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def planned_row(backend: str, op: str, sm_version: int | None, **select):
    """The run arguments the collector queues for the ONE planned case whose fields equal ``select``.

    The case comes from the op plan (collector/op_plans.plan: the shared cases minus
    collector/skips.py for this backend and SM) and its row from the plan's own rows /
    batch_rows, restricted to that case: a batch row then times exactly that point, and
    everything the plan decides per SM (a pinned dispatch, a KV pool) is what a
    collection would run. A select that matches no planned case, or several, raises:
    a capture must name a cell the collector actually measures.
    """
    import dataclasses

    from collector.op_plans import plan

    op_plan = plan(op, backend, sm_version)
    picked = [
        i
        for i, case in enumerate(op_plan.cases)
        if all(str(getattr(case, field)) == str(value) for field, value in select.items())
    ]
    if len(picked) != 1:
        raise SystemExit(f"{backend} {op} sm{sm_version}: select {select} matches {len(picked)} planned cases, want 1")
    i = picked[0]
    one = dataclasses.replace(
        op_plan, cases=[op_plan.cases[i]], dispatches=op_plan.dispatches[i : i + 1] if op_plan.dispatches else []
    )
    rows = one.rows
    if len(rows) != 1:
        raise SystemExit(f"{backend} {op}: the planned case {op_plan.cases[i]} runs as {len(rows)} rows, want 1")
    return rows[0]


def _bound_kwargs(entry, run_func, out_dir: str) -> dict:
    """The keyword arguments collect.py binds besides perf_filename and device."""
    from inspect import signature

    extra = {}
    if getattr(entry, "extra_perf_filenames", None) and "extra_perf_filenames" in signature(run_func).parameters:
        # multi-table producers (compute_scale -> computescale + scale_matrix): same kwarg collect.py binds
        extra["extra_perf_filenames"] = tuple(str(Path(out_dir) / str(p)) for p in entry.extra_perf_filenames)
    if getattr(entry, "run_kwargs", None):
        # fixed mode arguments the executor binds (collect.py run_kwargs partial), e.g. skip_indexer=True
        extra.update(dict(entry.run_kwargs))
    return extra


def _run_case(run_func, case, perf_path: str, device: str, extra: dict) -> None:
    if isinstance(case, dict) and "params" in case:  # collect.py task shape {"id", "params"} (collect.py:1696-1702)
        params = case["params"]
        if isinstance(params, dict):
            run_func(**params, perf_filename=perf_path, device=device, **extra)
        else:
            run_func(*params, perf_filename=perf_path, device=device, **extra)
    elif isinstance(case, dict):  # keyword-shaped cases bind by name
        run_func(**{k: v for k, v in case.items() if k != "id"}, perf_filename=perf_path, device=device, **extra)
    else:
        run_func(*case, perf_filename=perf_path, device=device, **extra)


def run_planned(backend: str, op: str, *, out_dir: str = "/tmp/smk", device: str = "cuda:0", **select) -> None:
    """Run ONE planned case in this process, as the collector runs it: the row planned_row()
    builds for this SM, through the backend registry's run function and perf file. A capture
    that calls this names its cell as a planned case and nothing else."""
    from collector.helper import get_sm_version

    registry = importlib.import_module(f"collector.{backend}.registry")
    entries = [e for e in registry.REGISTRY if e.op == op]
    if len(entries) != 1:
        raise SystemExit(f"op {op!r}: {len(entries)} registry entries in collector.{backend}.registry, want 1")
    entry = entries[0]
    module = importlib.import_module(entry.module)
    run_func = getattr(module, entry.run_func)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    row = planned_row(backend, op, get_sm_version(), **select)
    print(f"[op_smoke] {backend} {op}: {entry.run_func}{row}")
    _run_case(
        run_func, row, str(Path(out_dir) / str(entry.perf_filename)), device, _bound_kwargs(entry, run_func, out_dir)
    )
