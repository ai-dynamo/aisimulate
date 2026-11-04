# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-case kernel trace: which CUDA kernels a collector case launches.

With ``AIS_TRACE_KERNELS=1`` (``collect.py --trace-kernels`` sets it; the collection
pipeline passes it through ``AISIM_COLLECTOR_ENV``) every worker task runs once more
under ``torch.profiler`` before its timed run. That extra run writes its rows to a
scratch directory that is discarded, so the table rows are the same with or without
the trace (overview rule 5); the cost is one more execution per traced case. The
kernels land next to the perf file as ``<perf_filename>.kernels.jsonl``, one line per
task::

    {"task_id": ..., "case": "<the task's string form>", "kernels": [...], "error": null}

``kernels`` uses the same selection as the opharness ``path_diff --capture`` file
(profiler entries with device time, minus aten/memcpy/memset/runtime entries), so a
path gate can grade a traced shard instead of a hand-written capture script. A case
whose kernels run in a subprocess the worker spawns records an empty list.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import tempfile
from pathlib import Path

TRACE_ENV = "AIS_TRACE_KERNELS"
_NOT_KERNELS = ("aten::", "Memcpy", "Memset", "cuda", "Cuda")


def enabled() -> bool:
    return os.environ.get(TRACE_ENV) == "1"


def scratch_bound(func, scratch: Path):
    """``func`` with every perf file it writes redirected into ``scratch``."""
    keywords = getattr(func, "keywords", None) or {}
    if "perf_filename" not in keywords:
        raise ValueError("kernel trace needs a run function with perf_filename bound (collect.py binds it)")
    over = {"perf_filename": str(scratch / Path(keywords["perf_filename"]).name)}
    if keywords.get("extra_perf_filenames"):
        over["extra_perf_filenames"] = tuple(str(scratch / Path(p).name) for p in keywords["extra_perf_filenames"])
    return functools.partial(func, **over)


def kernel_names(prof) -> list[str]:
    return sorted(
        {
            ev.key
            for ev in prof.key_averages()
            if (getattr(ev, "self_device_time_total", 0) or getattr(ev, "self_cuda_time_total", 0)) > 0
            and not ev.key.startswith(_NOT_KERNELS)
        }
    )


def trace_case(func, task, device, task_id: str, restart_code: int) -> None:
    """Run ``func(*task, device=device)`` once under the profiler with its perf files
    discarded, and append the kernels it launched to ``<perf_filename>.kernels.jsonl``.
    A failure is recorded in the line; the timed run that follows reports it as usual."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    perf = Path(func.keywords["perf_filename"])
    scratch = Path(tempfile.mkdtemp(prefix="kernel_trace_"))
    error = None
    try:
        traced = scratch_bound(func, scratch)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            try:
                traced(*task, device=device)
            except SystemExit as exc:  # a collector recycling its worker after the case is not a failure
                if exc.code not in (0, None, restart_code):
                    error = f"exit={exc.code}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        line = {"task_id": task_id, "case": str(task), "kernels": kernel_names(prof), "error": error}
        with perf.with_name(perf.name + ".kernels.jsonl").open("a") as fh:
            fh.write(json.dumps(line) + "\n")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
