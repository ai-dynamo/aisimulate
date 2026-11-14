# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a collector's measurement child under torch.profiler for probe.trace_case.

Some collectors time each point in a child process (``python -c <code>`` or
``python -m <module>``), where the parent's profiler sees no kernels. While
probe.trace_case runs, ``AIS_TRACE_CHILD_DIR`` is set and ``helper.trace_child_argv``
starts such a child as ``python <this file> <dir> -c <code>`` (or ``-m <module> ...``).
This file runs the original child unchanged, with ``sys.path[0]`` set as Python sets it
for ``-c`` / ``-m`` (the working directory), and writes the kernels it launched to
``<dir>/child_<pid>.json``. It imports nothing from ``collector``: a child's working
directory need not make the package importable.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

# collector.helper.TRACE_TIMED_RANGE / TRACE_MEASURED_RANGE: benchmark_with_power's timed region
# and, inside it, the timed runs alone, while tracing (this file runs as a script in a
# measurement child and imports nothing from collector)
TIMED_RANGE = "ais::timed"
MEASURED_RANGE = "ais::measured"


def kernel_names(prof) -> tuple[list[str], bool, list[str] | None]:
    """The CUDA kernels the profiler saw, whether they are only the timed ones, and the kernels
    of the timed runs alone (None when no timed run completed).

    Device events only, as the serving probes take them: a custom op that launches its kernel
    with no aten op in between (deep_gemm, the ``_C`` extensions) carries self device time on
    the CPU side, so a device-time filter returned ``vllm::fp8_gemm_nt_op`` and ``_C::rms_norm``
    as kernels. When the run timed through benchmark_with_power, only kernels starting inside
    its timed range count: the module build and the cache fill before it are not the measured
    path. The measured set leaves out the warmups' one-time work too (JIT, autotune inputs):
    every kernel in it is paid on each timed call. GPU-side user annotations and memcpy/memset
    are not kernels."""
    events = prof.events()

    def inside(spans):
        return sorted({e.name for e in kernels if any(lo <= e.time_range.start <= hi for lo, hi in spans)})

    kernels = _kernel_events(events)
    timed, measured = _windows(events, TIMED_RANGE), _windows(events, MEASURED_RANGE)
    if not timed:
        return sorted({e.name for e in kernels}), False, None
    return inside(timed), True, inside(measured) if measured else None


def measured_timing(prof) -> list[dict] | None:
    """The device time of the timed runs, one table per measured window (each
    benchmark_with_power call, in launch order): ``{kernel: {"us": total, "launches": n}}``
    over the kernels that started inside it. The path gates compare the per-launch time of a
    window with serving's at the matching shape. None when no window was recorded."""
    events = prof.events()
    spans = sorted(_windows(events, MEASURED_RANGE))
    if not spans:
        return None
    tables: list[dict] = [{} for _ in spans]
    for e in _kernel_events(events):
        start = e.time_range.start
        for i, (lo, hi) in enumerate(spans):
            if lo <= start <= hi:
                slot = tables[i].setdefault(e.name, {"us": 0.0, "launches": 0})
                slot["us"] += e.time_range.end - e.time_range.start
                slot["launches"] += 1
                break
    for table in tables:
        for slot in table.values():
            slot["us"] = round(slot["us"], 3)
    return tables


def _windows(events, name):
    from torch.autograd import DeviceType

    return [
        (e.time_range.start, e.time_range.end) for e in events if e.name == name and e.device_type == DeviceType.CPU
    ]


def _kernel_events(events):
    from torch.autograd import DeviceType

    return [
        e
        for e in events
        if e.device_type == DeviceType.CUDA
        and not getattr(e, "is_user_annotation", False)
        and e.name not in (TIMED_RANGE, MEASURED_RANGE)
        and not e.name.startswith(("Memcpy", "Memset"))
    ]


def main() -> None:
    out_dir, mode, target, rest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    sys.path[0] = os.getcwd()
    import torch
    from torch.profiler import ProfilerActivity, profile

    status, failure = 0, None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        try:
            if mode == "-c":
                sys.argv = ["-c", *rest]
                exec(compile(target, "<string>", "exec"), {"__name__": "__main__"})
            else:
                sys.argv = [target, *rest]
                runpy.run_module(target, run_name="__main__", alter_sys=True)
        except SystemExit as exc:
            status = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if isinstance(exc.code, str):  # what Python prints for sys.exit("message"); the parent logs stderr
                print(exc.code, file=sys.stderr)
        except BaseException as exc:  # recorded below, then re-raised so the parent sees the failure
            failure = exc
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    record = {"mode": mode, "status": status, "error": None if failure is None else repr(failure)[:500]}
    record["kernels"], record["timed"], record["measured"] = kernel_names(prof)
    record["measured_timing"] = measured_timing(prof)
    (Path(out_dir) / f"child_{os.getpid()}.json").write_text(json.dumps(record))
    if failure is not None:
        raise failure
    sys.exit(status)


if __name__ == "__main__":
    main()
