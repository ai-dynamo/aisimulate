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

_NOT_KERNELS = ("aten::", "Memcpy", "Memset", "cuda", "Cuda")


def kernel_names(prof) -> list[str]:
    """Profiler entries with device time, minus aten/memcpy/memset/runtime entries."""
    return sorted(
        {
            ev.key
            for ev in prof.key_averages()
            if (getattr(ev, "self_device_time_total", 0) or getattr(ev, "self_cuda_time_total", 0)) > 0
            and not ev.key.startswith(_NOT_KERNELS)
        }
    )


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
        except BaseException as exc:  # recorded below, then re-raised so the parent sees the failure
            failure = exc
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    record = {"mode": mode, "status": status, "error": None if failure is None else repr(failure)[:500]}
    record["kernels"] = kernel_names(prof)
    (Path(out_dir) / f"child_{os.getpid()}.json").write_text(json.dumps(record))
    if failure is not None:
        raise failure
    sys.exit(status)


if __name__ == "__main__":
    main()
