# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure SGLang host costs on a serving host and write them as a `HostProfile`.

Frontend pools are measured by wrapping the real worker entry points of the
pinned SGLang release inside their own executors; scheduler-thread receive costs
run the real per-request preparation of the sampled frontend path on one thread.
Batch-level launch and result costs need a serving run with a GPU and are read
from an operator-supplied measurement; absent, the profile lists them as missing
and predictions fail closed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..profile import HostProfile, load_host_profile

_SUPERVISION_VARIABLES = (
    "_AISIMULATE_SUPERVISED_BUDGET",
    "_AISIMULATE_HOST_CPUS",
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
    "TOKENIZERS_PARALLELISM",
)


def calibration_command(
    *,
    output: str | Path,
    model: str,
    frontend: str,
    images: Mapping[str, Any],
    python: str | None = None,
    text_tokens: int | None = None,
) -> list[str]:
    """Command line that samples the given workload into ``output``."""
    command = [
        python or sys.executable,
        "-m",
        "aisimulate.vl.calibrate",
        "--frontend",
        frontend,
        "--model",
        model,
        "--images",
        f"{images['height']}x{images['width']}x{images.get('count', 1)}",
        "--encoding",
        str(images.get("encoding", "png")),
        "--output",
        str(output),
    ]
    if text_tokens is not None:
        command += ["--text-tokens", str(int(text_tokens))]
    for name in ("min_pixels", "max_pixels"):
        if images.get(name) is not None:
            command += [f"--{name.replace('_', '-')}", str(int(images[name]))]
    return command


def calibrate_in_subprocess(
    *,
    output: str | Path,
    model: str,
    frontend: str,
    images: Mapping[str, Any],
    python: str | None = None,
    text_tokens: int | None = None,
) -> HostProfile:
    """Sample the workload in a fresh interpreter and load the written profile.

    A supervised prediction pins its own process to a CPU slice and one thread
    per pool; the sampler must see the serving host's real cores and thread
    pools, so the supervision variables are dropped from its environment and the
    CPU mask the supervisor recorded before pinning is restored. The raised nice
    level cannot be undone without privileges; the sampler records it in the
    profile's provenance.
    """
    env = {name: value for name, value in os.environ.items() if name not in _SUPERVISION_VARIABLES}
    command = calibration_command(
        output=output, model=model, frontend=frontend, images=images, python=python, text_tokens=text_tokens
    )
    host_cpus = os.environ.get("_AISIMULATE_HOST_CPUS")
    preexec_fn = None
    if host_cpus and hasattr(os, "sched_setaffinity"):
        cpus = set(json.loads(host_cpus))

        def preexec_fn() -> None:
            os.sched_setaffinity(0, cpus)

    completed = subprocess.run(command, env=env, check=False, capture_output=True, text=True, preexec_fn=preexec_fn)
    if completed.returncode:
        raise RuntimeError(f"host calibration failed ({completed.returncode}):\n{completed.stderr[-8000:]}")
    return load_host_profile(output)


__all__ = ["calibrate_in_subprocess", "calibration_command"]
