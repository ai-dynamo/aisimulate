# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure SGLang host costs on a serving host and write them as a `HostProfile`.

Frontend pools are measured by wrapping the real worker functions of the pinned
SGLang release inside their own executors; scheduler-thread receive costs run the
real per-request preparation on one thread. Batch-level launch and result costs
need a serving run with a GPU and are read from an operator-supplied measurement;
absent, the profile lists them as missing and predictions fail closed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..profile import HostProfile, load_host_profile

_SUPERVISION_VARIABLES = (
    "_AISIMULATE_SUPERVISED_BUDGET",
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
) -> list[str]:
    """Command line that samples the given workload into ``output``."""
    return [
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


def calibrate_in_subprocess(
    *, output: str | Path, model: str, frontend: str, images: Mapping[str, Any], python: str | None = None
) -> HostProfile:
    """Sample the workload in a fresh interpreter and load the written profile.

    A supervised prediction pins its own process to one thread; the sampler must
    see the serving host's real thread pools, so the supervision variables are
    dropped from its environment.
    """
    env = {name: value for name, value in os.environ.items() if name not in _SUPERVISION_VARIABLES}
    command = calibration_command(output=output, model=model, frontend=frontend, images=images, python=python)
    completed = subprocess.run(command, env=env, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"host calibration failed ({completed.returncode}):\n{completed.stderr[-8000:]}")
    return load_host_profile(output)


__all__ = ["calibrate_in_subprocess", "calibration_command"]
