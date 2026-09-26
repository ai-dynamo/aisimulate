# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collectors for the pinned Rubin SGLang image and GLM-5.2 NVFP4 pilot."""

import importlib
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from pathlib import Path

_WORKER_BOOTSTRAP_ENV = "AISIM_SGLANG_RUBIN_WORKER_BOOTSTRAP"


@contextmanager
def _worker_helper_imports():
    """Scope the helper bootstrap to children of this collector invocation."""
    previous = os.environ.get(_WORKER_BOOTSTRAP_ENV)
    os.environ[_WORKER_BOOTSTRAP_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_WORKER_BOOTSTRAP_ENV, None)
        else:
            os.environ[_WORKER_BOOTSTRAP_ENV] = previous


def _shared_helper():
    """Bind legacy and package imports to one logger/restart-signal module."""
    helper_path = (Path(__file__).resolve().parents[1] / "helper.py").resolve()
    short = sys.modules.get("helper")
    qualified = sys.modules.get("collector.helper")
    for module in (short, qualified):
        if module is not None and Path(getattr(module, "__file__", "")).resolve() != helper_path:
            raise RuntimeError("A different helper module is already imported")
    if short is not None and qualified is not None and short is not qualified:
        raise RuntimeError("Shared collector helper was imported twice; start a fresh Python process")
    helper = short or qualified or importlib.import_module("collector.helper")
    sys.modules["helper"] = helper
    sys.modules["collector.helper"] = helper
    return helper


# The spawn interpreter can import this package from startup hooks even before
# prepare() changes its process name; at that point it carries the standard
# --multiprocessing-fork argv marker. Bind even if neither helper was loaded yet:
# otherwise the executor and MoE would disagree on WorkerRestartSignal's class.
# Main-process imports (including CPU --plan-only/runtime inventory) stay lazy.
if os.environ.get(_WORKER_BOOTSTRAP_ENV) == "1" and (
    mp.current_process().name != "MainProcess" or "--multiprocessing-fork" in sys.argv
):
    _shared_helper()
