# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime hooks that add per-request fields to the FPM stream of stock engines.

The learned forward-pass model's per-request feature presets (``sglang18``,
``hisim``) need ``extend_lengths`` / ``past_kv_lengths`` in
``ForwardPassMetrics.scheduled_requests``. Stock Dynamo (vLLM path) and stock
SGLang emit aggregates only. Instead of patching either code base, this package
installs the two lists at import time inside the engine process:

- SGLang (``_sglang.py``): wraps ``SchedulerMetricsReporter._build_scheduled_request_metrics`` and
  swaps ``ScheduledRequestMetrics`` for a subclass carrying the two lists
  (values from the schedule-time ``batch.extend_lens`` / ``batch.prefix_lens``).
- Dynamo vLLM (``_dynamo_vllm.py``): wraps ``InstrumentedScheduler._extract_scheduled`` the same way
  (values from ``SchedulerOutput.num_scheduled_tokens`` / ``num_computed_tokens``).

Both are no-ops when the producer already carries the fields natively. The
msgpack payload keeps FPM ``version`` 1 and is decoded by the Dynamo runtime
unchanged; aggregate-only consumers ignore the extra keys.

Activation (either):

    # 1. automatic, also inside spawned scheduler subprocesses
    export PYTHONPATH=$(python -c 'import aisimulate_core.fpm_hooks as h; print(h.hook_path())'):$PYTHONPATH
    python -m dynamo.sglang ...   # or python -m dynamo.vllm ...

    # 2. explicit runner (sets PYTHONPATH for the children itself)
    python -m aisimulate_core.fpm_hooks dynamo.sglang -- ...

Hooks depend on private engine internals (method names listed above); the
installer logs and skips when a target does not look as expected, so an
incompatible engine version degrades to aggregate-only FPM instead of failing.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from types import ModuleType

from ._dynamo_vllm import patch_dynamo_vllm_instrumented_scheduler
from ._sglang import patch_sglang_metrics_reporter

__all__ = [
    "TARGETS",
    "hook_path",
    "install",
    "patch_dynamo_vllm_instrumented_scheduler",
    "patch_sglang_metrics_reporter",
]

logger = logging.getLogger("aisimulate.fpm_hooks")

#: module name -> patch function applied right after that module is executed
TARGETS: dict[str, Callable[[ModuleType], None]] = {
    "sglang.srt.managers.scheduler_components.metrics_reporter": patch_sglang_metrics_reporter,
    "dynamo.vllm.instrumented_scheduler": patch_dynamo_vllm_instrumented_scheduler,
}

_INSTALLED_ATTR = "_aisimulate_fpm_hooks_installed"


def hook_path() -> str:
    """Directory to prepend to ``PYTHONPATH`` so ``sitecustomize`` installs the hooks.

    The hook modules are named ``_sglang`` / ``_dynamo_vllm`` on purpose: this directory
    precedes site-packages on ``sys.path``, so a module called ``sglang.py`` here would
    shadow the real ``sglang`` package.
    """
    return os.path.dirname(os.path.abspath(__file__))


class _PatchingLoader(importlib.abc.Loader):
    """Delegates to the real loader and runs the patch after ``exec_module``."""

    def __init__(self, inner: importlib.abc.Loader, patch: Callable[[ModuleType], None], name: str) -> None:
        self._inner = inner
        self._patch = patch
        self._name = name

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._inner.exec_module(module)
        try:
            self._patch(module)
        except Exception:
            logger.exception("fpm_hooks: patch for %s failed; FPM stays aggregate-only", self._name)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    def __init__(self, targets: dict[str, Callable[[ModuleType], None]]) -> None:
        self._targets = dict(targets)
        self._resolving: set[str] = set()

    def target_names(self) -> set[str]:
        return set(self._targets)

    def find_spec(self, fullname, path, target=None):
        patch = self._targets.get(fullname)
        if patch is None or fullname in self._resolving:
            return None
        self._resolving.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._resolving.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, patch, fullname)
        return spec


def install(targets: dict[str, Callable[[ModuleType], None]] | None = None) -> bool:
    """Register the post-import patcher (idempotent). Patches already-imported targets directly.

    Returns ``True`` when newly installed, ``False`` when it was already active.
    """
    if getattr(sys, _INSTALLED_ATTR, False) and targets is None:
        return False
    chosen = TARGETS if targets is None else targets
    # one finder per target set: repeated explicit installs must not stack finders
    for finder in sys.meta_path:
        if isinstance(finder, _PostImportPatcher) and finder.target_names() == set(chosen):
            return False
    sys.meta_path.insert(0, _PostImportPatcher(chosen))
    if targets is None:
        setattr(sys, _INSTALLED_ATTR, True)
    for name, patch in chosen.items():
        module = sys.modules.get(name)
        if module is not None:
            try:
                patch(module)
            except Exception:
                logger.exception("fpm_hooks: patch for already-imported %s failed", name)
    return True
