# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache capacity and per-request state memory estimation.

``kv_cache`` estimates rank-local KV-cache capacity. ``state`` resolves the
token and state cache geometry of hybrid models; Kimi K3's lives in ``kimi_k3``
and DeepSeek V4's in ``deepseek_v4``.

``kv_cache`` imports the model registry and the native engine, so its names
resolve on first use and state sizing stays cheap to import.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

from .state import estimate_state_cache as estimate_state_cache

__all__ = [
    "KVCacheEstimator",
    "NaiveKVCacheEstimator",
    "estimate_kv_cache",
    "estimate_num_gpu_blocks",
    "estimate_state_cache",
    "kv_cache_budget_bytes",
]


def __getattr__(name: str) -> object:
    """Resolve KV-capacity names from ``kv_cache`` on first use."""
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.kv_cache"), name)


class _MemoryPackage(ModuleType):
    """Write KV-capacity names through to ``kv_cache``.

    Names the package does not define belong to ``kv_cache``, so patching
    ``memory.get_model`` reaches the estimators that read it, as it did when
    ``kv_cache`` was this module.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if _owns(self, name):
            super().__setattr__(name, value)
        else:
            setattr(import_module(f"{self.__name__}.kv_cache"), name, value)

    def __delattr__(self, name: str) -> None:
        if _owns(self, name):
            super().__delattr__(name)
        else:
            delattr(import_module(f"{self.__name__}.kv_cache"), name)


def _owns(package: ModuleType, name: str) -> bool:
    """Dunders, package attributes and submodules stay on the package."""
    return name.startswith("__") or name in vars(package) or f"{package.__name__}.{name}" in sys.modules


sys.modules[__name__].__class__ = _MemoryPackage
