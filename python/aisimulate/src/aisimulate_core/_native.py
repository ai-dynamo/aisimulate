# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimator bindings backed by the unified ``aisimulate._runtime`` extension."""

from __future__ import annotations

from aisimulate import _runtime

__all__ = tuple(sorted(name for name in dir(_runtime) if not name.startswith("_")))
globals().update({name: getattr(_runtime, name) for name in __all__})


def __getattr__(name: str) -> object:
    try:
        return getattr(_runtime, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted({*globals(), *dir(_runtime)})
