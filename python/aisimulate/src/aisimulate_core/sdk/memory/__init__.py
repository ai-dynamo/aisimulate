# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache capacity and per-request state memory estimation.

``kv_cache`` estimates rank-local KV-cache capacity. ``state`` resolves the
token and state cache geometry of hybrid models; DeepSeek V4's lives in
``deepseek_v4``.

``kv_cache`` imports the model registry and the native engine, so its names
resolve on first use and state sizing stays cheap to import.
"""

from __future__ import annotations

from importlib import import_module

from .state import estimate_state_cache as estimate_state_cache


def __getattr__(name: str) -> object:
    """Resolve KV-capacity names from ``kv_cache`` on first use."""
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.kv_cache"), name)
