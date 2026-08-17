# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable AISimulate SDK facade backed by the migrated AIC core."""

from __future__ import annotations

from aiconfigurator_core import sdk as _compat_sdk
from aiconfigurator_core.sdk import *  # noqa: F403

__all__ = _compat_sdk.__all__
# Search the AISimulate shim directory first, then fall back to the migrated
# implementation tree for non-public/internal modules during the compatibility
# window.
__path__ = [*globals()["__path__"], *_compat_sdk.__path__]
__getattr__ = _compat_sdk.__getattr__
__dir__ = _compat_sdk.__dir__
