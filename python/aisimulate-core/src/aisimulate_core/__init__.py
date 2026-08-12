# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public AISimulate core namespace.

The implementation keeps the ``aiconfigurator_core`` package importable for
the AIC 0.12.0 compatibility window. New consumers should import this module or
``aisimulate_core.sdk``.
"""

from aiconfigurator_core import (
    AicEngine,
    RustForwardPassPerfModel,
    _build_smoke,
    engine_spec_bincode_from_json,
)

__all__ = [
    "AicEngine",
    "RustForwardPassPerfModel",
    "_build_smoke",
    "engine_spec_bincode_from_json",
]
