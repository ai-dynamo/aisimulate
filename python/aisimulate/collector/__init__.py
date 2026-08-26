# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ["get_sm_version", "log_perf"]


def __getattr__(name: str):
    """Load GPU runtime helpers only when callers request them.

    The application wheel ships the framework-neutral FPM planning workflow,
    whose ``collector.model_cases`` import must not pull the optional GPU
    benchmark stack into an installed planning process.
    """
    if name not in __all__:
        raise AttributeError(name)
    from . import helper

    return getattr(helper, name)
