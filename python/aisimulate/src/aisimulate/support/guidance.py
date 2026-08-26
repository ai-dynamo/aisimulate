# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actionable CLI guidance when a requested combination is unsupported."""

from __future__ import annotations

_SUPPORT_GAP_MARKERS = (
    "unsupported",
    "not supported",
    "unavailable",
    "unknown hardware",
    "no feasible",
    "no viable",
    "missing performance data",
    "performance database",
    "no database",
    "database version",
    "no system config",
    "no model config",
    "could not load model",
)


def is_support_gap(error: BaseException | str) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _SUPPORT_GAP_MARKERS)


def self_service_hint() -> str:
    return (
        "No validated support cell is available for this request. "
        "Run 'aisimulate support init --help' to create an exact model/GPU request, "
        "then use 'aisimulate support plan' and 'aisimulate support collect-fpm'."
    )
