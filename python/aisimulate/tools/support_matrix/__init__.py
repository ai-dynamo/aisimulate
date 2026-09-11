# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Support matrix generation and validation tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["SupportMatrix"]


def __getattr__(name: str) -> Any:
    """Keep the legacy export without loading the full matrix stack eagerly."""
    if name != "SupportMatrix":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tools.support_matrix.support_matrix import SupportMatrix

    globals()[name] = SupportMatrix
    return SupportMatrix


if TYPE_CHECKING:
    from tools.support_matrix.support_matrix import SupportMatrix
