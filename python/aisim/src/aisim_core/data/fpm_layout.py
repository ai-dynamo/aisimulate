# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical-to-physical path rule for FPM files after the move to ``data/fpm/`` (AISIM-2338).

The move was byte-identical, so 202 text files (every leaf ``manifest.json``, FPM
metadata, import records) still embed the legacy ``data/<model>/...`` spelling.
Those strings are *logical keys*: they keep configuration, case and observation
identities stable across the move. The physical repository path is
``data/fpm/<rest>`` (``data/history/...`` becomes ``data/fpm/history/...``).
"""

from __future__ import annotations

LEGACY_PREFIX = "data/"
PHYSICAL_PREFIX = "data/fpm/"


def physical_path(logical: str) -> str:
    """``data/<rest>`` -> ``data/fpm/<rest>``; an already physical path is returned unchanged."""
    if logical.startswith(PHYSICAL_PREFIX):
        return logical
    if not logical.startswith(LEGACY_PREFIX):
        raise ValueError(f"not a dataset path: {logical!r}")
    return PHYSICAL_PREFIX + logical[len(LEGACY_PREFIX) :]


def logical_path(physical: str) -> str:
    """Inverse of :func:`physical_path` (the legacy spelling stored inside moved files)."""
    if not physical.startswith(PHYSICAL_PREFIX):
        raise ValueError(f"not a physical FPM path: {physical!r}")
    return LEGACY_PREFIX + physical[len(PHYSICAL_PREFIX) :]
