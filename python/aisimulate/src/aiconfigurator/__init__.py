# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

try:
    __version__ = importlib.metadata.version("aisimulate")
except importlib.metadata.PackageNotFoundError:
    try:
        # One-release compatibility fallback for environments that still
        # install the legacy distribution name.
        __version__ = importlib.metadata.version("aiconfigurator")
    except importlib.metadata.PackageNotFoundError:
        # Importing directly from a source checkout has no distribution
        # metadata. Installed wheels always take the canonical branch above.
        __version__ = "0+unknown"
