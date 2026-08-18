# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

try:
    __version__ = importlib.metadata.version("aisimulate")
except importlib.metadata.PackageNotFoundError:
    # Source-tree and one-release compatibility fallback for environments
    # that still install the legacy distribution name.
    __version__ = importlib.metadata.version("aiconfigurator")
