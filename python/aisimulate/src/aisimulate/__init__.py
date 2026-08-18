# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public package marker for the unified AISimulate wheel."""

from importlib.metadata import version

__version__ = version("aisimulate")

__all__ = ["__version__"]
