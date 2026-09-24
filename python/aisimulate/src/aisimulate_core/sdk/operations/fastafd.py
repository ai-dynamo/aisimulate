# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAFD full-model MoE stage operation."""

import aisimulate_core._native as _core
from aisimulate_core.sdk.operations.base import OpShellKit


class FastAfdMoeStage(_core.FastAfdMoeStage, OpShellKit):
    """Exact measured decode latency for the complete MoE stage."""
