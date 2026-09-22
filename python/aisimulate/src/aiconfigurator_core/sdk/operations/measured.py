# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operations backed by externally measured complete stages."""

from __future__ import annotations

import aiconfigurator_core._aiconfigurator_core as _core

from aiconfigurator_core.sdk.operations.base import OpShellKit


class MeasuredStage(_core.MeasuredStage, OpShellKit):
    """Fixed latency for an externally measured complete stage.

    The caller must validate workload, topology, and provenance before
    construction. This operation has no energy value and reports source
    ``"external"``; it is not a packaged silicon-data lookup.
    """


__all__ = ["MeasuredStage"]
