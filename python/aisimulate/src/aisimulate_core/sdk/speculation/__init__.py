# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity-preserving facade for speculative decoding schemes."""

from aiconfigurator_core.sdk import speculation as _canonical
from aiconfigurator_core.sdk.speculation import *  # noqa: F403

__all__ = _canonical.__all__


def __getattr__(name: str) -> object:
    return getattr(_canonical, name)
