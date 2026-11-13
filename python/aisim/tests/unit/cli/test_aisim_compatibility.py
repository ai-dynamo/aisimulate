# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command identity contract for the application shipped by AISim."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.unit


def test_aisim_owns_the_single_simulation_cli() -> None:
    assert importlib.util.find_spec("aisim.main") is not None
    assert importlib.util.find_spec("aisim.__main__") is not None
    assert importlib.util.find_spec("aisim.replay.__main__") is None
    assert importlib.util.find_spec("aisim.sweeper.__main__") is None
