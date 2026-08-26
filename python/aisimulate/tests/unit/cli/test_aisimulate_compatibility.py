# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command identity contract for the application shipped by AISimulate."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.unit


def test_aisimulate_owns_the_single_simulation_cli() -> None:
    assert importlib.util.find_spec("aisimulate.main") is not None
    assert importlib.util.find_spec("aisimulate.__main__") is not None
    assert importlib.util.find_spec("aisimulate.replay.__main__") is None
    assert importlib.util.find_spec("aisimulate.sweeper.__main__") is None
