# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command identity contract for the application shipped by AISimulate."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.unit


def test_aisimulate_has_no_top_level_application_cli_modules() -> None:
    assert importlib.util.find_spec("aisimulate.main") is None
    assert importlib.util.find_spec("aisimulate.__main__") is None
