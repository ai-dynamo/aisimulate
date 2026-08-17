# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect AI Simulate tests only where its standalone package is available."""

from importlib.util import find_spec
from pathlib import Path

# Source-only tooling may inspect the repository before building the native
# extension. Ignore runtime suites until the standalone package is available.
try:
    _core_available = find_spec("aisimulate.runner") is not None
except ModuleNotFoundError:
    _core_available = False

try:
    _sweeper_available = find_spec("aisimulate.sweeper.config") is not None
except ModuleNotFoundError:
    _sweeper_available = False

collect_ignore = []
if not _core_available:
    collect_ignore.extend(
        str(Path(__file__).parent / test_file)
        for test_file in (
            "test_aic.py",
            "test_replay_cli.py",
            "test_runner.py",
            "test_traffic.py",
        )
    )
if not _sweeper_available:
    collect_ignore.append(str(Path(__file__).parent / "sweeper"))
