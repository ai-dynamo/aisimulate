# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importing support-matrix scripts must preserve the caller's package search path."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("script", ["compare_support_matrix", "generate_support_matrix"])
def test_script_import_preserves_package_search_path(script):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys

original_path = list(sys.path)
importlib.import_module(sys.argv[1])
assert sys.path == original_path, (original_path, sys.path)
""",
            f"tools.support_matrix.{script}",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
