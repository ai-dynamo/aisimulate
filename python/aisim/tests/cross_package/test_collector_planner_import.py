# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The collector's op planner imports on every interpreter the package supports.

The Python compatibility job runs this directory on each supported interpreter.
The planner is not part of the wheel, so this imports it from the checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_op_planner_imports_from_the_checkout() -> None:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PACKAGE_ROOT), str(PACKAGE_ROOT / "src")]))
    result = subprocess.run(
        [sys.executable, "-c", "import collector.op_plans; print(len(collector.op_plans.OP_PLANS))"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 0
