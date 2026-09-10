# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run producer state-machine stubs in an isolated process."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_real_kv_producer_lifecycle():
    root = Path(__file__).resolve().parents[3]
    runtime = root / "collector/fpm_forward/runtime/dsv41"
    environment = dict(os.environ)
    environment["AIC_FPM_DSV41_PRODUCER"] = str(runtime / "dsv41_scheduler.py")
    environment["AIC_FPM_NATIVE_ARTIFACT"] = str(root / "collector/fpm_forward/native_artifact.py")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fixtures") / "dsv41_producer_lifecycle.py")],
        env=environment,
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )
