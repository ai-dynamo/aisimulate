# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for repeated real hybrid requests and native dispatch."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_repeated_real_hybrid_request_lifecycle():
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["AIC_FPM_DSV41_PRODUCER"] = str(root / "collector/fpm_forward/runtime/dsv41/dsv41_scheduler.py")
    environment["AIC_FPM_GLM53FLASH_PRODUCER"] = str(
        root / "collector/fpm_forward/runtime/glm53flash/glm53flash_scheduler.py"
    )
    environment["AIC_FPM_NATIVE_ARTIFACT"] = str(root / "collector/fpm_forward/native_artifact.py")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("fixtures") / "glm53flash_producer_lifecycle.py")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
