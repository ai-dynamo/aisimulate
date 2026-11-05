# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The global conftest keeps CLI logging setup from leaking into later tests."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_INNER_TESTS = """
import logging

from aisimulate.logging_utils import setup_logging


def test_cli_logging_setup_under_capsys(capsys):
    setup_logging()


def test_later_test_sees_default_root_logging(capsys):
    logging.getLogger("aisimulate").warning("record")
    assert capsys.readouterr().err == ""
    assert logging.getLogger().level == logging.WARNING
"""


def test_cli_logging_setup_under_capsys_does_not_leak_into_later_tests(tmp_path):
    # A separate session runs both tests in a fixed order with the real global
    # conftest, so this check does not depend on the order of this suite. An
    # exported PYTEST_ADDOPTS, such as --log-level, must not change that session.
    shutil.copy(Path(__file__).parents[1] / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_inner.py").write_text(_INNER_TESTS)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
