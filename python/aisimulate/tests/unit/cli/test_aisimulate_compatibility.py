# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility contract for the executable shipped by the AISimulate wheel."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from aiconfigurator.main import main as aiconfigurator_main
from aisimulate.main import main as aisimulate_main

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("argv", [["--help"], ["version"]])
def test_aisimulate_delegates_to_the_complete_aic_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) if argv == ["--help"] else nullcontext():
        aisimulate_main(argv)
    aisimulate_output = capsys.readouterr()

    with pytest.raises(SystemExit) if argv == ["--help"] else nullcontext():
        aiconfigurator_main(argv)
    aiconfigurator_output = capsys.readouterr()

    assert aisimulate_output == aiconfigurator_output
