# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""helper.run_measurement_subprocess: a child that prints progress keeps its worker alive;
a silent one does not, and a timeout keeps the output for diagnosis."""

import os
import subprocess
import sys

import pytest
from collector import helper

pytestmark = pytest.mark.unit


@pytest.fixture
def beats(monkeypatch):
    calls = []
    monkeypatch.setattr(helper, "_WORKER_HEARTBEAT", lambda: calls.append(1))
    return calls


def _child(code):
    return [sys.executable, "-c", code]


def test_progress_beats_and_output_is_returned(beats):
    code = "import time\nfor i in range(3):\n    print('point', i)\n    time.sleep(0.3)\n"
    rc, out = helper.run_measurement_subprocess(_child(code), env=dict(os.environ), cwd=None, poll_sec=0.2)
    assert rc == 0
    assert out.decode().split() == ["point", "0", "point", "1", "point", "2"]
    assert beats  # unbuffered prints grew the output between polls


def test_a_silent_child_does_not_beat(beats):
    rc, _ = helper.run_measurement_subprocess(
        _child("import time; time.sleep(0.8)"), env=dict(os.environ), cwd=None, poll_sec=0.2
    )
    assert rc == 0
    assert beats == []


def test_timeout_kills_the_child_and_keeps_its_output(beats):
    code = "import time\nprint('started')\ntime.sleep(30)\n"
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        helper.run_measurement_subprocess(_child(code), env=dict(os.environ), cwd=None, timeout=0.6, poll_sec=0.2)
    assert b"started" in caught.value.output
