# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from collector.helper import PhaseClock, render_phase_totals

pytestmark = pytest.mark.unit


def test_phase_clock_marks_absorbs_and_renders():
    clock = PhaseClock()
    time.sleep(0.01)
    clock.mark("build")
    clock.mark("build")  # a second slice accumulates into the same phase
    clock.absorb({"warmup": 0.5, "timed": 0.25}, prefix="bench_")
    clock.absorb(None)
    assert clock.phases["build"] >= 0.01
    assert clock.phases["bench_warmup"] == 0.5
    assert clock.phases["bench_timed"] == 0.25
    rendered = clock.render()
    assert rendered.startswith("build=")
    assert "bench_warmup=0.50s" in rendered
    assert rendered.split()[-1].startswith("total=") and rendered.endswith("s")
    assert clock.total() >= clock.phases["build"]


def test_phase_totals_accumulate_points_and_render_without_the_counter():
    clock = PhaseClock()
    clock.mark("build")
    clock.absorb({"timed": 0.25}, prefix="bench_")
    totals: dict[str, float] = {}
    clock.add_to(totals)
    clock.add_to(totals)
    assert totals["points"] == 2
    assert totals["bench_timed"] == 0.5
    assert totals["total"] >= 2 * totals["build"]
    text = render_phase_totals(totals)
    assert text.startswith("2 point(s): ")
    assert "bench_timed=0.5s" in text
    assert "points=" not in text
    assert render_phase_totals({}) == "0 point(s): "
