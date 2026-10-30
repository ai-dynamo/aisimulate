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


def _fake_torch(monkeypatch, allocated):
    import sys
    import types

    state = {"allocated": allocated}
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: state["allocated"],
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    return state


def test_periodic_gc_collects_every_n_calls(monkeypatch):
    import gc

    from collector.helper import PeriodicGarbageCollector

    _fake_torch(monkeypatch, allocated=1000)
    calls = []
    monkeypatch.setattr(gc, "collect", lambda: calls.append(1) or 0)
    pgc = PeriodicGarbageCollector(every=4, growth_bytes=10**9)
    results = [pgc.maybe_collect() for _ in range(9)]
    assert results == [False, False, False, True, False, False, False, True, False]
    assert len(calls) == 2


def test_periodic_gc_collects_early_on_memory_growth(monkeypatch):
    import gc

    from collector.helper import PeriodicGarbageCollector

    state = _fake_torch(monkeypatch, allocated=1000)
    calls = []
    monkeypatch.setattr(gc, "collect", lambda: calls.append(1) or 0)
    pgc = PeriodicGarbageCollector(every=1000, growth_bytes=500)
    assert pgc.maybe_collect() is False  # first call sets the floor
    state["allocated"] = 1400
    assert pgc.maybe_collect() is False  # within the growth budget
    state["allocated"] = 1600
    assert pgc.maybe_collect() is True  # grew past floor + growth_bytes
    assert len(calls) == 1
    assert pgc.calls == 0
    assert pgc.floor == 1000  # the floor never rises


def test_periodic_gc_without_cuda_counts_calls_only(monkeypatch):
    import gc
    import sys
    import types

    from collector.helper import PeriodicGarbageCollector

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    calls = []
    monkeypatch.setattr(gc, "collect", lambda: calls.append(1) or 0)
    pgc = PeriodicGarbageCollector(every=2)
    assert [pgc.maybe_collect() for _ in range(4)] == [False, True, False, True]
    assert len(calls) == 2
