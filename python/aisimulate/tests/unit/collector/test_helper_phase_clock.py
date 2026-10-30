# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from collector.helper import PhaseClock, render_phase_totals

pytestmark = pytest.mark.unit


def test_phase_clock_accumulates_phases_and_point_totals():
    # The rendered text only feeds human-read log lines; this is a smoke test of the accounting.
    clock = PhaseClock()
    clock.mark("build")
    clock.mark("build")  # a second slice accumulates into the same phase
    clock.absorb({"warmup": 0.5, "timed": 0.25}, prefix="bench_")
    clock.absorb(None)
    assert clock.phases["bench_warmup"] == 0.5 and clock.phases["bench_timed"] == 0.25
    assert clock.total() >= clock.phases["build"]
    totals: dict[str, float] = {}
    clock.add_to(totals)
    clock.add_to(totals)
    assert totals["points"] == 2 and totals["bench_timed"] == 0.5
    assert isinstance(clock.render(), str) and isinstance(render_phase_totals(totals), str)


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
