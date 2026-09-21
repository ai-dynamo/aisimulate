# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lowering a worker recording: steady samples per level become one stage cost plus a sharing scale."""

import pytest

from aisimulate.vl.collect.lower import frontend_row, stage_costs
from aisimulate.vl.collect.samples import Span
from aisimulate.vl.table import SGLANG_REVISION, RowIdentity, Shape

pytestmark = pytest.mark.unit


def _level(concurrency: int, service_ms: float, rounds: int = 12) -> list[list[int]]:
    """`rounds` back-to-back rounds of `concurrency` fully overlapping jobs."""
    spans = []
    for round_index in range(rounds):
        start = round_index * int(service_ms * 1e6) * 2
        spans.extend([start, start + int(service_ms * 1e6)] for _ in range(concurrency))
    return spans


def test_stage_costs_take_the_alone_service_and_scale_by_sharing():
    curves = {1: [Span(*s) for s in _level(1, 10.0)], 2: [Span(*s) for s in _level(2, 15.0)]}
    cost, scale = stage_costs(curves, capacity=2)
    assert cost.const_ms == pytest.approx(10.0)
    assert scale == pytest.approx([1.0, 1.5])
    # A drained level is not steady: two jobs that barely overlap do not count for concurrency 2.
    curves[2] = [
        span
        for base in range(0, 6 * 40_000_000, 40_000_000)
        for span in (Span(base, base + 10_000_000), Span(base + 9_000_000, base + 19_000_000))
    ]
    with pytest.raises(ValueError, match="concurrency 2 has 0 steady samples"):
        stage_costs(curves, capacity=2)


def test_python_recordings_lower_to_pool_loop_and_receive_stages():
    recording = {
        "frontend": "python",
        "workers": 2,
        "levels": {"1": _level(1, 20.0), "2": _level(2, 24.0)},
        "tm_loop_ms": 3.5,
        "receive_ms": 7.25,
        "provenance": {"io_workers": 16},
    }
    row = frontend_row(
        recording,
        identity=RowIdentity(cpu="c", sglang_revision=SGLANG_REVISION, model="m", frontend="python"),
        shape=Shape(height=480, width=480, text_tokens=128),
    )
    assert [(s.resource, s.workers, s.cost.const_ms) for s in row.stages] == [
        ("pool", 2, pytest.approx(20.0)),
        ("tm_loop", 1, 3.5),
        ("pool", 1, 7.25),
    ]
    assert row.stages[0].concurrency_scale == pytest.approx([1.0, 1.2])
    assert row.provenance == {"io_workers": 16}
