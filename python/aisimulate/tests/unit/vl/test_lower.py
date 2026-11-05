# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lowering a worker recording: steady samples per level become one service time plus a sharing scale."""

import pytest

from aisimulate.config.engine import FrontendMeasurementConfig
from aisimulate.vl.collect.lower import environment, frontend_row, stage_costs
from aisimulate.vl.collect.samples import Span

pytestmark = pytest.mark.unit

ENVIRONMENT = {"cpu": "c", "sglang_version": "0.5.19", "python": "3.10.12"}


def _level(concurrency: int, service_ms: float, rounds: int = 12) -> list[list[int]]:
    """`rounds` back-to-back rounds of `concurrency` fully overlapping jobs."""
    spans = []
    for round_index in range(rounds):
        start = round_index * int(service_ms * 1e6) * 2
        spans.extend([start, start + int(service_ms * 1e6)] for _ in range(concurrency))
    return spans


def test_stage_costs_take_the_alone_service_and_scale_by_sharing():
    curves = {1: [Span(*s) for s in _level(1, 10.0)], 2: [Span(*s) for s in _level(2, 15.0)]}
    service_ms, scale = stage_costs(curves, capacity=2)
    assert service_ms == pytest.approx(10.0)
    assert scale == pytest.approx([1.0, 1.5])
    # A drained level is not steady: two jobs that barely overlap do not count for concurrency 2.
    curves[2] = [
        span
        for base in range(0, 6 * 40_000_000, 40_000_000)
        for span in (Span(base, base + 10_000_000), Span(base + 9_000_000, base + 19_000_000))
    ]
    with pytest.raises(ValueError, match="concurrency 2 has 0 steady samples"):
        stage_costs(curves, capacity=2)


def test_recordings_lower_to_each_frontends_stages():
    levels = {"1": _level(1, 20.0), "2": _level(2, 24.0)}
    python = frontend_row(
        {
            "frontend": "python",
            "workers": 2,
            "levels": levels,
            "send_ms": 3.5,
            "receive_ms": 7.25,
            "environment": ENVIRONMENT,
            "provenance": {"text_tokens": 128},
        },
        FrontendMeasurementConfig(model="m", frontend="python", feature_transport="shm", height=480, width=480),
    )
    assert [(s.workers, s.service_ms) for s in python.stages] == [(2, pytest.approx(20.0)), (1, 3.5), (1, 7.25)]
    assert python.stages[0].concurrency_scale == pytest.approx([1.0, 1.2])
    assert python.provenance == {"text_tokens": 128}
    rust = frontend_row(
        {"frontend": "rust", "workers": 2, "levels": levels, "receive_ms": 0.3, "environment": ENVIRONMENT},
        FrontendMeasurementConfig(model="m", frontend="rust", feature_transport="inline", height=480, width=480),
    )
    assert [(s.workers, s.service_ms) for s in rust.stages] == [(2, pytest.approx(20.0)), (1, 0.3)]
    assert environment({"environment": ENVIRONMENT}).cpu == "c"
