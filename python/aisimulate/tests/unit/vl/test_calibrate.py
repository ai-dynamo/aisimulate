# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host calibration lowering: measured intervals become engine cost tables."""

import json

import pytest

from aisimulate.vl.calibrate.rust_frontend import frontend_from_timing
from aisimulate.vl.calibrate.samples import Span, mean_active_concurrency, stage_costs

pytestmark = pytest.mark.unit


def _burst(concurrency: int, service_ms: float, repeats: int) -> list[Span]:
    """`repeats` back-to-back bursts of `concurrency` jobs that overlap exactly."""
    spans = []
    for burst in range(repeats):
        start = burst * int(service_ms * 1e6) * 2
        spans.extend(Span(start, start + int(service_ms * 1e6)) for _ in range(concurrency))
    return spans


def test_stage_costs_take_the_alone_service_and_scale_by_sharing():
    curves = {1: _burst(1, 4.0, 12), 2: _burst(2, 6.0, 6)}
    assert mean_active_concurrency(curves[2]) == [2.0] * 12
    cost, scale = stage_costs(curves)
    assert cost.const_ms == pytest.approx(4.0)
    assert scale == pytest.approx([1.0, 1.5])
    # A drained burst is not steady: two jobs that barely overlap do not count for concurrency 2.
    drained = [
        span
        for burst in range(6)
        for span in (
            Span(burst * 4_000_000, burst * 4_000_000 + 1_000_000),
            Span(burst * 4_000_000 + 900_000, burst * 4_000_000 + 1_900_000),
        )
    ]
    with pytest.raises(ValueError, match="steady samples"):
        stage_costs({1: curves[1], 2: drained})


def test_rust_timing_rows_lower_to_one_worker_stage(tmp_path):
    rows = []
    for burst in range(12):
        start = burst * 10_000_000
        rows.append({"started_ns": start, "ended_ns": start + 3_000_000, "image_timings_ns": [[1_000_000, 2_000_000]]})
        rows.append({"event": "boundary_span", "op": "rust_hash", "started_ns": start, "ended_ns": start + 10})
    path = tmp_path / "timing.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    frontend = frontend_from_timing(path, mm_workers=8)
    (stage,) = frontend.stages
    assert (stage.resource, stage.unit) == ("mm_worker", "request")
    assert stage.cost.const_ms == pytest.approx(3.0)
    assert stage.concurrency_scale == [1.0]
