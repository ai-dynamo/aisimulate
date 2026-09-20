# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower the Rust frontend timing recorded by `tools/frontend/sglang-v0.5.19-timing.patch`.

Each JSONL row is one request served by a multimodal worker: `image_timings_ns`
holds decode and processor nanoseconds per image, `started_ns`/`ended_ns` the
worker interval. The request's service time is the sum over its images; rows
without `image_timings_ns` are boundary diagnostics and are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config.engine import FrontendPredictionConfig, FrontendStageConfig
from .samples import Span, mean_active_concurrency, stage_costs


def load_worker_spans(path: str | Path) -> list[Span]:
    spans = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        timings = row.get("image_timings_ns")
        if timings is None:
            continue
        service_ns = sum(int(decode) + int(process) for decode, process in timings)
        started = int(row["started_ns"])
        spans.append(Span(started, started + service_ns))
    if not spans:
        raise ValueError(f"{path} holds no worker service rows (image_timings_ns)")
    return spans


def frontend_from_timing(path: str | Path, *, mm_workers: int) -> FrontendPredictionConfig:
    """One request-unit stage on the multimodal worker pool, scaled by observed sharing."""
    spans = load_worker_spans(path)
    curves: dict[int, list[Span]] = {}
    for span, active in zip(spans, mean_active_concurrency(spans), strict=True):
        curves.setdefault(min(max(round(active), 1), mm_workers), []).append(span)
    cost, scale = stage_costs(curves)
    return FrontendPredictionConfig(
        io_workers=1,
        processor_workers=1,
        mm_workers=mm_workers,
        stages=[FrontendStageConfig(resource="mm_worker", unit="request", cost=cost, concurrency_scale=scale)],
    )
