# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower the Rust frontend timing recorded by `tools/frontend/sglang-v0.5.19-timing.patch`.

The multimodal worker's service time is the `rust_worker` boundary span the
patch writes around `MmWorker::run`'s per-request `process` call: payload
conversion, fetch, content hash, decode, patchify, tokenization/token layout,
M-RoPE, feature packing, the optional shared-memory copy and the sidecar park.
The span is recorded only with `AIS_VL_BOUNDARY_TRACE=1`; the legacy rows that
carry `image_timings_ns` cover the decode/patchify part alone and are kept as
provenance, not as the worker occupancy.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config.engine import FrontendPredictionConfig, FrontendStageConfig
from .samples import Span, curves_by_active_concurrency, stage_costs

WORKER_OP = "rust_worker"
PROCESSOR_LABEL = "rust_worker@sglang-v0.5.19-timing.patch"
"""Stable processor label of the Rust pipeline in a profile identity."""


def load_worker_spans(path: str | Path) -> list[Span]:
    """Per-request `rust_worker` occupancy intervals; failed requests are not service."""
    spans = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "boundary_span" or row.get("op") != WORKER_OP:
            continue
        if row.get("metadata", {}).get("ok") is False:
            continue
        spans.append(Span(int(row["started_ns"]), int(row["ended_ns"])))
    if not spans:
        raise ValueError(
            f"{path} holds no {WORKER_OP} spans; record the serving run with AIS_VL_BOUNDARY_TRACE=1 "
            "and AIS_MM_TIMING_PATH set"
        )
    return spans


def frontend_from_timing(path: str | Path, *, mm_workers: int) -> FrontendPredictionConfig:
    """One request-unit stage on the multimodal worker pool, scaled by observed sharing.

    Every concurrency from one to `mm_workers` needs steady samples: the engine
    requires one scale entry per worker, and an unmeasured level must be
    reported, not defaulted.
    """
    spans = load_worker_spans(path)
    cost, scale = stage_costs(curves_by_active_concurrency(spans, mm_workers), capacity=mm_workers)
    return FrontendPredictionConfig(
        io_workers=1,
        processor_workers=1,
        mm_workers=mm_workers,
        stages=[FrontendStageConfig(resource="mm_worker", unit="request", cost=cost, concurrency_scale=scale)],
    )
