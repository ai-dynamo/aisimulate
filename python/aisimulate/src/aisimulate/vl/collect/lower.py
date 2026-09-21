# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower a worker recording into the frontend stages of one host cost table row."""

from __future__ import annotations

from typing import Any

from ...config.engine import CostFnConfig, FrontendStageConfig
from ..table import FrontendRow, RowIdentity, Shape
from .samples import Span, steady_means


def stage_costs(curves: dict[int, list[Span]], *, capacity: int) -> tuple[CostFnConfig, list[float]]:
    """Constant service cost of one request plus the per-concurrency scale of sharing the resource.

    Every sample processes the same shape, so the cost is a constant per job and
    sharing shows up as a scale relative to running alone.
    """
    means = steady_means(curves, capacity)
    alone = means[1]
    return CostFnConfig(const_ms=alone), [means[c] / alone for c in range(1, capacity + 1)]


def _curves(levels: dict[str, list[list[int]]]) -> dict[int, list[Span]]:
    return {int(level): [Span(int(started), int(ended)) for started, ended in spans] for level, spans in levels.items()}


def frontend_row(recording: dict[str, Any], *, identity: RowIdentity, shape: Shape) -> FrontendRow:
    """Stages of one frontend from the worker script's recording.

    Python: a pool stage for the multimodal processor path (image decode, HF
    processor, layout) whose width is the highest measured concurrency, a
    tokenizer-manager loop stage for the synchronous send continuation, and a
    single-worker pool for the scheduler's per-request receive preparation.
    Rust: one pool stage the width of the multimodal worker pool.
    """
    workers = int(recording["workers"])
    cost, scale = stage_costs(_curves(recording["levels"]), capacity=workers)
    stages = [FrontendStageConfig(resource="pool", workers=workers, cost=cost, concurrency_scale=scale)]
    if recording["frontend"] == "python":
        stages.append(
            FrontendStageConfig(resource="tm_loop", cost=CostFnConfig(const_ms=float(recording["tm_loop_ms"])))
        )
        stages.append(
            FrontendStageConfig(resource="pool", workers=1, cost=CostFnConfig(const_ms=float(recording["receive_ms"])))
        )
    return FrontendRow(identity=identity, shape=shape, stages=stages, provenance=dict(recording.get("provenance", {})))
