# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower a worker recording into the frontend stages of one host cost table row."""

from __future__ import annotations

from typing import Any

from ...config.engine import FrontendMeasurementConfig, FrontendStageConfig
from ..table import FrontendRow, TableEnvironment
from .samples import Span, steady_means


def stage_costs(curves: dict[int, list[Span]], *, capacity: int) -> tuple[float, list[float]]:
    """Service time of one request alone plus the per-concurrency scale of sharing the pool.

    Every sample processes the same workload, so the service time is one number
    per request and sharing shows up as a scale relative to running alone.
    """
    means = steady_means(curves, capacity)
    alone = means[1]
    return alone, [means[concurrency] / alone for concurrency in range(1, capacity + 1)]


def _curves(levels: dict[str, list[list[int]]]) -> dict[int, list[Span]]:
    return {int(level): [Span(int(started), int(ended)) for started, ended in spans] for level, spans in levels.items()}


def environment(recording: dict[str, Any]) -> TableEnvironment:
    return TableEnvironment.model_validate(recording["environment"])


def frontend_row(recording: dict[str, Any], *, model: str, measurement: FrontendMeasurementConfig) -> FrontendRow:
    """Stages of one frontend from the worker script's recording.

    Python: the multimodal processor path (image decode, HF processor, layout)
    as a pool the width of its IO executor, the tokenizer-manager loop's
    synchronous send as one worker, and the scheduler's per-request receive
    preparation as one worker. Rust: the multimodal worker pool, then receive.
    """
    workers = int(recording["workers"])
    service_ms, scale = stage_costs(_curves(recording["levels"]), capacity=workers)
    stages = [FrontendStageConfig(workers=workers, service_ms=service_ms, concurrency_scale=scale)]
    if recording["frontend"] == "python":
        stages.append(FrontendStageConfig(service_ms=float(recording["send_ms"])))
    stages.append(FrontendStageConfig(service_ms=float(recording["receive_ms"])))
    return FrontendRow(
        model=model,
        measured_for=measurement,
        stages=stages,
        provenance=dict(recording.get("provenance", {})),
    )


STAGE_LABELS = {"python": ("process", "send", "receive"), "rust": ("process", "receive")}
"""Display names of the stages `frontend_row` emits, in order."""
