# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn measured service intervals into the cost tables the engine consumes."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ...config.engine import CostFnConfig

STEADY_FRACTION = 0.9
"""A sample counts toward a concurrency level when its active overlap reaches this share of it."""

MIN_STEADY_SAMPLES = 10


@dataclass(frozen=True)
class Span:
    """One service interval measured inside a worker, executor wait excluded."""

    started_ns: int
    ended_ns: int

    @property
    def service_ms(self) -> float:
        return (self.ended_ns - self.started_ns) / 1e6


def mean_active_concurrency(spans: Sequence[Span]) -> list[float]:
    """Time-averaged number of spans overlapping each span, itself included.

    Bursts drain unevenly, so the concurrency a load generator targeted is not
    the concurrency a sample actually ran under; the overlap integral is.
    """
    result = []
    for span in spans:
        overlap = sum(
            max(0, min(span.ended_ns, other.ended_ns) - max(span.started_ns, other.started_ns)) for other in spans
        )
        result.append(overlap / max(span.ended_ns - span.started_ns, 1))
    return result


def steady_samples(spans: Sequence[Span], target: int) -> list[Span]:
    """Spans that ran at (nearly) the targeted concurrency."""
    return [
        span
        for span, active in zip(spans, mean_active_concurrency(spans), strict=True)
        if active >= STEADY_FRACTION * target
    ]


def mean_service_ms(spans: Iterable[Span]) -> float:
    return statistics.fmean(span.service_ms for span in spans)


def stage_costs(curves: dict[int, Sequence[Span]]) -> tuple[CostFnConfig, list[float]]:
    """Lower per-concurrency service curves of one fixed workload shape.

    Every sample processes the same shape, so the cost is a constant per job and
    sharing shows up as a per-concurrency scale relative to running alone.
    """
    if 1 not in curves:
        raise ValueError("stage curves must include the single-job concurrency")
    means = {}
    for concurrency, spans in sorted(curves.items()):
        steady = steady_samples(spans, concurrency)
        if len(steady) < MIN_STEADY_SAMPLES:
            raise ValueError(
                f"concurrency {concurrency} has {len(steady)} steady samples; at least {MIN_STEADY_SAMPLES} are needed"
            )
        means[concurrency] = mean_service_ms(steady)
    capacity = max(means)
    if set(means) != set(range(1, capacity + 1)):
        raise ValueError("stage curves must cover every concurrency from 1 to the worker count")
    alone = means[1]
    return CostFnConfig(const_ms=alone), [means[c] / alone for c in range(1, capacity + 1)]
