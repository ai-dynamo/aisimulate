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


def union_ms(spans: Iterable[Span]) -> float:
    """Length of the interval union: time during which at least one span was running.

    Pool work of one request runs in parallel, so its spans overlap; summing
    them would subtract the same wall-clock interval more than once.
    """
    total = 0
    current: tuple[int, int] | None = None
    for started, ended in sorted((span.started_ns, span.ended_ns) for span in spans):
        if current is not None and started <= current[1]:
            current = (current[0], max(current[1], ended))
            continue
        if current is not None:
            total += current[1] - current[0]
        current = (started, ended)
    if current is not None:
        total += current[1] - current[0]
    return total / 1e6


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


def curves_by_active_concurrency(spans: Sequence[Span], capacity: int) -> dict[int, list[Span]]:
    """Group observed spans by the (rounded) concurrency they ran under, clamped to `capacity`."""
    curves: dict[int, list[Span]] = {}
    for span, active in zip(spans, mean_active_concurrency(spans), strict=True):
        curves.setdefault(min(max(round(active), 1), capacity), []).append(span)
    return curves


def mean_service_ms(spans: Iterable[Span]) -> float:
    return statistics.fmean(span.service_ms for span in spans)


def stage_costs(curves: dict[int, Sequence[Span]], *, capacity: int | None = None) -> tuple[CostFnConfig, list[float]]:
    """Lower per-concurrency service curves of one fixed workload shape.

    Every sample processes the same shape, so the cost is a constant per job and
    sharing shows up as a per-concurrency scale relative to running alone. The
    scale must cover every level from one to `capacity` (the resource's worker
    count; the highest measured level when omitted) with steady samples, or the
    engine would reject the table or silently run unmeasured levels.
    """
    if 1 not in curves:
        raise ValueError("stage curves must include the single-job concurrency")
    capacity = max(curves) if capacity is None else capacity
    problems = []
    means = {}
    for concurrency in range(1, capacity + 1):
        steady = steady_samples(curves.get(concurrency, ()), concurrency)
        if len(steady) < MIN_STEADY_SAMPLES:
            problems.append(f"concurrency {concurrency} has {len(steady)} steady samples")
            continue
        means[concurrency] = mean_service_ms(steady)
    if problems:
        raise ValueError(
            f"stage curves must have at least {MIN_STEADY_SAMPLES} steady samples at every concurrency "
            f"from 1 to {capacity}: " + "; ".join(problems)
        )
    alone = means[1]
    return CostFnConfig(const_ms=alone), [means[c] / alone for c in range(1, capacity + 1)]
