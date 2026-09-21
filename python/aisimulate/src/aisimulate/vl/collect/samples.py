# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-interval statistics shared by the collector and its worker script.

Standard library only and Python 3.10 compatible: the worker script imports
this module by path under the serving host's SGLang interpreter.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

STEADY_FRACTION = 0.9
"""A sample counts toward a concurrency level when its active overlap reaches this share of it."""

MIN_STEADY_SAMPLES = 10


@dataclass(frozen=True)
class Span:
    """One service interval measured inside a worker, queue wait excluded."""

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


def steady_means(curves: dict[int, Sequence[Span]], capacity: int) -> dict[int, float]:
    """Mean steady service time per concurrency level from one to `capacity`.

    Every level needs `MIN_STEADY_SAMPLES` steady samples, or the engine would
    reject the table or silently run unmeasured levels.
    """
    if 1 not in curves:
        raise ValueError("stage curves must include the single-job concurrency")
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
    return means
