# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf request metric selectors over the native replay summary.

Native replay reports TTFT, request latency, per-request ITL and user TPS with
the request populations of ai-dynamo/aiperf b0276e19398169eb450f5a79db28509dd4bcb48f.
Native percentiles use the nearest rank; AIPerf interpolates linearly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

# AIPerf metric -> (selector name prefix, native summary suffix, native sample count).
_FAMILIES = {
    "time_to_first_token": ("TTFT", "ttft_ms", "num_ttft_samples"),
    "request_latency": ("LATENCY", "e2e_latency_ms", "num_e2e_latency_samples"),
    "inter_token_latency": ("ITL", "itl_ms", "num_tpot_samples"),
    "output_token_throughput_per_user": (
        "USER_TPS",
        "output_token_throughput_per_user",
        "num_output_token_throughput_per_user_samples",
    ),
}
# AIPerf statistic -> native summary prefix.
_STATS = {
    "avg": "mean",
    "p1": "p1",
    "p5": "p5",
    "p10": "p10",
    "p25": "p25",
    "p50": "median",
    "p75": "p75",
    "p90": "p90",
    "p95": "p95",
    "p99": "p99",
}


class _NativeMetric(str, Enum):
    @property
    def maximize(self) -> bool:
        return self.value.startswith("output_token_throughput_per_user.")

    @property
    def report_key(self) -> str:
        family, stat = self.value.rsplit(".", 1)
        return f"{_STATS[stat]}_{_FAMILIES[family][1]}"

    @property
    def count_key(self) -> str:
        return _FAMILIES[self.value.rsplit(".", 1)[0]][2]


# Members such as TTFT_P50 = "time_to_first_token.p50".
AIPerfMetric = _NativeMetric(
    "AIPerfMetric",
    {f"{short}_{stat.upper()}": f"{family}.{stat}" for family, (short, _, _) in _FAMILIES.items() for stat in _STATS},
)


class MetricConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: AIPerfMetric
    operator: Literal["lte", "gte"]
    threshold: float = Field(strict=True, ge=0, allow_inf_nan=False)

    @field_serializer("metric")
    def _serialize_metric(self, metric: AIPerfMetric) -> str:
        # Saved recommendation configs must be plain JSON even in Python dump mode.
        return metric.value


class AIPerfEvaluation(BaseModel):
    """Aggregate constraints that gate candidates only with strict filtering."""

    model_config = ConfigDict(extra="forbid")

    constraints: list[MetricConstraint] = Field(default_factory=list)


def metric_value(report: Mapping[str, float], metric: AIPerfMetric) -> float:
    """Missing samples never become a zero-latency or zero-throughput result."""
    count, value = report.get(metric.count_key), report.get(metric.report_key)
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, float))
        or not math.isfinite(count)
        or count <= 0
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{metric.value} is missing, invalid, or has no qualifying samples")
    return float(value)


def constraint_violations(report: Mapping[str, float], constraints: Sequence[MetricConstraint]) -> tuple[str, ...]:
    violations = []
    for constraint in constraints:
        try:
            value = metric_value(report, constraint.metric)
        except ValueError as exc:
            violations.append(str(exc))
            continue
        if (constraint.operator == "lte" and value > constraint.threshold) or (
            constraint.operator == "gte" and value < constraint.threshold
        ):
            violations.append(
                f"{constraint.metric.value}={value:g} violates {constraint.operator} {constraint.threshold:g}"
            )
    return tuple(violations)
