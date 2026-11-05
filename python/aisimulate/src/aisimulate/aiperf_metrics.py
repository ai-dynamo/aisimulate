# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf-compatible request metrics, separate from native token-gap metrics.

Definitions: ai-dynamo/aiperf at b0276e19398169eb450f5a79db28509dd4bcb48f,
src/aiperf/metrics/types/{ttft_metric,request_latency_metric,
inter_token_latency_metric,output_token_throughput_metrics}.py (Apache-2.0).
See THIRD_PARTY_NOTICES.md for attribution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class AIPerfMetric(str, Enum):
    TTFT_AVG = "time_to_first_token.avg"
    TTFT_P50 = "time_to_first_token.p50"
    TTFT_P90 = "time_to_first_token.p90"
    TTFT_P95 = "time_to_first_token.p95"
    TTFT_P99 = "time_to_first_token.p99"
    LATENCY_AVG = "request_latency.avg"
    LATENCY_P50 = "request_latency.p50"
    LATENCY_P90 = "request_latency.p90"
    LATENCY_P95 = "request_latency.p95"
    LATENCY_P99 = "request_latency.p99"
    ITL_AVG = "inter_token_latency.avg"
    ITL_P50 = "inter_token_latency.p50"
    ITL_P90 = "inter_token_latency.p90"
    ITL_P95 = "inter_token_latency.p95"
    ITL_P99 = "inter_token_latency.p99"
    USER_TPS_AVG = "output_token_throughput_per_user.avg"
    USER_TPS_P50 = "output_token_throughput_per_user.p50"
    USER_TPS_P90 = "output_token_throughput_per_user.p90"
    USER_TPS_P95 = "output_token_throughput_per_user.p95"
    USER_TPS_P99 = "output_token_throughput_per_user.p99"

    @property
    def maximize(self) -> bool:
        return self.value.startswith("output_token_throughput_per_user.")

    @property
    def report_key(self) -> str:
        return f"aiperf.{self.value}"

    @property
    def count_key(self) -> str:
        return f"aiperf.{self.value.rsplit('.', 1)[0]}.count"


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
    """Request AIPerf metrics from native per-request records.

    ``constraints`` gate candidates only when strict aggregate filtering is on.
    """

    model_config = ConfigDict(extra="forbid")

    per_chunk_usage: bool = Field(default=False, strict=True)
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


def summarize_requests(records: object, *, per_chunk_usage: bool = False) -> dict[str, float]:
    """Match ordinary AIPerf metrics, excluding errors and unfinished requests.

    Latency units are milliseconds. Percentiles use NumPy's linear interpolation.
    The failure-inflated ``adj_*`` metrics are separate in the pinned AIPerf version.
    """
    import numpy as np

    if not isinstance(records, list):
        raise ValueError("AIPerf metrics require native per-request records")
    samples: dict[str, list[float]] = {
        "time_to_first_token": [],
        "request_latency": [],
        "inter_token_latency": [],
        "output_token_throughput_per_user": [],
    }
    for record in records:
        if not isinstance(record, dict) or record.get("terminal_status") not in {
            "completed",
            "failed",
            "canceled",
            "rejected",
        }:
            raise ValueError("AIPerf metrics require records with terminal_status")
        if record["terminal_status"] != "completed":
            continue
        output = record.get("output_length")
        if type(output) is not int or output < 0:
            raise ValueError("AIPerf metrics require a nonnegative integer output_length")
        if output == 0:
            continue
        times = [record.get(key) for key in ("arrival_time_ms", "first_token_ms", "last_token_ms")]
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
            raise ValueError("AIPerf metrics require finite arrival and content-token timestamps")
        arrival, first, last = times
        if not arrival <= first <= last:
            raise ValueError("AIPerf content-token timestamps must be ordered")
        samples["time_to_first_token"].append(first - arrival)
        samples["request_latency"].append(last - arrival)
        if output < 2:
            continue
        first_chunk = record.get("first_content_chunk_tokens") if per_chunk_usage else None
        if first_chunk is not None and type(first_chunk) is not int:
            raise ValueError("first_content_chunk_tokens must be an integer or null")
        # AIPerf falls back to one when usage is absent, zero, or inconsistent.
        divisor = output - first_chunk if first_chunk is not None and 0 < first_chunk < output else output - 1
        itl = (last - first) / divisor
        samples["inter_token_latency"].append(itl)
        if itl > 0:
            samples["output_token_throughput_per_user"].append(1000.0 / itl)

    metrics = {}
    for name, values in samples.items():
        prefix = f"aiperf.{name}"
        metrics[f"{prefix}.count"] = float(len(values))
        if values:
            metrics[f"{prefix}.avg"] = float(np.mean(values))
            for percentile, value in zip((50, 90, 95, 99), np.percentile(values, [50, 90, 95, 99]), strict=True):
                metrics[f"{prefix}.p{percentile}"] = float(value)
    return metrics
