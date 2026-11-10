# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute ordinary AIPerf request metrics with pinned AIPerf classes; print JSON."""

import json
import sys
from types import SimpleNamespace

import numpy as np
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict, metric_result_from_array
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_sequence_length_metric import OutputSequenceLengthMetric
from aiperf.metrics.types.output_token_throughput_metrics import OutputTokenThroughputPerUserMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric

CLASSES = [TTFTMetric, RequestLatencyMetric, InterTokenLatencyMetric, OutputTokenThroughputPerUserMetric]


def aiperf_summary(rows, per_chunk):
    samples = {cls.tag: [] for cls in CLASSES}
    for first, last, output, chunk, status in rows:
        responses = (
            []
            if first is None
            else [SimpleNamespace(perf_ns=round(first * 1e6)), SimpleNamespace(perf_ns=round(last * 1e6))]
        )
        record = SimpleNamespace(
            valid=status == "completed",
            request=SimpleNamespace(start_perf_ns=0),
            start_perf_ns=0,
            content_responses=responses,
            token_counts=SimpleNamespace(first_content_chunk_tokens=chunk if per_chunk else None),
        )
        if not record.valid:
            continue  # AIPerf routes invalid records to error metrics only.
        metrics = MetricRecordDict({OutputSequenceLengthMetric.tag: output})
        for cls in CLASSES:
            try:
                value = cls().parse_record(record, metrics)
            except NoMetricValue:
                continue
            metrics[cls.tag] = value
            samples[cls.tag].append(value if cls is OutputTokenThroughputPerUserMetric else value / 1e6)
    out = {}
    for name, values in samples.items():
        out[f"aiperf.{name}.count"] = float(len(values))
        if values:
            arr = np.array(values, dtype=float)
            res = metric_result_from_array(name, name, "ms", arr, float(arr.sum()))
            for stat in ("avg", "p50", "p90", "p95", "p99"):
                out[f"aiperf.{name}.{stat}"] = float(getattr(res, stat))
    return out


if __name__ == "__main__":
    cases = json.load(sys.stdin)
    print(json.dumps([aiperf_summary(c["rows"], c["per_chunk"]) for c in cases], indent=1))
