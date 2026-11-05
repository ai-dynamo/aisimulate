# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic cases for the AIPerf definitions pinned in THIRD_PARTY_NOTICES.md."""

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisimulate.aiperf_metrics import AIPerfMetric, metric_value, summarize_requests
from aisimulate.config.cli import CoreRecommendationConfig
from aisimulate.recommend import _goal
from aisimulate.sweeper.config import Candidate, OptimizationGoal
from aisimulate.sweeper.score import aggregate_sla_violations, analyze_candidates, make_candidate


def record(first, last, output, *, status="completed", first_chunk=1):
    return {
        "arrival_time_ms": 0.0,
        "first_token_ms": first,
        "last_token_ms": last,
        "output_length": output,
        "first_content_chunk_tokens": first_chunk,
        "terminal_status": status,
    }


def test_request_populations_and_linear_percentiles():
    rows = [record(10, 20, 2), record(20, 110, 4), record(10, 10, 1), record(5, 5, 2)]
    rows += [record(0, 0, 100, status=status) for status in ("failed", "canceled", "rejected")]
    rows.append(record(None, None, 0))
    result = summarize_requests(rows)
    # Every row contributes at most one sample. Zero ITL has no reciprocal TPS.
    expected = {
        "time_to_first_token": (4, 11.25, 10, 17, 18.5, 19.7),
        "request_latency": (4, 36.25, 15, 83, 96.5, 107.3),
        "inter_token_latency": (3, 40 / 3, 10, 26, 28, 29.6),
        "output_token_throughput_per_user": (2, 200 / 3, 200 / 3, 280 / 3, 290 / 3, 298 / 3),
    }
    for name, values in expected.items():
        for stat, value in zip(("count", "avg", "p50", "p90", "p95", "p99"), values, strict=True):
            assert result[f"aiperf.{name}.{stat}"] == pytest.approx(value)


PINNED = json.loads((Path(__file__).parent / "fixtures" / "aiperf_b0276e19_request_metrics.json").read_text())


@pytest.mark.parametrize("case", PINNED["cases"], ids=lambda c: f"{len(c['rows'])}rows-chunk{c['per_chunk_usage']}")
def test_matches_pinned_aiperf_classes(case):
    # Expected values come from the pinned AIPerf metric classes (see the fixture's generator).
    rows = [
        record(first, last, output, status=status, first_chunk=chunk)
        for first, last, output, chunk, status in case["rows"]
    ]
    actual = summarize_requests(rows, per_chunk_usage=case["per_chunk_usage"])
    assert actual.keys() == case["expected"].keys()
    for key, expected in case["expected"].items():
        assert actual[key] == pytest.approx(expected, rel=1e-12, abs=1e-12), key


@pytest.mark.parametrize("first_chunk, expected", [(None, 7.5), (-1, 7.5), (0, 7.5), (2, 10), (5, 7.5)])
def test_first_chunk_matches_aiperf_usage_and_fallback(first_chunk, expected):
    rows = [record(10, 40, 5, first_chunk=first_chunk)]
    assert summarize_requests(rows)[AIPerfMetric.ITL_AVG.report_key] == 7.5
    with_usage = summarize_requests(rows, per_chunk_usage=True)
    assert with_usage[AIPerfMetric.ITL_AVG.report_key] == expected
    assert with_usage[AIPerfMetric.USER_TPS_AVG.report_key] == 1000 / expected


@pytest.mark.parametrize("rows", [None, [{}], [record(20, 10, 2)], [record(math.nan, 20, 2)]])
def test_invalid_request_evidence_is_rejected(rows):
    with pytest.raises(ValueError):
        summarize_requests(rows)


def test_no_samples_are_not_zero_latency():
    metrics = summarize_requests([])
    assert metrics[AIPerfMetric.TTFT_AVG.count_key] == 0
    assert AIPerfMetric.TTFT_AVG.report_key not in metrics
    with pytest.raises(ValueError, match="no qualifying samples"):
        metric_value(metrics, AIPerfMetric.TTFT_AVG)


@pytest.mark.parametrize("metric", list(AIPerfMetric))
def test_all_selectors_score_and_round_trip(metric):
    report = summarize_requests([record(10, 20, 2), record(20, 110, 4)])
    goal = OptimizationGoal(target=metric)
    assert OptimizationGoal.model_validate_json(goal.model_dump_json()) == goal
    candidate = make_candidate({"used_gpus": 1}, report, goal.target)
    sign = 1 if metric.maximize else -1
    assert candidate.score == sign * report[metric.report_key]
    assert Candidate.model_validate_json(candidate.model_dump_json()).metrics == candidate.metrics


@pytest.mark.parametrize("target", ["throughput", "goodput", "output_token_throughput_per_user.p50"])
def test_scoring_is_independent_of_strict_rejection(target):
    report = {
        **summarize_requests([record(10, 30, 3)]),
        "output_throughput_tok_s": 1000,
        "goodput_output_throughput_tok_s": 500,
    }
    goal = OptimizationGoal(
        target=target,
        sla={"ttft_ms": 100} if target == "goodput" else None,
        aiperf={
            "constraints": [{"metric": "output_token_throughput_per_user.p50", "operator": "gte", "threshold": 101}]
        },
    )
    # Preserve goodput's existing strict mean gate when it is explicitly enabled.
    report.update(completed_requests=1, num_ttft_samples=1, mean_ttft_ms=10)
    candidate = make_candidate({"used_gpus": 1}, report, goal.target)
    assert analyze_candidates([candidate], goal) == [candidate]
    assert analyze_candidates([candidate], goal.model_copy(update={"strict_sla": True})) == []
    at_bound = goal.metric_constraints[0].model_copy(update={"threshold": 100})
    assert aggregate_sla_violations(report, None, [at_bound]) == ()
    assert aggregate_sla_violations({}, None, [at_bound])


def test_cli_forwards_metric_policy_without_changing_goodput_sla():
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "workers": {"aggregated": {}},
            },
            "evaluation": {
                "aiperf": {
                    "per_chunk_usage": True,
                    "constraints": [
                        {"metric": "output_token_throughput_per_user.p50", "operator": "gte", "threshold": 50}
                    ],
                }
            },
            "optimization": {"target": "request_latency.p99", "strict_sla": True},
        }
    )
    restored = CoreRecommendationConfig.model_validate_json(config.model_dump_json())
    goal = OptimizationGoal.model_validate(_goal(restored))
    assert goal.target is AIPerfMetric.LATENCY_P99
    assert goal.strict_sla and goal.aiperf_per_chunk_usage and goal.uses_aiperf_metrics
    assert goal.sla is None
    with pytest.raises(ValidationError):
        OptimizationGoal(target="request_latency.p42")


def test_aiperf_metrics_work_as_pareto_axes():
    goal = OptimizationGoal(target="pareto", pareto_objectives=["throughput", "request_latency.p99"])
    report = {**summarize_requests([record(10, 20, 2)]), "output_throughput_tok_s": 100}
    candidate = make_candidate({"used_gpus": 1}, report, goal.target, pareto_objectives=goal.resolved_pareto_objectives)
    assert candidate.objectives == {"throughput": 100, "request_latency.p99": 20}
    assert analyze_candidates([candidate], goal) == [candidate]
