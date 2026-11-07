# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf selectors over native summary fields. Native values are tested in report.rs."""

import pytest
from pydantic import ValidationError

from aisimulate.aiperf_metrics import AIPerfMetric, metric_value
from aisimulate.config.cli import CoreRecommendationConfig
from aisimulate.recommend import _goal
from aisimulate.sweeper.config import Candidate, OptimizationGoal
from aisimulate.sweeper.score import aggregate_sla_violations, analyze_candidates, make_candidate


def native(value=10.0, count=2.0, **overrides):
    """A native summary where every selected statistic equals ``value``."""
    report = {key: value for metric in AIPerfMetric for key in [metric.report_key]}
    report.update({metric.count_key: count for metric in AIPerfMetric})
    return {**report, **overrides}


@pytest.mark.parametrize(
    "metric, key, count_key",
    [
        ("time_to_first_token.avg", "mean_ttft_ms", "num_ttft_samples"),
        ("request_latency.p99", "p99_e2e_latency_ms", "num_e2e_latency_samples"),
        ("inter_token_latency.p50", "median_itl_ms", "num_tpot_samples"),
        ("output_token_throughput_per_user.p75", "p75_output_token_throughput_per_user", "num_tpot_samples"),
    ],
)
def test_selectors_read_native_summary_fields(metric, key, count_key):
    metric = AIPerfMetric(metric)
    assert (metric.report_key, metric.count_key) == (key, count_key)
    assert metric_value({key: 3.0, count_key: 1}, metric) == 3.0


def test_selector_set_covers_aiperf_statistics():
    stats = {metric.value.rsplit(".", 1)[1] for metric in AIPerfMetric}
    assert stats == {"avg", "p50", "p75", "p90", "p95", "p99"}
    assert len(AIPerfMetric) == 24
    assert AIPerfMetric.TTFT_P75.value == "time_to_first_token.p75"


def test_no_samples_are_not_zero_latency():
    # Native reports zeros for an empty distribution; the sample count decides.
    with pytest.raises(ValueError, match="no qualifying samples"):
        metric_value(native(0.0, 0.0), AIPerfMetric.TTFT_AVG)


@pytest.mark.parametrize("metric", list(AIPerfMetric))
def test_all_selectors_score_and_round_trip(metric):
    report = native(25.0)
    goal = OptimizationGoal(target=metric)
    assert OptimizationGoal.model_validate_json(goal.model_dump_json()) == goal
    candidate = make_candidate({"used_gpus": 1}, report, goal.target)
    sign = 1 if metric.maximize else -1
    assert candidate.score == sign * 25.0
    assert Candidate.model_validate_json(candidate.model_dump_json()).metrics == candidate.metrics


@pytest.mark.parametrize("target", ["throughput", "goodput", "output_token_throughput_per_user.p50"])
def test_scoring_is_independent_of_strict_rejection(target):
    report = native(100.0, 1.0, output_throughput_tok_s=1000, goodput_output_throughput_tok_s=500)
    goal = OptimizationGoal(
        target=target,
        sla={"ttft_ms": 100} if target == "goodput" else None,
        aiperf={
            "constraints": [{"metric": "output_token_throughput_per_user.p50", "operator": "gte", "threshold": 101}]
        },
    )
    # Preserve goodput's existing strict mean gate when it is explicitly enabled.
    report.update(completed_requests=1, mean_ttft_ms=10)
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
    assert goal.strict_sla and goal.uses_aiperf_metrics
    assert goal.sla is None
    with pytest.raises(ValidationError):
        OptimizationGoal(target="request_latency.p42")
    with pytest.raises(ValidationError):
        OptimizationGoal(aiperf={"per_chunk_usage": True})


def test_aiperf_metrics_work_as_pareto_axes():
    goal = OptimizationGoal(target="pareto", pareto_objectives=["throughput", "request_latency.p99"])
    report = native(20.0, output_throughput_tok_s=100)
    candidate = make_candidate({"used_gpus": 1}, report, goal.target, pareto_objectives=goal.resolved_pareto_objectives)
    assert candidate.objectives == {"throughput": 100, "request_latency.p99": 20}
    assert analyze_candidates([candidate], goal) == [candidate]
