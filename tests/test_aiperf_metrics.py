# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf selectors over native summary fields. Native values are tested in report.rs."""

import pytest
from pydantic import ValidationError

from aisim.aiperf_metrics import AIPerfMetric, NoSamplesError, metric_value
from aisim.config.cli import CoreRecommendationConfig
from aisim.recommend import _goal
from aisim.sweeper.config import Candidate, OptimizationGoal
from aisim.sweeper.score import aggregate_sla_violations, analyze_candidates, make_candidate


def native(value=10.0, count=2.0, **overrides):
    """A native summary where every selected statistic equals ``value``."""
    report = {metric.report_key: value for metric in AIPerfMetric}
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
    assert len(AIPerfMetric) == 24
    with pytest.raises(ValidationError):
        OptimizationGoal(target="request_latency.p42")


@pytest.mark.parametrize("value, count, error", [(0.0, 0.0, NoSamplesError), (-1.0, 2.0, ValueError)])
def test_missing_samples_and_invalid_values_are_rejected(value, count, error):
    # Native reports zeros for an empty distribution; the sample count decides.
    with pytest.raises(error) as raised:
        metric_value(native(value, count), AIPerfMetric.TTFT_AVG)
    assert isinstance(raised.value, NoSamplesError) is (error is NoSamplesError)


@pytest.mark.parametrize("metric", [AIPerfMetric.LATENCY_P99, AIPerfMetric.USER_TPS_P50])
def test_selectors_score_in_their_direction_and_round_trip(metric):
    goal = OptimizationGoal(target=metric)
    assert OptimizationGoal.model_validate_json(goal.model_dump_json()) == goal
    candidate = make_candidate({"used_gpus": 1}, native(25.0), goal.target)
    assert candidate.score == (25.0 if metric.maximize else -25.0)
    assert Candidate.model_validate_json(candidate.model_dump_json()).metrics == candidate.metrics


def test_scoring_is_independent_of_strict_rejection():
    report = native(100.0, 1.0, output_throughput_tok_s=1000)
    goal = OptimizationGoal(
        target="throughput",
        metric_constraints=[{"metric": "output_token_throughput_per_user.p50", "operator": "gte", "threshold": 101}],
    )
    candidate = make_candidate({"used_gpus": 1}, report, goal.target)
    assert analyze_candidates([candidate], goal) == [candidate]
    assert analyze_candidates([candidate], goal.model_copy(update={"strict_sla": True})) == []
    at_bound = goal.metric_constraints[0].model_copy(update={"threshold": 100})
    assert aggregate_sla_violations(report, None, [at_bound]) == ()
    assert aggregate_sla_violations({}, None, [at_bound])


def test_aiperf_metrics_work_as_pareto_axes():
    goal = OptimizationGoal(target="pareto", pareto_objectives=["throughput", "request_latency.p99"])
    report = native(20.0, output_throughput_tok_s=100)
    candidate = make_candidate({"used_gpus": 1}, report, goal.target, pareto_objectives=goal.resolved_pareto_objectives)
    assert candidate.objectives == {"throughput": 100, "request_latency.p99": 20}
    assert analyze_candidates([candidate], goal) == [candidate]


def test_recommend_forwards_metric_constraints_to_the_search_goal():
    raw = {
        "engine": {
            "mode": "aggregated",
            "model": "example/model",
            "hardware": "h200_sxm",
            "workers": {"aggregated": {}},
        },
        "optimization": {
            "target": "request_latency.p99",
            "strict_sla": True,
            "constraints": {
                "metrics": [{"metric": "output_token_throughput_per_user.p50", "operator": "gte", "threshold": 50}]
            },
        },
    }
    config = CoreRecommendationConfig.model_validate(raw)
    goal = OptimizationGoal.model_validate(_goal(config))
    assert goal.target is AIPerfMetric.LATENCY_P99
    assert goal.metric_constraints == config.optimization.constraints.metrics
    raw["optimization"]["constraints"] = {}
    with pytest.raises(ValidationError, match="strict_sla requires"):
        CoreRecommendationConfig.model_validate(raw)
