# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimum-GPU target-load recommendation contracts."""

import pytest
from pydantic import ValidationError

from aisimulate.sweeper import (
    LoadRecommendation,
    LoadTarget,
    NoFeasibleLoadRecommendation,
    candidate_capacity,
    recommend_min_gpus,
    size_candidate,
)
from aisimulate.sweeper.config import (
    Candidate,
    OptimizationGoal,
    OptimizationTarget,
    SLATarget,
)
from aisimulate.sweeper.score import make_candidate


def _candidate(
    name: str,
    *,
    gpus: int,
    rate: float | None = None,
    concurrency: float | None = None,
    mode: str = "agg",
    metrics: dict[str, float] | None = None,
    config: dict | None = None,
) -> Candidate:
    candidate_metrics = dict(metrics or {})
    if rate is not None:
        candidate_metrics["request_throughput_rps"] = rate
    candidate_config = {
        "name": name,
        "deployment_mode": mode,
        **(config or {}),
    }
    if concurrency is not None:
        candidate_config["concurrency"] = concurrency
    return Candidate(
        config=candidate_config,
        used_gpus=gpus,
        score=1.0,
        metrics=candidate_metrics,
    )


def test_load_target_requires_exactly_one_finite_positive_shape():
    assert LoadTarget(request_rate=10).kind == "request_rate"
    assert LoadTarget(concurrency=32).kind == "concurrency"
    for payload in (
        {},
        {"request_rate": 1, "concurrency": 1},
        {"request_rate": 0},
        {"concurrency": float("inf")},
    ):
        with pytest.raises(ValidationError):
            LoadTarget.model_validate(payload)


def test_make_candidate_preserves_request_rate_capacity():
    candidate = make_candidate(
        {"used_gpus": 2, "deployment_mode": "agg"},
        {
            "output_throughput_tok_s": 1000.0,
            "request_throughput_rps": 12.5,
        },
        target=OptimizationTarget.THROUGHPUT,
    )
    assert candidate.metrics["request_throughput_rps"] == 12.5


def test_request_rate_recommendation_minimizes_true_gpu_count():
    narrow = _candidate("narrow", gpus=2, rate=10)
    wide = _candidate("wide", gpus=4, rate=21)
    results = recommend_min_gpus(
        [narrow, wide],
        LoadTarget(request_rate=21),
        goal=OptimizationGoal(),
    )

    assert [result.candidate.config["name"] for result in results] == [
        "wide",
        "narrow",
    ]
    assert results[0].replicas_needed == 1
    assert results[0].total_gpus_needed == 4
    assert results[1].replicas_needed == 3
    assert results[1].total_gpus_needed == 6


def test_concurrency_recommendation_uses_evaluated_candidate_load():
    candidate = _candidate(
        "disagg",
        gpus=2,
        concurrency=8,
        mode="disagg",
    )
    result = size_candidate(candidate, LoadTarget(concurrency=17))

    assert result.capacity_per_replica == 8
    assert result.replicas_needed == 3
    assert result.total_gpus_needed == 6
    assert result.limiting_role == "rate_matched_deployment"


def test_role_rates_identify_the_limiting_pool():
    candidate = _candidate(
        "heterogeneous",
        gpus=8,
        rate=12,
        mode="disagg",
        metrics={
            "prefill_request_throughput_rps": 18.0,
            "decode_request_throughput_rps": 12.0,
        },
    )

    assert candidate_capacity(candidate, LoadTarget(request_rate=20)) == (
        12.0,
        "decode",
    )


def test_strict_sla_filters_before_min_gpu_ranking():
    violating = _candidate(
        "fast-but-late",
        gpus=2,
        rate=20,
        metrics={
            "completed_requests": 1.0,
            "num_tpot_samples": 1.0,
            "mean_tpot_ms": 20.0,
        },
    )
    compliant = _candidate(
        "compliant",
        gpus=4,
        rate=10,
        metrics={
            "completed_requests": 1.0,
            "num_tpot_samples": 1.0,
            "mean_tpot_ms": 10.0,
        },
    )
    goal = OptimizationGoal(
        sla=SLATarget(itl_ms=10),
        strict_sla=True,
    )

    results = recommend_min_gpus(
        [violating, compliant],
        LoadTarget(request_rate=20),
        goal=goal,
    )
    assert [result.candidate.config["name"] for result in results] == ["compliant"]


def test_gpu_cap_requires_explicit_partial_service():
    candidate = _candidate("capped", gpus=4, rate=10)

    with pytest.raises(NoFeasibleLoadRecommendation, match="allow_partial=true"):
        recommend_min_gpus(
            [candidate],
            LoadTarget(request_rate=30, max_gpus=8),
            goal=OptimizationGoal(),
        )

    result = recommend_min_gpus(
        [candidate],
        LoadTarget(request_rate=30, max_gpus=8, allow_partial=True),
        goal=OptimizationGoal(),
    )[0]
    assert result.replicas_needed == 3
    assert result.total_gpus_needed == 12
    assert result.deployed_replicas == 2
    assert result.deployed_gpus == 8
    assert result.supported_load == 20
    assert result.load_served_pct == pytest.approx(66.6666667)
    assert result.partial


def test_full_service_ranks_ahead_of_partial_service():
    full = _candidate("full", gpus=8, rate=30)
    partial = _candidate("partial", gpus=4, rate=10)
    results = recommend_min_gpus(
        [partial, full],
        LoadTarget(request_rate=30, max_gpus=8, allow_partial=True),
        goal=OptimizationGoal(),
    )
    assert [result.candidate.config["name"] for result in results] == [
        "full",
        "partial",
    ]


def test_missing_or_impossible_capacity_has_actionable_no_feasible_error():
    missing = _candidate("missing", gpus=2)
    too_wide = _candidate("too-wide", gpus=16, rate=100)

    with pytest.raises(NoFeasibleLoadRecommendation) as exc_info:
        recommend_min_gpus(
            [missing, too_wide],
            LoadTarget(request_rate=10, max_gpus=8, allow_partial=True),
            goal=OptimizationGoal(),
        )

    message = str(exc_info.value)
    assert "does not expose" in message
    assert "one replica needs 16 GPUs" in message


def test_higher_target_is_monotonic_for_one_topology():
    candidate = _candidate("same-topology", gpus=4, rate=10)
    low = size_candidate(candidate, LoadTarget(request_rate=10))
    high = size_candidate(candidate, LoadTarget(request_rate=31))
    assert high.total_gpus_needed >= low.total_gpus_needed
    assert high.replicas_needed == 4


def test_equal_sizing_uses_efficiency_latency_and_config_ties():
    slower = _candidate(
        "z",
        gpus=4,
        rate=10,
        metrics={"mean_e2e_latency_ms": 200.0},
    )
    faster = _candidate(
        "a",
        gpus=4,
        rate=10,
        metrics={"mean_e2e_latency_ms": 100.0},
    )
    results = recommend_min_gpus(
        [slower, faster],
        LoadTarget(request_rate=20),
        goal=OptimizationGoal(),
    )
    assert [result.candidate.config["name"] for result in results] == ["a", "z"]


def test_recommendation_round_trips_as_machine_readable_json():
    recommendation = size_candidate(
        _candidate("round-trip", gpus=2, rate=8),
        LoadTarget(request_rate=15, max_gpus=8),
    )
    assert (
        LoadRecommendation.model_validate_json(recommendation.model_dump_json())
        == recommendation
    )
