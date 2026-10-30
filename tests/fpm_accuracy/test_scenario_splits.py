# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from math import ceil

import pytest
import test_scenarios as scenario_fixtures
from test_scenarios import observation

from scripts.fpm_accuracy import scenarios
from scripts.fpm_accuracy.scenario_contract import (
    DEFAULT_TRAIN_PERCENT,
    METHODS,
    PROTOCOL_METADATA,
    TRAIN_PERCENTAGES,
    _result,
    validate_scenarios,
)
from scripts.fpm_accuracy.scenarios import evaluate_scenario_case, score_scenario, split_observations

public_scenarios = scenario_fixtures.public_scenarios
recording = scenario_fixtures.recording
scenario_case = scenario_fixtures.scenario_case


@pytest.mark.parametrize("percent", TRAIN_PERCENTAGES)
def test_distributed_split_covers_each_workload_uniformly_at_every_prefix(percent):
    kinds = ("decode",) * 7 + ("prefill", "mixed", "decode") * 11 + ("prefill",) * 3
    points = tuple(observation(index, kind) for index, kind in enumerate(kinds))
    training, testing = split_observations(points, "b", train_percent=percent)
    train_ids = {point.observation_id for point in training}
    test_ids = {point.observation_id for point in testing}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {point.observation_id for point in points}
    assert [point.order for point in training] == sorted(point.order for point in training)
    assert [point.order for point in testing] == sorted(point.order for point in testing)
    seen, trained = Counter(), Counter()
    for point in points:
        seen[point.workload_kind] += 1
        trained[point.workload_kind] += point.observation_id in train_ids
        assert trained[point.workload_kind] == ceil(Fraction(seen[point.workload_kind] * percent, 100))


@pytest.mark.parametrize(
    "percent,expected",
    [(10, [0, 10, 20]), (30, [0, 3, 6, 10, 13, 16, 20]), (90, [*range(9), *range(10, 19), 20])],
)
def test_distributed_training_positions_are_spread_across_stream(percent, expected):
    training, _ = split_observations(tuple(observation(index) for index in range(21)), "b", train_percent=percent)
    assert [point.order for point in training] == expected


def test_default_split_matches_previous_odd_even_protocol_even_with_uneven_kinds():
    points = tuple(observation(index, ("decode", "prefill", "decode", "mixed")[index % 4]) for index in range(23))
    counts, odd, even = Counter(), [], []
    for point in points:
        counts[point.workload_kind] += 1
        (odd if counts[point.workload_kind] % 2 else even).append(point)
    assert DEFAULT_TRAIN_PERCENT == 50
    assert split_observations(points, "b") == (tuple(odd), tuple(even))
    assert split_observations(points, "bc") == (points[:11], points[11:])
    assert PROTOCOL_METADATA["split"]["train_percentages"] == list(TRAIN_PERCENTAGES)


@pytest.mark.parametrize("percent", TRAIN_PERCENTAGES)
@pytest.mark.parametrize("size", [0, 1, 2, 7, 11, 23])
def test_temporal_split_is_an_exact_prefix_and_suffix(percent, size):
    points = tuple(observation(index) for index in range(size))
    training, testing = split_observations(points, "bc", train_percent=percent)
    boundary = int(Fraction(size * percent, 100))
    assert training == points[:boundary]
    assert testing == points[boundary:]
    assert training + testing == points


@pytest.mark.parametrize("percent", TRAIN_PERCENTAGES)
@pytest.mark.parametrize("scenario", ["b", "bc"])
def test_partition_selection_never_reads_target_latency(scenario, percent):
    first = tuple(observation(index, "prefill" if index % 3 else "decode", actual=10) for index in range(21))
    changed = tuple(
        observation(index, "prefill" if index % 3 else "decode", actual=999 if index % 2 else 0.001)
        for index in range(21)
    )
    selected = lambda items: tuple(tuple(point.observation_id for point in subset) for subset in items)
    assert selected(split_observations(first, scenario, train_percent=percent)) == selected(
        split_observations(changed, scenario, train_percent=percent)
    )


@pytest.mark.parametrize("invalid", [True, False, 0, 100, 15, -10, 50.0, "50", None])
def test_invalid_training_percent_is_rejected_before_construction(scenario_case, recording, invalid):
    instances, options = recording
    with pytest.raises(ValueError, match="train_percent"):
        split_observations(scenario_case.observations, "b", train_percent=invalid)
    for method, scenario in (("linear", "b"), ("warmup", "c"), ("linear", "a")):
        with pytest.raises(ValueError, match="train_percent"):
            score_scenario(scenario_case, method, scenario, train_percent=invalid, **options)
    assert not instances


def test_small_partitions_report_missing_train_or_test_without_fitting(scenario_case, recording):
    instances, options = recording
    case = replace(scenario_case, observations=scenario_case.observations[:3])
    assert score_scenario(case, "linear", "b", train_percent=10, **options)["status"] == "no_test_data"
    assert score_scenario(case, "linear", "bc", train_percent=10, **options)["status"] == "no_training_data"
    assert not instances


@pytest.mark.parametrize("percent,boundary,updates", [(20, 4, [9, 14, 19]), (70, 14, [19]), (90, 18, [])])
def test_temporal_split_scores_before_update_and_anchors_at_first_test_point(
    scenario_case, recording, percent, boundary, updates
):
    instances, options = recording
    case = replace(scenario_case, observations=tuple(observation(index, actual=index + 1) for index in range(20)))
    result = score_scenario(case, "linear", "bc", train_percent=percent, **options)
    events = instances[0].events
    assert events[0] == ("fit", list(range(boundary)))
    assert [event[1] for event in events if event[0] == "predict"] == list(range(boundary, 20))
    assert [event[1] for event in events if event[0] == "tune"] == [[index] for index in updates]
    for index in updates:
        update_at = events.index(("tune", [index]))
        assert events[update_at - 1][:2] == ("predict", index)
        known_targets = list(range(1, boundary + 1)) + [earlier + 1 for earlier in updates if earlier < index]
        assert events[update_at - 1][2] == pytest.approx(sum(known_targets) / len(known_targets))
    assert result["training_count"] == boundary
    assert result["test_count"] == 20 - boundary
    assert result["online_schedule"]["elapsed_ms"] == (19 - boundary) * 1000


def test_temporal_boundary_allows_equal_timestamps(scenario_case, recording):
    _, options = recording
    case = replace(
        scenario_case, observations=tuple(replace(point, event_time_ms=123) for point in scenario_case.observations)
    )
    result = score_scenario(case, "linear", "bc", train_percent=30, **options)
    assert result["status"] == "evaluated"
    assert result["cost"]["predict_count"] == 9
    assert result["online_schedule"]["scheduled_update_count"] == 0


def test_grid_reuses_actual_default_results_and_runs_unsplit_scenarios_once(scenario_case, recording, monkeypatch):
    _, options = recording
    original, calls = scenarios.score_scenario, Counter()

    def record(case, method, scenario, **kwargs):
        calls[(scenario, method, kwargs.get("train_percent", 50))] += 1
        return original(case, method, scenario, **kwargs)

    monkeypatch.setattr(scenarios, "score_scenario", record)
    row = evaluate_scenario_case(scenario_case, **options)
    assert set(row["split_results"]) == {str(percent) for percent in TRAIN_PERCENTAGES}
    for scenario in ("b", "bc"):
        assert row["split_results"]["50"][scenario] is row["results"][scenario]
        for percent in TRAIN_PERCENTAGES:
            for method in METHODS:
                assert calls[(scenario, method, percent)] == 1
    for scenario in ("a", "c", "ac"):
        assert sum(count for key, count in calls.items() if key[0] == scenario) == len(METHODS)


@pytest.mark.parametrize(
    "mutation", ["missing_ratio", "extra_ratio", "missing_scenario", "extra_scenario", "missing_method"]
)
def test_v4_requires_the_exact_complete_split_grid(public_scenarios, mutation):
    document, summary = deepcopy(public_scenarios)
    grid = document["rows"][0]["split_results"]
    if mutation == "missing_ratio":
        del grid["10"]
    elif mutation == "extra_ratio":
        grid["15"] = deepcopy(grid["10"])
    elif mutation == "missing_scenario":
        del grid["10"]["bc"]
    elif mutation == "extra_scenario":
        grid["10"]["a"] = document["rows"][0]["results"]["a"]
    else:
        del grid["10"]["b"]["gbdt"]
    with pytest.raises(ValueError):
        validate_scenarios(document, summary)


def test_v4_rejects_copied_half_split_and_different_default_results(public_scenarios):
    document, summary = deepcopy(public_scenarios)
    grid = document["rows"][0]["split_results"]
    grid["10"]["b"] = deepcopy(grid["50"]["b"])
    with pytest.raises(ValueError, match="distributed holdout"):
        validate_scenarios(document, summary)
    document, summary = deepcopy(public_scenarios)
    grid = document["rows"][0]["split_results"]
    grid["50"] = deepcopy(grid["50"])
    grid["50"]["b"]["linear"]["cost"]["fit_ms"] += 1
    with pytest.raises(ValueError, match="default split"):
        validate_scenarios(document, summary)


@pytest.mark.parametrize("field", ["scoring_population_sha256", "training_population_sha256"])
def test_v4_requires_identical_partitions_across_methods(public_scenarios, field):
    document, summary = deepcopy(public_scenarios)
    document["rows"][0]["split_results"]["30"]["bc"]["linear"][field] = "f" * 64
    with pytest.raises(ValueError, match="different scenario populations"):
        validate_scenarios(document, summary)


def test_legacy_contract_does_not_accept_nondefault_ratio(public_scenarios):
    document, _ = public_scenarios
    row = document["rows"][0]
    with pytest.raises(ValueError, match="legacy scenarios"):
        _result(row["results"]["bc"]["linear"], row, "bc", "linear", 3, train_percent=30)
