# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.fpm_accuracy.exceptions import DependencyError
from scripts.fpm_accuracy.hf.models import (
    CaseStatus,
    FpmArtifact,
    MeasurementCase,
    MeasurementObservation,
    OrderingKind,
)
from scripts.fpm_accuracy.models.fpt_predictor import Prediction
from scripts.fpm_accuracy.scenario_contract import TRAIN_PERCENTAGES, validate_scenarios
from scripts.fpm_accuracy.scenarios import (
    METHODS,
    SCENARIOS,
    evaluate_scenario_case,
    scenario_document,
    score_scenario,
    select_artifact,
)
from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics


def observation(index, kind="decode", actual=10.0, *, capture_id="target", worker="worker"):
    scheduled = RequestMetrics(
        num_prefill_requests=int(kind != "decode"),
        sum_prefill_tokens=32 * int(kind != "decode"),
        num_decode_requests=int(kind != "prefill"),
        sum_decode_kv_tokens=64 * int(kind != "prefill"),
    )
    iteration = ForwardPassIteration.single_rank(
        ForwardPassMetric(
            fpm_id=index,
            configuration_id="config",
            version=1,
            worker_id=worker,
            dp_rank=0,
            counter_id=index,
            wall_time_s=actual / 1000,
            scheduled=scheduled,
            queued=RequestMetrics(),
            capture_id=capture_id,
        )
    )
    return MeasurementObservation(
        observation_id=f"{capture_id}/{index}",
        configuration_id="config",
        order=index,
        source_file_id="file",
        source_path="data/model/source.jsonl",
        source_row=index,
        iteration=iteration,
        event_time_ms=index * 1000,
    )


def artifact(identity, *, kv_off=False):
    path = f"data/model/{identity}.kv-off.parquet" if kv_off else f"data/model/{identity}.parquet"
    return FpmArtifact(
        identity,
        path,
        path + ".json",
        "a" * 64,
        "b" * 64,
        "fpm",
        ("decode",),
        2,
        Path(path),
        Path(path + ".json"),
        "",
        "",
    )


@pytest.fixture
def scenario_case():
    config = SimpleNamespace(
        configuration_id="config",
        configuration_path="data/model/config",
        snapshot_id="snapshot",
        model_id="test/model",
        gpu_family="test_gpu",
        system="test_gpu",
        framework="test_backend",
        framework_version="1.0",
        parallelism="tp1",
        worker_config_record=object(),
    )
    return MeasurementCase(
        case_id="case",
        measurement_membership_sha256="c" * 64,
        configuration=config,
        protocol_id="test-protocol",
        status=CaseStatus.READY,
        observations=tuple(observation(index, ("prefill", "decode", "mixed")[index % 3]) for index in range(12)),
        truth_files=(),
        helper_files=(),
        fpm_artifacts=(artifact("warmup"), artifact("nowarmup", kv_off=True)),
        worker_role="aggregated",
        ordering=OrderingKind.CHRONOLOGICAL,
        override_applied=False,
        override_sha256=None,
        override_effects=(),
    )


class RecordingPredictor:
    def __init__(self, method, training_iterations):
        self.method = method
        self.targets = [item.observed_time_ms for item in training_iterations]
        self.events = [("fit", [item.fpm_ids[0] for item in training_iterations])]
        self.closed = False
        self.online = False
        self.cold_start = False

    def predict(self, features):
        assert all("wall_time" not in rank for rank in features.rank_payloads)
        assert not hasattr(features, "actual_ms")
        predicted = sum(self.targets) / len(self.targets) if len(self.targets) >= 2 else None
        self.events.append(("predict", features.fpm_ids[0], predicted))
        return Prediction(predicted)

    def tune(self, iterations):
        self.events.append(("tune", [item.fpm_ids[0] for item in iterations]))
        self.targets.extend(item.observed_time_ms for item in iterations)

    def close(self):
        self.closed = True

    def diagnostics(self):
        if self.method in {"gbdt", "hybrid", "selector"}:
            from scripts.fpm_accuracy.scenario_contract import (
                GBDT_FROZEN_RECIPE,
                GBDT_ONLINE_RECIPE,
                HYBRID_ONLINE_RECIPE,
                SELECTOR_ONLINE_RECIPE,
            )

            return {
                "method": self.method,
                "model_scope": "case_worker_and_native_workload_store",
                "adaptation": "causal_selector"
                if self.method == "selector"
                else "online"
                if self.method == "gbdt" and self.online
                else "frozen",
                "recipe": {
                    "gbdt": GBDT_ONLINE_RECIPE if self.online else GBDT_FROZEN_RECIPE,
                    "hybrid": HYBRID_ONLINE_RECIPE,
                    "selector": SELECTOR_ONLINE_RECIPE,
                }[self.method],
                "scopes": [],
            }
        return {"model_scope": "worker_id_and_native_workload_store", "scopes": []}


@pytest.fixture
def recording():
    instances = []

    def factory(
        method, context, *, training_iterations=(), online=False, cold_start=False, mixed_fallback=False, scenario=None
    ):
        model = RecordingPredictor(method, training_iterations)
        model.online, model.cold_start = online, cold_start
        instances.append(model)
        return model

    def bootstrap(case, selected):
        assert selected in case.fpm_artifacts
        return tuple(observation(index, actual=5, capture_id="self-benchmark").iteration for index in (100, 101))

    return instances, {"factory": factory, "bootstrap": bootstrap}


def test_a_uses_external_training_and_never_updates_targets(scenario_case, recording):
    instances, options = recording
    result = score_scenario(scenario_case, "linear", "a", **options)
    assert instances[0].events[0] == ("fit", [100, 101])
    assert len([event for event in instances[0].events if event[0] == "predict"]) == 12
    assert all(event[0] != "tune" for event in instances[0].events)
    assert result["training_count"] == 2 and result["test_count"] == 12
    assert result["metrics"]["all"]["mape_pct"] == 50
    assert result["online_update_count"] == 0
    assert result["cost"]["update_us"] is None
    assert instances[0].closed


def test_interpolation_records_actual_method_without_inference(scenario_case, recording):
    _, options = recording
    evidence = {"method": "direct", "timing_only": True, "text_only": True, "metadata_recovered": False}
    model = RecordingPredictor("warmup", ())
    model.diagnostics = lambda: {"interpolation": evidence}
    result = score_scenario(scenario_case, "warmup", "a", **{**options, "factory": lambda *a, **k: model})
    assert result["interpolation"] == evidence
    assert model.closed
    old_runtime = score_scenario(scenario_case, "warmup", "a", **options)
    assert "interpolation" not in old_runtime


def test_failed_interpolation_diagnostics_closes_model_without_scoring(scenario_case, recording):
    _, options = recording
    model = RecordingPredictor("warmup", ())

    def fail():
        raise ValueError("invalid construction provenance")

    model.diagnostics = fail
    result = score_scenario(scenario_case, "warmup", "a", **{**options, "factory": lambda *a, **k: model})
    assert model.closed
    assert result["status"] == "predictor_error"
    assert result["cost"]["predict_count"] == 0
    assert result["metrics"]["all"]["error_count"] == 12


def test_b_alternates_each_workload_kind_and_freezes_after_fit(scenario_case, recording):
    instances, options = recording
    result = score_scenario(scenario_case, "spline", "b", **options)
    assert instances[0].events[0] == ("fit", [0, 1, 2, 6, 7, 8])
    assert [event[1] for event in instances[0].events[1:]] == [3, 4, 5, 9, 10, 11]
    assert all(event[0] != "tune" for event in instances[0].events)
    assert result["training_count"] == result["test_count"] == 6
    assert result["artifact"] is None


@pytest.mark.parametrize(
    "scenario,expected_training,test_ids,scheduled,warmup,skipped",
    [
        ("c", [], list(range(12)), [[5], [10]], [], 10),
        ("ac", [100, 101], list(range(12)), [[5], [10]], [], 10),
        ("bc", list(range(6)), list(range(6, 12)), [[11]], [], 5),
    ],
)
def test_streams_predict_before_current_target_updates(
    scenario_case, recording, scenario, expected_training, test_ids, scheduled, warmup, skipped
):
    instances, options = recording
    result = score_scenario(scenario_case, "linear", scenario, **options)
    events = instances[0].events
    assert events[0] == ("fit", expected_training)
    predictions = [event[1] for event in events if event[0] == "predict"]
    updates = [event[1] for event in events if event[0] == "tune"]
    assert predictions == test_ids
    assert updates == warmup + scheduled
    assert all(len(points) == 1 for points in updates)
    seen = set()
    for event in events[1:]:
        if event[0] == "predict":
            seen.add(event[1])
        else:
            assert set(event[1]) <= seen
    assert result["online_update_count"] == result["cost"]["update_count"] == len(updates)
    assert result["cost"]["predict_count"] == len(test_ids)
    assert result["online_schedule"] == {
        "interval_ms": 5000,
        "time_basis": "event_time_ms",
        "warmup_policy": "scheduled_only" if scenario == "c" else "none",
        "selection_policy": "first_observation_at_or_after_tick",
        "elapsed_ms": (test_ids[-1] - test_ids[0]) * 1000,
        "scheduled_update_count": len(scheduled),
        "warmup_update_count": len(warmup),
        "update_error_count": 0,
        "submitted_observation_count": len(test_ids) - skipped,
        "skipped_observation_count": skipped,
    }
    if scenario == "c":
        assert result["cold_start"] == {"unavailable_before_first_prediction": 11, "first_prediction_index": 11}
        assert result["metrics"]["all"]["unavailable_count"] == 11
        assert result["metrics"]["all"]["coverage_pct"] == pytest.approx(100 / 12)


def test_changing_current_and_future_targets_cannot_change_earlier_predictions(scenario_case, recording):
    instances, options = recording
    changed = list(scenario_case.observations)
    changed[5:] = [observation(item.order, str(item.workload_kind), actual=1000) for item in changed[5:]]
    score_scenario(scenario_case, "linear", "c", **options)
    score_scenario(replace(scenario_case, observations=tuple(changed)), "linear", "c", **options)
    streams = [[event for event in item.events if event[0] == "predict"] for item in instances]
    # The due observation is predicted before its target becomes training data.
    assert streams[0][:6] == streams[1][:6]
    assert streams[0][11] != streams[1][11]


@pytest.mark.parametrize(
    "scenario,skipped",
    [
        ("c", {0, 1, 2, 3, 4, 6, 7, 8, 9, 11}),
        ("ac", {0, 1, 2, 3, 4, 6, 7, 8, 9, 11}),
        ("bc", {6, 7, 8, 9, 10}),
    ],
)
def test_intervening_targets_are_never_accumulated_or_replayed(scenario_case, recording, scenario, skipped):
    instances, options = recording
    altered = tuple(
        observation(item.order, str(item.workload_kind), actual=1000 if item.order in skipped else 10)
        for item in scenario_case.observations
    )
    first = score_scenario(scenario_case, "linear", scenario, **options)
    second = score_scenario(replace(scenario_case, observations=altered), "linear", scenario, **options)
    # Scoring targets change, but labels on every nonselected point are never
    # retained for a later tick and cannot affect any prediction or model state.
    assert instances[0].events == instances[1].events
    assert instances[0].targets == instances[1].targets
    assert first["metrics"]["all"]["mape_pct"] != second["metrics"]["all"]["mape_pct"]
    assert first["online_schedule"]["skipped_observation_count"] == len(skipped)


@pytest.mark.parametrize("fail_due_warmup", [False, True])
def test_c_cold_update_consumes_due_tick_without_doublefeeding_or_retry(scenario_case, recording, fail_due_warmup):
    _, options = recording

    class WarmupPredictor(RecordingPredictor):
        def predict(self, features):
            index = features.fpm_ids[0]
            value = None if index < 2 else 10
            self.events.append(("predict", index, value))
            return Prediction(value)

        def tune(self, iterations):
            assert len(iterations) == 1
            super().tune(iterations)
            if fail_due_warmup and iterations[0].fpm_ids[0] == 1:
                raise ValueError("one failed warmup call")

    model = WarmupPredictor("linear", ())
    case = replace(
        scenario_case,
        observations=tuple(
            replace(observation(index), event_time_ms=time) for index, time in enumerate((0, 5000, 5000, 10000, 10000))
        ),
    )
    result = score_scenario(case, "linear", "c", **{**options, "factory": lambda *a, **k: model})
    assert [event[1] for event in model.events if event[0] == "tune"] == [[1], [3]]
    schedule = result["online_schedule"]
    assert schedule["warmup_update_count"] == 0
    assert schedule["scheduled_update_count"] == 2
    assert schedule["submitted_observation_count"] == 2
    assert schedule["skipped_observation_count"] == 3
    assert schedule["update_error_count"] == int(fail_due_warmup)
    assert result["online_update_count"] == 2 - int(fail_due_warmup)
    assert result["metrics"]["all"]["tuning_error_count"] == int(fail_due_warmup)


def test_c_does_not_train_until_first_five_second_tick(scenario_case, recording):
    instances, options = recording
    case = replace(
        scenario_case,
        observations=tuple(
            replace(observation(index), event_time_ms=time) for index, time in enumerate((0, 1000, 5000))
        ),
    )
    result = score_scenario(case, "linear", "c", **options)
    assert [event[1] for event in instances[0].events if event[0] == "tune"] == [[2]]
    assert result["cold_start"]["first_prediction_index"] is None
    assert result["online_schedule"]["warmup_update_count"] == 0
    assert result["online_schedule"]["scheduled_update_count"] == 1
    assert result["online_schedule"]["skipped_observation_count"] == 2


def test_fixed_ticks_select_current_point_after_scoring_with_ties_and_long_gaps(scenario_case, recording):
    instances, options = recording
    times = [1000, 2000, 5999, 6000, 6000, 6001, 25999, 26000, 26000, 28000]
    case = replace(
        scenario_case,
        observations=tuple(
            replace(observation(index), event_time_ms=time, event_time="not-a-clock")
            for index, time in enumerate(times)
        ),
    )
    result = score_scenario(case, "linear", "ac", **options)
    events = instances[0].events[1:]
    assert [event[1] for event in events if event[0] == "tune"] == [[3], [6], [7]]
    # The selected current point is scored first; tied later points cannot
    # consume the same tick. A long gap submits one point, not missed intervals.
    for index in (3, 6, 7):
        update = events.index(("tune", [index]))
        assert events[update - 1][:2] == ("predict", index)
    assert events[-1][:2] == ("predict", 9)
    schedule = result["online_schedule"]
    assert schedule["scheduled_update_count"] == 3
    assert schedule["warmup_update_count"] == 0
    assert schedule["submitted_observation_count"] == 3
    assert schedule["skipped_observation_count"] == 7
    assert schedule["elapsed_ms"] == 27000


def test_short_chronological_prefix_scores_without_an_early_final_update(scenario_case, recording):
    instances, options = recording
    observations = tuple(
        replace(observation(index), event_time_ms=100_000.125 + offset)
        for index, offset in enumerate((0, 1000, 4999, 4999.5))
    )
    result = score_scenario(replace(scenario_case, observations=observations), "gbdt", "ac", **options)
    assert result["metrics"]["all"]["predicted_count"] == 4
    assert result["online_update_count"] == result["cost"]["update_count"] == 0
    assert result["cost"]["update_us"] is None
    assert all(event[0] != "tune" for event in instances[0].events)
    assert result["online_schedule"]["skipped_observation_count"] == 4
    assert result["online_schedule"]["elapsed_ms"] == 4999.5


@pytest.mark.parametrize("scenario", ["c", "ac", "bc"])
@pytest.mark.parametrize("timestamp", [None, True, float("nan"), float("inf"), -1, "1000"])
def test_online_requires_valid_timestamps_for_entire_case(scenario_case, recording, scenario, timestamp):
    instances, options = recording
    observations = list(scenario_case.observations)
    # This lies in BC's training half: validating only the testing half would
    # incorrectly claim a temporal protocol for a source with a missing clock.
    observations[0] = replace(observations[0], event_time_ms=timestamp)
    result = score_scenario(replace(scenario_case, observations=tuple(observations)), "linear", scenario, **options)
    assert result["status"] == "chronology_unavailable"
    assert not instances
    assert result["cost"]["fit_ms"] is None
    assert result["cost"]["predict_count"] == result["cost"]["update_count"] == 0
    assert result["online_schedule"]["elapsed_ms"] is None
    assert all(value == 0 for key, value in result["online_schedule"].items() if key.endswith("_count"))


@pytest.mark.parametrize("scenario", ["c", "ac", "bc"])
def test_online_rejects_decreasing_or_nonchronological_source(scenario_case, recording, scenario):
    instances, options = recording
    decreasing = list(scenario_case.observations)
    decreasing[1] = replace(decreasing[1], event_time_ms=12000)
    for case in (
        replace(scenario_case, observations=tuple(decreasing)),
        replace(scenario_case, ordering=OrderingKind.FILE_ORDER_FALLBACK),
    ):
        assert score_scenario(case, "linear", scenario, **options)["status"] == "chronology_unavailable"
    assert not instances


@pytest.mark.parametrize("scenario", ["a", "b"])
def test_frozen_protocols_do_not_require_event_time(scenario_case, recording, scenario):
    _, options = recording
    case = replace(
        scenario_case,
        ordering=OrderingKind.FILE_ORDER_FALLBACK,
        observations=tuple(replace(item, event_time_ms=None) for item in scenario_case.observations),
    )
    result = score_scenario(case, "linear", scenario, **options)
    assert result["status"] == "evaluated"
    assert "online_schedule" not in result


def test_c_cold_workers_and_stores_cannot_bypass_schedule(scenario_case, recording):
    _, options = recording

    class ScopedPredictor(RecordingPredictor):
        def __init__(self):
            super().__init__("linear", ())
            self.counts = {}

        def predict(self, features):
            key = (features.rank_payloads[0]["worker_id"], str(features.workload_kind))
            index = features.fpm_ids[0]
            value = 10 if self.counts.get(key, 0) >= 2 and index != 3 else None
            self.events.append(("predict", index, value))
            if index == 10:
                raise ValueError("a warmed store has a query error")
            return Prediction(value)

        def tune(self, iterations):
            super().tune(iterations)
            for item in iterations:
                key = (item.ranks[0].worker_id, str(item.workload_kind))
                self.counts[key] = self.counts.get(key, 0) + 1

    model = ScopedPredictor()
    identities = [("a", "decode")] * 4 + [("b", "decode")] * 3 + [("a", "prefill")] * 4
    case = replace(
        scenario_case,
        observations=tuple(
            replace(observation(index, kind, worker=worker), event_time_ms=index * 100)
            for index, (worker, kind) in enumerate(identities)
        ),
    )
    result = score_scenario(case, "linear", "c", **{**options, "factory": lambda *a, **k: model})
    assert [event[1] for event in model.events if event[0] == "tune"] == []
    schedule = result["online_schedule"]
    assert schedule["warmup_update_count"] == schedule["submitted_observation_count"] == 0
    assert schedule["scheduled_update_count"] == 0
    assert schedule["skipped_observation_count"] == 11
    assert result["metrics"]["all"]["error_count"] == 1
    assert result["metrics"]["all"]["unavailable_count"] == 10


def test_c_mixed_rank_store_cannot_train_early(scenario_case, recording):
    _, options = recording
    observations = []
    for index, prefill_tokens in enumerate((1, 1000, 1, 1000)):
        item = observation(index, "prefill")
        prefill_rank = replace(
            item.iteration.ranks[0], scheduled=RequestMetrics(num_prefill_requests=1, sum_prefill_tokens=prefill_tokens)
        )
        decode_rank = replace(
            observation(index, "decode").iteration.ranks[0],
            dp_rank=1,
        )
        observations.append(
            replace(item, iteration=ForwardPassIteration((prefill_rank, decode_rank)), event_time_ms=index * 100)
        )
    # Display categories alternate as the representative rank changes, while
    # all four observations occupy the same native cross-rank store.
    assert [str(item.workload_kind) for item in observations] == ["decode", "prefill", "decode", "prefill"]

    class SharedStorePredictor(RecordingPredictor):
        def predict(self, features):
            value = 10 if self.targets and features.fpm_ids[0] != 2 else None
            self.events.append(("predict", features.fpm_ids[0], value))
            return Prediction(value)

    model = SharedStorePredictor("linear", ())
    result = score_scenario(
        replace(scenario_case, observations=tuple(observations)),
        "linear",
        "c",
        **{**options, "factory": lambda *a, **k: model},
    )
    assert [event[1] for event in model.events if event[0] == "tune"] == []
    assert result["online_schedule"]["warmup_update_count"] == 0
    assert result["online_schedule"]["skipped_observation_count"] == 4


def test_bc_uses_floor_half_and_does_not_claim_unordered_data_is_temporal(scenario_case, recording):
    instances, options = recording
    odd_case = replace(scenario_case, observations=scenario_case.observations[:5])
    result = score_scenario(odd_case, "linear", "bc", **options)
    assert instances[0].events[0] == ("fit", [0, 1])
    assert result["training_count"] == 2 and result["test_count"] == 3
    result = score_scenario(replace(odd_case, ordering=OrderingKind.INTENTIONAL_SWEEP), "linear", "bc", **options)
    assert len(instances) == 1
    assert result["status"] == "chronology_unavailable"
    assert result["metrics"]["all"]["unavailable_count"] == 3


def test_matrix_separates_predictor_state_and_keeps_same_test_population(scenario_case, recording):
    instances, options = recording
    row = evaluate_scenario_case(scenario_case, **options)
    # This12-point fixture has no B test rows at80% or90%, so those cells never construct a predictor.
    assert len(instances) == len({id(model) for model in instances}) == 81
    assert all(model.closed for model in instances)
    for scenario in SCENARIOS:
        supported = [cell for cell in row["results"][scenario].values() if cell is not None]
        assert len({cell["scoring_population_sha256"] for cell in supported}) == 1
        assert len({cell["test_count"] for cell in supported}) == 1
    for method in ("linear", "spline"):
        assert row["results"]["c"][method]["cold_start"]["first_prediction_index"] == 11
        assert row["results"]["ac"][method]["cold_start"]["first_prediction_index"] == 0
    document = scenario_document([row], snapshot={"hf_revision": "d" * 40})
    assert document["methods"] == list(METHODS)
    assert document["scenarios"] == list(SCENARIOS)
    assert document["protocol_id"] == "fpm-scenarios-v13"
    assert row["results"]["a"]["selector"] is None
    assert all(row["split_results"][str(percent)]["b"]["selector"] is None for percent in TRAIN_PERCENTAGES)
    # Only C, A+C, and the nine B+C splits construct selector state.
    assert sum(model.method == "selector" for model in instances) == 11


def test_unsupported_is_null_but_missing_artifacts_and_dependencies_have_results(scenario_case, recording):
    instances, options = recording
    assert score_scenario(scenario_case, "warmup", "ac", **options) is None
    assert score_scenario(scenario_case, "nowarmup", "c", **options) is None
    missing = score_scenario(replace(scenario_case, fpm_artifacts=()), "warmup", "a", **options)
    assert missing["status"] == "no_fpm_input"
    assert missing["metrics"]["all"]["unavailable_count"] == 12
    assert not instances

    def unavailable(*args, **kwargs):
        raise DependencyError("native wheel does not provide this model")

    result = score_scenario(scenario_case, "gbdt", "b", **{**options, "factory": unavailable})
    assert result["status"] == "unsupported_predictor"
    assert result["metrics"]["all"]["unavailable_count"] == 6
    assert result["metrics"]["all"]["error_count"] == 0


@pytest.mark.parametrize("scenario", ["a", "b"])
def test_hybrid_is_only_applicable_to_online_scenarios(scenario_case, recording, scenario):
    instances, options = recording
    assert score_scenario(scenario_case, "hybrid", scenario, **options) is None
    assert not instances


@pytest.mark.parametrize("scenario,train_percent", [("a", 50), *(("b", p) for p in TRAIN_PERCENTAGES)])
def test_selector_frozen_scenarios_skip_partition_training_and_construction(
    scenario_case, monkeypatch, scenario, train_percent
):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported selector scenario must not partition, bootstrap, or construct a predictor")

    monkeypatch.setattr("scripts.fpm_accuracy.scenarios.split_observations", forbidden)
    assert (
        score_scenario(
            scenario_case,
            "selector",
            scenario,
            train_percent=train_percent,
            factory=forbidden,
            bootstrap=forbidden,
        )
        is None
    )


@pytest.mark.parametrize("scenario", ["ac", "bc"])
@pytest.mark.parametrize("train_percent", [10, 50, 90])
def test_hybrid_missing_store_keeps_five_second_cadence_without_warmup(
    scenario_case, recording, scenario, train_percent
):
    _, options = recording
    records = tuple(observation(index) for index in range(61))

    class UnavailableStore(RecordingPredictor):
        def predict(self, features):
            self.events.append(("predict", features.fpm_ids[0], None))
            return Prediction(None)

    models = []

    def factory(method, context, *, training_iterations, online, **kwargs):
        assert method == "hybrid" and online is True
        model = UnavailableStore(method, training_iterations)
        models.append(model)
        return model

    result = score_scenario(
        replace(scenario_case, observations=records),
        "hybrid",
        scenario,
        train_percent=train_percent,
        **{**options, "factory": factory},
    )
    first = 0 if scenario == "ac" else len(records) * train_percent // 100
    expected_updates = list(range(first + 5, len(records), 5))
    events = models[0].events
    assert [event[1] for event in events if event[0] == "tune"] == [[i] for i in expected_updates]
    for index, event in enumerate(events):
        if event[0] == "tune":
            assert events[index - 1] == ("predict", event[1][0], None)
    assert result["online_schedule"]["warmup_update_count"] == 0
    assert result["online_schedule"]["submitted_observation_count"] == len(expected_updates)
    assert result["cost"]["predict_count"] == len(records) - first
    assert result["metrics"]["all"]["coverage_pct"] == 0
    assert result["online_model"]["recipe"]["correction"]["mode"] == "hybrid"
    assert models[0].closed


def test_artifact_selection_is_stable_and_independent_of_test_accuracy(scenario_case, recording):
    _, options = recording
    artifacts = (artifact("z"), artifact("a"), artifact("0", kv_off=True))
    for candidates in (artifacts, artifacts[::-1]):
        case = replace(scenario_case, fpm_artifacts=candidates)
        assert select_artifact(case, "linear").artifact_id == "a"
        assert select_artifact(case, "warmup").artifact_id == "a"
        assert select_artifact(case, "nowarmup").artifact_id == "0"
        assert score_scenario(case, "linear", "a", **options)["artifact"]["id"] == "a"
    assert select_artifact(replace(scenario_case, fpm_artifacts=(artifacts[2],)), "linear").artifact_id == "0"


def test_prediction_and_update_failures_retain_denominator_and_cleanup(scenario_case, recording):
    _, options = recording

    class FailingPredictor(RecordingPredictor):
        def predict(self, features):
            return Prediction(float("nan") if features.fpm_ids[0] % 2 else 10)

        def tune(self, iterations):
            self.events.append(("tune", [item.fpm_ids[0] for item in iterations]))
            raise ValueError("failed update")

    model = FailingPredictor("linear", ())
    result = score_scenario(scenario_case, "linear", "c", **{**options, "factory": lambda *a, **k: model})
    metric = result["metrics"]["all"]
    assert metric["measured_count"] == 12
    assert metric["tuning_error_count"] == 2
    assert metric["error_count"] == metric["predicted_count"] == 6
    assert metric["mape_pct"] == 0
    assert metric["coverage_pct"] == 50
    assert result["online_update_count"] == 0 and result["cost"]["update_count"] == 2
    assert result["online_schedule"]["update_error_count"] == 2
    assert result["online_schedule"]["skipped_observation_count"] == 10
    assert [event[1] for event in model.events if event[0] == "tune"] == [[5], [10]]
    assert {kind: result["metrics"][kind]["tuning_error_count"] for kind in ("prefill", "decode", "mixed")} == {
        "prefill": 0,
        "decode": 1,
        "mixed": 1,
    }
    assert model.closed


def test_empty_holdout_and_invalid_stream_are_explicit(scenario_case, recording):
    instances, options = recording
    result = score_scenario(
        replace(scenario_case, observations=scenario_case.observations[:3]), "linear", "b", **options
    )
    assert result["status"] == "no_test_data"
    assert result["test_count"] == 0 and result["metrics"]["all"]["mape_pct"] is None
    assert not instances
    with pytest.raises(ValueError, match="unique IDs"):
        evaluate_scenario_case(replace(scenario_case, observations=(scenario_case.observations[0],) * 2), **options)


@pytest.fixture
def public_scenarios(scenario_case, recording):
    _, options = recording
    row = evaluate_scenario_case(scenario_case, **options)
    fixture = Path(__file__).parent / "fixtures" / "summary.json"
    snapshot = {**json.loads(fixture.read_text())["snapshot"], "configuration_count": 1}
    source = {
        **row,
        "protocol_id": "test-protocol",
        "parser_policy_id": "test-parser",
        "skipped_count": 0,
        "configuration_manifest": "data/model/config/manifest.json",
        "measurement_manifest": "data/model/config/measurements.json",
        "results": {},
    }
    del source["split_results"]
    for method, cell in (
        ("warmup", row["results"]["a"]["warmup"]),
        ("nowarmup", row["results"]["a"]["nowarmup"]),
        ("regression", row["results"]["c"]["linear"]),
    ):
        source["results"][method] = {
            "status": cell["status"],
            "artifact": cell["artifact"],
            "metrics": {
                kind: {key: value for key, value in metric.items() if key != "coverage_pct"}
                for kind, metric in cell["metrics"].items()
            },
        }
    summary = {
        "schema_version": 1,
        "snapshot": snapshot,
        "methods": ["warmup", "nowarmup", "regression"],
        "rows": [source],
    }
    return scenario_document([row], snapshot=snapshot), summary


def test_public_scenario_contract_matches_source_and_is_dependency_free(public_scenarios):
    document, summary = public_scenarios
    assert validate_scenarios(document, summary) is document
    subprocess.run(
        [sys.executable, "-S", "-c", "from scripts.fpm_accuracy.scenario_contract import validate_scenarios"],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )


def test_public_contract_preserves_and_validates_interpolation_semantics(public_scenarios):
    document, summary = public_scenarios
    evidence = {"method": "direct", "timing_only": True, "text_only": True, "metadata_recovered": False}
    document["rows"][0]["results"]["a"]["warmup"]["interpolation"] = evidence
    assert validate_scenarios(document, summary) is document
    for field, value in (("method", "auto"), ("timing_only", "true"), ("method", "sol")):
        altered = deepcopy(document)
        altered["rows"][0]["results"]["a"]["warmup"]["interpolation"][field] = value
        with pytest.raises(ValueError, match="interpolation"):
            validate_scenarios(altered, summary)


@pytest.mark.parametrize(
    "field,value",
    [
        (("snapshot", "hf_revision"), "9" * 40),
        (("rows", 0, "membership_sha256"), "9" * 64),
        (("rows", 0, "configuration_path"), "data/wrong/configuration"),
        (("methods",), ["warmup", "nowarmup", "regression"]),
        (("rows", 0, "results", "a", "linear"), None),
        (("rows", 0, "results", "c", "gbdt"), {}),
        (("rows", 0, "results", "c", "linear", "scoring_population_sha256"), "9" * 64),
        (("rows", 0, "results", "b", "linear", "training_count"), 7),
        (("rows", 0, "results", "b", "linear", "training_population_sha256"), "9" * 64),
        (("rows", 0, "results", "c", "linear", "training_count"), 1),
        (("rows", 0, "results", "a", "linear", "cost", "fit_ms"), float("nan")),
        (("rows", 0, "results", "a", "linear", "cost", "predict_count"), True),
        (("rows", 0, "results", "a", "linear", "cost", "update_count"), 12),
        (("rows", 0, "results", "a", "linear", "artifact", "path"), "data/../secret"),
        (("rows", 0, "results", "a", "linear", "reason"), "file:///private/secret"),
        (("rows", 0, "results", "a", "linear", "metrics", "all", "coverage_pct"), 0),
        (("rows", 0, "results", "a", "linear", "metrics", "all", "mape_pct"), 200),
        (("rows", 0, "results", "c", "linear", "cold_start", "first_prediction_index"), 12),
    ],
)
def test_public_contract_rejects_changed_evidence_or_misleading_outcomes(public_scenarios, field, value):
    document, summary = public_scenarios
    altered = deepcopy(document)
    target = altered
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    with pytest.raises(ValueError):
        validate_scenarios(altered, summary)
