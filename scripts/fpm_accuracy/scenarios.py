# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible, target-blind offline and streaming FPM comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from time import perf_counter_ns

from scripts.fpm_accuracy.evaluate import WORKLOADS, Metric
from scripts.fpm_accuracy.exceptions import DependencyError
from scripts.fpm_accuracy.hf.models import FpmArtifact, MeasurementCase, MeasurementObservation
from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext
from scripts.fpm_accuracy.scenario_contract import (
    DEFAULT_TRAIN_PERCENT,
    GBDT_FROZEN_RECIPE,
    GBDT_ONLINE_RECIPE,
    HYBRID_ONLINE_RECIPE,
    METHODS,
    OFFLINE_ARTIFACT_SCENARIOS,
    ONLINE_SCENARIOS,
    ONLINE_UPDATE_INTERVAL_MS,
    PROTOCOL_ID,
    PROTOCOL_METADATA,
    SCENARIOS,
    SCHEMA_VERSION,
    SELECTOR_ONLINE_RECIPE,
    TRAIN_PERCENTAGES,
    applicable,
    validate_train_percent,
)


def split_observations(
    observations: Sequence[MeasurementObservation], scenario: str, *, train_percent: int = DEFAULT_TRAIN_PERCENT
) -> tuple[tuple[MeasurementObservation, ...], tuple[MeasurementObservation, ...]]:
    """Return only dataset-derived training rows and the fixed scoring population."""
    validate_train_percent(train_percent)
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if scenario == "b":
        counts = defaultdict(int)
        training, testing = [], []
        for observation in observations:
            kind = str(observation.workload_kind)
            index = counts[kind]
            counts[kind] += 1
            trains = ((index + 1) * train_percent + 99) // 100 > (index * train_percent + 99) // 100
            (training if trains else testing).append(observation)
        return tuple(training), tuple(testing)
    if scenario == "bc":
        boundary = len(observations) * train_percent // 100
        return tuple(observations[:boundary]), tuple(observations[boundary:])
    return (), tuple(observations)


def select_artifact(case: MeasurementCase, method: str) -> FpmArtifact | None:
    """Pin one artifact before reading target outcomes, never pick by test error."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    artifacts = sorted(case.fpm_artifacts, key=lambda item: (item.artifact_id, item.path, item.sha256))
    warmup = [artifact for artifact in artifacts if ".kv-off." not in artifact.path.lower()]
    nowarmup = [artifact for artifact in artifacts if ".kv-off." in artifact.path.lower()]
    candidates = nowarmup if method == "nowarmup" else warmup if method == "warmup" else warmup or nowarmup
    return candidates[0] if candidates else None


def _artifact_evidence(artifact: FpmArtifact | None) -> dict | None:
    if artifact is None:
        return None
    return {
        "id": artifact.artifact_id,
        "path": artifact.path,
        "sha256": artifact.sha256,
        "metadata_path": artifact.metadata_path,
        "metadata_sha256": artifact.metadata_sha256,
    }


def _population_hash(case: MeasurementCase, observations: Sequence[MeasurementObservation]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "configuration_id": case.configuration_id,
                "snapshot_id": case.configuration.snapshot_id,
                "membership_sha256": case.measurement_membership_sha256,
                "observation_ids": [observation.observation_id for observation in observations],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_observations(case: MeasurementCase) -> None:
    identities = [item.observation_id for item in case.observations]
    orders = [item.order for item in case.observations]
    if len(identities) != len(set(identities)) or orders != sorted(set(orders)):
        raise ValueError("measurement stream must have unique IDs and strictly increasing order")
    for observation in case.observations:
        if (
            not math.isfinite(observation.actual_ms)
            or observation.actual_ms <= 0
            or str(observation.workload_kind) not in WORKLOADS[1:]
        ):
            raise ValueError("invalid eligible measurement")


def _has_event_chronology(case: MeasurementCase) -> bool:
    """Require real, nondecreasing timestamps across the entire source case."""
    if str(case.ordering) != "chronological":
        return False
    previous = None
    for observation in case.observations:
        timestamp = getattr(observation, "event_time_ms", None)
        if isinstance(timestamp, bool) or not isinstance(timestamp, int | float):
            return False
        try:
            if not math.isfinite(timestamp) or timestamp < 0 or (previous is not None and timestamp < previous):
                return False
        except OverflowError:
            return False
        previous = timestamp
    return True


def score_scenario(
    case: MeasurementCase,
    method: str,
    scenario: str,
    *,
    train_percent: int = DEFAULT_TRAIN_PERCENT,
    factory: Callable | None = None,
    bootstrap: Callable | None = None,
) -> dict | None:
    """Own one fresh predictor's complete fit → predict → optional update lifetime."""
    validate_train_percent(train_percent)
    if not applicable(method, scenario):
        return None
    _validate_observations(case)
    if factory is None or bootstrap is None:
        from scripts.fpm_accuracy.models.scenario_predictors import (
            create_scenario_predictor,
            self_benchmark_iterations,
        )

        factory = factory or create_scenario_predictor
        bootstrap = bootstrap or self_benchmark_iterations
    training, testing = split_observations(case.observations, scenario, train_percent=train_percent)
    artifact = select_artifact(case, method) if scenario in OFFLINE_ARTIFACT_SCENARIOS else None
    metrics = {workload: Metric() for workload in WORKLOADS}
    predictor = None
    interpolation = None
    online_model = (
        {"recipe": HYBRID_ONLINE_RECIPE if method == "hybrid" else GBDT_ONLINE_RECIPE, "diagnostics": None}
        if method in ("gbdt", "hybrid") and scenario in ONLINE_SCENARIOS
        else None
    )
    if method == "selector":
        online_model = {"recipe": SELECTOR_ONLINE_RECIPE, "diagnostics": None}
    offline_model = (
        {"recipe": GBDT_FROZEN_RECIPE, "diagnostics": None} if method == "gbdt" and scenario in ("a", "b") else None
    )
    status, reason = "evaluated", None
    fit_ms = None
    predict_ns = update_ns = predict_count = update_count = online_update_count = 0
    first_prediction = None
    cold_unavailable = 0
    construction_failed = False
    training_iterations = tuple(observation.iteration for observation in training)
    training_count = len(training_iterations)
    online = scenario in ONLINE_SCENARIOS
    schedule = {
        "interval_ms": ONLINE_UPDATE_INTERVAL_MS,
        "time_basis": "event_time_ms",
        "warmup_policy": "scheduled_only" if scenario == "c" else "none",
        "selection_policy": "first_observation_at_or_after_tick",
        "elapsed_ms": None,
        "scheduled_update_count": 0,
        "warmup_update_count": 0,
        "update_error_count": 0,
        "submitted_observation_count": 0,
        "skipped_observation_count": 0,
    }
    if str(case.status) != "ready":
        status, reason = str(case.status), "The source case has no eligible evaluation stream."
    elif online and not _has_event_chronology(case):
        status, reason = (
            "chronology_unavailable",
            "Online replay requires finite, nondecreasing event timestamps for the complete source stream.",
        )
    elif not testing:
        status, reason = "no_test_data", "The deterministic split has no eligible test observations."
    elif scenario in ("b", "bc") and not training:
        status, reason = "no_training_data", "The deterministic split has no eligible training observations."
    elif scenario in OFFLINE_ARTIFACT_SCENARIOS and artifact is None:
        status, reason = "no_fpm_input", "No matching self-benchmark artifact is available."
    else:
        fit_start = perf_counter_ns()
        try:
            if artifact is not None:
                if method in ("warmup", "nowarmup"):
                    training_count = artifact.row_count
                else:
                    training_iterations = tuple(bootstrap(case, artifact))
                    training_count = len(training_iterations)
                    if not training_iterations:
                        status, reason = (
                            "no_training_data",
                            "The self-benchmark artifact has no eligible training rows.",
                        )
            if status == "evaluated":
                context = PredictorContext(
                    worker=case.configuration.worker_config_record,
                    worker_role=case.worker_role,
                    fpm_artifact=artifact,
                )
                online_kwargs = {"online": True} if method in ("gbdt", "hybrid", "selector") and online else {}
                if method in ("hybrid", "selector") and scenario == "c":
                    online_kwargs["cold_start"] = True
                if method in ("gbdt", "hybrid", "selector") and scenario in OFFLINE_ARTIFACT_SCENARIOS:
                    online_kwargs.update(mixed_fallback=True, scenario=scenario)
                predictor = factory(method, context, training_iterations=training_iterations, **online_kwargs)
                diagnostics = getattr(predictor, "diagnostics", None)
                if method in ("warmup", "nowarmup") and callable(diagnostics):
                    interpolation = diagnostics().get("interpolation")
        except Exception as exc:
            if predictor is not None:
                predictor.close()
                predictor = None
            print(f"{case.configuration_id} {scenario}/{method}: fit failed: {exc}", flush=True)
            construction_failed = not isinstance(exc, DependencyError)
            status = "predictor_error" if construction_failed else "unsupported_predictor"
            reason = (
                "The predictor failed during offline preparation."
                if construction_failed
                else "The installed runtime or self-benchmark format does not support this predictor."
            )
        finally:
            fit_ms = (perf_counter_ns() - fit_start) / 1_000_000

    def update(observation: MeasurementObservation, *, warmup: bool) -> None:
        nonlocal update_ns, update_count, online_update_count
        schedule["warmup_update_count" if warmup else "scheduled_update_count"] += 1
        schedule["submitted_observation_count"] += 1
        update_start = perf_counter_ns()
        try:
            predictor.tune([observation.iteration])
            online_update_count += 1
        except Exception:
            schedule["update_error_count"] += 1
            for key in ("all", str(observation.workload_kind)):
                metrics[key].tuning_error_count += 1
        finally:
            update_ns += perf_counter_ns() - update_start
            update_count += 1

    current_tick = 0
    anchor = testing[0].event_time_ms if online and predictor is not None else None
    if anchor is not None:
        schedule["elapsed_ms"] = testing[-1].event_time_ms - anchor
    try:
        for index, observation in enumerate(testing):
            predicted = None
            error = construction_failed
            update_due = False
            if predictor is not None:
                if online:
                    tick = math.floor((observation.event_time_ms - anchor) / schedule["interval_ms"])
                    if tick > current_tick:
                        # One event consumes the latest elapsed tick, including
                        # during C warmup. Empty ticks never replay older points.
                        update_due = True
                        current_tick = tick
                predict_start = perf_counter_ns()
                try:
                    predicted = predictor.predict(observation.prediction_input()).value_ms
                    if predicted is not None and (not math.isfinite(predicted) or predicted < 0):
                        raise ValueError("invalid prediction")
                except Exception:
                    predicted, error = None, True
                finally:
                    predict_ns += perf_counter_ns() - predict_start
                    predict_count += 1
                if predicted is not None and first_prediction is None:
                    first_prediction = index
                elif predicted is None and not error and first_prediction is None:
                    cold_unavailable += 1
            workload = str(observation.workload_kind)
            for key in ("all", workload):
                metrics[key].add(observation.actual_ms, predicted, error, False)
            # The current target becomes eligible only after scoring. Cold-start
            # training obeys exactly the same five-second schedule as all later
            # updates; intervening labels never enter any bootstrap buffer.
            if predictor is not None and online:
                if update_due:
                    update(observation, warmup=False)
                else:
                    schedule["skipped_observation_count"] += 1
    finally:
        if predictor is not None:
            try:
                if online_model is not None:
                    online_model["diagnostics"] = predictor.diagnostics()
                if offline_model is not None:
                    offline_model["diagnostics"] = predictor.diagnostics()
            finally:
                predictor.close()
    exported = {}
    for workload, metric in metrics.items():
        exported[workload] = {
            **metric.export(),
            "coverage_pct": metric.predicted_count / metric.measured_count * 100 if metric.measured_count else None,
        }
    return {
        "status": status,
        **({"reason": reason} if reason is not None else {}),
        **({"interpolation": interpolation} if interpolation is not None else {}),
        **({"online_model": online_model} if online_model is not None else {}),
        **({"offline_model": offline_model} if offline_model is not None else {}),
        "artifact": _artifact_evidence(artifact),
        "training_count": training_count,
        "test_count": len(testing),
        "online_update_count": online_update_count,
        "training_population_sha256": _population_hash(case, training) if training else None,
        "scoring_population_sha256": _population_hash(case, testing),
        "metrics": exported,
        "cost": {
            "fit_ms": fit_ms,
            "predict_us": predict_ns / predict_count / 1000 if predict_count else None,
            "update_us": update_ns / update_count / 1000 if update_count else None,
            "predict_count": predict_count,
            "update_count": update_count,
        },
        "cold_start": {
            "unavailable_before_first_prediction": cold_unavailable if predictor is not None else None,
            "first_prediction_index": first_prediction,
        },
        **({"online_schedule": schedule} if online else {}),
    }


def evaluate_scenario_case(
    case: MeasurementCase, *, factory: Callable | None = None, bootstrap: Callable | None = None
) -> dict:
    """Keep the full matrix, including unsupported and unavailable cells."""
    _validate_observations(case)
    config = case.configuration
    results = {
        scenario: {
            method: score_scenario(case, method, scenario, factory=factory, bootstrap=bootstrap) for method in METHODS
        }
        for scenario in SCENARIOS
    }
    split_results = {
        str(percent): {
            scenario: results[scenario]
            if percent == DEFAULT_TRAIN_PERCENT
            else {
                method: score_scenario(
                    case, method, scenario, train_percent=percent, factory=factory, bootstrap=bootstrap
                )
                for method in METHODS
            }
            for scenario in ("b", "bc")
        }
        for percent in TRAIN_PERCENTAGES
    }
    return {
        "configuration_id": config.configuration_id,
        "configuration_path": config.configuration_path,
        "snapshot_id": config.snapshot_id,
        "model": config.model_id,
        "gpu": config.gpu_family or config.system,
        "framework": config.framework,
        "framework_version": config.framework_version,
        "parallelism": config.parallelism,
        "worker_role": case.worker_role,
        "status": str(case.status),
        "ordering": str(case.ordering),
        "membership_sha256": case.measurement_membership_sha256,
        "measurement_count": len(case.observations),
        "results": results,
        "split_results": split_results,
    }


def scenario_document(rows: Sequence[dict], **metadata) -> dict:
    """Wrap rows with the immutable interpretation of this comparison protocol."""
    return {
        **metadata,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "methods": list(METHODS),
        "scenarios": list(SCENARIOS),
        "protocol_metadata": PROTOCOL_METADATA,
        "rows": list(rows),
    }
