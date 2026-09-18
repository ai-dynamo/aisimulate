# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce Gym's ordered, rank-aware cases into public overview aggregates."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from fpm_accuracy.exceptions import DependencyError
from fpm_accuracy.hf.models import MeasurementCase
from fpm_accuracy.models.aic_predictors import AicFpmPredictor, AicRegressionPredictor
from fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, PredictorContext
from fpm_accuracy.models.worker_regression import WorkerRegressionPredictor, infer_worker_roles

WORKLOADS = ("all", "prefill", "decode", "mixed")
METHODS = ("warmup", "nowarmup", "regression")


@dataclass
class Metric:
    measured_count: int = 0
    predicted_count: int = 0
    unavailable_count: int = 0
    error_count: int = 0
    tuning_error_count: int = 0
    ape_sum: float = 0.0

    def add(self, actual: float, predicted: float | None, error: bool, tuning_error: bool) -> None:
        self.measured_count += 1
        self.tuning_error_count += int(tuning_error)
        if error:
            self.error_count += 1
        elif predicted is None:
            self.unavailable_count += 1
        else:
            self.predicted_count += 1
            self.ape_sum += abs(predicted - actual) / actual * 100

    def export(self) -> dict:
        fields = asdict(self)
        fields.pop("ape_sum")
        fields["mape_pct"] = self.ape_sum / self.predicted_count if self.predicted_count else None
        return fields


def create_predictor(method: str, context: PredictorContext) -> ForwardPassTimePredictor:
    if method == "regression":
        return AicRegressionPredictor.create(context)
    if method == "aic-fpm":
        return AicFpmPredictor.create(context)
    raise ValueError(f"unsupported public predictor: {method}")


def score(
    case: MeasurementCase,
    method: str,
    artifact: Any = None,
    *,
    factory: Callable = create_predictor,
) -> dict:
    """Keep every eligible outcome; targets reach regression only after scoring."""
    metrics = {workload: Metric() for workload in WORKLOADS}
    status = "evaluated"
    predictor = None
    construction_failed = False
    no_input = method != "regression" and artifact is None
    if no_input:
        status = "no_fpm_input"
    else:
        context = PredictorContext(
            worker=case.configuration.worker_config_record, worker_role=case.worker_role, fpm_artifact=artifact
        )
        try:
            if method == "regression":
                roles = infer_worker_roles(item.iteration for item in case.observations)
                predictor = WorkerRegressionPredictor(context, roles, factory)
            else:
                predictor = factory("aic-fpm", context)
        except Exception as exc:
            # Raw diagnostics stay in Actions logs, never in the public JSON.
            print(f"{case.configuration_id} {method}: construction failed: {exc}", flush=True)
            construction_failed = not isinstance(exc, DependencyError)
            status = "predictor_error" if construction_failed else "unsupported_predictor"
    try:
        for observation in case.observations:
            actual = observation.actual_ms
            workload = str(observation.workload_kind)
            if not math.isfinite(actual) or actual <= 0 or workload not in WORKLOADS[1:]:
                raise ValueError("invalid eligible measurement")
            predicted = None
            error = construction_failed
            tuning_error = False
            if predictor is not None:
                try:
                    predicted = predictor.predict(observation.prediction_input()).value_ms
                    if predicted is not None and (not math.isfinite(predicted) or predicted < 0):
                        raise ValueError("invalid prediction")
                except Exception:
                    predicted, error = None, True
            # Score before the model sees this target.
            for key in ("all", workload):
                metrics[key].add(actual, predicted, error, False)
            if predictor is not None and method == "regression":
                try:
                    predictor.tune([observation.iteration])
                except Exception:
                    tuning_error = True
                if tuning_error:
                    for key in ("all", workload):
                        metrics[key].tuning_error_count += 1
    finally:
        if predictor is not None:
            predictor.close()
    return {
        "status": status,
        "artifact": None
        if artifact is None
        else {
            "id": artifact.artifact_id,
            "path": artifact.path,
            "sha256": artifact.sha256,
            "metadata_path": artifact.metadata_path,
            "metadata_sha256": artifact.metadata_sha256,
        },
        "metrics": {key: value.export() for key, value in metrics.items()},
    }


def choose_variant(results: list[dict]) -> dict:
    def rank(result):
        metric = result["metrics"]["all"]
        coverage = metric["predicted_count"] / metric["measured_count"] if metric["measured_count"] else 0
        mape = metric["mape_pct"]
        return (-coverage, mape if mape is not None else math.inf, result["artifact"]["id"])

    return min(results, key=rank)


def evaluate_case(case: MeasurementCase, *, factory: Callable = create_predictor) -> dict:
    config = case.configuration
    ids = [item.observation_id for item in case.observations]
    orders = [item.order for item in case.observations]
    if len(ids) != len(set(ids)) or orders != sorted(set(orders)):
        raise ValueError("measurement stream must have unique IDs and strictly increasing order")
    results = {}
    if str(case.status) == "ready":
        results["regression"] = score(case, "regression", factory=factory)
        for mode in METHODS[:2]:
            variants = [
                item
                for item in case.fpm_artifacts
                if ("nowarmup" if ".kv-off." in item.path.lower() else "warmup") == mode
            ]
            results[mode] = (
                choose_variant([score(case, mode, item, factory=factory) for item in variants])
                if variants
                else score(case, mode, factory=factory)
            )
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
        "protocol_id": case.protocol_id,
        "parser_policy_id": case.parser_policy_id,
        "ordering": str(case.ordering),
        "membership_sha256": case.measurement_membership_sha256,
        "measurement_count": len(case.observations),
        "skipped_count": sum(issue.count for issue in case.issues),
        "configuration_manifest": config.manifest_path,
        "measurement_manifest": config.measurements.manifest_path,
        "results": results,
    }
