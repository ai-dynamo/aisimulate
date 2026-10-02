# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce Gym's ordered, rank-aware cases into public overview aggregates."""

from __future__ import annotations

import hashlib
import json
import math
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from accuracy_digest import encode_points

from fpm_accuracy.dashboard.measurement_heatmaps import _axis_values, _bin_index, measurement_workload_heatmaps
from fpm_accuracy.exceptions import DependencyError
from fpm_accuracy.hf.models import MeasurementCase
from fpm_accuracy.models.aic_predictors import AicFpmPredictor, AicRegressionPredictor
from fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, PredictorContext
from fpm_accuracy.models.worker_regression import (
    WorkerRegressionPredictor,
    infer_worker_roles,
)

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
    include_points: bool = False,
    heatmaps: dict | None = None,
) -> dict:
    """Keep every eligible outcome; targets reach regression only after scoring."""
    cells = {}
    points = array("d")
    metrics = {workload: Metric() for workload in WORKLOADS}
    status = "evaluated"
    predictor = None
    construction_failed = False
    no_input = method != "regression" and artifact is None
    if no_input:
        status = "no_fpm_input"
    else:
        context = PredictorContext(
            worker=case.configuration.worker_config_record,
            worker_role=case.worker_role,
            fpm_artifact=artifact,
        )
        try:
            if method == "regression":
                roles = infer_worker_roles(item.iteration for item in case.observations)
                predictor = WorkerRegressionPredictor(context, roles, factory)
            else:
                predictor = factory("aic-fpm", context)
        except Exception as exc:
            # Raw diagnostics stay in Actions logs, never in the public JSON.
            print(
                f"{case.configuration_id} {method}: construction failed: {exc}",
                flush=True,
            )
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
            if heatmaps is not None:
                heatmap = heatmaps[workload]
                x, y = _axis_values(observation)
                key = (workload, _bin_index(x, heatmap.x_bins), _bin_index(y, heatmap.y_bins))
                cells.setdefault(key, Metric()).add(actual, predicted, error, False)
            if include_points:
                points.append(-1 if error or predicted is None else abs(predicted - actual) / actual * 100)
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
        **(
            {
                "_heatmaps": {
                    phase: {
                        **heatmap.model_dump(mode="json"),
                        "cells": [
                            {"x_index": x, "y_index": y, **metric.export()}
                            for (kind, x, y), metric in sorted(cells.items())
                            if kind == phase
                        ],
                    }
                    for phase, heatmap in heatmaps.items()
                }
            }
            if heatmaps is not None
            else {}
        ),
        **({"_points": encode_points(points)} if include_points else {}),
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
        return (
            -coverage,
            mape if mape is not None else math.inf,
            result["artifact"]["id"],
        )

    return min(results, key=rank)


def evaluate_case(
    case: MeasurementCase,
    *,
    factory: Callable = create_predictor,
    comparison: dict | None = None,
    details: list | None = None,
) -> dict:
    config = case.configuration
    ids = [item.observation_id for item in case.observations]
    orders = [item.order for item in case.observations]
    if len(ids) != len(set(ids)) or orders != sorted(set(orders)):
        raise ValueError("measurement stream must have unique IDs and strictly increasing order")
    results = {}
    variants_by_method = {}
    heatmaps = measurement_workload_heatmaps(case.observations) if details is not None else None
    if str(case.status) == "ready":
        results["regression"] = score(
            case, "regression", factory=factory, include_points=comparison is not None, heatmaps=heatmaps
        )
        for mode in METHODS[:2]:
            variants = [
                item
                for item in case.fpm_artifacts
                if ("nowarmup" if ".kv-off." in item.path.lower() else "warmup") == mode
            ]
            variants_by_method[mode] = [
                score(case, mode, item, factory=factory, include_points=comparison is not None, heatmaps=heatmaps)
                for item in variants
            ]
            results[mode] = (
                choose_variant(variants_by_method[mode])
                if variants
                else score(case, mode, factory=factory, include_points=comparison is not None, heatmaps=heatmaps)
            )
    if details is not None:
        details.append(
            {
                "configuration_id": config.configuration_id,
                "snapshot_id": config.snapshot_id,
                "membership_sha256": case.measurement_membership_sha256,
                "workload_heatmaps": {key: value.model_dump(mode="json") for key, value in heatmaps.items()},
                "methods": {
                    method: [
                        {key: value for key, value in candidate.items() if key != "_points"}
                        for candidate in (variants_by_method.get(method) or [result])
                    ]
                    for method, result in results.items()
                },
            }
        )
        for candidates in [list(results.values()), *variants_by_method.values()]:
            for candidate in candidates:
                candidate.pop("_heatmaps", None)
    if comparison is not None:
        comparison[config.configuration_id + "/" + config.snapshot_id] = {
            "order_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "methods": {method: result.pop("_points") for method, result in results.items()},
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
