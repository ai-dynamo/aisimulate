# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from aiconfigurator_core.sdk._cuda_graph_component_model import (
    COMPONENT_CATEGORICAL_FEATURES,
    COMPONENT_NUMERIC_FEATURES,
    COMPONENT_TARGET_FIELDS,
    component_observation,
    interpolate_component,
    reconstruct_reservation_bytes,
    required_components,
)
from tools.cuda_graph_profiles.common import sha256_file

GATES = {
    "minimum_gpu_families": 2,
    "minimum_holdout_coverage": 0.80,
    "minimum_model_identities": 2,
    "minimum_profiles": 20,
    "median_mape_max": 0.20,
    "p90_ape_max": 0.40,
    "upper_bound_coverage_min": 0.95,
    "underprediction_max": 0.20,
}


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    index = math.ceil((len(ordered) - 1) * quantile)
    return ordered[index]


def _eligible_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    eligible = frame.loc[
        frame["component_training_eligible"] & frame["estimated_cuda_graph_bytes"].notna() & ~frame["graph_disabled"]
    ].copy()
    eligible = eligible.sort_values("measurement_id").drop_duplicates("profile_id", keep="last")
    return eligible.to_dict(orient="records")


def _component_artifacts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for component_name, target_field in COMPONENT_TARGET_FIELDS.items():
        observations = [
            component_observation(row, component_name, target_bytes=int(row[target_field]))
            for row in rows
            if component_name in required_components(row) and row.get(target_field) is not None
        ]
        observations.sort(key=lambda observation: str(observation["profile_id"]))
        artifacts[component_name] = {
            "observation_count": len(observations),
            "observations": observations,
            "target_field": target_field,
        }
    return artifacts


def _predict_components(
    components: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    excluded_profile_id: str | None = None,
) -> float | None:
    predictions: dict[str, float] = {}
    for component_name in required_components(row):
        component = components.get(component_name)
        if component is None:
            return None
        prediction = interpolate_component(
            component["observations"],
            row,
            component_name,
            excluded_profile_id=excluded_profile_id,
        )
        if prediction is None:
            return None
        predictions[component_name] = prediction
    return reconstruct_reservation_bytes(row, predictions)


def _cross_validate(
    rows: list[dict[str, Any]], components: dict[str, dict[str, Any]]
) -> tuple[list[float], list[float]]:
    actual: list[float] = []
    predicted: list[float] = []
    for row in rows:
        prediction = _predict_components(components, row, excluded_profile_id=str(row["profile_id"]))
        if prediction is None:
            continue
        actual.append(float(row["estimated_cuda_graph_bytes"]))
        predicted.append(float(round(prediction)))
    return actual, predicted


def _training_domain(components: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[Any]]]:
    observations = [observation for component in components.values() for observation in component["observations"]]
    return {
        "categorical": {
            field: sorted({str(observation["categorical"][field]) for observation in observations})
            for field in COMPONENT_CATEGORICAL_FEATURES
        },
        "numeric": {
            field: (
                [
                    min(float(observation["numeric"][field]) for observation in observations),
                    max(float(observation["numeric"][field]) for observation in observations),
                ]
                if observations
                else []
            )
            for field in COMPONENT_NUMERIC_FEATURES
        },
    }


def train_model(parquet_path: Path, model_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(parquet_path)
    rows = _eligible_rows(frame)
    components = _component_artifacts(rows)
    actual, predicted = _cross_validate(rows, components)
    holdout_coverage = len(predicted) / len(rows) if rows else 0.0
    unique_models = len({str(row["model_id"]) for row in rows})
    unique_gpus = len({str(row["system"]).removesuffix("_sxm") for row in rows})

    model: dict[str, Any] = {
        "artifact_version": "cuda-graph-component-interpolation-v3",
        "categorical_features": list(COMPONENT_CATEGORICAL_FEATURES),
        "components": components,
        "enabled": False,
        "feature_design": "component-local-log-distance-v3",
        "gates": GATES,
        "holdout_group": "profile_id",
        "holdout_prediction_count": len(predicted),
        "holdout_prediction_coverage": holdout_coverage,
        "numeric_features": list(COMPONENT_NUMERIC_FEATURES),
        "parquet_sha256": sha256_file(parquet_path),
        "reconstruction": "max(full_first_capture,piecewise_first_capture)+sum((count-1)*per_graph)",
        "target_transform": "log1p_bytes",
        "training_domain": _training_domain(components),
        "training_profile_count": len(rows),
    }

    if actual:
        ape = [abs(prediction - target) / target for target, prediction in zip(actual, predicted, strict=True)]
        log_residual = [
            math.log1p(target) - math.log1p(prediction) for target, prediction in zip(actual, predicted, strict=True)
        ]
        lower_residual = _quantile(log_residual, 0.05)
        upper_residual = max(0.0, _quantile(log_residual, 0.95))
        upper = [float(round(math.expm1(math.log1p(value) + upper_residual))) for value in predicted]
        underprediction = [
            max(0.0, target - prediction) / target for target, prediction in zip(actual, predicted, strict=True)
        ]
        metrics = {
            "maximum_underprediction": max(underprediction),
            "median_mape": _quantile(ape, 0.50),
            "p90_ape": _quantile(ape, 0.90),
            "upper_bound_coverage": sum(bound >= target for bound, target in zip(upper, actual, strict=True))
            / len(actual),
        }
        model["holdout_metrics"] = metrics
        model["residual_log_interval"] = [lower_residual, upper_residual]
    else:
        metrics = None
        model["holdout_metrics"] = None
        model["residual_log_interval"] = None

    failures = []
    if len(rows) < GATES["minimum_profiles"]:
        failures.append(f"profiles:{len(rows)}<{GATES['minimum_profiles']}")
    if unique_models < GATES["minimum_model_identities"]:
        failures.append(f"model_identities:{unique_models}<{GATES['minimum_model_identities']}")
    if unique_gpus < GATES["minimum_gpu_families"]:
        failures.append(f"gpu_families:{unique_gpus}<{GATES['minimum_gpu_families']}")
    if holdout_coverage < GATES["minimum_holdout_coverage"]:
        failures.append(f"holdout_coverage:{holdout_coverage:.6f}")
    if metrics is not None:
        for metric, gate, comparator in (
            ("median_mape", "median_mape_max", "max"),
            ("p90_ape", "p90_ape_max", "max"),
            ("upper_bound_coverage", "upper_bound_coverage_min", "min"),
            ("maximum_underprediction", "underprediction_max", "max"),
        ):
            failed = metrics[metric] > GATES[gate] if comparator == "max" else metrics[metric] < GATES[gate]
            if failed:
                failures.append(f"{metric}:{metrics[metric]:.6f}")
    model["gate_failures"] = failures
    model["enabled"] = not failures
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model


def predict_from_model(model: dict[str, Any], row: dict[str, Any]) -> float:
    """Reference implementation used by tooling tests and offline evaluation."""
    if not model["enabled"]:
        raise ValueError("model is disabled")
    prediction = _predict_components(model["components"], row)
    if prediction is None:
        raise ValueError("row is outside the component interpolation domain")
    return prediction
