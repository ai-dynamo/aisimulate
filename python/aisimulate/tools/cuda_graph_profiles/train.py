# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk._cuda_graph_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    derived_feature_row,
)
from tools.cuda_graph_profiles.common import sha256_file

RIDGE_ALPHA = 1.0
GATES = {
    "minimum_gpu_families": 2,
    "minimum_model_identities": 2,
    "minimum_profiles": 20,
    "median_mape_max": 0.20,
    "p90_ape_max": 0.40,
    "upper_bound_coverage_min": 0.95,
    "underprediction_max": 0.20,
}


def _row_features(frame: pd.DataFrame) -> pd.DataFrame:
    records = [derived_feature_row(row) for row in frame.to_dict(orient="records")]
    return pd.DataFrame.from_records(records, index=frame.index)


def _levels(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {field: sorted({str(value) for value in frame[field].dropna().tolist()}) for field in CATEGORICAL_FEATURES}


def _feature_schema(levels: dict[str, list[str]]) -> list[str]:
    names = [f"standardized_log1p:{field}" for field in NUMERIC_FEATURES]
    for field in CATEGORICAL_FEATURES:
        names.extend(f"{field}={level}" for level in levels[field])
    return names


def _matrix(frame: pd.DataFrame, levels: dict[str, list[str]]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for field in NUMERIC_FEATURES:
        columns.append(np.log1p(frame[field].fillna(0).astype(float).to_numpy()))
    for field in CATEGORICAL_FEATURES:
        values = frame[field].fillna("unknown").astype(str).to_numpy()
        columns.extend((values == level).astype(float) for level in levels[field])
    return np.column_stack(columns)


def _normalization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros(x.shape[1])
    scales = np.ones(x.shape[1])
    numeric_count = len(NUMERIC_FEATURES)
    if len(x):
        centers[:numeric_count] = x[:, :numeric_count].mean(axis=0)
        numeric_scales = x[:, :numeric_count].std(axis=0)
        scales[:numeric_count] = np.where(numeric_scales > 1e-12, numeric_scales, 1.0)
    return centers, scales


def _standardize(x: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (x - centers) / scales


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return float(coefficients[0]), coefficients[1:]


def _cross_validate(frame: pd.DataFrame, levels: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(frame, levels)
    y = np.log1p(frame["estimated_cuda_graph_bytes"].astype(float).to_numpy())
    predictions = np.zeros(len(frame))
    model_ids = frame["model_id"].astype(str).to_numpy()
    for model_id in sorted(set(model_ids)):
        test = model_ids == model_id
        train = ~test
        if train.sum() < 2:
            predictions[test] = y[train].mean() if train.any() else y.mean()
            continue
        centers, scales = _normalization(x[train])
        intercept, coefficients = _fit(_standardize(x[train], centers, scales), y[train])
        predictions[test] = intercept + _standardize(x[test], centers, scales) @ coefficients
    return np.expm1(y), np.maximum(0, np.expm1(predictions))


def train_model(parquet_path: Path, model_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(parquet_path)
    eligible = frame.loc[
        frame["training_eligible"] & frame["estimated_cuda_graph_bytes"].notna() & ~frame["graph_disabled"]
    ].copy()
    eligible = eligible.sort_values("measurement_id").drop_duplicates("profile_id", keep="last")
    eligible = _row_features(eligible)
    levels = _levels(eligible)
    feature_schema = _feature_schema(levels)

    training_domain = {
        "model_id": sorted({str(value) for value in eligible["model_id"].dropna().tolist()}),
        **{
            field: sorted({str(value) for value in eligible[field].dropna().tolist()}) for field in CATEGORICAL_FEATURES
        },
    }
    model: dict[str, Any] = {
        "artifact_version": "cuda-graph-factorized-ridge-v2",
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categorical_levels": levels,
        "coefficients": [],
        "enabled": False,
        "feature_centers": [],
        "feature_design": "factorized-architecture-topology-v2",
        "feature_scales": [],
        "feature_schema": feature_schema,
        "gates": GATES,
        "holdout_group": "model_id",
        "intercept": None,
        "numeric_features": list(NUMERIC_FEATURES),
        "numeric_training_domain": {
            field: [float(eligible[field].min()), float(eligible[field].max())] for field in NUMERIC_FEATURES
        },
        "parquet_sha256": sha256_file(parquet_path),
        "ridge_alpha": RIDGE_ALPHA,
        "target_transform": "log1p_bytes",
        "training_domain": training_domain,
        "training_profile_count": len(eligible),
    }

    unique_models = eligible["model_id"].nunique()
    unique_gpus = eligible["gpu_family"].nunique()
    if len(eligible) >= 2:
        actual, predicted = _cross_validate(eligible, levels)
        ape = np.abs(predicted - actual) / actual
        log_residual = np.log1p(actual) - np.log1p(predicted)
        lower_residual = float(np.quantile(log_residual, 0.05))
        upper_residual = max(0.0, float(np.quantile(log_residual, 0.95, method="higher")))
        upper = np.expm1(np.log1p(predicted) + upper_residual)
        underprediction = np.maximum(0, actual - predicted) / actual
        metrics = {
            "median_mape": float(np.median(ape)),
            "p90_ape": float(np.quantile(ape, 0.90)),
            "upper_bound_coverage": float(np.mean(upper >= actual)),
            "maximum_underprediction": float(np.max(underprediction)),
        }
        model["holdout_metrics"] = metrics
        model["residual_log_interval"] = [lower_residual, upper_residual]
        gates_passed = (
            len(eligible) >= GATES["minimum_profiles"]
            and unique_models >= GATES["minimum_model_identities"]
            and unique_gpus >= GATES["minimum_gpu_families"]
            and metrics["median_mape"] <= GATES["median_mape_max"]
            and metrics["p90_ape"] <= GATES["p90_ape_max"]
            and metrics["upper_bound_coverage"] >= GATES["upper_bound_coverage_min"]
            and metrics["maximum_underprediction"] <= GATES["underprediction_max"]
        )
        x = _matrix(eligible, levels)
        centers, scales = _normalization(x)
        y = np.log1p(eligible["estimated_cuda_graph_bytes"].astype(float).to_numpy())
        intercept, coefficients = _fit(_standardize(x, centers, scales), y)
        model["intercept"] = intercept
        model["coefficients"] = [float(value) for value in coefficients]
        model["feature_centers"] = [float(value) for value in centers]
        model["feature_scales"] = [float(value) for value in scales]
        model["enabled"] = bool(gates_passed)
    else:
        model["holdout_metrics"] = None
        model["residual_log_interval"] = None

    failures = []
    if len(eligible) < GATES["minimum_profiles"]:
        failures.append(f"profiles:{len(eligible)}<{GATES['minimum_profiles']}")
    if unique_models < GATES["minimum_model_identities"]:
        failures.append(f"model_identities:{unique_models}<{GATES['minimum_model_identities']}")
    if unique_gpus < GATES["minimum_gpu_families"]:
        failures.append(f"gpu_families:{unique_gpus}<{GATES['minimum_gpu_families']}")
    if model.get("holdout_metrics"):
        metrics = model["holdout_metrics"]
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
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model


def predict_from_model(model: dict[str, Any], row: dict[str, Any]) -> float:
    """Reference implementation used by tooling tests and offline evaluation."""
    if not model["enabled"]:
        raise ValueError("model is disabled")
    features = derived_feature_row(row)
    numeric = [math.log1p(float(features.get(field) or 0)) for field in model["numeric_features"]]
    categorical = []
    for field in model["categorical_features"]:
        categorical.extend(float(str(features.get(field)) == level) for level in model["categorical_levels"][field])
    values = np.asarray(numeric + categorical, dtype=float)
    centers = np.asarray(model.get("feature_centers") or [0.0] * len(values), dtype=float)
    scales = np.asarray(model.get("feature_scales") or [1.0] * len(values), dtype=float)
    if len(values) != len(centers) or len(values) != len(scales):
        raise ValueError("model feature normalization does not match its feature schema")
    standardized = _standardize(values, centers, scales)
    log_prediction = float(model["intercept"]) + float(np.dot(model["coefficients"], standardized))
    return max(0.0, math.expm1(log_prediction))
