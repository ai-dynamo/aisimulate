# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.cuda_graph_profiles.common import backend_family, sha256_file

NUMERIC_FEATURES = (
    "cuda_graph_capture_count",
    "cuda_graph_largest_capture_size",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "dcp_size",
    "pcp_size",
    "moe_tp_size",
    "moe_ep_size",
    "speculative_tokens",
)
CATEGORICAL_FEATURES = (
    "model_id",
    "gpu_family",
    "vllm_family",
    "quantization",
    "compute_dtype",
    "kv_cache_dtype",
    "cuda_graph_mode",
    "attention_backend",
    "speculative_method",
)
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
    frame = frame.copy()
    frame["gpu_family"] = frame["system"].str.replace("_sxm", "", regex=False)
    frame["vllm_family"] = frame["backend_version"].map(backend_family)
    return frame


def _levels(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {field: sorted({str(value) for value in frame[field].dropna().tolist()}) for field in CATEGORICAL_FEATURES}


def _feature_schema(levels: dict[str, list[str]]) -> list[str]:
    names = [f"log1p:{field}" for field in NUMERIC_FEATURES]
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
    fold_ids = np.array([int(value[-2:], 16) % 5 for value in frame["profile_id"]])
    for fold in sorted(set(fold_ids)):
        test = fold_ids == fold
        train = ~test
        if train.sum() < 2:
            predictions[test] = y[train].mean() if train.any() else y.mean()
            continue
        intercept, coefficients = _fit(x[train], y[train])
        predictions[test] = intercept + x[test] @ coefficients
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

    model: dict[str, Any] = {
        "artifact_version": "cuda-graph-ridge-v1",
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categorical_levels": levels,
        "coefficients": [],
        "enabled": False,
        "feature_schema": feature_schema,
        "gates": GATES,
        "intercept": None,
        "numeric_features": list(NUMERIC_FEATURES),
        "parquet_sha256": sha256_file(parquet_path),
        "ridge_alpha": RIDGE_ALPHA,
        "target_transform": "log1p_bytes",
        "training_domain": {
            field: sorted({str(value) for value in eligible[field].dropna().tolist()}) for field in CATEGORICAL_FEATURES
        },
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
        y = np.log1p(eligible["estimated_cuda_graph_bytes"].astype(float).to_numpy())
        intercept, coefficients = _fit(x, y)
        model["intercept"] = intercept
        model["coefficients"] = [float(value) for value in coefficients]
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
    """Reference implementation used by tooling tests."""
    if not model["enabled"]:
        raise ValueError("model is disabled")
    numeric = [math.log1p(float(row.get(field) or 0)) for field in model["numeric_features"]]
    categorical = []
    for field in model["categorical_features"]:
        categorical.extend(float(str(row.get(field)) == level) for level in model["categorical_levels"][field])
    log_prediction = float(model["intercept"]) + float(np.dot(model["coefficients"], numeric + categorical))
    return max(0.0, math.expm1(log_prediction))
