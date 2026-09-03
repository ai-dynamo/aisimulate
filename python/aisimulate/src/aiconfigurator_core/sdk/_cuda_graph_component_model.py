# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local interpolation and structural composition for CUDA graph profiles."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from aiconfigurator_core.sdk._cuda_graph_features import CATEGORICAL_FEATURES, derived_feature_row

# vLLM shared-pool reservation composition (Apache-2.0), adapted for offline prediction:
# https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu_model_runner.py

COMPONENT_CATEGORICAL_FEATURES = ("model_id", *CATEGORICAL_FEATURES)
COMPONENT_NUMERIC_FEATURES = (
    "component_graph_count",
    "component_largest_capture_size",
    "component_second_largest_capture_size",
    "component_capture_size_sum",
    "component_capture_size_squared_sum",
    "component_capture_size_p50",
    "component_capture_size_p90",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "speculative_tokens",
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "dcp_size",
    "pcp_size",
    "moe_tp_size",
    "moe_ep_size",
    "layers_per_pipeline_rank",
    "attention_width_per_rank",
    "kv_width_per_rank",
    "dense_width_per_rank",
    "moe_width_per_rank",
    "active_expert_width_per_rank",
    "experts_per_rank",
)
COMPONENT_TARGET_FIELDS = {
    "full_first_capture": "full_first_capture_bytes",
    "full_per_graph": "full_per_graph_bytes",
    "piecewise_first_capture": "piecewise_first_capture_bytes",
    "piecewise_per_graph": "piecewise_per_graph_bytes",
}


def _capture_sizes(value: object) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        return []
    return sorted({int(item) for item in value})


def _mode_capture_sizes(row: Mapping[str, Any], mode: str) -> list[int]:
    sizes = _capture_sizes(row.get("cuda_graph_capture_sizes"))
    speculative_width = max(1, int(row.get("speculative_tokens") or 0) + 1)
    active = [size for size in sizes if speculative_width == 1 or size % speculative_width == 0] or sizes
    if mode == "piecewise":
        return active
    full_limit = int(row.get("max_num_seqs") or 0) * speculative_width
    return [size for size in active if not full_limit or size <= full_limit]


def _quantile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    return values[math.ceil((len(values) - 1) * quantile)]


def component_feature_row(row: Mapping[str, Any], component_name: str) -> dict[str, Any]:
    """Build the local interpolation coordinates for one graph component."""
    result = derived_feature_row(row)
    mode = "piecewise" if component_name.startswith("piecewise_") else "full"
    sizes = _mode_capture_sizes(row, mode)
    result.update(
        {
            "component_graph_count": len(sizes),
            "component_largest_capture_size": max(sizes, default=0),
            "component_second_largest_capture_size": sizes[-2] if len(sizes) > 1 else 0,
            "component_capture_size_sum": sum(sizes),
            "component_capture_size_squared_sum": sum(size * size for size in sizes),
            "component_capture_size_p50": _quantile(sizes, 0.50),
            "component_capture_size_p90": _quantile(sizes, 0.90),
        }
    )
    return result


def required_components(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return component names needed to reconstruct a decoder reservation."""
    features = derived_feature_row(row)
    required: list[str] = []
    for mode in ("full", "piecewise"):
        count = int(features[f"cuda_graph_{mode}_count"])
        if count > 0:
            required.append(f"{mode}_first_capture")
        if count > 1:
            required.append(f"{mode}_per_graph")
    return tuple(required)


def component_observation(
    row: Mapping[str, Any],
    component_name: str,
    *,
    target_bytes: int,
) -> dict[str, Any]:
    """Serialize one profile into the runtime-independent model artifact."""
    features = component_feature_row(row, component_name)
    return {
        "categorical": {field: str(features.get(field)) for field in COMPONENT_CATEGORICAL_FEATURES},
        "numeric": {field: float(features.get(field) or 0) for field in COMPONENT_NUMERIC_FEATURES},
        "profile_id": str(row["profile_id"]),
        "target_bytes": int(target_bytes),
    }


def interpolate_component(
    observations: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    component_name: str,
    *,
    excluded_profile_id: str | None = None,
) -> float | None:
    """Interpolate in one exact categorical cell without numeric extrapolation."""
    features = component_feature_row(row, component_name)
    categorical = {field: str(features.get(field)) for field in COMPONENT_CATEGORICAL_FEATURES}
    candidates = [
        observation
        for observation in observations
        if observation.get("profile_id") != excluded_profile_id and observation.get("categorical") == categorical
    ]
    if not candidates:
        return None

    numeric = {field: math.log1p(float(features.get(field) or 0)) for field in COMPONENT_NUMERIC_FEATURES}
    ranges: dict[str, tuple[float, float]] = {}
    for field in COMPONENT_NUMERIC_FEATURES:
        values = [math.log1p(float(observation["numeric"][field])) for observation in candidates]
        ranges[field] = (min(values), max(values))
        value = numeric[field]
        if value < ranges[field][0] - 1e-12 or value > ranges[field][1] + 1e-12:
            return None

    distances: list[tuple[float, Mapping[str, Any]]] = []
    for observation in candidates:
        squared = 0.0
        for field in COMPONENT_NUMERIC_FEATURES:
            low, high = ranges[field]
            scale = high - low
            observed = math.log1p(float(observation["numeric"][field]))
            delta = 0.0 if scale <= 1e-12 else (numeric[field] - observed) / scale
            squared += delta * delta
        distances.append((math.sqrt(squared), observation))
    distances.sort(key=lambda item: (item[0], str(item[1]["profile_id"])))

    exact = [float(observation["target_bytes"]) for distance, observation in distances if distance <= 1e-12]
    if exact:
        return max(exact)

    nearest = distances[: min(4, len(distances))]
    weights = [1.0 / (distance * distance) for distance, _ in nearest]
    log_prediction = sum(
        weight * math.log1p(float(observation["target_bytes"]))
        for weight, (_, observation) in zip(weights, nearest, strict=True)
    ) / sum(weights)
    return max(0.0, math.expm1(log_prediction))


def reconstruct_reservation_bytes(row: Mapping[str, Any], components: Mapping[str, float]) -> float | None:
    """Combine predicted components using shared first-capture pool semantics."""
    features = derived_feature_row(row)
    first_captures: list[float] = []
    incremental = 0.0
    for mode in ("full", "piecewise"):
        count = int(features[f"cuda_graph_{mode}_count"])
        if count <= 0:
            continue
        first_name = f"{mode}_first_capture"
        if first_name not in components:
            return None
        first_captures.append(float(components[first_name]))
        if count > 1:
            per_graph_name = f"{mode}_per_graph"
            if per_graph_name not in components:
                return None
            incremental += (count - 1) * float(components[per_graph_name])
    if not first_captures:
        return 0.0
    return max(first_captures) + incremental


__all__ = [
    "COMPONENT_CATEGORICAL_FEATURES",
    "COMPONENT_NUMERIC_FEATURES",
    "COMPONENT_TARGET_FIELDS",
    "component_feature_row",
    "component_observation",
    "interpolate_component",
    "reconstruct_reservation_bytes",
    "required_components",
]
