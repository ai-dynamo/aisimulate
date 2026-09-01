# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline CUDA graph reservation lookup and prediction."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

CudaGraphReservationSource = Literal["disabled", "profile", "modeled", "unavailable"]

_PROFILE_IDENTITY_FIELDS = (
    "model_id",
    "model_revision",
    "model_config_sha256",
    "system",
    "backend",
    "backend_version",
    "backend_build",
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "dcp_size",
    "pcp_size",
    "moe_tp_size",
    "moe_ep_size",
    "quantization",
    "compute_dtype",
    "kv_cache_dtype",
    "cuda_graph_mode",
    "cuda_graph_capture_sizes",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "attention_backend",
    "speculative_method",
    "speculative_tokens",
)


class CudaGraphProfileDatabaseError(ValueError):
    """The profile database is missing, corrupt, or incompatible."""


@dataclass(frozen=True, slots=True)
class CudaGraphReservationRequest:
    """Execution identity used to resolve a rank-local vLLM reservation."""

    model_id: str
    system: str
    backend: str = "vllm"
    backend_version: str | None = None
    backend_build: str | None = None
    model_revision: str | None = None
    model_config_sha256: str | None = None
    tp_size: int = 1
    pp_size: int = 1
    attention_dp_size: int = 1
    dcp_size: int = 1
    pcp_size: int = 1
    moe_tp_size: int | None = None
    moe_ep_size: int = 1
    quantization: str | None = None
    compute_dtype: str | None = None
    kv_cache_dtype: str | None = None
    cuda_graph_enabled: bool = True
    cuda_graph_mode: str | None = None
    cuda_graph_capture_sizes: tuple[int, ...] = ()
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
    attention_backend: str | None = None
    speculative_method: str = "none"
    speculative_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.system:
            raise ValueError("system must not be empty")
        integer_fields = (
            "tp_size",
            "pp_size",
            "attention_dp_size",
            "dcp_size",
            "pcp_size",
            "moe_ep_size",
        )
        if any(getattr(self, field) < 1 for field in integer_fields):
            raise ValueError("parallelism sizes must be positive")
        if self.moe_tp_size is not None and self.moe_tp_size < 1:
            raise ValueError("moe_tp_size must be positive")
        if self.speculative_tokens < 0:
            raise ValueError("speculative_tokens must be non-negative")
        sizes = tuple(sorted({int(value) for value in self.cuda_graph_capture_sizes}))
        if any(value < 1 for value in sizes):
            raise ValueError("CUDA graph capture sizes must be positive")
        object.__setattr__(self, "cuda_graph_capture_sizes", sizes)


@dataclass(frozen=True, slots=True)
class CudaGraphReservationEstimate:
    """Conservative reservation result with explicit provenance."""

    reservation_bytes: int | None
    central_estimate_bytes: int | None
    interval_lower_bytes: int | None
    interval_upper_bytes: int | None
    source: CudaGraphReservationSource
    profile_version: str | None
    model_version: str | None
    profile_ids: tuple[str, ...]
    identity_completeness: Literal["pinned", "unversioned_model"]
    miss_reason: str | None


@dataclass(frozen=True, slots=True)
class _Database:
    metadata: dict[str, object]
    model: dict[str, object]
    rows: tuple[dict[str, object], ...]


def _validate_enabled_model(model: dict[str, object]) -> None:
    if not model.get("enabled"):
        return
    try:
        gates = model["gates"]
        metrics = model["holdout_metrics"]
        domain = model["training_domain"]
        checks = (
            int(model["training_profile_count"]) >= int(gates["minimum_profiles"]),
            len(domain["model_id"]) >= int(gates["minimum_model_identities"]),
            len(domain["gpu_family"]) >= int(gates["minimum_gpu_families"]),
            float(metrics["median_mape"]) <= float(gates["median_mape_max"]),
            float(metrics["p90_ape"]) <= float(gates["p90_ape_max"]),
            float(metrics["upper_bound_coverage"]) >= float(gates["upper_bound_coverage_min"]),
            float(metrics["maximum_underprediction"]) <= float(gates["underprediction_max"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CudaGraphProfileDatabaseError("enabled CUDA graph model has incomplete safety-gate evidence") from exc
    if not all(checks):
        raise CudaGraphProfileDatabaseError("enabled CUDA graph model does not pass its declared safety gates")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_database_dir() -> Path:
    return Path(str(files("aiconfigurator_core") / "systems/cuda_graph_profiles/v1"))


def _database_dir(database_path: str | Path | None) -> Path:
    if database_path is None:
        return _default_database_dir()
    path = Path(database_path)
    return path.parent if path.name == "cuda_graph_profiles.parquet" else path


def _load_database(database_path: str | Path | None) -> _Database:
    root = _database_dir(database_path)
    parquet_path = root / "cuda_graph_profiles.parquet"
    metadata_path = root / "cuda_graph_profiles.metadata.json"
    model_path = root / "cuda_graph_reservation_model.json"
    if not all(path.is_file() for path in (parquet_path, metadata_path, model_path)):
        raise CudaGraphProfileDatabaseError(f"incomplete CUDA graph profile database: {root}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CudaGraphProfileDatabaseError("cannot read CUDA graph profile metadata") from exc
    parquet_sha256 = _sha256(parquet_path)
    if metadata.get("parquet_sha256") != parquet_sha256:
        raise CudaGraphProfileDatabaseError("CUDA graph profile Parquet checksum mismatch")
    if model.get("parquet_sha256") != parquet_sha256:
        raise CudaGraphProfileDatabaseError("CUDA graph model checksum does not match the Parquet database")
    _validate_enabled_model(model)
    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:
        raise CudaGraphProfileDatabaseError("cannot read CUDA graph profile Parquet") from exc
    missing = sorted(set(_PROFILE_IDENTITY_FIELDS) - set(table.column_names))
    if missing:
        raise CudaGraphProfileDatabaseError(f"CUDA graph profile database is missing columns: {missing}")
    return _Database(metadata=metadata, model=model, rows=tuple(table.to_pylist()))


def _identity_completeness(request: CudaGraphReservationRequest) -> Literal["pinned", "unversioned_model"]:
    if request.model_revision or request.model_config_sha256:
        return "pinned"
    return "unversioned_model"


def _capture_sizes(request: CudaGraphReservationRequest) -> str:
    return json.dumps(list(request.cuda_graph_capture_sizes), separators=(",", ":"))


def _request_identity(request: CudaGraphReservationRequest) -> dict[str, object]:
    return {
        "model_id": request.model_id,
        "model_revision": request.model_revision,
        "model_config_sha256": request.model_config_sha256,
        "system": request.system,
        "backend": request.backend,
        "backend_version": request.backend_version,
        "backend_build": request.backend_build,
        "tp_size": request.tp_size,
        "pp_size": request.pp_size,
        "attention_dp_size": request.attention_dp_size,
        "dcp_size": request.dcp_size,
        "pcp_size": request.pcp_size,
        "moe_tp_size": request.moe_tp_size or request.tp_size,
        "moe_ep_size": request.moe_ep_size,
        "quantization": request.quantization,
        "compute_dtype": request.compute_dtype,
        "kv_cache_dtype": request.kv_cache_dtype,
        "cuda_graph_mode": request.cuda_graph_mode,
        "cuda_graph_capture_sizes": _capture_sizes(request),
        "max_num_seqs": request.max_num_seqs,
        "max_num_batched_tokens": request.max_num_batched_tokens,
        "max_model_len": request.max_model_len,
        "attention_backend": request.attention_backend,
        "speculative_method": request.speculative_method,
        "speculative_tokens": request.speculative_tokens,
    }


def _exact_matches(database: _Database, request: CudaGraphReservationRequest) -> list[dict[str, object]]:
    identity = _request_identity(request)
    return [
        row
        for row in database.rows
        if row.get("estimated_cuda_graph_bytes") is not None
        and not row.get("graph_disabled")
        and all(row.get(field) == identity[field] for field in _PROFILE_IDENTITY_FIELDS)
    ]


def _backend_family(version: str | None) -> str | None:
    if not version:
        return None
    pieces = version.split(".")
    if len(pieces) >= 2 and all(piece.isdigit() for piece in pieces[:2]):
        return ".".join(pieces[:2])
    return version.split("+")[0]


def _model_features(request: CudaGraphReservationRequest) -> dict[str, object]:
    sizes = request.cuda_graph_capture_sizes
    return {
        **_request_identity(request),
        "cuda_graph_capture_count": len(sizes),
        "cuda_graph_largest_capture_size": max(sizes, default=0),
        "gpu_family": request.system.removesuffix("_sxm"),
        "vllm_family": _backend_family(request.backend_version),
    }


def _modeled_estimate(
    database: _Database, request: CudaGraphReservationRequest, completeness: Literal["pinned", "unversioned_model"]
) -> CudaGraphReservationEstimate | None:
    model = database.model
    if not model.get("enabled"):
        return None
    row = _model_features(request)
    for field, allowed in model["training_domain"].items():
        if row.get(field) is None or str(row[field]) not in allowed:
            return None
    if any(row.get(field) is None for field in model["numeric_features"]):
        return None

    values = [math.log1p(float(row[field])) for field in model["numeric_features"]]
    for field in model["categorical_features"]:
        values.extend(float(str(row[field]) == level) for level in model["categorical_levels"][field])
    coefficients = model["coefficients"]
    if len(coefficients) != len(values):
        raise CudaGraphProfileDatabaseError("CUDA graph model feature schema does not match its coefficients")
    weighted_features = zip(coefficients, values, strict=True)
    log_point = float(model["intercept"]) + sum(float(coefficient) * value for coefficient, value in weighted_features)
    point = max(0, round(math.expm1(log_point)))
    residual_interval = model.get("residual_log_interval")
    if not isinstance(residual_interval, list) or len(residual_interval) != 2:
        raise CudaGraphProfileDatabaseError("CUDA graph model is missing its calibrated interval")
    lower = max(0, round(math.expm1(log_point + float(residual_interval[0]))))
    upper = max(point, round(math.expm1(log_point + float(residual_interval[1]))))
    return CudaGraphReservationEstimate(
        reservation_bytes=upper,
        central_estimate_bytes=point,
        interval_lower_bytes=lower,
        interval_upper_bytes=upper,
        source="modeled",
        profile_version=str(database.metadata.get("database_version")),
        model_version=str(model.get("artifact_version")),
        profile_ids=(),
        identity_completeness=completeness,
        miss_reason=None,
    )


def estimate_cuda_graph_reservation(
    request: CudaGraphReservationRequest,
    *,
    database_path: str | Path | None = None,
) -> CudaGraphReservationEstimate:
    """Resolve an exact vLLM reservation or a validated in-domain upper bound."""
    completeness = _identity_completeness(request)
    if not request.cuda_graph_enabled:
        return CudaGraphReservationEstimate(
            reservation_bytes=0,
            central_estimate_bytes=0,
            interval_lower_bytes=0,
            interval_upper_bytes=0,
            source="disabled",
            profile_version=None,
            model_version=None,
            profile_ids=(),
            identity_completeness=completeness,
            miss_reason=None,
        )
    if request.backend.lower() != "vllm":
        return CudaGraphReservationEstimate(
            reservation_bytes=None,
            central_estimate_bytes=None,
            interval_lower_bytes=None,
            interval_upper_bytes=None,
            source="unavailable",
            profile_version=None,
            model_version=None,
            profile_ids=(),
            identity_completeness=completeness,
            miss_reason="unsupported_backend",
        )

    database = _load_database(database_path)
    matches = _exact_matches(database, request)
    if matches:
        reservation = max(int(row["estimated_cuda_graph_bytes"]) for row in matches)
        return CudaGraphReservationEstimate(
            reservation_bytes=reservation,
            central_estimate_bytes=reservation,
            interval_lower_bytes=reservation,
            interval_upper_bytes=reservation,
            source="profile",
            profile_version=str(database.metadata.get("database_version")),
            model_version=None,
            profile_ids=tuple(sorted({str(row["profile_id"]) for row in matches})),
            identity_completeness=completeness,
            miss_reason=None,
        )
    modeled = _modeled_estimate(database, request, completeness)
    if modeled is not None:
        return modeled
    reason = "model_disabled" if not database.model.get("enabled") else "out_of_domain"
    return CudaGraphReservationEstimate(
        reservation_bytes=None,
        central_estimate_bytes=None,
        interval_lower_bytes=None,
        interval_upper_bytes=None,
        source="unavailable",
        profile_version=str(database.metadata.get("database_version")),
        model_version=str(database.model.get("artifact_version")),
        profile_ids=(),
        identity_completeness=completeness,
        miss_reason=reason,
    )


__all__ = [
    "CudaGraphProfileDatabaseError",
    "CudaGraphReservationEstimate",
    "CudaGraphReservationRequest",
    "estimate_cuda_graph_reservation",
]
