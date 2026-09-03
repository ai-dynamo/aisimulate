# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from aiconfigurator_core.sdk._cuda_graph_features import (
    MODEL_ARCHITECTURE_FEATURES,
    graph_shape_features,
    model_architecture_features,
)
from tools.cuda_graph_profiles.common import (
    PROFILE_IDENTITY_FIELDS,
    REQUIRED_PROFILE_FIELDS,
    SCHEMA_VERSION,
    measurement_id,
    normalize_capture_sizes,
    profile_id,
    sha256_file,
    stable_hash,
)
from tools.cuda_graph_profiles.infx import verify_cache
from tools.cuda_graph_profiles.parser import ParsedMeasurement, ProfileParseError, load_json, load_yaml, parse_log_text

_MODEL_ARCHITECTURE_COLUMNS = (
    "model_architecture",
    "model_architecture_config_id",
    "model_architecture_config_sha256",
    *MODEL_ARCHITECTURE_FEATURES,
)
_GRAPH_SHAPE_COLUMNS = (
    "cuda_graph_capture_count",
    "cuda_graph_active_capture_count",
    "cuda_graph_capture_size_sum",
    "cuda_graph_active_capture_size_sum",
    "cuda_graph_capture_size_squared_sum",
    "cuda_graph_capture_size_p50",
    "cuda_graph_capture_size_p90",
    "cuda_graph_largest_capture_size",
    "cuda_graph_full_count",
    "cuda_graph_full_largest_capture_size",
    "cuda_graph_piecewise_count",
    "cuda_graph_piecewise_largest_capture_size",
)
DATABASE_COLUMNS = (
    *REQUIRED_PROFILE_FIELDS,
    *_MODEL_ARCHITECTURE_COLUMNS,
    "estimated_cuda_graph_bytes",
    "estimated_cuda_graph_bytes_min_rank",
    "actual_cuda_graph_pool_bytes",
    "actual_cuda_graph_pool_bytes_min_rank",
    "rank_count",
    "component_training_eligible",
    *_GRAPH_SHAPE_COLUMNS,
    "profiled_full_count",
    "profiled_full_largest_capture_size",
    "profiled_piecewise_count",
    "profiled_piecewise_largest_capture_size",
    "profiled_encoder_count",
    "profiled_encoder_largest_capture_size",
    "full_first_capture_bytes",
    "full_first_capture_bytes_min_rank",
    "full_per_graph_bytes",
    "full_per_graph_bytes_min_rank",
    "piecewise_first_capture_bytes",
    "piecewise_first_capture_bytes_min_rank",
    "piecewise_per_graph_bytes",
    "piecewise_per_graph_bytes_min_rank",
    "encoder_cuda_graph_bytes",
    "encoder_cuda_graph_bytes_min_rank",
    "available_kv_cache_bytes",
    "gpu_kv_cache_tokens",
    "exclusion_reason",
    "component_exclusion_reason",
    "source_url",
    "source_log_sha256",
    "recipe_fingerprint",
    "concurrency",
)

_IDENTITY_REQUIRED_FOR_TRAINING = (
    "model_id",
    "system",
    "backend_version",
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "dcp_size",
    "pcp_size",
    "quantization",
    "compute_dtype",
    "kv_cache_dtype",
    "cuda_graph_mode",
    "compilation_mode",
    "compilation_backend",
    "moe_backend",
    "linear_backend",
    "flashinfer_autotune",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "attention_backend",
    "speculative_method",
)
_REQUIRED_PUBLICATION_PROVENANCE = (
    "model_id",
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
    "compilation_mode",
    "compilation_backend",
    "moe_backend",
    "linear_backend",
    "flashinfer_autotune",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "attention_backend",
    "speculative_method",
    "speculative_tokens",
    "source_repository",
    "run_id",
    "run_attempt",
    "head_sha",
    "artifact_id",
    "artifact_name",
    "artifact_sha256",
)
_BANNED_PATH_MARKERS = ("/home/", "/Users/", "/tmp/", "/mnt/", "/scratch/", "/lustre/")


class ProfileValidationError(ValueError):
    """Raised when profile publication would violate the database contract."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_root(cache_dir: Path, source: dict[str, Any], artifact: dict[str, Any]) -> Path:
    return cache_dir / str(source["run_id"]) / str(artifact["artifact_id"])


def _load_companion_benchmark(
    source: dict[str, Any], server_artifact: dict[str, Any], cache_dir: Path
) -> dict[str, Any] | None:
    matches = [
        artifact
        for artifact in source["artifacts"]
        if artifact["role"] == "benchmark" and artifact["profile_key"] == server_artifact["profile_key"]
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ProfileValidationError(f"ambiguous benchmark companion for {server_artifact['artifact_name']}")
    root = _artifact_root(cache_dir, source, matches[0])
    candidates = sorted(root.rglob("*.json"))
    if len(candidates) != 1:
        raise ProfileValidationError(
            f"benchmark artifact {matches[0]['artifact_id']} must contain exactly one JSON file"
        )
    return load_json(candidates[0])


def _load_config(root: Path) -> dict[str, Any] | None:
    candidates = sorted(root.rglob("config.yaml"))
    if not candidates:
        return None
    if len(candidates) > 1:
        hashes = {sha256_file(path) for path in candidates}
        if len(hashes) > 1:
            raise ProfileValidationError("artifact contains incompatible config.yaml files")
    return load_yaml(candidates[0])


def _measurement_text(root: Path) -> tuple[str, list[Path]]:
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".log", ".out", ".txt"}
        and "benchmark" not in path.name.lower()
        and "workload_distribution" not in path.name.lower()
    )
    selected: list[Path] = []
    chunks: list[str] = []
    needles = ("CUDA graph", "Graph capturing", "Initializing a V1 LLM engine", "enforce_eager", "enforce-eager")
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(needle in text for needle in needles):
            selected.append(path)
            chunks.append(text)
    if not chunks:
        # A config can be authoritative for an explicit graph-disabled run.
        return "", []
    return "\n".join(chunks), selected


def _rank_range(values: dict[int, int], label: str, *, enforce_compatibility: bool) -> tuple[int | None, int | None]:
    if not values:
        return None, None
    minimum = min(values.values())
    maximum = max(values.values())
    if enforce_compatibility and minimum > 0 and maximum / minimum > 1.05:
        raise ProfileValidationError(f"incompatible rank-local {label}: min={minimum}, max={maximum}")
    return minimum, maximum


def _component_rank_values(
    measurement: ParsedMeasurement,
    mode: str,
    attribute: str,
) -> dict[int, int]:
    return {
        rank: int(getattr(component, attribute))
        for rank, components in measurement.graph_components_by_rank.items()
        if (component := components.get(mode)) is not None
    }


def _complete_component_ranks(measurement: ParsedMeasurement) -> set[int]:
    required_modes = {
        mode
        for mode, count in (
            ("full", measurement.profiled_full_count),
            ("piecewise", measurement.profiled_piecewise_count),
        )
        if count
    }
    complete = {
        rank for rank, components in measurement.graph_components_by_rank.items() if required_modes.issubset(components)
    }
    if measurement.profiled_encoder_count:
        complete &= measurement.encoder_graph_bytes_by_rank.keys()
    return complete


def _validate_component_reconstruction(measurement: ParsedMeasurement) -> bool:
    if not measurement.estimated_bytes_by_rank:
        return False
    complete = _complete_component_ranks(measurement)
    if complete != measurement.estimated_bytes_by_rank.keys():
        return False
    for rank, estimated in measurement.estimated_bytes_by_rank.items():
        components = measurement.graph_components_by_rank[rank]
        first_capture = max((component.first_capture_bytes for component in components.values()), default=0)
        incremental = sum(
            max(component.graph_count - 1, 0) * component.per_graph_bytes for component in components.values()
        )
        reconstructed = first_capture + incremental + measurement.encoder_graph_bytes_by_rank.get(rank, 0)
        tolerance = max(16 << 20, round(estimated * 0.05))
        if abs(reconstructed - estimated) > tolerance:
            raise ProfileValidationError(
                "CUDA graph components do not reconstruct the logged reservation "
                f"on rank {rank}: reconstructed={reconstructed}, estimated={estimated}"
            )
    return True


def _row_from_measurement(
    measurement: ParsedMeasurement,
    *,
    lock: dict[str, Any],
    source: dict[str, Any],
    artifact: dict[str, Any],
    log_files: list[Path],
    benchmark: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    estimated_min, estimated_max = _rank_range(
        measurement.estimated_bytes_by_rank, "reservation", enforce_compatibility=True
    )
    actual_min, actual_max = _rank_range(
        measurement.actual_bytes_by_rank, "pool measurement", enforce_compatibility=False
    )
    component_ranges = {}
    for mode in ("full", "piecewise"):
        for attribute in ("first_capture_bytes", "per_graph_bytes"):
            component_ranges[f"{mode}_{attribute}"] = _rank_range(
                _component_rank_values(measurement, mode, attribute),
                f"{mode} {attribute}",
                enforce_compatibility=True,
            )
    encoder_range = _rank_range(
        measurement.encoder_graph_bytes_by_rank,
        "encoder graph memory",
        enforce_compatibility=True,
    )
    component_reconstructs = _validate_component_reconstruction(measurement)
    identity = dict(measurement.identity)
    identity.setdefault("backend", "vllm")
    identity.setdefault("pp_size", 1)
    identity.setdefault("attention_dp_size", 1)
    identity.setdefault("dcp_size", 1)
    identity.setdefault("pcp_size", 1)
    identity.setdefault("moe_ep_size", 1)
    identity["moe_tp_size"] = identity.get("moe_tp_size") or identity.get("tp_size")
    identity["backend_build"] = identity.get("backend_build") or (benchmark or {}).get("image")
    identity["cuda_graph_capture_sizes"] = normalize_capture_sizes(identity.get("cuda_graph_capture_sizes"))
    architecture = model_architecture_features(identity.get("model_id"))
    graph_shape = graph_shape_features(identity)
    identity_completeness = (
        "pinned" if identity.get("model_revision") or identity.get("model_config_sha256") else "unversioned_model"
    )
    missing_training = [field for field in _IDENTITY_REQUIRED_FOR_TRAINING if identity.get(field) is None]
    missing_training.extend(field for field in _MODEL_ARCHITECTURE_COLUMNS if architecture.get(field) is None)
    training_eligible = bool(estimated_max is not None and not measurement.graph_disabled and not missing_training)
    component_training_eligible = bool(
        training_eligible and component_reconstructs and not measurement.profiled_encoder_count
    )
    exclusion_reason = None
    if measurement.graph_disabled:
        exclusion_reason = "graph_disabled"
    elif estimated_max is None:
        exclusion_reason = "actual_only_legacy_log"
    elif missing_training:
        exclusion_reason = "incomplete_training_identity:" + ",".join(missing_training)
    component_exclusion_reason = None
    if not component_training_eligible:
        if exclusion_reason:
            component_exclusion_reason = exclusion_reason
        elif measurement.profiled_encoder_count:
            component_exclusion_reason = "encoder_component_model_unsupported"
        else:
            component_exclusion_reason = "missing_component_breakdown"

    row: dict[str, Any] = {
        **{field: identity.get(field) for field in PROFILE_IDENTITY_FIELDS},
        "schema_version": SCHEMA_VERSION,
        "graph_disabled": measurement.graph_disabled,
        "training_eligible": training_eligible,
        "identity_completeness": identity_completeness,
        "source_repository": lock["repository"],
        "run_id": int(source["run_id"]),
        "run_attempt": int(source["run_attempt"]),
        "head_sha": source["head_sha"],
        "artifact_id": int(artifact["artifact_id"]),
        "artifact_name": artifact["artifact_name"],
        "artifact_sha256": artifact["extracted_content_sha256"],
        **architecture,
        "estimated_cuda_graph_bytes": estimated_max,
        "estimated_cuda_graph_bytes_min_rank": estimated_min,
        "actual_cuda_graph_pool_bytes": actual_max,
        "actual_cuda_graph_pool_bytes_min_rank": actual_min,
        "rank_count": max(
            len(measurement.estimated_bytes_by_rank),
            len(measurement.actual_bytes_by_rank),
            int(identity.get("tp_size") or 1),
        ),
        "component_training_eligible": component_training_eligible,
        **graph_shape,
        "profiled_full_count": measurement.profiled_full_count,
        "profiled_full_largest_capture_size": measurement.profiled_full_largest_capture_size,
        "profiled_piecewise_count": measurement.profiled_piecewise_count,
        "profiled_piecewise_largest_capture_size": measurement.profiled_piecewise_largest_capture_size,
        "profiled_encoder_count": measurement.profiled_encoder_count,
        "profiled_encoder_largest_capture_size": measurement.profiled_encoder_largest_capture_size,
        **{field: bounds[1] for field, bounds in component_ranges.items()},
        **{f"{field}_min_rank": bounds[0] for field, bounds in component_ranges.items()},
        "encoder_cuda_graph_bytes": encoder_range[1],
        "encoder_cuda_graph_bytes_min_rank": encoder_range[0],
        "available_kv_cache_bytes": measurement.available_kv_cache_bytes,
        "gpu_kv_cache_tokens": measurement.gpu_kv_cache_tokens,
        "exclusion_reason": exclusion_reason,
        "component_exclusion_reason": component_exclusion_reason,
        "source_url": source["html_url"],
        "source_log_sha256": stable_hash([sha256_file(path) for path in log_files]),
        "recipe_fingerprint": (benchmark or {}).get("recipe_fingerprint"),
        "concurrency": (benchmark or {}).get("conc"),
    }
    row["profile_id"] = profile_id(row)
    row["measurement_id"] = measurement_id(row)
    for field in (
        "full_count",
        "full_largest_capture_size",
        "piecewise_count",
        "piecewise_largest_capture_size",
    ):
        observed = row[f"profiled_{field}"]
        derived = row[f"cuda_graph_{field}"]
        if observed is not None and observed != derived:
            raise ProfileValidationError(f"logged CUDA graph {field}={observed} disagrees with derived value {derived}")
    reconciliation = {
        "artifact_id": row["artifact_id"],
        "identity_sources": measurement.identity_sources,
        "model_architecture_config_id": row["model_architecture_config_id"],
        "model_architecture_config_sha256": row["model_architecture_config_sha256"],
        "measurement_id": row["measurement_id"],
        "profile_id": row["profile_id"],
    }
    return row, reconciliation


def parse_locked_sources(lock_path: Path, cache_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_cache(lock, cache_dir)
    rows = []
    reconciliations = []
    for source in lock["sources"]:
        for artifact in source["artifacts"]:
            if artifact["role"] != "server_logs":
                continue
            root = _artifact_root(cache_dir, source, artifact)
            benchmark = _load_companion_benchmark(source, artifact, cache_dir)
            config = _load_config(root)
            text, log_files = _measurement_text(root)
            try:
                measurement = parse_log_text(
                    text,
                    benchmark=benchmark,
                    config=config,
                    artifact_name=artifact["artifact_name"],
                )
            except ProfileParseError as exc:
                raise ProfileValidationError(f"cannot parse artifact {artifact['artifact_id']}: {exc}") from exc
            row, reconciliation = _row_from_measurement(
                measurement,
                lock=lock,
                source=source,
                artifact=artifact,
                log_files=log_files,
                benchmark=benchmark,
            )
            rows.append(row)
            reconciliations.append(reconciliation)
    if not rows:
        raise ProfileValidationError("no server-log profiles were parsed")
    _validate_rows(rows)
    return rows, reconciliations


def _validate_duplicate_profiles(rows: list[dict[str, Any]]) -> None:
    fields = (
        "estimated_cuda_graph_bytes",
        "full_first_capture_bytes",
        "full_per_graph_bytes",
        "piecewise_first_capture_bytes",
        "piecewise_per_graph_bytes",
        "encoder_cuda_graph_bytes",
    )
    for field in fields:
        grouped: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            value = row.get(field)
            if value is not None:
                grouped[row["profile_id"]].append(int(value))
        for key, values in grouped.items():
            if len(values) > 1 and min(values) > 0 and max(values) / min(values) > 1.05:
                raise ProfileValidationError(
                    f"semantic profile {key} {field} differs by more than 5%: {min(values)}..{max(values)}"
                )


def _validate_component_row(row: dict[str, Any]) -> None:
    component_fields = (
        "full_first_capture_bytes",
        "full_per_graph_bytes",
        "piecewise_first_capture_bytes",
        "piecewise_per_graph_bytes",
        "encoder_cuda_graph_bytes",
    )
    if any(row.get(field) is not None and int(row[field]) < 0 for field in component_fields):
        raise ProfileValidationError("CUDA graph component bytes must be non-negative")
    if not row["component_training_eligible"]:
        if not row.get("component_exclusion_reason"):
            raise ProfileValidationError("component-ineligible profile must include an exclusion reason")
        return
    if row.get("component_exclusion_reason"):
        raise ProfileValidationError("component-eligible profile must not include an exclusion reason")
    if row.get("encoder_cuda_graph_bytes") is not None:
        raise ProfileValidationError("encoder graph profile cannot train the V3 decoder model")

    first_captures = []
    incremental = 0
    for mode in ("full", "piecewise"):
        count = int(row[f"cuda_graph_{mode}_count"])
        if count <= 0:
            continue
        first_field = f"{mode}_first_capture_bytes"
        if row.get(first_field) is None:
            raise ProfileValidationError(f"component-eligible profile is missing {first_field}")
        first_captures.append(int(row[first_field]))
        if count > 1:
            per_graph_field = f"{mode}_per_graph_bytes"
            if row.get(per_graph_field) is None:
                raise ProfileValidationError(f"component-eligible profile is missing {per_graph_field}")
            incremental += (count - 1) * int(row[per_graph_field])
    if not first_captures:
        raise ProfileValidationError("component-eligible profile contains no decoder graphs")
    reconstructed = max(first_captures) + incremental
    estimated = int(row["estimated_cuda_graph_bytes"])
    tolerance = max(16 << 20, round(estimated * 0.05))
    if abs(reconstructed - estimated) > tolerance:
        raise ProfileValidationError("published CUDA graph components do not reconstruct the reservation")


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        missing = [field for field in DATABASE_COLUMNS if field not in row]
        if missing:
            raise ProfileValidationError(f"profile is missing required columns: {missing}")
        missing_provenance = [field for field in _REQUIRED_PUBLICATION_PROVENANCE if row.get(field) is None]
        if missing_provenance:
            raise ProfileValidationError(f"profile has missing provenance: {missing_provenance}")
        if row["profile_id"] != profile_id(row):
            raise ProfileValidationError(f"profile hash mismatch: {row['profile_id']}")
        if row["measurement_id"] != measurement_id(row):
            raise ProfileValidationError(f"measurement hash mismatch: {row['measurement_id']}")
        if row["backend"] != "vllm":
            raise ProfileValidationError(f"unsupported backend in profile: {row['backend']}")
        if row["graph_disabled"] and row["estimated_cuda_graph_bytes"] != 0:
            raise ProfileValidationError("disabled graph profile must record a zero reservation")
        _validate_component_row(row)
        rendered = json.dumps(row, sort_keys=True)
        if any(marker in rendered for marker in _BANNED_PATH_MARKERS):
            raise ProfileValidationError(f"profile {row['measurement_id']} contains an internal filesystem path")
    _validate_duplicate_profiles(rows)


def validate_database(database_dir: Path, *, validate_model: bool = True) -> dict[str, Any]:
    parquet_path = database_dir / "cuda_graph_profiles.parquet"
    metadata_path = database_dir / "cuda_graph_profiles.metadata.json"
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise ProfileValidationError("database requires Parquet and metadata artifacts")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("parquet_sha256") != sha256_file(parquet_path):
        raise ProfileValidationError("Parquet checksum does not match metadata")
    table = pq.read_table(parquet_path)
    missing = sorted(set(DATABASE_COLUMNS) - set(table.column_names))
    if missing:
        raise ProfileValidationError(f"Parquet is missing required columns: {missing}")
    rows = table.to_pylist()
    _validate_rows(rows)
    if len(rows) != metadata.get("measurement_count"):
        raise ProfileValidationError("metadata measurement count does not match Parquet")
    if sum(bool(row["component_training_eligible"]) for row in rows) != metadata.get("component_training_count"):
        raise ProfileValidationError("metadata component training count does not match Parquet")
    model_path = database_dir / "cuda_graph_reservation_model.json"
    model = None
    if model_path.is_file():
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if validate_model and model.get("parquet_sha256") != metadata["parquet_sha256"]:
            raise ProfileValidationError("model artifact was not trained from the packaged Parquet")
    return {
        "measurement_count": len(rows),
        "model_enabled": bool(model.get("enabled")) if model else None,
        "model_version": model.get("artifact_version") if model else None,
        "parquet_sha256": metadata["parquet_sha256"],
        "profile_count": len({row["profile_id"] for row in rows}),
        "status": "valid",
    }


def publish_database(lock_path: Path, cache_dir: Path, database_dir: Path, reports_dir: Path) -> dict[str, Any]:
    database_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows, reconciliations = parse_locked_sources(lock_path, cache_dir)
    rows.sort(key=lambda row: row["measurement_id"])
    table = pa.Table.from_pylist([{column: row.get(column) for column in DATABASE_COLUMNS} for row in rows])
    parquet_path = database_dir / "cuda_graph_profiles.parquet"
    pq.write_table(table, parquet_path, compression="zstd", use_dictionary=False, write_statistics=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    metadata = {
        "database_version": "v1",
        "disabled_profile_count": sum(row["graph_disabled"] for row in rows),
        "identity_fields": list(PROFILE_IDENTITY_FIELDS),
        "lock_sha256": sha256_file(lock_path),
        "manifest_sha256": lock["manifest_sha256"],
        "measurement_count": len(rows),
        "parquet_sha256": sha256_file(parquet_path),
        "profile_count": len({row["profile_id"] for row in rows}),
        "reservation_training_count": sum(row["training_eligible"] for row in rows),
        "component_training_count": sum(row["component_training_eligible"] for row in rows),
        "schema_version": SCHEMA_VERSION,
        "source_run_ids": sorted({row["run_id"] for row in rows}),
    }
    _write_json(database_dir / "cuda_graph_profiles.metadata.json", metadata)
    _write_json(
        reports_dir / "source_mapping.report.json",
        {
            "measurements": [
                {
                    "artifact_id": row["artifact_id"],
                    "head_sha": row["head_sha"],
                    "measurement_id": row["measurement_id"],
                    "profile_id": row["profile_id"],
                    "run_attempt": row["run_attempt"],
                    "run_id": row["run_id"],
                }
                for row in rows
            ]
        },
    )
    _write_json(reports_dir / "reconciliation.report.json", {"measurements": reconciliations})
    _write_json(
        reports_dir / "exclusions.report.json",
        {
            "measurements": [
                {
                    "component_exclusion_reason": row["component_exclusion_reason"],
                    "exclusion_reason": row["exclusion_reason"],
                    "measurement_id": row["measurement_id"],
                }
                for row in rows
                if row["exclusion_reason"] or row["component_exclusion_reason"]
            ]
        },
    )
    validation = validate_database(database_dir, validate_model=False)
    _write_json(reports_dir / "validation.report.json", validation)
    return metadata
