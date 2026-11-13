# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Stage an immutable Hugging Face FPM parquet pair for AISim."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import yaml

from scripts.fpm_accuracy.contract import strict_json
from scripts.fpm_accuracy.exceptions import ConfigurationError, DependencyError


class FpmArtifactSource(Protocol):
    artifact_id: str
    local_path: Path
    local_metadata_path: Path
    sha256: str
    metadata_sha256: str


@dataclass(slots=True)
class PreparedAicFpmDatabase:
    """A temporary AISim systems root and its source provenance."""

    systems_root: Path
    diagnostics: dict[str, Any]
    _temporary_directory: TemporaryDirectory[str]

    def close(self) -> None:
        self._temporary_directory.cleanup()


def prepare_aic_fpm_database(
    engine_config: Mapping[str, Any],
    artifact: FpmArtifactSource,
) -> PreparedAicFpmDatabase:
    """Stage the pinned table, completing only verified reconstructed declarations.

    Original files remain immutable. Native validation still owns whether the
    staged table supports the requested estimator and workload.
    """

    parquet_path = Path(artifact.local_path).resolve()
    metadata_path = Path(artifact.local_metadata_path).resolve()
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise ConfigurationError(f"FPM artifact {artifact.artifact_id!r} is not materialized locally")
    metadata = _read_metadata(metadata_path)
    if metadata.get("schema_version") == 7:
        raise DependencyError("FPM schema v7 execution identity is not yet supported by the Gym worker adapter.")

    metadata, recovered = _recover_reconstructed_metadata(metadata, artifact)
    recovery = None
    if recovered:
        metadata_text = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
        recovery = {
            "policy": "schema6_reconstructed_row_contract_v1",
            "source_metadata_sha256": artifact.metadata_sha256,
            "staged_metadata_sha256": hashlib.sha256(metadata_text.encode()).hexdigest(),
            "recovered_fields": recovered,
        }
    temporary = TemporaryDirectory(prefix="aisim-fpm-")
    systems_root = Path(temporary.name)
    try:
        target_dir = _prepare_system_overlay(systems_root, engine_config)
        (target_dir / "fpm_forward_perf.parquet").symlink_to(parquet_path)
        staged_metadata = target_dir / "fpm_forward_perf.metadata.json"
        if recovery is not None:
            staged_metadata.write_text(metadata_text, encoding="utf-8")
        else:
            staged_metadata.symlink_to(metadata_path)
    except Exception:
        temporary.cleanup()
        raise
    return PreparedAicFpmDatabase(
        systems_root=systems_root,
        diagnostics={
            "fpm_artifact_id": artifact.artifact_id,
            "fpm_parquet_sha256": artifact.sha256,
            "fpm_row_count": metadata.get("row_count"),
            **({"fpm_metadata_recovery": recovery} if recovery is not None else {}),
        },
        _temporary_directory=temporary,
    )


def _recover_reconstructed_metadata(
    metadata: dict[str, Any], artifact: FpmArtifactSource
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Complete historical schema-6 sidecars using explicit, verified row facts.

    The schema-6 collector's row writer and sidecar publisher declare balanced
    iteration coordinates and the single-sample policy (collector/fpm_forward/
    database.py at e8828036e3d32d8361e2abb819e0ee60e569f518). This compatibility
    boundary accepts only that contract, not inferred producer history. It keeps
    provenance, unknown revisions and every existing declaration unchanged.
    Work is one digest pass and a bounded scan of contract/selector columns at
    preparation time; no measured latencies are loaded or rewritten here.
    """
    if metadata.get("schema_version") != 6 or metadata.get("metadata_provenance") != "reconstructed_from_parquet":
        return metadata, {}
    declarations = (
        "coordinate_system",
        "measurement_policy",
        "system",
        "backend",
        "backend_version",
    )
    if all(name in metadata for name in declarations):
        # Complete sidecars need no reconstruction proof. The native loader
        # validates their declared contract. Legacy single-sample tables may
        # omit per-row policy/repeat columns and sidecar repeat counts.
        return metadata, {}
    if type(metadata["schema_version"]) is not int or metadata.get("schema_name") != "aic_fpm_forward_perf":
        raise ConfigurationError("Reconstructed FPM sidecar requires the explicit schema-6 contract")
    parquet_path = Path(artifact.local_path)
    if (
        _file_sha256(parquet_path) != artifact.sha256
        or metadata.get("parquet_sha256") != artifact.sha256
        or _file_sha256(Path(artifact.local_metadata_path)) != artifact.metadata_sha256
    ):
        raise ConfigurationError("Reconstructed FPM sources differ from their pinned hashes")
    selector = metadata.get("configuration_selector")
    identity_fields = ("model_path", "system", "backend", "backend_version")
    if not isinstance(selector, dict) or not set(identity_fields) <= selector.keys():
        raise ConfigurationError("Reconstructed FPM sidecar requires an explicit configuration selector")
    row_count = metadata.get("row_count")
    if type(row_count) is not int or row_count <= 0:
        raise ConfigurationError("Reconstructed FPM sidecar requires a positive integer row_count")
    coordinate_fields = ("batch_size", "total_prefill_tokens", "total_kv_read_tokens")
    columns = list(
        dict.fromkeys(
            (
                *selector,
                *coordinate_fields,
                "partition_policy",
                "measurement_policy",
                "measurement_repeats",
                "warmup_repeats",
            )
        )
    )
    import pyarrow.parquet as pq

    try:
        parquet = pq.ParquetFile(parquet_path)
        if parquet.metadata.num_rows != row_count:
            raise ConfigurationError("Reconstructed FPM sidecar row_count differs from parquet")
        if not set(columns) <= set(parquet.schema_arrow.names):
            raise ConfigurationError("Reconstructed FPM sidecar requires explicit schema-6 row contract columns")
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            for row in batch.to_pylist():
                for name, expected in selector.items():
                    if type(row[name]) is not type(expected) or row[name] != expected:
                        raise ConfigurationError(f"Reconstructed FPM sidecar selector conflicts with row {name}")
                if any(not isinstance(row[name], str) or not row[name] for name in identity_fields):
                    raise ConfigurationError("Reconstructed FPM rows require explicit nonempty deployment identity")
                if row["partition_policy"] != "balanced_v1":
                    raise ConfigurationError("Reconstructed FPM rows require balanced_v1 partition_policy")
                if (
                    any(type(row[name]) is not int or row[name] < 0 for name in coordinate_fields)
                    or row["batch_size"] == 0
                ):
                    raise ConfigurationError("Reconstructed FPM rows have invalid iteration-total coordinates")
                if (
                    row["measurement_policy"] != "dynamo_native_single_sample_v1"
                    or type(row["measurement_repeats"]) is not int
                    or row["measurement_repeats"] != 1
                    or type(row["warmup_repeats"]) is not int
                    or row["warmup_repeats"] != 0
                ):
                    raise ConfigurationError("Reconstructed FPM rows require the explicit single-sample policy")
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError("Cannot validate reconstructed FPM row contracts") from exc
    facts = {
        "coordinate_system": "iteration_totals_balanced_v1",
        "measurement_policy": "dynamo_native_single_sample_v1",
        "measurement_repeats": 1,
        "warmup_repeats": 0,
        **{name: selector[name] for name in identity_fields[1:]},
    }
    for name, value in {**selector, **facts}.items():
        if name in metadata and (type(metadata[name]) is not type(value) or metadata[name] != value):
            raise ConfigurationError(f"Reconstructed FPM sidecar conflicts with row-derived {name}")
    recovered = {name: value for name, value in facts.items() if name not in metadata}
    return {**metadata, **recovered}, recovered


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        raw = strict_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot read FPM sidecar {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"FPM sidecar {path} must contain a JSON object")
    return raw


def _prepare_system_overlay(systems_root: Path, engine_config: Mapping[str, Any]) -> Path:
    try:
        from aisim_core.sdk import perf_database
    except ImportError as exc:
        raise DependencyError(
            "AISim FPM prediction requires aisim; install the AISim checkout in this environment"
        ) from exc

    system_name = str(engine_config["system_name"])
    source_yaml = next(
        (
            Path(root) / f"{system_name}.yaml"
            for root in perf_database.get_systems_paths()
            if (Path(root) / f"{system_name}.yaml").is_file()
        ),
        None,
    )
    if source_yaml is None:
        raise ConfigurationError(f"AISim has no system definition for {system_name!r}")
    target_yaml = systems_root / source_yaml.name
    shutil.copy2(source_yaml, target_yaml)
    system_spec = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
    if not isinstance(system_spec, dict) or not isinstance(system_spec.get("data_dir"), str):
        raise ConfigurationError(f"AISim system definition {source_yaml} has no data_dir")

    data_dir = _safe_relative_path(system_spec["data_dir"], field="system data_dir")
    source_data_root = (source_yaml.parent / data_dir).resolve()
    target_data_root = systems_root / data_dir
    backend = _safe_path_component(engine_config["backend"], field="backend")
    version = _safe_path_component(engine_config["backend_version"], field="backend version")
    target_version = target_data_root / backend / version
    target_version.mkdir(parents=True)
    if source_data_root.is_dir():
        _link_overlay_entries(source_data_root, target_data_root, backend, version)
    return target_version


def _link_overlay_entries(source_root: Path, target_root: Path, backend: str, version: str) -> None:
    """Expose op/SOL data while keeping the selected FPM directory isolated."""

    for source_entry in source_root.iterdir():
        if source_entry.name != backend:
            (target_root / source_entry.name).symlink_to(source_entry, target_is_directory=source_entry.is_dir())
    source_backend = source_root / backend
    target_backend = target_root / backend
    if not source_backend.is_dir():
        return
    for source_entry in source_backend.iterdir():
        if source_entry.name != version:
            (target_backend / source_entry.name).symlink_to(source_entry, target_is_directory=source_entry.is_dir())
    source_version = source_backend / version
    target_version = target_backend / version
    if not source_version.is_dir():
        return
    for source_entry in source_version.iterdir():
        if source_entry.name not in {"fpm_forward_perf.parquet", "fpm_forward_perf.metadata.json"}:
            (target_version / source_entry.name).symlink_to(source_entry, target_is_directory=source_entry.is_dir())


def _safe_relative_path(value: Any, *, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ConfigurationError(f"AISim {field} must be a safe relative path")
    return path


def _safe_path_component(value: Any, *, field: str) -> str:
    component = str(value)
    if component in {"", ".", ".."} or Path(component).name != component:
        raise ConfigurationError(f"AISim {field} must be one safe path component")
    return component


__all__ = ["FpmArtifactSource", "PreparedAicFpmDatabase", "prepare_aic_fpm_database"]
