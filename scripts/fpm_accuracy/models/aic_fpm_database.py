# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Stage an immutable Hugging Face FPM parquet pair for AISim."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import yaml

from fpm_accuracy.contract import strict_json
from fpm_accuracy.exceptions import ConfigurationError, DependencyError


class FpmArtifactSource(Protocol):
    artifact_id: str
    local_path: Path
    local_metadata_path: Path
    sha256: str


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
    """Overlay the exact HF parquet pair without rebuilding it from measurements."""

    parquet_path = Path(artifact.local_path).resolve()
    metadata_path = Path(artifact.local_metadata_path).resolve()
    if not parquet_path.is_file() or not metadata_path.is_file():
        raise ConfigurationError(f"FPM artifact {artifact.artifact_id!r} is not materialized locally")
    metadata = _read_metadata(metadata_path)

    temporary = TemporaryDirectory(prefix="aisim-fpm-")
    systems_root = Path(temporary.name)
    try:
        target_dir = _prepare_system_overlay(systems_root, engine_config)
        (target_dir / "fpm_forward_perf.parquet").symlink_to(parquet_path)
        (target_dir / "fpm_forward_perf.metadata.json").symlink_to(metadata_path)
    except Exception:
        temporary.cleanup()
        raise
    return PreparedAicFpmDatabase(
        systems_root=systems_root,
        diagnostics={
            "fpm_artifact_id": artifact.artifact_id,
            "fpm_parquet_sha256": artifact.sha256,
            "fpm_row_count": metadata.get("row_count"),
        },
        _temporary_directory=temporary,
    )


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
        from aisimulate_core.sdk import perf_database
    except ImportError as exc:
        raise DependencyError(
            "AISim FPM prediction requires aisimulate; install the AISim checkout in this environment"
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
