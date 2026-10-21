# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared public/private dataset contracts and offline profile preparation.

The catalog is additive: existing FPM catalogs retain their own protocols.
This module never authenticates, downloads, publishes, or runs stored scripts.
"""

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Category = Literal["op-latency", "fpm", "e2e"]


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError("expected a canonical repository-relative path")
    return value


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class Artifact(_Record):
    artifact_id: str = Field(min_length=1)
    path: str
    role: str = Field(min_length=1)
    format: str = Field(min_length=1)
    artifact_schema: str = Field(alias="schema", min_length=1)
    sha256: Digest
    bytes: int = Field(ge=0)
    rows: int | None = Field(default=None, ge=0)
    # Only runtime inputs have a destination in the materialized systems root.
    runtime_path: str | None = None

    @field_validator("path", "runtime_path")
    @classmethod
    def relative_paths(cls, value):
        return None if value is None else _relative(value)


class Identity(_Record):
    hardware: str
    framework: str
    framework_version: str
    operation_family: str | None
    model: str | None
    effective_settings: dict[str, JsonValue]
    unresolved_identity: dict[str, JsonValue] | None


class Manifest(_Record):
    schema_version: Literal["aisim-artifact-manifest/v1"]
    category: Category
    purpose: Literal["operation-tables", "execution-profile"]
    configuration_id: str = Field(pattern=r"^cfg-[0-9a-f]{64}$")
    identity: Identity
    requested_settings: dict[str, JsonValue] | None
    artifacts: list[Artifact] = Field(min_length=1)
    provenance: dict[str, JsonValue]
    evidence_gaps: list[dict[str, JsonValue]]


class Entry(_Record):
    configuration_id: str = Field(pattern=r"^cfg-[0-9a-f]{64}$")
    purpose: Literal["operation-tables", "execution-profile"]
    identity: Identity
    manifest: Artifact


class CategoryIndex(_Record):
    schema_version: Literal["aisim-category-index/v1"]
    category: Category
    entries: list[Entry]


class CategoryReference(_Record):
    root: str
    index: Artifact
    layout: Literal["category", "legacy-fpm"]

    _path = field_validator("root")(_relative)


class DatasetIndex(_Record):
    schema_version: Literal["aisim-dataset-index/v1"]
    dataset_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    categories: dict[Category, CategoryReference]


def configuration_id(category: str, purpose: str, identity: Identity) -> str:
    """Hash effective identity, including explicit nulls and unresolved source facts."""
    payload = {"version": 1, "category": category, "purpose": purpose, "identity": identity.model_dump()}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "cfg-" + hashlib.sha256(canonical.encode()).hexdigest()


def _file(root: Path, relative: str) -> Path:
    path = root
    for part in PurePosixPath(_relative(relative)).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in dataset path: {relative}")
    if not path.is_file():
        raise ValueError(f"missing dataset artifact: {relative}")
    return path


def _verify(root: Path, artifact: Artifact) -> Path:
    path = _file(root, artifact.path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != artifact.bytes or digest != artifact.sha256:
        raise ValueError(f"artifact integrity mismatch: {artifact.path}")
    if artifact.format == "parquet" and (
        artifact.rows is None or pq.ParquetFile(path).metadata.num_rows != artifact.rows
    ):
        raise ValueError(f"Parquet row count mismatch: {artifact.path}")
    return path


def check_manifest(root: Path, path: str) -> Manifest:
    """Check one manifest and all declared inputs without network access."""
    root = Path(root)
    manifest = Manifest.model_validate_json(_file(root, path).read_bytes())
    if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
        raise ValueError("configuration identity hash mismatch")
    if manifest.category != "op-latency":
        raise ValueError("use the existing category protocol for FPM/E2E manifests")
    if manifest.purpose == "operation-tables" and manifest.identity.model is not None:
        raise ValueError("operation table storage must remain model-independent")
    for field in ("artifact_id", "path", "runtime_path"):
        values = [getattr(a, field) for a in manifest.artifacts if getattr(a, field) is not None]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate artifact {field}")
    for artifact in manifest.artifacts:
        _verify(root, artifact)
    return manifest


def check_dataset(root: Path) -> DatasetIndex:
    """Verify shared discovery indexes; legacy FPM payload validation stays separate."""
    root = Path(root)
    index = DatasetIndex.model_validate_json(_file(root, "catalog/dataset.json").read_bytes())
    for category, reference in index.categories.items():
        path = _verify(root, reference.index)
        if reference.layout == "legacy-fpm":
            legacy = json.loads(path.read_bytes())
            if category != "fpm" or legacy.get("catalog_version") not in {5, 6, 7}:
                raise ValueError("unsupported legacy FPM index")
            continue
        entries = CategoryIndex.model_validate_json(path.read_bytes())
        if entries.category != category:
            raise ValueError("category index identity mismatch")
        seen = set()
        for entry in entries.entries:
            if entry.configuration_id in seen:
                raise ValueError("duplicate configuration ID")
            seen.add(entry.configuration_id)
            _verify(root, entry.manifest)
            manifest = check_manifest(root, entry.manifest.path)
            if (manifest.category, manifest.configuration_id, manifest.identity, manifest.purpose) != (
                category,
                entry.configuration_id,
                entry.identity,
                entry.purpose,
            ):
                raise ValueError("catalog and manifest identity mismatch")
    return index


def prepare_execution_profile(root: Path, manifest_path: str, output_dir: Path) -> dict:
    """Materialize verified local inputs and return canonical best_available config.

    output_dir must not exist. Canonical construction validates the compiled
    basis and retained runtime inputs without changing profile bytes or pins.
    Model metadata must be bundled or in an existing local model directory.
    Callers keep both the output and any local model directory for later reloads.
    """
    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk.common import DefaultHFModels

    root, output_dir = Path(root), Path(output_dir).absolute()
    manifest = check_manifest(root, manifest_path)
    if manifest.purpose != "execution-profile":
        raise ValueError("an execution-profile manifest is required")
    profiles = [a for a in manifest.artifacts if a.role == "execution-profile"]
    configs = [a for a in manifest.artifacts if a.role == "predictor-config"]
    if len(profiles) != 1 or len(configs) != 1:
        raise ValueError("exactly one profile and predictor config are required")
    profile, config_artifact = profiles[0], configs[0]
    profile_bytes = _verify(root, profile).read_bytes()
    config_bytes = _verify(root, config_artifact).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != config_artifact.sha256:
        raise ValueError("configuration changed during preparation")
    config = json.loads(config_bytes)
    model = config.get("model")
    if isinstance(model, str) and model not in DefaultHFModels and not Path(model).is_dir():
        raise ValueError(
            "offline profile preparation requires bundled model metadata or an existing local model directory"
        )
    selection = config["estimator_config"]["op_level"]["execution_profile"]
    if selection["sha256"] != profile.sha256:
        raise ValueError("configuration profile pin mismatch")
    declared_identity = (manifest.identity.hardware, manifest.identity.framework, manifest.identity.framework_version)
    if declared_identity != (config.get("system"), config.get("backend"), config.get("backend_version")):
        raise ValueError("manifest identity differs from predictor configuration")
    inputs = {a.runtime_path: a for a in manifest.artifacts if a.runtime_path is not None}
    retained = json.loads(profile_bytes)["retained_files"]
    declared = {f["path"]: (f["sha256"], f["size_bytes"]) for f in retained}
    actual = {p: (a.sha256, a.bytes) for p, a in inputs.items()}
    if len(declared) != len(retained) or declared != actual:
        raise ValueError("runtime inputs differ from the pinned profile inventory")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        for relative, artifact in inputs.items():
            destination = output_dir / "systems" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_verify(root, artifact), destination)
            _verify(output_dir / "systems", artifact.model_copy(update={"path": relative}))
        destination = output_dir / "profile.json"
        destination.write_bytes(profile_bytes)
        _verify(output_dir, profile.model_copy(update={"path": "profile.json"}))
        config["systems_paths"] = [str(output_dir / "systems")]
        selection["path"] = str(destination)
        # Use the same Rust-owned admission/readiness checks as prediction;
        # manifest/profile agreement alone cannot establish runtime completeness.
        RustForwardPassPerfModel.best_available(config)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return config
