# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-category loaders that turn verified local dataset files into AISim inputs.

A loader declares ``category``, accepted manifest schema versions and purposes,
``select(config) -> Selection`` (the catalog lookup for a resolved
configuration, ready for :func:`aisim_core.data.fetch.fetch`) and
``load(root, manifest_path, ...)`` (verified local files -> the structure AISim
already consumes). ``root`` is a dataset root such as ``FetchResult.root``.

Loaders never touch the network, never change performance-model behavior and are
not wired into any caller yet. They reject hash mismatches, unsupported schemas,
an inconsistent category/purpose/identity and invalid paths.

* :class:`OpLatencyLoader` materializes a ``systems`` root (``data/<system>/...``
  plus hardware specs) that can be appended to ``systems_paths``.
* :class:`FpmLoader` reuses :func:`prepare_execution_profile` for execution
  profiles and :func:`load_fpm_profile` for FPM model profiles.
* :class:`E2eLoader` returns pyarrow table views.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from aisim_core.data import (
    Artifact,
    Category,
    Identity,
    Manifest,
    _file,
    _verify,
    configuration_id,
    prepare_execution_profile,
)
from aisim_core.data.fetch import Selection
from aisim_core.data.parquet import read_parquet

MANIFEST_SCHEMA = "aisim-artifact-manifest/v1"


class LoaderError(ValueError):
    """A manifest or artifact is not acceptable to this loader."""


class Loader(ABC):
    """Contract every category loader implements."""

    category: ClassVar[Category]
    manifest_schemas: ClassVar[tuple[str, ...]] = (MANIFEST_SCHEMA,)
    # Manifest categories this loader accepts; defaults to ``(category,)``.
    manifest_categories: ClassVar[tuple[str, ...] | None] = None
    purposes: ClassVar[tuple[str, ...]]

    @abstractmethod
    def select(self, config: Mapping[str, Any]) -> Selection:
        """Catalog selection for a resolved configuration (no I/O)."""

    @abstractmethod
    def load(self, root: Path, manifest_path: str, **kwargs: Any) -> Any:
        """Verify the manifest and its files under ``root`` and build the result."""

    # -- shared helpers ------------------------------------------------------

    def _selection(self, config: Mapping[str, Any], **extra: Any) -> Selection:
        selection = Selection(
            category=self.category,
            hardware=config.get("system"),
            framework=config.get("backend"),
            framework_version=config.get("backend_version"),
            purpose=self.purposes[0] if len(self.purposes) == 1 else None,
            **extra,
        )
        selection.validate()
        return selection

    def read_manifest(self, root: Path, manifest_path: str, config: Mapping[str, Any] | None = None) -> Manifest:
        """Parse and fully verify a manifest of this category (hashes, identity, paths)."""
        root = Path(root)
        manifest = Manifest.model_validate_json(_file(root, manifest_path).read_bytes())
        if manifest.schema_version not in self.manifest_schemas:
            raise LoaderError(f"unsupported manifest schema: {manifest.schema_version}")
        categories = self.manifest_categories or (self.category,)
        if manifest.category not in categories:
            raise LoaderError(f"{type(self).__name__} cannot load a {manifest.category!r} manifest")
        if manifest.purpose not in self.purposes:
            raise LoaderError(f"{type(self).__name__} cannot load purpose {manifest.purpose!r}")
        if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
            raise LoaderError("configuration identity hash mismatch")
        for name in ("artifact_id", "path", "runtime_path"):
            values = [getattr(a, name) for a in manifest.artifacts if getattr(a, name) is not None]
            if len(values) != len(set(values)):
                raise LoaderError(f"duplicate artifact {name}")
        for artifact in manifest.artifacts:
            _verify(root, artifact)
        if config is not None:
            _check_identity(manifest.identity, config)
        return manifest


def _check_identity(identity: Identity, config: Mapping[str, Any]) -> None:
    declared = (identity.hardware, identity.framework, identity.framework_version)
    wanted = (config.get("system"), config.get("backend"), config.get("backend_version"))
    if declared != wanted:
        raise LoaderError(f"manifest identity {declared} differs from configuration {wanted}")
    model = config.get("model")
    if identity.model is not None and model is not None and identity.model != model:
        raise LoaderError(f"manifest model {identity.model!r} differs from configuration {model!r}")


# --------------------------------------------------------------------------- op-latency


@dataclass(frozen=True)
class OpLatencyData:
    """A materialized systems root, ready to append to ``systems_paths``."""

    systems_root: Path
    manifest: Manifest
    files: tuple[str, ...]  # runtime paths relative to ``systems_root``


class OpLatencyLoader(Loader):
    """Materialize verified operation tables as a ``systems/`` root.

    Only artifacts with a ``runtime_path`` are copied; the layout matches
    ``systems/data/<system>/<family>/<backend>/<version>/`` plus system specs,
    exactly what :func:`prepare_execution_profile` writes under ``systems``.
    """

    category: ClassVar[Category] = "op-latency"
    purposes: ClassVar[tuple[str, ...]] = ("operation-tables",)

    def select(self, config: Mapping[str, Any]) -> Selection:
        # Operation tables are model-independent: the model never selects them.
        return self._selection(config, operation_family=config.get("operation_family"))

    def load(
        self,
        root: Path,
        manifest_path: str,
        *,
        output_dir: Path,
        config: Mapping[str, Any] | None = None,
    ) -> OpLatencyData:
        """Copy verified runtime inputs to ``output_dir/systems`` (``output_dir`` must not exist)."""
        root, output_dir = Path(root), Path(output_dir).absolute()
        manifest = self.read_manifest(root, manifest_path, config)
        if manifest.identity.model is not None:
            raise LoaderError("operation table storage must remain model-independent")
        inputs = [a for a in manifest.artifacts if a.runtime_path is not None]
        if not inputs:
            raise LoaderError("manifest declares no runtime inputs")
        systems = output_dir / "systems"
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            for artifact in inputs:
                destination = systems / artifact.runtime_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_verify(root, artifact), destination)
                _verify(systems, artifact.model_copy(update={"path": artifact.runtime_path}))
        except BaseException:
            shutil.rmtree(output_dir)
            raise
        return OpLatencyData(systems, manifest, tuple(sorted(a.runtime_path for a in inputs)))


# --------------------------------------------------------------------------- fpm


@dataclass(frozen=True)
class FpmData:
    """Canonical ``best_available`` config and/or FPM model profile."""

    manifest: Manifest
    config: dict | None = None  # from prepare_execution_profile
    profile: Any | None = None  # FpmModelProfile from load_fpm_profile


class FpmLoader(Loader):
    """FPM data through the existing profile machinery.

    * ``execution-profile`` manifests go through :func:`prepare_execution_profile`,
      which verifies, materializes and returns the canonical config. Both the
      ``fpm`` stamp (shared-envelope layout) and the legacy ``op-latency``
      stamp are accepted; ``check_manifest`` accepts ``fpm`` for
      ``execution-profile`` manifests.
    * An ``execution-profile`` manifest whose artifacts include one
      ``fpm-profile`` JSON artifact (an FPM model profile) is read with
      :func:`load_fpm_profile` instead.
    """

    category: ClassVar[Category] = "fpm"
    manifest_categories: ClassVar[tuple[str, ...]] = ("fpm", "op-latency")
    purposes: ClassVar[tuple[str, ...]] = ("execution-profile",)

    def select(self, config: Mapping[str, Any]) -> Selection:
        return self._selection(config, model=config.get("model"))

    def load(
        self,
        root: Path,
        manifest_path: str,
        *,
        output_dir: Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> FpmData:
        root = Path(root)
        manifest = self.read_manifest(root, manifest_path, config)
        profiles = [a for a in manifest.artifacts if a.role == "fpm-profile"]
        if profiles:
            if len(profiles) != 1:
                raise LoaderError("exactly one fpm-profile artifact is required")
            return FpmData(manifest, profile=_load_fpm_profile_artifact(root, profiles[0]))
        if output_dir is None:
            raise LoaderError("output_dir is required to prepare an execution profile")
        return FpmData(manifest, config=prepare_execution_profile(root, manifest_path, Path(output_dir)))


def _load_fpm_profile_artifact(root: Path, artifact: Artifact):
    from aisim_core.fpm_profile import load_fpm_profile

    if artifact.format != "json":
        raise LoaderError("fpm-profile artifact must be JSON")
    return load_fpm_profile(_verify(root, artifact).read_text())


# --------------------------------------------------------------------------- e2e


@dataclass(frozen=True)
class E2eTables:
    """Verified Parquet artifacts as pyarrow tables keyed by ``artifact_id``."""

    manifest: Manifest
    tables: dict[str, pa.Table] = field(default_factory=dict)

    def __getitem__(self, artifact_id: str) -> pa.Table:
        return self.tables[artifact_id]


class E2eLoader(Loader):
    """E2E tables as pyarrow views; nothing is converted or materialized."""

    category: ClassVar[Category] = "e2e"
    purposes: ClassVar[tuple[str, ...]] = ("e2e-configuration",)

    def __init__(self, accepted_schemas: tuple[str, ...] | None = None):
        # None accepts exactly the schema each artifact declares in its manifest.
        self.accepted_schemas = accepted_schemas

    def select(self, config: Mapping[str, Any]) -> Selection:
        return self._selection(config, model=config.get("model"))

    def load(
        self,
        root: Path,
        manifest_path: str,
        *,
        config: Mapping[str, Any] | None = None,
        columns: Mapping[str, list[str]] | None = None,
    ) -> E2eTables:
        """Read every Parquet artifact; the stamped schema ID must match what is accepted."""
        root = Path(root)
        manifest = self.read_manifest(root, manifest_path, config)
        tables = {}
        for artifact in manifest.artifacts:
            if artifact.format != "parquet":
                continue
            if self.accepted_schemas is not None and artifact.artifact_schema not in self.accepted_schemas:
                raise LoaderError(f"unsupported table schema: {artifact.artifact_schema}")
            tables[artifact.artifact_id] = read_parquet(
                _verify(root, artifact),
                artifact.artifact_schema,
                columns=(columns or {}).get(artifact.artifact_id),
            )
        if not tables:
            raise LoaderError("manifest declares no Parquet tables")
        return E2eTables(manifest, tables)


LOADERS: dict[str, type[Loader]] = {cls.category: cls for cls in (OpLatencyLoader, FpmLoader, E2eLoader)}
