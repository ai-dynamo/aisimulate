# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed system spec copies for the dataset and the drift check.

Packaged ``aisim_core/systems/<system>.yaml`` stays the source of truth.
The dataset keeps an immutable copy per spec version::

    data/systems/<system>/<system>.<spec_sha256[:12]>.yaml
    data/systems/<system>/manifest.yaml

``spec_sha256`` is computed here, never stored in the YAML. It hashes the
canonical spec: the parsed mapping serialized as JSON with sorted keys and no
whitespace, so comments, key order and formatting do not change it. Published
copies are written in that normalized form (sorted keys, no comments).

This module is offline: it never downloads or publishes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_HASH_PREFIX_LEN = 12
MANIFEST_NAME = "manifest.yaml"
PACKAGED_SYSTEMS_DIR = Path(__file__).resolve().parent.parent / "systems"

_SYSTEM_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
# A packaged YAML is a system spec only if it carries these sections; other
# YAML in systems/ (query_versions, reuse manifests, lane defaults) is not.
_SPEC_SECTIONS = ("gpu", "node")


class SystemSpecError(ValueError):
    """A system spec, copy or manifest is invalid or out of sync."""


def _load_mapping(text: str, origin: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SystemSpecError(f"{origin}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemSpecError(f"{origin}: expected a YAML mapping")
    return data


def canonical_spec_bytes(spec: Mapping[str, Any]) -> bytes:
    """Canonical serialization: sorted keys, compact JSON, no comments."""
    try:
        return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise SystemSpecError(f"spec is not canonically serializable: {exc}") from exc


def spec_sha256(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_spec_bytes(spec)).hexdigest()


def spec_file_sha256(path: Path) -> str:
    """Loader-computed hash of a spec YAML file (packaged or published copy)."""
    return spec_sha256(_load_mapping(Path(path).read_text(encoding="utf-8"), str(path)))


def is_system_spec(spec: Mapping[str, Any]) -> bool:
    return all(isinstance(spec.get(s), dict) for s in _SPEC_SECTIONS)


def copy_filename(system: str, digest: str) -> str:
    _check_system(system)
    if not _DIGEST.match(digest):
        raise SystemSpecError("spec_sha256 must be 64 lowercase hex characters")
    return f"{system}.{digest[:SPEC_HASH_PREFIX_LEN]}.yaml"


def _check_system(system: str) -> None:
    if not _SYSTEM_ID.match(system):
        raise SystemSpecError(f"invalid system id {system!r}")


def packaged_system_specs(systems_dir: Path = PACKAGED_SYSTEMS_DIR) -> dict[str, Path]:
    """Map system id -> packaged spec YAML, skipping non-spec YAML files."""
    found: dict[str, Path] = {}
    for path in sorted(Path(systems_dir).glob("*.yaml")):
        spec = _load_mapping(path.read_text(encoding="utf-8"), str(path))
        if is_system_spec(spec):
            found[path.stem] = path
    return found


class FieldSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["vendor_spec", "calibrated"]
    evidence: str | None = Field(default=None, description="Eval record or measurement ID for a calibrated field")

    @model_validator(mode="after")
    def _evidence_for_calibrated(self) -> FieldSource:
        if self.source == "calibrated" and not self.evidence:
            raise ValueError("a calibrated field must cite its evidence")
        return self


class HistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_sha256: str = Field(pattern=_DIGEST.pattern)
    note: str | None = None


class SystemManifest(BaseModel):
    """``data/systems/<system>/manifest.yaml``."""

    model_config = ConfigDict(extra="forbid")

    system: str
    current: str = Field(pattern=_DIGEST.pattern, description="spec_sha256 of the current spec")
    history: list[HistoryEntry] = Field(default_factory=list, description="Oldest first; includes current")
    fields: dict[str, FieldSource] = Field(
        default_factory=dict, description="Dotted spec path (e.g. gpu.mem_bw) -> source and evidence"
    )

    @model_validator(mode="after")
    def _consistent(self) -> SystemManifest:
        _check_system(self.system)
        digests = [h.spec_sha256 for h in self.history]
        if len(set(digests)) != len(digests):
            raise ValueError("history repeats a spec_sha256")
        if not digests or digests[-1] != self.current:
            raise ValueError("history must end with the current spec_sha256")
        return self


def _flatten(spec: Mapping[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in spec.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths |= _flatten(value, f"{path}.")
        else:
            paths.add(path)
    return paths


def _dump_yaml(data: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(data), sort_keys=True, default_flow_style=False, allow_unicode=False)


def publish_system_spec(
    packaged_yaml: Path,
    dataset_root: Path,
    *,
    fields: Mapping[str, Mapping[str, Any]] | None = None,
    note: str | None = None,
) -> Path:
    """Write the content-addressed copy and update ``manifest.yaml``.

    ``dataset_root`` is the directory that contains ``systems/``. An existing
    copy is never rewritten (it is immutable); republishing an unchanged spec
    is a no-op. ``fields`` replaces the manifest's per-field sources when given;
    paths must exist in the spec, and fields not listed default to
    ``vendor_spec`` on read, so only calibrated fields need to be listed.
    """
    packaged_yaml = Path(packaged_yaml)
    system = packaged_yaml.stem
    spec = _load_mapping(packaged_yaml.read_text(encoding="utf-8"), str(packaged_yaml))
    if not is_system_spec(spec):
        raise SystemSpecError(f"{packaged_yaml} is not a system spec (missing {_SPEC_SECTIONS} sections)")
    digest = spec_sha256(spec)
    sysdir = Path(dataset_root) / "systems" / system
    sysdir.mkdir(parents=True, exist_ok=True)
    copy = sysdir / copy_filename(system, digest)
    if copy.exists():
        if spec_file_sha256(copy) != digest:
            raise SystemSpecError(f"{copy} exists but does not hash to {digest}; published copies are immutable")
    else:
        copy.write_text(_dump_yaml(spec), encoding="utf-8")

    manifest_path = sysdir / MANIFEST_NAME
    history: list[dict[str, Any]] = []
    prior_fields: dict[str, Any] = {}
    if manifest_path.exists():
        old = read_manifest(manifest_path)
        history = [h.model_dump(exclude_none=True) for h in old.history]
        prior_fields = {k: v.model_dump(exclude_none=True) for k, v in old.fields.items()}
    if not history or history[-1]["spec_sha256"] != digest:
        history = [h for h in history if h["spec_sha256"] != digest]
        entry: dict[str, Any] = {"spec_sha256": digest}
        if note:
            entry["note"] = note
        history.append(entry)
    new_fields = {k: dict(v) for k, v in fields.items()} if fields is not None else prior_fields
    valid = _flatten(spec)
    unknown = sorted(set(new_fields) - valid)
    if unknown:
        raise SystemSpecError(f"manifest fields not present in the spec: {unknown}")
    manifest = SystemManifest.model_validate(
        {"system": system, "current": digest, "history": history, "fields": new_fields}
    )
    manifest_path.write_text(_dump_yaml(manifest.model_dump(exclude_none=True)), encoding="utf-8")
    return copy


def read_manifest(path: Path) -> SystemManifest:
    return SystemManifest.model_validate(_load_mapping(Path(path).read_text(encoding="utf-8"), str(path)))


def check_published_system(system: str, dataset_root: Path, packaged_yaml: Path) -> list[str]:
    """Return drift problems for one system; empty means in sync."""
    problems: list[str] = []
    sysdir = Path(dataset_root) / "systems" / system
    manifest_path = sysdir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"{system}: missing {manifest_path}"]
    try:
        manifest = read_manifest(manifest_path)
    except (ValueError, SystemSpecError) as exc:
        return [f"{system}: invalid manifest: {exc}"]
    if manifest.system != system:
        problems.append(f"{system}: manifest names system {manifest.system!r}")
    packaged = spec_file_sha256(packaged_yaml)
    if manifest.current != packaged:
        problems.append(
            f"{system}: published spec {manifest.current[:SPEC_HASH_PREFIX_LEN]} != packaged "
            f"{packaged[:SPEC_HASH_PREFIX_LEN]}; republish the spec or revert the packaged edit"
        )
    for entry in manifest.history:
        copy = sysdir / copy_filename(system, entry.spec_sha256)
        if not copy.is_file():
            problems.append(f"{system}: missing published copy {copy.name}")
        elif spec_file_sha256(copy) != entry.spec_sha256:
            problems.append(f"{system}: {copy.name} content does not hash to its name")
    extra = {p.name for p in sysdir.glob(f"{system}.*.yaml")} - {
        copy_filename(system, h.spec_sha256) for h in manifest.history
    }
    if extra:
        problems.append(f"{system}: copies not in manifest history: {sorted(extra)}")
    return problems


def check_published_systems(
    dataset_root: Path,
    packaged_dir: Path = PACKAGED_SYSTEMS_DIR,
    *,
    private_systems: frozenset[str] = frozenset(),
) -> list[str]:
    """Check every packaged system against the published copy.

    ``private_systems`` are sensitive systems that live only in the private
    mirror: they must not be packaged. A published directory with no packaged
    spec is reported unless it is listed there.
    """
    problems: list[str] = []
    packaged = packaged_system_specs(packaged_dir)
    for system in sorted(set(packaged) & private_systems):
        problems.append(f"{system}: private system must not be packaged")
    for system, path in packaged.items():
        problems.extend(check_published_system(system, dataset_root, path))
    root = Path(dataset_root) / "systems"
    published = {p.name for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()
    for system in sorted(published - set(packaged) - private_systems):
        problems.append(f"{system}: published but no packaged spec (private systems must be declared)")
    return problems
