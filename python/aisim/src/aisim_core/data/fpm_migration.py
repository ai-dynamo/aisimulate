# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time move of the public FPM dataset into the unified ``data/fpm/`` layout (AISIM-2337).

The move follows DEP #435 "FPM alignment":

* Every file under ``data/`` keeps its exact bytes (same SHA-256). The only
  change is the path: ``data/<rest>`` becomes ``data/fpm/<rest>``. There is no
  schema change, no rewrite of any FPM file and no legacy adapter. Strings
  inside moved files that name the old ``data/...`` path (leaf manifests,
  metadata, import records) are historical values and stay as they are.
* Each configuration leaf (a directory holding the legacy ``manifest.json``)
  gets a new shared-envelope manifest (``aisim-artifact-manifest/v1``,
  category ``fpm``) next to it that lists every file in the leaf with its
  SHA-256, size and, for Parquet, row count.
* The current configurations are listed in a category index
  (``catalog/fpm/index.json``), the dataset index (``catalog/dataset.json``)
  and the coverage catalog shards (``catalog/fpm/<hardware>.parquet``).
  Historical snapshots get envelopes but are not indexed by default, so a
  ``--model/--hardware`` selection never downloads superseded snapshots.
* ``.gitattributes`` path-anchored LFS rules under ``data/`` follow the move.
  Everything else in the repository (README, schemas, scripts, audit, legacy
  ``catalog/*``) is copied unchanged.

Nothing here talks to the network except :func:`download_snapshot`, which only
reads. Publishing is never done by this module; :func:`plan_upload` only
computes the commit operations for a maintainer to approve.

CLI::

    python -m aisim_core.data.fpm_migration migrate --source S --output O [--listing L]
    python -m aisim_core.data.fpm_migration audit --source S --output O --report-dir R [--listing L]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from aisim_core.data import (
    Artifact,
    CategoryIndex,
    CategoryReference,
    DatasetIndex,
    Entry,
    Identity,
    Manifest,
    check_dataset,
    configuration_id,
)
from aisim_core.data.parquet import write_parquet

DEFAULT_REPO = "nvidia/aisimulate-fpm-dataset"
OLD_PREFIX = "data/"
NEW_PREFIX = "data/fpm/"
HISTORY_DIR = "history"
ENVELOPE_NAME = "aisim-manifest.json"
LEGACY_LEAF_MANIFEST = "manifest.json"
DATASET_INDEX = "catalog/dataset.json"
CATEGORY_INDEX = "catalog/fpm/index.json"
# Source-tree entries that are download metadata, never repository content.
_IGNORED_TOP = {".cache", ".git"}
_PATH_MENTION = re.compile(rb"data/(?:history/)?[A-Za-z0-9][A-Za-z0-9._-]*--[A-Za-z0-9._-]+/")

AUDIT_SCHEMA = "aisim-fpm-move-audit/v1"


class MigrationError(Exception):
    """Raised when the source tree cannot be migrated without guessing."""


# --------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_blob_id(path: Path) -> str:
    """The git object ID Hugging Face reports for a non-LFS file."""
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def target_path(source: str) -> str:
    """Pure path rule: ``data/<rest>`` -> ``data/fpm/<rest>``."""
    if not source.startswith(OLD_PREFIX) or source.startswith(NEW_PREFIX):
        raise MigrationError(f"not a legacy data path: {source}")
    return NEW_PREFIX + source[len(OLD_PREFIX) :]


def list_files(root: Path) -> list[str]:
    """Repository-relative POSIX paths of every file under ``root``, sorted. Symlinks are rejected."""
    out = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in _IGNORED_TOP:
            continue
        if path.is_symlink():
            raise MigrationError(f"symlink in dataset: {relative}")
        if path.is_file():
            out.append(relative.as_posix())
    return out


def _canonical(document: Any) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


# --------------------------------------------------------------------------- source listing


def load_listing(path: Path) -> dict[str, dict[str, Any]]:
    """Read a listing saved by :func:`download_snapshot`: path -> {size, lfs_sha256, blob_id}."""
    document = json.loads(Path(path).read_text())
    return {f["path"]: f for f in document["files"]}


def verify_source(root: Path, listing: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Check local files against the Hugging Face listing (LFS SHA-256 or git blob ID). Returns problems."""
    problems = []
    local = set(list_files(root))
    for missing in sorted(set(listing) - local):
        problems.append(f"missing locally: {missing}")
    for extra in sorted(local - set(listing)):
        problems.append(f"not in listing: {extra}")
    for relative in sorted(local & set(listing)):
        entry, path = listing[relative], root / relative
        if path.stat().st_size != entry["size"]:
            problems.append(f"size differs: {relative}")
        elif entry.get("lfs"):
            if sha256_file(path) != entry["lfs"]:
                problems.append(f"LFS SHA-256 differs: {relative}")
        elif git_blob_id(path) != entry["blob"]:
            problems.append(f"git blob ID differs: {relative}")
    return problems


def download_snapshot(repo: str, revision: str, dest: Path, listing_out: Path, token: str | None = None) -> str:
    """Read-only: list and download ``repo`` at ``revision`` into an empty ``dest``. Returns the commit SHA.

    ``revision`` is resolved once to a full SHA. Requires ``huggingface_hub`` (imported lazily).
    """
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    sha = api.dataset_info(repo, revision=revision).sha
    files = []
    for item in api.list_repo_tree(repo, repo_type="dataset", recursive=True, revision=sha):
        if hasattr(item, "size"):
            files.append(
                {
                    "path": item.path,
                    "size": item.size,
                    "lfs": item.lfs.sha256 if item.lfs else None,
                    "blob": item.blob_id,
                }
            )
    Path(listing_out).write_bytes(_canonical({"repo": repo, "revision": sha, "files": files}))
    snapshot_download(repo, repo_type="dataset", revision=sha, local_dir=str(dest), token=token)
    return sha


# --------------------------------------------------------------------------- envelope construction


@dataclass(frozen=True)
class Leaf:
    directory: str  # legacy path, e.g. data/<model>/<hw>/<fw>/<ver>/<par>
    manifest: dict[str, Any]
    files: tuple[str, ...]  # legacy paths of every file in the leaf

    @property
    def historical(self) -> bool:
        return self.directory.startswith(OLD_PREFIX + HISTORY_DIR + "/")

    @property
    def target_directory(self) -> str:
        return target_path(self.directory)


def discover_leaves(root: Path, files: Iterable[str]) -> list[Leaf]:
    """Group ``data/**`` files into configuration leaves. Every file must belong to exactly one leaf."""
    data_files = [f for f in files if f.startswith(OLD_PREFIX)]
    leaves: dict[str, dict[str, Any]] = {}
    for relative in data_files:
        path = PurePosixPath(relative)
        if path.name == LEGACY_LEAF_MANIFEST and "measurements" not in path.parts:
            manifest = json.loads((root / relative).read_text())
            leaves[path.parent.as_posix()] = manifest
    members: dict[str, list[str]] = defaultdict(list)
    orphans = []
    for relative in data_files:
        owners = [d for d in leaves if relative.startswith(d + "/")]
        if len(owners) != 1:
            orphans.append(relative)
        else:
            members[owners[0]].append(relative)
    if orphans:
        raise MigrationError(f"{len(orphans)} data files are not in exactly one leaf, e.g. {orphans[:3]}")
    return [Leaf(d, leaves[d], tuple(sorted(members[d]))) for d in sorted(leaves)]


def _fpm_schemas(root: Path) -> dict[str, str]:
    """Legacy path -> ``<schema_name>/v<version>`` for each FPM library, from the legacy catalog."""
    path = root / "catalog" / "fpm.json"
    if not path.is_file():
        return {}
    records = json.loads(path.read_text())["records"]
    return {r["path"]: f"{r['schema_name']}/v{r['schema_version']}" for r in records}


_FORMATS = {
    ".parquet": "parquet",
    ".json": "json",
    ".yaml": "yaml",
    ".md": "markdown",
    ".tsv": "tsv",
    ".gz": "gzip",
    ".jsonl": "jsonl",
    ".txt": "text",
    ".stdout": "text",
    ".stderr": "text",
}


def classify(leaf_relative: str, manifest_version: int | None = None) -> tuple[str, str]:
    """(role, format) of a file by its leaf-relative path."""
    path = PurePosixPath(leaf_relative)
    fmt = _FORMATS.get(path.suffix, "text" if path.name == "LICENSE" else "binary")
    parts = path.parts
    if leaf_relative == LEGACY_LEAF_MANIFEST:
        role = "legacy-configuration-manifest"
    elif parts[0] == "fpm":
        role = (
            "fpm-library"
            if path.suffix == ".parquet" and len(parts) == 2
            else ("fpm-library-metadata" if len(parts) == 2 else "fpm-provenance")
        )
    elif parts[0] == "measurements":
        if leaf_relative == "measurements/manifest.json":
            role = "measurement-manifest"
        elif len(parts) > 2 and parts[1] == "provenance":
            role = "measurement-provenance"
        else:
            role = "measurement-data"
    else:
        role = "other"
    return role, fmt


def _schema_label(role: str, legacy_path: str, schemas: Mapping[str, str], file: Path) -> str:
    if role == "fpm-library":
        try:
            return schemas[legacy_path]
        except KeyError:
            raise MigrationError(f"no legacy catalog record for FPM library {legacy_path}") from None
    if role in ("legacy-configuration-manifest", "measurement-manifest"):
        version = json.loads(file.read_text()).get("manifest_version")
        name = "configuration-manifest" if role == "legacy-configuration-manifest" else "measurement-manifest"
        return f"{name}/v{version}"
    return "unversioned"


def _parallelism(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    dims = {
        "tp": manifest["tp"],
        "pp": manifest["pp"],
        "dp": manifest["dp"],
        "cp": manifest["cp"],
        "ep": manifest["moe_ep"],
        "moe_tp": manifest["moe_tp"],
        "dcp": manifest.get("dcp", 1),
    }
    return {"aggregated": dims}


def build_identity(leaf: Leaf) -> Identity:
    m = leaf.manifest
    settings: dict[str, Any] = {
        "parallel_strategy": m["parallel_strategy"],
        "parallelism_label": m["parallelism"],
        "snapshot_status": m["snapshot_status"],
        "gpu_family": m["gpu_family"],
    }
    if m["snapshot_status"] != "current":
        settings["snapshot_id"] = m["snapshot_id"]
    for key in ("model_revision", "collection_method", "physical_gpu_count"):
        if m.get(key) is not None:
            settings[key] = m[key]
    return Identity(
        hardware=m["system"],
        framework=m["framework"],
        framework_version=m["framework_version"],
        operation_family=None,
        model=m["model_id"],
        effective_settings=settings,
        unresolved_identity={"model_revision": None} if m.get("model_revision") is None else None,
        parallelism=_parallelism(m),
        parallelism_complete=True,
        precision={"weights": m["weight_quantization"], "kv_cache": m["kv_cache_dtype"]},
        deployment=None,
    )


def _gaps(leaf: Leaf) -> list[dict[str, Any]]:
    m = leaf.manifest
    gaps: list[dict[str, Any]] = [
        {
            "field": "deployment",
            "reason": "the legacy configuration manifest does not record aggregated vs disaggregated",
        }
    ]
    if m.get("model_revision") is None:
        gaps.append({"field": "model_revision", "reason": "legacy configuration manifest has a null model_revision"})
    if m.get("aisim_commit_status") != "recorded":
        gaps.append(
            {"field": "aisim_commit", "reason": f"legacy aisim_commit_status is {m.get('aisim_commit_status')!r}"}
        )
    if (m.get("provenance") or {}).get("producer_identity_missing"):
        gaps.append({"field": "producer_identity", "reason": "legacy provenance reports producer_identity_missing"})
    return gaps


def build_manifest(
    root: Path, leaf: Leaf, schemas: Mapping[str, str], source_repo: str, source_revision: str
) -> Manifest:
    """Shared-envelope manifest for one leaf. Artifact paths are the NEW paths; hashes are of the unchanged bytes."""
    artifacts = []
    for legacy in leaf.files:
        relative = legacy[len(leaf.directory) + 1 :]
        role, fmt = classify(relative)
        file = root / legacy
        rows = pq.ParquetFile(file).metadata.num_rows if fmt == "parquet" else None
        artifacts.append(
            Artifact(
                artifact_id=relative,
                path=target_path(legacy),
                role=role,
                format=fmt,
                schema=_schema_label(role, legacy, schemas, file),
                sha256=sha256_file(file),
                bytes=file.stat().st_size,
                rows=rows,
            )
        )
    identity = build_identity(leaf)
    m = leaf.manifest
    return Manifest(
        schema_version="aisim-artifact-manifest/v1",
        category="fpm",
        purpose="execution-profile",
        configuration_id=configuration_id("fpm", "execution-profile", identity),
        identity=identity,
        requested_settings=None,
        artifacts=artifacts,
        provenance={
            "source_repository": f"https://huggingface.co/datasets/{source_repo}",
            "source_revision": source_revision,
            "path_mapping": {"from": OLD_PREFIX, "to": NEW_PREFIX},
            "legacy_configuration_path": leaf.directory,
            "legacy_snapshot_id": m["snapshot_id"],
            "legacy_manifest_version": m["manifest_version"],
            "evidence_date": m.get("evidence_date"),
            "aisim_commit_status": m.get("aisim_commit_status"),
            "legacy_provenance": m.get("provenance"),
        },
        evidence_gaps=_gaps(leaf),
    )


def _artifact_of(path: str, data: bytes, role: str, schema: str) -> Artifact:
    return Artifact(
        artifact_id=role,
        path=path,
        role=role,
        format="json",
        schema=schema,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )


def rewrite_gitattributes(text: str) -> str:
    """Move path-anchored ``data/`` LFS rules to ``data/fpm/``; other rules are unchanged."""
    out = []
    for line in text.splitlines(keepends=True):
        out.append(NEW_PREFIX + line[len(OLD_PREFIX) :] if line.startswith(OLD_PREFIX) else line)
    return "".join(out)


# --------------------------------------------------------------------------- migrate


@dataclass
class MigrationResult:
    revision: str
    moved: int
    leaves: int
    indexed: int
    history: int
    envelopes: list[str]
    new_files: list[str]
    modified_files: list[str]


def _place(source: Path, destination: Path, link: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copyfile(source, destination)


def migrate(
    source: Path,
    output: Path,
    *,
    revision: str,
    repo: str = DEFAULT_REPO,
    link: bool = False,
    index_history: bool = False,
    systems_dir: Path | None = None,
) -> MigrationResult:
    """Build the post-move repository tree under ``output`` (must not exist) from ``source``."""
    from aisim_core.data import catalog as cat

    source, output = Path(source), Path(output)
    if output.exists():
        raise MigrationError(f"output already exists: {output}")
    files = list_files(source)
    leaves = discover_leaves(source, files)
    if not leaves:
        raise MigrationError("no configuration leaves found")
    schemas = _fpm_schemas(source)
    output.mkdir(parents=True)
    try:
        # 1. Non-data files unchanged (except .gitattributes); data files moved by pure prefix rule.
        modified = []
        for relative in files:
            if relative.startswith(OLD_PREFIX):
                _place(source / relative, output / target_path(relative), link)
            elif relative == ".gitattributes":
                text = (source / relative).read_text()
                (output / relative).write_text(rewrite_gitattributes(text))
                modified.append(relative)
            else:
                _place(source / relative, output / relative, link)

        # 2. Envelope manifests, one per leaf.
        built: list[tuple[Leaf, Manifest, str, bytes]] = []
        seen_ids: dict[str, str] = {}
        for leaf in leaves:
            manifest = build_manifest(source, leaf, schemas, repo, revision)
            if manifest.configuration_id in seen_ids:
                raise MigrationError(
                    f"identity collision between {leaf.directory} and {seen_ids[manifest.configuration_id]}"
                )
            seen_ids[manifest.configuration_id] = leaf.directory
            path = f"{leaf.target_directory}/{ENVELOPE_NAME}"
            data = _canonical(manifest.model_dump(by_alias=True, mode="json"))
            (output / path).write_bytes(data)
            built.append((leaf, manifest, path, data))

        # 3. Category index (current leaves; history only on request), dataset index.
        entries = [
            Entry(
                configuration_id=manifest.configuration_id,
                purpose="execution-profile",
                identity=manifest.identity,
                manifest=_artifact_of(path, data, "manifest", "aisim-artifact-manifest/v1"),
            )
            for leaf, manifest, path, data in built
            if index_history or not leaf.historical
        ]
        entries.sort(key=lambda e: e.configuration_id)
        index_bytes = _canonical(
            CategoryIndex(schema_version="aisim-category-index/v1", category="fpm", entries=entries).model_dump(
                by_alias=True, mode="json"
            )
        )
        (output / CATEGORY_INDEX).parent.mkdir(parents=True, exist_ok=True)
        (output / CATEGORY_INDEX).write_bytes(index_bytes)
        dataset = DatasetIndex(
            schema_version="aisim-dataset-index/v1",
            dataset_id=repo,
            categories={
                "fpm": CategoryReference(
                    root="data/fpm",
                    layout="category",
                    index=_artifact_of(CATEGORY_INDEX, index_bytes, "index", "aisim-category-index/v1"),
                )
            },
        )
        (output / DATASET_INDEX).write_bytes(_canonical(dataset.model_dump(by_alias=True, mode="json")))

        # 4. Coverage catalog (fpm layer only) via the library builder.
        kwargs = {"systems_dir": systems_dir} if systems_dir else {}
        rows = [r for r in cat.build_catalog(output, write=False, **kwargs) if r.layer == "fpm"]
        cat._write(output, rows, cat.known_system_specs(**kwargs))

        new_files = sorted(
            set(list_files(output)) - set(files) - {target_path(f) for f in files if f.startswith(OLD_PREFIX)}
        )
        check_dataset(output)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return MigrationResult(
        revision=revision,
        moved=sum(f.startswith(OLD_PREFIX) for f in files),
        leaves=len(leaves),
        indexed=len(entries),
        history=sum(leaf.historical for leaf in leaves),
        envelopes=[path for _, _, path, _ in built],
        new_files=new_files,
        modified_files=modified,
    )


# --------------------------------------------------------------------------- audit


def audit(
    source: Path,
    output: Path,
    *,
    listing: Mapping[str, Mapping[str, Any]] | None = None,
    revision: str = "",
) -> dict[str, Any]:
    """Prove the move. Returns a JSON-serializable report; ``report['passed']`` is the exit gate.

    Checks: (a) the source matches the Hugging Face listing; (b) every ``data/**`` source file exists
    at ``data/fpm/**`` with equal size and SHA-256; (c) every other source file is unchanged except
    declared modifications; (d) the library accepts the dataset (``check_dataset``); (e) every moved
    file is declared by exactly one envelope manifest with the same hash; (f) the catalog lists every
    current legacy configuration, and every historical one has an envelope.
    """
    source, output = Path(source), Path(output)
    failures: list[str] = []
    source_files, output_files = list_files(source), set(list_files(output))
    if listing is not None:
        failures += [f"source/listing: {p}" for p in verify_source(source, listing)]

    moved, file_rows = 0, []
    for relative in source_files:
        if relative.startswith(OLD_PREFIX):
            destination = target_path(relative)
            moved += 1
        else:
            destination = relative
        if destination not in output_files:
            failures.append(f"missing in output: {destination}")
            continue
        before, after = sha256_file(source / relative), sha256_file(output / destination)
        size = (source / relative).stat().st_size
        unchanged = before == after and size == (output / destination).stat().st_size
        modified = (not unchanged) and relative == ".gitattributes"
        if not unchanged and not modified:
            failures.append(f"BYTES DIFFER: {relative} -> {destination}")
        file_rows.append(
            {
                "source_path": relative,
                "target_path": destination,
                "sha256_before": before,
                "sha256_after": after,
                "bytes": size,
                "moved": relative.startswith(OLD_PREFIX),
                "unchanged": unchanged,
            }
        )

    # (d) library check
    try:
        index = check_dataset(output)
    except Exception as exc:  # report, do not mask
        index, failures = None, failures + [f"check_dataset: {exc}"]

    # (e) envelope coverage
    declared: dict[str, tuple[str, str]] = {}
    envelope_ids: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in output_files if p.endswith("/" + ENVELOPE_NAME)):
        manifest = Manifest.model_validate_json((output / path).read_bytes())
        envelope_ids[path] = {"id": manifest.configuration_id, "artifacts": len(manifest.artifacts)}
        for artifact in manifest.artifacts:
            if artifact.path in declared:
                failures.append(f"declared by two envelopes: {artifact.path}")
            declared[artifact.path] = (artifact.sha256, path)
    for row in file_rows:
        if row["moved"]:
            got = declared.get(row["target_path"])
            if got is None:
                failures.append(f"moved file not declared by any envelope: {row['target_path']}")
            elif got[0] != row["sha256_before"]:
                failures.append(f"envelope hash differs from source bytes: {row['target_path']}")
    stray = sorted(set(declared) - {r["target_path"] for r in file_rows if r["moved"]})
    failures += [f"envelope declares a path that was not moved: {p}" for p in stray]

    # (f) catalog vs legacy catalog
    indexed_paths, listed = set(), []
    if index is not None:
        category = CategoryIndex.model_validate_json((output / index.categories["fpm"].index.path).read_bytes())
        listed = category.entries
        indexed_paths = {e.manifest.path for e in listed}
    legacy_current: list[str] = []
    legacy_history: list[str] = []
    legacy_index = source / "catalog" / "index.json"
    if legacy_index.is_file():
        document = json.loads(legacy_index.read_text())
        legacy_current = document.get("configuration_manifests", [])
        legacy_history = document.get("history_manifests", [])
    sibling = lambda p: target_path(p)[: -len(LEGACY_LEAF_MANIFEST)] + ENVELOPE_NAME
    expected_current = {sibling(p) for p in legacy_current}
    expected_history = {sibling(p) for p in legacy_history}
    for label, expected in (("current", expected_current), ("history", expected_history)):
        for p in sorted(expected - set(envelope_ids)):
            failures.append(f"legacy {label} configuration has no envelope: {p}")
    if legacy_current and not expected_current <= indexed_paths:
        failures += [
            f"legacy current configuration not in catalog: {p}" for p in sorted(expected_current - indexed_paths)
        ]
    if legacy_history and not expected_history <= set(envelope_ids):
        failures.append("some legacy history configurations lack envelopes")
    unknown = sorted(set(envelope_ids) - expected_current - expected_history) if legacy_index.is_file() else []
    failures += [f"envelope for a configuration absent from the legacy catalog: {p}" for p in unknown]

    # informational: moved text files that still mention old data/ paths
    mentions = 0
    for row in file_rows:
        scannable = row["moved"] and not row["target_path"].endswith((".parquet", ".gz")) and row["bytes"] < 5_000_000
        if scannable and _PATH_MENTION.search((output / row["target_path"]).read_bytes()):
            mentions += 1

    # coverage-catalog shards must account for every indexed configuration
    from aisim_core.data.catalog import Catalog, check_catalog_size

    shard_configs = 0
    try:
        shard_configs = sum(r.n_configs for r in Catalog(output).rows(layer="fpm"))
        size_report = check_catalog_size(output)
    except Exception as exc:
        failures.append(f"coverage catalog: {exc}")
        size_report = None
    if listed and shard_configs != len(listed):
        failures.append(f"coverage catalog counts {shard_configs} configurations, index lists {len(listed)}")

    ids = [e.configuration_id for e in listed]
    by_hw = Counter(e.identity.hardware for e in listed)
    by_model = Counter(e.identity.model for e in listed)
    return {
        "schema": AUDIT_SCHEMA,
        "source_revision": revision,
        "passed": not failures,
        "failures": failures,
        "files_in_source": len(source_files),
        "data_files_moved": moved,
        "data_files_hash_unchanged": sum(r["unchanged"] for r in file_rows if r["moved"]),
        "other_files_unchanged": sum(r["unchanged"] for r in file_rows if not r["moved"]),
        "other_files_modified": [r["source_path"] for r in file_rows if not r["unchanged"]],
        "new_files": sorted(output_files - {r["target_path"] for r in file_rows}),
        "bytes_moved": sum(r["bytes"] for r in file_rows if r["moved"]),
        "envelopes": len(envelope_ids),
        "catalog_configurations": len(listed),
        "catalog_unique_ids": len(set(ids)),
        "coverage_catalog_configurations": shard_configs,
        "coverage_catalog_bytes": size_report.total_bytes if size_report else None,
        "coverage_catalog_size_findings": [list(f) for f in size_report.findings] if size_report else [],
        "legacy_current_configurations": len(legacy_current),
        "legacy_history_configurations": len(legacy_history),
        "catalog_by_hardware": dict(sorted(by_hw.items())),
        "catalog_by_model": dict(sorted(by_model.items())),
        "moved_files_mentioning_legacy_data_paths": mentions,
        "file_rows": file_rows,
    }


def write_audit_report(report: Mapping[str, Any], directory: Path) -> list[Path]:
    """Write ``hash-audit.parquet`` (every file) and ``audit-summary.json`` (everything else)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = report["file_rows"]
    table = pa.table(
        {
            "source_path": [r["source_path"] for r in rows],
            "target_path": [r["target_path"] for r in rows],
            "sha256_before": [r["sha256_before"] for r in rows],
            "sha256_after": [r["sha256_after"] for r in rows],
            "bytes": pa.array([r["bytes"] for r in rows], pa.int64()),
            "moved": [r["moved"] for r in rows],
            "unchanged": [r["unchanged"] for r in rows],
        }
    )
    parquet = write_parquet(table, directory / "hash-audit.parquet", AUDIT_SCHEMA)
    summary = directory / "audit-summary.json"
    summary.write_bytes(_canonical({k: v for k, v in report.items() if k != "file_rows"}))
    return [parquet, summary]


# --------------------------------------------------------------------------- upload plan


def plan_upload(listing: Mapping[str, Mapping[str, Any]], output: Path, source: Path) -> dict[str, Any]:
    """Commit operations that turn the source revision into ``output``. Computes only; publishes nothing.

    * ``copy``   LFS data files: server-side copy, no upload (``CommitOperationCopy`` works for LFS only).
    * ``add``    non-LFS moved files, new envelopes/catalog files and modified root files (uploaded).
    * ``delete`` every legacy ``data/<rest>`` path; old commits still resolve them.
    """
    output, source = Path(output), Path(source)
    moved = [p for p in listing if p.startswith(OLD_PREFIX)]
    ops: list[dict[str, Any]] = []
    for legacy in sorted(moved):
        if listing[legacy].get("lfs"):
            ops.append({"op": "copy", "src": legacy, "dst": target_path(legacy), "bytes": listing[legacy]["size"]})
        else:
            ops.append(
                {"op": "add", "src": target_path(legacy), "dst": target_path(legacy), "bytes": listing[legacy]["size"]}
            )
    out_files = set(list_files(output))
    moved_targets = {target_path(p) for p in moved}
    for relative in sorted(out_files - moved_targets):
        in_listing = relative in listing
        if in_listing and sha256_file(output / relative) == sha256_file(source / relative):
            continue
        ops.append({"op": "add", "src": relative, "dst": relative, "bytes": (output / relative).stat().st_size})
    ops += [{"op": "delete", "src": p, "dst": p, "bytes": listing[p]["size"]} for p in sorted(moved)]
    by_kind = Counter(o["op"] for o in ops)
    return {
        "operations": ops,
        "counts": dict(by_kind),
        "uploaded_bytes": sum(o["bytes"] for o in ops if o["op"] == "add"),
        "copied_lfs_bytes": sum(o["bytes"] for o in ops if o["op"] == "copy"),
    }


def write_upload_plan(plan: Mapping[str, Any], directory: Path) -> list[Path]:
    """Write ``upload-operations.parquet`` (one row per commit operation) and ``upload-summary.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ops = plan["operations"]
    table = pa.table(
        {
            "op": [o["op"] for o in ops],
            "src": [o["src"] for o in ops],
            "dst": [o["dst"] for o in ops],
            "bytes": pa.array([o["bytes"] for o in ops], pa.int64()),
        }
    )
    parquet = write_parquet(table, directory / "upload-operations.parquet", "aisim-fpm-move-upload-plan/v1")
    summary = directory / "upload-summary.json"
    summary.write_bytes(_canonical({k: v for k, v in plan.items() if k != "operations"}))
    return [parquet, summary]


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", type=Path, required=True, help="snapshot of the legacy repository")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--listing", type=Path, help="listing JSON written by `download`")

    down = sub.add_parser("download", help="read-only download of a pinned revision")
    down.add_argument("--repo", default=DEFAULT_REPO)
    down.add_argument("--revision", required=True)
    down.add_argument("--dest", type=Path, required=True)
    down.add_argument("--listing-out", type=Path, required=True)

    mig = sub.add_parser("migrate", help="build the data/fpm/ tree locally")
    common(mig)
    mig.add_argument("--revision", required=True, help="full source commit SHA")
    mig.add_argument("--repo", default=DEFAULT_REPO)
    mig.add_argument("--link", action="store_true", help="hardlink instead of copy when possible")
    mig.add_argument("--index-history", action="store_true", help="also list historical snapshots in the index")

    aud = sub.add_parser("audit", help="hash audit and catalog check")
    common(aud)
    aud.add_argument("--revision", default="")
    aud.add_argument("--report-dir", type=Path, required=True)

    plan = sub.add_parser("plan-upload", help="compute (never run) the commit operations")
    common(plan)
    plan.add_argument("--plan-out", type=Path, required=True, help="directory for the operations table and summary")

    args = parser.parse_args(argv)
    if args.command == "download":
        token = os.environ.get("HF_TOKEN")
        print(download_snapshot(args.repo, args.revision, args.dest, args.listing_out, token))
        return 0
    if args.command == "migrate":
        listing = load_listing(args.listing) if args.listing else None
        if listing is not None:
            problems = verify_source(args.source, listing)
            if problems:
                print("\n".join(problems[:20]), file=sys.stderr)
                return 1
        result = migrate(
            args.source,
            args.output,
            revision=args.revision,
            repo=args.repo,
            link=args.link,
            index_history=args.index_history,
        )
        print(json.dumps({k: v for k, v in result.__dict__.items() if k != "envelopes"}, indent=2))
        return 0
    listing = load_listing(args.listing) if args.listing else None
    if args.command == "audit":
        report = audit(args.source, args.output, listing=listing, revision=args.revision)
        for path in write_audit_report(report, args.report_dir):
            print(path)
        for failure in report["failures"][:50]:
            print("FAIL", failure, file=sys.stderr)
        return 0 if report["passed"] else 1
    if listing is None:
        parser.error("plan-upload requires --listing")
    plan = plan_upload(listing, args.output, args.source)
    write_upload_plan(plan, args.plan_out)
    print(json.dumps({k: v for k, v in plan.items() if k != "operations"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
