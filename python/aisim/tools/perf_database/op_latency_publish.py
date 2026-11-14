# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert collector/packaged operation tables into the ``data/op-latency/`` dataset layout.

This is a supported publish step, not library code: it stays until AISIM-2373 makes the collector emit the
layout directly. The permanent contract (row models, schema IDs, layout paths, content hashes) lives in
:mod:`aisim_core.data.op_schema`; this module only builds and audits a local publish tree.

Build the local ``data/op-latency/`` publish tree from the packaged operation measurements.

Source: ``aisim_core/systems/data/<system>/<family>/<backend>/<version>/`` plus the lookup and reuse
metadata ``systems/perf_data_reuse_manifest.yaml`` and ``systems/query_versions.yaml``. Nothing is read from
or written to the network, and nothing under the source root is modified or removed (the packaged path stays
in the wheel).

Output (all paths relative to ``out``)::

    data/op-latency/<system>/<family>/<backend>/<version>/<table>.parquet   restamped zstd Parquet
    data/op-latency/<system>/<family>/<backend>/<version>/collection_meta.yaml, reuse.yaml, errors*.json
                                                                            byte-identical copies
    data/op-latency/<system>/<family>/<backend>/<version>/manifest.json     aisim-artifact-manifest/v1
    data/op-latency/<system>/<extra files>                                  byte-identical copies
    data/op-latency/perf_data_reuse_manifest.parquet                        converted from YAML
    data/op-latency/query_versions.parquet                                  converted from YAML
    data/op-latency/source_map.parquet                                      source -> destination + hashes
    catalog/dataset.json, catalog/op-latency.json                           dataset and category indexes
    catalog/op-latency-reuse.parquet                                        declared reuse with donors resolved
    catalog/op-latency/<system>.parquet, catalog/systems.parquet            coverage catalog (build_catalog)

Hash policy (:func:`audit_tree` re-verifies every row from disk):

* ``file-bytes``      byte-identical copy; source and destination SHA-256 are equal.
* ``arrow-rows``      Parquet restamped to zstd with the schema ID; the SHA-256 of the Arrow IPC serialization
                      of the schema-metadata-free table is equal on both sides.
* ``embedded-source`` YAML converted to Parquet; the verbatim source text is embedded in the file metadata and
                      its SHA-256 equals the source file's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

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
from aisim_core.data.catalog import build_catalog
from aisim_core.data.op_schema import (
    CATEGORY_INDEX_PATH,
    DATA_PREFIX,
    DATASET_INDEX_PATH,
    DECLARED_REUSE_PATH,
    OP_DECLARED_REUSE_SCHEMA_ID,
    OP_QUERY_VERSIONS_SCHEMA_ID,
    OP_REUSE_MANIFEST_SCHEMA_ID,
    OP_SOURCE_MAP_SCHEMA_ID,
    OP_TABLE_SCHEMA_ID,
    QUERY_VERSIONS_NAME,
    REUSE_MANIFEST_NAME,
    SKIPPED_NAMES,
    SOURCE_MAP_PATH,
    file_sha256,
    table_content_sha256,
)
from aisim_core.data.parquet import read_parquet, read_parquet_schema_id, write_parquet

# Recorded in every manifest; kept at its original value so rebuilt trees stay byte-comparable.
PUBLISHER = "aisimulate_core.data.op_publish/v1"
REUSE_GROUP_KEYS = {
    "op_file",
    "kernel_source",
    "tier",
    "systems",
    "frameworks",
    "total_raw_rows",
    "within_framework_dedup_rows",
    "median_pct_divergence",
}


class PublishError(ValueError):
    """Raised when the source tree cannot be published faithfully."""


# --------------------------------------------------------------------------- helpers


def _json_sha256(document: Any) -> str:
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


# --------------------------------------------------------------------------- source scan


@dataclass(frozen=True)
class SourceInfo:
    root: Path
    toplevel: Path | None
    commit: str | None
    blobs: dict[str, str]  # path relative to root -> git blob id

    def repo_path(self, relative: str) -> str:
        return self.absolute_repo_path(self.root / relative)

    def blob_of(self, path: Path) -> str | None:
        """Git blob id of any tracked file in the same repository (None outside git)."""
        if self.toplevel is None:
            return None
        out = _git(self.toplevel, "ls-files", "-s", "--", self.absolute_repo_path(path)) or ""
        fields = out.split()
        return fields[1] if len(fields) > 1 else None

    def absolute_repo_path(self, path: Path) -> str:
        path = Path(path).resolve()
        return path.relative_to(self.toplevel).as_posix() if self.toplevel else f"{self.root.name}/{path.name}"


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout


def inspect_source(root: Path, *, allow_dirty: bool = False) -> SourceInfo:
    """Resolve git provenance for ``root``; require a clean tree so hashes belong to the commit."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None:
        return SourceInfo(root, None, None, {})
    commit = (_git(root, "rev-parse", "HEAD") or "").strip() or None
    dirty = (_git(root, "status", "--porcelain", "--", ".") or "").strip()
    if dirty and not allow_dirty:
        raise PublishError(f"source tree {root} has uncommitted changes; commit them or pass allow_dirty=True")
    blobs = {}
    for line in (_git(root, "ls-files", "-s", "-z", "--", ".") or "").split("\0"):
        if line:
            meta, _, name = line.partition("\t")
            blobs[name] = meta.split()[1]
    return SourceInfo(root, Path(top.strip()), commit, blobs)


@dataclass(frozen=True)
class SourceFile:
    relative: str  # path relative to the systems data root
    kind: str  # table | collection-meta | reuse-declaration | auxiliary | extra | skipped
    hardware: str
    family: str | None = None
    framework: str | None = None
    version: str | None = None

    @property
    def name(self) -> str:
        return self.relative.rsplit("/", 1)[-1]

    @property
    def version_dir(self) -> tuple[str, str, str, str] | None:
        if self.family is None:
            return None
        return (self.hardware, self.family, self.framework, self.version)  # type: ignore[return-value]


def scan_source(root: Path) -> list[SourceFile]:
    """List every file under the data root, classified by its position in the layout."""
    root = Path(root)
    found = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
        if path.is_symlink():
            raise PublishError(f"symlink in source tree: {path}")
        relative = path.relative_to(root).as_posix()
        parts = relative.split("/")
        if parts[-1] in SKIPPED_NAMES:
            found.append(SourceFile(relative, "skipped", parts[0]))
        elif len(parts) == 5:
            name = parts[-1]
            if name == "collection_meta.yaml":
                kind = "collection-meta"
            elif name == "reuse.yaml":
                kind = "reuse-declaration"
            elif name.endswith(".parquet"):
                kind = "table"
            else:
                kind = "auxiliary"
            found.append(SourceFile(relative, kind, *parts[:4]))
        else:
            found.append(SourceFile(relative, "extra", parts[0]))
    return found


# --------------------------------------------------------------------------- per-file publication


@dataclass
class MapRow:
    source_path: str | None
    destination_path: str
    kind: str
    conversion: str  # copy | parquet-restamp | yaml-to-parquet | skipped
    hash_basis: str  # file-bytes | arrow-rows | embedded-source
    source_sha256: str | None
    source_bytes: int | None
    source_git_blob: str | None
    destination_sha256: str | None
    destination_bytes: int | None
    source_content_sha256: str | None
    destination_content_sha256: str | None
    rows: int | None
    hardware: str | None
    family: str | None
    framework: str | None
    version: str | None
    configuration_id: str | None = None


_MAP_COLUMNS = {
    "source_path": pa.string(),
    "destination_path": pa.string(),
    "kind": pa.string(),
    "conversion": pa.string(),
    "hash_basis": pa.string(),
    "source_sha256": pa.string(),
    "source_bytes": pa.int64(),
    "source_git_blob": pa.string(),
    "destination_sha256": pa.string(),
    "destination_bytes": pa.int64(),
    "source_content_sha256": pa.string(),
    "destination_content_sha256": pa.string(),
    "rows": pa.int64(),
    "hardware": pa.string(),
    "family": pa.string(),
    "framework": pa.string(),
    "version": pa.string(),
    "configuration_id": pa.string(),
}


def _map_table(rows: list[MapRow]) -> pa.Table:
    return pa.table({n: pa.array([getattr(r, n) for r in rows], t) for n, t in _MAP_COLUMNS.items()})


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _publish_copy(src: Path, out: Path, sf: SourceFile, info: SourceInfo, dest_rel: str) -> MapRow:
    dst = out / dest_rel
    _copy(src, dst)
    digest = file_sha256(src)
    if file_sha256(dst) != digest:
        raise PublishError(f"copy changed bytes: {sf.relative}")
    size = src.stat().st_size
    return MapRow(
        source_path=info.repo_path(sf.relative),
        destination_path=dest_rel,
        kind=sf.kind,
        conversion="copy",
        hash_basis="file-bytes",
        source_sha256=digest,
        source_bytes=size,
        source_git_blob=info.blobs.get(sf.relative),
        destination_sha256=digest,
        destination_bytes=size,
        source_content_sha256=digest,
        destination_content_sha256=digest,
        rows=None,
        hardware=sf.hardware,
        family=sf.family,
        framework=sf.framework,
        version=sf.version,
    )


def _publish_table(src: Path, out: Path, sf: SourceFile, info: SourceInfo, dest_rel: str) -> MapRow:
    source_table = pq.read_table(src)
    source_hash = table_content_sha256(source_table)
    dst = out / dest_rel
    write_parquet(
        source_table,
        dst,
        OP_TABLE_SCHEMA_ID,
        extra_metadata={
            "aisim.source_path": info.repo_path(sf.relative),
            "aisim.source_rows_sha256": source_hash,
            "aisim.op_file": sf.name,
        },
    )
    published = read_parquet(dst, OP_TABLE_SCHEMA_ID)
    published_hash = table_content_sha256(published)
    if published_hash != source_hash or not source_table.replace_schema_metadata(None).equals(
        published.replace_schema_metadata(None)
    ):
        raise PublishError(f"rows changed by restamp: {sf.relative}")
    return MapRow(
        source_path=info.repo_path(sf.relative),
        destination_path=dest_rel,
        kind="table",
        conversion="parquet-restamp",
        hash_basis="arrow-rows",
        source_sha256=file_sha256(src),
        source_bytes=src.stat().st_size,
        source_git_blob=info.blobs.get(sf.relative),
        destination_sha256=file_sha256(dst),
        destination_bytes=dst.stat().st_size,
        source_content_sha256=source_hash,
        destination_content_sha256=published_hash,
        rows=published.num_rows,
        hardware=sf.hardware,
        family=sf.family,
        framework=sf.framework,
        version=sf.version,
    )


# --------------------------------------------------------------------------- YAML metadata -> Parquet


def _leading_comment(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
        else:
            break
    return "\n".join(lines)


_OPTIONAL_REUSE_STATS = ("total_raw_rows", "within_framework_dedup_rows", "median_pct_divergence")


def reuse_manifest_table(document: dict) -> pa.Table:
    if set(document) != {"groups"} or not isinstance(document["groups"], list):
        raise PublishError("perf_data_reuse_manifest.yaml must contain only a 'groups' list")
    for group in document["groups"]:
        unknown = set(group) - REUSE_GROUP_KEYS
        if unknown:
            raise PublishError(f"unknown reuse group keys: {sorted(unknown)}")
    groups = document["groups"]
    str_list = pa.list_(pa.string())
    return pa.table(
        {
            "op_file": pa.array([g["op_file"] for g in groups], pa.string()),
            "kernel_source": pa.array([g["kernel_source"] for g in groups], pa.string()),
            "tier": pa.array([g["tier"] for g in groups], pa.string()),
            "systems": pa.array([list(g["systems"]) for g in groups], str_list),
            "frameworks": pa.array([list(g["frameworks"]) for g in groups], str_list),
            "total_raw_rows": pa.array([g.get("total_raw_rows") for g in groups], pa.int64()),
            "within_framework_dedup_rows": pa.array([g.get("within_framework_dedup_rows") for g in groups], pa.int64()),
            "median_pct_divergence": pa.array([g.get("median_pct_divergence") for g in groups], pa.float64()),
        }
    )


def reuse_manifest_document(table: pa.Table) -> dict:
    """Inverse of :func:`reuse_manifest_table` (absent optional keys stay absent)."""
    groups = []
    for row in table.to_pylist():
        group = {k: v for k, v in row.items() if k not in _OPTIONAL_REUSE_STATS}
        for key in _OPTIONAL_REUSE_STATS:
            if row[key] is not None:
                group[key] = row[key]
        groups.append(group)
    return {"groups": groups}


def query_versions_table(document: dict) -> pa.Table:
    if set(document) != {"schema_version", "defaults", "overrides"}:
        raise PublishError("query_versions.yaml must contain schema_version, defaults and overrides")
    rows = [("defaults", None, fw, slot) for fw, slot in document["defaults"].items()]
    rows += [
        ("overrides", system, fw, slot)
        for system, per_framework in document["overrides"].items()
        for fw, slot in per_framework.items()
    ]
    for _, _, _, slot in rows:
        if set(slot) - {"current", "previous"}:
            raise PublishError(f"unknown version slot keys: {sorted(slot)}")
    return pa.table(
        {
            "scope": pa.array([r[0] for r in rows], pa.string()),
            "system": pa.array([r[1] for r in rows], pa.string()),
            "framework": pa.array([r[2] for r in rows], pa.string()),
            "current": pa.array([r[3].get("current") for r in rows], pa.string()),
            "previous": pa.array([r[3].get("previous") for r in rows], pa.string()),
        }
    )


def query_versions_document(table: pa.Table, schema_version: int) -> dict:
    document: dict = {"schema_version": schema_version, "defaults": {}, "overrides": {}}
    for row in table.to_pylist():
        slot = {"current": row["current"], "previous": row["previous"]}
        if row["scope"] == "defaults":
            document["defaults"][row["framework"]] = slot
        else:
            document["overrides"].setdefault(row["system"], {})[row["framework"]] = slot
    return document


def _publish_yaml_table(
    src: Path,
    out: Path,
    info: SourceInfo,
    dest_rel: str,
    schema_id: str,
    build,
    rebuild,
) -> MapRow:
    relative = info.absolute_repo_path(src)
    raw = src.read_bytes()
    text = raw.decode("utf-8")
    document = yaml.safe_load(text)
    table = build(document)
    dst = out / dest_rel
    source_sha = hashlib.sha256(raw).hexdigest()
    write_parquet(
        table,
        dst,
        schema_id,
        extra_metadata={
            "aisim.source_path": relative,
            "aisim.source_sha256": source_sha,
            "aisim.source_yaml": text,
            "aisim.source_header": _leading_comment(text),
        },
    )
    published = read_parquet(dst, schema_id)
    embedded = (pq.read_metadata(dst).metadata or {})[b"aisim.source_yaml"]
    if hashlib.sha256(embedded).hexdigest() != source_sha:
        raise PublishError(f"embedded source text differs: {relative}")
    if _json_sha256(_jsonable(rebuild(published, document))) != _json_sha256(_jsonable(document)):
        raise PublishError(f"converted rows do not reproduce the YAML document: {relative}")
    return MapRow(
        source_path=relative,
        destination_path=dest_rel,
        kind="lookup-metadata",
        conversion="yaml-to-parquet",
        hash_basis="embedded-source",
        source_sha256=source_sha,
        source_bytes=len(raw),
        source_git_blob=info.blob_of(src),
        destination_sha256=file_sha256(dst),
        destination_bytes=dst.stat().st_size,
        source_content_sha256=source_sha,
        destination_content_sha256=hashlib.sha256(embedded).hexdigest(),
        rows=published.num_rows,
        hardware=None,
        family=None,
        framework=None,
        version=None,
    )


# --------------------------------------------------------------------------- manifests and indexes


def _artifact(out: Path, rel: str, *, artifact_id: str, role: str, fmt: str, schema: str, **extra) -> Artifact:
    path = out / rel
    return Artifact.model_validate(
        {
            "artifact_id": artifact_id,
            "path": rel,
            "role": role,
            "format": fmt,
            "schema": schema,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            **extra,
        }
    )


def _sidecar_schema(kind: str, source: Path) -> tuple[str, dict]:
    if kind in ("collection-meta", "reuse-declaration"):
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        version = document.get("schema_version", 1) if isinstance(document, dict) else 1
        prefix = "aisim-collection-meta" if kind == "collection-meta" else "aisim-reuse-declaration"
        return f"{prefix}/v{version}", document if isinstance(document, dict) else {}
    return "collector-errors/unversioned", {}


_ROLES = {
    "table": "operation-measurements",
    "collection-meta": "collection-meta",
    "reuse-declaration": "reuse-declaration",
    "auxiliary": "collection-auxiliary",
}


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _build_manifest(
    out: Path,
    source: Path,
    key: tuple[str, str, str, str],
    files: list[SourceFile],
    rows_by_dest: dict[str, MapRow],
    info: SourceInfo,
) -> tuple[Manifest, str, list[dict]]:
    hardware, family, framework, version = key
    identity = Identity(
        hardware=hardware,
        framework=framework,
        framework_version=version,
        operation_family=family,
        model=None,
        effective_settings={},
        unresolved_identity=None,
    )
    cfg = configuration_id("op-latency", "operation-tables", identity)
    artifacts, mapping, declared, collection = [], [], [], {}
    for sf in files:
        dest_rel = f"{DATA_PREFIX}/{sf.relative}"
        row = rows_by_dest[dest_rel]
        row.configuration_id = cfg
        runtime_path = f"data/{sf.relative}"
        if sf.kind == "table":
            artifact = _artifact(
                out,
                dest_rel,
                artifact_id=sf.name,
                role=_ROLES["table"],
                fmt="parquet",
                schema=OP_TABLE_SCHEMA_ID,
                rows=row.rows,
                runtime_path=runtime_path,
            )
        else:
            schema, document = _sidecar_schema(sf.kind, source / sf.relative)
            suffix = sf.name.rsplit(".", 1)[-1]
            artifact = _artifact(
                out,
                dest_rel,
                artifact_id=sf.name,
                role=_ROLES.get(sf.kind, "collection-auxiliary"),
                fmt="yaml" if suffix == "yaml" else suffix,
                schema=schema,
                runtime_path=runtime_path,
            )
            if sf.kind == "reuse-declaration":
                declared = [
                    {k: str(e[k]) for k in ("table", "from_version", "approved_by")} for e in document.get("reuse", [])
                ]
            if sf.kind == "collection-meta":
                collection = _jsonable(
                    {k: document[k] for k in ("schema_version", "provenance", "runtime") if k in document}
                )
        artifacts.append(artifact)
        mapping.append(
            {
                "artifact_id": sf.name,
                "source_path": row.source_path,
                "source_git_blob": row.source_git_blob,
                "source_sha256": row.source_sha256,
                "hash_basis": row.hash_basis,
                "source_content_sha256": row.source_content_sha256,
            }
        )
    manifest = Manifest(
        schema_version="aisim-artifact-manifest/v1",
        category="op-latency",
        purpose="operation-tables",
        configuration_id=cfg,
        identity=identity,
        requested_settings=None,
        artifacts=artifacts,
        provenance={
            "publisher": PUBLISHER,
            "source_commit": info.commit,
            "source_directory": info.repo_path("/".join(key)),
            "collection_meta": collection,
            "declared_reuse": declared,
            "source_to_destination": mapping,
        },
        evidence_gaps=[],
    )
    manifest_rel = f"{DATA_PREFIX}/{'/'.join(key)}/manifest.json"
    _write_json(out / manifest_rel, manifest.model_dump(by_alias=True, mode="json"))
    return manifest, manifest_rel, declared


def _declared_reuse_table(rows: list[dict]) -> pa.Table:
    columns = {
        "hardware": pa.string(),
        "family": pa.string(),
        "framework": pa.string(),
        "version": pa.string(),
        "table": pa.string(),
        "from_version": pa.string(),
        "approved_by": pa.string(),
        "reason": pa.string(),
        "declaration_path": pa.string(),
        "consumer_manifest": pa.string(),
        "donor_manifest": pa.string(),
        "donor_configuration_id": pa.string(),
        "donor_table_path": pa.string(),
        "status": pa.string(),
    }
    return pa.table({n: pa.array([r[n] for r in rows], t) for n, t in columns.items()})


def _resolve_declared_reuse(out: Path, manifests: dict[tuple, tuple[Manifest, str]]) -> list[dict]:
    """One row per ``reuse.yaml`` entry, with the donor's manifest and table resolved or flagged."""
    rows = []
    for (hardware, family, framework, version), (manifest, manifest_rel) in sorted(manifests.items()):
        decl = next((a for a in manifest.artifacts if a.role == "reuse-declaration"), None)
        if decl is None:
            continue
        document = yaml.safe_load((out / decl.path).read_text(encoding="utf-8"))
        for entry in document["reuse"]:
            donor = manifests.get((hardware, family, framework, str(entry["from_version"])))
            table_name = f"{entry['table']}.parquet"
            donor_table = None
            if donor is not None:
                donor_table = next((a.path for a in donor[0].artifacts if a.path.endswith(f"/{table_name}")), None)
            status = "donor-missing" if donor is None else "resolved" if donor_table else "table-missing"
            rows.append(
                {
                    "hardware": hardware,
                    "family": family,
                    "framework": framework,
                    "version": version,
                    "table": str(entry["table"]),
                    "from_version": str(entry["from_version"]),
                    "approved_by": str(entry["approved_by"]),
                    "reason": str(entry["reason"]).strip(),
                    "declaration_path": decl.path,
                    "consumer_manifest": manifest_rel,
                    "donor_manifest": donor[1] if donor else None,
                    "donor_configuration_id": donor[0].configuration_id if donor else None,
                    "donor_table_path": donor_table,
                    "status": status,
                }
            )
    return rows


# --------------------------------------------------------------------------- build


@dataclass
class BuildResult:
    out: Path
    source_commit: str | None
    n_manifests: int
    n_files: int
    n_converted_tables: int
    n_copied: int
    n_skipped: int
    source_bytes: int
    published_bytes: int
    declared_reuse: dict[str, int] = field(default_factory=dict)
    map_rows: list[MapRow] = field(default_factory=list)


def build_publish_tree(
    source_root: str | Path,
    reuse_manifest: str | Path,
    query_versions: str | Path,
    out: str | Path,
    *,
    dataset_id: str = "local/op-latency-publish",
    systems_dir: str | Path | None = None,
    allow_dirty: bool = False,
) -> BuildResult:
    """Build the publish tree under ``out`` (which must not exist). Local only; never touches the network."""
    source, out = Path(source_root).resolve(), Path(out).absolute()
    if out.exists():
        raise PublishError(f"output directory already exists: {out}")
    if out == source or source in out.parents:
        raise PublishError("output must not be inside the source tree")
    info = inspect_source(source, allow_dirty=allow_dirty)
    files = scan_source(source)
    out.mkdir(parents=True)
    try:
        return _build(source, Path(reuse_manifest), Path(query_versions), out, files, info, dataset_id, systems_dir)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def _build(source, reuse_manifest, query_versions, out, files, info, dataset_id, systems_dir) -> BuildResult:
    rows: list[MapRow] = []
    for sf in files:
        dest_rel = f"{DATA_PREFIX}/{sf.relative}"
        if sf.kind == "skipped":
            rows.append(
                MapRow(
                    source_path=info.repo_path(sf.relative),
                    destination_path=dest_rel,
                    kind="skipped",
                    conversion="skipped",
                    hash_basis="file-bytes",
                    source_sha256=file_sha256(source / sf.relative),
                    source_bytes=(source / sf.relative).stat().st_size,
                    source_git_blob=info.blobs.get(sf.relative),
                    destination_sha256=None,
                    destination_bytes=None,
                    source_content_sha256=None,
                    destination_content_sha256=None,
                    rows=None,
                    hardware=sf.hardware,
                    family=None,
                    framework=None,
                    version=None,
                )
            )
        elif sf.kind == "table":
            rows.append(_publish_table(source / sf.relative, out, sf, info, dest_rel))
        else:
            rows.append(_publish_copy(source / sf.relative, out, sf, info, dest_rel))

    for name, src, build, rebuild, schema in (
        (
            REUSE_MANIFEST_NAME,
            reuse_manifest,
            reuse_manifest_table,
            lambda table, _doc: reuse_manifest_document(table),
            OP_REUSE_MANIFEST_SCHEMA_ID,
        ),
        (
            QUERY_VERSIONS_NAME,
            query_versions,
            query_versions_table,
            lambda table, doc: query_versions_document(table, doc["schema_version"]),
            OP_QUERY_VERSIONS_SCHEMA_ID,
        ),
    ):
        row = _publish_yaml_table(src, out, info, f"{DATA_PREFIX}/{name}.parquet", schema, build, rebuild)
        rows.append(row)

    by_dest = {r.destination_path: r for r in rows}
    groups: dict[tuple, list[SourceFile]] = defaultdict(list)
    for sf in files:
        if sf.version_dir is not None and sf.kind != "skipped":
            groups[sf.version_dir].append(sf)
    manifests: dict[tuple, tuple[Manifest, str]] = {}
    for key in sorted(groups):
        manifest, manifest_rel, _ = _build_manifest(out, source, key, groups[key], by_dest, info)
        manifests[key] = (manifest, manifest_rel)

    entries = []
    for key, (manifest, manifest_rel) in sorted(manifests.items(), key=lambda kv: kv[1][1]):
        entries.append(
            Entry(
                configuration_id=manifest.configuration_id,
                purpose="operation-tables",
                identity=manifest.identity,
                manifest=_artifact(
                    out,
                    manifest_rel,
                    artifact_id="manifest",
                    role="manifest",
                    fmt="json",
                    schema="aisim-artifact-manifest/v1",
                ),
            )
        )
    _write_json(
        out / CATEGORY_INDEX_PATH,
        CategoryIndex(schema_version="aisim-category-index/v1", category="op-latency", entries=entries).model_dump(
            by_alias=True, mode="json"
        ),
    )
    _write_json(
        out / DATASET_INDEX_PATH,
        DatasetIndex(
            schema_version="aisim-dataset-index/v1",
            dataset_id=dataset_id,
            categories={
                "op-latency": CategoryReference(
                    root=DATA_PREFIX,
                    layout="category",
                    index=_artifact(
                        out,
                        CATEGORY_INDEX_PATH,
                        artifact_id="op-latency-index",
                        role="index",
                        fmt="json",
                        schema="aisim-category-index/v1",
                    ),
                )
            },
        ).model_dump(by_alias=True, mode="json"),
    )

    declared = _resolve_declared_reuse(out, manifests)
    write_parquet(_declared_reuse_table(declared), out / DECLARED_REUSE_PATH, OP_DECLARED_REUSE_SCHEMA_ID)
    build_catalog(out, systems_dir=systems_dir if systems_dir is not None else source.parent)

    # The source map is written last because it records every other file's hashes.
    rows.sort(key=lambda r: (r.source_path or "", r.destination_path))
    write_parquet(
        _map_table(rows),
        out / SOURCE_MAP_PATH,
        OP_SOURCE_MAP_SCHEMA_ID,
        extra_metadata={"aisim.source_commit": info.commit or "", "aisim.publisher": PUBLISHER},
    )
    check_dataset(out)

    published = [r for r in rows if r.conversion != "skipped"]
    counts: dict[str, int] = defaultdict(int)
    for row in declared:
        counts[row["status"]] += 1
    return BuildResult(
        out=out,
        source_commit=info.commit,
        n_manifests=len(manifests),
        n_files=len(published),
        n_converted_tables=sum(1 for r in published if r.conversion != "copy"),
        n_copied=sum(1 for r in published if r.conversion == "copy"),
        n_skipped=len(rows) - len(published),
        source_bytes=sum(r.source_bytes or 0 for r in rows),
        published_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        declared_reuse=dict(counts),
        map_rows=rows,
    )


# --------------------------------------------------------------------------- audit


@dataclass
class AuditReport:
    checked: int = 0
    by_basis: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    unmapped_destinations: list[str] = field(default_factory=list)
    unmapped_sources: list[str] = field(default_factory=list)
    reuse_status: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.problems or self.unmapped_destinations or self.unmapped_sources)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "by_hash_basis": self.by_basis,
            "problems": self.problems,
            "unmapped_destinations": self.unmapped_destinations,
            "unmapped_sources": self.unmapped_sources,
            "declared_reuse_status": self.reuse_status,
        }


def audit_tree(out: str | Path, source_root: str | Path, *, reuse_manifest: str | Path, query_versions: str | Path):
    """Re-verify every published file against its source from disk, independent of the build.

    Checks, per source-map row: destination bytes/SHA-256 on disk equal the recorded values, the source
    still has the recorded hash, and the per-basis content hashes agree (see module docstring). Also checks
    that the destination inventory equals the map plus the generated metadata files, and that every source
    file is mapped.
    """
    out, source = Path(out), Path(source_root).resolve()
    report = AuditReport()
    table = read_parquet(out / SOURCE_MAP_PATH, OP_SOURCE_MAP_SCHEMA_ID).to_pylist()
    external = {
        f"{DATA_PREFIX}/{REUSE_MANIFEST_NAME}.parquet": Path(reuse_manifest),
        f"{DATA_PREFIX}/{QUERY_VERSIONS_NAME}.parquet": Path(query_versions),
    }
    mapped_sources = set()
    for row in table:
        dest_rel = row["destination_path"]
        if row["conversion"] == "skipped":
            continue
        report.checked += 1
        basis = row["hash_basis"]
        report.by_basis[basis] = report.by_basis.get(basis, 0) + 1
        src_path = external.get(dest_rel) or source / dest_rel.removeprefix(f"{DATA_PREFIX}/")
        dst = out / dest_rel
        if not dst.is_file() or not src_path.is_file():
            report.problems.append(f"missing file for {dest_rel}")
            continue
        mapped_sources.add(src_path.resolve())
        src_sha, dst_sha = file_sha256(src_path), file_sha256(dst)
        if src_sha != row["source_sha256"]:
            report.problems.append(f"source changed since build: {dest_rel}")
        if dst_sha != row["destination_sha256"] or dst.stat().st_size != row["destination_bytes"]:
            report.problems.append(f"destination differs from recorded hash: {dest_rel}")
        if basis == "file-bytes":
            if src_sha != dst_sha:
                report.problems.append(f"byte hash mismatch: {dest_rel}")
        elif basis == "arrow-rows":
            if read_parquet_schema_id(dst) != OP_TABLE_SCHEMA_ID:
                report.problems.append(f"schema ID missing: {dest_rel}")
            a, b = table_content_sha256(pq.read_table(src_path)), table_content_sha256(pq.read_table(dst))
            if not (a == b == row["source_content_sha256"] == row["destination_content_sha256"]):
                report.problems.append(f"row hash mismatch: {dest_rel}")
        elif basis == "embedded-source":
            embedded = (pq.read_metadata(dst).metadata or {}).get(b"aisim.source_yaml", b"")
            if hashlib.sha256(embedded).hexdigest() != src_sha:
                report.problems.append(f"embedded source hash mismatch: {dest_rel}")
        else:
            report.problems.append(f"unknown hash basis {basis!r}: {dest_rel}")

    generated = {SOURCE_MAP_PATH}
    in_tree = {p.relative_to(out).as_posix() for p in (out / DATA_PREFIX).rglob("*") if p.is_file()}
    mapped_dest = {r["destination_path"] for r in table if r["conversion"] != "skipped"}
    manifests = {p for p in in_tree if p.endswith("/manifest.json")}
    report.unmapped_destinations = sorted(in_tree - mapped_dest - manifests - generated)
    for sf in scan_source(source):
        if sf.kind != "skipped" and (source / sf.relative).resolve() not in mapped_sources:
            report.unmapped_sources.append(sf.relative)
    declared = read_parquet(out / DECLARED_REUSE_PATH, OP_DECLARED_REUSE_SCHEMA_ID).to_pylist()
    for row in declared:
        report.reuse_status[row["status"]] = report.reuse_status.get(row["status"], 0) + 1
    try:
        check_dataset(out)  # manifest/index hashes, sizes, row counts, identity IDs
    except ValueError as exc:
        report.problems.append(f"check_dataset: {exc}")
    return report
