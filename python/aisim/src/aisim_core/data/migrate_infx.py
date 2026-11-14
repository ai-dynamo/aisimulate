# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time migration of the private InferenceX cache to ``aisim-e2e/v1``.

The cache (``ai-dynamo/aisim-infx-cache``) stores ``e2e-source-index/v1``,
``inferencex-input-snapshot/v1`` and ``inferencex-{observations,metrics}-v1``
Parquet partitions. :func:`read_infx_cache` reads and *verifies* a local copy of a
pinned revision (hashes, row counts, record hashes, partition identity, snapshot
consistency; the same checks the current HF reader applies). :func:`migrate_infx_cache`
converts it with the shared :func:`~aisim_core.data.e2e.inferencex_to_configurations`
converter into a local ``aisim-e2e/v1`` dataset:

``data/e2e/<model>/<hardware>/<framework>/<version>/<parallelism>/<cfg-id>/``
    manifest, observations, metrics and one raw-output bundle (every raw record, hash-checked per row)
``catalog/e2e.json``, ``catalog/dataset.json``
    category and dataset indexes
``catalog/e2e/snapshots/<snapshot-id>/``
    frozen membership and acquisition tables
``data/e2e/responses/``
    the cached raw API responses, byte-identical, referenced by hash from the snapshot

Offline only. Nothing here reads or writes Hugging Face; callers download a
revision first and publish (or not) afterwards.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from aisim_core.data import Artifact, CategoryIndex, CategoryReference, DatasetIndex, Entry
from aisim_core.data.e2e import (
    PURPOSE,
    E2eConfigurationData,
    SnapshotAcquisition,
    SnapshotMember,
    inferencex_to_configurations,
    load_e2e,
    read_snapshot,
    write_configuration,
    write_snapshot,
)

SOURCE_INDEX_PATH = "catalog/e2e/inferencex.json"
SOURCE_INDEX_SCHEMA = "e2e-source-index/v1"
SOURCE_SNAPSHOT_SCHEMA = "inferencex-input-snapshot/v1"
SOURCE_OBSERVATIONS_SCHEMA = "inferencex-observations-v1"
SOURCE_METRICS_SCHEMA = "inferencex-metrics-v1"
SOURCE_API_SCHEMA = "inferencex-api-json/v1"
PRODUCER_URL = re.compile(
    r"https://github\.com/SemiAnalysisAI/InferenceX/actions/runs/([1-9][0-9]*)/attempts/([1-9][0-9]*)"
)
DATA_PREFIX = "data"
SNAPSHOT_DIR = "catalog/e2e/snapshots"
RESPONSES_DIR = "data/e2e/responses"


class MigrationError(ValueError):
    """The source cache is incomplete, inconsistent or fails a verification."""


def canonical(value: Any) -> bytes:
    """Canonical JSON bytes; the cache's record-hash contract."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe(root: Path, relative: str) -> Path:
    """Resolve a cache-relative path, refusing escapes and symlinks."""
    parts = relative.split("/")
    if not relative or relative.startswith("/") or "\\" in relative or any(p in {"", ".", ".."} for p in parts):
        raise MigrationError(f"unsafe cache path: {relative!r}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise MigrationError(f"symlink in cache path: {relative}")
    if not path.is_file():
        raise MigrationError(f"missing cache artifact: {relative}")
    return path


def _verified(root: Path, entry: Mapping[str, Any]) -> Path:
    path = _safe(root, entry["path"])
    data = path.read_bytes()
    if len(data) != entry["bytes"] or _sha(data) != entry["sha256"]:
        raise MigrationError(f"cache artifact checksum or size mismatch: {entry['path']}")
    return path


# --------------------------------------------------------------------------- reading the cache


@dataclass(frozen=True)
class InfxCache:
    """A verified local copy of one revision of the private InferenceX cache."""

    root: Path
    index: dict[str, Any]
    snapshot: dict[str, Any]
    records: list[dict[str, Any]]  # raw InferenceX records, one per observation
    observation_rows: list[dict[str, Any]]  # ``inferencex-observations-v1`` rows, unchanged
    metric_rows: list[dict[str, Any]]  # ``inferencex-metrics-v1`` rows, unchanged
    partition_of: dict[str, tuple[str, str, str]]  # observation_id -> (source_model, hardware, framework)
    response_paths: list[str] = field(default_factory=list)


def read_infx_cache(root: Path | str) -> InfxCache:
    """Read and verify a cache revision (index, snapshot, partitions and their records)."""
    root = Path(root)
    index = json.loads(_safe(root, SOURCE_INDEX_PATH).read_bytes())
    if index.get("schema_version") != SOURCE_INDEX_SCHEMA or index.get("source_kind") != "inferencex":
        raise MigrationError("unsupported source index")
    snapshot_ref = index["snapshot"]
    snapshot = json.loads(_verified(root, snapshot_ref).read_bytes())
    if (
        snapshot.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA
        or snapshot.get("snapshot_id") != index["snapshot_id"]
        or snapshot.get("completeness") != "complete"
        or snapshot.get("partitions") != index["partitions"]
    ):
        raise MigrationError("snapshot is incomplete or inconsistent with the index")
    records, observation_rows, metric_rows = [], [], []
    partition_of: dict[str, tuple[str, str, str]] = {}
    for partition in index["partitions"]:
        by_schema = {a["schema"]: a for a in partition["artifacts"]}
        if set(by_schema) != {SOURCE_OBSERVATIONS_SCHEMA, SOURCE_METRICS_SCHEMA} or len(partition["artifacts"]) != 2:
            raise MigrationError("partition requires exactly one observation and one metric artifact")
        tables = {}
        for schema, entry in by_schema.items():
            table = pq.ParquetFile(_verified(root, entry)).read()
            if table.num_rows != entry["rows"] or (table.schema.metadata or {}).get(b"schema_id") != schema.encode():
                raise MigrationError(f"schema or row count mismatch: {entry['path']}")
            tables[schema] = table.to_pylist()
        for row in tables[SOURCE_OBSERVATIONS_SCHEMA]:
            raw = json.loads(row["raw_record_json"])
            if (
                _sha(canonical(raw)) != row["record_sha256"]
                or row["observation_id"] != "inferencex:" + str(raw["id"])
                or any(
                    raw[key] != partition[field_]
                    for key, field_ in (("model", "source_model"), ("hardware", "hardware"), ("framework", "framework"))
                )
            ):
                raise MigrationError(f"observation identity mismatch: {row['observation_id']}")
            if row["observation_id"] in partition_of:
                raise MigrationError(f"duplicate observation: {row['observation_id']}")
            partition_of[row["observation_id"]] = (
                partition["source_model"],
                partition["hardware"],
                partition["framework"],
            )
            records.append(raw)
            observation_rows.append(row)
        known = {r["observation_id"] for r in tables[SOURCE_OBSERVATIONS_SCHEMA]}
        stray = {r["observation_id"] for r in tables[SOURCE_METRICS_SCHEMA]} - known
        if stray:
            raise MigrationError(f"metric rows reference unknown observations: {sorted(stray)[:3]}")
        metric_rows.extend(tables[SOURCE_METRICS_SCHEMA])
    if len(records) != snapshot["counts"]["observations_distinct"]:
        raise MigrationError("snapshot observation count mismatch")
    responses = []
    for acquisition in snapshot.get("acquisitions", []):
        artifact = acquisition["artifact"]
        if artifact["schema"] != SOURCE_API_SCHEMA:
            raise MigrationError("unsupported acquisition artifact schema")
        _verified(root, artifact)
        responses.append(artifact["path"])
    return InfxCache(root, index, snapshot, records, observation_rows, metric_rows, partition_of, responses)


# --------------------------------------------------------------------------- converting


@dataclass(frozen=True)
class MigrationResult:
    """What :func:`migrate_infx_cache` wrote; the dataset is local only."""

    output: Path
    snapshot_id: str
    legacy_snapshot_id: str
    configurations: int
    observations: int
    metric_rows: int
    manifest_paths: dict[str, str]  # configuration_id -> manifest path (relative to ``output``)
    observation_manifest: dict[str, str]  # observation_id -> manifest path
    bytes: int = 0
    files: int = 0


def _artifact(root: Path, relative: str, *, artifact_id: str, role: str, fmt: str, schema: str) -> Artifact:
    data = (root / relative).read_bytes()
    return Artifact(
        artifact_id=artifact_id,
        path=relative,
        role=role,
        format=fmt,
        artifact_schema=schema,
        sha256=_sha(data),
        bytes=len(data),
    )


def dataset_size(root: Path) -> tuple[int, int]:
    """(bytes, files) of every regular file under ``root``."""
    sizes = [p.stat().st_size for p in Path(root).rglob("*") if p.is_file()]
    return sum(sizes), len(sizes)


def migrate_infx_cache(
    cache: InfxCache,
    output: Path | str,
    *,
    runs: Mapping[str, Mapping[str, Any]] | None = None,
    source: Mapping[str, Any] | None = None,
    dataset_id: str = "ai-dynamo/aisim-infx-cache",
) -> MigrationResult:
    """Convert a verified cache into a new local ``aisim-e2e/v1`` dataset at ``output``.

    ``runs`` maps a producer ``run_url`` to GitHub run metadata (``head_sha``,
    ``run_attempt``, ``run_started_at``, ``head_branch``, ``status``, ``conclusion``);
    it is kept verbatim in each manifest because freshness and image decisions order
    rows by it. ``source`` (repo, revision) is recorded in the snapshot scope.
    ``output`` must not exist or must be empty.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise MigrationError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    configurations: list[E2eConfigurationData] = inferencex_to_configurations(cache.records, runs)
    manifest_paths: dict[str, str] = {}
    observation_manifest: dict[str, str] = {}
    members: list[SnapshotMember] = []
    entries: list[Entry] = []
    for data in configurations:
        path = write_configuration(output, data, prefix=DATA_PREFIX)
        cid = data.body.configuration_id()
        manifest_paths[cid] = path
        manifest_artifact = _artifact(
            output, path, artifact_id=cid, role="manifest", fmt="json", schema="aisim-artifact-manifest/v1"
        )
        entries.append(
            Entry(configuration_id=cid, purpose=PURPOSE, identity=data.body.identity(), manifest=manifest_artifact)
        )
        for obs in data.observations:
            observation_manifest[obs.observation_id] = path
            members.append(
                SnapshotMember(
                    configuration_id=cid,
                    observation_id=obs.observation_id,
                    record_sha256=obs.record_sha256,
                    manifest_path=path,
                )
            )

    # Acquisition: one row per observation, from the response of its display model.
    by_model = {a["model"]: a for a in cache.snapshot.get("acquisitions", [])}
    display = {
        (p["source_model"], p["hardware"], p["framework"]): p["display_model"] for p in cache.index["partitions"]
    }
    acquisition = []
    for obs_id, key in sorted(cache.partition_of.items()):
        response = by_model.get(display[key])
        acquisition.append(
            SnapshotAcquisition(
                observation_id=obs_id,
                acquired_at=response["retrieved_at"] if response else None,
                source_uri=response["url"] if response else None,
                status="acquired" if response else "missing",
            )
        )
    responses = output / RESPONSES_DIR
    responses.mkdir(parents=True, exist_ok=True)
    for relative in cache.response_paths:
        shutil.copyfile(cache.root / relative, responses / Path(relative).name)
    legacy = {
        key: value
        for key, value in cache.snapshot.items()
        if key not in {"partitions", "source_scope", "snapshot_id", "schema_version"}
    }
    legacy.update(
        schema_version=cache.snapshot["schema_version"],
        snapshot_id=cache.snapshot["snapshot_id"],
        source_index_schema=cache.index["schema_version"],
        # Responses keep the cache's path layout; the files sit under RESPONSES_DIR here.
        acquisitions=[
            {**a, "artifact": {**a["artifact"], "path": f"{RESPONSES_DIR}/{Path(a['artifact']['path']).name}"}}
            for a in cache.snapshot.get("acquisitions", [])
        ],
    )
    scope = {**cache.snapshot["source_scope"], "legacy": legacy, "migrated_from": dict(source or {})}
    temporary = f"{SNAPSHOT_DIR}/pending/snapshot.json"
    (output / temporary).parent.mkdir(parents=True, exist_ok=True)
    snapshot = write_snapshot(
        output,
        temporary,
        members,
        source_scope=scope,
        selection_policy=cache.snapshot["selection_policy"],
        completeness="complete",
        acquisition=acquisition,
    )
    final = f"{SNAPSHOT_DIR}/{snapshot.snapshot_id}"
    shutil.move(str(output / SNAPSHOT_DIR / "pending"), str(output / final))
    # The table artifact paths embedded in the snapshot document name the pending directory; rewrite.
    snapshot, _ = _rewrite_snapshot_paths(output, final)

    entries.sort(key=lambda e: e.configuration_id)
    index = CategoryIndex(schema_version="aisim-category-index/v1", category="e2e", entries=entries)
    (output / "catalog").mkdir(exist_ok=True)
    (output / "catalog/e2e.json").write_text(index.model_dump_json(by_alias=True, indent=2) + "\n")
    reference = CategoryReference(
        root=f"{DATA_PREFIX}/e2e",
        index=_artifact(
            output,
            "catalog/e2e.json",
            artifact_id="e2e-index",
            role="category-index",
            fmt="json",
            schema="aisim-category-index/v1",
        ),
        layout="category",
    )
    dataset_index = DatasetIndex(
        schema_version="aisim-dataset-index/v1", dataset_id=dataset_id, categories={"e2e": reference}
    )
    (output / "catalog/dataset.json").write_text(dataset_index.model_dump_json(by_alias=True, indent=2) + "\n")
    size, files = dataset_size(output)
    return MigrationResult(
        output=output,
        snapshot_id=snapshot.snapshot_id,
        legacy_snapshot_id=cache.snapshot["snapshot_id"],
        configurations=len(configurations),
        observations=len(members),
        metric_rows=sum(len(c.metrics) for c in configurations),
        manifest_paths=manifest_paths,
        observation_manifest=observation_manifest,
        bytes=size,
        files=files,
    )


def _rewrite_snapshot_paths(root: Path, directory: str):
    """Point the snapshot document's table artifacts at their final directory and refresh the hashes."""
    from aisim_core.data.e2e import E2eInputSnapshot

    path = f"{directory}/snapshot.json"
    document = E2eInputSnapshot.model_validate_json((root / path).read_bytes())
    members = document.members.model_copy(update={"path": f"{directory}/{Path(document.members.path).name}"})
    acquisition = (
        document.acquisition.model_copy(update={"path": f"{directory}/{Path(document.acquisition.path).name}"})
        if document.acquisition
        else None
    )
    document = document.model_copy(update={"members": members, "acquisition": acquisition})
    (root / path).write_text(
        json.dumps(document.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True) + "\n"
    )
    return read_snapshot(root, path)


# --------------------------------------------------------------------------- verifying the output


@dataclass(frozen=True)
class DatasetCheck:
    configurations: int
    observations: int
    metric_rows: int
    snapshot_id: str
    problems: list[str]


def verify_migrated_dataset(root: Path | str) -> DatasetCheck:
    """Re-read the written dataset through the production loader; collect every problem."""
    root = Path(root)
    problems: list[str] = []
    dataset_index = DatasetIndex.model_validate_json((root / "catalog/dataset.json").read_bytes())
    reference = dataset_index.categories["e2e"]
    index = CategoryIndex.model_validate_json((root / reference.index.path).read_bytes())
    if _sha((root / reference.index.path).read_bytes()) != reference.index.sha256:
        problems.append("category index hash differs from dataset.json")
    observations = metrics = 0
    seen_obs: Counter[str] = Counter()
    for entry in index.entries:
        try:
            data = load_e2e(root, entry.manifest.path)
        except Exception as error:
            problems.append(f"{entry.manifest.path}: {error}")
            continue
        if data.manifest.configuration_id != entry.configuration_id or data.body.identity() != entry.identity:
            problems.append(f"{entry.manifest.path}: index and manifest identity differ")
        if _sha((root / entry.manifest.path).read_bytes()) != entry.manifest.sha256:
            problems.append(f"{entry.manifest.path}: index hash differs from manifest")
        observations += len(data.observations)
        metrics += len(data.metrics)
        seen_obs.update(o.observation_id for o in data.observations)
    problems.extend(f"observation in several configurations: {o}" for o, n in seen_obs.items() if n > 1)
    snapshots = sorted((root / SNAPSHOT_DIR).glob("*/snapshot.json"))
    snapshot_id = ""
    if len(snapshots) != 1:
        problems.append(f"expected one snapshot, found {len(snapshots)}")
    else:
        snapshot, members = read_snapshot(root, str(snapshots[0].relative_to(root)))
        snapshot_id = snapshot.snapshot_id
        if {m.observation_id for m in members} != set(seen_obs):
            problems.append("snapshot membership differs from the configurations' observations")
    return DatasetCheck(len(index.entries), observations, metrics, snapshot_id, problems)
