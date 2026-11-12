# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permanent op-latency dataset contract: row models, schema IDs, layout paths and content hashes.

The op-latency category of the unified dataset (``data/op-latency/``, ``catalog/op-latency*.{json,parquet}``) is
read by the data library (``schemas.py`` registers the row models) and written by the one-off converter in
``tools/perf_database/op_latency_publish.py``. This module holds only what both sides share; it reads no files
and has no side effects.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

OP_TABLE_SCHEMA_ID = "aisim-op-latency/v1"
OP_REUSE_MANIFEST_SCHEMA_ID = "aisim-op-reuse-manifest/v1"
OP_QUERY_VERSIONS_SCHEMA_ID = "aisim-op-query-versions/v1"
OP_DECLARED_REUSE_SCHEMA_ID = "aisim-op-declared-reuse/v1"
OP_SOURCE_MAP_SCHEMA_ID = "aisim-op-source-map/v1"

# --- Row models for the registered schemas (JSON Schema snapshots in schemas.py) ---


class _OpRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class OpLatencyRow(BaseModel):
    """One row of an op measurement table (``aisim-op-latency/v1``).

    Columns differ per operation family and file (116 distinct names across the
    packaged tables); only ``kernel_source`` is common to all. The schema
    therefore pins ``kernel_source`` and allows the family-specific columns,
    whose names, types and units are defined by the collector for each family.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    kernel_source: str


class OpReuseManifestRow(_OpRow):
    """One group of ``perf_data_reuse_manifest.yaml`` (``aisim-op-reuse-manifest/v1``)."""

    op_file: str
    kernel_source: str
    tier: str
    systems: list[str]
    frameworks: list[str]
    total_raw_rows: int
    within_framework_dedup_rows: int
    median_pct_divergence: float | None = None


class OpQueryVersionsRow(_OpRow):
    """One default or per-system override slot of ``query_versions.yaml`` (``aisim-op-query-versions/v1``)."""

    scope: str  # "defaults" | "overrides"
    system: str | None
    framework: str
    current: str | None
    previous: str | None


class OpDeclaredReuseRow(_OpRow):
    """One declared reuse with its donor resolved (``aisim-op-declared-reuse/v1``)."""

    hardware: str
    family: str
    framework: str
    version: str
    table: str
    from_version: str | None
    approved_by: str | None
    reason: str | None
    declaration_path: str
    consumer_manifest: str
    donor_manifest: str | None
    donor_configuration_id: str | None
    donor_table_path: str | None
    status: str


class OpSourceMapRow(_OpRow):
    """One source-to-destination file mapping (``aisim-op-source-map/v1``)."""

    source_path: str | None
    destination_path: str
    kind: str
    conversion: str
    hash_basis: str
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


DATA_PREFIX = "data/op-latency"
REUSE_MANIFEST_NAME = "perf_data_reuse_manifest"
QUERY_VERSIONS_NAME = "query_versions"
SOURCE_MAP_PATH = f"{DATA_PREFIX}/source_map.parquet"
DECLARED_REUSE_PATH = "catalog/op-latency-reuse.parquet"
CATEGORY_INDEX_PATH = "catalog/op-latency.json"
DATASET_INDEX_PATH = "catalog/dataset.json"
SKIPPED_NAMES = {".gitkeep"}

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


# --------------------------------------------------------------------------- hashing


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table_content_sha256(table: pa.Table) -> str:
    """SHA-256 of the Arrow IPC stream of ``table`` without schema-level metadata.

    Independent of compression, row-group layout, chunking and the stamped schema ID, so it compares rows
    across a restamp.
    """
    table = table.replace_schema_metadata(None).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()
