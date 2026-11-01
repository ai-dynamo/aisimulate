# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector provenance: the collector hash and the ``collection_meta.yaml`` writer.

Collector V3 makes the collector finalize step the provenance authority
(design: docs/perf_database/collector-v3-op-centric-design.md §5). This module
provides the building blocks:

- ``collector_hash``: a content hash over the collector source (see
  ``_HASHED_SOURCES``). Every table collected from the same source carries the
  same hash. Content-based, so it survives rebases.
- ``case_plan_hash``: a hash of the resolved/attested case-id set collected
  at run time (GPU needed to produce it — never recomputed in CI).
- ``write_collection_meta``: renders the design-§5 YAML deterministically.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import yaml

# The source collector_hash covers: the collector package, plus the SDK code and
# packaged model configs its runners import. JSON under collector/ is left out
# because the TRT-LLM MoE autotuners write their caches there.
_HASHED_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("collector", (".patch", ".py", ".sh", ".yaml", ".yml")),
    ("src/aisimulate_core/sdk", (".py",)),
    ("src/aisimulate_core/model_configs", (".json",)),
)

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
COLLECTION_META_SCHEMA_VERSION = 1
MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION = 2

_RUNTIME_FIELD_ORDER = (
    "framework",
    "version",
    "image",
    "image_variant",
    "image_digest",
    "source_commit",
    "abi",
    "live_abi",
    "transport",
    "backend_capability",
    "backend_abis",
    "backend_capabilities",
)
_RUNTIME_STRING_FIELDS = ("image", "image_variant", "image_digest", "source_commit")
_RUNTIME_MAPPING_FIELDS = (
    "abi",
    "live_abi",
    "transport",
    "backend_capability",
    "backend_abis",
    "backend_capabilities",
)
_TABLE_FIELD_ORDER = (
    "collector_ref",
    "collector_hash",
    "case_plan_hash",
    "collected_at",
    "rows",
    "classified_failures",
    "status",
)
_TABLE_REQUIRED_FIELD_ORDER = tuple(field for field in _TABLE_FIELD_ORDER if field != "classified_failures")
_COLLECTION_OPTIONAL_FIELD_ORDER = ("source_campaign_rows", "source_campaign_status", "runtime")


def hashed_files(repo_root: str | Path) -> list[str]:
    """The sorted repo-relative paths ``_HASHED_SOURCES`` selects, skipping ``__pycache__``."""
    root = Path(repo_root)
    return sorted(
        path.relative_to(root).as_posix()
        for directory, suffixes in _HASHED_SOURCES
        for path in (root / directory).rglob("*")
        if path.suffix in suffixes and "__pycache__" not in path.parts and path.is_file()
    )


def collector_hash(repo_root: str | Path) -> str:
    """Content hash over the collector source, the same for every table a source tree collects.

    Sha256 over sorted (relpath, file-bytes) pairs of :func:`hashed_files`,
    "sha256:<hex>" formatted. Content-based (never absolute paths or a commit
    SHA), so it is stable across rebases and repo relocation.
    """
    root = Path(repo_root)
    digest = hashlib.sha256()
    for relpath in hashed_files(root):
        content = (root / relpath).read_bytes()
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def case_plan_hash(case_ids: list[str]) -> str:
    """Hash of the sorted, deduplicated case-id set (the attested expanded plan)."""
    unique_sorted = sorted(set(case_ids))
    digest = hashlib.sha256("\n".join(unique_sorted).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def derive_table_status(*, unresolved_failed_count: int, had_module_failure: bool) -> str:
    """complete unless a ModuleCollectionFailure was recorded for one of the
    table's producing ops.

    Owner decision: recorded per-case
    failures do NOT demote a table. failure_handling.md's core doctrine is
    that a classified failure is DATA — deterministic framework limits (OOM
    at sweep extremes, kernel grid caps) land in the failure log by design,
    and demoting the table for them made every honest campaign
    unpublishable. The anti-false-success guarantees are unaffected and live
    elsewhere: a run that dies mid-way never finalizes (parquet without a
    matching ``tables`` entry fails the coverage gate / strict loader), and
    an op that produced zero rows still demotes via ``had_module_failure``.
    ``unresolved_failed_count`` is retained for observability at call sites
    (logged, recorded in error summaries) but no longer affects status.
    """
    del unresolved_failed_count  # observability-only; see docstring
    if had_module_failure:
        return STATUS_PARTIAL
    return STATUS_COMPLETE


def spdx_header() -> str:
    """The repo-standard copyright header, dated to the year of emission
    (the copyright CI check requires the year to cover the file's last commit).
    """
    return (
        f"# SPDX-FileCopyrightText: Copyright (c) {date.today().year} NVIDIA CORPORATION & AFFILIATES."
        " All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "\n"
    )


def _ordered_runtime(runtime: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise ValueError(f"{field} must be a mapping")
    unknown = sorted(set(runtime) - set(_RUNTIME_FIELD_ORDER))
    if unknown:
        raise ValueError(f"{field} has unsupported field(s): {', '.join(unknown)}")
    for key in ("framework", "version"):
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            raise ValueError(f"{field}.{key} must be a non-empty string")
    for key in _RUNTIME_STRING_FIELDS:
        if key in runtime and (not isinstance(runtime[key], str) or not runtime[key].strip()):
            raise ValueError(f"{field}.{key} must be a non-empty string when provided")
    for key in _RUNTIME_MAPPING_FIELDS:
        if key in runtime and not isinstance(runtime[key], dict):
            raise ValueError(f"{field}.{key} must be a mapping when provided")
    return {key: runtime[key] for key in _RUNTIME_FIELD_ORDER if key in runtime}


def _ordered_collection_event(event: Any, *, table: str, index: int) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError(f"{table}.collections[{index}] must be a mapping")

    missing = [field for field in _TABLE_REQUIRED_FIELD_ORDER if field not in event]
    if missing:
        raise ValueError(f"{table}.collections[{index}] missing required field(s): {', '.join(missing)}")

    allowed = {*_TABLE_FIELD_ORDER, *_COLLECTION_OPTIONAL_FIELD_ORDER}
    unknown = sorted(set(event) - allowed)
    if unknown:
        raise ValueError(f"{table}.collections[{index}] has unsupported field(s): {', '.join(unknown)}")

    for field in ("collector_ref", "collector_hash", "case_plan_hash", "collected_at"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise ValueError(f"{table}.collections[{index}].{field} must be a non-empty string")
    if event["status"] not in (STATUS_COMPLETE, STATUS_PARTIAL):
        raise ValueError(f"{table}.collections[{index}].status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'")
    if not isinstance(event["rows"], int) or isinstance(event["rows"], bool) or event["rows"] < 0:
        raise ValueError(f"{table}.collections[{index}].rows must be a non-negative integer")
    classified_failures = event.get("classified_failures")
    if classified_failures is not None and (
        not isinstance(classified_failures, int) or isinstance(classified_failures, bool) or classified_failures < 0
    ):
        raise ValueError(f"{table}.collections[{index}].classified_failures must be a non-negative integer")
    source_rows = event.get("source_campaign_rows")
    if source_rows is not None and (
        not isinstance(source_rows, int) or isinstance(source_rows, bool) or source_rows < event["rows"]
    ):
        raise ValueError(
            f"{table}.collections[{index}].source_campaign_rows must be an integer at least as large as rows"
        )
    source_status = event.get("source_campaign_status")
    if source_status is not None and source_status not in (STATUS_COMPLETE, STATUS_PARTIAL):
        raise ValueError(
            f"{table}.collections[{index}].source_campaign_status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'"
        )

    ordered = {
        field: event[field]
        for field in (*_TABLE_FIELD_ORDER, "source_campaign_rows", "source_campaign_status")
        if field in event
    }
    if "runtime" in event:
        ordered["runtime"] = _ordered_runtime(event["runtime"], field=f"{table}.collections[{index}].runtime")
    return ordered


def _ordered_multi_event_table(table: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{table} must be a mapping")
    if "rows" not in entry or "status" not in entry:
        raise ValueError(f"{table} missing required merged-table rows/status")
    if not isinstance(entry["rows"], int) or isinstance(entry["rows"], bool) or entry["rows"] < 0:
        raise ValueError(f"{table}.rows must be a non-negative integer")
    if entry["status"] not in (STATUS_COMPLETE, STATUS_PARTIAL):
        raise ValueError(f"{table}.status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'")

    collections = entry.get("collections")
    if collections is None:
        collections = [entry]
    if not isinstance(collections, list) or not collections:
        raise ValueError(f"{table}.collections must be a non-empty list")

    return {
        "rows": entry["rows"],
        "status": entry["status"],
        "collections": [
            _ordered_collection_event(event, table=table, index=index) for index, event in enumerate(collections)
        ],
    }


def validate_collection_meta_for_update(
    document: Any,
    *,
    tables_to_update: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Validate an existing sidecar before a collector mutates table data.

    Historical ``local``/``collected`` v1 sidecars may contain reduced table
    summaries. They can be preserved beside a new table, but cannot take part
    in a v2 history promotion when any existing table is updated.
    """
    if not isinstance(document, dict):
        raise ValueError("collection_meta.yaml must be a mapping")
    unknown_document_fields = sorted(set(document) - {"schema_version", "provenance", "runtime", "tables"})
    if unknown_document_fields:
        raise ValueError(f"collection_meta.yaml has unsupported field(s): {', '.join(unknown_document_fields)}")
    schema_version = document.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in (COLLECTION_META_SCHEMA_VERSION, MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION)
    ):
        raise ValueError(f"unsupported collection_meta.yaml schema_version {schema_version!r}")
    provenance_tier = document.get("provenance")
    if provenance_tier is not None and (not isinstance(provenance_tier, str) or not provenance_tier.strip()):
        raise ValueError("collection_meta.yaml provenance must be a non-empty string when provided")
    if provenance_tier == "legacy":
        raise ValueError("legacy-tier collection_meta.yaml cannot be updated by a fresh collection")
    _ordered_runtime(document.get("runtime"), field="runtime")

    tables = document.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("collection_meta.yaml tables must be a mapping")
    if any(not isinstance(table, str) or not table for table in tables):
        raise ValueError("collection_meta.yaml table names must be non-empty strings")

    if schema_version == MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION:
        for table, entry in tables.items():
            if not isinstance(entry, dict) or set(entry) != {"rows", "status", "collections"}:
                raise ValueError(f"{table} must contain exactly rows, status, and collections")
            _ordered_multi_event_table(table, entry)
        return

    reduced_tables: set[str] = set()
    for table, entry in tables.items():
        if isinstance(entry, dict) and set(entry) == {"status"} and provenance_tier in {"local", "collected"}:
            if entry["status"] not in (STATUS_COMPLETE, STATUS_PARTIAL):
                raise ValueError(f"{table}.status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'")
            reduced_tables.add(table)
            continue
        if (
            isinstance(entry, dict)
            and set(entry) == {"collected_at", "rows", "status"}
            and provenance_tier in {"local", "collected"}
        ):
            if not isinstance(entry["collected_at"], str) or not entry["collected_at"].strip():
                raise ValueError(f"{table}.collected_at must be a non-empty string")
            if not isinstance(entry["rows"], int) or isinstance(entry["rows"], bool) or entry["rows"] < 0:
                raise ValueError(f"{table}.rows must be a non-negative integer")
            if entry["status"] not in (STATUS_COMPLETE, STATUS_PARTIAL):
                raise ValueError(f"{table}.status must be '{STATUS_COMPLETE}' or '{STATUS_PARTIAL}'")
            reduced_tables.add(table)
            continue
        if not isinstance(entry, dict) or not set(_TABLE_REQUIRED_FIELD_ORDER) <= set(entry) <= set(_TABLE_FIELD_ORDER):
            raise ValueError(
                f"{table} must contain {', '.join(_TABLE_REQUIRED_FIELD_ORDER)} and may include classified_failures"
            )
        _ordered_collection_event(entry, table=table, index=0)

    if reduced_tables and set(tables_to_update) & set(tables):
        raise ValueError(
            "reduced historical table metadata cannot be promoted while updating an existing sidecar table"
        )


def append_collection_event(
    existing: dict[str, Any], current: dict[str, Any], *, table: str, merged_rows: int
) -> dict[str, Any]:
    """Append a fresh collection event while retaining an existing valid history.

    A full v1 entry is promoted to the first v2 event. Incomplete v1/local
    entries cannot be promoted honestly and fail instead of being discarded.
    ``current.rows`` describes this event; ``merged_rows`` describes the
    accumulated parquet table as shipped.
    """
    if "collections" in existing:
        existing_history = _ordered_multi_event_table(table, existing)["collections"]
    else:
        existing_history = [_ordered_collection_event(existing, table=table, index=0)]
    current_event = _ordered_collection_event(current, table=table, index=len(existing_history))
    return {
        "rows": merged_rows,
        "status": current["status"],
        "collections": [*existing_history, current_event],
    }


def render_collection_meta(
    runtime_meta: dict[str, Any],
    tables: dict[str, dict[str, Any]],
    *,
    provenance_tier: str | None = None,
) -> bytes:
    """Render the exact design-§5 ``collection_meta.yaml`` bytes."""
    schema_version = (
        MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION
        if any("collections" in entry for entry in tables.values())
        else COLLECTION_META_SCHEMA_VERSION
    )
    if schema_version == MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION:
        rendered_tables = {table: _ordered_multi_event_table(table, tables[table]) for table in sorted(tables)}
    else:
        rendered_tables = {
            table: {key: tables[table][key] for key in _TABLE_FIELD_ORDER if key in tables[table]}
            for table in sorted(tables)
        }

    doc = {"schema_version": schema_version}
    if provenance_tier is not None:
        if not isinstance(provenance_tier, str) or not provenance_tier.strip():
            raise ValueError("provenance_tier must be a non-empty string when provided")
        doc["provenance"] = provenance_tier
    doc["runtime"] = (
        _ordered_runtime(runtime_meta, field="runtime")
        if schema_version == MULTI_EVENT_COLLECTION_META_SCHEMA_VERSION
        else {key: runtime_meta[key] for key in _RUNTIME_FIELD_ORDER if key in runtime_meta}
    )
    doc["tables"] = rendered_tables

    return (
        spdx_header()
        + yaml.safe_dump(
            doc,
            sort_keys=False,
            default_flow_style=False,
        )
    ).encode()


def write_collection_meta(
    out_dir: str | Path,
    runtime_meta: dict[str, Any],
    tables: dict[str, dict[str, Any]],
    *,
    provenance_tier: str | None = None,
) -> Path:
    """Write ``collection_meta.yaml`` per design §5, with deterministic key order."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    meta_path = out_path / "collection_meta.yaml"
    meta_path.write_bytes(
        render_collection_meta(
            runtime_meta,
            tables,
            provenance_tier=provenance_tier,
        )
    )
    return meta_path
