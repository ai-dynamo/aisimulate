# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate Dynamo-native iteration totals and publish the formal FPM database."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from aisimulate.fpm_contract import FPM_RESOLVED_CONFIG_GLOB
from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY

from .native_artifact import validate_native_collection
from .planner import FPMCell, FPMCollectionPlan, backend_identity_columns

logger = logging.getLogger(__name__)

_ROW_KEY = (
    "cell_id",
    "model_path",
    "system",
    "backend",
    "backend_version",
    "weight_quantization",
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "kv_cache_dtype",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
    "moe_backend",
    "attention_backend",
    "enable_wideep",
    "enable_eplb",
    *EXECUTION_COLUMNS,
    "workload_kind",
    "batch_size",
    "total_prefill_tokens",
    "total_kv_read_tokens",
    "partition_policy",
)
_RUN_IDENTITY_FIELDS = (
    "source_plan_sha256",
    "collector_attempt_id",
    "runtime_run_id",
    "runtime_grid_digest",
)


def _dotted_get(payload: object, path: str) -> object:
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _validate_backend_markers(cell: FPMCell, cell_dir: Path) -> None:
    expected = cell.backend_policy.expected_markers
    if not expected:
        return
    paths = sorted((cell_dir / "raw").glob(f"**/{FPM_RESOLVED_CONFIG_GLOB}"))
    if not paths:
        raise ValueError(f"backend policy {cell.backend_policy.policy_id} requires resolved-config evidence")
    for path in paths:
        payload = json.loads(path.read_text())
        mismatches = {}
        for marker_path, marker_value in expected.items():
            try:
                actual = _dotted_get(payload, marker_path)
            except KeyError:
                actual = "<missing>"
            # Markers are declared as strings while the resolved config keeps
            # native JSON types (enable_eplb: true vs expected "True");
            # compare canonical string forms so a type gap is not a mismatch.
            if str(actual) != str(marker_value):
                mismatches[marker_path] = {"actual": actual, "expected": marker_value}
        if mismatches:
            raise ValueError(f"backend marker mismatch in {path}: {mismatches}")


def aggregate_cell(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    cell_dir: Path,
    *,
    expected_attempt_id: str,
) -> list[dict[str, Any]]:
    """Validate native rank artifacts and take max-rank latency per grid point."""

    if not expected_attempt_id:
        raise ValueError(f"cannot aggregate {cell.cell_id} without an expected Collector attempt identity")
    _validate_backend_markers(cell, cell_dir)
    collection = validate_native_collection(
        cell,
        cell_dir / "raw",
        expected_plan_sha256=plan.sha256,
        expected_attempt_id=expected_attempt_id,
        expected_backend_version=plan.capability.aic_database_version if cell.state_protocol else None,
    )
    backend_version = collection.backend_version
    capability = plan.capability

    # The steady-state decode policy clamps every per-sequence context below
    # the measurable minimum up to it, so several requested plan points can
    # collapse onto one achieved physical coordinate (e.g. batch=3 with
    # requested total-kv 3/4/5/6 all measure total-kv 6). Repeated samples of
    # one coordinate would violate the database's unique-key contract, so keep
    # exactly one per coordinate: the native (unclamped) sample when present,
    # otherwise the clamped sample with the lowest benchmark_id. Two native
    # samples on one coordinate remain a hard error — the grid itself
    # guarantees native coordinates are unique.
    grouped: dict[tuple[str, int, int, int], list[Any]] = {}
    for measurement in collection.points:
        point = measurement.point
        key = (
            str(point["point_type"]),
            int(point["batch_size"]),
            int(point["total_prefill_tokens"]),
            int(point["total_kv_read_tokens"]),
        )
        grouped.setdefault(key, []).append(measurement)
    selected: list[Any] = []
    dropped_clamped = 0
    for key, measurements in grouped.items():
        natives = [m for m in measurements if "context_clamped" not in (m.point.get("sample_reasons") or ())]
        if len(natives) > 1:
            raise ValueError(
                f"conflicting FPM measurements for physical coordinate {key} in "
                f"{cell.cell_id}: {len(natives)} unclamped samples share one key"
            )
        if natives:
            selected.append(natives[0])
        else:
            selected.append(min(measurements, key=lambda m: int(m.point["benchmark_id"])))
        dropped_clamped += len(measurements) - 1
    if dropped_clamped:
        logger.info(
            "FPM %s: consolidated %d context-clamped duplicate sample(s) onto their achieved physical coordinates",
            cell.cell_id,
            dropped_clamped,
        )

    # KV seed regime (schema v6 additive column): a per-row RECORD of the
    # engine's measurement protocol for this point, consumed by the modeling
    # loader (which may exclude fake_fallback rows). The collector records
    # and never filters (layer_permissions: run it or raise).
    #
    # Only protocol-approved warm-ineligible reasons may override per-point
    # evidence. Other skips mean real KV was not established and must stay in
    # the fake_fallback bucket so consumers cannot mistake biased points for a
    # legitimate topology regime.
    kvwarm_meta = collection.kvwarm_meta
    approved_skip_reasons = {"moe_tp_balanced_by_construction"}

    def _kv_seed_regime(point: dict[str, Any], phase: str) -> str:
        if phase == "prefill" and (not cell.execution_identity[0] or int(point["total_kv_read_tokens"]) == 0):
            return "n/a"
        if kvwarm_meta is None:
            return "legacy"
        skip_reason = kvwarm_meta.get("skip_reason")
        if skip_reason in approved_skip_reasons:
            return f"skip:{skip_reason}"
        reasons = point.get("sample_reasons") or []
        if "kvwarm_real_kv" in reasons:
            if skip_reason:
                raise ValueError(f"native point reports real KV under warm-up skip_reason={skip_reason!r}")
            return "real_kv"
        if "kvwarm_fake_fallback" in reasons:
            return "fake_fallback"
        if skip_reason:
            return "fake_fallback"
        return "legacy"

    rows = []
    for measurement in selected:
        point = measurement.point
        phase = str(point["point_type"])
        batch = int(point["batch_size"])
        total_prefill = int(point["total_prefill_tokens"])
        total_kv = int(point["total_kv_read_tokens"])
        if (
            cell.execution_identity[0]
            and (phase == "decode" or total_kv > 0)
            and _kv_seed_regime(point, phase) != "real_kv"
        ):
            raise ValueError("DeepSeek-V4.1 cached prefill/decode publication requires real_kv measurements")
        rows.append(
            {
                "cell_id": cell.cell_id,
                "model_path": plan.model_path,
                "system": plan.system,
                "backend": plan.backend,
                "backend_version": backend_version,
                "weight_quantization": cell.weight_quantization,
                "gemm_quant_mode": cell.gemm_quant_mode,
                "moe_quant_mode": cell.moe_quant_mode,
                "fmha_quant_mode": cell.fmha_quant_mode,
                "fmha_resolution": cell.fmha_resolution,
                "comm_quant_mode": cell.comm_quant_mode,
                "kv_cache_dtype": cell.kv_cache_dtype,
                "parallel_strategy": cell.parallel_strategy,
                "tp": cell.topology.tp,
                "pp": cell.topology.pp,
                "dp": cell.topology.dp,
                "moe_tp": cell.topology.moe_tp,
                "moe_ep": cell.topology.moe_ep,
                "cp": cell.topology.cp,
                **backend_identity_columns(cell.backend_policy),
                **dict(zip(EXECUTION_COLUMNS, cell.execution_identity, strict=True)),
                "workload_kind": phase,
                "batch_size": batch,
                "total_prefill_tokens": total_prefill,
                "total_kv_read_tokens": total_kv,
                "partition_policy": "balanced_v1",
                "kv_seed_regime": _kv_seed_regime(point, phase),
                "latency_ms": max(latency for _rank, latency in measurement.rank_wall_times) * 1000.0,
                "global_warmup_iterations": 0 if cell.state_protocol else plan.options.warmup_iterations,
                "warmup_repeats": 5 if cell.state_protocol else 0,
                "measurement_repeats": 10 if cell.state_protocol else 1,
                "measurement_policy": (
                    f"{cell.backend}_native_real_hybrid_median_v1"
                    if cell.state_protocol
                    else "dynamo_native_single_sample_v1"
                ),
                **(
                    {
                        "state_protocol": cell.state_protocol,
                        "timing_boundary": "sglang_native_forward_device_timer"
                        if cell.backend == "sglang"
                        else "vllm_native_scheduler_output_interval",
                    }
                    if cell.state_protocol
                    else {}
                ),
                "model_support_level": capability.support_level,
                "model_template_id": capability.template_id,
                "model_template_version": capability.template_version,
                "aic_database_version": capability.aic_database_version,
                "source_plan_sha256": plan.sha256,
                "collector_attempt_id": collection.collector_attempt_id,
                "runtime_run_id": collection.runtime_run_id,
                "runtime_grid_digest": collection.runtime_grid_digest,
                "input_text_sha256": (collection.input_provenance or {}).get("text_sha256"),
                "input_token_ids_sha256": (collection.input_provenance or {}).get("token_ids_sha256"),
                "input_tokenizer_revision": (collection.input_provenance or {}).get("tokenizer_revision"),
            }
        )
    return rows


def _run_identities_by_cell(rows: list[dict[str, Any]], *, source: str) -> dict[str, tuple[str, ...]]:
    identities: dict[str, tuple[str, ...]] = {}
    for row in rows:
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(f"{source} FPM row has no cell identity")
        values = tuple(row.get(field) for field in _RUN_IDENTITY_FIELDS)
        if not all(isinstance(value, str) and value for value in values):
            invalid = [
                field
                for field, value in zip(_RUN_IDENTITY_FIELDS, values, strict=True)
                if not isinstance(value, str) or not value
            ]
            raise ValueError(f"{source} FPM row for {cell_id} has invalid run identity fields: {invalid}")
        typed_values = tuple(str(value) for value in values)
        existing = identities.setdefault(cell_id, typed_values)
        if existing != typed_values:
            raise ValueError(f"{source} FPM rows mix run identities for cell_id={cell_id!r}")
    return identities


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_formal_database_commit(
    parquet_path: Path,
    metadata_path: Path,
    plan: FPMCollectionPlan,
    *,
    expected_attempt_ids: dict[str, str] | None = None,
    reused_cell_ids: tuple[str, ...] = (),
    expected_cell_rows: dict[str, str] | None = None,
    required_cells: tuple[FPMCell, ...] | None = None,
) -> dict[str, Any]:
    """Validate the sealed table and requested cells, returning per-cell row hashes.

    A terminal checkpoint supplies its published row hashes and attempt IDs.
    First-publisher reuse may keep another plan/attempt only for explicitly
    recorded reused cells; their identity and coordinate coverage still apply.
    """

    if not parquet_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            "completed FPM database checkpoint references missing output: "
            f"parquet={parquet_path}, metadata={metadata_path}"
        )
    payload = json.loads(metadata_path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"FPM database commit record must be a mapping: {metadata_path}")
    expected = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 7,
        "system": plan.system,
        "backend": plan.backend,
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"FPM database commit record does not match the frozen plan: {mismatches}")
    if payload.get("parquet_sha256") != _sha256(parquet_path):
        raise ValueError(f"FPM database does not match its commit record: {parquet_path}")

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("validating fpm_forward_perf.parquet requires pyarrow") from error
    parquet = pq.ParquetFile(parquet_path)
    required = {*_ROW_KEY, *_RUN_IDENTITY_FIELDS}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"committed FPM database is missing schema-v7 columns: {missing}")
    row_count = payload.get("row_count")
    if not isinstance(row_count, int) or row_count < 1 or parquet.metadata.num_rows != row_count:
        raise ValueError(
            "FPM database row count does not match its commit record: "
            f"parquet={parquet.metadata.num_rows}, metadata={row_count!r}"
        )
    rows = parquet.read().to_pylist()
    identities = _run_identities_by_cell(rows, source="committed")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["cell_id"], []).append(row)
    cells = plan.cells if required_cells is None else required_cells
    if expected_cell_rows is not None and not isinstance(expected_cell_rows, dict):
        raise TypeError("FPM checkpoint cell row hashes must be a mapping")
    required_ids = {cell.cell_id for cell in cells}
    if set(reused_cell_ids) - required_ids:
        raise ValueError("FPM first-publisher reuse names cells outside the frozen plan")
    explicit_json = getattr(plan.options, "benchmark_points_json", None)
    explicit = json.loads(explicit_json) if explicit_json is not None else None
    cell_rows = {}
    for cell in cells:
        selected = grouped.get(cell.cell_id, [])
        if not selected:
            raise ValueError(f"committed FPM database does not cover required cell {cell.cell_id}")
        expected_identity = {
            "model_path": plan.model_path,
            "system": plan.system,
            "backend": plan.backend,
            "backend_version": payload.get("backend_version"),
            "weight_quantization": cell.weight_quantization,
            "gemm_quant_mode": cell.gemm_quant_mode,
            "moe_quant_mode": cell.moe_quant_mode,
            "fmha_quant_mode": cell.fmha_quant_mode,
            "comm_quant_mode": cell.comm_quant_mode,
            "fmha_resolution": cell.fmha_resolution,
            "kv_cache_dtype": cell.kv_cache_dtype,
            "parallel_strategy": cell.parallel_strategy,
            "workload_kind": cell.workload_kind,
            "partition_policy": "balanced_v1",
            **{name: getattr(cell.topology, name) for name in ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")},
            **backend_identity_columns(cell.backend_policy),
            **dict(zip(EXECUTION_COLUMNS, cell.execution_identity, strict=True)),
        }
        if cell.input_text_sha256:
            expected_identity["input_text_sha256"] = cell.input_text_sha256
        for row in selected:
            if any(row.get(key) != value for key, value in expected_identity.items()):
                raise ValueError(f"committed FPM cell identity differs from frozen plan: {cell.cell_id}")
            latency = row.get("latency_ms")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, int | float)
                or not math.isfinite(latency)
                or latency <= 0
            ):
                raise ValueError(f"committed FPM cell has an invalid latency: {cell.cell_id}")
        if cell.cell_id not in reused_cell_ids:
            if identities[cell.cell_id][0] != plan.sha256:
                raise ValueError(f"committed FPM cell has a different source plan: {cell.cell_id}")
            if expected_attempt_ids is not None and identities[cell.cell_id][1] != expected_attempt_ids.get(
                cell.cell_id
            ):
                raise ValueError(f"committed FPM cell has a different Collector attempt: {cell.cell_id}")
        coordinates = {
            (row["batch_size"], row["total_prefill_tokens"], row["total_kv_read_tokens"]) for row in selected
        }
        if len(coordinates) != len(selected):
            raise ValueError(f"committed FPM cell contains duplicate coordinates: {cell.cell_id}")
        # New checkpoints bind the actually achieved native coordinates,
        # including legal context clamping. Legacy explicit checkpoints and
        # first-publisher reuse must prove requested coverage from the table.
        if explicit is not None and (
            cell.cell_id in reused_cell_ids or (expected_attempt_ids is not None and expected_cell_rows is None)
        ):
            requested = {
                (point["batch_size"], point.get("total_prefill_tokens", 0), point["total_kv_read_tokens"])
                for point in explicit[cell.workload_kind]
            }
            if not requested <= coordinates:
                raise ValueError(f"committed FPM cell does not cover requested coordinates: {cell.cell_id}")
        canonical_rows = sorted(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) for row in selected
        )
        digest = hashlib.sha256("\n".join(canonical_rows).encode()).hexdigest()
        if expected_cell_rows is not None and expected_cell_rows.get(cell.cell_id) != digest:
            raise ValueError(f"committed FPM cell differs from its published checkpoint rows: {cell.cell_id}")
        if expected_attempt_ids is not None and expected_cell_rows is None and explicit is None:
            raise ValueError("terminal FPM checkpoint has no sealed cell coverage; raw artifacts must be revalidated")
        cell_rows[cell.cell_id] = digest
    return {**payload, "cell_rows": cell_rows}


def _curated_systems_root() -> Path:
    """The data tree the SDK's default systems path actually reads.

    Resolved through the same package the SDK's ``--systems-paths default``
    resolves to (``aisimulate_core``), so default publications land where
    the fpm forward-model consumer looks regardless of how the package is
    installed. In a repo checkout this is the aic-core tree (which
    src/aisimulate_core/systems symlinks to); in an installed environment it is
    the site-packages tree the SDK reads.
    """
    from importlib import resources

    return Path(os.fspath(resources.files("aisimulate_core") / "systems" / "data"))


# The SDK's reuse/partial markers declare a version without holding measured
# data; a version dir counts as measured evidence only when it holds at least
# one real perf file beyond these (mirrors perf_database's marker set).
_CURATED_VERSION_MARKER_FILES = frozenset(
    {"reuse.yaml", "SHARED_LAYER_REUSE.txt", "collection_meta.yaml", "INCOMPLETE.txt"}
)


def _family_version_dir_is_partial(version_dir: Path) -> bool:
    """Mirror the SDK's partial veto (_database_version_dir_is_declared): a
    version dir mid-collection — any table with status partial, or the legacy
    INCOMPLETE marker — is undeclared to version discovery no matter how many
    perf files it holds, so it must not count as curated evidence either. An
    unreadable collection_meta.yaml fails closed: declaredness cannot be
    proven from it."""
    if (version_dir / "INCOMPLETE.txt").is_file():
        return True
    meta_path = version_dir / "collection_meta.yaml"
    if not meta_path.is_file():
        return False
    try:
        meta = yaml.safe_load(meta_path.read_text())
    except Exception:
        return True
    tables = meta.get("tables") if isinstance(meta, dict) else None
    if not isinstance(tables, dict):
        return False
    return any(isinstance(table, dict) and table.get("status") == "partial" for table in tables.values())


def _version_has_curated_measurements(systems_root: Path, system: str, backend: str, version: str) -> bool:
    """True when an existing curated family dir holds measured data for version.

    The SDK's version discovery treats ANY populated version directory under
    the curated tree as a declared database version, so materializing
    <system>/<backend>/<version> for a version nothing else declares would
    make get_latest_database_version return a dataless version and poison
    every later default-version resolution for this system. Evidence therefore
    mirrors the SDK's own declaredness: dot-prefixed dirs are invisible to
    discovery, marker files declare without measuring, and partial
    (mid-collection) dirs are vetoed outright — so marker-only and
    partial-only versions stay excluded from default-version resolution
    exactly as they were before an FPM publication.
    """
    system_dir = systems_root / system
    if not system_dir.is_dir():
        return False
    for family_dir in system_dir.iterdir():
        if family_dir.name.startswith("."):
            continue
        candidate = family_dir / backend / version
        if not candidate.is_dir() or _family_version_dir_is_partial(candidate):
            continue
        for entry in candidate.iterdir():
            if entry.is_file() and not entry.name.startswith(".") and entry.name not in _CURATED_VERSION_MARKER_FILES:
                return True
    return False


@contextmanager
def _publication_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _temporary_path(destination: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    return Path(raw_path)


def write_formal_database(
    plan: FPMCollectionPlan,
    rows: list[dict[str, Any]],
    *,
    systems_root: Path | None = None,
    reject_replaced_cells: bool = False,
) -> tuple[Path, Path, tuple[str, ...]]:
    """Atomically merge conflict-free native-grid rows into the AIC data tree.

    Returns ``(parquet_path, metadata_path, skipped_cells)``.
    ``skipped_cells`` implements first-publisher-wins: cells whose cell_id is
    already published under a different run identity are dropped from this
    publication (loudly) instead of failing the whole merge - published rows
    are sha-sealed evidence and are never overwritten or mixed, while the
    incoming plan's non-overlapping cells still land.
    """

    if not rows:
        raise ValueError("refusing to write an empty FPM database")
    incoming_identities = _run_identities_by_cell(rows, source="incoming")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("writing fpm_forward_perf.parquet requires pyarrow") from error

    versions = {str(row.get("backend_version") or "") for row in rows}
    if len(versions) != 1 or not next(iter(versions)):
        raise ValueError(f"FPM rows must contain one non-empty runtime backend_version, got {sorted(versions)!r}")
    version = next(iter(versions))
    # backend_version comes from pod-reported provenance: reject anything that
    # is not a plain version token before it becomes a path component.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version):
        raise ValueError(f"pod-reported backend_version {version!r} is not a safe database directory name")
    curated_root = systems_root is None
    if systems_root is None:
        systems_root = _curated_systems_root()
    # The FPM parquet's agreed consumer location is <system>/<backend>/<version>
    # (the fpm forward-model loader reads exactly this path); the guard below
    # keeps that from introducing versions the curated tree does not measure.
    destination = systems_root / plan.system / plan.backend / version
    if curated_root and not _version_has_curated_measurements(systems_root, plan.system, plan.backend, version):
        raise ValueError(
            f"pod-reported backend_version {version!r} is not a curated AIC database version for "
            f"{plan.system}/{plan.backend}: no <family>/{plan.backend}/{version} directory with "
            f"measured data exists under {systems_root / plan.system}. Publish against a curated "
            "version or pass --fpm-database-root to write into an explicit tree"
        )
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        if not curated_root:
            raise
        raise ValueError(
            f"cannot create the default curated FPM publication directory {destination}: {error}. "
            "The tree the SDK reads by default is not writable here; pass --fpm-database-root to "
            "publish into an explicit tree"
        ) from error
    parquet_path = destination / "fpm_forward_perf.parquet"
    metadata_path = destination / "fpm_forward_perf.metadata.json"

    lock_path = destination / ".fpm_forward_perf.lock"
    with _publication_lock(lock_path):
        merged = []
        if parquet_path.exists():
            # The metadata sidecar is the commit record of the last successful
            # publication: refuse to merge onto a parquet whose bytes it does
            # not vouch for (partial write, manual edit, or a foreign file).
            if not metadata_path.exists():
                raise ValueError(f"existing FPM database has no commit record; refusing to merge: {parquet_path}")
            committed = json.loads(metadata_path.read_text())
            if committed.get("parquet_sha256") != _sha256(parquet_path):
                raise ValueError(f"existing FPM database does not match its commit record: {parquet_path}")
            table = pq.read_table(parquet_path)
            # The merge below indexes every _ROW_KEY and run-identity column,
            # so the gate must require exactly that set: a narrower hand-picked
            # list turns schema drift into a bare KeyError instead of this
            # actionable error, and silently rots when _ROW_KEY grows.
            required = {*_ROW_KEY, *_RUN_IDENTITY_FIELDS}
            if committed.get("schema_version") == 6:
                required -= set(EXECUTION_COLUMNS)
            elif committed.get("schema_version") != 7:
                raise ValueError("existing FPM database has unsupported schema version")
            missing = sorted(required - set(table.column_names))
            if missing:
                raise ValueError(
                    "existing FPM database does not satisfy the attempt-bound schema-v7 row-key "
                    f"contract (missing columns: {missing}); publish to a clean destination: {parquet_path}"
                )
            existing_rows = table.to_pylist()
            if committed.get("schema_version") == 6:
                for row in existing_rows:
                    row.update(zip(EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY, strict=True))
            merged.extend(existing_rows)
        existing_versions = {str(row.get("backend_version") or "") for row in merged}
        if existing_versions and existing_versions != {version}:
            raise ValueError(
                f"existing FPM database runtime version mismatch: actual={sorted(existing_versions)!r} "
                f"expected={version!r}"
            )
        existing_identities = _run_identities_by_cell(merged, source="existing")
        skipped_cells: list[str] = []
        for cell_id, incoming_identity in sorted(incoming_identities.items()):
            existing_identity = existing_identities.get(cell_id)
            if existing_identity is not None and existing_identity != incoming_identity:
                skipped_cells.append(cell_id)
                logger.warning(
                    "fpm_forward: cell_id=%s is already published under a different run identity "
                    "(existing=%s, incoming=%s); first publisher wins - skipping this cell's rows",
                    cell_id,
                    existing_identity,
                    incoming_identity,
                )
        if skipped_cells:
            if reject_replaced_cells:
                raise ValueError(f"complete shard union conflicts with published child attempts: {skipped_cells}")
            skipped_set = set(skipped_cells)
            rows = [row for row in rows if row.get("cell_id") not in skipped_set]
            if not rows:
                logger.warning(
                    "fpm_forward: every incoming cell is already published under a different run "
                    "identity; database left unchanged"
                )
                return parquet_path, metadata_path, tuple(skipped_cells)
        index = {tuple(row[key] for key in _ROW_KEY): row for row in merged}
        if len(index) != len(merged):
            raise ValueError(f"existing FPM database contains duplicate physical keys: {parquet_path}")
        for row in rows:
            key = tuple(row[name] for name in _ROW_KEY)
            existing = index.get(key)
            if existing is not None:
                if existing != row:
                    raise ValueError(f"conflicting FPM database row for key={key}")
                continue
            index[key] = row
            merged.append(row)
        merged.sort(key=lambda row: tuple(row[name] for name in _ROW_KEY))

        # Additive schema evolution: rows merged from a parquet written before
        # kv_seed_regime existed lack the key, and pyarrow's from_pylist
        # derives its schema from the first rows -- without normalization the
        # new column silently vanishes from the merged file. Old rows publish
        # as null instead.
        for row in merged:
            row.setdefault("kv_seed_regime", None)
            for field in ("input_text_sha256", "input_token_ids_sha256", "input_tokenizer_revision"):
                row.setdefault(field, None)
        if any(row.get("state_protocol") for row in merged):
            for row in merged:
                row.setdefault("state_protocol", None)
                row.setdefault("timing_boundary", None)

        temporary = _temporary_path(parquet_path)
        temporary_metadata = _temporary_path(metadata_path)
        try:
            pq.write_table(pa.Table.from_pylist(merged), temporary, compression="zstd")
            metadata = {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 7,
                "coordinate_system": "iteration_totals_balanced_v1",
                "measurement_policy": (
                    next(iter(policies))
                    if len(policies := {row["measurement_policy"] for row in merged}) == 1
                    else "per_row"
                ),
                "warmup_repeats": (
                    next(iter(warmups)) if len(warmups := {row["warmup_repeats"] for row in merged}) == 1 else None
                ),
                "measurement_repeats": (
                    next(iter(repeats)) if len(repeats := {row["measurement_repeats"] for row in merged}) == 1 else None
                ),
                "row_count": len(merged),
                "parquet_sha256": _sha256(temporary),
                "source_plan_sha256": sorted({str(row["source_plan_sha256"]) for row in merged}),
                "collector_attempt_ids": sorted({str(row["collector_attempt_id"]) for row in merged}),
                "runtime_run_ids": sorted({str(row["runtime_run_id"]) for row in merged}),
                "runtime_grid_digests": sorted({str(row["runtime_grid_digest"]) for row in merged}),
                "aic_revision": plan.aic_revision,
                "model_paths": sorted({str(row["model_path"]) for row in merged}),
                "system": plan.system,
                "backend": plan.backend,
                "backend_version": version,
            }
            temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, parquet_path)
            # Metadata is the commit record: readers must validate its parquet
            # digest and ignore an unmatched pair after an interrupted writer.
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
    return parquet_path, metadata_path, tuple(skipped_cells)
