# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-0 coverage and accuracy catalog: one summary row per model x hardware x framework x layer.

The catalog answers "what do we have, what is missing or stale, how accurate was
the last prediction, how good was the collection and what did it cost" without
touching any payload. It is Parquet (zstd), sharded by category and hardware so
an agent working on one GPU reads one small file per layer::

    catalog/op-latency/<hardware>.parquet    schema aisim-catalog/v1
    catalog/fpm/<hardware>.parquet
    catalog/e2e/<hardware>.parquet
    catalog/systems.parquet                  schema aisim-catalog-systems/v1

:func:`build_catalog` is the only code that reads manifests and eval records.
:class:`Catalog` (the query API) and :func:`check_catalog_size` read only the
files above. Output is deterministic (no timestamps), so a rebuild of unchanged
inputs is byte-for-byte stable up to Parquet writer determinism.

Cell status:

* ``have``    data (manifest) or eval evidence exists and its latest accuracy
              record, if any, was made against the current system spec.
* ``stale``   the latest accuracy record used a different ``spec_sha256`` than
              the system's current packaged spec (calibration changed since).
* ``missing`` the cell is expected but has no data. Expected cells are every
              model x hardware x framework seen anywhere in the dataset, crossed
              with all known systems, for the fpm and e2e layers; the op layer is
              model-independent (``model`` is null) and expected per
              hardware x framework. Callers can widen the universe through
              ``expected_models`` / ``expected_frameworks``.

``stale`` is deliberately narrow and offline-decidable. Unknown stays null:
a cell with no recorded GPU time has ``gpu_seconds`` null, never zero.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from aisim_core.data import CategoryIndex, DatasetIndex, Manifest
from aisim_core.data.eval import EvalRecord, Layer, current_records
from aisim_core.data.parquet import read_parquet, write_parquet
from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR, packaged_system_specs, spec_file_sha256

CATALOG_SCHEMA_ID = "aisim-catalog/v1"
SYSTEMS_SCHEMA_ID = "aisim-catalog-systems/v1"
CATALOG_DIR = "catalog"
SYSTEMS_FILE = "systems.parquet"

LAYERS: tuple[Layer, ...] = ("op", "fpm", "e2e")
LAYER_CATEGORY: dict[str, str] = {"op": "op-latency", "fpm": "fpm", "e2e": "e2e"}
CATEGORY_LAYER = {v: k for k, v in LAYER_CATEGORY.items()}

Status = Literal["have", "missing", "stale"]

# Budgets proposed by AISIM-2315 (soft: warn; hard: plan refused without opt-in).
MIB = 1024 * 1024
KIB = 1024


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class CatalogRow(_Row):
    """One cell summary. ``model`` is null for the model-independent op layer."""

    layer: Layer
    model: str | None
    hardware: str
    framework: str
    status: Status
    versions: str = ""  # comma-joined framework versions with data, sorted
    n_configs: int = Field(ge=0)  # manifests in the category index
    n_evidence_gaps: int = Field(ge=0)  # summed manifest evidence gaps
    n_eval_records: int = Field(ge=0)  # current (non-superseded) eval records
    acc_mape_pct: float | None = None  # last accuracy; null = never evaluated
    acc_coverage: float | None = None
    acc_n_scored: int | None = Field(default=None, ge=0)
    acc_tier: str | None = None
    acc_record_id: str | None = None
    acc_created_at: str | None = None
    acc_spec_sha256: str | None = None
    collection_quality_json: str | None = None  # latest record's collection_quality, canonical JSON
    gpu_seconds: float | None = Field(default=None, ge=0)  # sum over recorded gpu_time; null if none recorded
    gpu_n_recorded: int = Field(default=0, ge=0)  # records contributing to gpu_seconds
    spec_sha256_current: str | None = None  # current packaged spec hash for this hardware
    stale_reason: str | None = None


class SystemRow(_Row):
    """A known system and the spec hash current in this checkout."""

    system: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


_ROW_FIELDS = {
    "layer": pa.string(),
    "model": pa.string(),
    "hardware": pa.string(),
    "framework": pa.string(),
    "status": pa.string(),
    "versions": pa.string(),
    "n_configs": pa.int32(),
    "n_evidence_gaps": pa.int32(),
    "n_eval_records": pa.int32(),
    "acc_mape_pct": pa.float64(),
    "acc_coverage": pa.float64(),
    "acc_n_scored": pa.int32(),
    "acc_tier": pa.string(),
    "acc_record_id": pa.string(),
    "acc_created_at": pa.string(),
    "acc_spec_sha256": pa.string(),
    "collection_quality_json": pa.string(),
    "gpu_seconds": pa.float64(),
    "gpu_n_recorded": pa.int32(),
    "spec_sha256_current": pa.string(),
    "stale_reason": pa.string(),
}
assert list(_ROW_FIELDS) == list(CatalogRow.model_fields)


def _table(rows: list[CatalogRow]) -> pa.Table:
    return pa.table(
        {name: pa.array([getattr(r, name) for r in rows], typ) for name, typ in _ROW_FIELDS.items()},
    )


def _row_sort_key(row: CatalogRow) -> tuple:
    return (row.layer, row.model or "", row.framework)


# --- paths --------------------------------------------------------------------


def shard_path(root: str | Path, layer: str, hardware: str) -> Path:
    return Path(root) / CATALOG_DIR / LAYER_CATEGORY[layer] / f"{hardware}.parquet"


def catalog_files(root: str | Path) -> list[Path]:
    """Every catalog Parquet file under ``root`` (shards and the systems table)."""
    base = Path(root) / CATALOG_DIR
    files = [base / SYSTEMS_FILE] if (base / SYSTEMS_FILE).is_file() else []
    for category in LAYER_CATEGORY.values():
        files.extend(sorted((base / category).glob("*.parquet")))
    return files


# --- builder ------------------------------------------------------------------


@dataclass
class _Cell:
    versions: set[str] = field(default_factory=set)
    n_configs: int = 0
    n_gaps: int = 0
    records: list[EvalRecord] = field(default_factory=list)


def _read_indexed_cells(root: Path, cells: dict[tuple, _Cell]) -> None:
    """Fold manifests listed by the dataset index into ``cells`` (identity only, no payloads)."""
    index_path = root / CATALOG_DIR / "dataset.json"
    if not index_path.is_file():
        return
    index = DatasetIndex.model_validate_json(index_path.read_bytes())
    for category, ref in index.categories.items():
        layer = CATEGORY_LAYER.get(category)
        if layer is None:
            continue
        entries = CategoryIndex.model_validate_json((root / ref.index.path).read_bytes())
        for entry in entries.entries:
            ident = entry.identity
            manifest = Manifest.model_validate_json((root / entry.manifest.path).read_bytes())
            cell = cells[(layer, None if layer == "op" else ident.model, ident.hardware, ident.framework)]
            cell.versions.add(ident.framework_version)
            cell.n_configs += 1
            cell.n_gaps += len(manifest.evidence_gaps)


def _read_eval_cells(root: Path, cells: dict[tuple, _Cell]) -> None:
    eval_root = root / "eval"
    if not eval_root.is_dir():
        return
    directories = sorted({p.parent for p in eval_root.rglob("*.json")})
    for directory in directories:
        for record in current_records(directory):
            ident = record.truth.identity
            layer = record.truth.layer
            cell = cells[(layer, None if layer == "op" else ident.model, ident.hardware, ident.framework)]
            cell.versions.add(ident.framework_version)
            cell.records.append(record)


def _latest_accuracy(records: list[EvalRecord]) -> EvalRecord | None:
    scored = [r for r in records if r.results.mape_pct is not None and r.status != "failed"]
    return max(scored, key=lambda r: (r.created_at, r.record_id), default=None)


def _summarize(key: tuple, cell: _Cell | None, current_specs: Mapping[str, str]) -> CatalogRow:
    layer, model, hardware, framework = key
    cell = cell or _Cell()
    current = current_specs.get(hardware)
    latest = _latest_accuracy(cell.records)
    have = cell.n_configs > 0 or bool(cell.records)
    status: Status = "have" if have else "missing"
    reason = None
    if latest is not None and current is not None and latest.prediction.spec_sha256 != current:
        status, reason = "stale", "latest accuracy record used a different system spec hash"
    recorded = [r.gpu_time.gpu_seconds for r in cell.records if r.gpu_time and r.gpu_time.gpu_seconds is not None]
    quality = latest.results.collection_quality if latest else None
    return CatalogRow(
        layer=layer,
        model=model,
        hardware=hardware,
        framework=framework,
        status=status,
        versions=",".join(sorted(cell.versions)),
        n_configs=cell.n_configs,
        n_evidence_gaps=cell.n_gaps,
        n_eval_records=len(cell.records),
        acc_mape_pct=latest.results.mape_pct if latest else None,
        acc_coverage=latest.results.coverage if latest else None,
        acc_n_scored=latest.results.n_scored if latest else None,
        acc_tier=latest.tier if latest else None,
        acc_record_id=latest.record_id if latest else None,
        acc_created_at=latest.created_at if latest else None,
        acc_spec_sha256=latest.prediction.spec_sha256 if latest else None,
        collection_quality_json=(
            None if quality is None else json.dumps(quality, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ),
        gpu_seconds=sum(recorded) if recorded else None,
        gpu_n_recorded=len(recorded),
        spec_sha256_current=current,
        stale_reason=reason,
    )


def known_system_specs(systems_dir: str | Path = PACKAGED_SYSTEMS_DIR) -> dict[str, str]:
    """System id -> current spec hash for every packaged system spec."""
    return {name: spec_file_sha256(path) for name, path in packaged_system_specs(Path(systems_dir)).items()}


def build_catalog(
    dataset_root: str | Path,
    *,
    systems_dir: str | Path = PACKAGED_SYSTEMS_DIR,
    expected_models: Iterable[str] = (),
    expected_frameworks: Iterable[str] = (),
    write: bool = True,
) -> list[CatalogRow]:
    """Build the catalog from a dataset root's category indexes, manifests and eval records.

    Returns every row (sorted by hardware, layer, model, framework). With
    ``write=True`` (default) it replaces the shards under ``<root>/catalog/``
    and removes stale shards it no longer produces.
    """
    root = Path(dataset_root)
    specs = known_system_specs(systems_dir)
    cells: dict[tuple, _Cell] = defaultdict(_Cell)
    _read_indexed_cells(root, cells)
    _read_eval_cells(root, cells)

    models = {k[1] for k in cells if k[1] is not None} | set(expected_models)
    frameworks = {k[3] for k in cells} | set(expected_frameworks)
    hardware = {k[2] for k in cells} | set(specs)
    keys = set(cells)
    for hw in hardware:
        for fw in frameworks:
            keys.add(("op", None, hw, fw))
            for model in models:
                keys.update((layer, model, hw, fw) for layer in ("fpm", "e2e"))

    rows = [_summarize(k, cells.get(k), specs) for k in keys]
    rows.sort(key=lambda r: (r.hardware, *_row_sort_key(r)))
    if write:
        _write(root, rows, specs)
    return rows


def _write(root: Path, rows: list[CatalogRow], specs: Mapping[str, str]) -> None:
    grouped: dict[tuple[str, str], list[CatalogRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.layer, row.hardware)].append(row)
    keep = set()
    for (layer, hw), shard_rows in grouped.items():
        keep.add(write_parquet(_table(shard_rows), shard_path(root, layer, hw), CATALOG_SCHEMA_ID))
    systems = [SystemRow(system=s, spec_sha256=h) for s, h in sorted(specs.items())]
    systems_table = pa.table(
        {
            "system": pa.array([s.system for s in systems], pa.string()),
            "spec_sha256": pa.array([s.spec_sha256 for s in systems], pa.string()),
        }
    )
    keep.add(write_parquet(systems_table, root / CATALOG_DIR / SYSTEMS_FILE, SYSTEMS_SCHEMA_ID))
    for stale in set(catalog_files(root)) - keep:
        stale.unlink()


# --- query API (reads catalog files only) --------------------------------------


class Catalog:
    """Read-only view over a dataset's ``catalog/`` directory.

    Never opens anything outside ``<root>/catalog/*.parquet``, so it works on a
    catalog-only download. A query naming a hardware reads only that
    hardware's shards.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / CATALOG_DIR).is_dir():
            raise FileNotFoundError(f"no catalog directory under {self.root}")

    def _shards(self, hardware: str | None, layer: str | None) -> list[Path]:
        layers = [layer] if layer else list(LAYERS)
        out = []
        for lay in layers:
            directory = self.root / CATALOG_DIR / LAYER_CATEGORY[lay]
            if hardware is not None:
                path = directory / f"{hardware}.parquet"
                out.extend([path] if path.is_file() else [])
            else:
                out.extend(sorted(directory.glob("*.parquet")))
        return out

    def rows(
        self,
        *,
        hardware: str | None = None,
        model: str | None = None,
        framework: str | None = None,
        layer: str | None = None,
        status: Status | None = None,
    ) -> list[CatalogRow]:
        out = []
        for path in self._shards(hardware, layer):
            for item in read_parquet(path, CATALOG_SCHEMA_ID).to_pylist():
                row = CatalogRow(**item)
                if (
                    (model is None or row.model == model)
                    and (framework is None or row.framework == framework)
                    and (status is None or row.status == status)
                ):
                    out.append(row)
        return sorted(out, key=lambda r: (r.hardware, *_row_sort_key(r)))

    def worst_cells(
        self, hardware: str, *, n: int = 10, layer: str | None = None, model: str | None = None
    ) -> list[CatalogRow]:
        """Cells with a recorded accuracy, highest MAPE first (null MAPE never listed)."""
        scored = [r for r in self.rows(hardware=hardware, layer=layer, model=model) if r.acc_mape_pct is not None]
        return sorted(scored, key=lambda r: (-r.acc_mape_pct, _row_sort_key(r)))[:n]

    def empty_cells(self, hardware: str, *, layer: str | None = None, model: str | None = None) -> list[CatalogRow]:
        """Expected cells with no data on ``hardware``."""
        return self.rows(hardware=hardware, layer=layer, model=model, status="missing")

    def stale_cells(self, hardware: str | None = None, *, layer: str | None = None) -> list[CatalogRow]:
        return self.rows(hardware=hardware, layer=layer, status="stale")

    def unevaluated_cells(self, hardware: str, *, layer: str | None = None) -> list[CatalogRow]:
        """Cells with data but no accuracy record yet."""
        return [r for r in self.rows(hardware=hardware, layer=layer) if r.status == "have" and r.acc_mape_pct is None]

    def systems(self) -> dict[str, str]:
        """Known system -> current spec hash."""
        path = self.root / CATALOG_DIR / SYSTEMS_FILE
        if not path.is_file():
            return {}
        table = read_parquet(path, SYSTEMS_SCHEMA_ID)
        return dict(zip(table["system"].to_pylist(), table["spec_sha256"].to_pylist(), strict=True))

    def hardware(self) -> list[str]:
        names = {p.stem for layer in LAYERS for p in self._shards(None, layer)}
        return sorted(names)


# --- size budget ----------------------------------------------------------------


@dataclass(frozen=True)
class CatalogBudget:
    """Configurable limits in bytes. Defaults are the AISIM-2315 proposal."""

    soft_total: int = 1 * MIB
    hard_total: int = 2 * MIB
    soft_shard: int = 256 * KIB


@dataclass(frozen=True)
class SizeReport:
    total_bytes: int
    shard_bytes: dict[str, int]
    findings: tuple[tuple[Literal["error", "warn"], str], ...]

    @property
    def within_budget(self) -> bool:
        return not self.findings

    @property
    def refused(self) -> bool:
        """True when the hard limit is exceeded (a plan would need explicit opt-in)."""
        return any(sev == "error" for sev, _ in self.findings)


def check_catalog_size(root: str | Path, budget: CatalogBudget | None = None) -> SizeReport:
    """Compare catalog bytes (Parquet shards only) to ``budget``; read-only, stat calls only."""
    budget = budget or CatalogBudget()
    root = Path(root)
    sizes = {str(p.relative_to(root)): p.stat().st_size for p in catalog_files(root)}
    total = sum(sizes.values())
    findings: list[tuple[Literal["error", "warn"], str]] = []
    if total > budget.hard_total:
        findings.append(("error", f"catalog is {total} bytes, over the hard limit {budget.hard_total}"))
    elif total > budget.soft_total:
        findings.append(("warn", f"catalog is {total} bytes, over the soft limit {budget.soft_total}"))
    for name, size in sorted(sizes.items()):
        if size > budget.soft_shard:
            findings.append(("warn", f"{name} is {size} bytes, over the per-shard limit {budget.soft_shard}"))
    return SizeReport(total_bytes=total, shard_bytes=sizes, findings=tuple(findings))
