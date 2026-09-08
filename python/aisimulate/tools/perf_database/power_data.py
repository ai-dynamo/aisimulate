# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate paired power metrics and their checked-in provenance manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

POWER_COLUMNS = ("power", "power_limit")
MEASUREMENT_COLUMNS = ("latency", *POWER_COLUMNS)
EXPECTED_UNITS = {"latency": "ms", "power": "W", "power_limit": "W"}
MAX_POWER_LIMIT_RATIO = 1.05
MANIFEST_BASENAME = "power_data_provenance.json"


def power_metric_issues(table: pa.Table) -> list[str]:
    """Return violations of the committed power-data contract.

    A table may omit power entirely. Once either optional metric is present,
    both columns must be float64 and every row must be either a measured
    positive pair or the typed ``0.0``/``0.0`` unavailable sentinel.
    """
    present = [name for name in POWER_COLUMNS if name in table.column_names]
    if not present:
        return []
    if len(present) != len(POWER_COLUMNS):
        return [f"power and power_limit must be present together (found: {', '.join(present)})"]

    issues: list[str] = []
    columns: dict[str, list[float | None]] = {}
    valid_for_pair_checks = True
    for name in POWER_COLUMNS:
        field = table.schema.field(name)
        column = table.column(name)
        if not pa.types.is_float64(field.type):
            issues.append(f"{name} must be double, found {field.type}")
            valid_for_pair_checks = False
            continue
        values = column.to_pylist()
        columns[name] = values
        if column.null_count:
            issues.append(f"{name} contains {column.null_count} null cells")
            valid_for_pair_checks = False
        invalid_count = sum(value is not None and (not math.isfinite(value) or value < 0) for value in values)
        if invalid_count:
            issues.append(f"{name} contains {invalid_count} non-finite or negative values")
            valid_for_pair_checks = False

    if not valid_for_pair_checks:
        return issues

    pairs = zip(columns["power"], columns["power_limit"], strict=True)
    invalid_pairs = 0
    over_limit = 0
    for power, power_limit in pairs:
        assert power is not None and power_limit is not None
        sentinel = power == 0.0 and power_limit == 0.0
        measured = power > 0.0 and power_limit > 0.0
        if not sentinel and not measured:
            invalid_pairs += 1
        elif measured and power > MAX_POWER_LIMIT_RATIO * power_limit:
            over_limit += 1
    if invalid_pairs:
        issues.append(f"power/power_limit contains {invalid_pairs} rows that are neither positive pairs nor 0.0 pairs")
    if over_limit:
        issues.append(f"power exceeds {MAX_POWER_LIMIT_RATIO:.2f}x power_limit in {over_limit} rows")
    return issues


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(entry: dict[str, Any], name: str, *, context: str, issues: list[str]) -> int | None:
    value = entry.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        issues.append(f"{context}: {name} must be a non-negative integer")
        return None
    return value


def table_identity_evidence(table: pa.Table) -> tuple[list[str], dict[str, dict[str, int | float]]]:
    """Return ordered identity columns and numeric shape bounds for one table."""
    identity_columns = [name for name in table.column_names if name not in MEASUREMENT_COLUMNS]
    shape_bounds: dict[str, dict[str, int | float]] = {}
    for name in identity_columns:
        field_type = table.schema.field(name).type
        if not (pa.types.is_integer(field_type) or pa.types.is_floating(field_type)):
            continue
        extrema = pc.min_max(table.column(name)).as_py()
        if extrema["min"] is not None and extrema["max"] is not None:
            shape_bounds[name] = {"min": extrema["min"], "max": extrema["max"]}
    return identity_columns, shape_bounds


def validate_manifest(manifest_path: Path) -> list[str]:
    """Validate one adjacent power provenance manifest and its parquet files."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{manifest_path}: cannot read manifest: {exc}"]

    issues: list[str] = []
    if manifest.get("schema_version") != 1:
        issues.append(f"{manifest_path}: schema_version must be 1")
    source = manifest.get("source")
    if not isinstance(source, dict):
        issues.append(f"{manifest_path}: source must be an object")
    else:
        commit = source.get("commit")
        if not isinstance(commit, str) or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            issues.append(f"{manifest_path}: source.commit must be a lowercase 40-character Git SHA")
        if source.get("license") != "Apache-2.0":
            issues.append(f"{manifest_path}: source.license must identify Apache-2.0")

    root = manifest_path.parent.resolve()
    dataset = manifest.get("dataset")
    backend: str | None = None
    version: str | None = None
    if not isinstance(dataset, dict):
        issues.append(f"{manifest_path}: dataset must be an object")
    else:
        if dataset.get("system") != root.name:
            issues.append(f"{manifest_path}: dataset.system must match the manifest directory")
        for name in ("backend", "version"):
            value = dataset.get(name)
            if not isinstance(value, str) or not value:
                issues.append(f"{manifest_path}: dataset.{name} must be a non-empty string")
            elif name == "backend":
                backend = value
            else:
                version = value
        if dataset.get("units") != EXPECTED_UNITS:
            issues.append(f"{manifest_path}: dataset.units must be {EXPECTED_UNITS}")
        if dataset.get("unavailable_sentinel") != {"power": 0.0, "power_limit": 0.0}:
            issues.append(f"{manifest_path}: dataset.unavailable_sentinel must be the paired 0.0 sentinel")
        if not isinstance(dataset.get("anomalies"), list):
            issues.append(f"{manifest_path}: dataset.anomalies must be an array")

    entries = manifest.get("tables")
    if not isinstance(entries, list) or not entries:
        return [*issues, f"{manifest_path}: tables must be a non-empty array"]

    seen_paths: set[str] = set()
    retained_rows_by_path: dict[str, int] = {}
    actual_totals = {"upstream_rows": 0, "packaged_rows": 0, "measured_rows": 0, "zero_sentinel_rows": 0}
    for index, raw_entry in enumerate(entries):
        context = f"{manifest_path}: tables[{index}]"
        if not isinstance(raw_entry, dict):
            issues.append(f"{context} must be an object")
            continue
        relative = raw_entry.get("path")
        if not isinstance(relative, str) or not relative.endswith("_perf.parquet"):
            issues.append(f"{context}: path must name a *_perf.parquet file")
            continue
        if relative in seen_paths:
            issues.append(f"{context}: duplicate path {relative}")
            continue
        seen_paths.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            issues.append(f"{context}: path escapes manifest directory: {relative}")
            continue
        if not path.is_file():
            issues.append(f"{context}: missing file {relative}")
            continue
        relative_parts = Path(relative).parts
        if (
            len(relative_parts) != 4
            or backend is None
            or version is None
            or relative_parts[1:3] != (backend, version)
        ):
            issues.append(f"{context}: path does not match the declared backend/version: {relative}")

        packaged_sha = raw_entry.get("packaged_sha256")
        actual_sha = _sha256(path)
        if packaged_sha != actual_sha:
            issues.append(f"{context}: packaged_sha256 mismatch for {relative}")

        mode = raw_entry.get("import_mode")
        upstream_sha = raw_entry.get("upstream_sha256")
        if mode not in {"exact-copy", "identity-merge"}:
            issues.append(f"{context}: import_mode must be exact-copy or identity-merge")
        if (
            not isinstance(upstream_sha, str)
            or len(upstream_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in upstream_sha)
        ):
            issues.append(f"{context}: upstream_sha256 must be a SHA-256 digest")
        elif mode == "exact-copy" and upstream_sha != packaged_sha:
            issues.append(f"{context}: exact-copy hashes differ for {relative}")

        expected = {
            name: _integer(raw_entry, name, context=context, issues=issues)
            for name in actual_totals
        }
        try:
            table = pq.read_table(path)
        except Exception as exc:
            issues.append(f"{context}: cannot read {relative}: {exc}")
            continue
        for problem in power_metric_issues(table):
            issues.append(f"{context}: {relative}: {problem}")
        if not all(name in table.column_names for name in POWER_COLUMNS):
            issues.append(f"{context}: manifested table must include power and power_limit: {relative}")
            continue

        identity_columns, shape_bounds = table_identity_evidence(table)
        if raw_entry.get("identity_columns") != identity_columns:
            issues.append(f"{context}: identity_columns do not match {relative}")
        if raw_entry.get("shape_bounds") != shape_bounds:
            issues.append(f"{context}: shape_bounds do not match {relative}")
        if not identity_columns:
            issues.append(f"{context}: table has no identity columns: {relative}")
        elif table.select(identity_columns).group_by(identity_columns).aggregate([]).num_rows != table.num_rows:
            issues.append(f"{context}: identity_columns do not uniquely identify every row in {relative}")

        power = table.column("power").to_pylist()
        power_limit = table.column("power_limit").to_pylist()
        measured_rows = sum(
            a is not None and b is not None and a > 0.0 and b > 0.0
            for a, b in zip(power, power_limit, strict=True)
        )
        zero_rows = sum(a == 0.0 and b == 0.0 for a, b in zip(power, power_limit, strict=True))
        actual = {
            "packaged_rows": table.num_rows,
            "measured_rows": measured_rows,
            "zero_sentinel_rows": zero_rows,
        }
        if measured_rows + zero_rows != table.num_rows:
            issues.append(f"{context}: measured and zero-sentinel counts do not cover every row")
        for name, value in actual.items():
            if expected[name] is not None and expected[name] != value:
                issues.append(f"{context}: {name} is {value}, expected {expected[name]}")
        if expected["upstream_rows"] is not None and expected["packaged_rows"] is not None:
            if mode == "exact-copy" and expected["upstream_rows"] != expected["packaged_rows"]:
                issues.append(f"{context}: exact-copy row counts differ for {relative}")
            if mode == "identity-merge" and expected["upstream_rows"] > expected["packaged_rows"]:
                issues.append(f"{context}: identity-merge dropped upstream rows for {relative}")
            retained_rows_by_path[relative] = expected["packaged_rows"] - expected["upstream_rows"]
        for name in actual_totals:
            if expected[name] is not None:
                actual_totals[name] += expected[name]

    if backend is not None and version is not None:
        packaged_paths = {
            path.relative_to(root).as_posix()
            for path in root.glob(f"*/{backend}/{version}/*_perf.parquet")
        }
        if missing := sorted(packaged_paths - seen_paths):
            issues.append(f"{manifest_path}: manifest omits packaged tables: {', '.join(missing)}")
        if extra := sorted(seen_paths - packaged_paths):
            issues.append(f"{manifest_path}: manifest lists tables outside the dataset: {', '.join(extra)}")

    totals = manifest.get("totals")
    if not isinstance(totals, dict):
        issues.append(f"{manifest_path}: totals must be an object")
    else:
        for name, actual in actual_totals.items():
            expected_total = _integer(totals, name, context=f"{manifest_path}: totals", issues=issues)
            if expected_total is not None and expected_total != actual:
                issues.append(f"{manifest_path}: totals.{name} is {expected_total}, table sum is {actual}")
        packaged_total = totals.get("packaged_rows")
        measured_total = totals.get("measured_rows")
        zero_total = totals.get("zero_sentinel_rows")
        valid_total_counts = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (packaged_total, measured_total, zero_total)
        )
        if valid_total_counts and measured_total + zero_total != packaged_total:
            issues.append(f"{manifest_path}: totals do not cover every packaged row")

    if isinstance(dataset, dict) and isinstance(dataset.get("anomalies"), list):
        anomalies = dataset["anomalies"]
        anomaly_rows = 0
        affected_paths: set[str] = set()
        for index, anomaly in enumerate(anomalies):
            context = f"{manifest_path}: dataset.anomalies[{index}]"
            if not isinstance(anomaly, dict):
                issues.append(f"{context} must be an object")
                continue
            rows = _integer(anomaly, "rows", context=context, issues=issues)
            if rows is not None:
                anomaly_rows += rows
            if anomaly.get("kind") != "aisimulate-only-identities":
                issues.append(f"{context}: kind must identify aisimulate-only-identities")
            if anomaly.get("treatment") != "paired-zero-sentinel":
                issues.append(f"{context}: treatment must identify the paired-zero sentinel")
            affected = anomaly.get("tables")
            if not isinstance(affected, list) or not affected:
                issues.append(f"{context}: tables must be a non-empty array")
            else:
                affected_rows = 0
                for table_index, affected_table in enumerate(affected):
                    table_context = f"{context}.tables[{table_index}]"
                    if not isinstance(affected_table, dict):
                        issues.append(f"{table_context} must be an object")
                        continue
                    path = affected_table.get("path")
                    if path not in seen_paths:
                        issues.append(f"{table_context}: path must identify a manifested table")
                    value = _integer(affected_table, "rows", context=table_context, issues=issues)
                    if value is not None:
                        affected_rows += value
                        expected_retained = retained_rows_by_path.get(path)
                        if expected_retained is not None and value != expected_retained:
                            issues.append(
                                f"{table_context}: rows is {value}, expected {expected_retained} retained identities"
                            )
                    if isinstance(path, str):
                        if path in affected_paths:
                            issues.append(f"{table_context}: duplicate anomaly table path")
                        affected_paths.add(path)
                if rows is not None and affected_rows != rows:
                    issues.append(f"{context}: rows does not match the affected-table sum")
        retained_rows = actual_totals["packaged_rows"] - actual_totals["upstream_rows"]
        if anomaly_rows != retained_rows:
            issues.append(
                f"{manifest_path}: anomaly rows are {anomaly_rows}, expected {retained_rows} retained identities"
            )
        retained_paths = {path for path, rows in retained_rows_by_path.items() if rows}
        if retained_paths != affected_paths:
            issues.append(f"{manifest_path}: anomaly table paths do not match tables with retained identities")
    return issues


def _default_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "aiconfigurator_core" / "systems" / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="*", type=Path, help="manifest paths; defaults to every packaged manifest")
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    args = parser.parse_args(argv)
    manifests = args.manifests or sorted(args.data_root.rglob(MANIFEST_BASENAME))
    if not manifests:
        print(f"no {MANIFEST_BASENAME} files found under {args.data_root}", file=sys.stderr)
        return 1
    issues = [issue for manifest in manifests for issue in validate_manifest(manifest)]
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    print(f"validated {len(manifests)} power-data manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
