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
import pyarrow.parquet as pq

POWER_COLUMNS = ("power", "power_limit")
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

    entries = manifest.get("tables")
    if not isinstance(entries, list) or not entries:
        return [*issues, f"{manifest_path}: tables must be a non-empty array"]

    root = manifest_path.parent.resolve()
    seen_paths: set[str] = set()
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
            continue

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
        for name in actual_totals:
            if expected[name] is not None:
                actual_totals[name] += expected[name]

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
