# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared power-field storage checks for Collector V3 review tooling."""

from __future__ import annotations

import math

import pyarrow as pa

POWER_COLUMNS = ("power", "power_limit")


def power_metric_issues(table: pa.Table) -> list[str]:
    """Return violations of the committed power-data contract.

    A table may omit power entirely. Once either optional metric is present,
    both columns must be float64 and every row must be either a measured
    positive pair or the typed ``0.0``/``0.0`` unavailable sentinel.
    Measurement-quality thresholds are outside this storage contract.
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
    for power, power_limit in pairs:
        assert power is not None and power_limit is not None
        sentinel = power == 0.0 and power_limit == 0.0
        measured = power > 0.0 and power_limit > 0.0
        if not sentinel and not measured:
            invalid_pairs += 1
    if invalid_pairs:
        issues.append(f"power/power_limit contains {invalid_pairs} rows that are neither positive pairs nor 0.0 pairs")
    return issues
