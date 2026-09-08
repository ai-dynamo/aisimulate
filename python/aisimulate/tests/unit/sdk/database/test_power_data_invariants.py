# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power/energy data invariants — version-agnostic by construction.

Policy (2026-08): power/energy tests pin no values and bind to no specific
backend version. They assert query-surface invariants over WHATEVER power
data is currently shipped: every parquet that carries power columns must
satisfy the energy model's input contract (paired float64 metrics; either a
positive measurement pair or the typed 0.0/0.0 unavailable sentinel). The
energy MATH is anchored by the rust synthetic
oracles on power-carrying fixtures (``energy_test_fixtures`` tests in
``operators/{gemm,attention}.rs``); this test guards the shipped data plane
those models consume. If no power-carrying parquet is shipped at all, the
suite records that state explicitly instead of passing vacuously.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

import aiconfigurator_core

pytestmark = pytest.mark.unit

_DATA_ROOT = Path(aiconfigurator_core.__file__).parent / "systems" / "data"
_POWER_COLUMNS = ("power", "power_limit")


def _power_carrying_files() -> list[Path]:
    files = []
    for path in sorted(_DATA_ROOT.rglob("*_perf.parquet")):
        schema = pq.read_schema(path)
        if any(col in schema.names for col in _POWER_COLUMNS):
            files.append(path)
    return files


def test_power_columns_satisfy_energy_model_input_contract():
    files = _power_carrying_files()
    if not files:
        pytest.skip("no power-carrying parquet shipped (energy path idle)")
    problems = []
    for path in files:
        rel = path.relative_to(_DATA_ROOT)
        schema = pq.read_schema(path)
        present = [column for column in _POWER_COLUMNS if column in schema.names]
        if len(present) != len(_POWER_COLUMNS):
            problems.append(f"{rel}: power and power_limit must be present together")
            continue
        if any(str(schema.field(column).type) != "double" for column in _POWER_COLUMNS):
            problems.append(f"{rel}: power and power_limit must be float64")
            continue
        table = pq.read_table(path, columns=list(_POWER_COLUMNS))
        frame = table.to_pandas()
        power = frame["power"]
        power_limit = frame["power_limit"]
        bad_values = power.isna() | power_limit.isna() | ~np.isfinite(power) | ~np.isfinite(power_limit)
        bad_values |= (power < 0) | (power_limit < 0)
        if bad_values.any():
            problems.append(f"{rel}: {int(bad_values.sum())} rows with null/non-finite/negative power metrics")
            continue
        sentinel = (power == 0.0) & (power_limit == 0.0)
        measured = (power > 0.0) & (power_limit > 0.0)
        bad_pairs = ~(sentinel | measured)
        if bad_pairs.any():
            problems.append(f"{rel}: {int(bad_pairs.sum())} invalid power/power_limit pairs")
        over_limit = measured & (power > 1.05 * power_limit)
        if over_limit.any():
            problems.append(f"{rel}: {int(over_limit.sum())} rows above 1.05x power_limit")
    assert not problems, "power data violates the energy-model input contract:\n" + "\n".join(problems)
