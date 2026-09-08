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

import importlib.util
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import aiconfigurator_core

pytestmark = pytest.mark.unit

_DATA_ROOT = Path(aiconfigurator_core.__file__).parent / "systems" / "data"
_POWER_DATA = Path(__file__).resolve().parents[4] / "tools" / "perf_database" / "power_data.py"
_SPEC = importlib.util.spec_from_file_location("power_data_invariants", _POWER_DATA)
assert _SPEC is not None and _SPEC.loader is not None
_POWER_DATA_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_POWER_DATA_MODULE)
_POWER_COLUMNS = _POWER_DATA_MODULE.POWER_COLUMNS
_power_metric_issues = _POWER_DATA_MODULE.power_metric_issues


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
        table = pq.read_table(path, columns=present)
        problems.extend(f"{rel}: {issue}" for issue in _power_metric_issues(table))
    assert not problems, "power data violates the energy-model input contract:\n" + "\n".join(problems)
