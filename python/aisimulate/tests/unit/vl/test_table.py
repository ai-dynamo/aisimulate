# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host cost table: exact lookup, one serving environment per table, and the collect command on a miss."""

import pytest

from aisimulate.config.engine import FrontendMeasurementConfig, FrontendStageConfig, HostProfileConfig
from aisimulate.vl.table import (
    FrontendRow,
    HostCostTable,
    MissingRow,
    TableEnvironment,
    lookup,
    resolve_frontend,
    update_table,
)

pytestmark = pytest.mark.unit


def _environment(cpu: str = "test-cpu", sglang: str = "0.5.19") -> TableEnvironment:
    return TableEnvironment(cpu=cpu, host="serving-host", threads=8, sglang_version=sglang, python="3.10.12")


def _measurement(**overrides) -> FrontendMeasurementConfig:
    base = {"frontend": "python", "feature_transport": "shm", "height": 480, "width": 480, "count": 2}
    return FrontendMeasurementConfig(**{**base, **overrides})


def _row(service_ms: float = 4.0, **overrides) -> FrontendRow:
    return FrontendRow(
        model="m",
        measured_for=_measurement(**overrides),
        stages=[FrontendStageConfig(workers=2, service_ms=service_ms)],
        provenance={"sampled_at": "now"},
    )


def test_a_remeasured_row_replaces_the_old_one_and_lookup_returns_it(tmp_path):
    path = tmp_path / "table.json"
    update_table(path, _environment(), _row())
    table = update_table(path, _environment(), _row(service_ms=5.0))
    assert len(table.rows) == 1
    frontend, digest = resolve_frontend(
        HostProfileConfig(path=str(path), frontend="python"),
        model="m",
        images={"height": 480, "width": 480, "count": 2},
        tensor=1,
    )
    assert frontend.stages[0].service_ms == 5.0
    assert frontend.measured_for == _measurement()
    # The digest covers the environment and the costs, not the provenance.
    same_costs = _row(service_ms=5.0).model_copy(update={"provenance": {"sampled_at": "later"}})
    assert digest == table.digest(same_costs) != table.digest(_row(service_ms=6.0))


def test_a_table_belongs_to_one_serving_environment(tmp_path):
    path = tmp_path / "table.json"
    update_table(path, _environment(), _row())
    with pytest.raises(ValueError, match="one table per serving environment"):
        update_table(path, _environment(cpu="other-cpu"), _row(frontend="rust", feature_transport="inline"))


def test_a_miss_names_the_command_that_measures_the_row(tmp_path):
    table = HostCostTable(environment=_environment(), rows=[_row()])
    measurement = FrontendMeasurementConfig(
        frontend="rust", feature_transport="shm", height=1024, width=1024, encoding="jpeg", max_pixels=1_000_000
    )
    with pytest.raises(MissingRow) as error:
        lookup(table, tmp_path / "t.json", model="m", measurement=measurement)
    message = str(error.value)
    assert "no row for m rust shm 1024x1024x1 jpeg max_pixels=1000000" in message
    for flag in ("--frontend rust", "--tp 2", "--images 1024x1024x1", "--encoding jpeg", "--max-pixels 1000000"):
        assert flag in message
    # A table measured on another sglang release is refused, not matched.
    foreign = HostCostTable(environment=_environment(sglang="0.6.0"), rows=[_row()])
    with pytest.raises(ValueError, match="sglang 0.6.0"):
        lookup(foreign, tmp_path / "t.json", model="m", measurement=_measurement())
