# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host cost table: exact-shape lookup, content digest, and the collect command on a miss."""

import pytest

from aisimulate.config.engine import CostFnConfig, FrontendStageConfig, HostProfileConfig
from aisimulate.vl.table import (
    SGLANG_REVISION,
    FrontendRow,
    HostCostTable,
    MissingRow,
    RowIdentity,
    Shape,
    lookup,
    resolve_frontend,
    save_table,
    upsert_row,
)

pytestmark = pytest.mark.unit


def _row(const_ms: float = 4.0, **identity) -> FrontendRow:
    return FrontendRow(
        identity=RowIdentity(
            **{"cpu": "test-cpu", "sglang_revision": SGLANG_REVISION, "model": "m", "frontend": "python", **identity}
        ),
        shape=Shape(height=480, width=480, count=2, encoding="png", text_tokens=128),
        stages=[FrontendStageConfig(resource="pool", workers=2, cost=CostFnConfig(const_ms=const_ms))],
        provenance={"sampled_at": "now"},
    )


def test_lookup_returns_the_exact_shape_and_digests_the_costs_only(tmp_path):
    table = upsert_row(HostCostTable(), _row())
    # Re-measuring the same identity and shape replaces the row instead of duplicating it.
    table = upsert_row(table, _row(const_ms=5.0))
    assert len(table.rows) == 1
    path = tmp_path / "table.json"
    save_table(path, table)
    frontend, digest = resolve_frontend(
        HostProfileConfig(path=str(path), frontend="python"),
        model="m",
        images={"height": 480, "width": 480, "count": 2},
        text_tokens=128,
    )
    assert frontend.stages[0].cost.const_ms == 5.0
    same_costs = _row(const_ms=5.0).model_copy(update={"provenance": {"sampled_at": "later"}})
    assert digest == same_costs.digest() != _row(const_ms=6.0).digest()


def test_a_miss_names_the_command_that_measures_the_row(tmp_path):
    table = HostCostTable(rows=[_row(), _row(frontend="rust")])
    shape = Shape(height=1024, width=1024, count=1, encoding="jpeg", text_tokens=256, max_pixels=1_000_000)
    with pytest.raises(MissingRow) as error:
        lookup(table, tmp_path / "t.json", model="m", frontend="rust", shape=shape)
    message = str(error.value)
    assert "no row for rust 1024x1024x1 jpeg text_tokens=256 max_pixels=1000000" in message
    flags = ("--frontend rust", "--images 1024x1024x1", "--encoding jpeg", "--text-tokens 256", "--max-pixels 1000000")
    for flag in flags:
        assert flag in message
    # Rows from two serving hosts for one key are ambiguous, not averaged or picked.
    two_cpus = upsert_row(table, _row(cpu="other-cpu"))
    with pytest.raises(ValueError, match="several CPUs"):
        lookup(two_cpus, tmp_path / "t.json", model="m", frontend="python", shape=_row().shape)
