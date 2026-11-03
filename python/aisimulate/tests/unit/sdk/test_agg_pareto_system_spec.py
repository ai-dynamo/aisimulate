# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""agg_pareto selects the backend from the database's loaded system spec."""

from unittest.mock import MagicMock

import pytest

from aisimulate.sdk import config, pareto_analysis

pytestmark = pytest.mark.unit


class _Stop(Exception):
    pass


def test_agg_pareto_forwards_loaded_system_spec_to_backend_selection(monkeypatch) -> None:
    # Custom-root systems: classify from the loaded spec, not a name re-load.
    captured = {}

    def fake_get_backend(name, system=None, system_spec=None):
        captured.update(system=system, system_spec=system_spec)
        raise _Stop

    monkeypatch.setattr(pareto_analysis, "get_backend", fake_get_backend)
    monkeypatch.setattr(pareto_analysis, "get_model", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(pareto_analysis, "resolve_sglang_mla_compute", lambda *a, **k: None)
    database = MagicMock()
    database.system = "review_b60"
    database.system_spec = {"gpu": {"mem_capacity": 1}}

    with pytest.raises(RuntimeError):  # every config failed -> agg_pareto gives up
        pareto_analysis.agg_pareto(
            "test-model",
            config.RuntimeConfig(isl=1024, osl=1, ttft=1.0, tpot=1.0),
            database,
            "vllm",
            config.ModelConfig(),
            [(1, 1, 1, 1, 1, 1)],
        )
    assert captured == {"system": "review_b60", "system_spec": database.system_spec}
