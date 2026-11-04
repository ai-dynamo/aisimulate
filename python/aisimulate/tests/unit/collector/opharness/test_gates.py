# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gates.yaml names every path gate as one planned collector case: the probe.trace_case
shard it schedules must select exactly one planned case, fit the pipeline's case-filter
charset, and grade with a hint its backend defines. A gate whose case is planned on no
SM grades nothing (the GLM-5 DSA module capture once timed 128 heads, a cell no plan holds)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture(scope="module")
def gates():
    spec = importlib.util.spec_from_file_location("gates_t", COMPONENTS / "gates.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gates_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ids():
    import yaml

    return [g["gate"] + "@" + g["backend"] for g in yaml.safe_load((COMPONENTS / "gates.yaml").read_text())["gates"]]


@pytest.mark.parametrize("gate_id", _ids())
def test_every_gate_names_one_planned_case(gates, gate_id):
    name, backend = gate_id.split("@")
    (gate,) = [g for g in gates.load()["gates"] if g["gate"] == name and g["backend"] == backend]
    assert gate["hint"] in gates.load()["hints"][backend]
    cases = gates._cases(gate)
    planned_on = [sm for sm in (90, 100) if gates._matches(backend, gate["op"], sm, gate["select"])]
    assert planned_on, f"{gate_id}: select {gate['select']} matches no single planned case on SM90 or SM100"
    for _name, select in cases:
        op = select.pop("op", gate["op"])
        gates.case_filter(backend, op, select, gate.get("env"))  # raises when it does not fit the pipeline


def test_shard_plans_parse_as_pipeline_plans(gates, tmp_path):
    plan = gates.shard_plan("sglang", "sm100")
    assert plan["backend"] == "sglang" and plan["shards"]
    assert all(
        s["op"] == "probe.trace_case" and s["case_filter"].startswith("backend=sglang,op=") for s in plan["shards"]
    )
    assert len({s["name"] for s in plan["shards"]}) == len(plan["shards"])
