# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""case_inventory: the collector's case plan is recorded on a CPU and diffed
against a base revision — the test-case regression check (job 469988017:
paged_mqa out of plan, V4-Pro calib dropped by the getter, mla_context_module
without a producer were all visible in the case set). Cases leave the plan on
purpose only under a collector/skips.py entry."""

import csv
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture()
def ci():
    spec = importlib.util.spec_from_file_location("case_inventory_t", COMPONENTS / "case_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summarize_cases_separates_categorical_from_shape_columns(ci):
    rows = [{"seq_len": str(n), "dtype": "bf16" if n % 2 else "fp8", "model_path": "m"} for n in range(200)]
    f = ci.summarize_cases(rows)
    assert f["seq_len"] == {"kind": "numeric", "min": 0, "max": 199, "distinct": ">64"}
    assert f["dtype"] == {"kind": "categorical", "values": ["bf16", "fp8"]}
    assert f["model_path"]["values"] == ["m"]


def test_build_inventory_reads_the_plan_report_and_its_table(ci, tmp_path):
    table = tmp_path / "vllm_sm90_moe.csv"
    with table.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["moe_type", "num_tokens"])
        writer.writerows([["fp8", 1], ["bfloat16", 2]])
    report = {
        "ops": ["moe", "moe_a2a"],
        "plans": {
            "moe": {
                "cases": 2,
                "by_dtype": {"fp8": 1, "bfloat16": 1},
                "removed": [{"region": "dtype=nvfp4", "cases": 5, "reason": "{'sm': [90]}"}],
                "plan_hash": "sha256:x",
                "table": str(table),
            }
        },
    }
    ops = ci.build_inventory(report)
    assert ops["moe"]["removed"] == {"dtype=nvfp4": 5} and ops["moe"]["plan_hash"] == "sha256:x"
    assert ops["moe"]["fields"]["moe_type"]["values"] == ["bfloat16", "fp8"]
    assert ops["moe_a2a"]["cases"] is None and "no op plan" in ops["moe_a2a"]["note"]


def _inv(ops):
    return {"_meta": {"framework": "sglang", "version": "x"}, "ops": ops}


def test_diff_fails_losses_no_skips_entry_accounts_for(ci):
    cat = {"kind": "categorical"}
    prev = _inv(
        {
            "dsv4_paged_mqa_logits_module": {"cases": 22, "removed": {}, "fields": {}},
            "dsv4_csa_topk_calib": {
                "cases": 22,
                "removed": {},
                "fields": {"model_path": {**cat, "values": ["Flash", "Pro"]}},
            },
            "gemm": {"cases": 100, "removed": {}, "fields": {"n": {"kind": "numeric", "min": 1, "max": 9}}},
        }
    )
    new = _inv(
        {
            "dsv4_csa_topk_calib": {"cases": 11, "removed": {}, "fields": {"model_path": {**cat, "values": ["Flash"]}}},
            "gemm": {"cases": 120, "removed": {}, "fields": {"n": {"kind": "numeric", "min": 1, "max": 12}}},
            "new_op": {"cases": 5, "removed": {}, "fields": {}},
        }
    )
    d = ci.diff_inventories(prev, new)
    assert "dsv4_paged_mqa_logits_module: op left the plan" in d["regressions"]
    assert any("topk_calib: cases 22 -> 11 (0 under skips.py)" in r for r in d["regressions"])
    assert any("topk_calib[model_path] lost values ['Pro']" in r for r in d["regressions"])
    assert d["additions"] == ["gemm: cases 100 -> 120", "new_op: new op (5 cases)"]
    # numeric range changes are never regressions (shape grids move; the SDK interpolates)
    assert not any("gemm" in r for r in d["regressions"])


def test_diff_explains_losses_under_a_new_skips_entry(ci):
    prev = _inv(
        {"moe": {"cases": 10, "removed": {}, "fields": {"dtype": {"kind": "categorical", "values": ["fp8", "nvfp4"]}}}}
    )
    new = _inv(
        {
            "moe": {
                "cases": 6,
                "removed": {"dtype=nvfp4": 4},
                "fields": {"dtype": {"kind": "categorical", "values": ["fp8"]}},
            }
        }
    )
    d = ci.diff_inventories(prev, new)
    assert d["regressions"] == []
    assert d["explained"] == [
        "moe: cases 10 -> 6, skips.py ['dtype=nvfp4']",
        "moe[dtype] lost ['nvfp4'] under skips.py",
    ]


def test_ops_without_an_op_plan_are_not_diffed(ci):
    d = ci.diff_inventories(_inv({"moe_a2a": {"cases": None}}), _inv({"moe_a2a": {"cases": None}}))
    assert d == {"regressions": [], "explained": [], "additions": []}


def test_the_base_extract_carries_every_plan_input(ci):
    """op_plans reads packaged checkpoint configs outside collector/ (max_position, checkpoint
    quant); a base plan computed without them silently differs."""
    assert ci.PLAN_INPUTS == ("collector", "src/aisimulate_core/model_configs")
