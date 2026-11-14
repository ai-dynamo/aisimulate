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


def test_build_inventory_and_tables_read_the_plan_report(ci, tmp_path):
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
    assert ops["moe_a2a"]["cases"] is None and "no op plan" in ops["moe_a2a"]["note"]
    assert ci.read_tables(report) == {"moe": (["moe_type", "num_tokens"], [["fp8", "1"], ["bfloat16", "2"]])}


H = ["tp_size", "kv_cache_dtype", "seq_len"]


def _rows(tps=("1", "8"), kvs=("bfloat16", "fp8"), seqs=("1", "16")):
    return [[t, k, q] for t in tps for k in kvs for q in seqs]


def _diff(ci, prev_rows, new_rows, new_removed=None, prev_removed=None, header=H, new_header=None):
    prev = {"attn": {"cases": len(prev_rows), "removed": prev_removed or {}}}
    new = {"attn": {"cases": len(new_rows), "removed": new_removed or {}}}
    return ci.diff_plans(prev, new, {"attn": (header, prev_rows)}, {"attn": (new_header or header, new_rows)})


def test_a_lost_value_is_not_explained_by_a_region_that_merely_contains_it(ci):
    """A bug drops every tp_size=8 case while an unrelated skip removes fp8-KV cases: the
    substring rule credited '8' to 'kv_cache_dtype=fp8'; each lost case must carry the region."""
    new = [r for r in _rows(tps=("1",)) if r[1] != "fp8"]
    d = _diff(ci, _rows(), new, {"kv_cache_dtype=fp8": 4})
    # the fp8-KV cases (tp 1 and 8) carry the region; the tp_size=8 bf16 cases do not
    assert d["explained"] == ["attn: 4 cases left under skips.py ['kv_cache_dtype=fp8']"]
    (reg,) = d["regressions"]
    assert reg.startswith("attn: 2 cases left the plan without a skips.py entry")
    assert "'tp_size': '8', 'kv_cache_dtype': 'bfloat16'" in reg


def test_a_region_must_match_every_pair_exactly(ci):
    # seq_len=1 names seq_len 1, never 16; dtype=<v> names a case carrying v in any dtype field
    new = [r for r in _rows() if r[2] != "1"]
    assert _diff(ci, _rows(), new, {"seq_len=1": 4})["regressions"] == []
    new = [r for r in _rows() if r[2] != "16"]
    assert _diff(ci, _rows(), new, {"seq_len=1": 4})["regressions"]
    new = [r for r in _rows() if r[1] != "fp8"]
    assert _diff(ci, _rows(), new, {"dtype=fp8": 4})["regressions"] == []


def test_additions_never_offset_a_loss(ci):
    new = [r for r in _rows() if r[0] != "8"] + [
        ["4", "bfloat16", "1"],
        ["4", "fp8", "1"],
        ["4", "fp8", "16"],
        ["4", "bfloat16", "16"],
    ]
    d = _diff(ci, _rows(), new)
    assert d["additions"] == ["attn: +4 cases"] and d["regressions"]


def test_a_sweep_that_gains_points_still_covers_the_old_case(ci):
    """moe num_tokens gained 32768 and 65536 (a widened sweep, not a loss); a sweep that drops a
    point loses the case."""
    h = ["moe_type", "num_tokens"]
    prev = [["fp8", "(1, 2, 4)"], ["bf16", "(1, 2, 4)"]]
    d = _diff(ci, prev, [["fp8", "(1, 2, 4, 32768)"], ["bf16", "(1, 2, 4, 32768)"]], header=h)
    assert d["regressions"] == [] and d["additions"] == ["attn: 2 cases whose sweep gained points"]
    d = _diff(ci, prev, [["fp8", "(1, 4)"], ["bf16", "(1, 2, 4)"]], header=h)
    assert len(d["regressions"]) == 1 and "'moe_type': 'fp8'" in d["regressions"][0]


def test_cases_compare_on_shared_columns_without_the_dispatch(ci):
    new = [[*r, "flashinfer"] for r in _rows()]
    d = _diff(ci, [[*r, "fa3"] for r in _rows()], new, header=[*H, "dispatch"])
    assert d == {"regressions": [], "explained": [], "additions": []}
    d = _diff(ci, _rows(), [[*r, "x"] for r in _rows()], new_header=[*H, "new_field"])
    assert d == {"regressions": [], "explained": [], "additions": []}


def test_model_entries_are_credited_by_count_and_lost_ops_are_regressions(ci):
    new = [r for r in _rows() if r[0] != "8"]
    d = _diff(ci, _rows(), new, {"model=DeepseekV4ForCausalLM": 4})
    assert d["regressions"] == [] and d["explained"] == ["attn: 4 cases left under skips.py ['model entries']"]
    d = ci.diff_plans({"paged": {"cases": 2}}, {}, {}, {})
    assert d["regressions"] == ["paged: op left the plan"]
    assert ci.diff_plans({"a2a": {"cases": None}}, {"a2a": {"cases": None}}, {}, {})["regressions"] == []


def test_the_base_extract_carries_every_plan_input(ci):
    """op_plans reads packaged checkpoint configs outside collector/ (max_position, checkpoint
    quant); a base plan computed without them silently differs."""
    assert ci.PLAN_INPUTS == ("collector", "src/aisimulate_core/model_configs")
