# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""case_inventory: the collector's case set is derived without collecting and
diffed version-to-version — the test-case regression check (job 469988017:
paged_mqa out of plan, V4-Pro calib dropped by the getter, mla_context_module
without a producer were all visible in the case inventory)."""
import importlib.util
import sys
import types
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


def test_summarize_cases_separates_categorical_from_shape_fields(ci):
    cases = [(n, "bf16" if n % 2 else "fp8", "sgl-project/DeepSeek-V4-Pro-FP8", True) for n in range(200)]
    f = ci.summarize_cases(cases)
    assert f["0"]["kind"] == "numeric" and f["0"]["min"] == 0 and f["0"]["max"] == 199
    assert f["1"] == {"kind": "categorical", "values": ["bf16", "fp8"]}
    assert f["2"]["values"] == ["sgl-project/DeepSeek-V4-Pro-FP8"]
    assert f["3"]["values"] == ["True"]
    assert ci.summarize_cases([{"id": "x", "params": {"dtype": "fp8", "n": 1}}])["dtype"]["values"] == ["fp8"]


def _inv(ops):
    return {"_meta": {"framework": "sglang", "version": "x"}, "ops": ops}


def test_diff_reports_op_loss_plan_loss_count_loss_and_value_loss(ci):
    prev = _inv({
        "dsv4_paged_mqa_logits_module": {"in_plan": True, "cases": 22, "fields": {}},
        "dsv4_csa_topk_calib": {"in_plan": True, "cases": 22,
                                "fields": {"2": {"kind": "categorical", "values": ["Flash", "Pro"]}}},
        "mla_context_module": {"in_plan": True, "cases": 10, "fields": {}},
        "gemm": {"in_plan": True, "cases": 100, "fields": {"0": {"kind": "numeric", "min": 1, "max": 9}}},
    })
    new = _inv({
        "dsv4_paged_mqa_logits_module": {"in_plan": False, "cases": 22, "fields": {}},
        "dsv4_csa_topk_calib": {"in_plan": True, "cases": 11,
                                "fields": {"2": {"kind": "categorical", "values": ["Flash"]}}},
        "gemm": {"in_plan": True, "cases": 120, "fields": {"0": {"kind": "numeric", "min": 1, "max": 12}}},
        "new_op": {"in_plan": True, "cases": 5, "fields": {}},
    })
    d = ci.diff_inventories(prev, new)
    assert any("paged_mqa" in r and "registry-only" in r for r in d["regressions"])
    assert any("topk_calib: cases 22 -> 11" in r for r in d["regressions"])
    assert any("topk_calib[2] lost values ['Pro']" in r for r in d["regressions"])
    assert any("mla_context_module: op gone" in r for r in d["regressions"])
    assert d["additions"] == ["gemm: cases 100 -> 120", "new_op: new op (5 cases)"]
    # numeric range changes are never regressions (shape grids move; the SDK interpolates)
    assert not any("gemm" in r for r in d["regressions"])


def test_waivers_turn_signed_losses_into_waived(ci):
    prev = _inv({"a": {"in_plan": True, "cases": 2, "fields": {"1": {"kind": "categorical", "values": ["x", "y"]}}},
                 "b": {"in_plan": True, "cases": 3, "fields": {}}})
    new = _inv({"a": {"in_plan": True, "cases": 2, "fields": {"1": {"kind": "categorical", "values": ["x"]}}}})
    d = ci.diff_inventories(prev, new, {"ops": {"b": "retired (owner 2026-10-06)"},
                                        "fields": {"a": {1: "y lane removed from serving"}}})
    assert d["regressions"] == [] and len(d["waived"]) == 2


def test_getter_error_is_a_regression(ci):
    d = ci.diff_inventories(_inv({"a": {"in_plan": True, "cases": 2, "fields": {}}}),
                            _inv({"a": {"in_plan": True, "cases": None, "error": "ImportError: boom"}}))
    assert d["regressions"] == ["a: getter fails to enumerate (ImportError: boom)"]


def test_previous_inventory_picks_the_newest_older_version(ci, tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "HARNESS", tmp_path)
    d = tmp_path / "results" / "sm100" / "cases"
    d.mkdir(parents=True)
    for v in ("0.5.14", "0.5.17", "0.5.21", "0.5.9"):
        (d / f"sglang-{v}.yaml").write_text("{}")
    (d / "sglang-0.5.21.waivers.yaml").write_text("{}")
    assert ci.previous_inventory("sm100", "sglang", "0.5.21").name == "sglang-0.5.17.yaml"
    assert ci.previous_inventory("sm100", "sglang", "0.5.9") is None
    assert ci.inventory_path("sm100", "sglang", "0.5.21") == d / "sglang-0.5.21.yaml"


def test_enumerate_op_uses_the_executor_contract_under_ais_sm(ci, monkeypatch):
    """get_func + capabilities filter, SM taken from AIS_SM (no GPU)."""
    seen = {}

    def get_cases():
        from collector.helper import get_sm_version

        seen["sm"] = get_sm_version()
        return [(1, "bf16"), (2, "fp8")]

    sys.modules["fake_collector_mod"] = types.SimpleNamespace(get_cases=get_cases)
    entry = types.SimpleNamespace(op="gemm", module="fake_collector_mod", get_func="get_cases", unverified_sms=())
    monkeypatch.setenv("AIS_SM", "sm100")
    kept, dropped = ci.enumerate_op("sglang", entry, 100)
    assert seen["sm"] == 100 and len(kept) == 2 and dropped == []
