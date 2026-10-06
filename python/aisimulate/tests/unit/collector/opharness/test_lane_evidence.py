# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""lane_evidence grades a collector lane guard against identity records.

The int4_wo/SM100 cell of the b200_sxm sglang 0.5.21 run (job 469988017) is
the reference case: the guard was closed while results/sm100 + the moe_auto
retest confirmed the declared flashinfer_trtllm backend -> inconsistent."""
import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture()
def le():
    spec = importlib.util.spec_from_file_location("lane_evidence_t", COMPONENTS / "lane_evidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "guard, evidence, consistent",
    [
        ("open", "match", True),
        ("open", "contradict", False),
        ("open", "absent", False),
        ("closed", "match", False),      # int4_wo / SM100 before 95085922
        ("closed", "contradict", True),
        ("closed", "absent", True),
    ],
)
def test_grade_table(le, guard, evidence, consistent):
    ok, why = le.grade(guard, evidence)
    assert ok is consistent and why


def test_evidence_state_reads_only_backend_bearing_strings(le):
    rx = "trtllm_gen_moe|FLASHINFER_TRTLLM"
    assert le.evidence_state(["CompressedTensorsFusedMoE->trtllm_gen_moe"], rx) == "match"
    assert le.evidence_state(["CompressedTensorsFusedMoE->marlin"], rx) == "contradict"
    assert le.evidence_state([], "marlin") == "absent"
    assert le.evidence_state(["marlin"], None) == "absent"


GUARD_SRC = '''
def _raise_if_unverified_moe_lane(moe_type):
    installed_version = _dist_version("sglang")
    if moe_type != "int4_wo":
        return installed_version
    if _check_compat("sglang>=0.5.21,<0.5.22", installed_version) and get_sm_version() in (89, 90):
        return installed_version
    raise RuntimeError(f"{moe_type} unverified: {installed_version} SM{get_sm_version()}")
'''


def _harness(tmp_path, moe_identity, sm="sm100"):
    h = tmp_path / "harness"
    (h / "results" / sm).mkdir(parents=True)
    (h / "results" / sm / "sglang-0.5.21.yaml").write_text(yaml.safe_dump({
        "_meta": {"version": "0.5.21"},
        "results": {"moonshotai/Kimi-K2.5": {"verdict": "pass", "moe": moe_identity}}}))
    src = tmp_path / "collect_moe_fake.py"
    src.write_text(GUARD_SRC)
    rules = {"guards": {"sglang.moe": {"source": str(src), "function": "_raise_if_unverified_moe_lane", "lanes": {
        "int4_wo": {"models": ["moonshotai/Kimi-K2.5"],
                    "by_sm": {"sm100": {"declared": "flashinfer_trtllm", "serving": "trtllm_gen_moe|FLASHINFER_TRTLLM"},
                              "sm90": {"declared": "marlin", "serving": "marlin"}}}}}}}
    return h, rules


def test_closed_guard_with_confirming_evidence_is_inconsistent(le, tmp_path):
    h, rules = _harness(tmp_path, "CompressedTensorsFusedMoE->trtllm_gen_moe")
    state = le.evaluate("sglang", "0.5.21", "sm100", harness=h, rules=rules)
    (cell,) = state["cells"]
    assert cell["guard_state"] == "closed" and cell["evidence"] == "match"
    assert state["consistent"] is False and "refused" in cell["why"]


def test_open_guard_with_confirming_evidence_is_consistent(le, tmp_path):
    h, rules = _harness(tmp_path, "CompressedTensorsFusedMoE->marlin", sm="sm90")
    state = le.evaluate("sglang", "0.5.21", "sm90", harness=h, rules=rules)
    (cell,) = state["cells"]
    assert cell["guard_state"] == "open" and cell["evidence"] == "match" and state["consistent"] is True


def test_bare_module_name_is_no_evidence(le, tmp_path):
    h, rules = _harness(tmp_path, "Mxfp4MoE")  # older record shape: no "->kernel_family"
    state = le.evaluate("sglang", "0.5.21", "sm100", harness=h, rules=rules)
    (cell,) = state["cells"]
    assert cell["evidence"] == "absent" and cell["guard_state"] == "closed" and state["consistent"] is True


def test_repo_rules_grade_the_real_sm100_sglang_0521_guard_consistent(le):
    """The committed guard + committed records: every sglang lane cell on sm100 agrees
    (this is the state 95085922 restored; the pre-fix guard reads int4_wo inconsistent)."""
    state = le.evaluate("sglang", "0.5.21", "sm100")
    assert state["cells"] and state["consistent"], [c for c in state["cells"] if not c["consistent"]]
