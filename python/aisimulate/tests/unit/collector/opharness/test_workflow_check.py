# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""workflow_check's four checks are relative to the manifest pin, the recorded plan diff,
the DECLARED gates and skips.py: one aligned file does not satisfy every declared gate,
and a gate set that lost a registry op's family is not aligned."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture()
def wc(tmp_path, monkeypatch):
    """A scratch harness dir (results, captures, workflows) beside a scratch collector dir."""
    spec = importlib.util.spec_from_file_location("workflow_check_t", COMPONENTS / "workflow_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["workflow_check_t"] = mod
    spec.loader.exec_module(mod)
    harness = tmp_path / "harness"
    (harness / "components" / "captures").mkdir(parents=True)
    (harness / "results" / "sm90" / "pathdiff").mkdir(parents=True)
    (harness / "workflows").mkdir()
    monkeypatch.setattr(mod, "HARNESS", harness)
    return mod


def _gates_script(wc, gates):
    (wc.HARNESS / "components" / "captures" / "verdicts_all.sh").write_text(
        "run() { python3 $PD --diff --framework vllm --version 0.29.0 --save-verdict $OUT/$2.json; }\n"
        + "".join(f"run cap_{g} {g} org/m auto hint\n" for g in gates)
        + "# run cap_x explained_x org/m auto hint\n"
    )


def _verdict(wc, gate, verdict="aligned", fw="vllm", version="0.29.0"):
    d = wc.HARNESS / "results" / "pathdiff" / "sm90" / "vllm-0.29.0"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{gate}.json").write_text(json.dumps({"verdict": verdict, "framework": fw, "version": version}))


def _verdict(wc, gate, verdict="aligned", fw="vllm", version="0.29.0", repo="org/m"):
    d = wc.HARNESS / "results" / "pathdiff" / "sm90" / "vllm-0.29.0"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{gate}.json").write_text(json.dumps({"verdict": verdict, "framework": fw, "version": version, "repo": repo}))


P = {"fw": "vllm", "version": "0.29.0", "sm": "sm90"}


def test_one_identity_free_verdict_does_not_satisfy_the_declared_gates(wc):
    _gates_script(wc, ["gemm_fp8_M", "attn_ctx_M"])
    _verdict(wc, "gemm_fp8_M")
    ok, reason = wc.pred_gates_aligned(P)
    assert ok is False and "1/2 declared gates without a verdict" in reason
    # a verdict graded for another framework/version does not count
    _verdict(wc, "attn_ctx_M", fw="sglang", version="0.5.16")
    ok, reason = wc.pred_gates_aligned(P)
    assert ok is False and "another framework/version" in reason
    _verdict(wc, "attn_ctx_M", verdict="diverged")
    ok, reason = wc.pred_gates_aligned(P)
    assert ok is False and "attn_ctx_M=diverged" in reason
    _verdict(wc, "attn_ctx_M")
    assert wc.pred_gates_aligned(P) == (True, "2 gates in scope, all aligned")
    assert wc.declared_gates("vllm", "0.29.0") == {"gemm_fp8_M", "attn_ctx_M"}  # commented lines are not gates


def test_no_declared_gates_is_not_aligned(wc):
    ok, reason = wc.pred_gates_aligned(P)
    assert ok is False and "no gates declared" in reason


def test_repo_and_family_narrow_the_gates_in_scope(wc):
    """onboard_model passes repo=, new_op_collector passes family=: the same check, scoped."""
    _gates_script(wc, ["gemm_fp8_M", "attn_ctx_M"])
    _verdict(wc, "gemm_fp8_M", repo="org/new")
    assert wc.pred_gates_aligned({**P, "family": "gemm"}) == (True, "1 gates in scope, all aligned")
    assert wc.pred_gates_aligned({**P, "family": "moe"})[0] is False
    assert wc.pred_gates_aligned({**P, "repo": "org/new"}) == (True, "1 gates in scope, all aligned")
    ok, reason = wc.pred_gates_aligned({**P, "repo": "org/other"})
    assert ok is False and "no gate verdict names org/other" in reason


def test_pin_is_reads_the_manifest_default_or_a_family_pin(wc):
    manifest = yaml.safe_load((wc.COLLECTOR / "framework_manifest.yaml").read_text())["frameworks"]["vllm"]
    default = str(manifest["default"]["version"])
    assert wc.pred_pin_is({"fw": "vllm", "version": default})[0] is True
    for family in (manifest.get("families") or {}).values():
        assert wc.pred_pin_is({"fw": "vllm", "version": str(family["version"])})[0] is True
    ok, reason = wc.pred_pin_is({"fw": "vllm", "version": "0.0.1"})
    assert ok is False and "want 0.0.1" in reason


def test_evaluate_names_the_first_todo_and_flags_unknown_checks(wc):
    (wc.HARNESS / "workflows" / "w.yaml").write_text(
        yaml.safe_dump(
            {
                "workflow": "w",
                "steps": [
                    {"id": "pin", "done_when": {"check": "pin_is"}},
                    {"id": "odd", "done_when": {"check": "no_such_check"}},
                ],
            }
        )
    )
    manifest = yaml.safe_load((wc.COLLECTOR / "framework_manifest.yaml").read_text())["frameworks"]["vllm"]
    state = wc.evaluate("w", {"fw": "vllm", "version": str(manifest["default"]["version"]), "sm": "sm90"})
    assert [(s["id"], s["status"]) for s in state["steps"]] == [("pin", "done"), ("odd", "todo")]
    assert state["first_todo"] == "odd" and state["all_done"] is False
    assert "unknown check" in state["steps"][1]["reason"]


def test_gate_coverage_is_checked_against_the_registry_not_the_script(wc, tmp_path):
    """Review 2026-09-29: the 0.30 gate rebuild kept 19 of 32 gates and
    the path check reported 19/19. gates_aligned without a repo/family filter requires
    a declared gate family for every registry op (or a recorded gate-free reason)."""
    (wc.HARNESS.parent / "vllm").mkdir(parents=True)
    (wc.HARNESS.parent / "vllm" / "registry.py").write_text(
        'REGISTRY = [OpEntry(op="gemm"), OpEntry(op="gdn"), '
        'OpEntry(op="compute_scale"), OpEntry(op="msa_context_module")]\n'
    )
    sh = wc.HARNESS / "components" / "captures" / "verdicts_vllm_0300.sh"
    sh.write_text(
        "python3 $PD --diff --framework vllm --version 0.30.0 \n"
        "run gemm_bf16 gemm_bf16_Llama meta-llama/Llama auto ''\n"
        "run gdn_ctx gdn_ctx_Qwen Qwen/Q auto ''\n"
    )
    ok, reason = wc.pred_gates_aligned({"fw": "vllm", "version": "0.30.0"})
    assert not ok and "gdn:^gdn_gen" in reason and "msa_context_module" in reason
    sh.write_text(
        sh.read_text() + "run gdn_gen gdn_gen_Qwen Qwen/Q auto ''\nrun msa_ctx msa_ctx_M3 MiniMaxAI/M3 auto ''\n"
    )
    ok, reason = wc.pred_gates_aligned({"fw": "vllm", "version": "0.30.0"})
    assert not ok and "without a verdict" in reason  # covered now; the verdicts are still missing
    # an op the table does not know is a finding, never silently gate-free
    (wc.HARNESS.parent / "vllm" / "registry.py").write_text('REGISTRY = [OpEntry(op="brand_new_op")]\n')
    ok, reason = wc.pred_gates_aligned({"fw": "vllm", "version": "0.30.0"})
    assert not ok and "brand_new_op" in reason


def test_platform_floor_gates_are_declared_coverage_but_not_expected_on_that_sm(wc):
    """A gate marked FLOOR_SM=<sm> has no serving instance on that SM by framework
    fact (sm120 fp8-KV dense MLA): coverage still counts it, path_aligned on that
    SM does not wait for a verdict, other SMs grade it normally."""
    sh = wc.HARNESS / "components" / "captures" / "verdicts_vllm_0300.sh"
    sh.write_text(
        "python3 $PD --diff --framework vllm --version 0.30.0 \n"
        'run mla_ctx_bf16 mla_ctx_bf16_DeepSeek-V3 deepseek-ai/DeepSeek-V3 auto "$MLA"\n'
        'FLOOR_SM=sm120 FLOOR_NOTE="TRITON_MLA fp8-KV smem" '
        'run mla_ctx_fp8 mla_ctx_fp8_DeepSeek-R1 deepseek-ai/DeepSeek-R1 fp8 "$MLA"\n'
        'SERVING_RAW=facts/x.json run dsa_ctx_fp8 dsa_ctx_fp8_s512 deepseek-ai/DeepSeek-V3.2 fp8 "$DSA"\n'
    )
    assert wc.declared_gates("vllm", "0.30.0") == {
        "mla_ctx_bf16_DeepSeek-V3",
        "mla_ctx_fp8_DeepSeek-R1",
        "dsa_ctx_fp8_s512",
    }
    assert wc.declared_gates("vllm", "0.30.0", "sm120") == {"mla_ctx_bf16_DeepSeek-V3", "dsa_ctx_fp8_s512"}
    assert wc.declared_gates("vllm", "0.30.0", "sm90") == wc.declared_gates("vllm", "0.30.0")


def test_floor_sm_accepts_a_comma_list():
    """FLOOR_SM=sm120,sm89 declares one platform floor for both SMs; an SM not in the list still owns the gate."""
    import importlib.util
    import sys
    from pathlib import Path

    comp = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"
    spec = importlib.util.spec_from_file_location("wfc_floor", comp / "workflow_check.py")
    wfc = importlib.util.module_from_spec(spec)
    sys.modules["wfc_floor"] = wfc
    spec.loader.exec_module(wfc)
    full = wfc.declared_gates("vllm", "0.30.0")
    assert "mla_ctx_fp8_DeepSeek-R1" in full
    for sm in ("sm120", "sm89"):
        assert "mla_ctx_fp8_DeepSeek-R1" not in wfc.declared_gates("vllm", "0.30.0", sm)
    assert "mla_ctx_fp8_DeepSeek-R1" in wfc.declared_gates("vllm", "0.30.0", "sm90")


def test_skips_rechecked_needs_a_kept_record_with_evidence_for_every_entry_in_scope(wc):
    """Rule 7: a pin bump re-checks every collector/skips.py entry in scope, by source or run."""
    p = {"fw": "vllm", "version": "0.33.0", "sm": "sm90", "ops": "dsv4_hca_context_module"}
    ok, reason = wc.pred_skips_rechecked(p)
    assert ok is False and "no recheck record" in reason
    ci = wc._case_inventory_module()
    keys = [ci.skip_key(e) for e in ci.skips_in_scope("vllm", ["dsv4_hca_context_module"])]
    assert keys and len(keys) < len(ci.skips_in_scope("vllm"))
    f = wc.HARNESS / "results" / "sm90" / "rechecks" / "vllm-0.33.0.yaml"
    f.parent.mkdir(parents=True)
    items = [{"entry": k, "outcome": "kept", "how": "source", "ref": "vllm/x.py:f"} for k in keys]
    f.write_text(yaml.safe_dump({"skips": items[1:]}))
    ok, reason = wc.pred_skips_rechecked(p)
    assert ok is False and "not rechecked" in reason
    items[0]["ref"] = None
    f.write_text(yaml.safe_dump({"skips": items}))
    ok, reason = wc.pred_skips_rechecked(p)
    assert ok is False and "without kept" in reason
    items[0]["ref"] = "run 78103835"
    f.write_text(yaml.safe_dump({"skips": items}))
    ok, reason = wc.pred_skips_rechecked(p)
    assert ok is True and f"all {len(keys)}" in reason


def test_case_set_no_regression_reads_the_recorded_plan_diff(wc):
    p = {"fw": "sglang", "version": "0.5.21", "sm": "sm100"}
    ok, reason = wc.pred_case_set_no_regression(p)
    assert ok is False and "no case plan" in reason
    d = wc.HARNESS / "results" / "sm100" / "cases"
    d.mkdir(parents=True)
    doc = {"_meta": {"framework": "sglang", "version": "0.5.21", "revision": "abc", "dirty": False}, "ops": {}}
    (d / "sglang-0.5.21.yaml").write_text(yaml.safe_dump(doc))
    ok, reason = wc.pred_case_set_no_regression(p)
    assert ok is False and "no diff" in reason
    doc["_meta"]["base"] = {"rev": "upstream/main"}
    doc["diff"] = {"regressions": ["dsv4_paged_mqa_logits_module: op left the plan"], "explained": [], "additions": []}
    (d / "sglang-0.5.21.yaml").write_text(yaml.safe_dump(doc))
    ok, reason = wc.pred_case_set_no_regression(p)
    assert ok is False and "paged_mqa" in reason
    doc["diff"] = {"regressions": [], "explained": ["moe: cases 10 -> 8, skips.py ['dtype=nvfp4']"], "additions": []}
    (d / "sglang-0.5.21.yaml").write_text(yaml.safe_dump(doc))
    ok, reason = wc.pred_case_set_no_regression(p)
    assert ok is True and "1 losses under skips.py" in reason
    doc["_meta"]["dirty"] = True
    (d / "sglang-0.5.21.yaml").write_text(yaml.safe_dump(doc))
    assert wc.pred_case_set_no_regression(p)[0] is False
