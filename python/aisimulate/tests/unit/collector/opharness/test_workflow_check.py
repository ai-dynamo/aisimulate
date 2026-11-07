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
    """A scratch harness dir (workflows), workspace results and gates table beside a scratch collector dir."""
    spec = importlib.util.spec_from_file_location("workflow_check_t", COMPONENTS / "workflow_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["workflow_check_t"] = mod
    spec.loader.exec_module(mod)
    harness = tmp_path / "harness"
    (harness / "components").mkdir(parents=True)
    (harness / "workflows").mkdir()
    monkeypatch.setattr(mod, "HARNESS", harness)
    monkeypatch.setattr(mod, "_results", lambda: tmp_path / "ws" / "results")
    monkeypatch.setattr(mod, "registry_ops", lambda fw: [])  # coverage is tested on its own
    monkeypatch.setattr(mod, "GATES_TABLE", tmp_path / "gates.yaml")
    return mod


def _gates_table(wc, rows, planned=None):
    """A scratch gates.yaml for backend vllm. ``planned`` maps SM -> gate names whose case the
    plan holds there (default: every gate on every SM); it stands in for the op plan."""
    doc = {"gates": []}
    for row in rows:
        row = {"gate": row} if isinstance(row, str) else row
        base = {"backend": "vllm", "op": "gemm", "select": {"_gate": row["gate"]}, "repo": "org/m", "kv": "auto"}
        doc["gates"].append({**base, **row})
    wc.GATES_TABLE.write_text(yaml.safe_dump(doc))
    names = {g["gate"] for g in doc["gates"]}
    gm = wc._gates_module()
    gm._matches = lambda backend, op, sm, select: select["_gate"] in (planned or {}).get(sm, names)
    gm.gate_version = lambda g: g.get("version", "0.29.0")  # stands in for the manifest pin


def _verdict(wc, gate, verdict="aligned", fw="vllm", version="0.29.0", repo="org/m"):
    d = wc._results() / "pathdiff" / "sm90" / "vllm-0.29.0"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{gate}.json").write_text(json.dumps({"verdict": verdict, "framework": fw, "version": version, "repo": repo}))


P = {"fw": "vllm", "version": "0.29.0", "sm": "sm90"}


def test_one_identity_free_verdict_does_not_satisfy_the_declared_gates(wc):
    _gates_table(wc, ["gemm_fp8_M", "attn_ctx_M"])
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
    assert wc.declared_gates("vllm", "0.29.0") == {"gemm_fp8_M", "attn_ctx_M"}


def test_no_declared_gates_is_not_aligned(wc):
    _gates_table(wc, [])
    ok, reason = wc.pred_gates_aligned(P)
    assert ok is False and "no gates declared" in reason


def test_repo_and_family_narrow_the_gates_in_scope(wc):
    """onboard_model passes repo=, new_op_collector passes family=: the same check, scoped."""
    _gates_table(wc, ["gemm_fp8_M", "attn_ctx_M"])
    _verdict(wc, "gemm_fp8_M", repo="org/new")
    assert wc.pred_gates_aligned({**P, "family": "gemm"}) == (True, "1 gates in scope, all aligned")
    assert wc.pred_gates_aligned({**P, "family": "moe"})[0] is False
    assert wc.pred_gates_aligned({**P, "repo": "org/new"}) == (True, "1 gates in scope, all aligned")
    ok, reason = wc.pred_gates_aligned({**P, "repo": "org/other"})
    assert ok is False and "no gate verdict names org/other" in reason


def test_a_gate_belongs_to_the_release_its_op_is_collected_with(wc):
    """vLLM gemm/moe/kda are pinned apart from the default: their gates are not the default
    release's, and an upgrade done op by op checks one op's gates."""
    _gates_table(wc, ["dsa_ctx_M", {"gate": "gemm_M", "version": "0.31.0"}, {"gate": "moe_M", "op": "moe"}])
    assert wc.declared_gates("vllm", "0.29.0") == {"dsa_ctx_M", "moe_M"}
    assert wc.declared_gates("vllm", "0.31.0") == {"gemm_M"}
    assert wc.declared_gates("vllm", "0.29.0", op="moe") == {"moe_M"}
    _verdict(wc, "moe_M")
    assert wc.pred_gates_aligned({**P, "op": "moe"}) == (True, "1 gates in scope, all aligned")


def test_pin_is_reads_the_manifest_default_or_a_family_pin(wc):
    manifest = yaml.safe_load((wc.COLLECTOR / "framework_manifest.yaml").read_text())["frameworks"]["vllm"]
    default = str(manifest["default"]["version"])
    assert wc.pred_pin_is({"fw": "vllm", "version": default})[0] is True
    for family in (manifest.get("families") or {}).values():
        assert wc.pred_pin_is({"fw": "vllm", "version": str(family["version"])})[0] is True
    ok, reason = wc.pred_pin_is({"fw": "vllm", "version": "0.0.1"})
    assert ok is False and "want 0.0.1" in reason


def test_evaluate_names_the_first_todo_and_flags_unknown_checks(wc):
    wc.WORKFLOWS["w"] = [("pin", "owner", "pin_is"), ("odd", "script", "no_such_check")]
    manifest = yaml.safe_load((wc.COLLECTOR / "framework_manifest.yaml").read_text())["frameworks"]["vllm"]
    state = wc.evaluate("w", {"fw": "vllm", "version": str(manifest["default"]["version"]), "sm": "sm90"})
    assert [(s["id"], s["status"]) for s in state["steps"]] == [("pin", "done"), ("odd", "todo")]
    assert state["first_todo"] == "odd" and state["all_done"] is False
    assert "unknown check" in state["steps"][1]["reason"]


def test_gate_coverage_is_checked_against_the_registry_not_the_table(wc, monkeypatch):
    """Review 2026-09-29: the 0.30 gate rebuild kept 19 of 32 gates (kda decode among them) and
    the path check reported 19/19. Without a repo/family filter every registry op needs a gate
    row naming it, one per planned phase when its cases carry a phase, or an UNGATED reason."""
    ops = ["gemm", "gdn", "compute_scale", "msa_context_module"]
    monkeypatch.setattr(wc, "registry_ops", lambda fw: ops)
    gdn = lambda name, phase: {"gate": name, "op": "gdn", "select": {"_gate": name, "phase": phase}}
    _gates_table(wc, ["gemm_bf16_Llama", gdn("gdn_ctx_Qwen", "context")])
    ok, reason = wc.pred_gates_aligned({"fw": "vllm", "version": "0.29.0", "sm": "sm90"})
    assert not ok and "gdn:generation" in reason and "msa_context_module" in reason
    assert "compute_scale" not in reason  # listed in UNGATED with its reason
    msa = {"gate": "msa_ctx_M3", "op": "msa_context_module"}
    _gates_table(wc, ["gemm_bf16_Llama", gdn("gdn_ctx_Qwen", "context"), gdn("gdn_gen_Qwen", "generation"), msa])
    ok, reason = wc.pred_gates_aligned({"fw": "vllm", "version": "0.29.0", "sm": "sm90"})
    assert not ok and "without a verdict" in reason  # covered now; the verdicts are still missing
    # an op the table does not know is a finding, never silently gate-free
    monkeypatch.setattr(wc, "registry_ops", lambda fw: ["brand_new_op"])
    ok, reason = wc.pred_gates_aligned({"fw": "vllm", "version": "0.29.0", "sm": "sm90"})
    assert not ok and "brand_new_op" in reason


def test_kda_verify_is_the_one_ungated_phase(wc):
    names = lambda *phases: [{"gate": f"kda_{p}", "op": "kda", "select": {"phase": p}} for p in phases]
    assert wc.uncovered("vllm", ["kda"], names("context")) == ["kda:generation"]
    assert wc.uncovered("vllm", ["kda"], names("context", "generation")) == []


def test_the_registries_are_read_from_the_collector():
    spec = importlib.util.spec_from_file_location("workflow_check_reg", COMPONENTS / "workflow_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {"gemm", "moe"} <= set(mod.registry_ops("vllm")) and "dsv41_module" in mod.registry_ops("sglang")


def test_the_plan_and_no_serving_floors_decide_the_gates_on_an_sm(wc):
    """A gate exists on an SM only where its case is planned; a no_serving floor (the model cannot
    be served on that SM's probe box) keeps it out of path_aligned there while coverage counts it."""
    rows = [
        "mla_ctx_bf16_DeepSeek-V3",
        {"gate": "mla_ctx_fp8_DeepSeek-R1", "no_serving": {"sm120": "TRITON_MLA fp8-KV smem", "sm89": "capacity"}},
        "dsa_ctx_fp8_s512",
    ]
    _gates_table(wc, rows, planned={100: {"mla_ctx_bf16_DeepSeek-V3", "mla_ctx_fp8_DeepSeek-R1"}})
    full = {"mla_ctx_bf16_DeepSeek-V3", "mla_ctx_fp8_DeepSeek-R1", "dsa_ctx_fp8_s512"}
    assert wc.declared_gates("vllm", "0.29.0") == full
    assert wc.declared_gates("vllm", "0.29.0", "sm120") == {"mla_ctx_bf16_DeepSeek-V3", "dsa_ctx_fp8_s512"}
    assert wc.declared_gates("vllm", "0.29.0", "sm89") == {"mla_ctx_bf16_DeepSeek-V3", "dsa_ctx_fp8_s512"}
    assert wc.declared_gates("vllm", "0.29.0", "sm90") == full
    # on SM100 the plan holds only the two MLA cases: the DSA gate is not a gate there
    assert wc.declared_gates("vllm", "0.29.0", "sm100") == {"mla_ctx_bf16_DeepSeek-V3", "mla_ctx_fp8_DeepSeek-R1"}


def test_skips_rechecked_needs_a_kept_record_with_evidence_for_every_entry_in_scope(wc):
    """Rule 7: a pin bump re-checks every collector/skips.py entry in scope, by source or run."""
    p = {"fw": "vllm", "version": "0.33.0", "sm": "sm90", "op": "dsv4_hca_context_module"}
    ok, reason = wc.pred_skips_rechecked(p)
    assert ok is False and "no recheck record" in reason
    ci = wc._case_inventory_module()
    keys = [ci.skip_key(e) for e in ci.skips_in_scope("vllm", ["dsv4_hca_context_module"])]
    assert keys and len(keys) < len(ci.skips_in_scope("vllm"))
    f = wc._results() / "sm90" / "rechecks" / "vllm-0.33.0.yaml"
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
    d = wc._results() / "sm100" / "cases"
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
    # an upgrade done op by op: another op's regression is that op's step
    assert wc.pred_case_set_no_regression({**p, "op": "moe"})[0] is True
    assert wc.pred_case_set_no_regression({**p, "op": "moe,dsv4_paged_mqa_logits_module"})[0] is False
    doc["diff"] = {"regressions": [], "explained": ["moe: cases 10 -> 8, skips.py ['dtype=nvfp4']"], "additions": []}
    (d / "sglang-0.5.21.yaml").write_text(yaml.safe_dump(doc))
    ok, reason = wc.pred_case_set_no_regression(p)
    assert ok is True and "1 losses under skips.py" in reason
    doc["_meta"]["dirty"] = True
    (d / "sglang-0.5.21.yaml").write_text(yaml.safe_dump(doc))
    assert wc.pred_case_set_no_regression(p)[0] is False
