# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""workflow_check predicates are relative to the PLAN and the DECLARED gates
(review 2026-09-25 P1/P2 #4): an empty matrix is not complete, one aligned
file does not satisfy every declared gate, and an implemented component never
completes a step by itself."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture()
def wc(tmp_path, monkeypatch):
    """A scratch harness dir (targets, results, captures) and a scratch workspace."""
    os.environ["AIS_PROBE_WORKSPACE"] = str(tmp_path / "ws")
    spec = importlib.util.spec_from_file_location("workflow_check_t", COMPONENTS / "workflow_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["workflow_check_t"] = mod
    spec.loader.exec_module(mod)
    harness = tmp_path / "harness"
    (harness / "components" / "captures").mkdir(parents=True)
    (harness / "results" / "sm90" / "pathdiff").mkdir(parents=True)
    (harness / "workflows").mkdir()
    (harness / "targets.yaml").write_text(yaml.safe_dump(
        {"backends": {"vllm": {"versions": ["0.29.0"]}}, "families": {}, "topologies": []}))
    (tmp_path / "ws" / "archive").mkdir(parents=True)
    monkeypatch.setattr(mod, "HARNESS", harness)
    monkeypatch.setattr(mod, "ROOT", tmp_path / "ws")
    return mod


def _matrix(wc, cells):
    p = wc.HARNESS / "results" / "sm90" / "vllm-0.29.0.yaml"
    p.write_text(yaml.safe_dump({"_meta": {"version": "0.29.0"}, "results": cells}))


def _plan(wc, repos, version="0.29.0"):
    (wc.ROOT / "archive" / "plan.json").write_text(json.dumps(
        [{"id": f"i{n}", "repo": r, "backend": "vllm", "version": version} for n, r in enumerate(repos)]))


def test_review_empty_matrix_is_not_complete(wc):
    _matrix(wc, {})
    ok, reason = wc.pred_matrix_complete({"fw": "vllm", "version": "0.29.0"})
    assert ok is False and "no plan runs" in reason
    _plan(wc, ["org/a", "org/b"])
    ok, reason = wc.pred_matrix_complete({"fw": "vllm", "version": "0.29.0"})
    assert ok is False and "2 planned repos without a cell" in reason


def test_matrix_complete_means_every_planned_repo_has_a_verdict(wc):
    _plan(wc, ["org/a", "org/b"])
    _matrix(wc, {"org/a": {"verdict": "pass"}, "org/b": {"verdict": "fail", "cause": "generator rejects"}})
    assert wc.pred_matrix_complete({"fw": "vllm", "version": "0.29.0"})[0] is True
    # a new planned repo reopens the step
    _plan(wc, ["org/a", "org/b", "org/c"])
    ok, reason = wc.pred_matrix_complete({"fw": "vllm", "version": "0.29.0"})
    assert ok is False and "org/c" in reason
    # stale evidence is not completion either
    _plan(wc, ["org/a"])
    _matrix(wc, {"org/a": {"verdict": "fail", "cause": "stale evidence"}})
    assert "stale" in wc.pred_matrix_complete({"fw": "vllm", "version": "0.29.0"})[1]


def _gates_script(wc, gates):
    (wc.HARNESS / "components" / "captures" / "verdicts_all.sh").write_text(
        'run() { python3 $PD --diff --framework vllm --version 0.29.0 --save-verdict $OUT/$2.json; }\n'
        + "".join(f"run cap_{g} {g} org/m auto hint\n" for g in gates)
        + "# run cap_x explained_x org/m auto hint\n")


def _verdict(wc, gate, verdict="aligned", fw="vllm", version="0.29.0"):
    d = wc.HARNESS / "results" / "pathdiff" / "sm90" / "vllm-0.29.0"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{gate}.json").write_text(json.dumps({"verdict": verdict, "framework": fw, "version": version}))


def test_review_one_identity_free_verdict_does_not_satisfy_the_declared_gates(wc):
    _gates_script(wc, ["gemm_fp8_M", "attn_ctx_M"])
    _verdict(wc, "gemm_fp8_M")
    ok, reason = wc.pred_path_verdicts_aligned({"fw": "vllm", "version": "0.29.0"})
    assert ok is False and "1/2 declared gates without a verdict" in reason
    # a verdict graded for another framework/version does not count
    _verdict(wc, "attn_ctx_M", fw="sglang", version="0.5.16")
    ok, reason = wc.pred_path_verdicts_aligned({"fw": "vllm", "version": "0.29.0"})
    assert ok is False and "another framework/version" in reason
    _verdict(wc, "attn_ctx_M", verdict="diverged")
    ok, reason = wc.pred_path_verdicts_aligned({"fw": "vllm", "version": "0.29.0"})
    assert ok is False and "attn_ctx_M=diverged" in reason
    _verdict(wc, "attn_ctx_M")
    assert wc.pred_path_verdicts_aligned({"fw": "vllm", "version": "0.29.0"}) == (True, "2 declared gates, all aligned")
    assert wc.declared_gates("vllm", "0.29.0") == {"gemm_fp8_M", "attn_ctx_M"}  # commented lines are not gates


def test_no_declared_gates_is_not_aligned(wc):
    ok, reason = wc.pred_path_verdicts_aligned({"fw": "vllm", "version": "0.29.0"})
    assert ok is False and "no gates declared" in reason


def test_review_implemented_component_never_completes_a_step(wc):
    ok, reason = wc.pred_component_pending({"component": "path_diff"})
    assert ok is False and "no completion predicate" in reason
    ok, reason = wc.pred_component_pending({"component": "sanity"})
    assert ok is False and "not implemented" in reason
    (wc.HARNESS / "workflows" / "w.yaml").write_text(yaml.safe_dump({"workflow": "w", "steps": [
        {"id": "a", "actor": "script", "done_when": {"check": "component_pending", "args": {"component": "path_diff"}}},
        {"id": "b", "actor": "script", "done_when": {"check": "component_pending", "args": {"component": "sanity"}}},
    ]}))
    state = wc.evaluate("w", {})
    assert [(s["id"], s["status"]) for s in state["steps"]] == [("a", "todo"), ("b", "blocked")]
    assert state["all_done"] is False


def test_review_nonexistent_family_shows_nothing_done(wc):
    for pred in ("family_observed", "family_unit_defined", "family_collector_exists", "family_gates_aligned"):
        ok, _ = wc.PREDICATES[pred]({"family": "nonexistent-review-op", "sm": "sm90"})
        assert ok is False, pred
