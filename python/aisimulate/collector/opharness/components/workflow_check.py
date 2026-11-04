#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: workflow step checker. Progress is derived from artifacts, never self-reported.

Each step of a workflow manifest (workflows/<name>.yaml) names one of four checks:

  pin_is                  collector/framework_manifest.yaml pins fw at version
  case_set_no_regression  the CPU plan diff (case_inventory.py) lost nothing skips.py does not explain
  gates_aligned           every declared path_diff gate in scope has an aligned verdict; without a
                          repo/family filter, every registry op is covered by a gate or listed gate-free
  skips_rechecked         rule 7: every collector/skips.py entry in scope was re-checked and kept

and reports done or todo with the reason; the driving loop is "do the first todo, re-run".
Judgment steps without an artifact (triage, review) are prose in the workflow's .md.

Usage:
  AIS_PROBE_WORKSPACE=<ws> python3 workflow_check.py upgrade_op --param fw=sglang --param version=0.5.21 [--json]
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
COLLECTOR = HARNESS.parent


@functools.cache
def _probe_driver_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe_driver_wc", HERE / "probe_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pred_pin_is(p):
    """The manifest pins fw at version: its framework default, or a family pin (a family upgrade)."""
    fw = yaml.safe_load((COLLECTOR / "framework_manifest.yaml").read_text())["frameworks"].get(p["fw"]) or {}
    pins = [str((fw.get("default") or {}).get("version"))]
    pins += [str(f.get("version")) for f in (fw.get("families") or {}).values()]
    if str(p["version"]) in pins:
        return True, f"framework_manifest.yaml pins {p['fw']} {p['version']}"
    return False, f"framework_manifest.yaml pins {p['fw']} at {sorted(set(pins))}, want {p['version']}"


def pred_case_set_no_regression(p):
    """The collector's case plan for (fw, sm) was recorded and diffed against a
    base revision (components/case_inventory.py, CPU only): no op left the plan
    and no case or categorical value left it without a collector/skips.py entry
    accounting for it. Case-set regressions are derivable before any
    collection: job 469988017 lost paged_mqa (op out of plan), the V4-Pro topk
    calibration (a getter's drop) and the sglang mla_context_module table (no
    producer) — all visible in a plan diff."""
    sm = p.get("sm", "sm90")
    inv = HARNESS / "results" / sm / "cases" / f"{p['fw']}-{p['version']}.yaml"
    if not inv.exists():
        return False, f"no case plan ({inv.relative_to(HARNESS)}) — run components/case_inventory.py --base <rev>"
    doc = yaml.safe_load(inv.read_text()) or {}
    meta = doc.get("_meta") or {}
    if meta.get("framework") != p["fw"] or str(meta.get("version")) != str(p["version"]):
        return False, "case plan is for another framework/version"
    if meta.get("dirty"):
        return False, f"case plan was recorded from an uncommitted collector at {meta.get('revision')}"
    if "diff" not in doc:
        return False, "case plan has no diff against a base revision (recorded with --base none)"
    d = doc["diff"]
    base = (meta.get("base") or {}).get("rev")
    if d.get("regressions"):
        return False, f"{len(d['regressions'])} case-set regressions vs {base} — {d['regressions'][0]}"
    return True, (
        f"no case-set regression vs {base} ({len(d.get('additions') or [])} additions, "
        f"{len(d.get('explained') or [])} losses under skips.py)"
    )


def pred_skips_rechecked(p):
    """Rule 7 of the collector overview: at a pin bump every collector/skips.py
    entry in scope for this backend is re-checked, by a source check or a run,
    and kept or deleted. results/<sm>/rechecks/<fw>-<version>.yaml records one
    item per entry (components/case_inventory.py --skips lists them; --param
    ops=a,b narrows a family pin to the entries naming its ops): its key,
    outcome kept|deleted, how source|run, and a ref (framework file:symbol or a
    run id). A kept entry needs its evidence; a deleted one must be gone.
    Deviation notes are comments at the code and are re-checked in review.
    The Kimi-K3 96-head gap sat in a conclusion for two days while the collector
    swept num_heads=96 on B200 and failed 362 cases 'unexpected': a conclusion
    about what a backend cannot run belongs in skips.py, where the plan reads it."""
    sm = p.get("sm", "sm90")
    ci = _case_inventory_module()
    ops = [op for op in str(p.get("ops") or "").split(",") if op]
    in_scope = {ci.skip_key(entry) for entry in ci.skips_in_scope(p["fw"], ops or None)}
    rec_path = HARNESS / "results" / sm / "rechecks" / f"{p['fw']}-{p['version']}.yaml"
    if not rec_path.exists():
        return False, f"{len(in_scope)} skips.py entries in scope, no recheck record ({rec_path.relative_to(HARNESS)})"
    items = {it.get("entry"): it for it in (yaml.safe_load(rec_path.read_text()) or {}).get("skips") or []}
    missing = sorted(in_scope - set(items))
    if missing:
        return False, f"{len(missing)}/{len(in_scope)} skips.py entries not rechecked (e.g. {missing[0][:90]})"
    bad = [
        key
        for key in in_scope
        if items[key].get("outcome") != "kept"
        or items[key].get("how") not in ("source", "run")
        or not items[key].get("ref")
    ]
    if bad:
        return (
            False,
            f"{len(bad)} in-scope entries without kept + how(source|run) + ref, or recorded deleted: {bad[0][:90]}",
        )
    return True, f"all {len(in_scope)} skips.py entries in scope for {p['fw']} rechecked on {p['version']}"


def _case_inventory_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("case_inventory_wc", HERE / "case_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GATES_TABLE = HERE / "gates.yaml"


@functools.cache
def _gates_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gates_wc", HERE / "gates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def declared_gates(fw: str, version: str, sm: str | None = None) -> set:
    """Gate names gates.yaml declares for backend ``fw`` (the version graded is the manifest's;
    ``version`` names the verdict directory). With ``sm``, only gates whose case is planned on
    that SM and that SM is not a ``no_serving`` floor for: the plan decides where a gate exists,
    and a floor gets a platform-floor verdict instead. Coverage checks pass sm=None."""
    rows = _gates_module().gates_in_scope(fw, sm, GATES_TABLE)
    if sm is not None:
        rows = [g for g in rows if sm not in (g.get("no_serving") or {})]
    return {g["gate"] for g in rows}


# Which gate-name families must exist for a backend registry op before the
# op's serving alignment can be called verified. A registry op with no
# mapping is a finding (the table must grow with the registry), an op mapped
# to None is deliberately gate-free with the reason recorded here. Review
# 2026-09-29: rebuilding the vllm gate set for 0.30.0 silently dropped 11 of
# the 32 gates (attention, dsa, encoder, gemm m=1, msa, kda decode) and
# the path check happily reported 19/19 — coverage is now checked
# against the registry, not against whatever the script happens to declare.
_OP_GATE_FAMILIES = {
    "gemm": [r"^gemm_"],
    "compute_scale": None,  # deviation note in collector/vllm/collect_computescale.py; no serving instance to gate
    "attention_context": [r"^attn\w*_ctx"],
    "attention_generation": [r"^attn\w*_gen"],
    "encoder_attention": [r"^encoder_attn"],
    "moe": [r"^moe_"],
    "mla_bmm_gen_pre": [r"^mla_bmm"],
    "mla_bmm_gen_post": [r"^mla_bmm"],
    "mla_context": [r"^mla_ctx"],
    "mla_context_module": [r"^mla_ctx"],
    "mla_generation": [r"^mla_gen"],
    "mla_generation_module": [r"^mla_gen"],
    "dsa_context_module": [r"^dsa_ctx"],
    "dsa_context_module_skip_indexer": None,  # sglang variant of the same path
    "dsa_generation_module": [r"^dsa_gen"],
    "dsa_generation_module_skip_indexer": None,
    "msa_context_module": [r"^msa\w*_ctx"],
    "msa_generation_module": [r"^msa\w*_gen"],
    "dsv4_csa_context_module": [r"^dsv4_csa_ctx"],
    "dsv4_csa_generation_module": [r"^dsv4_csa_gen"],
    "dsv4_csa_topk_calib": None,  # calibration helper, not a serving op
    "dsv4_hca_context_module": [r"^dsv4_hca_ctx"],
    "dsv4_hca_generation_module": [r"^dsv4_hca_gen"],
    "dsv4_paged_mqa_logits_module": [r"^dsv4_paged_mqa"],
    "glm5_dsa_attn_module": [r"^glm5_dsa"],
    "glm5_mqa_logits_module": [r"^glm5_mqa"],
    "glm5_topk_module": [r"^glm5_topk"],
    # mhc_module: the collector measures the no-norm mhc_pre by producer/consumer contract (the SDK bills
    # attn_norm separately); serving fuses the norm -> explained deviation, kept under facts/pathdiff/explained
    "mhc_module": None,
    "gdn": [r"^gdn_ctx", r"^gdn_gen"],
    "kda": [r"^kda_ctx", r"^kda_gen"],
    "mamba2": [r"^mamba2"],
}


def registry_ops(fw: str) -> list:
    """Op names the backend's collector registry declares (text-parsed: the
    harness never imports collector machinery). Wide-EP registries are the
    multi-node plane and are not gated here."""
    reg = HARNESS.parent / fw / "registry.py"
    if not reg.exists():
        return []
    return sorted(set(re.findall(r'op="([a-z0-9_]+)"', reg.read_text())))


def pred_gates_aligned(p):
    """Every gate declared for (fw, version), minus platform floors on this SM, has a verdict
    file that names this framework/version and reads 'aligned'. --param repo= or family=
    narrows the scope to the gates that name the repo or start with the family; without a
    filter, every registry op must also be covered (a rebuilt gate set once kept 19 of 32)."""
    sm, fw, version = p.get("sm", "sm90"), p["fw"], p["version"]
    expected = declared_gates(fw, version, sm)
    if not p.get("repo") and not p.get("family"):
        ops = registry_ops(fw)
        unmapped = [op for op in ops if op not in _OP_GATE_FAMILIES]
        if unmapped:
            return False, f"registry ops without a gate-family mapping: {unmapped[:4]} (extend _OP_GATE_FAMILIES)"
        all_gates = declared_gates(fw, version)
        uncovered = [
            f"{op}:{rx}"
            for op in ops
            for rx in _OP_GATE_FAMILIES[op] or []
            if not any(re.match(rx, g) for g in all_gates)
        ]
        if uncovered:
            return False, f"{len(uncovered)} gate families undeclared for {fw} {version}: {uncovered[:4]}"
    if p.get("family"):
        expected = {g for g in expected if g.startswith(p["family"])}
    if not expected:
        return False, f"no gates declared in scope for {fw} {version} in components/gates.yaml"
    vd = HARNESS / "results" / "pathdiff" / sm / f"{fw}-{version}"
    missing, bad, foreign, named = [], [], [], 0
    for g in sorted(expected):
        f = vd / f"{g}.json"
        if not f.exists():
            missing.append(g)
            continue
        d = json.loads(f.read_text())
        if p.get("repo") and d.get("repo") != p["repo"]:
            continue
        named += 1
        if d.get("framework") != fw or str(d.get("version")) != str(version):
            foreign.append(g)
        elif d.get("verdict") != "aligned":
            bad.append(f"{g}={d.get('verdict')}")
    if missing and not p.get("repo"):
        return False, f"{len(missing)}/{len(expected)} declared gates without a verdict (e.g. {missing[0]})"
    if p.get("repo") and not named:
        return False, f"no gate verdict names {p['repo']} for {fw} {version}"
    if foreign:
        return False, f"verdicts graded for another framework/version: {foreign[:3]}"
    if bad:
        return False, f"not aligned: {bad[:3]}"
    return True, f"{named} gates in scope, all aligned"


PREDICATES = {
    "pin_is": pred_pin_is,
    "case_set_no_regression": pred_case_set_no_regression,
    "gates_aligned": pred_gates_aligned,
    "skips_rechecked": pred_skips_rechecked,
}


def evaluate(workflow: str, params: dict) -> dict:
    params = {"sm": _probe_driver_module().current_sm(), **{k: v for k, v in params.items() if v is not None}}
    manifest = yaml.safe_load((HARNESS / "workflows" / f"{workflow}.yaml").read_text())
    steps = []
    for step in manifest["steps"]:
        check = step["done_when"]["check"]
        try:
            ok, reason = PREDICATES[check]({**params, **(step["done_when"].get("args") or {})})
        except KeyError as e:
            ok, reason = False, f"unknown check or missing param: {e}"
        steps.append(
            {
                "id": step["id"],
                "actor": step.get("actor", "script"),
                "status": "done" if ok else "todo",
                "reason": reason,
            }
        )
    first = next((s["id"] for s in steps if s["status"] == "todo"), None)
    return {"workflow": workflow, "params": params, "steps": steps, "first_todo": first, "all_done": first is None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow", help="manifest name under workflows/ (without .yaml)")
    ap.add_argument("--param", action="append", default=[], help="key=value campaign parameter")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    state = evaluate(args.workflow, dict(kv.split("=", 1) for kv in args.param))
    if args.json:
        print(json.dumps(state, indent=1, ensure_ascii=False))
    else:
        for s in state["steps"]:
            print(f" {'✓' if s['status'] == 'done' else '•'} [{s['actor']:6}] {s['id']:24} {s['reason']}")
        print(f"\n first todo: {state['first_todo'] or '—'}   all done: {state['all_done']}")
    return 0 if state["all_done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
