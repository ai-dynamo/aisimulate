#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: workflow step checker. Progress is derived from artifacts, never self-reported.

Each checked step of a workflow (WORKFLOWS) names one of four checks:

  pin_is                  collector/framework_manifest.yaml pins fw at version
  case_set_no_regression  the CPU plan diff (case_inventory.py) lost nothing skips.py does not explain
  gates_aligned           every declared path_diff gate in scope has an aligned verdict; without a
                          repo/family filter, every registry op is covered by a gate or listed gate-free
  skips_rechecked         rule 7: every collector/skips.py entry in scope was re-checked and kept

and reports done or todo with the reason; the driving loop is "do the first todo, re-run".
Judgment steps without an artifact (triage, review) are prose in the workflow's .md.

Usage:
  python3 workflow_check.py upgrade_op --param fw=sglang --param version=0.5.21 --param sm=sm100 [--json]

The artifacts are the probe workspace's results/ on the control machine (probe_driver.ROOT).
"""

from __future__ import annotations

import argparse
import functools
import json
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


def _results() -> Path:
    return _probe_driver_module().ROOT / "results"


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
    inv = _results() / p["sm"] / "cases" / f"{p['fw']}-{p['version']}.yaml"
    if not inv.exists():
        return False, f"no case plan ({inv}) — run components/case_inventory.py --base <rev>"
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
    ci = _case_inventory_module()
    ops = [op for op in str(p.get("ops") or "").split(",") if op]
    in_scope = {ci.skip_key(entry) for entry in ci.skips_in_scope(p["fw"], ops or None)}
    rec_path = _results() / p["sm"] / "rechecks" / f"{p['fw']}-{p['version']}.yaml"
    if not rec_path.exists():
        return False, f"{len(in_scope)} skips.py entries in scope, no recheck record ({rec_path})"
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


def declared_gates(fw: str, version: str, sm: str | None = None, op: str | None = None) -> set:
    """Gate names gates.yaml declares for backend ``fw`` at ``version``: a gate runs at the
    release its op is collected with (gates.gate_version), so a family pinned apart from the
    default has its gates under its own version. With ``op``, only that op's gates. With
    ``sm``, only gates whose case is planned on that SM and that SM is not a ``no_serving``
    floor for: the plan decides where a gate exists, and a floor gets a platform-floor verdict
    instead. Coverage checks pass sm=None."""
    gm = _gates_module()
    rows = [g for g in gm.gates_in_scope(fw, sm, GATES_TABLE, op=op) if gm.gate_version(g) == str(version)]
    if sm is not None:
        rows = [g for g in rows if sm not in (g.get("no_serving") or {})]
    return {g["gate"] for g in rows}


# Registry ops (or op:phase) no gate can prove, each with its reason. Every other registry op
# needs a gates.yaml row naming it (`op`, or an `also` case), and an op whose cases carry a
# `phase` needs one per planned phase: coverage is checked against the registry, not the
# table (review 2026-09-29: a rebuilt vllm gate set kept 19 of 32 gates, kda decode among
# the lost, and the check reported 19/19).
UNGATED = {
    "compute_scale": "deviation note in collector/vllm/collect_computescale.py; no serving instance to gate",
    "mhc_module": "measures the no-norm mhc_pre by contract (the SDK bills attn_norm); serving fuses the norm",
    "dsa_context_module_skip_indexer": "the sglang dsa_context_module path without the indexer",
    "dsa_generation_module_skip_indexer": "the sglang dsa_generation_module path without the indexer",
    "dsv4_csa_topk_calib": "calibration helper, not a serving op",
    "dsv41_module": "the native runner times TP ranks in spawned processes; probe.trace_case refuses it",
    "kda:verify": "the serving probe runs no speculative decoding",
}


def registry_ops(fw: str) -> list:
    """Op names the backend's collector registry declares (the wide-EP registries are the
    multi-node plane and are not gated here)."""
    import importlib
    import sys

    sys.path.insert(0, str(COLLECTOR.parent))
    try:
        registry = importlib.import_module(f"collector.{fw}.registry").REGISTRY
    except ModuleNotFoundError:
        return []
    return sorted({entry.op for entry in registry})


def _phases(op: str, fw: str) -> list:
    """The `phase` values an op plans (any SM), or [] when its cases carry none."""
    import contextlib
    import os

    from collector.op_plans import OP_PLANS, plan

    if op not in OP_PLANS or "phase" not in OP_PLANS[op].case_type._fields:
        return []
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        return sorted({case.phase for case in plan(op, fw, None).cases})


def uncovered(fw: str, ops: list, gates: list) -> list:
    """Registry ops (op or op:phase) of ``fw`` that no gate names and UNGATED does not list."""
    named = set()
    for g in gates:
        for op, select in [(g["op"], g["select"]), *((a["op"], a["select"]) for a in g.get("also") or [])]:
            named.add(op)
            if "phase" in select:
                named.add(f"{op}:{select['phase']}")
    missing = []
    for op in ops:
        if op in UNGATED:
            continue
        if op not in named:
            missing.append(op)
            continue
        missing += [f"{op}:{ph}" for ph in _phases(op, fw) if f"{op}:{ph}" not in named | set(UNGATED)]
    return missing


def pred_gates_aligned(p):
    """Every gate declared for (fw, version), minus platform floors on this SM, has a verdict
    file that names this framework/version and reads 'aligned'. --param op=, repo= or family=
    narrows the scope to one op's gates (an upgrade done op by op), the gates that name the
    repo, or those starting with the family; without a filter, every registry op must also be
    covered (a rebuilt gate set once kept 19 of 32)."""
    sm, fw, version = p["sm"], p["fw"], p["version"]
    expected = declared_gates(fw, version, sm, p.get("op"))
    if not p.get("repo") and not p.get("family") and not p.get("op"):
        gaps = uncovered(fw, registry_ops(fw), _gates_module().gates_in_scope(fw, None, GATES_TABLE))
        if gaps:
            return False, f"registry ops no gate names (add a gates.yaml row or an UNGATED reason): {gaps[:4]}"
    if p.get("family"):
        expected = {g for g in expected if g.startswith(p["family"])}
    if not expected:
        return False, f"no gates declared in scope for {fw} {version} in components/gates.yaml"
    vd = _results() / "pathdiff" / sm / f"{fw}-{version}"
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


# The checked steps of each workflow (workflows/<name>.md holds the full step table):
# (step id, who does it, the check that says it is done).
WORKFLOWS = {
    "upgrade_op": [
        ("pin_bumped", "owner", "pin_is"),
        ("case_set_no_regression", "script", "case_set_no_regression"),
        ("path_aligned", "script", "gates_aligned"),
        ("skips_rechecked", "ai", "skips_rechecked"),
    ],
    "onboard_model": [
        ("case_set_no_regression", "script", "case_set_no_regression"),
        ("path_aligned", "script", "gates_aligned"),
    ],
    "new_op_collector": [
        ("case_set_no_regression", "script", "case_set_no_regression"),
        ("path_aligned", "script", "gates_aligned"),
    ],
}


def evaluate(workflow: str, params: dict) -> dict:
    params = {k: v for k, v in params.items() if v is not None}
    steps = []
    for step_id, actor, check in WORKFLOWS[workflow]:
        try:
            ok, reason = PREDICATES[check](params)
        except KeyError as e:
            ok, reason = False, f"unknown check or missing param: {e}"
        steps.append({"id": step_id, "actor": actor, "status": "done" if ok else "todo", "reason": reason})
    first = next((s["id"] for s in steps if s["status"] == "todo"), None)
    return {"workflow": workflow, "params": params, "steps": steps, "first_todo": first, "all_done": first is None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow", choices=sorted(WORKFLOWS))
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
