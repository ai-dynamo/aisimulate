#!/usr/bin/env python3
"""Component: workflow step checker — the anti-drift loop for workflows.

Progress through a workflow is NEVER self-reported by the agent driving it;
it is derived, every time, from artifacts (workspace files, results, targets
declarations, findings). This checker evaluates the `done_when` predicate of
every step in a workflow manifest (workflows/<name>.yaml) and reports:

  done     the artifact evidence for this step exists and is consistent
  todo     actionable now, evidence missing (with the reason)
  blocked  the step depends on a component that is not implemented yet

The driving loop is then trivial and re-entrant:

  while not workflow_check(...).all_done:
      do the FIRST todo step
      re-run workflow_check          # artifacts decide, not the agent

Every invocation appends one observation line to
results/campaigns/<workflow>__<params>.jsonl (timestamp, git commit, per-step
status) — an append-only ledger of how the campaign actually progressed. The
ledger is audit history only; state is always re-derived, never read back.

Judgment steps (actor: ai/owner) complete by PRODUCING a declared artifact
(a findings entry, a retest record, a signed exclusion); their predicates
check form and completeness — content quality remains review/owner territory
and this tool does not pretend otherwise.

Usage:
  AIC_PROBE_WORKSPACE=<ws> python3 workflow_check.py upgrade_op \
      --param fw=sglang --param version=0.5.17 [--json] [--no-ledger]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
ROOT = Path(os.environ.get("AIC_PROBE_WORKSPACE", Path.cwd()))

IMPLEMENTED_COMPONENTS = {"probe_driver", "dummies", "probes", "build_images", "workflow_check"}


def _load_targets() -> dict:
    return yaml.safe_load((HARNESS / "targets.yaml").read_text())


def _load_findings() -> dict:
    p = HARNESS / "results" / "findings.yaml"
    return (yaml.safe_load(p.read_text()) or {}).get("findings", {}) if p.exists() else {}


def _matrix_path(fw: str, version: str, sm: str) -> Path:
    return HARNESS / "results" / sm / f"{fw}-{version}.yaml"


def _load_matrix(fw: str, version: str, sm: str) -> dict | None:
    p = _matrix_path(fw, version, sm)
    return yaml.safe_load(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------
# predicate registry: name -> fn(params) -> (ok: bool, reason: str)
# Predicates consult artifacts only. Params come from the manifest step's
# `args` merged over the campaign --param values.

def pred_component_pending(p):
    c = p["component"]
    if c in IMPLEMENTED_COMPONENTS:
        return True, f"component {c} implemented"
    return False, f"component {c} not implemented yet"


def pred_pin_is(p):
    be = _load_targets()["backends"].get(p["fw"]) or {}
    vers = be.get("versions") or []
    if vers == [p["version"]]:
        return True, f"targets pin = {p['version']}"
    return False, f"targets pin is {vers}, want [{p['version']}]"


def pred_plan_has_version(p):
    for pf in sorted((ROOT / "archive").glob("plan*.json")):
        runs = json.loads(pf.read_text())
        hits = [r for r in runs if isinstance(r, dict)
                and r.get("backend") == p["fw"] and r.get("version") == p["version"]
                and "skip" not in r]
        if hits:
            return True, f"{pf.name}: {len(hits)} runs at {p['version']}"
    return False, f"no plan file contains {p['fw']} runs at {p['version']}"


def pred_matrix_complete(p):
    m = _load_matrix(p["fw"], p["version"], p.get("sm", "sm90"))
    if m is None:
        return False, f"results/{p.get('sm','sm90')}/{p['fw']}-{p['version']}.yaml missing"
    if str(m.get("_meta", {}).get("version")) != str(p["version"]):
        return False, "matrix _meta.version mismatch"
    bad = [r for r, c in m.get("results", {}).items() if not (c or {}).get("verdict")]
    if bad:
        return False, f"{len(bad)} cells without a verdict (e.g. {bad[0]})"
    return True, f"{len(m['results'])} cells, all carry verdicts"


def pred_fails_root_caused(p):
    """Every fail cell carries a cause, and every repo failing NEWLY (vs the
    previous version's matrix, when one exists) is mentioned in findings."""
    sm = p.get("sm", "sm90")
    m = _load_matrix(p["fw"], p["version"], sm)
    if m is None:
        return False, "matrix missing"
    fails = {r: c for r, c in m["results"].items() if c.get("verdict") == "fail"}
    uncaused = [r for r, c in fails.items() if not c.get("cause")]
    if uncaused:
        return False, f"{len(uncaused)} fail cells without a cause (e.g. {uncaused[0]})"
    prev = sorted(q for q in (HARNESS / "results" / sm).glob(f"{p['fw']}-*.yaml")
                  if q != _matrix_path(p["fw"], p["version"], sm))
    if prev:
        old = yaml.safe_load(prev[-1].read_text())["results"]
        newly = [r for r in fails if (old.get(r) or {}).get("verdict") not in (None, "fail")]
        blob = json.dumps(_load_findings(), ensure_ascii=False)
        missing = [r for r in newly if r not in blob]
        if missing:
            return False, f"{len(missing)} newly-failing repos absent from findings (e.g. {missing[0]})"
    return True, f"{len(fails)} fails, all caused; new fails covered in findings"


def pred_customizations_retested(p):
    """Every per-checkpoint cli_extra_args for this fw needs a retest record
    at the new version: results/retests/<fw>-<version>.yaml maps each repo to
    still_needed|dropped. Produced by the AI step; this checks completeness."""
    t = _load_targets()
    custom = set()
    for fam in t["families"].values():
        for ck in fam.get("checkpoints") or []:
            if p["fw"] in (ck.get("cli_extra_args") or {}):
                custom.add(ck["repo"])
        for repo, o in (fam.get("checkpoint_overrides") or {}).items():
            if p["fw"] in ((o or {}).get("cli_extra_args") or {}):
                custom.add(repo)
    if not custom:
        return True, "no per-checkpoint customizations for this framework"
    rp = HARNESS / "results" / "retests" / f"{p['fw']}-{p['version']}.yaml"
    if not rp.exists():
        return False, f"{len(custom)} customizations, no retest record ({rp.name})"
    rec = yaml.safe_load(rp.read_text()) or {}
    missing = sorted(custom - set(rec))
    badval = [r for r, v in rec.items() if v not in ("still_needed", "dropped")]
    if missing:
        return False, f"retest record missing {len(missing)} repos (e.g. {missing[0]})"
    if badval:
        return False, f"invalid retest outcomes for {badval[:2]}"
    return True, f"all {len(custom)} customizations retested"


def pred_model_inputs_ready(p):
    """configs fetched for the repo, or a signed owner exclusion."""
    repo = p["repo"]
    if (ROOT / "configs" / (repo.replace("/", "_") + ".json")).exists():
        return True, "config fetched"
    for fam in _load_targets()["families"].values():
        for e in fam.get("excluded") or []:
            if e.get("repo") == repo:
                if e.get("decided_by") and e.get("reason"):
                    return True, f"owner-excluded by {e['decided_by']}"
                return False, "exclusion entry lacks decided_by/reason"
    return False, "no fetched config and no signed exclusion — OWNER DECISION NEEDED"


def pred_dummies_built(p):
    name = p["repo"].split("/", 1)[1]
    dirs = list((ROOT / "dummy_models").glob(f"*/{name}__*"))
    if not dirs:
        return False, "no dummy variants"
    bare = [d.name for d in dirs if not (d / "tokenizer.json").exists()]
    if bare:
        return False, f"variants missing tokenizer: {bare[:2]}"
    return True, f"{len(dirs)} variants with tokenizers"


def pred_model_probed(p):
    sm = p.get("sm", "sm90")
    missing = []
    for fw, be in _load_targets()["backends"].items():
        ver = (be.get("versions") or ["?"])[0]
        m = _load_matrix(fw, ver, sm)
        if m is None or p["repo"] not in m.get("results", {}):
            missing.append(f"{fw}-{ver}")
    if missing:
        return False, f"no matrix cell yet on: {', '.join(missing)}"
    return True, "cells present on all pinned backends"


def pred_model_fails_dispositioned(p):
    """Each fail cell for the repo is either root-caused in findings or
    rescued into a cli_extra_args customization (pass+custom)."""
    sm = p.get("sm", "sm90")
    blob = json.dumps(_load_findings(), ensure_ascii=False)
    open_fails, seen = [], 0
    for fw, be in _load_targets()["backends"].items():
        ver = (be.get("versions") or ["?"])[0]
        m = _load_matrix(fw, ver, sm)
        cell = (m or {}).get("results", {}).get(p["repo"]) or {}
        if cell:
            seen += 1
        if cell.get("verdict") == "fail" and p["repo"] not in blob:
            open_fails.append(fw)
    if seen == 0:
        return False, "not evaluable: no matrix cells yet (probe first)"
    if open_fails:
        return False, f"fail cells without findings coverage: {open_fails}"
    return True, "every fail cell is findings-covered (or rescued)"


PREDICATES = {fn.__name__[5:]: fn for fn in [
    pred_component_pending, pred_pin_is, pred_plan_has_version,
    pred_matrix_complete, pred_fails_root_caused, pred_customizations_retested,
    pred_model_inputs_ready, pred_dummies_built, pred_model_probed,
    pred_model_fails_dispositioned,
]}


# --------------------------------------------------------------------------

def evaluate(workflow: str, params: dict) -> dict:
    manifest = yaml.safe_load((HARNESS / "workflows" / f"{workflow}.yaml").read_text())
    steps = []
    for step in manifest["steps"]:
        check = step["done_when"]["check"]
        args = {**params, **(step["done_when"].get("args") or {})}
        try:
            ok, reason = PREDICATES[check](args)
            status = "done" if ok else (
                "blocked" if check == "component_pending" else "todo")
        except KeyError as e:
            status, reason = "todo", f"unknown predicate/param: {e}"
        steps.append({"id": step["id"], "actor": step.get("actor", "script"),
                      "status": status, "reason": reason})
    first = next((s["id"] for s in steps if s["status"] == "todo"), None)
    return {"workflow": workflow, "params": params, "steps": steps,
            "first_todo": first,
            "all_done": all(s["status"] == "done" for s in steps)}


def append_ledger(state: dict) -> Path:
    slug = state["workflow"] + "__" + "_".join(
        f"{k}-{v}" for k, v in sorted(state["params"].items())) or state["workflow"]
    slug = re.sub(r"[^\w.\-]+", "-", slug)
    led = HARNESS / "results" / "campaigns" / f"{slug}.jsonl"
    led.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(HARNESS), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    with led.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit,
                            "status": {s["id"]: s["status"] for s in state["steps"]},
                            "first_todo": state["first_todo"]}) + "\n")
    return led


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow", help="manifest name under workflows/ (without .yaml)")
    ap.add_argument("--param", action="append", default=[], help="key=value campaign parameter")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    params = dict(kv.split("=", 1) for kv in args.param)
    state = evaluate(args.workflow, params)
    if not args.no_ledger:
        append_ledger(state)
    if args.json:
        print(json.dumps(state, indent=1, ensure_ascii=False))
    else:
        for s in state["steps"]:
            mark = {"done": "✓", "todo": "•", "blocked": "▧"}[s["status"]]
            print(f" {mark} [{s['actor']:6}] {s['id']:28} {s['reason']}")
        print(f"\n first todo: {state['first_todo'] or '—'}   all done: {state['all_done']}")
    return 0 if state["all_done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
