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
  AIS_PROBE_WORKSPACE=<ws> python3 workflow_check.py upgrade_op \
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
ROOT = Path(os.environ.get("AIS_PROBE_WORKSPACE")
            or os.environ.get("AIC_PROBE_WORKSPACE")  # legacy name
            or Path.cwd())

IMPLEMENTED_COMPONENTS = {"probe_driver", "dummies", "probes", "build_images", "workflow_check", "path_diff",
                          "decompose", "e2e_align"}


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
    """A component being implemented is a PRECONDITION of a step, never its
    completion (review 2026-09-25 P1/P2 #4: four new_op_collector steps read
    'done' for a nonexistent op because their tool existed). Steps that still
    point here are unfinished until they get an artifact predicate."""
    c = p["component"]
    if c in IMPLEMENTED_COMPONENTS:
        return False, f"component {c} implemented, but this step has no completion predicate yet"
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


def _planned_repos(fw: str, version: str) -> set:
    """Repos the plan files declare for (fw, version) — rendered AND rejected
    runs (a generator rejection is still a cell the matrix must show)."""
    repos = set()
    for pf in sorted((ROOT / "archive").glob("plan*.json")):
        try:
            runs = json.loads(pf.read_text())
        except Exception:
            continue
        for r in runs:
            if isinstance(r, dict) and r.get("backend") == fw and str(r.get("version")) == str(version) and r.get("repo"):
                repos.add(r["repo"])
    return repos


def pred_matrix_complete(p):
    """Complete = every repo the PLAN declares for (fw, version) has a cell
    with a verdict. An empty matrix, or one missing planned repos, is not
    complete (review 2026-09-25: '0 cells, all carry verdicts' read as done)."""
    sm = p.get("sm", "sm90")
    expected = _planned_repos(p["fw"], p["version"])
    if not expected:
        return False, f"no plan runs for {p['fw']} {p['version']} (emit queues first)"
    m = _load_matrix(p["fw"], p["version"], sm)
    if m is None:
        return False, f"results/{sm}/{p['fw']}-{p['version']}.yaml missing"
    if str(m.get("_meta", {}).get("version")) != str(p["version"]):
        return False, "matrix _meta.version mismatch"
    cells = m.get("results", {}) or {}
    missing = sorted(expected - set(cells))
    if missing:
        return False, f"{len(missing)} planned repos without a cell (e.g. {missing[0]})"
    bad = [r for r, c in cells.items() if not (c or {}).get("verdict")]
    if bad:
        return False, f"{len(bad)} cells without a verdict (e.g. {bad[0]})"
    stale = [r for r, c in cells.items() if (c or {}).get("cause") == "stale evidence"]
    if stale:
        return False, f"{len(stale)} cells rest on stale evidence (e.g. {stale[0]}) — re-probe"
    return True, f"{len(expected)} planned repos, all cells carry verdicts"


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
    at the new version: results/retests/<sm>/<fw>-<version>.yaml maps each repo to
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
    # SM is a gate dimension (B300 finding 2026-09-20: (fw, version)-keyed
    # evidence let a fresh arch inherit another arch's green checks and let
    # new verdicts overwrite the old arch's files).
    rp = HARNESS / "results" / "retests" / p.get("sm", "sm90") / f"{p['fw']}-{p['version']}.yaml"
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


def declared_gates(fw: str, version: str) -> set:
    """Gate names the verdict scripts (components/captures/verdicts_*.sh)
    declare for this (fw, version): every `run <capture> <gate> ...` line whose
    script grades --framework fw --version version and whose output dir is the
    gate dir (explained deviations write elsewhere and are not gates)."""
    gates = set()
    for sh in sorted((HARNESS / "components" / "captures").glob("verdicts_*.sh")):
        text = sh.read_text()
        if f"--framework {fw} " not in text or f"--version {version} " not in text:
            continue
        for line in text.splitlines():
            m = re.match(r"^run\s+(\S+)\s+(\S+)", line)
            if m:
                gates.add(m.group(2))
    return gates


def pred_path_verdicts_aligned(p):
    """Every gate the verdict scripts declare for (fw, version) has a verdict
    file that names this framework/version and reads 'aligned'. Extra files
    do not count; a missing gate, or a verdict for another (fw, version), is
    incomplete (review 2026-09-25: one identity-free file used to pass)."""
    sm = p.get("sm", "sm90")
    expected = declared_gates(p["fw"], p["version"])
    if not expected:
        return False, f"no gates declared for {p['fw']} {p['version']} in components/captures/verdicts_*.sh"
    vd = HARNESS / "results" / "pathdiff" / sm / f"{p['fw']}-{p['version']}"
    missing, bad, foreign = [], [], []
    for g in sorted(expected):
        f = vd / f"{g}.json"
        if not f.exists():
            missing.append(g)
            continue
        d = json.loads(f.read_text())
        if d.get("framework") != p["fw"] or str(d.get("version")) != str(p["version"]):
            foreign.append(g)
        elif d.get("verdict") != "aligned":
            bad.append(f"{g}={d.get('verdict')}")
    if missing:
        return False, f"{len(missing)}/{len(expected)} declared gates without a verdict (e.g. {missing[0]})"
    if foreign:
        return False, f"verdicts graded for another framework/version: {foreign[:3]}"
    if bad:
        return False, f"not aligned: {bad[:3]}"
    return True, f"{len(expected)} declared gates, all aligned"


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
    # tokenizer artifacts differ per family: HF tokenizer.json, sentencepiece
    # tokenizer.model, or tiktoken (Kimi: tiktoken.model + tokenization_*.py)
    def _has_tokenizer(d: Path) -> bool:
        return any((d / n).exists() for n in ("tokenizer.json", "tokenizer.model", "tiktoken.model")) \
            or ((d / "tokenizer_config.json").exists() and any(d.glob("tokenization_*.py")))
    bare = [d.name for d in dirs if not _has_tokenizer(d)]
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


def _decompositions(repo: str, sm: str) -> dict[str, dict]:
    """{fw-version: decomposition entry} for every pinned backend whose matrix
    cell for the repo PASSES (fail cells have no execution to decompose)."""
    out = {}
    for fw, be in _load_targets()["backends"].items():
        ver = (be.get("versions") or ["?"])[0]
        cell = ((_load_matrix(fw, ver, sm) or {}).get("results") or {}).get(repo) or {}
        if not str(cell.get("verdict", "")).startswith("pass"):
            continue
        p = HARNESS / "results" / sm / "decompose" / f"{fw}-{ver}.yaml"
        entry = ((yaml.safe_load(p.read_text()) or {}).get("results") or {}).get(repo) if p.exists() else None
        out[f"{fw}-{ver}"] = entry
    return out


def pred_model_decomposed(p):
    """components/decompose.py produced an entry for the repo on every pinned
    backend where it passes (results/<sm>/decompose/<fw>-<ver>.yaml)."""
    d = _decompositions(p["repo"], p.get("sm", "sm90"))
    if not d:
        return False, "not evaluable: no passing matrix cell yet (probe first)"
    missing = [k for k, v in d.items() if v is None]
    if missing:
        return False, f"no decomposition on: {', '.join(missing)} (run components/decompose.py)"
    res = sum(len(v.get("residue") or []) for v in d.values())
    return True, f"decomposed on {len(d)} backends, {res} residue kernels"


def pred_residue_dispositioned(p):
    """Owner granularity call: every residue kernel of the repo is named in
    findings (new family / new table / absorb), or there is no residue."""
    d = _decompositions(p["repo"], p.get("sm", "sm90"))
    if not d or any(v is None for v in d.values()):
        return False, "not evaluable: decompose first"
    residue = sorted({k for v in d.values() for k in (v.get("residue") or [])})
    if not residue:
        return True, "no residue — nothing to decide"
    blob = json.dumps(_load_findings(), ensure_ascii=False)
    undecided = [k for k in residue if k not in blob]
    if undecided:
        return False, f"{len(undecided)} residue kernels without a findings decision (e.g. {undecided[0]})"
    return True, f"{len(residue)} residue kernels dispositioned in findings"


def pred_e2e_admitted(p):
    """components/e2e_align.py verdicts exist for the repo on every pinned
    backend where it passes, and all of them are 'aligned'."""
    sm = p.get("sm", "sm90")
    tag = p["repo"].replace("/", "_")
    missing, bad, n = [], [], 0
    for fw, be in _load_targets()["backends"].items():
        ver = (be.get("versions") or ["?"])[0]
        cell = ((_load_matrix(fw, ver, sm) or {}).get("results") or {}).get(p["repo"]) or {}
        if not str(cell.get("verdict", "")).startswith("pass"):
            continue
        files = sorted((HARNESS / "results" / sm / "e2e" / f"{fw}-{ver}").glob(f"{tag}*.json"))
        if not files:
            missing.append(f"{fw}-{ver}")
            continue
        for f in files:
            n += 1
            if json.loads(f.read_text()).get("verdict") != "aligned":
                bad.append(f.name)
    if not missing and n == 0:
        return False, "not evaluable: no passing matrix cell yet"
    if missing:
        return False, f"no e2e verdict on: {', '.join(missing)} (needs a live measurement — see TODO.md)"
    if bad:
        return False, f"e2e verdicts not aligned: {bad[:3]}"
    return True, f"{n} e2e verdicts, all aligned"


_FAMILY_ROLE = {"attention": "attention", "mla": "attention", "msa": "attention", "sparse_attention": "attention",
                "dsa": "dsa_indexer", "gdn": "linear_attention", "kda": "linear_attention",
                "linear_attention": "linear_attention", "gemm": "gemm", "moe": "moe_gemm", "mhc": "mhc",
                "quant": "quant", "quantize": "quant"}


def _pins():
    return [(fw, str((be.get("versions") or ["?"])[0])) for fw, be in _load_targets()["backends"].items()]


def pred_family_observed(p):
    """Serving truth for an op family: some passing record decomposes into
    kernels of the family's role (results/<sm>/decompose/)."""
    sm, fam = p.get("sm", "sm90"), p["family"]
    role = _FAMILY_ROLE.get(fam, fam)
    hits = []
    for fw, ver in _pins():
        f = HARNESS / "results" / sm / "decompose" / f"{fw}-{ver}.yaml"
        if not f.exists():
            continue
        for repo, entry in ((yaml.safe_load(f.read_text()) or {}).get("results") or {}).items():
            if role in (entry.get("families") or {}):
                hits.append(f"{fw}:{repo}")
    if not hits:
        return False, f"no decomposition shows role '{role}' for family '{fam}' (decompose first, or the family is unobserved)"
    return True, f"role '{role}' observed on {len(hits)} (backend, repo) cells"


def pred_family_unit_defined(p):
    fam = p["family"]
    files = sorted((HARNESS.parent / "cases" / "base_ops").glob(f"{fam}*.yaml"))
    return (True, f"{len(files)} base_ops declarations") if files else (False, f"no collector/cases/base_ops/{fam}*.yaml")


def pred_family_collector_exists(p):
    fam = p["family"]
    files = sorted((HARNESS.parent).glob(f"*/collect_{fam}*.py"))
    return (True, f"collectors: {[f.parent.name for f in files]}") if files else (False, f"no collector/*/collect_{fam}*.py")


def pred_family_gates_aligned(p):
    sm, fam = p.get("sm", "sm90"), p["family"]
    seen, bad = 0, []
    for fw, ver in _pins():
        for f in sorted((HARNESS / "results" / "pathdiff" / sm / f"{fw}-{ver}").glob(f"{fam}_*.json")):
            seen += 1
            if json.loads(f.read_text()).get("verdict") != "aligned":
                bad.append(f"{fw}:{f.stem}")
    if not seen:
        return False, f"no path_diff gate named {fam}_* on any pinned backend"
    if bad:
        return False, f"not aligned: {bad[:3]}"
    return True, f"{seen} {fam} gates aligned across pins"


def pred_model_gates_aligned(p):
    """Every path_diff verdict that names this repo (on the pinned backends)
    is aligned, and at least one exists."""
    sm, repo = p.get("sm", "sm90"), p["repo"]
    seen, bad = 0, []
    for fw, ver in _pins():
        for f in sorted((HARNESS / "results" / "pathdiff" / sm / f"{fw}-{ver}").glob("*.json")):
            d = json.loads(f.read_text())
            if d.get("repo") != repo:
                continue
            seen += 1
            if d.get("verdict") != "aligned":
                bad.append(f"{fw}:{f.stem}={d.get('verdict')}")
    if not seen:
        return False, f"no path_diff gate names {repo} on the pinned backends"
    if bad:
        return False, f"not aligned: {bad[:3]}"
    return True, f"{seen} gates name {repo}, all aligned"


PREDICATES = {fn.__name__[5:]: fn for fn in [
    pred_component_pending, pred_pin_is, pred_plan_has_version, pred_path_verdicts_aligned,
    pred_matrix_complete, pred_fails_root_caused, pred_customizations_retested,
    pred_model_inputs_ready, pred_dummies_built, pred_model_probed,
    pred_model_fails_dispositioned, pred_model_decomposed, pred_residue_dispositioned, pred_e2e_admitted,
    pred_family_observed, pred_family_unit_defined, pred_family_collector_exists, pred_family_gates_aligned,
    pred_model_gates_aligned,
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
                "blocked" if check == "component_pending" and args.get("component") not in IMPLEMENTED_COMPONENTS
                else "todo")
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
