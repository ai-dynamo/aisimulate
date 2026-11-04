#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: workflow step checker — the anti-drift loop for workflows.

Progress through a workflow is NEVER self-reported by the agent driving it;
it is derived, every time, from artifacts (workspace files, results, targets
declarations, the collector's plan and skips.py, committed evidence). This
checker evaluates the `done_when` predicate of every step in a workflow
manifest (workflows/<name>.yaml) and reports:

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
(a skips.py entry, an evidence README, a retest or recheck record); their predicates
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
COLLECTOR = HARNESS.parent
ROOT = Path(
    os.environ.get("AIS_PROBE_WORKSPACE")
    or os.environ.get("AIC_PROBE_WORKSPACE")  # legacy name
    or Path.cwd()
)

IMPLEMENTED_COMPONENTS = {
    "probe_driver",
    "dummies",
    "probes",
    "workflow_check",
    "path_diff",
    "op_smoke",
    "case_inventory",
}


def _load_targets() -> dict:
    return yaml.safe_load((HARNESS / "targets.yaml").read_text())


def _conclusions_text() -> str:
    """Where a conclusion lives once drawn (V4: no findings file): collector/skips.py
    (a model the backend cannot serve, a block that cannot run), the collector source
    (a fix, a FIXME(kernel-limit), a deviation note at the deviating code) and the
    committed evidence READMEs under benchmarks/evidence/collector/."""
    own = COLLECTOR / "opharness"  # the harness's own sources are not where a conclusion lives
    parts = [path.read_text(errors="replace") for path in sorted(COLLECTOR.glob("**/*.py")) if own not in path.parents]
    evidence = COLLECTOR.parents[2] / "benchmarks" / "evidence" / "collector"
    parts += [path.read_text(errors="replace") for path in sorted(evidence.glob("*/README.md"))]
    return "\n".join(parts)


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
        hits = [
            r
            for r in runs
            if isinstance(r, dict)
            and r.get("backend") == p["fw"]
            and r.get("version") == p["version"]
            and "skip" not in r
        ]
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
            if (
                isinstance(r, dict)
                and r.get("backend") == fw
                and str(r.get("version")) == str(version)
                and r.get("repo")
            ):
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
    previous version's matrix, when one exists) is named where its conclusion
    lives (_conclusions_text: skips.py, the collector source, an evidence README)."""
    sm = p.get("sm", "sm90")
    m = _load_matrix(p["fw"], p["version"], sm)
    if m is None:
        return False, "matrix missing"
    fails = {r: c for r, c in m["results"].items() if c.get("verdict") == "fail"}
    uncaused = [r for r, c in fails.items() if not c.get("cause")]
    if uncaused:
        return False, f"{len(uncaused)} fail cells without a cause (e.g. {uncaused[0]})"
    prev = sorted(
        q for q in (HARNESS / "results" / sm).glob(f"{p['fw']}-*.yaml") if q != _matrix_path(p["fw"], p["version"], sm)
    )
    if prev:
        old = yaml.safe_load(prev[-1].read_text())["results"]
        newly = [r for r in fails if (old.get(r) or {}).get("verdict") not in (None, "fail")]
    else:
        # first campaign on this SM (no baseline matrix): every fail that is not a
        # generator reject is new here and must be explained (RTX 6000 handoff
        # 2026-09-30: the check used to go green on 24 unexplained fails)
        newly = [r for r, c in fails.items() if "generator" not in str(c.get("cause") or "")]
    blob = _conclusions_text()
    missing = [r for r in newly if r not in blob]
    if missing:
        return False, (
            f"{len(missing)} newly-failing repos named in no skips.py entry, collector note or evidence README "
            f"(e.g. {missing[0]})"
        )
    return True, f"{len(fails)} fails, all caused; {len(newly)} new fails explained"


def pred_customizations_retested(p):
    """Every per-checkpoint cli_extra_args for this fw needs a retest record
    at the new version: results/retests/<sm>/<fw>-<version>.yaml maps each repo to
    still_needed|dropped. Produced by the AI step; this checks completeness."""
    t = _load_targets()
    sm = p.get("sm", "sm90")

    def _active(entry) -> bool:  # an entry scoped with `sms:` counts only on those SMs
        e = (entry or {}).get(p["fw"])
        if e is None:
            return False
        return not (isinstance(e, dict) and e.get("sms")) or sm in [str(x) for x in e["sms"]]

    custom = set()
    for fam in t["families"].values():
        for ck in fam.get("checkpoints") or []:
            if _active(ck.get("cli_extra_args")):
                custom.add(ck["repo"])
        for repo, o in (fam.get("checkpoint_overrides") or {}).items():
            if _active((o or {}).get("cli_extra_args")):
                custom.add(repo)
    if not custom:
        return True, "no per-checkpoint customizations for this framework on this SM"
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


# --------------------------------------------------------------------------
# Evidence-flow predicates (b200_sxm sglang 0.5.21 shard run, GitLab job
# 469988017, 2026-10-05: 7 of 8 failed shards were harness knowledge that never
# reached the place it was needed — a guard edited against the identity
# records, findings that named a framework gap no collector grid guard ever
# learned, a smoke that never passed through collect.py's plan/resume/finalize).


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


_GATE_LINE = re.compile(r'^(?:(?:SERVING_RAW|FLOOR_SM|FLOOR_NOTE)=(?:"[^"]*"|\S+)\s+)*run\s+(\S+)\s+(\S+)')


def declared_gates(fw: str, version: str, sm: str | None = None) -> set:
    """Gate names the verdict scripts (components/captures/verdicts_*.sh)
    declare for this (fw, version): every `run <capture> <gate> ...` line whose
    script grades --framework fw --version version and whose output dir is the
    gate dir (explained deviations write elsewhere and are not gates). Leading
    SERVING_RAW= / FLOOR_SM= / FLOOR_NOTE= assignments still declare a gate;
    with ``sm`` given, gates declared as a platform floor FOR THAT SM are left
    out (they have no serving instance there by framework fact and carry a
    platform-floor verdict file instead) — coverage checks pass sm=None."""
    gates = set()
    for sh in sorted((HARNESS / "components" / "captures").glob("verdicts_*.sh")):
        text = sh.read_text()
        if f"--framework {fw} " not in text or f"--version {version} " not in text:
            continue
        for line in text.splitlines():
            m = _GATE_LINE.match(line)
            if not m:
                continue
            floor = re.match(r"^FLOOR_SM=(\S+)", line)
            if sm is not None and floor and sm in floor.group(1).split(","):  # FLOOR_SM=sm120,sm89
                continue
            gates.add(m.group(2))
    return gates


# Which gate-name families must exist for a backend registry op before the
# op's serving alignment can be called verified. A registry op with no
# mapping is a finding (the table must grow with the registry), an op mapped
# to None is deliberately gate-free with the reason recorded here. Review
# 2026-09-29: rebuilding the vllm gate set for 0.30.0 silently dropped 11 of
# the 32 gates (attention, dsa, encoder, gemm m=1, msa, kda decode) and
# path_verdicts_aligned happily reported 19/19 — coverage is now checked
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


def pred_gates_cover_registry_ops(p):
    """Every op the backend registry declares has at least one gate of each
    of its gate families declared for (fw, version) — or is listed gate-free
    with a reason in _OP_GATE_FAMILIES. Catches a rebuilt gate set that
    quietly lost families (the 0.30.0 rebuild kept 19 of 32)."""
    fw, version = p["fw"], p["version"]
    ops = registry_ops(fw)
    if not ops:
        return False, f"no collector registry found for {fw}"
    gates = declared_gates(fw, version)
    unmapped, uncovered = [], []
    for op in ops:
        if op not in _OP_GATE_FAMILIES:
            unmapped.append(op)
            continue
        fams = _OP_GATE_FAMILIES[op]
        if fams is None:
            continue
        for rx in fams:
            if not any(re.match(rx, g) for g in gates):
                uncovered.append(f"{op}:{rx}")
    if unmapped:
        return False, f"registry ops without a gate-family mapping: {unmapped[:4]} (extend _OP_GATE_FAMILIES)"
    if uncovered:
        return False, f"{len(uncovered)} gate families undeclared for {fw} {version}: {uncovered[:4]}"
    return True, f"{len(gates)} gates cover all {len(ops)} registry ops"


def pred_path_verdicts_aligned(p):
    """Every gate the verdict scripts declare for (fw, version) has a verdict
    file that names this framework/version and reads 'aligned'. Extra files
    do not count; a missing gate, or a verdict for another (fw, version), is
    incomplete (review 2026-09-25: one identity-free file used to pass)."""
    sm = p.get("sm", "sm90")
    expected = declared_gates(p["fw"], p["version"], sm)
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
        return any((d / n).exists() for n in ("tokenizer.json", "tokenizer.model", "tiktoken.model")) or (
            (d / "tokenizer_config.json").exists() and any(d.glob("tokenization_*.py"))
        )

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
    """Each fail cell for the repo is either explained where conclusions live
    (_conclusions_text) or rescued into a cli_extra_args customization (pass+custom)."""
    sm = p.get("sm", "sm90")
    blob = _conclusions_text()
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
        return False, f"fail cells explained nowhere (skips.py, collector, evidence README): {open_fails}"
    return True, "every fail cell is explained (or rescued)"


_FAMILY_ROLE = {
    "attention": "attention",
    "mla": "attention",
    "msa": "attention",
    "sparse_attention": "attention",
    "dsa": "dsa_indexer",
    "gdn": "linear_attention",
    "kda": "linear_attention",
    "linear_attention": "linear_attention",
    "gemm": "gemm",
    "moe": "moe_gemm",
    "mhc": "mhc",
    "quant": "quant",
    "quantize": "quant",
}


def _pins():
    return [(fw, str((be.get("versions") or ["?"])[0])) for fw, be in _load_targets()["backends"].items()]


def pred_family_observed(p):
    """Serving truth for an op family: a probe record on a pinned backend holds a
    kernel the taxonomy labels with the family's role (archive/records.jsonl)."""
    fam = p["family"]
    role = _FAMILY_ROLE.get(fam, fam)
    recs = ROOT / "archive" / "records.jsonl"
    if not recs.exists():
        return False, "no probe records (run probe_driver.py --records)"
    pd = _probe_driver_module()
    pins = set(_pins())
    hits = set()
    for line in recs.open():
        r = json.loads(line)
        rt, tgt = r.get("runtime") or {}, r.get("target") or {}
        if (rt.get("backend"), str(rt.get("version"))) not in pins:
            continue
        kernels = {k for op in r.get("ops") or [] for k in op.get("kernels") or []}
        if any((pd.kernel_role(k) or (None, None))[1] == role for k in kernels):
            hits.add(f"{rt['backend']}:{tgt.get('repo')}")
    if not hits:
        return False, f"no probe record shows role '{role}' for family '{fam}' (probe first, or it is unobserved)"
    return True, f"role '{role}' observed on {len(hits)} (backend, repo) cells"


def _probe_driver_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe_driver_wc", HERE / "probe_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pred_family_unit_defined(p):
    fam = p["family"]
    files = sorted((HARNESS.parent / "cases" / "base_ops").glob(f"{fam}*.yaml"))
    return (
        (True, f"{len(files)} base_ops declarations") if files else (False, f"no collector/cases/base_ops/{fam}*.yaml")
    )


def pred_family_collector_exists(p):
    fam = p["family"]
    files = sorted((HARNESS.parent).glob(f"*/collect_{fam}*.py"))
    return (
        (True, f"collectors: {[f.parent.name for f in files]}")
        if files
        else (False, f"no collector/*/collect_{fam}*.py")
    )


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


PREDICATES = {
    fn.__name__[5:]: fn
    for fn in [
        pred_component_pending,
        pred_pin_is,
        pred_plan_has_version,
        pred_gates_cover_registry_ops,
        pred_path_verdicts_aligned,
        pred_matrix_complete,
        pred_fails_root_caused,
        pred_customizations_retested,
        pred_model_inputs_ready,
        pred_dummies_built,
        pred_model_probed,
        pred_model_fails_dispositioned,
        pred_family_observed,
        pred_family_unit_defined,
        pred_family_collector_exists,
        pred_family_gates_aligned,
        pred_model_gates_aligned,
        pred_skips_rechecked,
        pred_case_set_no_regression,
    ]
}


# --------------------------------------------------------------------------


def default_sm() -> str:
    """AIS_SM if exported, else targets.yaml platform.sm, else sm90 — the same
    rule as probe_driver.current_sm (a forgotten export must not silently
    read another SM's results)."""
    env = os.environ.get("AIS_SM")
    if env:
        return env
    try:
        plat = _load_targets().get("platform") or {}
        if plat.get("sm"):
            return f"sm{int(plat['sm'])}"
    except Exception:
        pass
    return "sm90"


def evaluate(workflow: str, params: dict) -> dict:
    params = {"sm": default_sm(), **{k: v for k, v in params.items() if v is not None}}
    manifest = yaml.safe_load((HARNESS / "workflows" / f"{workflow}.yaml").read_text())
    steps = []
    for step in manifest["steps"]:
        check = step["done_when"]["check"]
        args = {**params, **(step["done_when"].get("args") or {})}
        try:
            ok, reason = PREDICATES[check](args)
            status = (
                "done"
                if ok
                else (
                    "blocked"
                    if check == "component_pending" and args.get("component") not in IMPLEMENTED_COMPONENTS
                    else "todo"
                )
            )
        except KeyError as e:
            status, reason = "todo", f"unknown predicate/param: {e}"
        steps.append({"id": step["id"], "actor": step.get("actor", "script"), "status": status, "reason": reason})
    first = next((s["id"] for s in steps if s["status"] == "todo"), None)
    return {
        "workflow": workflow,
        "params": params,
        "steps": steps,
        "first_todo": first,
        "all_done": all(s["status"] == "done" for s in steps),
    }


def append_ledger(state: dict) -> Path:
    slug = (
        state["workflow"] + "__" + "_".join(f"{k}-{v}" for k, v in sorted(state["params"].items())) or state["workflow"]
    )
    slug = re.sub(r"[^\w.\-]+", "-", slug)
    led = HARNESS / "results" / "campaigns" / f"{slug}.jsonl"
    led.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "-C", str(HARNESS), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    with led.open("a") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "commit": commit,
                    "status": {s["id"]: s["status"] for s in state["steps"]},
                    "first_todo": state["first_todo"],
                }
            )
            + "\n"
        )
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
