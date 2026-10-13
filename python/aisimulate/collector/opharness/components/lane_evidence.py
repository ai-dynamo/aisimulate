#!/usr/bin/env python3
"""Component: grade a collector's version/SM lane guard against identity evidence.

Why this exists (b200_sxm sglang 0.5.21, GitLab job 469988017, 2026-10-05):
moe/int4_wo returned 0/3,078 rows because collect_moe's lane guard refused
0.5.21 on SM100 from a SOURCE reading ("auto moved to Triton"), while the
harness's own B200 identity records of the same day showed Kimi-K2.5 serving
on trtllm_gen_moe / FLASHINFER_TRTLLM — exactly the backend the cases declare.
The evidence existed; nothing read it when the guard was edited (on an SM90
box). This component makes that reading mechanical: lane_evidence.yaml says
which checkpoints' serving identity confirms a lane on an SM, the guard is
evaluated for (version, SM) the way the executor would call it (AST-extracted,
no framework import), and every cell is graded consistent / inconsistent.
workflow_check's `lane_guards_match_evidence` predicate consumes the grading.

Usage:
  python3 lane_evidence.py --fw sglang --version 0.5.21 --sm sm100 [--json]
  exit 0 = every cell consistent, 1 = at least one inconsistent cell
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import types
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
COLLECTOR = HARNESS.parent  # python/aisimulate/collector
RULES_PATH = HERE / "lane_evidence.yaml"

_EVIDENCE_FIELDS = ("moe",)                       # results/<sm>/<fw>-<ver>.yaml cells
_RETEST_FIELDS = ("moe_runner", "moe_kernel_families")  # results/retests/<sm>/<fw>-<ver>.moe_auto.yaml cells


def load_rules(path: Path = RULES_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _check_compat():
    """collector.version_resolver._check_compat loaded by file path (packaging
    grammar, no collector package import side effects)."""
    spec = importlib.util.spec_from_file_location("ais_version_resolver_le", COLLECTOR / "version_resolver.py")
    mod = importlib.util.module_from_spec(spec)
    import sys

    # version_resolver imports collector.registry_types; make the parent importable the way tests do
    sys.path.insert(0, str(COLLECTOR.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod._check_compat


def load_guard(source: Path, function: str, *, installed_version: str, sm_version: int):
    """Extract the guard function from source and bind it to (version, SM) —
    the same technique as tests/unit/collector/sglang/test_collect_moe_unverified_lane_guard.py,
    so the grading never needs the framework installed."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function), None)
    if node is None:
        raise ValueError(f"{source}: no function {function}")
    fake_distribution = types.SimpleNamespace(version=installed_version)
    namespace = {
        "pkg_resources": types.SimpleNamespace(get_distribution=lambda _name: fake_distribution),
        "_check_compat": _check_compat(),
        "_dist_version": lambda _name: installed_version,
        "get_sm_version": lambda: sm_version,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function]


def guard_state(guard, lane: str) -> str:
    try:
        guard(lane)
    except RuntimeError:
        return "closed"
    return "open"


def _evidence_strings(harness: Path, fw: str, version: str, sm: str, repo: str) -> list[str]:
    out: list[str] = []
    matrix = harness / "results" / sm / f"{fw}-{version}.yaml"
    if matrix.exists():
        cell = ((yaml.safe_load(matrix.read_text()) or {}).get("results") or {}).get(repo) or {}
        if cell.get("verdict", "").startswith("pass"):
            # "Module->kernel_family" is evidence; a bare module name (older records,
            # e.g. sm90 "Mxfp4MoE") names no backend and must not read as a contradiction
            out += [str(cell[f]) for f in _EVIDENCE_FIELDS if cell.get(f) and "->" in str(cell[f])]
    retest = harness / "results" / "retests" / sm / f"{fw}-{version}.moe_auto.yaml"
    if retest.exists():
        cell = ((yaml.safe_load(retest.read_text()) or {}).get("results") or {}).get(repo) or {}
        if str(cell.get("verdict", "")).startswith("pass"):
            for f in _RETEST_FIELDS:
                v = cell.get(f)
                if isinstance(v, list):
                    out += [str(x) for x in v]
                elif v:
                    out.append(str(v))
    return out


def evidence_state(strings: list[str], serving_regex: str | None) -> str:
    """match / contradict / absent for one (lane, SM) given its evidence strings."""
    if not serving_regex or not strings:
        return "absent"
    rx = re.compile(serving_regex, re.IGNORECASE)
    return "match" if any(rx.search(s) for s in strings) else "contradict"


def grade(guard_state_: str, evidence_state_: str) -> tuple[bool, str]:
    if guard_state_ == "open":
        if evidence_state_ == "match":
            return True, "open, serving confirms the declared backend"
        if evidence_state_ == "contradict":
            return False, "OPEN but serving picks another backend — wrong-kernel data risk"
        return False, "OPEN without hardware evidence — probe the lane's models or close the guard"
    if evidence_state_ == "match":
        return False, ("CLOSED although serving confirms the declared backend — rows are being refused "
                       "(int4_wo/SM100 class)")
    if evidence_state_ == "contradict":
        return True, "closed, serving contradicts the declaration"
    return True, "closed, no evidence"


def evaluate(fw: str, version: str, sm: str, *, harness: Path = HARNESS, rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    sm_version = int(str(sm).replace("sm", ""))
    cells = []
    for guard_name, g in (rules.get("guards") or {}).items():
        if not guard_name.startswith(f"{fw}."):
            continue
        # guard sources are code (python/aisimulate/<source>), results are evidence (harness/results/)
        source = Path(g["source"]) if Path(g["source"]).is_absolute() else COLLECTOR.parent / g["source"]
        guard = load_guard(source, g["function"], installed_version=version, sm_version=sm_version)
        for lane, spec in (g.get("lanes") or {}).items():
            rule = ((spec.get("by_sm") or {}).get(sm)) or {}
            strings: list[str] = []
            models_seen = []
            for repo in spec.get("models") or []:
                s = _evidence_strings(harness, fw, version, sm, repo)
                if s:
                    models_seen.append(repo)
                strings += s
            gs = guard_state(guard, lane)
            es = evidence_state(strings, rule.get("serving"))
            ok, why = grade(gs, es)
            cells.append({"guard": guard_name, "lane": lane, "sm": sm, "guard_state": gs, "evidence": es,
                          "declared": rule.get("declared"), "models_with_evidence": models_seen,
                          "evidence_strings": sorted(set(strings)), "consistent": ok, "why": why})
    return {"fw": fw, "version": version, "sm": sm, "cells": cells,
            "consistent": all(c["consistent"] for c in cells) and bool(cells)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--sm", required=True, help="sm90 / sm100 / ...")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    state = evaluate(args.fw, args.version, args.sm)
    if args.json:
        print(json.dumps(state, indent=1))
    else:
        for c in state["cells"]:
            mark = "✓" if c["consistent"] else "✗"
            print(f" {mark} {c['guard']}:{c['lane']:18} {c['sm']:6} guard={c['guard_state']:6} "
                  f"evidence={c['evidence']:10} declared={c['declared']}  {c['why']}")
            if c["evidence_strings"]:
                print(f"      evidence: {', '.join(c['evidence_strings'])}  ({', '.join(c['models_with_evidence'])})")
        print(f"\n consistent: {state['consistent']}")
    return 0 if state["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
