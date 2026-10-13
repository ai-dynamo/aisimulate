#!/usr/bin/env python3
"""Component: the collector's TEST-CASE INVENTORY for (fw, version, sm), and its diff.

A collector op's case set is a pure function of (SM, framework version, case
files, collector code): every registry op's get_func, filtered by
cases/capabilities.yaml. It can therefore be enumerated WITHOUT collecting —
CPU only, inside the framework image (getters import the framework),
AIS_SM=<target> standing in for the GPU. Diffing two inventories is the
"test-case regression" check the harness lacked: the b200_sxm sglang 0.5.21
shard run (job 469988017) lost dsv4_paged_mqa_logits_module (op not in the
plan), the DeepSeek-V4-Pro topk calibration (getter's default-plan drop) and
the sglang mla_context_module table (no producer at all) — all visible in the
case inventory, none needing a GPU hour to discover.

Output (committed): results/<sm>/cases/<fw>-<version>.yaml
  _meta  : fw / version / sm / harness commit / counts
  ops    : per registry op — in_plan, cases (count), dropped_by_capabilities,
           fields: per positional field of the case tuple, the value set when
           it is small (categorical: dtypes, backends, models, lanes) or
           min/max/distinct when it is numeric (shapes)
  full case lists go to <AIS_PROBE_WORKSPACE>/archive/cases/<fw>-<version>-<sm>.jsonl
  (evidence, not committed).

Usage (inside the framework image, CPU is enough):
  AIS_SM=sm100 python3 case_inventory.py --fw sglang --version 0.5.21 --sm sm100 [--ops a,b]
  python3 case_inventory.py --diff results/sm100/cases/sglang-0.5.17.yaml results/sm100/cases/sglang-0.5.21.yaml
Waivers (owner-signed, next to the inventory): results/<sm>/cases/<fw>-<version>.waivers.yaml
  ops:    {<op>: <reason>}                      # op retired / moved on purpose
  fields: {<op>: {<field index>: <reason>}}     # a categorical value dropped on purpose
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
COLLECTOR = HARNESS.parent
PKG_ROOT = COLLECTOR.parent
CATEGORICAL_MAX = 64


def _version_key(v: str):
    return tuple(int(x) if x.isdigit() else x for x in re.findall(r"\d+|[a-zA-Z]+", str(v)))


def _case_params(case):
    if isinstance(case, dict) and "params" in case:  # collect.py task shape {"id", "params"}
        return case["params"]
    return case


def _field_values(case):
    p = _case_params(case)
    if isinstance(p, dict):
        return [(k, v) for k, v in p.items()]
    if isinstance(p, (list, tuple)):
        return [(i, v) for i, v in enumerate(p)]
    return [(0, p)]


def _repr(v) -> str:
    s = str(v)
    return s if len(s) <= 200 else hashlib.sha256(s.encode()).hexdigest()[:16]


def summarize_cases(cases: list) -> dict:
    """Per-field summary of a case list: value sets for categorical fields,
    ranges for numeric ones. Field keys are positions (legacy tuples) or names."""
    fields: dict = {}
    for case in cases:
        for key, value in _field_values(case):
            slot = fields.setdefault(str(key), {"values": set(), "numeric": True, "min": None, "max": None})
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                slot["numeric"] = False
            else:
                slot["min"] = value if slot["min"] is None else min(slot["min"], value)
                slot["max"] = value if slot["max"] is None else max(slot["max"], value)
            if len(slot["values"]) <= CATEGORICAL_MAX:
                slot["values"].add(_repr(value))
    out = {}
    for key, slot in fields.items():
        if len(slot["values"]) <= CATEGORICAL_MAX:
            out[key] = {"kind": "categorical", "values": sorted(slot["values"])}
        elif slot["numeric"]:
            out[key] = {"kind": "numeric", "min": slot["min"], "max": slot["max"], "distinct": f">{CATEGORICAL_MAX}"}
        else:
            out[key] = {"kind": "wide", "distinct": f">{CATEGORICAL_MAX}"}
    return out


def enumerate_op(fw: str, entry, sm_version: int, model_path: str | None = None) -> tuple[list, list]:
    """The executor's enumeration contract (collect.py _get_test_cases_for_model +
    capabilities.filter_cases), as op_smoke mirrors it."""
    from inspect import signature

    from collector.capabilities import filter_cases
    from collector.version_resolver import resolve_module

    module_name = entry.module or resolve_module(entry, os.environ.get("AIS_FW_VERSION", ""))
    module = importlib.import_module(module_name)
    get_func = getattr(module, entry.get_func)
    if model_path is not None and "model_path" in signature(get_func).parameters:
        cases = get_func(model_path=model_path)
    else:
        cases = get_func()
    kept, dropped = filter_cases(cases, op=entry.op, sm_version=sm_version)
    return list(kept), list(dropped)


def planned_ops(fw: str, *, python: str, sm: str) -> list[str]:
    cmd = [python, str(COLLECTOR / "collect.py"), "--backend", fw, "--model-cases-full", "--plan-only",
           "--sm", str(sm).replace("sm", "")]
    cp = subprocess.run(cmd, cwd=str(PKG_ROOT), capture_output=True, text=True)
    if cp.returncode != 0:
        raise SystemExit(f"--plan-only failed (exit {cp.returncode}):\n{cp.stderr[-2000:]}")
    start = cp.stdout.find("{")
    plan = json.loads(cp.stdout[start:]) if start >= 0 else {}
    return sorted(str(o) for o in (plan.get("ops") or plan.get("selected_ops") or []))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def build_inventory(fw: str, version: str, sm: str, *, ops: list[str] | None, python: str,
                    archive_dir: Path | None) -> dict:
    sm_version = int(str(sm).replace("sm", ""))
    os.environ["AIS_SM"] = str(sm_version)   # getters read get_sm_version(); no GPU needed
    sys.path.insert(0, str(PKG_ROOT))
    registry = importlib.import_module(f"collector.{fw}.registry").REGISTRY
    in_plan = set(planned_ops(fw, python=python, sm=sm))
    wanted = set(ops) if ops else {e.op for e in registry}
    doc_ops: dict = {}
    archive = None
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = (archive_dir / f"{fw}-{version}-{sm}.jsonl").open("w")
    seen = set()
    for entry in registry:
        if entry.op not in wanted or entry.op in seen:
            continue
        seen.add(entry.op)
        rec: dict = {"in_plan": entry.op in in_plan, "unverified_sms": list(entry.unverified_sms)}
        try:
            kept, dropped = enumerate_op(fw, entry, sm_version)
            rec.update({"cases": len(kept), "dropped_by_capabilities": len(dropped), "fields": summarize_cases(kept)})
            if archive is not None:
                for c in kept:
                    archive.write(json.dumps({"op": entry.op, "case": _repr(_case_params(c))}) + "\n")
        except Exception as e:  # a getter that cannot even enumerate is itself a finding
            rec.update({"cases": None, "error": f"{type(e).__name__}: {e!s}"[:320]})
        doc_ops[entry.op] = rec
        print(f" {entry.op:36} in_plan={rec['in_plan']!s:5} cases={rec.get('cases')}"
              + (f"  ERROR {rec['error']}" if rec.get("error") else ""), flush=True)
    if archive is not None:
        archive.close()
    return {
        "_meta": {"framework": fw, "version": version, "sm": sm, "harness_commit": _git("rev-parse", "--short", "HEAD"),
                  "enumerated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "what": "registry op case sets for this (fw, version, sm): get_func + capabilities filter, no GPU",
                  "ops_total": len(doc_ops), "ops_in_plan": sum(1 for r in doc_ops.values() if r["in_plan"]),
                  "cases_total": sum(r.get("cases") or 0 for r in doc_ops.values())},
        "ops": doc_ops,
    }


def diff_inventories(prev: dict, new: dict, waivers: dict | None = None) -> dict:
    """Regressions = ops present before and gone/unplanned/smaller now, and
    categorical field values present before and absent now. Additions are
    reported but never fail."""
    waivers = waivers or {}
    op_waivers = waivers.get("ops") or {}
    field_waivers = waivers.get("fields") or {}
    regressions, additions, waived = [], [], []
    pops, nops = prev.get("ops") or {}, new.get("ops") or {}
    for op, p in pops.items():
        n = nops.get(op)
        if op in op_waivers:
            waived.append(f"{op}: {op_waivers[op]}")
            continue
        if n is None:
            regressions.append(f"{op}: op gone from the registry")
            continue
        if p.get("in_plan") and not n.get("in_plan"):
            regressions.append(f"{op}: was in the plan, now registry-only (not collected)")
        if n.get("error"):
            regressions.append(f"{op}: getter fails to enumerate ({n['error'][:80]})")
            continue
        pc, nc = p.get("cases") or 0, n.get("cases") or 0
        if nc < pc:
            regressions.append(f"{op}: cases {pc} -> {nc}")
        elif nc > pc:
            additions.append(f"{op}: cases {pc} -> {nc}")
        for field, pf in (p.get("fields") or {}).items():
            nf = (n.get("fields") or {}).get(field)
            if pf.get("kind") != "categorical" or nf is None:
                continue
            lost = sorted(set(pf.get("values") or []) - set(nf.get("values") or []))
            if not lost:
                continue
            op_fw = field_waivers.get(op) or {}
            reason = op_fw.get(field) or (op_fw.get(int(field)) if str(field).isdigit() else None)
            if reason:
                waived.append(f"{op}[{field}] lost {lost[:4]}: {reason}")
            else:
                regressions.append(f"{op}[{field}] lost values {lost[:6]}")
    for op in nops:
        if op not in pops:
            additions.append(f"{op}: new op ({(nops[op].get('cases'))} cases)")
    return {"regressions": regressions, "additions": additions, "waived": waived}


def inventory_path(sm: str, fw: str, version: str) -> Path:
    return HARNESS / "results" / sm / "cases" / f"{fw}-{version}.yaml"


def previous_inventory(sm: str, fw: str, version: str) -> Path | None:
    """The newest inventory of the same (fw, sm) at an OLDER version."""
    cands = []
    for p in (HARNESS / "results" / sm / "cases").glob(f"{fw}-*.yaml"):
        if p.name.endswith(".waivers.yaml"):
            continue
        v = p.stem[len(fw) + 1:]
        if _version_key(v) < _version_key(version):
            cands.append((_version_key(v), p))
    return max(cands)[1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw")
    ap.add_argument("--version")
    ap.add_argument("--sm")
    ap.add_argument("--ops", default=None, help="comma list; default every registry op")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-archive", action="store_true",
                    help="do not write the full case list to the workspace archive")
    ap.add_argument("--diff", nargs=2, metavar=("PREV", "NEW"), help="diff two inventory files and exit")
    ap.add_argument("--waivers", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.diff:
        prev, new = (yaml.safe_load(Path(p).read_text()) for p in args.diff)
        waivers = yaml.safe_load(Path(args.waivers).read_text()) if args.waivers else None
        d = diff_inventories(prev, new, waivers)
        print(json.dumps(d, indent=1) if args.json else
              "\n".join([*(f" ✗ {r}" for r in d["regressions"]), *(f" ~ {w}" for w in d["waived"]),
                         *(f" + {a}" for a in d["additions"])]) or " no differences")
        return 0 if not d["regressions"] else 1

    if not (args.fw and args.version and args.sm):
        ap.error("--fw --version --sm are required (or --diff PREV NEW)")
    ws = os.environ.get("AIS_PROBE_WORKSPACE") or os.environ.get("AIC_PROBE_WORKSPACE")
    archive_dir = None if args.no_archive else Path(ws or Path.cwd()) / "archive" / "cases"
    doc = build_inventory(args.fw, args.version, args.sm, ops=args.ops.split(",") if args.ops else None,
                          python=args.python, archive_dir=archive_dir)
    out = Path(args.out) if args.out else inventory_path(args.sm, args.fw, args.version)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"\n wrote {out}  ({doc['_meta']['ops_in_plan']}/{doc['_meta']['ops_total']} ops in plan, "
          f"{doc['_meta']['cases_total']} cases)")
    prev = previous_inventory(args.sm, args.fw, args.version)
    if prev is not None:
        wv = out.with_name(out.stem + ".waivers.yaml")
        waivers = yaml.safe_load(wv.read_text()) if wv.exists() else None
        d = diff_inventories(yaml.safe_load(prev.read_text()), doc, waivers)
        print(f" diff vs {prev.name}: {len(d['regressions'])} regressions, {len(d['additions'])} additions, "
              f"{len(d['waived'])} waived")
        for r in d["regressions"]:
            print(f"  ✗ {r}")
        return 0 if not d["regressions"] else 1
    print(" no older inventory for this (fw, sm) — first baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
