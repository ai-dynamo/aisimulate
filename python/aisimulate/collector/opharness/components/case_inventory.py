#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: the collector's CASE PLAN for (fw, sm), and its diff against a base revision.

A case plan is a pure function of the collector source: the op plans
(collector/op_plans.py) expand the case YAML and collector/skips.py leaves
cases out, without importing any framework. `collect.py --plan-only
--plan-table` therefore enumerates every op of a backend on any CPU in
seconds. This component records that plan and diffs it against the plan at a
base revision of the collector, which is the "test-case regression" check: the
b200_sxm sglang 0.5.21 shard run (job 469988017) lost dsv4_paged_mqa_logits_module
(op not in the plan), the DeepSeek-V4-Pro top-k calibration (a getter's drop)
and the sglang mla_context_module table (no producer) — all visible in a plan,
none needing a GPU hour to discover.

Cases leave a plan on purpose only through collector/skips.py, and every such
entry is counted under `removed` with its reason. A loss the growth of
`removed` does not account for is a regression; there is no waiver file.

Output (the probe workspace on the control machine, never committed): results/<sm>/cases/<fw>-<version>.yaml
  _meta : fw / version (the pin under test, a label) / sm / revisions / counts
  ops   : per planned op — cases, by_dtype, by_dispatch, removed {region: cases},
          plan_hash (the case_plan_hash of a run that attempts every case), and
          fields: per plan-table column, the value set when small (categorical)
          or min/max when numeric
  diff  : regressions / explained / additions against the base plan

Usage (CPU, no framework image):
  python3 case_inventory.py --fw vllm --version 0.33.0 --sm sm90 --base <rev>|none [--ops a,b]
  python3 case_inventory.py --diff PREV.yaml NEW.yaml
  python3 case_inventory.py --fw vllm --skips [--ops a,b]   # rule-7 recheck skeleton for a pin bump
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
COLLECTOR = HARNESS.parent
PKG_ROOT = COLLECTOR.parent
CATEGORICAL_MAX = 64


def _repr(v) -> str:
    s = str(v)
    return s if len(s) <= 200 else hashlib.sha256(s.encode()).hexdigest()[:16]


def summarize_cases(cases: list[dict]) -> dict:
    """Per-column summary of plan-table rows: value sets for categorical columns,
    ranges for numeric ones."""
    fields: dict = {}
    for case in cases:
        for key, raw in case.items():
            slot = fields.setdefault(key, {"values": set(), "numeric": True, "min": None, "max": None})
            try:
                value = float(raw) if "." in str(raw) else int(raw)
            except (TypeError, ValueError):
                value = None
                slot["numeric"] = False
            if value is not None:
                slot["min"] = value if slot["min"] is None else min(slot["min"], value)
                slot["max"] = value if slot["max"] is None else max(slot["max"], value)
            if len(slot["values"]) <= CATEGORICAL_MAX:
                slot["values"].add(_repr(raw))
    out = {}
    for key, slot in fields.items():
        if len(slot["values"]) <= CATEGORICAL_MAX:
            out[key] = {"kind": "categorical", "values": sorted(slot["values"])}
        elif slot["numeric"]:
            out[key] = {"kind": "numeric", "min": slot["min"], "max": slot["max"], "distinct": f">{CATEGORICAL_MAX}"}
        else:
            out[key] = {"kind": "wide", "distinct": f">{CATEGORICAL_MAX}"}
    return out


def plan_report(fw: str, sm: str, pkg_root: Path, table_dir: Path, *, python: str, ops: list[str] | None) -> dict:
    """collect.py --plan-only --plan-table for one backend and SM, run from ``pkg_root``
    (a checkout's python/aisimulate, or a base revision extracted from git)."""
    cmd = [python, str(pkg_root / "collector" / "collect.py"), "--backend", fw, "--model-cases-full"]
    cmd += ["--sm", str(sm).replace("sm", ""), "--plan-only", "--plan-table", str(table_dir)]
    if ops:
        cmd += ["--ops", *ops]
    cp = subprocess.run(cmd, cwd=str(pkg_root), capture_output=True, text=True)
    if cp.returncode != 0 and "unrecognized arguments: --plan-table" in cp.stderr:
        # A collector from before the op plans: it lists the planned ops, not their cases.
        cmd = cmd[: cmd.index("--plan-table")] + cmd[cmd.index("--plan-table") + 2 :]
        cp = subprocess.run(cmd, cwd=str(pkg_root), capture_output=True, text=True)
    if cp.returncode != 0:
        raise SystemExit(f"--plan-only failed at {pkg_root} (exit {cp.returncode}):\n{cp.stderr[-2000:]}")
    start = cp.stdout.find("{")
    return json.loads(cp.stdout[start:]) if start >= 0 else {}


def build_inventory(report: dict) -> dict:
    """The per-op plan record from a --plan-only report and the tables it wrote."""
    ops = {}
    for op, plan in (report.get("plans") or {}).items():
        table = Path(plan["table"]) if plan.get("table") else None
        rows = list(csv.DictReader(table.open(newline=""))) if table and table.exists() else []
        ops[op] = {
            "cases": plan.get("cases"),
            "by_dtype": plan.get("by_dtype") or {},
            "by_dispatch": plan.get("by_dispatch") or {},
            "removed": {r["region"]: r["cases"] for r in plan.get("removed") or []},
            "removed_runs": {r["region"]: r["cases"] for r in plan.get("removed_runs") or []},
            "plan_hash": plan.get("plan_hash"),
            "fields": summarize_cases(rows),
        }
    for op in report.get("ops") or []:
        ops.setdefault(op, {"cases": None, "note": "no op plan: the collector builds its own population (moe_a2a)"})
    return ops


def diff_inventories(prev: dict, new: dict) -> dict:
    """Regressions = an op that left the plan, or cases and categorical values that
    left it without a collector/skips.py entry accounting for them. Additions and
    explained losses are reported but never fail."""
    regressions, explained, additions = [], [], []
    pops, nops = prev.get("ops") or {}, new.get("ops") or {}
    for op, p in pops.items():
        n = nops.get(op)
        if n is None:
            regressions.append(f"{op}: op left the plan")
            continue
        if p.get("cases") is None or n.get("cases") is None:
            continue
        grown = {
            region: count - (p.get("removed") or {}).get(region, 0)
            for region, count in (n.get("removed") or {}).items()
            if count > (p.get("removed") or {}).get(region, 0)
        }
        lost = p["cases"] - n["cases"]
        if lost > 0:
            if sum(grown.values()) >= lost:
                explained.append(f"{op}: cases {p['cases']} -> {n['cases']}, skips.py {sorted(grown)}")
            else:
                regressions.append(f"{op}: cases {p['cases']} -> {n['cases']} ({sum(grown.values())} under skips.py)")
        elif lost < 0:
            additions.append(f"{op}: cases {p['cases']} -> {n['cases']}")
        for field, pf in (p.get("fields") or {}).items():
            nf = (n.get("fields") or {}).get(field)
            if pf.get("kind") != "categorical" or nf is None or nf.get("kind") != "categorical":
                continue
            gone = sorted(set(pf.get("values") or []) - set(nf.get("values") or []))
            if not gone:
                continue
            accounted = [v for v in gone if any(f"{field}={v}" in region or v in region for region in grown)]
            if accounted:
                explained.append(f"{op}[{field}] lost {accounted[:4]} under skips.py")
            rest = [v for v in gone if v not in accounted]
            if rest:
                regressions.append(f"{op}[{field}] lost values {rest[:6]}")
    for op, n in nops.items():
        if op not in pops:
            additions.append(f"{op}: new op ({n.get('cases')} cases)")
    return {"regressions": regressions, "explained": explained, "additions": additions}


def _git(*args: str, cwd: Path = HARNESS) -> str:
    cp = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    return cp.stdout.strip() if cp.returncode == 0 else ""


# What a plan reads: the collector, and the packaged checkpoint configs op_plans takes
# max_position_embeddings and checkpoint quantization from (helper._AIC_MODEL_CONFIG_DIR).
PLAN_INPUTS = ("collector", "src/aisimulate_core/model_configs")


def extract_collector(rev: str, dest: Path) -> Path:
    """The plan's inputs (PLAN_INPUTS) at a git revision, extracted under ``dest`` (no checkout)."""
    top = Path(_git("rev-parse", "--show-toplevel"))
    prefix = PKG_ROOT.relative_to(top).as_posix()
    blob = subprocess.run(
        ["git", "-C", str(top), "archive", "--format=tar", rev, *(f"{prefix}/{p}" for p in PLAN_INPUTS)],
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    return dest / prefix


def skips_in_scope(fw: str, ops: list[str] | None = None) -> list[dict]:
    """The collector/skips.py entries a pin bump of this backend re-checks (rule 7 of the
    collector overview): those that name it or no backend, and, for a family pin, name one
    of its ops or no op."""
    sys.path.insert(0, str(PKG_ROOT))
    try:
        from collector.skips import SKIPS
    finally:
        sys.path.pop(0)
    return [
        entry
        for entry in SKIPS
        if fw in _as_list(entry.get("backend", fw))
        and (not ops or "op" not in entry or set(ops) & set(_as_list(entry["op"])))
    ]


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]


def skip_key(entry: dict) -> str:
    """A stable key for a skips.py entry (callables by name), as recheck records name it."""
    return json.dumps(
        {k: getattr(v, "__name__", v) if callable(v) else v for k, v in entry.items()}, sort_keys=True, default=str
    )


def inventory_path(sm: str, fw: str, version: str) -> Path:
    sys.path.insert(0, str(HERE))
    import probe_driver

    return probe_driver.ROOT / "results" / sm / "cases" / f"{fw}-{version}.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw")
    ap.add_argument("--version", help="the pin under test; names the output file")
    ap.add_argument("--sm")
    ap.add_argument("--ops", default=None, help="comma list; default every planned op")
    ap.add_argument(
        "--base",
        required=False,
        help="git revision whose collector plan is the baseline, e.g. the merge base with upstream main "
        "('none' records the plan without a diff)",
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", nargs=2, metavar=("PREV", "NEW"), help="diff two inventory files and exit")
    ap.add_argument(
        "--skips",
        action="store_true",
        help="print the recheck-record skeleton of the collector/skips.py entries in scope for --fw and exit",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.diff:
        prev, new = (yaml.safe_load(Path(p).read_text()) for p in args.diff)
        d = diff_inventories(prev, new)
        print(json.dumps(d, indent=1) if args.json else _format_diff(d))
        return 0 if not d["regressions"] else 1

    if args.skips:
        if not args.fw:
            ap.error("--skips needs --fw")
        scope = skips_in_scope(args.fw, args.ops.split(",") if args.ops else None)
        items = [{"entry": skip_key(e), "outcome": None, "how": None, "ref": None} for e in scope]
        print(yaml.safe_dump({"skips": items}, sort_keys=False, allow_unicode=True, width=200))
        return 0
    if not (args.fw and args.version and args.sm and args.base):
        ap.error("--fw --version --sm --base are required (or --diff PREV NEW)")
    ops = args.ops.split(",") if args.ops else None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        report = plan_report(args.fw, args.sm, PKG_ROOT, tmp / "new", python=args.python, ops=ops)
        doc = {
            "_meta": {
                "framework": args.fw,
                "version": args.version,
                "sm": args.sm,
                "revision": _git("rev-parse", "--short", "HEAD"),
                "dirty": bool(_git("status", "--porcelain", "--", str(COLLECTOR))),
                "planned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "what": "collect.py --plan-only --plan-table on a CPU: the cases every planned op attempts",
            },
            "ops": build_inventory(report),
        }
        doc["_meta"]["cases_total"] = sum(r.get("cases") or 0 for r in doc["ops"].values())
        if args.base != "none":
            base_root = extract_collector(args.base, tmp / "base")
            base_report = plan_report(args.fw, args.sm, base_root, tmp / "old", python=args.python, ops=ops)
            doc["_meta"]["base"] = {"rev": args.base, "commit": _git("rev-parse", "--short", args.base)}
            doc["diff"] = diff_inventories({"ops": build_inventory(base_report)}, doc)
    out = Path(args.out) if args.out else inventory_path(args.sm, args.fw, args.version)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f" wrote {out}  ({len(doc['ops'])} ops, {doc['_meta']['cases_total']} cases)")
    if "diff" not in doc:
        return 0
    print(f" diff vs {args.base}:\n{_format_diff(doc['diff'])}")
    return 0 if not doc["diff"]["regressions"] else 1


def _format_diff(d: dict) -> str:
    lines = [*(f" ✗ {r}" for r in d["regressions"]), *(f" ~ {e}" for e in d["explained"])]
    lines += [f" + {a}" for a in d["additions"]]
    return "\n".join(lines) or " no differences"


if __name__ == "__main__":
    raise SystemExit(main())
