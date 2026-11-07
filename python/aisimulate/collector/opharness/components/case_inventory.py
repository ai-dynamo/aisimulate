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
          plan_hash (the case_plan_hash of a run that attempts every case)
  diff  : regressions / explained / additions against the base plan, case by case: a case
          of the base plan that the new plan lacks is explained only by a skips.py region
          of the new plan whose every key=value it carries

Usage (CPU, no framework image):
  python3 case_inventory.py --fw vllm --version 0.33.0 --sm sm90 --base <rev>|none [--ops a,b]
  python3 case_inventory.py --fw vllm --skips [--ops a,b]   # rule-7 recheck skeleton for a pin bump
"""

from __future__ import annotations

import argparse
import ast
import csv
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


def plan_report(fw: str, sm: str, pkg_root: Path, table_dir: Path, *, python: str, ops: list[str] | None) -> dict:
    """collect.py --plan-only --plan-table for one backend and SM, run from ``pkg_root``
    (a checkout's python/aisimulate, or a base revision extracted from git)."""
    cmd = [python, str(pkg_root / "collector" / "collect.py"), "--backend", fw, "--model-cases-full"]
    cmd += ["--sm", str(sm).replace("sm", ""), "--plan-only", "--plan-table", str(table_dir)]
    if ops:
        cmd += ["--ops", *ops]
    cp = subprocess.run(cmd, cwd=str(pkg_root), capture_output=True, text=True)
    if cp.returncode != 0:
        raise SystemExit(f"--plan-only failed at {pkg_root} (exit {cp.returncode}):\n{cp.stderr[-2000:]}")
    start = cp.stdout.find("{")
    return json.loads(cp.stdout[start:]) if start >= 0 else {}


def build_inventory(report: dict) -> dict:
    """The per-op plan record from a --plan-only report."""
    ops = {}
    for op, plan in (report.get("plans") or {}).items():
        ops[op] = {
            "cases": plan.get("cases"),
            "by_dtype": plan.get("by_dtype") or {},
            "by_dispatch": plan.get("by_dispatch") or {},
            "removed": {r["region"]: r["cases"] for r in plan.get("removed") or []},
            "removed_runs": {r["region"]: r["cases"] for r in plan.get("removed_runs") or []},
            "plan_hash": plan.get("plan_hash"),
        }
    for op in report.get("ops") or []:
        ops.setdefault(op, {"cases": None, "note": "no op plan: the collector builds its own population (moe_a2a)"})
    return ops


def read_tables(report: dict) -> dict:
    """op -> (header, rows) of the case tables a --plan-only --plan-table report wrote."""
    tables = {}
    for op, plan in (report.get("plans") or {}).items():
        path = Path(plan["table"]) if plan.get("table") else None
        if path and path.exists():
            with path.open(newline="") as fh:
                rows = list(csv.reader(fh))
            tables[op] = (rows[0], rows[1:]) if rows else ([], [])
    return tables


def _pairs(region: str) -> list[tuple[str, str]]:
    """``"dispatch=a,kv_cache_dtype=fp8"`` -> [("dispatch", "a"), ("kv_cache_dtype", "fp8")]; a piece
    without ``=`` continues the previous value (a value holding a comma)."""
    pairs: list[list[str]] = []
    for piece in region.split(","):
        key, sep, value = piece.partition("=")
        if sep and key.isidentifier():
            pairs.append([key, value])
        elif pairs:
            pairs[-1][1] += "," + piece
    return [(k, v) for k, v in pairs]


def _region_holds(pairs: list[tuple[str, str]], case: dict) -> bool:
    """Does a skips.py region (skips._first_match: the entry's key=value pairs on a case) name
    this case? ``dtype`` is one of the case's dtype fields; ``op=`` names the whole op."""
    for key, value in pairs:
        if key == "op":
            continue
        if key == "dtype":
            if value not in case.values():
                return False
        elif case.get(key) != value:
            return False
    return True


def _sweep(value: str) -> frozenset | None:
    """A sweep field's values (``"(1, 2, 4)"``), or None for a scalar."""
    if not (value.startswith("(") or value.startswith("[")):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    return frozenset(parsed) if isinstance(parsed, (tuple, list)) else None


def _widened(lost: set, after: set) -> set:
    """The lost cases a new case still covers: the same scalar fields, and every sweep field
    of the new case a superset of the old one (a sweep that gained points)."""
    by_scalars: dict[tuple, list[tuple]] = {}
    for case in after:
        sweeps = tuple(_sweep(v) for v in case)
        if any(sweeps):
            key = tuple(v for v, w in zip(case, sweeps, strict=True) if w is None)
            by_scalars.setdefault(key, []).append(sweeps)
    covered = set()
    for case in lost:
        sweeps = tuple(_sweep(v) for v in case)
        if not any(sweeps):
            continue
        key = tuple(v for v, w in zip(case, sweeps, strict=True) if w is None)
        if any(all(w is None or w <= nw for w, nw in zip(sweeps, new, strict=True)) for new in by_scalars.get(key, [])):
            covered.add(case)
    return covered


def diff_plans(prev_ops: dict, new_ops: dict, prev_tables: dict, new_tables: dict) -> dict:
    """Regressions = an op that left the plan, or a case of the base plan that the new plan
    lacks and no skips.py region of the new plan names. Cases are compared on the columns
    both tables share, without the dispatch (a case moving to another kernel is not a loss).
    A sweep field (a tuple such as moe ``num_tokens``) may gain points: the old case is still
    covered. Model entries (``model=``) remove whole model rows and are credited by count.
    Additions and explained losses are reported but never fail, and never offset a loss."""
    regressions, explained, additions = [], [], []
    for op, p in prev_ops.items():
        n = new_ops.get(op)
        if n is None:
            regressions.append(f"{op}: op left the plan")
            continue
        if op not in prev_tables or op not in new_tables:
            continue  # no op plan (moe_a2a): nothing to compare case by case
        (ph, prows), (nh, nrows) = prev_tables[op], new_tables[op]
        shared = [c for c in ph if c in nh and c != "dispatch"]
        pi, ni = [ph.index(c) for c in shared], [nh.index(c) for c in shared]
        before = {tuple(r[i] for i in pi) for r in prows}
        after = {tuple(r[i] for i in ni) for r in nrows}
        lost, added = before - after, after - before
        widened = _widened(lost, after)
        lost -= widened
        if widened:
            additions.append(f"{op}: {len(widened)} cases whose sweep gained points")
        if len(added) > len(widened):
            additions.append(f"{op}: +{len(added) - len(widened)} cases")
        if not lost:
            continue
        regions = [(region, _pairs(region)) for region in n.get("removed") or {} if not region.startswith("model=")]
        by_region: dict[str, int] = {}
        unexplained = []
        for case in sorted(lost):
            fields = dict(zip(shared, case, strict=True))
            region = next((r for r, pairs in regions if _region_holds(pairs, fields)), None)
            if region is None:
                unexplained.append(fields)
            else:
                by_region[region] = by_region.get(region, 0) + 1
        model_growth = sum(
            max(0, count - (p.get("removed") or {}).get(region, 0))
            for region, count in (n.get("removed") or {}).items()
            if region.startswith("model=")
        )
        credited = min(model_growth, len(unexplained))
        if credited:
            by_region["model entries"] = credited
            unexplained = unexplained[credited:]
        if by_region:
            explained.append(f"{op}: {sum(by_region.values())} cases left under skips.py {sorted(by_region)}")
        if unexplained:
            regressions.append(
                f"{op}: {len(unexplained)} cases left the plan without a skips.py entry, e.g. {unexplained[0]}"
            )
    for op, n in new_ops.items():
        if op not in prev_ops:
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
    ap.add_argument(
        "--skips",
        action="store_true",
        help="print the recheck-record skeleton of the collector/skips.py entries in scope for --fw and exit",
    )
    args = ap.parse_args()

    if args.skips:
        if not args.fw:
            ap.error("--skips needs --fw")
        scope = skips_in_scope(args.fw, args.ops.split(",") if args.ops else None)
        items = [{"entry": skip_key(e), "outcome": None, "how": None, "ref": None} for e in scope]
        print(yaml.safe_dump({"skips": items}, sort_keys=False, allow_unicode=True, width=200))
        return 0
    if not (args.fw and args.version and args.sm and args.base):
        ap.error("--fw --version --sm --base are required")
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
            doc["diff"] = diff_plans(
                build_inventory(base_report), doc["ops"], read_tables(base_report), read_tables(report)
            )
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
