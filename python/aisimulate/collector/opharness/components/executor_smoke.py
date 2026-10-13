#!/usr/bin/env python3
"""Component: smoke the collector through its REAL execution path.

op_smoke.py calls (get_func, run_func) directly — it proves a collector can
build and time one case on the new framework. It never passes through
collect.py's main: the --model-cases-full case plan, the checkpoint ledger,
--resume, and parquet finalization. The b200_sxm sglang 0.5.21 shard run
(GitLab job 469988017, 2026-10-05) failed 7 shards in exactly those layers
while every op_smoke cell was green: three ops were not in the case plan at
all, three skip_indexer shards and one all-failed shard died in --resume
finalization. This component closes that gap with two modes:

  --shards <plan.yaml>      CPU only. For every shard op, run
                            `collect.py --backend <fw> --model-cases-full --plan-only --ops <op>`
                            and report which ops the case plan refuses — the
                            pipeline's shard plan is checked against the
                            collector's plan BEFORE a GPU hour is spent.
  --run                     In the framework image. For every op of the case
                            plan (or --ops), run collect.py the way the
                            pipeline does — `--model-cases-full --ops <op>
                            --limit N --checkpoint-dir ... --resume` — in a
                            scratch result dir, then judge the ARTIFACTS:
                            exit code, parquet produced, errors reported.
                            Writes results/<sm>/executor_smoke/<fw>-<ver>.yaml
                            for workflow_check `executor_smoked`.

A per-op status is one of:
  ok            exit 0 and at least one parquet table finalized
  all_failed    exit 0, no parquet, every attempted case failed (classified or
                not — the error report says which; a lane guard refusing the
                whole op lands here and must be explained in findings)
  not_in_plan   collect.py refused the op ("not present in the collector v2 case plan")
  fail          anything else (non-zero exit, finalize crash, no artifacts)

Usage:
  python3 executor_smoke.py --fw sglang --shards shards_sm100_sglang_0521.yaml
  python3 executor_smoke.py --fw sglang --version 0.5.21 --sm sm100 --run --limit 1 [--ops gemm,moe] \\
      [--python /opt/sglang/bin/python3] [--work /tmp/executor_smoke]
"""
from __future__ import annotations

import argparse
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
COLLECTOR = HARNESS.parent            # python/aisimulate/collector
PKG_ROOT = COLLECTOR.parent           # python/aisimulate (collect.py imports `collector.*` from here)
COLLECT_PY = COLLECTOR / "collect.py"
NOT_IN_PLAN_RX = re.compile(r"Requested ops are not present in the collector v2 case plan: (.*)")


def _run(
    cmd: list[str], *, cwd: Path, env: dict | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)


def plan_check_one(fw: str, op: str, *, python: str, sm: str | None = None) -> dict:
    """One `--plan-only` invocation: does the collector's full case plan contain op?"""
    cmd = [python, str(COLLECT_PY), "--backend", fw, "--model-cases-full", "--plan-only", "--ops", op]
    if sm:
        cmd += ["--sm", str(sm).replace("sm", "")]
    cp = _run(cmd, cwd=PKG_ROOT)
    text = (cp.stdout or "") + (cp.stderr or "")
    m = NOT_IN_PLAN_RX.search(text)
    if m:
        return {"op": op, "status": "not_in_plan", "detail": m.group(1).strip()}
    if cp.returncode != 0:
        detail = text.strip().splitlines()[-1] if text.strip() else f"exit {cp.returncode}"
        return {"op": op, "status": "fail", "detail": detail}
    return {"op": op, "status": "in_plan"}


def shard_ops(shards_yaml: Path) -> list[str]:
    data = yaml.safe_load(shards_yaml.read_text(encoding="utf-8")) or {}
    ops = []
    for shard in data.get("shards") or []:
        op = shard.get("op") if isinstance(shard, dict) else None
        if op and op not in ops:
            ops.append(op)
    return ops


def planned_ops(fw: str, *, python: str, sm: str | None = None) -> list[str]:
    """Ops the collector's --model-cases-full plan lists (from --plan-only JSON)."""
    cmd = [python, str(COLLECT_PY), "--backend", fw, "--model-cases-full", "--plan-only"]
    if sm:
        cmd += ["--sm", str(sm).replace("sm", "")]
    cp = _run(cmd, cwd=PKG_ROOT)
    if cp.returncode != 0:
        raise SystemExit(f"--plan-only failed (exit {cp.returncode}):\n{cp.stderr[-2000:]}")
    start = cp.stdout.find("{")
    plan = json.loads(cp.stdout[start:]) if start >= 0 else {}
    ops = plan.get("ops") or plan.get("selected_ops") or []
    if not ops:
        raise SystemExit("--plan-only printed no ops; cannot enumerate the plan")
    return sorted(str(o) for o in ops)


def judge_run(exit_code: int, result_dir: Path) -> dict:
    parquets = sorted(p.name for p in result_dir.glob("*.parquet"))
    rows = 0
    for p in result_dir.glob("*.parquet"):
        try:
            import pyarrow.parquet as pq  # optional: only for the row count

            rows += pq.read_metadata(p).num_rows
        except Exception:
            pass
    errors = 0
    for ej in result_dir.glob("errors_*.json"):
        try:
            errors += len(json.loads(ej.read_text()))
        except Exception:
            pass
    if exit_code == 0 and parquets:
        status = "ok"
    elif exit_code == 0 and not parquets and errors:
        status = "all_failed"
    else:
        status = "fail"
    return {"status": status, "exit": exit_code, "parquet": parquets, "rows": rows, "errors": errors}


def run_one(fw: str, op: str, *, python: str, work: Path, limit: int, sm: str | None, timeout: int) -> dict:
    result_dir = work / op
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, str(COLLECT_PY), "--backend", fw, "--model-cases-full", "--ops", op,
           "--limit", str(limit), "--checkpoint-dir", str(result_dir / "collector_checkpoint"), "--resume"]
    if sm:
        cmd += ["--sm", str(sm).replace("sm", "")]
    t0 = time.time()
    try:
        cp = _run(cmd, cwd=result_dir, env={**os.environ, "PYTHONPATH": str(PKG_ROOT)}, timeout=timeout)
        exit_code, tail = cp.returncode, ((cp.stdout or "") + (cp.stderr or ""))[-3000:]
    except subprocess.TimeoutExpired:
        exit_code, tail = -1, f"timeout after {timeout}s"
    (result_dir / "executor_smoke.log").write_text(tail)
    verdict = judge_run(exit_code, result_dir)
    m = NOT_IN_PLAN_RX.search(tail)
    if m:
        verdict["status"] = "not_in_plan"
    verdict["seconds"] = round(time.time() - t0)
    verdict["cmd"] = " ".join(cmd)
    return verdict


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", required=True)
    ap.add_argument("--version", default=None, help="framework version the run is recorded under (--run)")
    ap.add_argument("--sm", default=None, help="sm90 / sm100 / ... (also passed to collect.py --sm)")
    ap.add_argument("--python", default=sys.executable, help="interpreter with the framework (in --run mode)")
    ap.add_argument("--shards", default=None, help="pipeline shard plan yaml to check against the case plan (CPU)")
    ap.add_argument("--run", action="store_true", help="run every planned op through collect.py main (GPU)")
    ap.add_argument("--ops", default=None, help="comma list: restrict --run to these ops")
    ap.add_argument("--limit", type=int, default=1, help="collect.py --limit per op in --run mode")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per op in --run mode")
    ap.add_argument("--work", default=None, help="scratch dir for --run result dirs")
    ap.add_argument("--out", default=None, help="override results/<sm>/executor_smoke/<fw>-<version>.yaml")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.shards:
        ops = shard_ops(Path(args.shards))
        rows = [plan_check_one(args.fw, op, python=args.python, sm=args.sm) for op in ops]
        bad = [r for r in rows if r["status"] != "in_plan"]
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                mark = "✓" if r["status"] == "in_plan" else "✗"
                print(f" {mark} {r['op']:36} {r['status']}" + (f"  {r.get('detail')}" if r.get("detail") else ""))
            print(f"\n {len(ops) - len(bad)}/{len(ops)} shard ops are in the {args.fw} --model-cases-full plan")
        return 0 if not bad else 1

    if not args.run:
        ap.error("pass --shards <yaml> (plan check) or --run (execute through collect.py)")
    if not (args.version and args.sm):
        ap.error("--run needs --version and --sm (the result file is keyed by them)")
    plan = planned_ops(args.fw, python=args.python, sm=args.sm)
    wanted = [o.strip() for o in args.ops.split(",")] if args.ops else plan
    not_in_plan = sorted(set(wanted) - set(plan))
    work = Path(args.work or (HARNESS / "results" / args.sm / "executor_smoke" / f"work_{args.fw}-{args.version}"))
    results = {}
    for op in [o for o in wanted if o in plan]:
        results[op] = run_one(args.fw, op, python=args.python, work=work, limit=args.limit, sm=args.sm,
                              timeout=args.timeout)
        print(f" {'✓' if results[op]['status'] == 'ok' else '✗'} {op:36} {results[op]['status']:12} "
              f"rows={results[op]['rows']} errors={results[op]['errors']} {results[op]['seconds']}s", flush=True)
    default_out = HARNESS / "results" / args.sm / "executor_smoke" / f"{args.fw}-{args.version}.yaml"
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "_meta": {"framework": args.fw, "version": args.version, "sm": args.sm, "limit": args.limit,
                  "python": args.python, "harness_commit": _git("rev-parse", "--short", "HEAD"),
                  "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "what": "collect.py main path per op: --model-cases-full plan, checkpoint ledger, --resume, finalize",
                  "summary": {s: sum(1 for r in results.values() if r["status"] == s)
                              for s in ("ok", "all_failed", "not_in_plan", "fail")}},
        "planned_ops": plan,
        "not_in_plan": not_in_plan,
        "ops": results,
    }
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"\n wrote {out}")
    return 0 if all(r["status"] == "ok" for r in results.values()) and not not_in_plan else 1


if __name__ == "__main__":
    raise SystemExit(main())
