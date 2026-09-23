#!/usr/bin/env python3
"""Component: generic single-case runner for any registered op collector.

Motivation (owner rule 2026-09-19: fix drift with one standard, never one
config per model/op): several collectors export only (get_func, run_func)
pairs and rely on collect.py's executor for invocation — they have no
__main__, so "smoke one case on the new framework version" used to mean
writing ad-hoc driver code per op. That ad-hoc code is a drift point: each
rewrite risks inventing its own invocation semantics. This component fixes
the invocation ONCE, taken verbatim from the executor's contract:

    cases = get_func()                      # collect.py:1103-1111 (optional
                                            #   model_path kwarg honored)
    run_func(*case, perf_filename=..., device=...)
                                            # collect.py:1597 (*task, device)
                                            # collect.py:2179 (perf_filename
                                            #   bound by partial)

The op -> (module, get_func, run_func, perf_filename) mapping comes from the
backend's own registry — never hand-typed here. --module overrides the
registry module for smoking unrouted version lanes (e.g. collect_*_029.py
forks parked until the manifest pin moves) with the SAME registry-declared
function names and perf file.

Runs INSIDE the framework container (same interpreter as the collector):

  PYTHONPATH=/ws python3 op_smoke.py --backend vllm --op compute_scale
  PYTHONPATH=/ws python3 op_smoke.py --backend vllm --op mla_bmm_gen_pre \
      --module collector.vllm.collect_mla_bmm_029 --cases 2

The one law applies: a case runs or the component raises — no skip, no
substitute, no catch-and-continue.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from inspect import signature
from pathlib import Path


def _get_cases(get_func, model_path: str | None):
    # Mirrors collect.py:_get_test_cases_for_model (:1103-1111): pass
    # model_path only to getters that accept it.
    if model_path is not None and "model_path" in signature(get_func).parameters:
        return get_func(model_path=model_path)
    return get_func()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="collector backend package (vllm/sglang/trtllm)")
    ap.add_argument("--op", required=True, help="op name as declared in the backend registry")
    ap.add_argument("--module", default=None,
                    help="override the registry module (smoke an unrouted version lane); "
                         "get/run function names and perf filename still come from the registry entry")
    ap.add_argument("--cases", type=int, default=1, help="how many cases to run (from the head of the plan)")
    ap.add_argument("--case-index", type=int, default=None,
                    help="run exactly this case index instead of the head slice")
    ap.add_argument("--case-filter", default=None,
                    help="comma list of substrings; run the first --cases cases whose str(case) "
                         "contains all of them (collect.py --case-filter semantics) — picks a "
                         "representative cell (model, precision, shape) without hand-typing the tuple")
    ap.add_argument("--model-path", default=None, help="forwarded to get_func when it accepts model_path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=".", help="directory for the perf file (registry filename)")
    args = ap.parse_args()

    registry = importlib.import_module(f"collector.{args.backend}.registry")
    entries = [e for e in registry.REGISTRY if e.op == args.op]
    if not entries:
        known = ", ".join(sorted({e.op for e in registry.REGISTRY}))
        raise SystemExit(f"op {args.op!r} not in collector.{args.backend}.registry (known: {known})")
    if len(entries) > 1 and args.module is None:
        raise SystemExit(
            f"op {args.op!r} has {len(entries)} registry entries (version routes); "
            "pass --module to pick the lane explicitly"
        )
    entry = entries[0]

    module_name = args.module or entry.module
    module = importlib.import_module(module_name)
    get_func = getattr(module, entry.get_func)
    run_func = getattr(module, entry.run_func)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    perf_path = str(Path(args.out_dir) / str(entry.perf_filename))

    cases = _get_cases(get_func, args.model_path)
    if not cases:
        raise SystemExit(f"{entry.get_func} returned 0 cases — nothing to smoke "
                         "(platform floor or empty plan; see the drop log above)")
    if args.case_filter:
        subs = [s.strip() for s in args.case_filter.split(",") if s.strip()]
        matching = [c for c in cases if all(s in str(c) for s in subs)]
        if not matching:
            raise SystemExit(f"--case-filter {args.case_filter!r} matches 0/{len(cases)} cases")
        picked = matching[: args.cases]
        print(f"[op_smoke] --case-filter {args.case_filter!r}: {len(matching)}/{len(cases)} cases match")
    else:
        picked = ([cases[args.case_index]] if args.case_index is not None
                  else cases[: args.cases])
    print(f"[op_smoke] {args.op}: module={module_name} run={entry.run_func} "
          f"perf={perf_path} — {len(picked)}/{len(cases)} case(s)")
    for i, case in enumerate(picked):
        print(f"[op_smoke] case {i}: {case}")
        run_func(*case, perf_filename=perf_path, device=args.device)
    print(f"[op_smoke] OK: {len(picked)} case(s) ran; rows appended to {perf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
