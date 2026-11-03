#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
  PYTHONPATH=/ws python3 op_smoke.py --backend trtllm --op dsv4_hca_context_module \
      --select model_path=sgl-project/DeepSeek-V4-Flash-FP8,seq_len=4096,batch_size=1,tp_size=1,\
gemm_type=fp8_block,prefix_len=0

--select names one case of the op plan (collector/op_plans.py) by its fields and
runs the row the collector queues for it on this SM, so a capture's cell is a
planned case by construction (planned_row()).

The one law applies: a case runs or the component raises — no skip, no
substitute, no catch-and-continue.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from inspect import signature
from pathlib import Path


def planned_row(backend: str, op: str, sm_version: int | None, **select):
    """The run arguments the collector queues for the ONE planned case whose fields equal ``select``.

    The case comes from the op plan (collector/op_plans.plan: the shared cases minus
    collector/skips.py for this backend and SM) and its row from the plan's own rows /
    batch_rows, restricted to that case: a batch row then times exactly that point, and
    everything the plan decides per SM (a pinned dispatch, a KV pool) is what a
    collection would run. A select that matches no planned case, or several, raises:
    a capture must name a cell the collector actually measures.
    """
    import dataclasses

    from collector.op_plans import plan

    op_plan = plan(op, backend, sm_version)
    picked = [
        i
        for i, case in enumerate(op_plan.cases)
        if all(str(getattr(case, field)) == str(value) for field, value in select.items())
    ]
    if len(picked) != 1:
        raise SystemExit(f"{backend} {op} sm{sm_version}: select {select} matches {len(picked)} planned cases, want 1")
    i = picked[0]
    one = dataclasses.replace(
        op_plan, cases=[op_plan.cases[i]], dispatches=op_plan.dispatches[i : i + 1] if op_plan.dispatches else []
    )
    rows = one.rows
    if len(rows) != 1:
        raise SystemExit(f"{backend} {op}: the planned case {op_plan.cases[i]} runs as {len(rows)} rows, want 1")
    return rows[0]


def _parse_select(text: str) -> dict:
    pairs = [item.split("=", 1) for item in text.split(",") if item.strip()]
    if any(len(pair) != 2 for pair in pairs):
        raise SystemExit(f"--select takes field=value pairs, got {text!r}")
    return {field.strip(): value.strip() for field, value in pairs}


def _bound_kwargs(entry, run_func, out_dir: str) -> dict:
    """The keyword arguments collect.py binds besides perf_filename and device."""
    from inspect import signature

    extra = {}
    if getattr(entry, "extra_perf_filenames", None) and "extra_perf_filenames" in signature(run_func).parameters:
        # multi-table producers (compute_scale -> computescale + scale_matrix): same kwarg collect.py binds
        extra["extra_perf_filenames"] = tuple(str(Path(out_dir) / str(p)) for p in entry.extra_perf_filenames)
    if getattr(entry, "run_kwargs", None):
        # fixed mode arguments the executor binds (collect.py run_kwargs partial), e.g. skip_indexer=True
        extra.update(dict(entry.run_kwargs))
    return extra


def _run_case(run_func, case, perf_path: str, device: str, extra: dict) -> None:
    if isinstance(case, dict) and "params" in case:  # collect.py task shape {"id", "params"} (collect.py:1696-1702)
        params = case["params"]
        if isinstance(params, dict):
            run_func(**params, perf_filename=perf_path, device=device, **extra)
        else:
            run_func(*params, perf_filename=perf_path, device=device, **extra)
    elif isinstance(case, dict):  # keyword-shaped cases bind by name
        run_func(**{k: v for k, v in case.items() if k != "id"}, perf_filename=perf_path, device=device, **extra)
    else:
        run_func(*case, perf_filename=perf_path, device=device, **extra)


def run_planned(backend: str, op: str, *, out_dir: str = "/tmp/smk", device: str = "cuda:0", **select) -> None:
    """Run ONE planned case in this process, as the collector runs it: the row planned_row()
    builds for this SM, through the backend registry's run function and perf file. A capture
    that calls this names its cell as a planned case and nothing else."""
    from collector.helper import get_sm_version

    registry = importlib.import_module(f"collector.{backend}.registry")
    entries = [e for e in registry.REGISTRY if e.op == op]
    if len(entries) != 1:
        raise SystemExit(f"op {op!r}: {len(entries)} registry entries in collector.{backend}.registry, want 1")
    entry = entries[0]
    module = importlib.import_module(entry.module)
    run_func = getattr(module, entry.run_func)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    row = planned_row(backend, op, get_sm_version(), **select)
    print(f"[op_smoke] {backend} {op}: {entry.run_func}{row}")
    _run_case(
        run_func, row, str(Path(out_dir) / str(entry.perf_filename)), device, _bound_kwargs(entry, run_func, out_dir)
    )


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
    ap.add_argument(
        "--module",
        default=None,
        help="override the registry module (smoke an unrouted version lane); "
        "get/run function names and perf filename still come from the registry entry",
    )
    ap.add_argument("--cases", type=int, default=1, help="how many cases to run (from the head of the plan)")
    ap.add_argument(
        "--case-index", type=int, default=None, help="run exactly this case index instead of the head slice"
    )
    ap.add_argument(
        "--case-filter",
        default=None,
        help="comma list of substrings; run the first --cases cases whose str(case) "
        "contains all of them (collect.py --case-filter semantics) — picks a "
        "representative cell (model, precision, shape) without hand-typing the tuple",
    )
    ap.add_argument(
        "--case-prefix",
        default=None,
        help="literal prefix of str(case) (commas allowed, no splitting): pins the leading "
        'shape fields exactly, e.g. "[1, 4096, 32, 8, 128, 0, False, False, True"',
    )
    ap.add_argument(
        "--select",
        default=None,
        help="field=value pairs naming ONE planned case of the op (its op-plan case fields); runs the row "
        "the collector queues for it on this SM (AIS_SM overrides the device)",
    )
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

    if args.select:
        from collector.helper import get_sm_version

        cases = [planned_row(args.backend, args.op, get_sm_version(), **_parse_select(args.select))]
    else:
        cases = _get_cases(get_func, args.model_path)
    if not cases:
        raise SystemExit(
            f"{entry.get_func} returned 0 cases — nothing to smoke "
            "(platform floor or empty plan; see the drop log above)"
        )
    if args.case_filter or args.case_prefix:
        subs = [s.strip() for s in (args.case_filter or "").split(",") if s.strip()]
        matching = [
            c
            for c in cases
            if all(s in str(c) for s in subs) and (not args.case_prefix or str(c).startswith(args.case_prefix))
        ]
        if not matching:
            raise SystemExit(
                f"--case-filter {args.case_filter!r} --case-prefix {args.case_prefix!r} matches 0/{len(cases)} cases"
            )
        picked = matching[: args.cases]
        print(
            f"[op_smoke] --case-filter {args.case_filter!r} --case-prefix {args.case_prefix!r}: "
            f"{len(matching)}/{len(cases)} cases match"
        )
    else:
        picked = [cases[args.case_index]] if args.case_index is not None else cases[: args.cases]
    print(
        f"[op_smoke] {args.op}: module={module_name} run={entry.run_func} "
        f"perf={perf_path} — {len(picked)}/{len(cases)} case(s)"
    )
    extra = _bound_kwargs(entry, run_func, args.out_dir)
    for i, case in enumerate(picked):
        print(f"[op_smoke] case {i}: {case}")
        _run_case(run_func, case, perf_path, args.device, extra)
    print(f"[op_smoke] OK: {len(picked)} case(s) ran; rows appended to {perf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
