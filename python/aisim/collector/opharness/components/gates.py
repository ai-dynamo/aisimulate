#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: the path gates of gates.yaml, scheduled and graded.

A gate runs at the release its op is collected with (collector/framework_manifest.yaml: the
framework default, or a family pin such as vLLM gemm/moe/kda): its trace runs in that image
and its serving record is probed at that release. Every command scopes by ``--fw`` (a comma
list; default every backend), ``--op`` (the gates whose case, or an ``also`` case, is that op:
one op across the frameworks), ``--repo`` or ``--family``.

  plan   --sm <sm> --out <dir> [scope] [--serving <campaign dir>]
         One shard plan per (backend, release), ``<dir>/<backend>-<version>.yaml``, with a
         ``probe.trace_case`` shard per gate planned on that SM (plus one per ``also`` case);
         ``--serving`` appends that campaign's probe.serving shards of the same (backend,
         release), so one job per plan runs both sides. A gate whose select matches no
         planned case on the SM is not a gate there.
  grade  --sm <sm> --results <dir> [<dir> ...] [scope]
         Every gate in scope: the ``probe_trace_case.json`` of its shards (found under the
         result dirs) against the serving record of its repo, kv, isl and release on the same
         SM (probe_driver campaigns), through path_diff.diff; verdicts land in the workspace,
         results/pathdiff/<sm>/<backend>-<version>/<gate>.json. ``no_serving`` SMs get a
         platform-floor verdict. The exit code is 0 when every gate in scope is aligned
         or a platform floor.

  paths  --sm <sm> [scope]
         What the gates prove, read from the op plans: every path a plan collects on that SM
         (the cases of one op that share a phase, the plan's dtype fields and its dispatch)
         with the gates naming a case on it, the ``ungated`` reason, the gate that can be
         derived for it, or NO GATE. No GPU.

``--derive`` (plan, grade; probe_driver --gates) adds a gate for each path no row names
whose cases name a roster checkpoint: the case nearest the probe point, graded against
that checkpoint at the case's KV dtype. A path with neither needs a row (a case that
names no model: gemm, attention) or a reason.

gates.yaml is the one home of the gates and of the reason an op has none (``ungated``);
``uncovered`` lists the registry ops with neither, and a unit test keeps that list empty.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
COLLECTOR = HARNESS.parent
TABLE = HERE / "gates.yaml"
SHARD_OP = "probe.trace_case"
# aisim-auto-collector's shard-plan charset for a case filter (security._CASE_FILTER_RE)
_CASE_FILTER = re.compile(r"^[A-Za-z0-9 _.,:/'\[\]()=+-]{1,256}$")


def load(table: Path = TABLE) -> dict:
    return yaml.safe_load(table.read_text())


@functools.cache
def _planned(backend: str, op: str, sm: int | None):
    """The op plan of (backend, op) on an SM (CPU only)."""
    sys.path.insert(0, str(COLLECTOR.parent))
    from collector.op_plans import plan

    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):  # planning prints model notes
        return plan(op, backend, sm)


def _matching(backend: str, op: str, sm: int, select: dict) -> list[int]:
    """The plan indices of the cases of (backend, op) on this SM that ``select`` matches."""
    sys.path.insert(0, str(COLLECTOR.parent))
    from collector.probes.trace_case import _matches as case_matches

    return [i for i, case in enumerate(_planned(backend, op, sm).cases) if case_matches(case, select)]


def _matches(backend: str, op: str, sm: int, select: dict) -> bool:
    """Exactly one planned case of (backend, op) on this SM matches ``select``."""
    return len(_matching(backend, op, sm, select)) == 1


def _ops(gate: dict) -> list[str]:
    return [gate["op"], *(a["op"] for a in gate.get("also") or [])]


def gates_in_scope(
    fw: str | None = None, sm: str | None = None, table: Path = TABLE, op: str | None = None, derive: bool = False
) -> list[dict]:
    """The gates of backend ``fw`` and op ``op`` (comma lists; None: every one); with ``sm``
    ("sm100"), only those planned on that SM, and with ``derive`` also the gates derived
    there for the paths no row names (:func:`derived`)."""
    fws = set(fw.split(",")) if fw else None
    ops = set(op.split(",")) if op else None
    gates = [
        g for g in load(table)["gates"] if (fws is None or g["backend"] in fws) and (ops is None or ops & set(_ops(g)))
    ]
    if sm is None:
        return gates
    n = int(str(sm).removeprefix("sm"))
    gates = [g for g in gates if _matches(g["backend"], g["op"], n, g["select"])]
    if derive:
        gates += [p["derived"] for p in paths(sm, fw, op, table) if p.get("derived")]
    return gates


def registry_ops(backend: str) -> list[str]:
    """The ops a backend's collector registry declares (the wide-EP registries are the
    multi-node plane and are not gated here)."""
    import importlib

    sys.path.insert(0, str(COLLECTOR.parent))
    try:
        registry = importlib.import_module(f"collector.{backend}.registry").REGISTRY
    except ModuleNotFoundError:
        return []
    return sorted({entry.op for entry in registry})


def _phases(op: str, backend: str) -> list[str]:
    """The ``phase`` values an op plans on any SM; [] when its cases carry none."""
    sys.path.insert(0, str(COLLECTOR.parent))
    from collector.op_plans import OP_PLANS, plan

    if op not in OP_PLANS or "phase" not in OP_PLANS[op].case_type._fields:
        return []
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        return sorted({case.phase for case in plan(op, backend, None).cases})


def uncovered(backend: str, table: Path = TABLE, ops: list[str] | None = None) -> list[str]:
    """Registry ops (``op``, or ``op:phase`` for an op whose cases carry a phase) of a backend
    that no gate names and ``ungated`` gives no reason for. Coverage is read from the
    registry, not the table: a rebuilt gate set once kept 19 of 32 gates and still reported
    every gate aligned (review 2026-09-29)."""
    doc = load(table)
    ungated = set(doc.get("ungated") or {})
    named = set()
    for g in doc["gates"]:
        if g["backend"] != backend:
            continue
        for op, select in [(g["op"], g["select"]), *((a["op"], a["select"]) for a in g.get("also") or [])]:
            named.add(op)
            if "phase" in select:
                named.add(f"{op}:{select['phase']}")
    missing = []
    for op in registry_ops(backend) if ops is None else ops:
        if op in ungated:
            continue
        if op not in named:
            missing.append(op)
            continue
        missing += [f"{op}:{ph}" for ph in _phases(op, backend) if f"{op}:{ph}" not in named | ungated]
    return missing


_MODEL_FIELDS = ("model_path", "model_name")


def _probe_point(isl: int) -> dict:
    """The point a serving record is probed at: one request of ``isl`` tokens on one GPU."""
    point = {"batch_size": 1, "seq_len": isl, "prefix_len": 0, "tp": 1, "tp_size": 1, "ep": 1}
    return {**point, "distribution": "balanced"}


def _derive(backend: str, op: str, key: tuple, cases: list, runners: list, roster: set[str], isl: int) -> dict | None:
    """The gate of a path no row names: its case nearest the probe point among those that
    name a roster checkpoint and run on this dispatch alone, selected by every field it has,
    graded against that checkpoint at the case's KV dtype. None when no case qualifies or
    its filter does not fit the pipeline."""
    point = _probe_point(isl)
    best = None
    for order, (case, on) in enumerate(zip(cases, runners, strict=True)):
        repo = next((getattr(case, f) for f in _MODEL_FIELDS if getattr(case, f, None)), None)
        if repo not in roster or len(on) != 1:
            continue
        off = sum(1 for f, v in point.items() if getattr(case, f, v) != v)
        rank = (off, -int(getattr(case, "num_heads", 0) or 0), order)
        if best is None or rank < best[0]:
            best = (rank, case, repo)
    if best is None:
        return None
    _rank, case, repo = best
    select = {}
    for field, value in case._asdict().items():
        if isinstance(value, tuple):  # a sweep: the probe point's values, or its smallest
            value = [v for v in value if v in (1, isl)] or [min(value)]
        if value is not None:
            select[field] = value
    try:
        case_filter(backend, op, select)
    except ValueError:
        return None
    phase, dtypes, dispatch = key
    name = "_".join(str(part) for part in ("auto", op, phase, *dtypes, dispatch, repo.rsplit("/", 1)[-1]) if part)
    kv = getattr(case, "kv_cache_dtype", "bfloat16")
    return {
        "gate": re.sub(r"[^A-Za-z0-9_.-]+", "-", name),
        "backend": backend,
        "op": op,
        "select": select,
        "repo": repo,
        "kv": "auto" if kv == "bfloat16" else kv,
    }


def paths(sm: str, fw: str | None = None, op: str | None = None, table: Path = TABLE) -> list[dict]:
    """The paths the plans in scope collect on ``sm`` and the gates that name them. A path is
    the cases of one (backend, op) that share a phase, the op plan's dtype fields and its
    dispatch: what selects the kernels. Each row carries ``gates`` (the table rows whose case
    is on the path), or ``derived`` (:func:`_derive`) when none is, or ``ungated`` (the
    table's reason for the op or its phase)."""
    sys.path[:0] = [str(HERE), str(COLLECTOR.parent)]
    import probe_driver
    from collector.op_plans import OP_PLANS

    n, isl = int(str(sm).removeprefix("sm")), probe_driver.DEFAULT_ISL
    doc = load(table)
    ungated = doc.get("ungated") or {}
    roster = set(probe_driver.roster_repos(probe_driver.load_targets(HARNESS / "targets.yaml")))
    rows = []
    for backend in fw.split(",") if fw else ("vllm", "sglang", "trtllm"):
        for case_op in sorted(set(registry_ops(backend)) & set(OP_PLANS)):
            if op and case_op not in op.split(","):
                continue
            spec, planned = OP_PLANS[case_op], _planned(backend, case_op, n)
            runners = planned.dispatches or [(None,)] * len(planned.cases)
            named: dict[int, list[str]] = {}
            for g in doc["gates"]:
                selects = [(g["op"], g["select"]), *((a["op"], a["select"]) for a in g.get("also") or [])]
                if g["backend"] != backend:
                    continue
                for i in [i for o, s in selects if o == case_op for i in _matching(backend, o, n, s)]:
                    named.setdefault(i, []).append(g["gate"])
            members: dict[tuple, list[int]] = {}
            for i, (case, on) in enumerate(zip(planned.cases, runners, strict=True)):
                for dispatch in on:
                    members.setdefault((getattr(case, "phase", None), spec.dtypes(case), dispatch), []).append(i)
            for key, idx in members.items():
                row = {"backend": backend, "op": case_op, "phase": key[0], "dtypes": key[1], "dispatch": key[2]}
                row["cases"] = len(idx)
                row["gates"] = sorted({name for i in idx for name in named.get(i, [])})
                reason = ungated.get(case_op) or ungated.get(f"{case_op}:{key[0]}")
                if reason:
                    row["ungated"] = reason
                elif not row["gates"]:
                    cases = [planned.cases[i] for i in idx]
                    row["derived"] = _derive(backend, case_op, key, cases, [runners[i] for i in idx], roster, isl)
                rows.append(row)
    return rows


@functools.cache
def op_version(backend: str, op: str) -> str:
    """The release (backend, op) is collected with: the framework default or its family pin."""
    sys.path.insert(0, str(COLLECTOR.parent))
    from collector.framework_manifest import resolve_op_runtime

    return resolve_op_runtime(backend, op).version


def release_image(backend: str, version: str) -> str:
    """The manifest image of (backend, release). The pipeline runs a shard plan on the
    framework's DEFAULT image unless AISIM_BACKEND_DOCKER_IMAGE names another, so a family
    pinned apart from the default must name its image at trigger time."""
    spec = yaml.safe_load((COLLECTOR / "framework_manifest.yaml").read_text())["frameworks"][backend]
    for runtime in [spec["default"], *(spec.get("families") or {}).values()]:
        if str(runtime["version"]) == version:
            return str(runtime["images"]["default"])
    raise KeyError(f"framework_manifest.yaml pins no {backend} {version}")


def gate_version(gate: dict) -> str:
    """The release a gate runs at; its ``also`` cases share it (tested)."""
    return op_version(gate["backend"], gate["op"])


_PHASE = {"context": "prefill", "generation": "decode"}


def gate_target(op: str, select: dict) -> tuple[tuple[str, ...], str | None]:
    """(the taxonomy roles a gate proves, the serving phase its evidence comes from), by its op:
    the gate name carries no meaning."""
    if op in ("gemm", "mla_bmm_gen_pre", "mla_bmm_gen_post"):
        return ("gemm",), None
    if op == "compute_scale":
        return ("quant",), None
    if op == "moe":
        return ("moe_gemm", "routing"), None
    if op == "mhc_module":
        return ("mhc",), None
    if op in ("gdn", "kda", "mamba2"):
        return ("linear_attention",), _PHASE.get(select.get("phase"))
    if op == "encoder_attention":  # the vision encoder runs in vLLM's profile run and the probes' vision forward
        return ("attention",), "profile_run"
    phase = "prefill" if "context" in op else "decode" if "generation" in op else None
    return ("attention", "dsa_indexer"), phase


def _value(v) -> str:
    return "+".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


def case_filter(backend: str, op: str, select: dict, env: dict | None = None, row: int | None = None) -> str:
    pairs = [f"backend={backend}", f"op={op}", *(f"{k}={_value(v)}" for k, v in select.items())]
    if row is not None:  # which row of a case the plan runs as several (probe.trace_case planned_row)
        pairs.append(f"row={int(row)}")
    pairs += [f"env.{k}={v}" for k, v in (env or {}).items()]
    text = ",".join(pairs)
    if not _CASE_FILTER.fullmatch(text):
        raise ValueError(f"{op} {select}: case filter {text!r} does not fit the pipeline's charset or 256 chars")
    return text


def _cases(gate: dict) -> list[tuple[str, dict]]:
    """(shard name, select) of the gate's own case and of each ``also`` case."""
    out = [(gate["gate"], dict(gate["select"]))]
    for extra in gate.get("also") or []:  # {op, select}: another planned case traced into the same verdict
        out.append((f"{gate['gate']}__{extra['op']}", {**extra["select"], "op": extra["op"]}))
    return out


def shard_plans(
    sm: str,
    fw: str | None = None,
    op: str | None = None,
    repo: str | None = None,
    family: str | None = None,
    derive: bool = False,
) -> dict[tuple[str, str], dict]:
    """{(backend, release): pipeline shard plan} of the gates in scope on ``sm``; ``repo`` /
    ``family`` narrow them to the gates naming a checkpoint or starting with a family."""
    plans: dict[tuple[str, str], dict] = {}
    for g in gates_in_scope(fw, sm, op=op, derive=derive):
        if (repo and g["repo"] != repo) or (family and not g["gate"].startswith(family)):
            continue
        doc = plans.setdefault((g["backend"], gate_version(g)), {"backend": g["backend"], "shards": []})
        for name, select in _cases(g):
            case_op = select.pop("op", g["op"])
            filt = case_filter(g["backend"], case_op, select, g.get("env"), g.get("row"))
            doc["shards"].append({"op": SHARD_OP, "name": name, "case_filter": filt})
    return plans


def serving_shards(campaign: Path, backend: str, version: str) -> list[dict]:
    """The single-GPU probe.serving shards probe_driver emitted for (backend, release) in a
    campaign: they share the trace job. A tp>1 run needs its own job with that many GPUs."""
    f = Path(campaign) / "shards" / "tp1" / f"{backend}-{version}.yaml"
    return yaml.safe_load(f.read_text())["shards"] if f.exists() else []


def verdict_dir(sm: str, fw: str, version: str) -> Path:
    """Where the verdicts of (sm, fw, version) live: the probe workspace on the control machine."""
    sys.path.insert(0, str(HERE))
    import probe_driver

    return probe_driver.ROOT / "results" / "pathdiff" / sm / f"{fw}-{version}"


def _trace(results: list[Path], name: str, backend: str, version: str, sm: int | None = None) -> Path | None:
    """The ``backend`` trace of shard ``name`` among the result dirs (shard names repeat across
    backends and systems), from a run on ``sm`` and at ``version`` when it says so. A trace that
    ran on another SM is never used (grading sm90 once read a B200 trace of the same shard); one
    that died before recording its SM is the last resort, and a trace from another release fails
    the runtime check and grades invalid."""
    found = []
    for d in results:
        for f in sorted(Path(d).rglob(f"{SHARD_OP}__{name}/probe_trace_case.json")):
            doc = json.loads(f.read_text())
            if (doc.get("args") or {}).get("backend") != backend:
                continue
            res = doc.get("result") or {}
            ran_sm = res.get("sm")
            if sm is not None and ran_sm is not None and int(ran_sm) != sm:
                continue
            found.append((ran_sm is None, res.get("framework_version") not in (version, None), f))
    return min(found)[2] if found else None


def grade(
    sm: str,
    results: list[Path],
    fw: str | None = None,
    op: str | None = None,
    repo: str | None = None,
    derive: bool = False,
) -> dict[tuple[str, str], str]:
    """Grade every gate in scope; returns {(backend, gate): verdict}."""
    sys.path.insert(0, str(HERE))
    import path_diff
    import probe_driver

    verdicts = {}
    for g in gates_in_scope(fw, sm, op=op, derive=derive):
        if repo and g["repo"] != repo:
            continue
        backend, version, key = g["backend"], gate_version(g), (g["backend"], g["gate"])
        out_dir = verdict_dir(sm, backend, version)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{g['gate']}.json"
        floor = (g.get("no_serving") or {}).get(sm)
        if floor:
            record = {"verdict": "platform-floor", "gate_name": g["gate"], "repo": g["repo"], "framework": backend}
            record.update(version=version, sm=sm, note=floor, graded_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            out.write_text(json.dumps(record, indent=1))
            verdicts[key] = "platform-floor"
            continue
        files = [_trace(results, name, backend, version, int(sm.removeprefix("sm"))) for name, _ in _cases(g)]
        out.unlink(missing_ok=True)  # a verdict of an earlier run never stands in for this one
        if not all(files):
            verdicts[key] = "missing-trace"
            continue
        roles, phase = gate_target(g["op"], g["select"])
        rc = path_diff.diff(
            [str(f) for f in files],
            g["repo"],
            backend,
            version,
            sm,
            save=str(out),
            kv_dtype=g["kv"],
            gate_name=g["gate"],
            target_roles=roles,
            phase=phase,
            isl=g.get("isl", probe_driver.DEFAULT_ISL),
        )
        verdicts[key] = "no-serving-record" if rc == 2 else json.loads(out.read_text())["verdict"]
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    g = sub.add_parser("grade")
    c = sub.add_parser("paths")
    for parser in (p, g, c):
        parser.add_argument("--sm", required=True, help="e.g. sm100")
        parser.add_argument("--fw", default=None, help="comma list of backends (default: every backend)")
        parser.add_argument("--op", default=None, help="only the gates of these collector ops (comma list)")
    for parser in (p, g):
        parser.add_argument("--repo", default=None, help="only the gates that name this checkpoint")
        parser.add_argument("--derive", action="store_true", help="also the gates derived for unnamed paths")
    p.add_argument("--family", default=None, help="only the gates whose name starts with this family")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--serving", default=None, type=Path, help="a probe_driver campaign dir whose shards to add")
    g.add_argument("--results", required=True, nargs="+", type=Path, help="unpacked pipeline result dirs")
    args = ap.parse_args()
    if args.cmd == "paths":
        rows = paths(args.sm, args.fw, args.op)
        for r in rows:
            d = r.get("derived")
            how = ", ".join(r["gates"]) or (f"ungated: {r['ungated']}" if "ungated" in r else "")
            how = how or (f"derivable: {d['repo']} kv={d['kv']}" if d else "NO GATE")
            path = f"{r['phase'] or '-'} {'/'.join(r['dtypes']) or '-'} {r['dispatch'] or '-'}"
            print(f"{r['backend']:7} {r['op']:34} {path:44} {r['cases']:6}  {how}")
        gated = sum(1 for r in rows if r["gates"])
        derivable = sum(1 for r in rows if r.get("derived"))
        bare = sum(1 for r in rows if not r["gates"] and "ungated" not in r and not r.get("derived"))
        print(f"{args.sm}: {len(rows)} paths, {gated} gated, {derivable} derivable, {bare} with no gate")
        return 0
    if args.cmd == "plan":
        args.out.mkdir(parents=True, exist_ok=True)
        plans = shard_plans(args.sm, args.fw, args.op, args.repo, args.family, args.derive)
        for f in sorted(args.serving.glob("shards/tp*/*.yaml")) if args.serving else []:
            if f.parent.name != "tp1":
                print(f"{f}: its own job, AISIM_GPU_NUM={f.parent.name.removeprefix('tp')}")
        for (backend, version), doc in sorted(plans.items()):
            if args.serving:
                doc["shards"] = serving_shards(args.serving, backend, version) + doc["shards"]
            path = args.out / f"{backend}-{version}.yaml"
            path.write_text(yaml.safe_dump(doc, sort_keys=False, width=1000))
            image = release_image(backend, version)
            print(f"{len(doc['shards'])} shards -> {path}")
            print(f"  trigger: --frameworks {backend}:{version} --var AISIM_BACKEND_DOCKER_IMAGE={image}")
        return 0
    verdicts = grade(args.sm, args.results, args.fw, args.op, args.repo, args.derive)
    for (backend, gate), verdict in sorted(verdicts.items()):
        print(f"{backend:7} {gate:44} {verdict}")
    return 0 if all(v in ("aligned", "platform-floor") for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
