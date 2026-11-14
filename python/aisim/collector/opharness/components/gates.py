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
         platform-floor verdict.

The table is the one home of the gates; workflow_check reads it through ``gates_in_scope``.
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


def _matches(backend: str, op: str, sm: int, select: dict) -> bool:
    """Exactly one planned case of (backend, op) on this SM matches ``select`` (CPU only)."""
    sys.path.insert(0, str(COLLECTOR.parent))
    from collector.op_plans import plan
    from collector.probes.trace_case import _matches as case_matches

    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):  # planning prints model notes
        cases = plan(op, backend, sm).cases
    return sum(1 for case in cases if case_matches(case, select)) == 1


def _ops(gate: dict) -> list[str]:
    return [gate["op"], *(a["op"] for a in gate.get("also") or [])]


def gates_in_scope(
    fw: str | None = None, sm: str | None = None, table: Path = TABLE, op: str | None = None
) -> list[dict]:
    """The gates of backend ``fw`` and op ``op`` (comma lists; None: every one); with ``sm``
    ("sm100"), only those planned on that SM."""
    fws = set(fw.split(",")) if fw else None
    ops = set(op.split(",")) if op else None
    gates = [
        g for g in load(table)["gates"] if (fws is None or g["backend"] in fws) and (ops is None or ops & set(_ops(g)))
    ]
    if sm is None:
        return gates
    n = int(str(sm).removeprefix("sm"))
    return [g for g in gates if _matches(g["backend"], g["op"], n, g["select"])]


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


def case_filter(backend: str, op: str, select: dict, env: dict | None = None) -> str:
    pairs = [f"backend={backend}", f"op={op}", *(f"{k}={_value(v)}" for k, v in select.items())]
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
) -> dict[tuple[str, str], dict]:
    """{(backend, release): pipeline shard plan} of the gates in scope on ``sm``; ``repo`` /
    ``family`` narrow them as workflow_check scopes onboard_model / new_op_collector."""
    plans: dict[tuple[str, str], dict] = {}
    for g in gates_in_scope(fw, sm, op=op):
        if (repo and g["repo"] != repo) or (family and not g["gate"].startswith(family)):
            continue
        doc = plans.setdefault((g["backend"], gate_version(g)), {"backend": g["backend"], "shards": []})
        for name, select in _cases(g):
            case_op = select.pop("op", g["op"])
            filt = case_filter(g["backend"], case_op, select, g.get("env"))
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


def _trace(results: list[Path], name: str, backend: str, version: str) -> Path | None:
    """The ``backend`` trace of shard ``name`` among the result dirs (shard names repeat across
    backends), from a run at ``version`` when one says so (a trace from another release fails the
    runtime check and grades invalid)."""
    found = []
    for d in results:
        for f in sorted(Path(d).rglob(f"{SHARD_OP}__{name}/probe_trace_case.json")):
            doc = json.loads(f.read_text())
            if (doc.get("args") or {}).get("backend") == backend:
                found.append((f, (doc.get("result") or {}).get("framework_version")))
    for f, ran in found:
        if ran in (version, None):
            return f
    return found[0][0] if found else None


def grade(
    sm: str, results: list[Path], fw: str | None = None, op: str | None = None, repo: str | None = None
) -> dict[tuple[str, str], str]:
    """Grade every gate in scope; returns {(backend, gate): verdict}."""
    sys.path.insert(0, str(HERE))
    import path_diff
    import probe_driver

    verdicts = {}
    for g in gates_in_scope(fw, sm, op=op):
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
        files = [_trace(results, name, backend, version) for name, _ in _cases(g)]
        if not all(files):
            verdicts[key] = "missing-trace"
            continue
        out.unlink(missing_ok=True)
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
    for parser in (p, g):
        parser.add_argument("--sm", required=True, help="e.g. sm100")
        parser.add_argument("--fw", default=None, help="comma list of backends (default: every backend)")
        parser.add_argument("--op", default=None, help="only the gates of these collector ops (comma list)")
        parser.add_argument("--repo", default=None, help="only the gates that name this checkpoint")
    p.add_argument("--family", default=None, help="only the gates whose name starts with this family")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--serving", default=None, type=Path, help="a probe_driver campaign dir whose shards to add")
    g.add_argument("--results", required=True, nargs="+", type=Path, help="unpacked pipeline result dirs")
    args = ap.parse_args()
    if args.cmd == "plan":
        args.out.mkdir(parents=True, exist_ok=True)
        plans = shard_plans(args.sm, args.fw, args.op, args.repo, args.family)
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
    verdicts = grade(args.sm, args.results, args.fw, args.op, args.repo)
    for (backend, gate), verdict in sorted(verdicts.items()):
        print(f"{backend:7} {gate:44} {verdict}")
    return 0 if all(v in ("aligned", "platform-floor") for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
