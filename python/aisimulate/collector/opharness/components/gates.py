#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: the path gates of gates.yaml, scheduled and graded.

  plan   --fw <backend> --sm <sm> --out <dir>
         A shard plan (``<dir>/<backend>.yaml``) with one ``probe.trace_case`` shard per
         gate planned on that SM (plus one per ``also`` case), for the collection pipeline:
         collect.py runs each planned case once under the profiler. A gate whose select
         matches no planned case on the SM is not a gate there.
  grade  --fw <backend> --sm <sm> --results <dir> [--version <v>]
         For every gate in scope, the ``probe_trace_case.json`` of its shards (found under
         ``<dir>``) against the serving record of its repo and kv on the same SM (probe_driver
         campaigns), through path_diff.diff; verdicts land in the workspace,
         results/pathdiff/<sm>/<fw>-<version>/<gate>.json. ``no_serving`` SMs get a
         platform-floor verdict. The version defaults to the manifest's default.

The table is the one home of the gates; workflow_check reads it through ``gates_in_scope``.
"""

from __future__ import annotations

import argparse
import json
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

    return sum(1 for case in plan(op, backend, sm).cases if case_matches(case, select)) == 1


def gates_in_scope(fw: str, sm: str | None = None, table: Path = TABLE) -> list[dict]:
    """The gates of backend ``fw``; with ``sm`` ("sm100"), only those planned on that SM."""
    gates = [g for g in load(table)["gates"] if g["backend"] == fw]
    if sm is None:
        return gates
    n = int(str(sm).removeprefix("sm"))
    return [g for g in gates if _matches(fw, g["op"], n, g["select"])]


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


def shard_plan(fw: str, sm: str) -> dict:
    shards = []
    for g in gates_in_scope(fw, sm):
        for name, select in _cases(g):
            op = select.pop("op", g["op"])
            shards.append({"op": SHARD_OP, "name": name, "case_filter": case_filter(fw, op, select, g.get("env"))})
    return {"backend": fw, "shards": shards}


def verdict_dir(sm: str, fw: str, version: str) -> Path:
    """Where the verdicts of (sm, fw, version) live: the probe workspace on the control machine."""
    sys.path.insert(0, str(HERE))
    import probe_driver

    return probe_driver.ROOT / "results" / "pathdiff" / sm / f"{fw}-{version}"


def _manifest_default(fw: str) -> str:
    manifest = yaml.safe_load((COLLECTOR / "framework_manifest.yaml").read_text())["frameworks"][fw]
    return str(manifest["default"]["version"])


def grade(fw: str, sm: str, results: Path, version: str | None = None) -> dict:
    """Grade every gate in scope; returns {gate: verdict}."""
    sys.path.insert(0, str(HERE))
    import path_diff

    version = version or _manifest_default(fw)
    out_dir = verdict_dir(sm, fw, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    hints = load()["hints"][fw]
    verdicts = {}
    for g in gates_in_scope(fw, sm):
        out = out_dir / f"{g['gate']}.json"
        floor = (g.get("no_serving") or {}).get(sm)
        if floor:
            record = {"verdict": "platform-floor", "gate_name": g["gate"], "repo": g["repo"], "framework": fw}
            record.update(version=version, sm=sm, note=floor, graded_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            out.write_text(json.dumps(record, indent=1))
            verdicts[g["gate"]] = "platform-floor"
            continue
        files = [
            next(iter(sorted(results.rglob(f"{SHARD_OP}__{name}/probe_trace_case.json"))), None)
            for name, _ in _cases(g)
        ]
        if not all(files):
            verdicts[g["gate"]] = "missing-trace"
            continue
        out.unlink(missing_ok=True)
        rc = path_diff.diff(
            [str(f) for f in files],
            g["repo"],
            fw,
            version,
            sm,
            hints[g["hint"]],
            save=str(out),
            kv_dtype=g["kv"],
            gate_name=g["gate"],
        )
        verdicts[g["gate"]] = "no-serving-record" if rc == 2 else json.loads(out.read_text())["verdict"]
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--fw", required=True)
    p.add_argument("--sm", required=True, help="e.g. sm100")
    p.add_argument("--out", required=True, type=Path)
    g = sub.add_parser("grade")
    g.add_argument("--fw", required=True)
    g.add_argument("--sm", required=True)
    g.add_argument("--results", required=True, type=Path, help="unpacked pipeline result dir (shard dirs inside)")
    g.add_argument("--version", default=None)
    args = ap.parse_args()
    if args.cmd == "plan":
        args.out.mkdir(parents=True, exist_ok=True)
        doc = shard_plan(args.fw, args.sm)
        (args.out / f"{args.fw}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, width=1000))
        print(f"{len(doc['shards'])} probe shards -> {args.out / f'{args.fw}.yaml'}")
        return 0
    verdicts = grade(args.fw, args.sm, args.results, args.version)
    for gate, verdict in sorted(verdicts.items()):
        print(f"{gate:44} {verdict}")
    return 0 if all(v in ("aligned", "platform-floor") for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
