# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation probes: one question about serving, answered on a GPU, never a perf table.

A probe is a module here with ``run(args: dict[str, str], out_dir: Path) -> dict``
(``"failed": True`` in the dict makes the exit code 1, with the result still written). It
runs through collect.py as ``--ops probe.<module>`` with its arguments as
``--case-filter key=value,key=value``, so the collection pipeline schedules it like any
shard (a shard plan entry ``{op: probe.<module>, name: <label>, case_filter: "..."}``,
run in the image the pipeline resolves). The result is ``probe_<module>.json`` in the
working directory, which is the shard directory under the pipeline. A probe has no plan,
no rows and no collection_meta; cite its job at the code that uses its answer.
"""

from __future__ import annotations

import importlib
import json
import traceback
from pathlib import Path

PROBE_PREFIX = "probe."


def probe_op_names() -> list[str]:
    here = Path(__file__).resolve().parent
    return sorted(f"{PROBE_PREFIX}{p.stem}" for p in here.glob("*.py") if not p.stem.startswith("_"))


def parse_args(filters: list[str]) -> dict[str, str]:
    """``["a=1,b=x", "c=2"]`` -> ``{"a": "1", "b": "x", "c": "2"}``; a pair without ``=`` is an error."""
    args: dict[str, str] = {}
    for text in filters:
        for pair in (item.strip() for item in text.split(",")):
            if not pair:
                continue
            key, sep, value = pair.partition("=")
            if not sep or not key:
                raise ValueError(f"probe argument {pair!r} is not key=value")
            args[key.strip()] = value.strip()
    return args


def run_probe(name: str, filters: list[str], out_dir: Path) -> int:
    """Run probe ``name``; write ``probe_<name>.json`` into ``out_dir``. Exit code 0 on success."""
    out = Path(out_dir) / f"probe_{name}.json"
    record: dict = {"probe": name}
    try:
        record["args"] = parse_args(filters)
        module = importlib.import_module(f"collector.probes.{name}")
        record["result"] = module.run(record["args"], Path(out_dir))
        status = 1 if record["result"].get("failed") else 0
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-4000:]
        status = 1
    out.write_text(json.dumps(record, indent=1, default=str))
    print(f"probe.{name}: {'ok' if status == 0 else record.get('error', 'failed')} -> {out}", flush=True)
    return status
