#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: a collected run against the previous release, before its tables are integrated.

A path gate proves one case runs serving's kernels. It cannot see a slow node, a key the new
release stopped serving, or a timing protocol that changed: those show only when the new
table is laid next to the previous one. This is that comparison, for every table at once.

  tables    <new root> [--base <root>] [--system S] [--backend B] [--family F] [--detail]
            Both roots hold <system>/<family>/<backend>/<version>/<table>.parquet (the packaged
            data layout). A table is new when its version is the one the manifest pins for
            its backend and family; it is compared with the same table at the greatest older
            version under the base root (default: the new root). Per table: rows, cases shared, lost and
            gained, key buckets lost, and the median new/old latency over the shared cases
            whose old latency is in the bottom half (``small``) and in the top fifth
            (``large``).
  failures  <result dir> [...]
            Per shard directory (one holding ``errors_*.json`` next to its parquet): rows,
            failed tasks, their share, and the most frequent messages.

Reading ``tables`` (all are leads for triage, none is a verdict):
  large >= 1.25, small ~ 1   compute-bound rows slowed and launch-bound rows did not: a slow
                             node (recollect elsewhere) or a slower kernel (source check)
  both shifted               the timed region or the kernel changed: name it in the PR
  buckets lost               values of the exactly matched columns the previous table had and
                             this one lacks: a skip, a plan change or an unavailable key
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
COLLECTOR = HERE.parents[1]
# columns that are not part of a case: where and what ran, and what was measured
_NOT_KEY = frozenset({"framework", "version", "device", "kernel_source", "latency", "power", "power_limit"})
# case columns the SDK interpolates over; every other case column is matched exactly and
# names a key bucket (.claude/rules/collector/overview.md, "Failures are counted per consumer key")
_SWEPT = frozenset(
    {"batch_size", "isl", "seq_len", "num_tokens", "num_token", "step", "prefix", "m", "n", "k", "message_size"}
)
LARGE_SLOWDOWN = 1.25
SHIFT = 0.15


def pinned_version(backend: str, family: str) -> str | None:
    """The release the manifest pins for a backend's data family: the family pin or the default."""
    spec = yaml.safe_load((COLLECTOR / "framework_manifest.yaml").read_text())["frameworks"].get(backend)
    if not spec:
        return None
    return str(((spec.get("families") or {}).get(family) or spec["default"])["version"])


def _version_key(version: str):
    from packaging.version import InvalidVersion, Version

    try:
        return (0, Version(version))
    except InvalidVersion:
        return (1, version)


def previous_version(table_dir: Path, version: str, table: str) -> str | None:
    """The greatest version older than ``version`` under a <backend>/ directory that holds the table."""
    older = [
        d.name
        for d in table_dir.iterdir()
        if d.is_dir() and (d / table).exists() and _version_key(d.name) < _version_key(version)
    ]
    return max(older, key=_version_key) if older else None


def compare(new, old) -> dict:
    """The comparison of two frames of one table (see the module docstring for the fields)."""
    keys = [c for c in new.columns if c not in _NOT_KEY and c in old.columns and new[c].dtype.kind != "f"]
    buckets = [c for c in keys if c not in _SWEPT]
    out = {"rows": len(new), "rows_old": len(old), "keys": keys, "buckets": buckets}
    n = new.groupby(keys, dropna=False)["latency"].min()
    o = old.groupby(keys, dropna=False)["latency"].min()
    shared = n.to_frame("new").join(o.to_frame("old"), how="inner")
    out.update(shared=len(shared), lost=len(o) - len(shared), gained=len(n) - len(shared))
    if buckets:
        had = set(map(tuple, old[buckets].drop_duplicates().itertuples(index=False)))
        has = set(map(tuple, new[buckets].drop_duplicates().itertuples(index=False)))
        out["buckets_lost"] = sorted(had - has, key=str)
        out["buckets_gained"] = len(has - had)
    else:
        out["buckets_lost"], out["buckets_gained"] = [], 0
    shared = shared[(shared.old > 0) & (shared.new > 0)]
    if len(shared) >= 20:
        ratio = shared.new / shared.old
        out["small"] = round(float(ratio[shared.old <= shared.old.quantile(0.5)].median()), 2)
        out["large"] = round(float(ratio[shared.old >= shared.old.quantile(0.8)].median()), 2)
    return out


def lead(result: dict) -> str:
    """The one-word reading of a comparison; empty when nothing stands out."""
    words = []
    small, large = result.get("small"), result.get("large")
    if large is not None:
        if large >= LARGE_SLOWDOWN and abs(small - 1) < SHIFT:
            words.append("large-slower")
        elif abs(large - 1) >= SHIFT or abs(small - 1) >= SHIFT:
            words.append("shifted")
    if result["buckets_lost"]:
        words.append("buckets-lost")
    return ",".join(words)


def tables(new_root: Path, base_root: Path, system=None, backend=None, family=None, alias=None) -> list[dict]:
    """One comparison per new table under ``new_root``. ``alias`` maps a system directory of
    the new root to its name under the base root."""
    import pandas as pd

    rows = []
    for path in sorted(new_root.glob("*/*/*/*/*.parquet")):
        sys_name, fam, be, version = path.relative_to(new_root).parts[:4]
        if (system and sys_name != system) or (backend and be != backend) or (family and fam != family):
            continue
        if version != pinned_version(be, fam):
            continue
        row = {"system": sys_name, "family": fam, "backend": be, "version": version, "table": path.stem}
        base_dir = base_root / (alias or {}).get(sys_name, sys_name) / fam / be
        prev = previous_version(base_dir, version, path.name) if base_dir.is_dir() else None
        row["previous"] = prev
        if prev:
            new, old = pd.read_parquet(path), pd.read_parquet(base_dir / prev / path.name)
            if "latency" in new.columns and "latency" in old.columns:
                row.update(compare(new, old))
                row["lead"] = lead(row)
        rows.append(row)
    return rows


def failures(result_dirs: list[Path]) -> list[dict]:
    """One row per shard directory under the result dirs."""
    import pyarrow.parquet as pq

    rows = []
    shard_dirs = sorted({f.parent for d in result_dirs for f in Path(d).rglob("errors_*.json*")})
    for shard in shard_dirs:
        errors = []
        for f in shard.glob("errors_*.json*"):
            with (gzip.open if f.suffix == ".gz" else open)(f, "rt") as fh:
                doc = json.load(fh)
            items = doc if isinstance(doc, list) else doc.get("errors", [])
            # UnresolvedFailures is the run's closing summary record, not a failed task
            errors += [e for e in items if isinstance(e, dict) and e.get("error_type") != "UnresolvedFailures"]
        n_rows = sum(pq.read_metadata(p).num_rows for p in shard.glob("*.parquet"))
        first_lines = ((str(e.get("error_message") or "").strip().splitlines() or [""])[0] for e in errors)
        messages = Counter(re.sub(r"\d+", "N", line)[:110] for line in first_lines)
        rows.append(
            {
                "shard": str(shard),
                "rows": n_rows,
                "failed": len(errors),
                "classes": dict(Counter(e.get("classification", "unclassified") for e in errors)),
                "messages": messages.most_common(3),
            }
        )
    return rows


def _print_tables(rows: list[dict], detail: bool) -> None:
    head = ("system", "family", "backend", "table", "version", "previous", "rows", "shared", "lost", "gained")
    head += ("small", "large")
    print(" ".join(head) + " lead")
    for r in rows:
        print(" ".join(str(r.get(k, "-")) for k in head), r.get("lead") or ("no-previous" if not r["previous"] else ""))
        if detail and r.get("buckets_lost"):
            print(f"    buckets ({', '.join(r['buckets'])}) lost: {len(r['buckets_lost'])}")
            for bucket in r["buckets_lost"][:8]:
                print(f"      {bucket}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tables")
    t.add_argument("new", type=Path)
    t.add_argument("--base", type=Path, default=None)
    t.add_argument("--system", default=None)
    t.add_argument("--backend", default=None)
    t.add_argument("--family", default=None)
    t.add_argument("--alias", action="append", default=[], metavar="NEW=BASE", help="a system named apart")
    t.add_argument("--detail", action="store_true", help="list the lost key buckets")
    t.add_argument("--json", type=Path, default=None, help="also write the rows here")
    f = sub.add_parser("failures")
    f.add_argument("results", nargs="+", type=Path)
    args = ap.parse_args()
    if args.cmd == "failures":
        for r in failures(args.results):
            share = r["failed"] / max(r["rows"] + r["failed"], 1)
            print(f"{r['shard']}: rows={r['rows']} failed={r['failed']} ({share:.1%}) {r['classes']}")
            for message, count in r["messages"]:
                print(f"    {count:6d} {message}")
        return 0
    alias = dict(a.split("=", 1) for a in args.alias)
    rows = tables(args.new, args.base or args.new, args.system, args.backend, args.family, alias)
    _print_tables(rows, args.detail)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
