#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: a collected run against the tables it can be read next to, before it is integrated.

A path gate proves one case runs serving's kernels. It cannot see a slow node, a key the new
release stopped serving, or a timing protocol that changed: those show only when the new
table is laid next to another one. This is that comparison, for every table at once.

  tables    <new root> [--base <root>] [--sibling SYSTEM]... [--system S] [--backend B] [--family F] [--detail]
            Both roots hold <system>/<family>/<backend>/<version>/<table>.parquet (the packaged
            data layout). A table is new when its version is the one the manifest pins for
            its backend and family. Three views, each with the same fields: rows, cases shared,
            lost and gained, the median new/other latency over the shared cases whose other
            latency is in the bottom half (``small``) and in the top fifth (``large``), and the
            cases more than 2x apart in each direction.
              previous   the same table at the greatest older version under the base root
                         (default: the new root); also the key buckets lost
              replaces   the table of the same version under the base root: what a recollection
                         replaces
              sibling    the same table of a system named with --sibling, for the new table and
                         for the base table (the replaced one, else the previous release, against
                         the sibling at that version), to read side by side
            ``spikes`` counts the rows of the new table far above both neighbours of their series.
  failures  <result dir> [...] [--base <result dir>...]
            Per shard directory (one holding ``errors_*.json`` next to its parquet): rows,
            failed tasks, their share, and the most frequent messages; ``partial`` counts the
            failed tasks that still wrote rows (a task that times several points and loses
            some, such as those above a capacity limit); with --base, the previous run of
            the same shards, the failed tasks that are new, gone and unchanged.

Reading ``tables`` (all are leads for triage, none is a verdict):
  large >= 1.25, small ~ 1   compute-bound rows slowed and launch-bound rows did not: a slow
                             node (recollect elsewhere) or a slower kernel (source check)
  both shifted               the timed region or the kernel changed: name it in the PR
  buckets lost               values of the exactly matched columns the previous table had and
                             this one lacks: a skip, a plan change or an unavailable key
  slow host?                 the new table sits above the sibling system where the base table
                             stood with it: recollect on another node
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
# a row this many times above both its neighbours along a swept column is an isolated spike
SPIKE = 4.0


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
    ratio = shared.new / shared.old
    out.update(over2=int((ratio > 2).sum()), under2=int((ratio < 0.5).sum()))
    if len(shared) >= 20:
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


def spikes(frame) -> int:
    """The rows more than SPIKE times above both neighbours along a swept column, the other case
    columns fixed. Latency grows along a sweep, so a row above the next point too is no trend."""
    keys = [c for c in frame.columns if c not in _NOT_KEY and frame[c].dtype.kind != "f"]
    hit = None
    for axis in [c for c in keys if c in _SWEPT]:
        ordered = frame.sort_values(axis, kind="stable")
        rest = [c for c in keys if c != axis]
        series = ordered.groupby(rest, dropna=False, sort=False)["latency"] if rest else ordered["latency"]
        above = (ordered["latency"] > SPIKE * series.shift(1)) & (ordered["latency"] > SPIKE * series.shift(-1))
        hit = above if hit is None else hit | above
    return 0 if hit is None else int(hit.sum())


def _read(path: Path):
    """The frame of a table, or None when there is no such table or it times nothing."""
    import pandas as pd

    frame = pd.read_parquet(path) if path.is_file() else None
    return frame if frame is not None and "latency" in frame.columns else None


def against_sibling(new, base, base_version: str | None, sibling_dirs: list[Path], version: str, table: str) -> dict:
    """The new table and the base table of one system, each against a sibling system's table of
    its own version (the first of ``sibling_dirs``, <root>/<sibling>/<family>/<backend>, that
    holds it). ``slow host?``: the new table sits LARGE_SLOWDOWN above the sibling on the
    launch-bound or the compute-bound cases while the base table stood below 1 + SHIFT."""
    view = {}
    for name, frame, at in (("new", new, version), ("base", base, base_version)):
        theirs = next((t for d in sibling_dirs if at and (t := _read(d / at / table)) is not None), None)
        if frame is not None and theirs is not None:
            view[name] = compare(frame, theirs)
    if "base" in view:
        view["base_version"] = base_version
        stats = [(view["new"].get(k), view["base"].get(k)) for k in ("small", "large")] if "new" in view else []
        slow = any(n is not None and b is not None and n >= LARGE_SLOWDOWN and b < 1 + SHIFT for n, b in stats)
        view["lead"] = "slow host?" if slow else ""
    return view


def tables(
    new_root: Path, base_root: Path, system=None, backend=None, family=None, alias=None, siblings=()
) -> list[dict]:
    """One row per new table under ``new_root``: the comparison with the previous release in
    the row itself, with the table it replaces under ``replaces`` and with each sibling system
    under ``siblings``. ``alias`` maps a system directory of the new root to its name under the
    base root."""
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
        new, old = _read(path), _read(base_dir / prev / path.name) if prev else None
        if new is not None and old is not None:
            row.update(compare(new, old))
            row["lead"] = lead(row)
        if new is not None:
            row["spikes"] = spikes(new)
            same = base_dir / version / path.name
            replaced = _read(same) if same.resolve() != path.resolve() else None
            if replaced is not None:
                row["replaces"] = compare(new, replaced)
                row["replaces"]["lead"] = lead(row["replaces"])
            base, base_version = (replaced, version) if replaced is not None else (old, prev)
            for sibling in siblings:
                dirs = [root / sibling / fam / be for root in (base_root, new_root)]
                view = against_sibling(new, base, base_version, dirs, version, path.name)
                if view:
                    row.setdefault("siblings", {})[sibling] = view
        rows.append(row)
    return rows


def _shard_errors(result_dirs: list[Path]) -> dict[Path, tuple[Path, list[dict]]]:
    """The failed tasks of every shard directory under the result dirs, keyed by the shard's
    path below its result dir (what two runs of the same shards share)."""
    found = {}
    for result_dir in result_dirs:
        for shard in sorted({f.parent for f in Path(result_dir).rglob("errors_*.json*")}):
            errors = []
            for f in shard.glob("errors_*.json*"):
                with (gzip.open if f.suffix == ".gz" else open)(f, "rt") as fh:
                    doc = json.load(fh)
                items = doc if isinstance(doc, list) else doc.get("errors", [])
                # UnresolvedFailures is the run's closing summary record, not a failed task
                errors += [e for e in items if isinstance(e, dict) and e.get("error_type") != "UnresolvedFailures"]
            found[shard.relative_to(result_dir)] = (shard, errors)
    return found


def _task(error: dict) -> str:
    return str(error.get("task_id") or error.get("task_params") or "")


# a task that times several points and loses some raises "<lost>/<points> point(s) ..." after the rest ran
_POINTS = re.compile(r"(\d+)/(\d+) point\(s\)")


def partial(errors: list[dict]) -> list[str]:
    """The failed tasks that still wrote rows. A task that times several points runs the others
    when some fail or are refused (above a capacity limit, say) and raises afterwards with the
    count, so it is recorded failed and is no whole-task failure. A series that loses exactly
    one point re-raises that point's own error, without a count, and is not recognised."""
    lost = (_POINTS.search(str(e.get("error_message") or "")) for e in errors)
    return [_task(e) for e, m in zip(errors, lost, strict=True) if m and int(m.group(1)) < int(m.group(2))]


def failures(result_dirs: list[Path], base_dirs: list[Path] | None = None) -> list[dict]:
    """One row per shard directory under the result dirs; with ``base_dirs`` (the previous run
    of the same shards) also the failed tasks that are new, gone and unchanged."""
    import pyarrow.parquet as pq

    before = _shard_errors(base_dirs) if base_dirs else {}
    rows = []
    for key, (shard, errors) in sorted(_shard_errors(result_dirs).items()):
        n_rows = sum(pq.read_metadata(p).num_rows for p in shard.glob("*.parquet"))
        first_lines = ((str(e.get("error_message") or "").strip().splitlines() or [""])[0] for e in errors)
        messages = Counter(re.sub(r"\d+", "N", line)[:110] for line in first_lines)
        row = {
            "shard": str(shard),
            "rows": n_rows,
            "failed": len(errors),
            "classes": dict(Counter(e.get("classification", "unclassified") for e in errors)),
            "messages": messages.most_common(3),
            "partial": partial(errors) if n_rows else [],
        }
        if base_dirs:
            now, then = {_task(e) for e in errors}, {_task(e) for e in before.get(key, (None, []))[1]}
            row.update(previous=key in before, new=sorted(now - then), gone=sorted(then - now))
            row["unchanged"] = len(now & then)
        rows.append(row)
    return rows


_VIEW = ("rows", "shared", "lost", "gained", "small", "large", "over2", "under2")


def _print_tables(rows: list[dict], detail: bool) -> None:
    head = ("system", "family", "backend", "table", "version", "previous", "rows", "shared", "lost", "gained")
    head += ("small", "large", "over2", "under2", "spikes")
    print(" ".join(head) + " lead")
    for r in rows:
        print(" ".join(str(r.get(k, "-")) for k in head), r.get("lead") or ("no-previous" if not r["previous"] else ""))
        views = [("replaces", r["replaces"], r["replaces"]["lead"])] if "replaces" in r else []
        for sibling, view in (r.get("siblings") or {}).items():
            views += [(f"{sibling} new", view["new"], view.get("lead", ""))] if "new" in view else []
            views += [(f"{sibling} base@{view['base_version']}", view["base"], "")] if "base" in view else []
        for name, view, word in views:
            print(f"    {name}: " + " ".join(f"{k}={view.get(k, '-')}" for k in _VIEW), word)
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
    t.add_argument("--sibling", action="append", default=[], metavar="SYSTEM", help="a system to read next to")
    t.add_argument("--detail", action="store_true", help="list the lost key buckets")
    t.add_argument("--json", type=Path, default=None, help="also write the rows here")
    f = sub.add_parser("failures")
    f.add_argument("results", nargs="+", type=Path)
    f.add_argument("--base", nargs="+", type=Path, default=None, help="result dirs of the previous run")
    args = ap.parse_args()
    if args.cmd == "failures":
        for r in failures(args.results, args.base):
            share = r["failed"] / max(r["rows"] + r["failed"], 1)
            line = f"{r['shard']}: rows={r['rows']} failed={r['failed']} ({share:.1%}) partial={len(r['partial'])}"
            if "new" in r:
                line += f" new={len(r['new'])} gone={len(r['gone'])} unchanged={r['unchanged']}"
                line += "" if r["previous"] else " no-previous-run"
            print(line, r["classes"])
            for message, count in r["messages"]:
                print(f"    {count:6d} {message}")
            for task in r.get("new", [])[:5]:
                print(f"    new: {task[:150]}")
        return 0
    alias = dict(a.split("=", 1) for a in args.alias)
    rows = tables(args.new, args.base or args.new, args.system, args.backend, args.family, alias, args.sibling)
    _print_tables(rows, args.detail)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
