#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report how much of the wheel the packaged production operation data takes.

Two sources, same report:

* ``--wheel PATH``: sizes read from an existing wheel (authoritative).
* default: an estimate from the source tree, deflating every file the maturin
  ``include`` list ships from ``systems/`` (what ``maturin build`` would add).

Packaged operation data is ``aisim_core/systems/data/**``; the rest of
``systems/`` (specs, version tables, profiles) is reported as ``systems-other``.
Operation data now comes from the pinned Hugging Face dataset, so the wheel must stay small:
``--max-wheel-mib N`` fails when the wheel file exceeds N MiB or still packages operation tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import zipfile
import zlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "aisim"
DATA_PREFIX = "aisim_core/systems/data/"
SYSTEMS_PREFIX = "aisim_core/systems/"


def bucket(name: str) -> str:
    """Report bucket for a wheel member name."""
    if name.startswith(DATA_PREFIX):
        parts = name[len(DATA_PREFIX) :].split("/")
        return f"data/{parts[0]}" if len(parts) > 1 else "data"
    if name.startswith(SYSTEMS_PREFIX):
        return "systems-other"
    if name.endswith((".so", ".pyd")):
        return "native"
    return "other"


def summarize(members: Iterable[tuple[str, int, int]]) -> dict:
    """``members`` are ``(name, size, compressed_size)``; returns totals and per-bucket sizes."""
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for name, size, compressed in members:
        entry = buckets[bucket(name)]
        entry[0] += 1
        entry[1] += size
        entry[2] += compressed
    total = [sum(v[i] for v in buckets.values()) for i in range(3)]
    data = [sum(v[i] for k, v in buckets.items() if k.startswith("data")) for i in range(3)]

    def row(v: list[int]) -> dict:
        return {"files": v[0], "bytes": v[1], "compressed_bytes": v[2]}

    return {
        "total": row(total),
        "packaged_op_data": row(data),
        "data_share_of_compressed": round(data[2] / total[2], 4) if total[2] else 0.0,
        "without_data_compressed_bytes": total[2] - data[2],
        "buckets": {k: row(v) for k, v in sorted(buckets.items())},
    }


def wheel_members(wheel: Path) -> list[tuple[str, int, int]]:
    with zipfile.ZipFile(wheel) as archive:
        return [(i.filename, i.file_size, i.compress_size) for i in archive.infolist() if not i.is_dir()]


def tree_members(package: Path = PACKAGE) -> list[tuple[str, int, int]]:
    """Estimate: deflate each file the maturin include list ships from ``src/aisim_core``."""
    config = tomllib.loads((package / "pyproject.toml").read_text())
    patterns = [p for p in config["tool"]["maturin"]["include"] if isinstance(p, str)]
    source = package / "src"
    seen: dict[str, Path] = {}
    for pattern in patterns:
        if not pattern.startswith("src/aisim_core/"):
            continue
        for path in source.parent.glob(pattern):
            if path.is_file():
                seen[path.relative_to(source).as_posix()] = path
    members = []
    for name, path in sorted(seen.items()):
        raw = path.read_bytes()
        members.append((name, len(raw), len(zlib.compress(raw, 6))))
    return members


WHEEL_LIMIT_MIB = 100


def check_wheel_limit(wheel: Path, limit_mib: float, report: dict) -> list[str]:
    """Problems that make a wheel unacceptable: over the size limit or still carrying op data."""
    problems = []
    size = wheel.stat().st_size
    if size > limit_mib * 1024 * 1024:
        problems.append(f"{wheel.name} is {size / 1024 / 1024:.1f} MiB, above the {limit_mib:g} MiB limit")
    if report["packaged_op_data"]["files"]:
        problems.append(f"{wheel.name} packages {report['packaged_op_data']['files']} operation data files")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, help="measure this wheel instead of estimating from the tree")
    parser.add_argument(
        "--max-wheel-mib",
        type=float,
        nargs="?",
        const=WHEEL_LIMIT_MIB,
        help=f"with --wheel: exit 1 if the wheel exceeds this size (default {WHEEL_LIMIT_MIB}) or packages op data",
    )
    args = parser.parse_args(argv)
    if args.max_wheel_mib is not None and not args.wheel:
        parser.error("--max-wheel-mib needs --wheel")
    members = wheel_members(args.wheel) if args.wheel else tree_members()
    report = {"source": str(args.wheel) if args.wheel else "source-tree-estimate", **summarize(members)}
    if args.wheel:
        report["wheel_file_bytes"] = args.wheel.stat().st_size
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if args.max_wheel_mib is not None:
        problems = check_wheel_limit(args.wheel, args.max_wheel_mib, report)
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
