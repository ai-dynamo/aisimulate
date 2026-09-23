#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a dev-version suffix to every AISimulate version site, in place.

Invoked by nightly CI before `scripts/build_release_artifacts.py`. Takes one
argument -- a PEP 440 suffix like '.dev202609170000001234' -- and rewrites:
  - [project].version in python/aisimulate/pyproject.toml (PEP 440 form)
  - [package].version in crates/core/Cargo.toml and
    [workspace.package].version in Cargo.toml (SemVer form: dash instead of
    dot, so '0.13.0-dev.202609170000001234' -- cargo rejects the PEP 440 spelling)

The suffix is the UTC creation date followed by a ten-digit workflow run
number; legacy date-only suffixes remain accepted. The crate form keeps the
suffix as a dotted numeric identifier so SemVer pre-release
ordering compares it numerically (matching the crate's published lineage on
crates.io). Idempotent: re-running with the same suffix is a no-op, and an
empty suffix changes nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path("python/aisimulate/pyproject.toml")
# Each Cargo manifest below has exactly one line-anchored `version = "..."`
# (external deps use inline tables, which the regex skips). The publish=false
# test crate pins no version on its path dep, so it needs no stamping.
CARGO_MANIFESTS = [Path("Cargo.toml"), Path("crates/core/Cargo.toml")]

VERSION_LINE_RE = re.compile(r'^(\s*version\s*=\s*")([^"]+)(")\s*$', re.MULTILINE)

SUFFIX_RE = re.compile(r"^\.dev[0-9]{8}(?:[0-9]{10})?$")


def semver(suffix: str) -> str:
    return "-dev." + suffix[len(".dev") :]


def rewrite(path: Path, tail: str) -> str:
    text = path.read_text()

    def bump(match: re.Match[str]) -> str:
        if match.group(2).endswith(tail):
            return match.group(0)  # already stamped
        return f"{match.group(1)}{match.group(2)}{tail}{match.group(3)}"

    new_text, count = VERSION_LINE_RE.subn(bump, text, count=1)
    if count != 1:
        raise SystemExit(f"no version line found in {path}")
    path.write_text(new_text)
    return str(VERSION_LINE_RE.search(new_text).group(2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suffix", help="e.g. .dev20260827 (empty = no-op)")
    parser.add_argument("root", nargs="?", default=".", help="repo root")
    args = parser.parse_args()

    if not args.suffix:
        print("apply_dev_version: empty suffix, no-op", file=sys.stderr)
        return 0
    if not SUFFIX_RE.fullmatch(args.suffix):
        raise SystemExit(f"suffix must be .devYYYYMMDD with an optional ten-digit run number, got {args.suffix!r}")

    root = Path(args.root).resolve()
    stamped = [rewrite(root / PYPROJECT, args.suffix)]
    stamped += [rewrite(root / path, semver(args.suffix)) for path in CARGO_MANIFESTS]
    print(f"apply_dev_version: {', '.join(stamped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
