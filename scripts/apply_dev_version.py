#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a dev-version suffix to every AISimulate version site, in place.

Invoked by nightly CI before `scripts/build_release_artifacts.py`. Takes one
argument -- a PEP 440 suffix like '.dev202609170000001234' -- and rewrites:
  - [project].version in Python package manifests (PEP 440 form)
  - [package].version in crates/core/Cargo.toml and
    [workspace.package].version in Cargo.toml (SemVer form: dash instead of
    dot, so '0.13.0-dev.202609170000001234' -- cargo rejects the PEP 440 spelling)
  - optional policy package versions and exact base-package dependency pins
  - local package versions in the shared Cargo.lock, so --locked builds
    retain the same resolved third-party dependencies after stamping

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


def stage_version(path: Path, tail: str, changes: dict[Path, str]) -> str:
    text = path.read_text()

    def bump(match: re.Match[str]) -> str:
        if match.group(2).endswith(tail):
            return match.group(0)  # already stamped
        return f"{match.group(1)}{match.group(2)}{tail}{match.group(3)}"

    new_text, count = VERSION_LINE_RE.subn(bump, text, count=1)
    if count != 1:
        raise SystemExit(f"no version line found in {path}")
    changes[path] = new_text
    return str(VERSION_LINE_RE.search(new_text).group(2))


def stage_lock(path: Path, package_versions: dict[str, str], changes: dict[Path, str]) -> None:
    """Stamp only local package records, preserving every resolved dependency."""
    if not path.is_file():
        return  # Historical release fixtures may not contain a lockfile.
    text = path.read_text()
    for name, version in package_versions.items():
        pattern = re.compile(r'(\[\[package\]\]\nname = "' + re.escape(name) + r'"\nversion = ")[^"]+("\n)')
        text, count = pattern.subn(lambda match: f"{match.group(1)}{version}{match.group(2)}", text)
        if count != 1:
            raise SystemExit(f"expected one local {name} package in {path}, found {count}")
    changes[path] = text


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
    # Validate every transformation before writing any file. Invalid pins or
    # lock records must not leave a partially stamped release source tree.
    changes: dict[Path, str] = {}
    stamped = [stage_version(root / PYPROJECT, args.suffix, changes)]
    stamped += [stage_version(root / path, semver(args.suffix), changes) for path in CARGO_MANIFESTS]
    package_versions = {"aisimulate-core": stamped[2]}
    plugin_py = root / "python/aisimulate-dynamo-policy/pyproject.toml"
    if plugin_py.is_file():
        plugin_native = root / "crates/dynamo-policy/Cargo.toml"
        stamped.append(stage_version(plugin_py, args.suffix, changes))
        stamped.append(stage_version(plugin_native, semver(args.suffix), changes))
        plugin_text, count = re.subn(r'"aisimulate==[^"]+"', f'"aisimulate=={stamped[0]}"', changes[plugin_py])
        if count != 1:
            raise SystemExit(f"expected one exact aisimulate pin in {plugin_py}, found {count}")
        changes[plugin_py] = plugin_text
        native_text, count = re.subn(
            r'(aisimulate-core\s*=\s*\{[^\n]*version\s*=\s*")=[^"]+("[^\n]*\})',
            rf"\g<1>={stamped[2]}\2",
            changes[plugin_native],
        )
        if count != 1:
            raise SystemExit(f"expected one exact aisimulate-core pin in {plugin_native}")
        changes[plugin_native] = native_text
        package_versions["aisimulate-dynamo-policy"] = stamped[-1]
    stage_lock(root / "Cargo.lock", package_versions, changes)
    for path, text in changes.items():
        path.write_text(text)
    print(f"apply_dev_version: {', '.join(stamped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
