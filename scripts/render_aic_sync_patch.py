#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render an AIConfigurator diff against AISimulate's stable mirror paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "scripts" / "aic_sync.toml"


def _git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(source), *args))


def _require_source_path(source: Path, ref: str, path: str) -> None:
    result = subprocess.run(
        ("git", "-C", str(source), "cat-file", "-e", f"{ref}:{path}"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"AIC sync source path does not exist at {ref}: {path}")


def render(source: Path, from_ref: str, to_ref: str) -> bytes:
    config = tomllib.loads(LEDGER.read_text())
    _git(source, "rev-parse", "--verify", f"{from_ref}^{{commit}}")
    _git(source, "rev-parse", "--verify", f"{to_ref}^{{commit}}")

    chunks: list[bytes] = []
    for mapping in config["mirror"]:
        upstream = str(mapping["source"]).strip("/")
        target = str(mapping["target"]).strip("/")
        _require_source_path(source, from_ref, upstream)
        patch = _git(
            source,
            "diff",
            "--binary",
            "--full-index",
            "--find-renames",
            f"--relative={upstream}",
            f"--src-prefix=a/{target}/",
            f"--dst-prefix=b/{target}/",
            from_ref,
            to_ref,
            "--",
            upstream,
        )
        if patch:
            chunks.append(patch.rstrip(b"\n") + b"\n")
    return b"".join(chunks)


def main() -> None:
    config = tomllib.loads(LEDGER.read_text())
    parser = argparse.ArgumentParser(
        description="Create a binary-safe, path-rewritten AIC synchronization patch."
    )
    parser.add_argument("--source", type=Path, required=True, help="AIC git checkout")
    parser.add_argument(
        "--from-ref", default=config["upstream"]["last_synced"], help="old AIC ref"
    )
    parser.add_argument("--to-ref", required=True, help="new AIC ref")
    parser.add_argument("--output", type=Path, help="write patch here (default: stdout)")
    args = parser.parse_args()

    patch = render(args.source.resolve(), args.from_ref, args.to_ref)
    if args.output:
        args.output.write_bytes(patch)
    else:
        sys.stdout.buffer.write(patch)


if __name__ == "__main__":
    main()
