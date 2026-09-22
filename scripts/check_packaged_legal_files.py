#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify wheel-local legal files are exact copies of the root originals."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python" / "aisimulate"
LEGAL_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md")


def main() -> int:
    mismatched: list[str] = []
    packages = [PACKAGE_ROOT]
    plugin = ROOT / "python" / "aisimulate-dynamo-policy"
    if plugin.is_dir():
        packages.append(plugin)
    for package in packages:
        for name in LEGAL_FILES:
            canonical = ROOT / name
            packaged = package / name
            if not canonical.is_file() or not packaged.is_file():
                mismatched.append(f"{package.name}/{name}: canonical or packaging copy is missing")
            elif canonical.read_bytes() != packaged.read_bytes():
                mismatched.append(f"{package.name}/{name}: packaged copy differs from root")

    if mismatched:
        for message in mismatched:
            print(message)
        return 1

    print("wheel-local LICENSE and THIRD_PARTY_NOTICES.md match the root files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
