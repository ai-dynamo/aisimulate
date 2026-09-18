# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute a checkout-location-independent digest for staged collector code."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_tree_sha256(root: Path) -> str:
    """Hash sorted relative paths and content digests beneath ``root``."""
    root = root.resolve()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix()
    )
    if not files:
        raise ValueError(f"collector code root contains no files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(compute_tree_sha256(args.root))


if __name__ == "__main__":
    main()
