# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover the exact own-repository comparison sources used for this study."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = "be0c44dfd1e319139d16659f2d96bf9a88337c53"
PREFIX = "data/experimental/deepseek-v41/verification-plan"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REVISION, PREFIX],
        cwd=args.repo,
        text=True,
    ).splitlines()
    hashes = {}
    for path in paths:
        if not path.endswith(".py"):
            continue
        data = subprocess.check_output(
            ["git", "show", f"{REVISION}:{path}"], cwd=args.repo
        )
        (args.output / Path(path).name).write_bytes(data)
        hashes[path] = hashlib.sha256(data).hexdigest()
    (args.output / "source-manifest.json").write_text(
        json.dumps({"revision": REVISION, "source_files_sha256": hashes}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
