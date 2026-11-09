# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash the vLLM files that the FPM observers read, at each audited tag.

Prints the SOURCE_MANIFESTS constant of fpm_memory_observer.py. With --check, compares the hashes
of the pinned commits with that constant and returns 1 on a difference.

The list of files is the set of keys of the committed constant, which comes from the observer audit.
The script recomputes the hashes of those files at the pinned tags and checks them. It does not
discover files: a new file must be added to the list by hand at an audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .runtime import fpm_memory_observer as observer

VLLM_TAGS: dict[str, tuple[str, str]] = {
    "0.32.0": ("v0.32.0", "5df63a7203cdb9328d824a27b682e77c7b9928be"),
    "0.33.0": ("v0.33.0", "4ba787ec9959e566d3b587a796dff07673324da5"),
}


def _run(vllm_source: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(vllm_source), *args], capture_output=True)


def _git(vllm_source: Path, *args: str) -> bytes:
    result = _run(vllm_source, *args)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.decode().strip()}")
    return result.stdout


def build_manifests(vllm_source: Path) -> dict[str, dict[str, str]]:
    manifests: dict[str, dict[str, str]] = {}
    for release, (tag, commit) in VLLM_TAGS.items():
        resolved = _run(vllm_source, "rev-parse", f"{tag}^{{commit}}")
        if resolved.returncode:
            raise ValueError(f"{tag} does not resolve to {commit}: {resolved.stderr.decode().strip()}")
        if resolved.stdout.decode().strip() != commit:
            raise ValueError(f"{tag} does not resolve to {commit}")
        manifests[release] = {
            path: hashlib.sha256(_git(vllm_source, "show", f"{commit}:{path}")).hexdigest()
            for path in observer.SOURCE_MANIFESTS[release]
        }
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True, help="vLLM git checkout")
    parser.add_argument("--check", action="store_true", help="compare with SOURCE_MANIFESTS instead of printing")
    args = parser.parse_args(argv)
    manifests = build_manifests(args.vllm_source)
    if not args.check:
        print(json.dumps(manifests, indent=4, sort_keys=True))
        return 0
    differing = [
        f"{release} {path}"
        for release, manifest in manifests.items()
        for path, digest in manifest.items()
        if observer.SOURCE_MANIFESTS[release][path] != digest
    ]
    for item in differing:
        print(item)
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
