# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage an exact Docker context from separately retained, checksum-pinned artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def copy_verified(source: Path, destination: Path, expected: str) -> None:
    with source.open("rb") as artifact:
        actual = hashlib.file_digest(artifact, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {source.name}: {actual} != {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Dereference only explicitly named metadata files; never copy an HF auth/cache tree.
    shutil.copyfile(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--hf-hub-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Output must be a new directory; existing contexts are never overwritten")
    wheels = json.loads((bundle / "image-provenance/wheelhouse-manifest.json").read_text())
    hf = json.loads((bundle / "image-provenance/hf-metadata.json").read_text())
    output.mkdir(parents=True)
    for name in ["Dockerfile", ".dockerignore", "requirements.lock"]:
        shutil.copyfile(bundle / name, output / name)
    for name in ["image-provenance", "smoke"]:
        shutil.copytree(bundle / name, output / name)
    for wheel in wheels:
        copy_verified(
            args.wheelhouse / wheel["filename"],
            output / "wheelhouse" / wheel["filename"],
            wheel["sha256"],
        )
    repository_dir = "models--" + hf["repository"].replace("/", "--")
    snapshot = Path(repository_dir) / "snapshots" / hf["revision"]
    for entry in hf["files"]:
        copy_verified(
            args.hf_hub_cache / snapshot / entry["name"],
            output / "hf-cache/hub" / snapshot / entry["name"],
            entry["sha256"],
        )
    refs = output / "hf-cache/hub" / repository_dir / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(hf["revision"])
    print(json.dumps({"context": str(output), "wheels": len(wheels), "hf_files": len(hf["files"]), "status": "verified"}))


if __name__ == "__main__":
    main()
