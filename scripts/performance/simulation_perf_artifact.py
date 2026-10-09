# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record and verify the exact wheel and locked requirements passed between CI jobs."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def record(directory: Path, side: str, revision: str, source: Path) -> None:
    actual = command("git", "-C", str(source), "rev-parse", "HEAD")
    if actual != revision:
        raise ValueError(f"build source is {actual}, expected {revision}")
    wheels = list(directory.glob("aisimulate-*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one AISimulate wheel")
    manifest = {
        "side": side,
        "source_sha": revision,
        "model_data_tree": command("git", "-C", str(source), "rev-parse", "HEAD:python/aisimulate/src/aisimulate_core"),
        "files": {p.name: sha256(p) for p in [wheels[0], directory / "requirements.txt"]},
        "build": {
            "container": os.environ["BUILD_CONTAINER"],
            "python": platform.python_version(),
            "rustc": command("rustc", "-vV"),
            "maturin": command("python", "-m", "maturin", "--version"),
            "uv": command("uv", "--version"),
            "flags": ["--release", "--locked"],
        },
        "build_seconds": int(os.environ["BUILD_SECONDS"]),
    }
    manifest["dynamo"] = json.loads((directory / "dynamo-build.json").read_text())
    for path in [*directory.glob("ai_dynamo*.whl"), *directory.glob("dynamo-*")]:
        manifest["files"][path.name] = sha256(path)
    (directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify(directory: Path, side: str, revision: str) -> dict:
    manifest = json.loads((directory / "provenance.json").read_text())
    if manifest["side"] != side or manifest["source_sha"] != revision:
        raise ValueError(f"{side}: artifact revision or role mismatch")
    wheels = list(directory.glob("aisimulate-*.whl"))
    runtime = list(directory.glob("ai_dynamo_runtime-*.whl"))
    adapter = list(directory.glob("ai_dynamo-*.whl"))
    if len(wheels) != 1 or len(runtime) != 1 or len(adapter) != 1:
        raise ValueError(f"{side}: expected AISimulate, Dynamo runtime and adapter wheels")
    expected_wheels = {wheels[0].name, runtime[0].name, adapter[0].name}
    if {path.name for path in directory.glob("*.whl")} != expected_wheels:
        raise ValueError(f"{side}: unexpected wheel in artifact")
    expected = {
        *expected_wheels,
        "requirements.txt",
        "dynamo-requirements.txt",
        "dynamo-build.json",
        "dynamo-build.patch",
        "dynamo-Cargo.lock",
    }
    if set(manifest["files"]) != expected or manifest["dynamo"]["aisimulate_sha"] != revision:
        raise ValueError(f"{side}: incomplete artifact or wrong embedded AISimulate revision")
    for name, checksum in manifest["files"].items():
        if sha256(directory / name) != checksum:
            raise ValueError(f"{side}: checksum mismatch for {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("record")
    build.add_argument("directory", type=Path)
    build.add_argument("side", choices=("base", "head"))
    build.add_argument("revision")
    build.add_argument("source", type=Path)
    check = subparsers.add_parser("verify")
    for side in ("base", "head"):
        check.add_argument(f"--{side}-dir", type=Path, required=True)
        check.add_argument(f"--{side}-sha", required=True)
    args = parser.parse_args()
    if args.action == "record":
        record(args.directory, args.side, args.revision, args.source)
    else:
        base = verify(args.base_dir, "base", args.base_sha)
        head = verify(args.head_dir, "head", args.head_sha)
        if base["build"] != head["build"]:
            raise ValueError("base and head build settings differ")
        if base["dynamo"]["source_sha"] != head["dynamo"]["source_sha"]:
            raise ValueError("base and head Dynamo revisions differ")
        if base["files"]["dynamo-requirements.txt"] != head["files"]["dynamo-requirements.txt"]:
            raise ValueError("base and head Dynamo requirements differ")
        if base["build"]["python"] != platform.python_version():
            raise ValueError("comparison Python differs from build Python")


if __name__ == "__main__":
    main()
