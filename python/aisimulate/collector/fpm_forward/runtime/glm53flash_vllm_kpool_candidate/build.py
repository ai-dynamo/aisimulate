# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build an actual versioned wheel with upstream's precompiled build route.

This original build driver never changes the installed stock package. A produced
wheel is an unqualified candidate until separate native correctness tests pass.
"""

import argparse
import email.parser
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def member_digest(archive, member):
    with archive.open(member) as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--stock-wheel", type=Path, required=True)
    parser.add_argument("--stock-wheel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if platform.machine() != "aarch64":
        raise RuntimeError("build must use the pinned ARM64 native image")
    stock = importlib.metadata.distribution("vllm")
    if stock.version != "0.30.0":
        raise RuntimeError("installed base package is not the exact stock release")
    identity = json.loads((root / "patch-identity.json").read_text())
    archive = json.loads((root / "source-archive.json").read_text())
    if digest(args.source_archive) != archive["sha256"]:
        raise RuntimeError("source archive hash mismatch")
    origin = json.loads((root / "binary-base-origin.json").read_text())
    if origin["archive_sha256"] != args.stock_wheel_sha256:
        raise RuntimeError("binary base differs from frozen image-derived origin receipt")
    if origin["upstream_source_revision"] != "ced6857afa0ea7b2e3f0846a62e1394e90f15607":
        raise RuntimeError("binary base origin source revision differs")
    if digest(args.stock_wheel) != args.stock_wheel_sha256:
        raise RuntimeError("provided binary base wheel hash mismatch")
    if digest(stock.locate_file(identity["source_path"])) != identity["base_sha256"]:
        raise RuntimeError("installed original helper differs from immutable base")
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {}
    legal_files = {}
    with zipfile.ZipFile(args.stock_wheel) as wheel:
        (metadata_path,) = [x for x in wheel.namelist() if x.endswith(".dist-info/METADATA")]
        metadata = email.parser.Parser().parsestr(wheel.read(metadata_path).decode())
        if metadata["Name"] != "vllm" or metadata["Version"] != "0.30.0":
            raise RuntimeError("binary wheel metadata is not the stock base release")
        for name in wheel.namelist():
            if any(
                marker in Path(name).name.upper() for marker in ("LICENSE", "COPYING", "NOTICE")
            ) and not name.endswith("/"):
                legal_files[name] = member_digest(wheel, name)
        for item in stock.files:
            name = str(item)
            if not name.startswith("vllm/") or not (name.endswith(".so") or name == "vllm/vllm-rs"):
                continue
            installed_sha = digest(stock.locate_file(item))
            if name not in wheel.namelist() or member_digest(wheel, name) != installed_sha:
                raise RuntimeError(f"binary base differs from actual stock install: {name}")
            binaries[name] = installed_sha
    if not binaries:
        raise RuntimeError("no actual native binary identity was verified")
    source_dir = args.output / "source"
    source_dir.mkdir()
    with tarfile.open(args.source_archive) as source:
        source.extractall(source_dir, filter="data")
    (source_root,) = list(source_dir.iterdir())
    original = source_root / identity["source_path"]
    if digest(original) != identity["base_sha256"]:
        raise RuntimeError("source helper differs from immutable base")
    patch = root / "retained-tail-prefill.patch"
    patch_digest = digest(patch)
    if patch_digest != identity["patch_sha256"]:
        raise RuntimeError("repair patch differs from frozen candidate")
    with patch.open("rb") as patch_input:
        subprocess.run(["patch", "--batch", "--forward", "-p1"], cwd=source_root, stdin=patch_input, check=True)
    if digest(original) != identity["patched_sha256"]:
        raise RuntimeError("applied source repair differs from frozen candidate bytes")
    version = "0.30.0+glm53kpool." + patch_digest[:12]
    # Native setuptools build writes both distribution METADATA and _version.py.
    # No package-version monkeypatch or edited installed distribution is used.
    env = dict(
        os.environ,
        VLLM_USE_PRECOMPILED="1",
        VLLM_PRECOMPILED_WHEEL_LOCATION=str(args.stock_wheel.resolve()),
        VLLM_VERSION_OVERRIDE=version,
    )
    destination = args.output / "dist"
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        ".",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(destination.resolve()),
    ]
    (args.output / "build-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (args.output / "build.log").open("w") as log:
        subprocess.run(
            command,
            cwd=source_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    (built,) = list(destination.glob("*.whl"))
    with zipfile.ZipFile(built) as wheel:
        (metadata_path,) = [x for x in wheel.namelist() if x.endswith(".dist-info/METADATA")]
        metadata = email.parser.Parser().parsestr(wheel.read(metadata_path).decode())
        if metadata["Name"] != "vllm" or metadata["Version"] != version:
            raise RuntimeError("build did not produce the requested real version metadata")
        version_namespace = {}
        exec(
            compile(wheel.read("vllm/_version.py"), "built-vllm-version", "exec"),
            version_namespace,
        )
        if version_namespace.get("__version__") != version:
            raise RuntimeError("actual packaged version differs from distribution metadata")
        if member_digest(wheel, identity["source_path"]) != identity["patched_sha256"]:
            raise RuntimeError("build lost the source repair")
        for name, expected in binaries.items():
            if name not in wheel.namelist() or member_digest(wheel, name) != expected:
                raise RuntimeError(f"native binary changed or disappeared during build: {name}")
        built_legal = {
            member_digest(wheel, name)
            for name in wheel.namelist()
            if any(marker in Path(name).name.upper() for marker in ("LICENSE", "COPYING", "NOTICE"))
            and not name.endswith("/")
        }
        if not legal_files or not set(legal_files.values()) <= built_legal:
            raise RuntimeError("upstream binary-base license/notice material missing from built distribution")
    receipt = {
        "status": "built_unqualified_candidate",
        "version": version,
        "wheel": built.name,
        "wheel_sha256": digest(built),
        "source_archive": archive,
        "patch": identity,
        "patch_sha256": patch_digest,
        "binary_base_origin": origin,
        "binary_base_origin_receipt_sha256": digest(root / "binary-base-origin.json"),
        "binary_base_wheel": args.stock_wheel.name,
        "binary_base_sha256": args.stock_wheel_sha256,
        "unchanged_native_binaries": binaries,
        "preserved_binary_base_legal_files": legal_files,
        "correctness_acceptance": "NOT_EVALUATED",
        "accuracy_acceptance": "NOT_EVALUATED",
    }
    (args.output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
