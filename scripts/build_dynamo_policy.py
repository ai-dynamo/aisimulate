#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build matching base/optional-policy wheels from one checkout, without overrides."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

if __package__:
    from .build_release_artifacts import check_manifests
else:
    from build_release_artifacts import check_manifests

ROOT = Path(__file__).resolve().parents[1]
DYNAMO_REVISION = "d9eb42db1168131fdae318eef77255637e4d3495"


def check_policy_dependency(root: Path = ROOT) -> None:
    manifest = tomllib.loads((root / "crates/dynamo-policy/Cargo.toml").read_text())
    dependency = manifest["dependencies"]["dynamo-kv-router"]
    assert dependency["git"] == "https://github.com/ai-dynamo/dynamo"
    assert dependency["rev"] == DYNAMO_REVISION
    assert dependency["default-features"] is False and dependency["features"] == ["standalone-selection"]
    for path in (root / "Cargo.toml", root / "crates/dynamo-policy/Cargo.toml"):
        candidate = tomllib.loads(path.read_text())
        assert not candidate.get("patch") and not candidate.get("replace"), "Dynamo overrides are forbidden"
    assert "path" not in dependency and "branch" not in dependency and "tag" not in dependency, (
        "Dynamo must use the immutable Git revision without an override"
    )
    lock = tomllib.loads((root / "Cargo.lock").read_text())
    dynamo = [entry for entry in lock["package"] if entry["name"].startswith("dynamo-")]
    assert {entry["name"] for entry in dynamo} == {
        "dynamo-kv-router",
        "dynamo-tokens",
        "dynamo-kv-hashing",
        "dynamo-truthy",
    }, "adapter dependency graph must not import Dynamo runtime, LLM or Mocker"
    expected_source = f"git+https://github.com/ai-dynamo/dynamo?rev={DYNAMO_REVISION}#{DYNAMO_REVISION}"
    assert all(entry.get("source") == expected_source for entry in dynamo), "Dynamo lock source differs from the pin"


def source_identity(root: Path, supplied_revision: str | None) -> tuple[str, bool | None]:
    """Require provenance even when a Docker/source archive omits Git metadata."""
    if supplied_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", supplied_revision):
        raise ValueError("source revision must be a full 40-character Git SHA")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        if supplied_revision is None:
            raise ValueError("source archive/container builds require --source-revision") from None
        return supplied_revision, None  # Archive contents cannot prove Git cleanliness.
    if supplied_revision is not None and supplied_revision != revision:
        raise ValueError("supplied source revision differs from the checkout HEAD")
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    return revision, dirty


def verify_wheels(wheels: list[Path], version: str) -> None:
    expected = {"aisimulate", "aisimulate-dynamo-policy"}
    observed = set()
    for path in wheels:
        with zipfile.ZipFile(path) as wheel:
            metadata_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            assert len(metadata_paths) == 1, f"missing or duplicate wheel metadata: {path}"
            metadata = email.message_from_bytes(wheel.read(metadata_paths[0]))
            name = metadata["Name"]
            assert name in expected and name not in observed, f"unexpected or duplicate wheel: {name}"
            assert metadata["Version"] == version, f"wheel version differs from its manifest: {name}"
            if name == "aisimulate-dynamo-policy":
                assert metadata.get_all("Requires-Dist") == [f"aisimulate=={version}"], "wheel lost its exact base pin"
            observed.add(name)
            for legal in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
                matches = [entry for entry in wheel.namelist() if entry.endswith(f".dist-info/licenses/{legal}")]
                assert len(matches) == 1 and wheel.read(matches[0]) == (ROOT / legal).read_bytes()
    assert observed == expected, "expected one base wheel and one optional policy wheel"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--debug", action="store_true", help="Build unoptimized local qualification wheels")
    parser.add_argument(
        "--source-revision", help="Full checkout SHA (required for source archives and Docker contexts)"
    )
    args = parser.parse_args()
    version, _ = check_manifests()
    check_policy_dependency()
    try:
        revision, dirty = source_identity(ROOT, args.source_revision)
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be empty")
    # The immutable Dynamo source requires this toolchain. Both wheels use it.
    environment = {**os.environ, "RUSTUP_TOOLCHAIN": "1.96.1"}
    for package in ("aisimulate", "aisimulate-dynamo-policy"):
        command = [sys.executable, "-m", "maturin", "build", "--locked", "--out", str(output)]
        if not args.debug:
            command.append("--release")
        subprocess.run(command, cwd=ROOT / "python" / package, env=environment, check=True)
    wheels = sorted(output.glob("*.whl"))
    verify_wheels(wheels, version)
    receipt = {
        "version": version,
        "dynamo_revision": DYNAMO_REVISION,
        "rust_toolchain": "1.96.1",
        "profile": "debug" if args.debug else "release",
        "source_revision": revision,
        "source_dirty": dirty,
        "wheels": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in wheels},
    }
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
