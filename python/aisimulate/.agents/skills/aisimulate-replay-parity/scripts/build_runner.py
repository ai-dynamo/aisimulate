#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the exact skill runner source against an isolated AI Simulate checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parent
RUNNER_SOURCE = SKILL_DIR / "replay-parity-runner"
DEPENDENCY_LINE = 'aisimulate-core = { path = "../../../../../../../crates/core" }'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runner_source_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (Path("Cargo.toml"), Path("Cargo.lock"), Path("src/main.rs")):
        payload = (RUNNER_SOURCE / relative).read_bytes()
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _prepare_empty_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path {path} is not a directory")
        if any(path.iterdir()):
            raise ValueError(f"output directory {path} is not empty")
    else:
        path.mkdir(parents=True)


def _checkout_revision(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _checkout_is_dirty(checkout: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


def _materialize_runner(checkout: Path, output: Path) -> Path:
    core_manifest = checkout / "crates/core/Cargo.toml"
    if not core_manifest.is_file():
        raise ValueError(f"checkout {checkout} has no crates/core/Cargo.toml")
    _prepare_empty_directory(output)
    (output / "src").mkdir()
    shutil.copy2(RUNNER_SOURCE / "src/main.rs", output / "src/main.rs")
    shutil.copy2(RUNNER_SOURCE / "Cargo.lock", output / "Cargo.lock")

    manifest = (RUNNER_SOURCE / "Cargo.toml").read_text(encoding="utf-8")
    if manifest.count(DEPENDENCY_LINE) != 1:
        raise ValueError("runner Cargo.toml has an unexpected aisimulate-core dependency")
    dependency = f"aisimulate-core = {{ path = {json.dumps(str(core_manifest.parent.resolve()))} }}"
    (output / "Cargo.toml").write_text(
        manifest.replace(DEPENDENCY_LINE, dependency), encoding="utf-8"
    )
    return output / "Cargo.toml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--toolchain", default="1.93.1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-dirty-checkout", action="store_true")
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    output = args.output_dir.resolve()
    revision = _checkout_revision(checkout)
    checkout_dirty = _checkout_is_dirty(checkout)
    if checkout_dirty and not args.allow_dirty_checkout:
        raise ValueError(
            f"checkout {checkout} has uncommitted or untracked files; "
            "use an immutable clean worktree or pass --allow-dirty-checkout explicitly"
        )
    manifest = _materialize_runner(checkout, output)
    result = {
        "checkout": str(checkout),
        "checkout_dirty": checkout_dirty,
        "checkout_revision": revision,
        "manifest": str(manifest),
        "runner_source_sha256": _runner_source_sha256(),
    }
    if not args.prepare_only:
        subprocess.run(
            [
                args.cargo,
                f"+{args.toolchain}",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                str(manifest),
            ],
            check=True,
        )
        binary = output / "target/release/aisimulate-replay-parity-runner"
        if not binary.is_file():
            raise RuntimeError(f"runner build did not produce {binary}")
        result["binary"] = str(binary)
        result["binary_sha256"] = _sha256(binary)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
