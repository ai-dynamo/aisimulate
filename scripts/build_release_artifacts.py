#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify the only two approved AISimulate release artifacts."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.12.0"
# Nightly CI stamps a dev suffix via scripts/apply_dev_version.py:
# PEP 440 `0.12.0.devYYYYMMDD` in the wheel, SemVer `0.12.0-dev.YYYYMMDD` in
# the crate, optionally followed by a ten-digit run number in both formats.
# Cargo rejects the PEP 440 spelling. The release contract still
# anchors on VERSION; only this suffix pair is additionally accepted.
DEV_SUFFIX_RE = re.compile(r"\.dev[0-9]{8}(?:[0-9]{10})?")

EXPECTED_PYTHON_PROJECTS = {
    ROOT / "python" / "aisimulate" / "pyproject.toml": "aisimulate",
}
EXPECTED_CRATE = ROOT / "crates" / "core" / "Cargo.toml"
LEGAL_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md")
IGNORED_DISCOVERY_DIRS = {".git", ".venv", "dist", "target", "release-tooling"}


def _toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text())


def _is_source_manifest(path: Path) -> bool:
    return not IGNORED_DISCOVERY_DIRS.intersection(path.relative_to(ROOT).parts)


def check_manifests() -> tuple[str, str]:
    """Validate the manifest set; return (wheel version, crate version)."""
    pyprojects = {
        path: str(_toml(path)["project"]["name"]) for path in ROOT.rglob("pyproject.toml") if _is_source_manifest(path)
    }
    assert pyprojects == EXPECTED_PYTHON_PROJECTS, (
        f"publishable Python manifest set changed:\nexpected={EXPECTED_PYTHON_PROJECTS}\nactual={pyprojects}"
    )

    publishable_crates: dict[Path, str] = {}
    for path in ROOT.rglob("Cargo.toml"):
        if not _is_source_manifest(path):
            continue
        manifest = _toml(path)
        package = manifest.get("package")
        if isinstance(package, dict) and package.get("publish", True) is not False:
            publishable_crates[path] = str(package["name"])
    assert publishable_crates == {EXPECTED_CRATE: "aisimulate-core"}, (
        "publishable Rust manifest set changed:\n"
        f"expected={{{EXPECTED_CRATE!r}: 'aisimulate-core'}}\nactual={publishable_crates}"
    )

    app = _toml(ROOT / "python" / "aisimulate" / "pyproject.toml")["project"]
    crate = _toml(EXPECTED_CRATE)["package"]
    py_version = str(app["version"])
    crate_version = str(crate["version"])
    dev_suffix = py_version.removeprefix(VERSION)
    assert py_version.startswith(VERSION) and (dev_suffix == "" or DEV_SUFFIX_RE.fullmatch(dev_suffix)), (
        f"wheel version must be {VERSION} or a supported numeric nightly version, got {py_version}"
    )
    expected_crate_version = f"{VERSION}-dev.{dev_suffix[len('.dev') :]}" if dev_suffix else VERSION
    assert crate_version == expected_crate_version, (
        f"crate version {crate_version} does not match wheel version {py_version} (expected {expected_crate_version})"
    )
    optional_dependencies = app.get("optional-dependencies", {})
    dependencies = [
        *app["dependencies"],
        *(dependency for group in optional_dependencies.values() for dependency in group),
    ]
    assert not any(str(dep).lower().startswith(("aisimulate-core", "aiconfigurator")) for dep in dependencies)
    assert not any(str(dep).lower().startswith(("dynamo", "ai-dynamo")) for dep in dependencies)
    assert app["scripts"] == {
        "aiconfigurator": "aiconfigurator.main:main",
        "aisimulate": "aisimulate.main:main",
    }
    return py_version, crate_version


def _run(
    *command: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd or ROOT, env=env, check=True)


def build(output: Path, py_version: str, crate_version: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")

    if sys.platform.startswith("linux"):
        _run(
            sys.executable,
            str(ROOT / "scripts" / "build_manylinux_wheel.py"),
            "--output-dir",
            str(output),
        )
    else:
        _run(
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--out",
            str(output),
            cwd=ROOT / "python" / "aisimulate",
        )

    with tempfile.TemporaryDirectory(prefix="aisimulate-crate-") as temp:
        target = Path(temp) / "target"
        _run(
            "cargo",
            "package",
            "--manifest-path",
            str(EXPECTED_CRATE),
            "--allow-dirty",
            "--target-dir",
            str(target),
            env={**os.environ, "PYO3_PYTHON": sys.executable},
        )
        crate = target / "package" / f"aisimulate-core-{crate_version}.crate"
        if not crate.is_file():
            raise SystemExit(f"cargo did not produce {crate}")
        shutil.copy2(crate, output / crate.name)

    verify_output(output, py_version, crate_version)


def verify_output(output: Path, py_version: str, crate_version: str) -> None:
    names = sorted(path.name for path in output.iterdir() if path.is_file())
    expected_crate = f"aisimulate-core-{crate_version}.crate"
    app_wheels = [name for name in names if name.startswith(f"aisimulate-{py_version}-") and name.endswith(".whl")]
    assert len(names) == 2, f"expected exactly two artifacts, got {names}"
    assert len(app_wheels) == 1, f"missing or duplicate aisimulate wheel: {names}"
    assert expected_crate in names, f"missing {expected_crate}: {names}"
    wheel_path = output / app_wheels[0]
    with zipfile.ZipFile(wheel_path) as wheel:
        for legal_file in LEGAL_FILES:
            matches = [name for name in wheel.namelist() if name.endswith(f".dist-info/licenses/{legal_file}")]
            assert len(matches) == 1, f"expected one packaged {legal_file}, found {matches}"
            assert wheel.read(matches[0]) == (ROOT / legal_file).read_bytes(), (
                f"packaged {legal_file} differs from the root original"
            )
    print("verified release artifacts:")
    for name in names:
        print(f"- {name}")


def main() -> None:
    global ROOT, VERSION, EXPECTED_PYTHON_PROJECTS, EXPECTED_CRATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="Release source checkout")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.root is not None:
        ROOT = args.root.resolve()
        # A historical release has its own base version. Anchor the contract
        # to its committed manifest, never to the stamped working copy.
        manifest = subprocess.check_output(
            ["git", "show", "HEAD:python/aisimulate/pyproject.toml"], cwd=ROOT, text=True
        )
        VERSION = str(tomllib.loads(manifest)["project"]["version"])
    EXPECTED_PYTHON_PROJECTS = {ROOT / "python" / "aisimulate" / "pyproject.toml": "aisimulate"}
    EXPECTED_CRATE = ROOT / "crates" / "core" / "Cargo.toml"

    py_version, crate_version = check_manifests()
    if not args.check_only:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / "dist"
        build(output_dir.resolve(), py_version, crate_version)


if __name__ == "__main__":
    main()
