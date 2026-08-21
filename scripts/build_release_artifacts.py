#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify the only two approved AISimulate release artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.13.0"

EXPECTED_PYTHON_PROJECTS = {
    ROOT / "python" / "aisimulate" / "pyproject.toml": "aisimulate",
}
EXPECTED_CRATE = ROOT / "crates" / "core" / "Cargo.toml"


def _toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text())


def check_manifests() -> None:
    pyprojects = {
        path: str(_toml(path)["project"]["name"])
        for path in ROOT.rglob("pyproject.toml")
        if ".venv" not in path.parts and "target" not in path.parts
    }
    assert pyprojects == EXPECTED_PYTHON_PROJECTS, (
        "publishable Python manifest set changed:\n"
        f"expected={EXPECTED_PYTHON_PROJECTS}\nactual={pyprojects}"
    )

    publishable_crates: dict[Path, str] = {}
    for path in ROOT.rglob("Cargo.toml"):
        if ".venv" in path.parts or "target" in path.parts:
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
    assert app["version"] == crate["version"] == VERSION
    optional_dependencies = app.get("optional-dependencies", {})
    dependencies = [
        *app["dependencies"],
        *(
            dependency
            for group in optional_dependencies.values()
            for dependency in group
        ),
    ]
    assert not any(
        str(dep).lower().startswith(("aisimulate-core", "aiconfigurator"))
        for dep in dependencies
    )
    assert not any(
        str(dep).lower().startswith(("dynamo", "ai-dynamo"))
        for dep in dependencies
    )
    assert app["scripts"] == {
        "aiconfigurator": "aiconfigurator.main:main",
        "aisimulate": "aisimulate.main:main",
    }


def _run(
    *command: str,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")

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
        crate = target / "package" / f"aisimulate-core-{VERSION}.crate"
        if not crate.is_file():
            raise SystemExit(f"cargo did not produce {crate}")
        shutil.copy2(crate, output / crate.name)

    verify_output(output)


def verify_output(output: Path) -> None:
    names = sorted(path.name for path in output.iterdir() if path.is_file())
    expected_crate = f"aisimulate-core-{VERSION}.crate"
    app_wheels = [
        name
        for name in names
        if name.startswith(f"aisimulate-{VERSION}-") and name.endswith(".whl")
    ]
    assert len(names) == 2, f"expected exactly two artifacts, got {names}"
    assert len(app_wheels) == 1, f"missing or duplicate aisimulate wheel: {names}"
    assert expected_crate in names, f"missing {expected_crate}: {names}"
    print("verified release artifacts:")
    for name in names:
        print(f"- {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    check_manifests()
    if not args.check_only:
        build(args.output_dir.resolve())


if __name__ == "__main__":
    main()
