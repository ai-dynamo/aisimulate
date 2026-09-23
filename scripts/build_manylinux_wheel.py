#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Linux wheel and repair it for the manylinux 2.28 policy."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_PROJECT = ROOT / "python" / "aisimulate"
PLATFORM_BY_MACHINE = {
    "aarch64": "manylinux_2_28_aarch64",
    "arm64": "manylinux_2_28_aarch64",
    "amd64": "manylinux_2_28_x86_64",
    "x86_64": "manylinux_2_28_x86_64",
}


def manylinux_platform(machine: str | None = None) -> str:
    """Return the required manylinux policy for the native architecture."""
    machine = (machine or platform.machine()).lower()
    try:
        return PLATFORM_BY_MACHINE[machine]
    except KeyError as error:
        supported = ", ".join(sorted(PLATFORM_BY_MACHINE))
        raise SystemExit(f"unsupported wheel architecture {machine!r}; expected one of: {supported}") from error


def _run(*command: str, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _single_wheel(directory: Path, description: str) -> Path:
    wheels = sorted(directory.glob("aisimulate-*.whl"))
    if len(wheels) != 1:
        names = [path.name for path in wheels]
        raise SystemExit(f"expected exactly one {description}, found {names}")
    return wheels[0]


def build(output: Path) -> Path:
    """Build an unrepaired wheel, then make auditwheel produce the final wheel."""
    if not sys.platform.startswith("linux"):
        raise SystemExit("the manylinux wheel builder must run on Linux")
    if shutil.which("auditwheel") is None:
        raise SystemExit("auditwheel is required; run inside the pinned manylinux image")

    policy = manylinux_platform()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")

    with tempfile.TemporaryDirectory(prefix="aisimulate-unrepaired-wheel-") as temp:
        raw_output = Path(temp)
        _run(
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--locked",
            "--release",
            "--auditwheel",
            "skip",
            "--out",
            str(raw_output),
            cwd=PYTHON_PROJECT,
        )
        raw_wheel = _single_wheel(raw_output, "unrepaired AISimulate wheel")
        _run(
            "auditwheel",
            "repair",
            "--plat",
            policy,
            "--wheel-dir",
            str(output),
            str(raw_wheel),
        )

    repaired_wheel = _single_wheel(output, "repaired AISimulate wheel")
    if policy not in repaired_wheel.name:
        raise SystemExit(f"repaired wheel does not carry required tag {policy}: {repaired_wheel.name}")
    _run("auditwheel", "show", str(repaired_wheel))
    print(f"built {repaired_wheel}")
    return repaired_wheel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    build(args.output_dir.resolve())


if __name__ == "__main__":
    main()
