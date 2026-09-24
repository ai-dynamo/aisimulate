# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-wheel regression for the installed FPM workflow."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

# Release wheel builds on arm64 routinely approach the suite's 180-second
# per-test limit, so keep a larger guard for this build-and-install workflow.
pytestmark = [pytest.mark.e2e, pytest.mark.build, pytest.mark.timeout(600)]

REPO_ROOT = Path(__file__).resolve().parents[5]
APP_ROOT = REPO_ROOT / "python" / "aisimulate"
VERIFY_INSTALLED_LAYERS = APP_ROOT / "tools" / "verify_installed_package_layers.py"


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: int = 300):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_built_application_wheel_runs_installed_fpm_plan_and_resolves_runtime_assets(tmp_path):
    supplied_wheel = os.environ.get("AISIMULATE_TEST_WHEEL")
    if supplied_wheel:
        wheels = (Path(supplied_wheel).resolve(),)
    else:
        wheel_dir = tmp_path / "wheel"
        wheel_dir.mkdir()
        _run(
            [sys.executable, "-m", "maturin", "build", "--release", "--out", str(wheel_dir)],
            cwd=APP_ROOT,
            timeout=600,
        )
        wheels = tuple(wheel_dir.glob("aisimulate-*.whl"))
    assert len(wheels) == 1
    assert wheels[0].is_file()

    unrelated_repository = tmp_path / "unrelated-git"
    unrelated_repository.mkdir()
    _run(["git", "init"], cwd=unrelated_repository)
    (unrelated_repository / "host-source.txt").write_text("unrelated source\n", encoding="utf-8")
    _run(["git", "add", "host-source.txt"], cwd=unrelated_repository)
    _run(
        [
            "git",
            "-c",
            "user.name=AISimulate Build Test",
            "-c",
            "user.email=aisimulate-build-test@example.invalid",
            "commit",
            "-m",
            "unrelated host commit",
        ],
        cwd=unrelated_repository,
    )
    unrelated_head = _run(["git", "rev-parse", "HEAD"], cwd=unrelated_repository).stdout.strip()

    # Put the complete virtual environment under an unrelated Git checkout.
    # The installed planner must use wheel provenance before it can observe
    # this ambient repository and its deliberately different HEAD.
    install_root = unrelated_repository / "installed"
    _run([sys.executable, "-m", "venv", str(install_root)], cwd=unrelated_repository)
    installed_python = install_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        [str(installed_python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
        cwd=unrelated_repository,
    )

    # Reuse the already-installed test dependencies without exposing the app
    # source tree. Local editable environments may keep the core package in a
    # separate source directory, while CI's preceding core-wheel step already
    # places it in the current purelib directory.
    dependency_paths = [Path(sysconfig.get_path("purelib")).resolve()]
    core_spec = importlib.util.find_spec("aisimulate_core")
    assert core_spec is not None and core_spec.submodule_search_locations
    core_root = Path(next(iter(core_spec.submodule_search_locations))).resolve().parent
    if core_root not in dependency_paths:
        dependency_paths.append(core_root)
    installed_purelib = Path(
        _run(
            [str(installed_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            cwd=unrelated_repository,
        ).stdout.strip()
    )
    (installed_purelib / "aisimulate-build-test-dependencies.pth").write_text(
        "".join(f"{path}\n" for path in dependency_paths),
        encoding="utf-8",
    )

    env = {
        key: value for key, value in os.environ.items() if key not in {"FPM_COLLECTOR_SOURCE_REVISION", "PYTHONPATH"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    completed = _run(
        [str(installed_python), str(VERIFY_INSTALLED_LAYERS), "--expect", "fpm"],
        cwd=unrelated_repository,
        env=env,
    )
    installed_version = _run(
        [
            str(installed_python),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('aisimulate'))",
        ],
        cwd=unrelated_repository,
        env=env,
    ).stdout.strip()

    expected_version = tomllib.loads((APP_ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert installed_version == expected_version
    assert f"Verified installed AISimulate {installed_version} FPM workflow" in completed.stdout
    assert f"installed:aisimulate=={installed_version}:record-sha256:" in completed.stdout
    assert unrelated_head not in completed.stdout
    assert str(unrelated_repository) not in completed.stdout
