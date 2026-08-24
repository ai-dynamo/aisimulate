# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version and wire-schema contracts shared by the unified wheel and crate."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import aiconfigurator_core
from aiconfigurator_core.sdk import engine

APPLICATION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUST_CRATE = REPOSITORY_ROOT / "crates" / "core"
RUST_CONFIG = RUST_CRATE / "src" / "perfmodel" / "config.rs"
SUPPORTED_PYTHON = ">=3.11,<3.14"
SUPPORTED_NUMPY = "numpy>=2.1,<3"
SUPPORTED_PLOTEXT = "plotext>=5.3.2,<6"


def _project_version(path: Path) -> str:
    return str(tomllib.loads(path.read_text())["project"]["version"])


def _crate_version(path: Path) -> str:
    return str(tomllib.loads(path.read_text())["package"]["version"])


def _rust_u32_constant(path: Path, name: str) -> int:
    source = path.read_text()
    match = re.search(rf"^pub const {re.escape(name)}: u32 = (\d+);$", source, re.MULTILINE)
    assert match is not None, f"missing public Rust constant {name}"
    return int(match.group(1))


def test_aisimulate_wheel_and_core_crate_versions_match() -> None:
    assert _project_version(APPLICATION_ROOT / "pyproject.toml") == _crate_version(RUST_CRATE / "Cargo.toml")


def test_python_and_numpy_support_contracts_match() -> None:
    project = tomllib.loads((APPLICATION_ROOT / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == SUPPORTED_PYTHON
    assert [dependency for dependency in project["dependencies"] if dependency.startswith("numpy")] == [
        SUPPORTED_NUMPY
    ]


def test_plotext_support_contract_excludes_incompatible_v6() -> None:
    project = tomllib.loads((APPLICATION_ROOT / "pyproject.toml").read_text())["project"]
    assert [dependency for dependency in project["dependencies"] if dependency.startswith("plotext")] == [
        SUPPORTED_PLOTEXT
    ]


def test_engine_schema_versions_match_across_python_and_rust() -> None:
    assert _rust_u32_constant(RUST_CONFIG, "ENGINE_CONFIG_SCHEMA_VERSION") == engine.ENGINE_CONFIG_SCHEMA_VERSION
    assert _rust_u32_constant(RUST_CONFIG, "ENGINE_SPEC_SCHEMA_VERSION") == engine.ENGINE_SPEC_SCHEMA_VERSION
    assert aiconfigurator_core._build_smoke() == engine.ENGINE_CONFIG_SCHEMA_VERSION
