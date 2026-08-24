# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_aic_sync_patch.py"
SPEC = importlib.util.spec_from_file_location("render_aic_sync_patch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def test_sync_source_validation_rejects_missing_mapped_path(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "AISimulate test")
    _git(tmp_path, "config", "user.email", "aisimulate-test@nvidia.com")
    source = tmp_path / "aic-core" / "src" / "aiconfigurator_core"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    _git(tmp_path, "add", "aic-core/src/aiconfigurator_core/__init__.py")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    commit = _git(tmp_path, "rev-parse", "HEAD")

    SYNC._require_source_path(tmp_path, commit, "aic-core/src/aiconfigurator_core")
    with pytest.raises(ValueError, match="does not exist"):
        SYNC._require_source_path(tmp_path, commit, "src/aiconfigurator_core")
