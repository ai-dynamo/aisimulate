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


def test_manual_changes_require_and_populate_a_report(tmp_path: Path, monkeypatch) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "AISimulate test")
    _git(tmp_path, "config", "user.email", "aisimulate-test@nvidia.com")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "value.txt").write_text("before\n")
    (tmp_path / "pyproject.toml").write_text("before\n")
    _git(tmp_path, "add", "mirror/value.txt", "pyproject.toml")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    (mirror / "value.txt").write_text("after\n")
    (tmp_path / "pyproject.toml").write_text("after\n")
    _git(tmp_path, "add", "mirror/value.txt", "pyproject.toml")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "target")
    target = _git(tmp_path, "rev-parse", "HEAD")

    ledger = tmp_path / "ledger.toml"
    ledger.write_text(
        """
[upstream]
last_synced = "unused"

[[mirror]]
source = "mirror"
target = "mapped"

[[manual]]
source = "pyproject.toml"
reason = "adapt the combined wheel manifest"
""".strip()
        + "\n"
    )
    monkeypatch.setattr(SYNC, "LEDGER", ledger)

    with pytest.raises(SYNC.ManualChangesRequired, match="pyproject.toml"):
        SYNC.render(tmp_path, base, target)

    report = tmp_path / "manual.md"
    patch = SYNC.render(tmp_path, base, target, manual_report=report)
    assert b"a/mapped/value.txt" in patch
    report_text = report.read_text()
    assert "`pyproject.toml`" in report_text
    assert "`M\tpyproject.toml`" in report_text
    assert "adapt the combined wheel manifest" in report_text
