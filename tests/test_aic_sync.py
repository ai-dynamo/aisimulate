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
        SYNC._require_source_path(tmp_path, commit, "src/aisimulate_core")


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


def test_render_prefixes_rename_metadata_with_the_mirror_target(tmp_path: Path, monkeypatch) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "AISimulate test")
    _git(tmp_path, "config", "user.email", "aisimulate-test@nvidia.com")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "old.txt").write_text("stable\n")
    _git(tmp_path, "add", "mirror/old.txt")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    _git(tmp_path, "mv", "mirror/old.txt", "mirror/new.txt")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "rename")
    target = _git(tmp_path, "rev-parse", "HEAD")

    ledger = tmp_path / "ledger.toml"
    ledger.write_text(
        """
[upstream]
last_synced = "unused"

[[mirror]]
source = "mirror"
target = "mapped"
""".strip()
        + "\n"
    )
    monkeypatch.setattr(SYNC, "LEDGER", ledger)

    patch = SYNC.render(tmp_path, base, target)
    assert b"rename from mapped/old.txt" in patch
    assert b"rename to mapped/new.txt" in patch

    destination = tmp_path / "destination"
    destination.mkdir()
    _git(destination, "init")
    (destination / "mapped").mkdir()
    (destination / "mapped" / "old.txt").write_text("stable\n")
    patch_path = tmp_path / "rename.patch"
    patch_path.write_bytes(patch)
    subprocess.run(
        ("git", "-C", str(destination), "apply", "--check", str(patch_path)),
        check=True,
    )


def test_repository_ledger_maps_original_packages_without_restoring_old_names(tmp_path: Path) -> None:
    import tomllib

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "AISimulate test")
    _git(tmp_path, "config", "user.email", "aisimulate-test@nvidia.com")
    ledger = tomllib.loads(SYNC.LEDGER.read_text())
    for mapping in ledger["mirror"]:
        root = tmp_path / mapping["source"]
        root.mkdir(parents=True, exist_ok=True)
        (root / "fixture.txt").write_text("base\n")
    paths = {
        "aic-core/src/aiconfigurator_core/sdk/engine.py": "python/aisimulate/src/aisimulate_core/sdk/engine.py",
        "src/aiconfigurator/sdk/task_v2.py": "python/aisimulate/src/aisimulate/sdk/task_v2.py",
        "src/aiconfigurator/generator/api.py": "python/aisimulate/src/aisimulate/generator/api.py",
        "src/aiconfigurator/cli/main.py": "python/aisimulate/src/aisimulate/legacy_cli/main.py",
    }
    for source in paths:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("before\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    for source in paths:
        (tmp_path / source).write_text("after\n")
    (tmp_path / "src/aiconfigurator/main.py").write_text("entrypoint\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-m", "target")
    target = _git(tmp_path, "rev-parse", "HEAD")

    with pytest.raises(SYNC.ManualChangesRequired, match="src/aiconfigurator"):
        SYNC.render(tmp_path, base, target)
    report = tmp_path / "manual.md"
    patch = SYNC.render(tmp_path, base, target, manual_report=report)
    for destination in paths.values():
        assert f"+++ b/{destination}".encode() in patch
    assert b"python/aisimulate/src/aiconfigurator" not in patch
    assert "src/aiconfigurator/main.py" in report.read_text()
