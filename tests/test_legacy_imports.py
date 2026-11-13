# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The renamed application must share one runtime through old and new APIs."""

import importlib
import importlib.metadata
import importlib.resources
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "legacy,canonical",
    [
        ("aisimulate", "aisim"),
        ("aisimulate.traffic", "aisim.traffic"),
        ("aisimulate.config", "aisim.config"),
        ("aisimulate.runner", "aisim.runner"),
        ("aisimulate._runtime", "aisim._runtime"),
        ("aisimulate.sdk.memory", "aisim.sdk.memory"),
        ("aisimulate_core", "aisim_core"),
        ("aisimulate_core.sdk.common", "aisim_core.sdk.common"),
        ("aisimulate_core._native", "aisim_core._native"),
    ],
)
def test_old_and_new_imports_share_module_objects(legacy, canonical):
    old = importlib.import_module(legacy)
    new = importlib.import_module(canonical)
    assert old is new
    assert old.__spec__.name == new.__spec__.name


def test_aliases_share_mutable_state_and_bundled_resources(monkeypatch):
    old = importlib.import_module("aisimulate_core.sdk.common")
    new = importlib.import_module("aisim_core.sdk.common")
    marker = object()
    monkeypatch.setattr(old, "_rename_test_marker", marker, raising=False)
    assert new._rename_test_marker is marker
    assert importlib.resources.files("aisimulate_core") == importlib.resources.files("aisim_core")
    assert (importlib.resources.files("aisimulate_core") / "systems/h100_sxm.yaml").is_file()
    # GLOBAL opcodes from historical pickles still resolve through the alias.
    assert pickle.loads(b"caisimulate_core.sdk.common\nGEMMQuantMode\n.") is new.GEMMQuantMode


def test_legacy_plugin_namespace_remains_discoverable(monkeypatch):
    from aisim._legacy import plugin_entry_points

    entries = importlib.metadata.EntryPoints(
        [
            importlib.metadata.EntryPoint(name="old", value="example:old", group="aisimulate.runner_factories"),
            importlib.metadata.EntryPoint(name="new", value="example:new", group="aisim.runner_factories"),
            importlib.metadata.EntryPoint(name="both", value="example:both", group="aisim.runner_factories"),
            importlib.metadata.EntryPoint(name="both", value="example:both", group="aisimulate.runner_factories"),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)
    assert [entry.name for entry in plugin_entry_points("aisim.runner_factories")] == ["new", "both", "old"]


def test_module_execution_and_console_commands_share_cli():
    environment = {**os.environ, "NO_COLOR": "1"}
    results = []
    for program, command in (
        ("aisim", [sys.executable, "-m", "aisim", "--help"]),
        ("aisimulate", [sys.executable, "-m", "aisimulate", "--help"]),
        ("aisim", [str(Path(sys.executable).parent / "aisim"), "--help"]),
        ("aisimulate", [str(Path(sys.executable).parent / "aisimulate"), "--help"]),
    ):
        output = subprocess.run(command, capture_output=True, text=True, check=True, env=environment, timeout=30).stdout
        prefix = f"usage: {program} "
        assert output.startswith(prefix), output
        # Launcher identity intentionally differs; all remaining help must match.
        results.append("usage: <command> " + output.removeprefix(prefix))
    assert len(set(results)) == 1


@pytest.mark.parametrize("flag", ["--aisim-revision", "--aisimulate-revision"])
def test_onboarding_revision_flag_preserves_its_saved_field(flag):
    import argparse

    from aisim.support.cli import add_support_parser

    parser = argparse.ArgumentParser()
    add_support_parser(parser.add_subparsers(dest="command"))
    arguments = parser.parse_args(["onboard", "init", "--output-dir", "unused", flag, "a" * 40])
    assert arguments.aisimulate_revision == "a" * 40


@pytest.mark.parametrize("module", ["aisim", "aisim_core", "aisimulate", "aisimulate_core"])
@pytest.mark.parametrize("order", [("aisimulate", "aisim"), ("aisim", "aisimulate")])
def test_overlapping_distributions_fail_before_loading_runtime(tmp_path, module, order):
    for name in order:
        metadata = tmp_path / f"{name}-0.14.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.14.0\n")
    source = Path(__file__).resolve().parents[1] / "python/aisim/src"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            f"""
import importlib
import sys
try:
    importlib.import_module({module!r})
except ImportError as error:
    assert 'aisim._runtime' not in sys.modules
    assert 'aisim_core._native' not in sys.modules
    print(error)
    sys.exit(23)
raise AssertionError('mixed distributions must be rejected')
""",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(source), str(tmp_path)))},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    assert "aisim and aisimulate are both installed" in result.stdout
    assert "pip uninstall -y aisimulate" in result.stdout
    assert "pip install --force-reinstall" in result.stdout


@pytest.mark.parametrize("installed", [(), ("aisim",), ("aisimulate",)])
def test_distribution_guard_allows_single_distribution_and_source_checkout(tmp_path, installed):
    for name in installed:
        metadata = tmp_path / f"{name}-0.14.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.14.0\n")
    source = Path(__file__).resolve().parents[1] / "python/aisim/src"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys, aisim, aisim_core, aisimulate, aisimulate_core; "
            "assert aisimulate is aisim; assert aisimulate_core is aisim_core; "
            "assert 'aisim._runtime' not in sys.modules; "
            f"assert aisim.__version__ == {('0.14.0' if 'aisim' in installed else '0+unknown')!r}",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(source), str(tmp_path)))},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
