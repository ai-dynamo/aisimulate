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
    results = [
        subprocess.run(command, capture_output=True, text=True, check=True, env=environment).stdout
        for command in (
            [sys.executable, "-m", "aisim", "--help"],
            [sys.executable, "-m", "aisimulate", "--help"],
            [str(Path(sys.executable).parent / "aisim"), "--help"],
            [str(Path(sys.executable).parent / "aisimulate"), "--help"],
        )
    ]
    assert results[0]
    assert len(set(results)) == 1


@pytest.mark.parametrize("flag", ["--aisim-revision", "--aisimulate-revision"])
def test_onboarding_revision_flag_preserves_its_saved_field(flag):
    import argparse

    from aisim.support.cli import add_support_parser

    parser = argparse.ArgumentParser()
    add_support_parser(parser.add_subparsers(dest="command"))
    arguments = parser.parse_args(["onboard", "init", "--output-dir", "unused", flag, "a" * 40])
    assert arguments.aisimulate_revision == "a" * 40
