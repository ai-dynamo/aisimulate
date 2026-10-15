# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from collector.fpm_forward.runtime_instrumentation import freeze_instrumentation, load_instrumentation

pytestmark = pytest.mark.unit


def _manifest(tmp_path: Path):
    (tmp_path / "observer.py").write_text("raise AssertionError('observer must never execute on the host')\n")
    (tmp_path / "source.md").write_text("Source mapping with bytes and lifecycle points.\n")
    value = {
        "schema_version": "aisimulate-runtime-instrumentation/v1",
        "runtime": {"framework": "vllm", "version": "0.28.0", "source_revision": "a" * 40},
        "files": ["observer.py", "source.md"],
        "source_notes": "source.md",
        "worker_class": "observer.ObservedWorker",
        "scheduler_class": "observer.ObservedScheduler",
        "observation_schema": "aisimulate-runtime-observation/v1",
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(json.dumps(value))
    return path, value


def test_load_and_freeze_preserve_hashes_without_executing_observer(tmp_path):
    path, value = _manifest(tmp_path)
    bundle = load_instrumentation(path, expected_version="0.28.0")
    frozen = freeze_instrumentation(bundle, tmp_path / "frozen")
    assert frozen.sha256 == bundle.sha256
    assert frozen.manifest == value
    assert frozen.manifest_path == tmp_path / "frozen" / "manifest.json"
    assert frozen.files.keys() == {"observer.py", "source.md"}
    (tmp_path / "observer.py").write_text("raise RuntimeError('changed')\n")
    assert load_instrumentation(path).sha256 != frozen.sha256
    assert load_instrumentation(frozen.manifest_path).sha256 == frozen.sha256
    with pytest.raises(ValueError, match="changed"):
        freeze_instrumentation(bundle, tmp_path / "another")
    with pytest.raises(ValueError, match="fresh"):
        freeze_instrumentation(frozen, tmp_path / "frozen")


@pytest.mark.parametrize(
    "name", ["../observer.py", "/observer.py", "./observer.py", "x//observer.py", "x\\observer.py"]
)
def test_declared_paths_are_unambiguous_and_contained(tmp_path, name):
    path, value = _manifest(tmp_path)
    value["files"].append(name)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="relative|normalized"):
        load_instrumentation(path)


def test_symlinks_and_duplicate_modules_are_rejected(tmp_path):
    path, value = _manifest(tmp_path)
    (tmp_path / "alias.py").symlink_to("observer.py")
    value["files"].append("alias.py")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="symlink"):
        load_instrumentation(path)
    value["files"].remove("alias.py")
    (tmp_path / "observer").mkdir()
    (tmp_path / "observer" / "__init__.py").write_text("")
    value["files"].append("observer/__init__.py")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ambiguous"):
        load_instrumentation(path)


def test_version_unknown_keys_duplicate_keys_and_undeclared_modules_rejected(tmp_path):
    path, value = _manifest(tmp_path)
    with pytest.raises(ValueError, match="version"):
        load_instrumentation(path, expected_version="0.27.0")
    value["worker_class"] = "not_declared.Worker"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="declared"):
        load_instrumentation(path)
    path.write_text('{"schema_version":"x", "schema_version":"y"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_instrumentation(path)
    path.write_text("schema_version: x\nschema_version: y\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_instrumentation(path)
