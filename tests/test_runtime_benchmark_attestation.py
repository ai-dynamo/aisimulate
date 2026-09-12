# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_migration_runtime", Path(__file__).resolve().parents[1] / "scripts/benchmark_migration_runtime.py"
)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


@pytest.mark.parametrize("tracked", [False, True])
def test_attestation_rejects_changed_source_before_import(tmp_path, tracked):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "src"
    source.mkdir()
    module = source / "shadow.py"
    module.write_text("value = 1\n")
    if tracked:
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        module.write_text("value = 2\n")
    variant = {
        "label": "fixture",
        "kind": "aisimulate",
        "root": str(tmp_path),
        "sources": [str(source)],
        "python": "/must-not-run-python",
    }
    with pytest.raises(RuntimeError, match="uncommitted source changes"):
        BENCHMARK.attest(variant, {})


def test_attestation_rejects_an_extension_from_another_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    other = tmp_path / "other/_runtime.abi3.so"
    other.parent.mkdir()
    other.write_bytes(b"different build")

    def output(command, **kwargs):
        del kwargs
        if "status" in command:
            return ""
        if "-c" in command:
            return json.dumps({"native": str(other), "python": "test", "dependencies": {}})
        raise AssertionError(command)

    monkeypatch.setattr(BENCHMARK.subprocess, "check_output", output)
    variant = {"label": "fixture", "kind": "aisimulate", "root": str(root), "sources": [str(root)], "python": "python"}
    with pytest.raises(ValueError):
        BENCHMARK.attest(variant, {})
