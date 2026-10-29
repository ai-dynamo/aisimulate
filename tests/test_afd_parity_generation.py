# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation is atomic and cannot bless changed source or native binaries."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location("afd_generation", Path(__file__).parents[1] / "scripts/afd_parity.py")
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)


@pytest.mark.parametrize("stale", ["source", "binary"])
def test_build_identity_rejects_stale_input(tmp_path, monkeypatch, stale):
    source = tmp_path / "source.rs"
    source.write_text("checked source")
    manifest = {
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "data_tree": "c" * 40,
        "binary_sha256": "wrong binary",
        "source_files": {"source.rs": parity._sha256(source)},
    }
    if stale == "source":
        source.write_text("changed after build")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    with pytest.raises(
        ValueError,
        match="source changed" if stale == "source" else "native binary differs",
    ):
        parity._verified_build(path)


@pytest.mark.parametrize("fail_after_run", [False, True])
def test_partial_refresh_is_atomic_and_keeps_other_case_identity(tmp_path, monkeypatch, fail_after_run):
    fixture = {
        "cases": {
            name: {
                "config": f"{name}.yaml",
                "summary": {"completed_requests": 1},
                "generation": {"source_commit": "original"},
            }
            for name in ["selected", "untouched"]
        }
    }
    path = tmp_path / "afd_e2e.json"
    path.write_text(json.dumps(fixture))
    before = path.read_bytes()
    (tmp_path / "selected.yaml").write_text("input")
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    monkeypatch.setattr(parity, "GOLDEN_DIR", tmp_path)
    calls = []

    def verify(_):
        calls.append(1)
        if len(calls) > 1 and fail_after_run:
            raise ValueError("source changed after build")
        return {"source_commit": "verified"}

    def run(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        (output / "prediction.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "completed_requests": 1,
                        "prediction_provenance": {"data_root": "/machine/local"},
                    }
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(parity, "_verified_build", verify)
    monkeypatch.setattr(parity.subprocess, "run", run)
    if fail_after_run:
        with pytest.raises(ValueError, match="source changed"):
            parity._refresh_e2e(["selected"], "explained change", build_manifest=tmp_path / "manifest")
        assert path.read_bytes() == before
    else:
        parity._refresh_e2e(["selected"], "explained change", build_manifest=tmp_path / "manifest")
        actual = json.loads(path.read_text())
        assert actual["cases"]["untouched"] == fixture["cases"]["untouched"]
        assert actual["cases"]["selected"]["generation"]["source_commit"] == "verified"
        assert "prediction_provenance" not in actual["cases"]["selected"]["summary"]
