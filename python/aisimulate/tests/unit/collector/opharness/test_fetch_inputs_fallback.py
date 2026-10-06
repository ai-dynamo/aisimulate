# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated / vanished repos fall back to the SDK's bundled HF config with a
PROVENANCE note (every Blackwell box without a token staged meta-llama by hand
from exactly these files, 2026-09-30)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_inputs_t", COMPONENTS / "fetch_inputs.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["fetch_inputs_t"] = mod; spec.loader.exec_module(mod)
    return mod


def test_gated_repo_without_token_uses_bundled_config(tmp_path, monkeypatch):
    fi = _load()
    monkeypatch.setattr(fi, "_get", lambda url, token, binary=False: json.dumps({"gated": True, "siblings": [{"rfilename": "config.json"}]}))
    configs = tmp_path / "configs"; configs.mkdir()
    got = fi.fetch("meta-llama/Meta-Llama-3.1-8B", configs, token=None)
    assert got["config"] and "bundled" in got["source"]
    assert json.loads((configs / "meta-llama_Meta-Llama-3.1-8B.json").read_text())["architectures"]
    prov = json.loads((configs / "aux_files" / "meta-llama_Meta-Llama-3.1-8B" / "PROVENANCE.json").read_text())
    assert prov["reason"] == "gated, no token" and "model_configs" in prov["config_source"]


def test_hub_failure_without_bundled_copy_is_still_an_owner_decision(tmp_path, monkeypatch):
    fi = _load()
    def boom(url, token, binary=False):
        raise OSError("404")
    monkeypatch.setattr(fi, "_get", boom)
    with pytest.raises(SystemExit, match="OWNER DECISION"):
        fi.fetch("nobody/NoSuchModel", tmp_path, token=None)
