# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One pin for every FPM consumer of the unified dataset (AISIM-2338)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("set_fpm_dataset_pin", ROOT / "scripts/ci/set_fpm_dataset_pin.py")
pin_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pin_tool)


def test_every_unified_consumer_carries_the_single_pin():
    pin = json.loads(pin_tool.PIN_FILE.read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"])
    consumers = pin_tool.consumers()
    assert len(consumers) >= 15
    for path in consumers:
        document = json.loads(path.read_text())
        assert pin_tool.is_unified(path), path
        assert (document["repo_id"], document["revision"]) == (pin["repo_id"], pin["revision"]), path


def test_only_the_history_profiles_stay_on_the_retired_layout():
    """dsv41_fpm_hf.json pins sidecars that were rewritten after b35883ee: it keeps its old repo and commit."""
    document = json.loads((pin_tool.SYSTEMS / "dsv41_fpm_hf.json").read_text())
    assert document["repo_id"] == "nvidia/aisimulate-fpm-dataset"
    assert document["revision"] == "b35883ee5f8b4a82e24844a7e872056aff64287f"
    assert all(item["path"].startswith("data/history/") for e in document["profiles"].values() for item in e["files"])


@pytest.mark.parametrize("revision", ["c" * 40])
def test_set_pin_rewrites_every_consumer_and_check_agrees(tmp_path, monkeypatch, revision):
    """Run against a copy so the checked-in pins are untouched."""
    copy = tmp_path / "repo"
    for source in [pin_tool.PIN_FILE, *pin_tool.consumers()]:
        target = copy / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    monkeypatch.setattr(pin_tool, "ROOT", copy)
    monkeypatch.setattr(pin_tool, "PIN_FILE", copy / "scripts/fpm_accuracy/dataset_pin.json")
    monkeypatch.setattr(pin_tool, "SYSTEMS", copy / "python/aisim/src/aisim_core/systems")
    monkeypatch.setattr(pin_tool, "RECIPES", copy / "python/aisim/collector/fpm_forward/recipes")
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--revision", revision])
    assert pin_tool.main() == 0
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--check", "--require-set"])
    assert pin_tool.main() == 0
    assert {json.loads(p.read_text())["revision"] for p in pin_tool.consumers()} == {revision}
