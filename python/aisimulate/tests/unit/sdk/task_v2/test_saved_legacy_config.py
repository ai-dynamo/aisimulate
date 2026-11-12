# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading experiment artifacts saved before legacy AFD removal."""

import copy
import warnings
from pathlib import Path

import pytest
import yaml

from aisimulate.legacy_cli.main import build_experiment_tasks
from aisimulate.sdk import task_v2
from aisimulate.sdk.task_v2 import Task

pytestmark = pytest.mark.unit
_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"


@pytest.fixture(autouse=True)
def _reset_afd_warning(monkeypatch):
    monkeypatch.setattr(task_v2, "_warned_removed_afd_keys", False)


def _saved_config(mode):
    return yaml.safe_load((_FIXTURES / f"pr320-pre-removal-{mode}.yaml").read_text())


@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_saved_experiment_loads_without_changing_active_configuration(mode):
    saved = _saved_config(mode)
    original = copy.deepcopy(saved)
    removed = {key for key in saved if key.startswith("afd_")}
    assert len(removed) == 26
    expected = Task.from_yaml({key: value for key, value in saved.items() if key not in removed})

    with pytest.warns(DeprecationWarning, match="Removed legacy AFD Task keys"):
        restored = Task.from_yaml(saved)

    assert saved == original
    assert restored.to_dict() == expected.to_dict()
    exported = yaml.safe_load(restored.to_yaml())
    assert not removed.intersection(exported)
    assert Task.from_yaml(exported).to_dict() == restored.to_dict()


@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_saved_experiment_builds_through_legacy_cli(mode, tmp_path):
    saved = _saved_config(mode)
    path = tmp_path / "rerun.yaml"
    path.write_text(yaml.safe_dump({"exps": ["rerun"], "rerun": saved}))

    with pytest.warns(DeprecationWarning, match="Removed legacy AFD Task keys"):
        tasks = build_experiment_tasks(yaml_path=str(path))

    assert set(tasks) == {"rerun"}
    assert tasks["rerun"].serving_mode == mode
    assert tasks["rerun"].total_gpus == saved["total_gpus"]


def test_removed_fields_warn_and_log_once_per_process(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        Task.from_yaml(_saved_config("agg"))
        Task.from_yaml(_saved_config("disagg"))

    messages = [str(w.message) for w in caught if "Removed legacy AFD Task keys" in str(w.message)]
    assert len(messages) == 1
    assert "afd_total_gpus" in messages[0]
    assert "aisimulate recommend" in messages[0]
    assert sum("Removed legacy AFD Task keys" in record.message for record in caplog.records) == 1


def test_saved_experiment_preserves_overrides():
    with pytest.warns(DeprecationWarning, match="Removed legacy AFD Task keys"):
        restored = Task.from_yaml(_saved_config("agg"), isl=1024, total_gpus=16, serving_mode=None)
    assert restored.isl == 1024
    assert restored.total_gpus == 16
    assert restored.serving_mode == "agg"


@pytest.mark.parametrize("key", ["afd_typo", "totally_made_up_field", "predictor"])
def test_saved_experiment_still_rejects_other_ignored_keys(key):
    saved = _saved_config("agg")
    saved[key] = "invalid"
    with (
        pytest.warns(DeprecationWarning, match="Removed legacy AFD Task keys"),
        pytest.raises(ValueError, match=key),
    ):
        Task.from_yaml(saved)


@pytest.mark.parametrize("use_override", [False, True])
def test_removed_serving_mode_has_actionable_migration_error(use_override):
    saved = _saved_config("agg")
    overrides = {"serving_mode": "afd"} if use_override else {}
    if not use_override:
        saved["serving_mode"] = "afd"
    with pytest.raises(ValueError, match="serving_mode 'afd' is no longer supported") as caught:
        Task.from_yaml(saved, **overrides)
    assert "aisimulate recommend" in str(caught.value)
    assert "aisimulate predict" in str(caught.value)
    assert "workers.<role>.afd" in str(caught.value)


def test_valid_serving_mode_override_takes_precedence():
    saved = _saved_config("agg")
    saved["serving_mode"] = "afd"
    with pytest.warns(DeprecationWarning, match="Removed legacy AFD Task keys"):
        assert Task.from_yaml(saved, serving_mode="agg").serving_mode == "agg"
