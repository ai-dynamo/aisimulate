# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The collect command without sglang: a kept recording lowers into a table that `--show` prints with its provenance."""

import json
from pathlib import Path

import pytest

from aisimulate.config.engine import FrontendMeasurementConfig
from aisimulate.vl.collect.__main__ import main

from .test_lower import ENVIRONMENT, _level

pytestmark = pytest.mark.unit


MEASUREMENT = FrontendMeasurementConfig(model="m", frontend="python", feature_transport="shm", height=480, width=480)


def _recording() -> dict:
    return {
        "frontend": "python",
        "workers": 2,
        "levels": {"1": _level(1, 20.0), "2": _level(2, 24.0)},
        "send_ms": 3.5,
        "receive_ms": 7.25,
        "environment": ENVIRONMENT,
        "provenance": {
            "host": "serving-host",
            "threads": 8,
            "image_processor_backend": "pil",
            "processor_workers": 2,
            "transformers": "5.12.1",
            "torch": "2.13.0",
        },
        "measured_for": MEASUREMENT.model_dump(mode="json"),
    }


def test_a_recording_lowers_into_a_table_that_show_prints_with_its_provenance(tmp_path, capsys):
    kept = tmp_path / "recording.json"
    kept.write_text(json.dumps(_recording()))
    table = tmp_path / "table.json"
    arguments = ["--table", str(table), "--model", "m", "--frontend", "python", "--images", "480x480"]
    assert main(["--recording", str(kept), *arguments]) == 0
    assert main(["--table", str(table), "--show"]) == 0
    shown = capsys.readouterr().out
    assert MEASUREMENT.describe() in shown
    assert "host serving-host" in shown and "transformers 5.12.1" in shown
    absent = tmp_path / "absent.json"
    with pytest.raises(SystemExit, match="absent.json") as error:
        main(["--table", str(absent), "--show"])
    assert "Traceback" not in str(error.value)


def test_a_tilde_table_path_lands_under_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    kept = tmp_path / "recording.json"
    kept.write_text(json.dumps(_recording()))
    arguments = ["--table", "~/table.json", "--model", "m", "--frontend", "python", "--images", "480x480"]
    assert main(["--recording", str(kept), *arguments]) == 0
    assert (tmp_path / "table.json").exists()
    assert Path(capsys.readouterr().out.split(":", 1)[0]).is_absolute()
