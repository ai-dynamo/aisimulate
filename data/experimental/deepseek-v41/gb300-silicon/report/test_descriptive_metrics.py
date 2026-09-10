# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard historical-pair preservation, weighting and incomplete coverage."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("metrics_under_test", ROOT / "descriptive_metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def frozen(root):
    (root / "comparison.json").write_text('{"original":true}\n')
    (root / "original-artifact-hashes.json").write_text(
        json.dumps({"comparison.json": metrics.sha(root / "comparison.json")})
    )
    value = {
        "original_artifact_hashes_sha256": metrics.sha(root / "original-artifact-hashes.json"),
        "comparison_files_sha256": {"comparison.json": metrics.sha(root / "comparison.json")},
    }
    (root / "descriptive-inputs.json").write_text(json.dumps(value))


def test_unequal_duration_and_missing_pairs():
    result = metrics.paired([(10, 20), (100, 90)], 3)
    assert result == {
        "observed": 3,
        "predicted": 2,
        "missing": 1,
        "mape_percent": pytest.approx(55),
        "wape_percent": pytest.approx(100 * 20 / 110),
    }
    assert result["wape_percent"] != 0  # Signed errors cancel; absolute errors do not.


@pytest.mark.parametrize("pairs,count", [([(0, 1)], 1), ([(1, float("nan"))], 1), ([(1, 1)], 0)])
def test_invalid_values_are_not_missing_fallback(pairs, count):
    with pytest.raises(ValueError):
        metrics.paired(pairs, count)


def test_empty_support_has_no_zero_error():
    assert metrics.paired([], 2) == {"observed": 2, "predicted": 0, "missing": 2}


def test_frozen_comparison_mutation_is_rejected(tmp_path):
    frozen(tmp_path)
    metrics.validate_frozen(tmp_path)
    (tmp_path / "comparison.json").write_text('{"original":false}\n')
    with pytest.raises(ValueError, match="frozen comparison changed"):
        metrics.validate_frozen(tmp_path)


def test_native_mape_uses_same_fully_predicted_trials_as_wape(tmp_path, monkeypatch):
    frozen(tmp_path)
    monkeypatch.setattr(metrics, "plot_serving", lambda *args: None)
    good = [
        {"status": "predicted", "dispatch_id": 1, "phase": "decode", "observed_ms": 10, "predicted_ms": 20},
        {"status": "predicted", "dispatch_id": 2, "phase": "decode", "observed_ms": 100, "predicted_ms": 90},
    ]
    partial = [
        {"status": "predicted", "dispatch_id": 3, "phase": "prefill", "observed_ms": 5, "predicted_ms": 5},
        {"status": "missing", "dispatch_id": 4, "phase": "decode", "observed_ms": 1000},
    ]
    trace = {
        "cohorts": [
            {"cohort_id": "a", "purpose": "case", "intervals": good},
            {"cohort_id": "b", "purpose": "case", "intervals": partial},
        ],
        "independent_trial_summary": {"case": {"fully_predicted_trials": 1, "interval_wape_percent": 100 * 20 / 110}},
    }
    e2e = {
        "cohorts": [
            {"cohort_id": "a", "status": "predicted", "observed": {"ttft_ms": 10}, "prediction": {"ttft_ms": 20}}
        ]
    }
    result = metrics.serving(tmp_path, {"hybrid": (e2e, trace)}, ["ttft_ms"])
    assert trace["independent_trial_summary"]["case"]["interval_mape_percent"] == pytest.approx(55)
    assert result["trace"]["hybrid"]["all"]["predicted"] == 3
    assert result["trace"]["hybrid"]["all"]["observed"] == 4
    assert result["native_fully_predicted_trials"]["hybrid"]["case"]["predicted"] == 2
    assert (tmp_path / "comparison.json").read_text() == '{"original":true}\n'


def test_all_historical_inputs_stay_frozen():
    for group in (
        ".",
        "precision-v2",
        "prefix-refinement-v1",
        "serving-v4/off-baseline",
        "serving-v4/off-prefix-refined",
        "serving-v4/on-segmented",
    ):
        metrics.validate_frozen(ROOT / group)


def test_rendered_mape_tables_have_matching_columns():
    for group in (
        ".",
        "precision-v2",
        "prefix-refinement-v1",
        "serving-v4/off-baseline",
        "serving-v4/off-prefix-refined",
        "serving-v4/on-segmented",
    ):
        lines = (ROOT / group / "README.md").read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("|") and "MAPE" in line:
                columns = len(line.split("|"))
                for row in lines[i + 1 :]:
                    if not row.startswith("|"):
                        break
                    assert len(row.split("|")) == columns, (group, row)
