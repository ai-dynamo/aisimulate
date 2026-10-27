# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the prediction-regression-gate old-vs-new comparison logic."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from tools.prediction_regression_gate import compare, grid

pytestmark = pytest.mark.unit

HEADER = [*compare.KEY_FIELDS, "status", "value_ms", "err"]


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(status: str = "OK", value_ms: str = "10.000000", err: str = "", **overrides) -> dict:
    row = {
        "model": "org/model",
        "tp": "4",
        "pp": "1",
        "adp": "1",
        "moe_tp": "",
        "moe_ep": "",
        "quant": "default",
        "phase": "ctx",
        "bs": "1",
        "isl": "1024",
        "status": status,
        "value_ms": value_ms,
        "err": err,
    }
    row.update(overrides)
    return row


def _compare(tmp_path: Path, base_rows: list[dict], cur_rows: list[dict], rtol: float = 1e-4):
    baseline = _write(tmp_path / "baseline.csv", base_rows)
    current = _write(tmp_path / "current.csv", cur_rows)
    return compare.compare_combo("sys/backend/1.0.csv", baseline, current, rtol=rtol)


def test_identical_rows_produce_no_diffs(tmp_path: Path) -> None:
    rows = [_row(), _row(phase="gen", bs="32", status="DATA_MISS", value_ms="")]
    result = _compare(tmp_path, rows, rows)
    assert result.diffs == []
    assert result.rows_compared == 2


@pytest.mark.parametrize("side", ["old", "new"])
@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_identity_rejected_on_either_side(tmp_path: Path, side: str, reverse: bool) -> None:
    duplicates = [_row(), _row(status="DATA_MISS", value_ms="")]
    if reverse:
        duplicates.reverse()
    old = duplicates if side == "old" else [_row()]
    new = duplicates if side == "new" else [_row()]
    with pytest.raises(ValueError, match="duplicate snapshot identity"):
        _compare(tmp_path, old, new)


def test_ok_to_miss_is_regression(tmp_path: Path) -> None:
    result = _compare(tmp_path, [_row()], [_row(status="DATA_MISS", value_ms="")])
    assert [d.category for d in result.diffs] == ["REGRESSION"]


def test_miss_to_ok_is_gain(tmp_path: Path) -> None:
    result = _compare(tmp_path, [_row(status="DATA_MISS", value_ms="")], [_row()])
    assert [d.category for d in result.diffs] == ["GAIN"]


def test_value_drift_beyond_rtol_flagged(tmp_path: Path) -> None:
    result = _compare(tmp_path, [_row(value_ms="10.000000")], [_row(value_ms="10.002000")])
    assert [d.category for d in result.diffs] == ["DRIFT"]


def test_value_drift_within_rtol_ignored(tmp_path: Path) -> None:
    result = _compare(tmp_path, [_row(value_ms="10.000000")], [_row(value_ms="10.000500")], rtol=1e-4)
    assert result.diffs == []


def test_invalid_error_type_change_flagged(tmp_path: Path) -> None:
    base = [_row(status="INVALID", value_ms="", err="ValueError")]
    cur = [_row(status="INVALID", value_ms="", err="KeyError")]
    result = _compare(tmp_path, base, cur)
    assert [d.category for d in result.diffs] == ["STATUS_CHANGE"]


def test_added_and_removed_rows_flagged(tmp_path: Path) -> None:
    base = [_row(), _row(isl="8192")]
    cur = [_row(), _row(isl="32768")]
    result = _compare(tmp_path, base, cur)
    assert sorted(d.category for d in result.diffs) == ["ROWS_ADDED", "ROWS_REMOVED"]


def test_summarize_and_report(tmp_path: Path) -> None:
    result = _compare(tmp_path, [_row()], [_row(status="INVALID", value_ms="", err="ValueError")])
    text = compare.summarize([result])
    assert "REGRESSION" in text
    report = tmp_path / "report.csv"
    compare.write_report([result], report)
    lines = report.read_text().splitlines()
    assert len(lines) == 2 and "REGRESSION" in lines[1]


# ---------------------------------------------------------------------------
# Declared model relabels
# ---------------------------------------------------------------------------

# The plain identifier moves from the BF16 checkpoint to the FP8 one, which
# lacks data on this combo; the BF16 checkpoint moves to a suffixed name.
RENAME = {"org/m-FP8": "org/m", "org/m": "org/m-BF16"}
PRE_RENAME = [_row(model="org/m-FP8", status="DATA_MISS", value_ms=""), _row(model="org/m", value_ms="10.000000")]
POST_RENAME = [_row(model="org/m", status="DATA_MISS", value_ms=""), _row(model="org/m-BF16", value_ms="10.000000")]


def _compare_relabeled(tmp_path: Path, old_rows: list[dict], new_rows: list[dict], relabels: dict[str, str]):
    old = _write(tmp_path / "old.csv", old_rows)
    new = _write(tmp_path / "new.csv", new_rows)
    return compare.compare_combo("sys/backend/1.0.csv", old, new, relabels=relabels)


def test_rename_without_declaration_reads_as_regression(tmp_path: Path) -> None:
    result = _compare_relabeled(tmp_path, PRE_RENAME, POST_RENAME, relabels={})
    assert sorted(d.category for d in result.diffs) == ["REGRESSION", "ROWS_ADDED", "ROWS_REMOVED"]


def test_declared_rename_compares_each_model_under_its_new_identifier(tmp_path: Path) -> None:
    result = _compare_relabeled(tmp_path, PRE_RENAME, POST_RENAME, relabels=RENAME)
    assert [(d.category, d.key, d.detail) for d in result.diffs] == [
        ("RELABELED", ("org/m",), "1 old rows compared as org/m-BF16"),
        ("RELABELED", ("org/m-FP8",), "1 old rows compared as org/m"),
    ]
    assert result.rows_compared == 2
    assert not any(d.category in compare.BLOCKING_CATEGORIES for d in result.diffs)


def test_declared_rename_still_blocks_a_real_regression(tmp_path: Path) -> None:
    broken = [POST_RENAME[0], _row(model="org/m-BF16", status="INVALID", value_ms="", err="ValueError")]
    result = _compare_relabeled(tmp_path, PRE_RENAME, broken, relabels=RENAME)
    regressions = [d for d in result.diffs if d.category == "REGRESSION"]
    assert [d.key[0] for d in regressions] == ["org/m-BF16"]


@pytest.mark.parametrize(
    "old_rows",
    [
        pytest.param(POST_RENAME, id="rename-already-on-base"),
        pytest.param([*PRE_RENAME, _row(model="org/m-BF16")], id="introduced-identifier-already-on-base"),
    ],
)
def test_declared_rename_is_inert_unless_old_side_predates_it(tmp_path: Path, old_rows: list[dict]) -> None:
    result = _compare_relabeled(tmp_path, old_rows, POST_RENAME, relabels=RENAME)
    assert "RELABELED" not in {d.category for d in result.diffs}


def test_declared_rename_is_inert_while_new_side_keeps_a_retired_identifier(tmp_path: Path) -> None:
    result = _compare_relabeled(tmp_path, PRE_RENAME, PRE_RENAME, relabels=RENAME)
    assert result.diffs == []


def test_declared_rename_rejects_a_new_side_that_lacks_an_introduced_identifier(tmp_path: Path) -> None:
    # The new snapshot lost the BF16 rows while the plain identifier changed status. Without the
    # check, the old plain rows surface as report-only ROWS_REMOVED and the gate passes.
    incomplete = [POST_RENAME[0]]
    with pytest.raises(ValueError, match="incomplete.*org/m-BF16"):
        _compare_relabeled(tmp_path, PRE_RENAME, incomplete, relabels=RENAME)


def test_relabels_must_be_one_to_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        _compare_relabeled(tmp_path, PRE_RENAME, POST_RENAME, relabels={"org/a": "org/m", "org/b": "org/m"})


def test_shipped_relabels_match_the_bundled_model_list() -> None:
    bundled = set(grid.bundled_models())
    targets = set(compare.MODEL_RELABELS.values())
    assert len(targets) == len(compare.MODEL_RELABELS)
    assert targets <= bundled
    assert not (compare.MODEL_RELABELS.keys() - targets) & bundled


# ---------------------------------------------------------------------------
# Tier-2 comparison
# ---------------------------------------------------------------------------

TIER2_HEADER = ["id", "status", "ttft_ms", "tpot_ms"]


def _tier2_row(config_id: str = "dense_agg_c8", status: str = "OK", ttft: str = "100.0", tpot: str = "10.0") -> dict:
    return {
        "id": config_id,
        "status": status,
        "ttft_ms": ttft if status == "OK" else "",
        "tpot_ms": tpot if status == "OK" else "",
    }


def _compare_tier2(tmp_path: Path, old_rows: list[dict], new_rows: list[dict], rtol: float = 1e-4):
    def write(name: str, rows: list[dict]) -> Path:
        path = tmp_path / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TIER2_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    return compare.compare_tier2(write("old.csv", old_rows), write("new.csv", new_rows), rtol=rtol)


def test_tier2_identical_no_diffs(tmp_path: Path) -> None:
    rows = [_tier2_row(), _tier2_row("oom_boundary", status="RuntimeError")]
    result = _compare_tier2(tmp_path, rows, rows)
    assert result.diffs == []
    assert result.rows_compared == 2


def test_tier2_ok_to_exception_is_regression(tmp_path: Path) -> None:
    result = _compare_tier2(tmp_path, [_tier2_row()], [_tier2_row(status="ValueError")])
    assert [d.category for d in result.diffs] == ["REGRESSION"]


def test_tier2_per_metric_drift(tmp_path: Path) -> None:
    result = _compare_tier2(tmp_path, [_tier2_row(ttft="100.0", tpot="10.0")], [_tier2_row(ttft="103.0", tpot="10.2")])
    assert [d.category for d in result.diffs] == ["DRIFT", "DRIFT"]
    assert "ttft_ms" in result.diffs[0].detail and "tpot_ms" in result.diffs[1].detail


def test_tier2_exception_type_change_flagged(tmp_path: Path) -> None:
    result = _compare_tier2(tmp_path, [_tier2_row(status="ValueError")], [_tier2_row(status="RuntimeError")])
    assert [d.category for d in result.diffs] == ["STATUS_CHANGE"]


def test_tier2_config_added_and_removed(tmp_path: Path) -> None:
    result = _compare_tier2(tmp_path, [_tier2_row("a"), _tier2_row("b")], [_tier2_row("a"), _tier2_row("c")])
    assert sorted(d.category for d in result.diffs) == ["ROWS_ADDED", "ROWS_REMOVED"]
