# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped missing-data exceptions never turn an unavailable prediction into OK."""

from __future__ import annotations

import collections
import csv
import hashlib
import json

import pytest
from tools.prediction_regression_gate import compare, expectations, report

pytestmark = pytest.mark.unit
COMBO = "h200_sxm/vllm/0.24.0.csv"
KEY = ("zai-org/GLM-5.2-FP8", "8", "1", "1", "1", "8", "fp8", "ctx", "1", "1024")


def cause(phase="context"):
    return (
        f"perf database error: {phase} DSA module data missing for DsaKey {{ "
        'architecture: "GlmMoeDsaForCausalLM", fmha_quant: "bfloat16", '
        'kv_quant: "fp8", gemm_quant: "fp8" }'
    )


def row(status="OK", *, diagnostic="", err="", key=KEY):
    return dict(zip(compare.KEY_FIELDS, key, strict=True)) | {
        "status": status,
        "value_ms": "1.0" if status == "OK" else "",
        "err": err,
        "data_miss_detail": diagnostic,
    }


def snapshot(root, rows, combo=COMBO):
    path = root / combo
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*compare.KEY_FIELDS, "status", "value_ms", "err", "data_miss_detail"])
        writer.writeheader()
        writer.writerows(rows)
    return root


def evaluate(tmp_path, old, new, *, combo=COMBO, key=KEY, enabled=True):
    old_dir = snapshot(tmp_path / "old", old, combo)
    new_dir = snapshot(tmp_path / "new", new, combo)
    results = report.compare_snapshots(old_dir, new_dir, rtol=1e-4)
    expectations.apply_expectations(results, old_dir, new_dir, [expectations.ExpectedDataMiss(COMBO, key, enabled)])
    return [d for result in results for d in result.diffs]


@pytest.mark.parametrize("phase,label", [("context", "ctx"), ("generation", "gen")])
def test_only_exact_native_data_gap_is_expected(tmp_path, phase, label):
    key = (*KEY[:7], label, *KEY[8:])
    diagnostic = cause(phase) + " at /checkout/systems/data/h200_sxm/vllm/0.24.0"
    diffs = evaluate(tmp_path, [row(key=key)], [row("DATA_MISS", diagnostic=diagnostic, key=key)], key=key)
    assert [d.category for d in diffs] == ["EXPECTED_DATA_MISS"]
    assert (diffs[0].old_status, diffs[0].new_status) == ("OK", "DATA_MISS")
    assert diffs[0].new_data_miss_detail == diagnostic
    assert not any(d.category in compare.BLOCKING_CATEGORIES for d in diffs)


def test_after_merge_identical_missing_data_remains_visible(tmp_path):
    missing = row("DATA_MISS", diagnostic=cause())
    diffs = evaluate(tmp_path, [missing], [missing])
    assert [d.category for d in diffs] == ["EXPECTED_DATA_MISS"]
    assert diffs[0].old_status == diffs[0].new_status == "DATA_MISS"
    assert diffs[0].old_data_miss_detail == diffs[0].new_data_miss_detail == cause()


def test_followup_reopens_same_missing_case_even_after_engine_merge(tmp_path):
    missing = row("DATA_MISS", diagnostic=cause())
    diffs = evaluate(tmp_path, [missing], [missing], enabled=False)
    assert [d.category for d in diffs] == ["REOPENED_FAILURE"]
    assert diffs[0].category in compare.BLOCKING_CATEGORIES
    assert diffs[0].old_status == diffs[0].new_status == "DATA_MISS"


def test_followup_measured_coverage_passes_as_gain_without_xpass(tmp_path):
    diffs = evaluate(tmp_path, [row("DATA_MISS", diagnostic=cause())], [row()], enabled=False)
    assert [d.category for d in diffs] == ["GAIN"]
    assert not any(d.category in compare.BLOCKING_CATEGORIES for d in diffs)


def test_followup_missing_entire_combo_still_blocks(tmp_path):
    results = []
    expectations.apply_expectations(
        results,
        tmp_path / "absent-old",
        tmp_path / "absent-new",
        [expectations.ExpectedDataMiss(COMBO, KEY, False)],
    )
    assert len(results) == 1 and results[0].combo == COMBO
    assert [d.category for d in results[0].diffs] == ["REOPENED_FAILURE"]
    assert results[0].diffs[0].category in compare.BLOCKING_CATEGORIES


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", ""])
def test_reopened_ok_requires_valid_positive_latency(tmp_path, value):
    diffs = evaluate(tmp_path, [row("DATA_MISS", diagnostic=cause())], [row() | {"value_ms": value}], enabled=False)
    assert "REOPENED_FAILURE" in {d.category for d in diffs}


@pytest.mark.parametrize("extra", [{"err": "ValueError"}, {"data_miss_detail": cause()}])
def test_reopened_ok_cannot_hide_an_error(tmp_path, extra):
    diffs = evaluate(tmp_path, [row()], [row() | extra], enabled=False)
    assert "REOPENED_FAILURE" in {d.category for d in diffs}


@pytest.mark.parametrize("baseline", [row(), row("DATA_MISS", diagnostic=cause())])
def test_restored_data_is_blocking_xpass_even_after_merge(tmp_path, baseline):
    diffs = evaluate(tmp_path, [baseline], [row()])
    xpass = next(d for d in diffs if d.category == "XPASS")
    assert xpass.category in compare.BLOCKING_CATEGORIES
    assert "disable the expected-failure waiver" in xpass.detail


@pytest.mark.parametrize(
    "candidate",
    [
        row("INVALID", err="ValueError"),
        row("DATA_MISS", diagnostic="perf database error: GEMM data missing"),
        row("DATA_MISS", diagnostic=cause().replace('gemm_quant: "fp8"', 'gemm_quant: "nvfp4"')),
        row("DATA_MISS", diagnostic=cause().replace('kv_quant: "fp8"', 'kv_quant: "bfloat16"')),
        row("DATA_MISS", diagnostic=cause().replace('fmha_quant: "bfloat16"', 'fmha_quant: "fp8"')),
        row("DATA_MISS", diagnostic=cause().replace("GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM")),
        row("DATA_MISS", diagnostic=cause("generation")),
        row("DATA_MISS", diagnostic=cause(), err="RuntimeError"),
        row("DATA_MISS", diagnostic=cause()) | {"value_ms": "1.0"},
        row("DATA_MISS"),  # old snapshots without native diagnostic are insufficient
    ],
)
def test_another_failure_is_never_waived(tmp_path, candidate):
    diffs = evaluate(tmp_path, [row()], [candidate])
    assert {d.category for d in diffs} == {"REGRESSION", "EXPECTATION_MISMATCH"}
    assert all(d.category in compare.BLOCKING_CATEGORIES for d in diffs)


@pytest.mark.parametrize(
    "baseline",
    [
        row("INVALID", err="ValueError"),
        row("DATA_MISS", diagnostic="unrelated old op missing"),
    ],
)
def test_nonmatching_baseline_is_not_a_known_data_gap(tmp_path, baseline):
    diffs = evaluate(tmp_path, [baseline], [row("DATA_MISS", diagnostic=cause())])
    assert "EXPECTED_DATA_MISS" not in {d.category for d in diffs}
    assert "EXPECTATION_MISMATCH" in {d.category for d in diffs}


@pytest.mark.parametrize(
    "combo",
    [
        "b200_sxm/vllm/0.24.0.csv",
        "h200_sxm/sglang/0.5.14.csv",
        "h200_sxm/trtllm/1.3.0.csv",
        "h200_sxm/vllm/0.25.1.csv",
    ],
)
def test_other_gpu_backend_version_regressions_unchanged(tmp_path, combo):
    diffs = evaluate(tmp_path, [row()], [row("DATA_MISS", diagnostic=cause())], combo=combo)
    assert [d.category for d in diffs] == ["REGRESSION"]


@pytest.mark.parametrize("field,value", [("model", "deepseek-ai/DeepSeek-V3.2"), ("quant", "nvfp4"), ("isl", "2048")])
def test_unlisted_model_precision_shape_stays_blocking(tmp_path, field, value):
    old = row() | {field: value}
    new = row("DATA_MISS", diagnostic=cause()) | {field: value}
    diffs = evaluate(tmp_path, [old], [new])
    assert "REGRESSION" in {d.category for d in diffs}
    assert "EXPECTED_DATA_MISS" not in {d.category for d in diffs}


def test_removing_expected_row_blocks(tmp_path):
    diffs = evaluate(tmp_path, [row()], [])
    assert {d.category for d in diffs} == {"ROWS_REMOVED", "EXPECTATION_MISMATCH"}


def test_duplicate_snapshot_cannot_hide_a_failure(tmp_path):
    root = snapshot(tmp_path, [row(), row("INVALID", err="ValueError")])
    with pytest.raises(ValueError, match="duplicate snapshot identity"):
        compare.load_rows(root / COMBO)


def write_expectations(path, entries):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(expectations.HEADER)
        writer.writerows(entries)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "case_count": len(entries),
                "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_failure_enabled": True,
            }
        )
    )
    return path


def test_nonempty_alternate_list_requires_manifest(tmp_path):
    path = write_expectations(tmp_path / "expect.csv", [(COMBO, *KEY)])
    assert len(expectations.load_expectations(path)) == 1
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="require a reviewed manifest"):
        expectations.load_expectations(path)


@pytest.mark.parametrize("field,value", [("case_count", 0), ("csv_sha256", ""), ("expected_failure_enabled", None)])
def test_alternate_list_manifest_cannot_bypass_reviewed_policy(tmp_path, field, value):
    path = write_expectations(tmp_path / "expect.csv", [(COMBO, *KEY)])
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="count/hash mismatch|must be an explicit boolean"):
        expectations.load_expectations(path)


@pytest.mark.parametrize(
    "entry",
    [
        ("b200_sxm/vllm/0.24.0.csv", *KEY),
        ("h200_sxm/sglang/0.5.14.csv", *KEY),
        ("h200_sxm/trtllm/1.3.0.csv", *KEY),
        ("h200_sxm/vllm/0.25.1.csv", *KEY),
        (COMBO, "*", *KEY[1:]),
        (COMBO, *KEY[:6], "nvfp4", *KEY[7:]),
    ],
)
def test_loading_out_of_scope_expectation_fails(tmp_path, entry):
    path = write_expectations(tmp_path / "expect.csv", [entry])
    with pytest.raises(ValueError, match="outside the authorized"):
        expectations.load_expectations(path)


def test_duplicate_expectations_fail(tmp_path):
    path = write_expectations(tmp_path / "expect.csv", [(COMBO, *KEY)] * 2)
    with pytest.raises(ValueError, match="duplicate expectation"):
        expectations.load_expectations(path)


def test_checked_in_list_matches_reviewed_manifest():
    entries = expectations.load_expectations()
    manifest = json.loads(expectations.DEFAULT_PATH.with_suffix(".json").read_text())
    assert len(entries) == manifest["case_count"] == 540
    assert (
        hashlib.sha256(expectations.DEFAULT_PATH.read_bytes()).hexdigest()
        == "95f2d1294cec9fa2961dbe4572b4b554c5edb5e8fbd082b667500e249e317c79"
    )
    assert dict(collections.Counter(e.combo for e in entries)) == manifest["by_combo"]
    assert manifest["source_head"] == "3e9fddd487cf163a58ec2d2bd58927c7c3ac4527"
    assert all(e.combo.split("/")[1] == "vllm" and e.combo.split("/")[0] != "b200_sxm" for e in entries)


def test_report_preserves_missing_count_and_causes_and_exit_code(tmp_path, monkeypatch):
    expectation_path = write_expectations(tmp_path / "expect.csv", [(COMBO, *KEY)])
    old = snapshot(tmp_path / "old", [row()])
    new = snapshot(tmp_path / "new", [row("DATA_MISS", diagnostic=cause())])
    out = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "report.py",
            "--old",
            str(old),
            "--new",
            str(new),
            "--report-dir",
            str(out),
            "--expected-data-misses",
            str(expectation_path),
        ],
    )
    assert report.main() == 0
    text = (out / "summary.md").read_text()
    assert "EXPECTED_DATA_MISS | 1" in text and "not successful estimates" in text
    with (out / "drift_report.csv").open() as f:
        recorded = list(csv.DictReader(f))
    assert recorded[0]["old_status"] == "OK" and recorded[0]["new_status"] == "DATA_MISS"
    assert recorded[0]["new_data_miss_detail"] == cause()
    snapshot(new, [row()])
    assert report.main() == 1
    assert "XPASS" in (out / "summary.md").read_text()
    # The stacked data follow-up disables the waiver while retaining the case.
    expectation_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "case_count": 1,
                "csv_sha256": hashlib.sha256(expectation_path.read_bytes()).hexdigest(),
                "expected_failure_enabled": False,
            }
        )
    )
    assert len(expectations.load_expectations(expectation_path)) == 1
    assert report.main() == 0
    snapshot(new, [row("DATA_MISS", diagnostic=cause())])
    snapshot(old, [row("DATA_MISS", diagnostic=cause())])
    assert report.main() == 1
    assert "REOPENED_FAILURE" in (out / "summary.md").read_text()
    # The report's old-harness-missing degradation must not bypass reopened cases.
    (old / COMBO).unlink()
    assert report.main() == 1
    assert "REOPENED_FAILURE" in (out / "summary.md").read_text()


def run_report(monkeypatch, old, new, output, expectation_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "report.py",
            "--old",
            str(old),
            "--new",
            str(new),
            "--report-dir",
            str(output),
            "--expected-data-misses",
            str(expectation_path),
        ],
    )
    return report.main()


@pytest.mark.parametrize(
    "candidate,category",
    [
        (row("INVALID", err="ValueError"), "EXPECTATION_MISMATCH"),
        (row("DATA_MISS", diagnostic="unrelated GEMM missing"), "EXPECTATION_MISMATCH"),
        (row("DATA_MISS", diagnostic=cause()), "EXPECTATION_MISMATCH"),
        (row(), "XPASS"),
    ],
)
def test_enabled_cases_are_checked_without_baseline_harness(tmp_path, monkeypatch, candidate, category):
    expectation_path = write_expectations(tmp_path / "expect.csv", [(COMBO, *KEY)])
    new = snapshot(tmp_path / "new", [candidate])
    output = tmp_path / "report"
    assert run_report(monkeypatch, tmp_path / "absent-old", new, output, expectation_path) == 1
    with (output / "drift_report.csv").open() as f:
        recorded = list(csv.DictReader(f))
    blocked = [item for item in recorded if item["category"] in compare.BLOCKING_CATEGORIES]
    assert len(blocked) == 1 and blocked[0]["category"] == category
    assert blocked[0]["old_status"] == ""
    assert blocked[0]["new_status"] == candidate["status"]
    assert blocked[0]["new_err"] == candidate["err"]
    assert blocked[0]["new_data_miss_detail"] == candidate["data_miss_detail"]


def test_enabled_no_harness_unlisted_only_snapshot_retains_statistics_path(tmp_path, monkeypatch):
    expectation_path = write_expectations(tmp_path / "expect.csv", [(COMBO, *KEY)])
    new = snapshot(tmp_path / "new", [row("INVALID", err="ValueError")], "h200_sxm/sglang/0.5.14.csv")
    output = tmp_path / "report"
    assert run_report(monkeypatch, tmp_path / "absent-old", new, output, expectation_path) == 0
    summary = (output / "summary.md").read_text()
    assert "Old side has no snapshot" in summary and "New-side statistics" in summary


def test_observed_combo_requires_listed_rows_even_without_baseline_harness(tmp_path, monkeypatch):
    expectation_path = write_expectations(tmp_path / "expect.csv", [(COMBO, *KEY)])
    new = snapshot(tmp_path / "new", [row() | {"model": "unlisted/model"}])
    output = tmp_path / "report"
    assert run_report(monkeypatch, tmp_path / "absent-old", new, output, expectation_path) == 1
    assert "EXPECTATION_MISMATCH | 1" in (output / "summary.md").read_text()


def test_disabled_reviewed_list_requires_all_540_candidates_without_baseline(tmp_path, monkeypatch):
    # The follow-up changes the boolean only; the actual reviewed identities
    # remain byte-identical and mandatory, even for combos absent on both sides.
    expectation_path = tmp_path / "expect.csv"
    original_bytes = expectations.DEFAULT_PATH.read_bytes()
    expectation_path.write_bytes(original_bytes)
    manifest = json.loads(expectations.DEFAULT_PATH.with_suffix(".json").read_text())
    manifest["expected_failure_enabled"] = False
    expectation_path.with_suffix(".json").write_text(json.dumps(manifest))
    entries = expectations.load_expectations(expectation_path)
    assert len(entries) == 540 and all(not entry.expected_failure_enabled for entry in entries)
    by_combo = collections.defaultdict(list)
    for entry in entries:
        by_combo[entry.combo].append(row(key=entry.key))
    new = tmp_path / "new"
    for combo, rows in by_combo.items():
        snapshot(new, rows, combo)
    output = tmp_path / "report"
    old = tmp_path / "absent-old"
    assert run_report(monkeypatch, old, new, output, expectation_path) == 0

    missing_combo = sorted(by_combo)[0]
    (new / missing_combo).unlink()
    assert run_report(monkeypatch, old, new, output, expectation_path) == 1
    with (output / "drift_report.csv").open() as f:
        reopened = [item for item in csv.DictReader(f) if item["category"] == "REOPENED_FAILURE"]
    assert len(reopened) == 108
    assert {item["combo"] for item in reopened} == {missing_combo}
    assert all(item["old_status"] == item["new_status"] == "" for item in reopened)

    unlisted = snapshot(tmp_path / "unlisted", [row()], "h200_sxm/sglang/0.5.14.csv")
    assert run_report(monkeypatch, old, unlisted, output, expectation_path) == 1
    with (output / "drift_report.csv").open() as f:
        reopened = [item for item in csv.DictReader(f) if item["category"] == "REOPENED_FAILURE"]
    assert len(reopened) == 540
    assert expectation_path.read_bytes() == original_bytes
