# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for paired forward-performance comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.forward_perf_gate import PROTOCOL_VERSION, cases, compare, worker

pytestmark = pytest.mark.unit


def _response(
    case: dict,
    *,
    cold_us: float = 100_000.0,
    warm_us: float = 100.0,
    status: str = "OK",
) -> dict:
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": case["case_id"],
        "case_hash": worker.canonical_case_hash(case),
        "status": status,
    }
    if status == "OK":
        response.update({"cold_us": cold_us, "warm": {"call_median_us": warm_us}})
    else:
        response["error"] = {"type": status, "message": "not available"}
    return response


def _raw(
    warm_ratios: list[float],
    *,
    cold_ratio: float = 1.0,
    warm_base: float = 100.0,
    cold_base: float = 100_000.0,
) -> dict:
    case = cases.expand_cases()[0]
    rounds = []
    for index, warm_ratio in enumerate(warm_ratios, start=1):
        rounds.append(
            {
                "round": index,
                "base": _response(case, cold_us=cold_base, warm_us=warm_base),
                "head": _response(
                    case,
                    cold_us=cold_base * cold_ratio,
                    warm_us=warm_base * warm_ratio,
                ),
            }
        )
    return {
        "base_revision": "base",
        "head_revision": "head",
        "configuration": {
            "mode": "full",
            "matrix_case_count": len(cases.expand_cases()),
            "selected_case_count": 1,
            "expected_case_ids": [case["case_id"]],
        },
        "run_errors": [],
        "cases": [{"case": case, "rounds": rounds}],
    }


def _point(result: dict, metric: str) -> dict:
    return next(point for point in result["points"] if point["metric"] == metric)


def test_four_of_five_rounds_confirm_a_regression() -> None:
    result = compare.compare_raw(_raw([1.11, 1.12, 1.11, 1.12, 1.0]))
    assert _point(result, "warm")["classification"] == "REGRESSION"
    assert _point(result, "warm")["exceed_count"] == 4
    assert _point(result, "warm")["consensus_required"] == 4
    assert result["blocking"] is True


@pytest.mark.parametrize("round_count", [1, 3])
def test_quorum_is_derived_from_available_rounds(round_count: int) -> None:
    raw = _raw([1.12] * round_count)
    if round_count == 1:
        raw["configuration"]["mode"] = "smoke"
    result = compare.compare_raw(raw)
    point = _point(result, "warm")
    assert point["classification"] == "REGRESSION"
    assert point["consensus_required"] == round_count
    if round_count == 1:
        assert "advisory — smoke" in compare.render_markdown(result)


def test_three_of_five_rounds_are_unstable_not_blocking() -> None:
    result = compare.compare_raw(_raw([1.11, 1.12, 1.11, 1.0, 1.0]))
    assert _point(result, "warm")["classification"] == "UNSTABLE"
    assert result["blocking"] is False


@pytest.mark.parametrize("metric", ["cold", "warm"])
def test_absolute_floor_filters_small_relative_change(metric: str) -> None:
    raw = _raw([1.20] * 5, cold_ratio=1.20, warm_base=5.0, cold_base=5.0)
    result = compare.compare_raw(raw)
    assert _point(result, metric)["classification"] == "OK"


def test_cold_regression_can_exceed_its_absolute_floor() -> None:
    result = compare.compare_raw(_raw([1.0] * 5, cold_ratio=1.12))
    point = _point(result, "cold")
    assert point["classification"] == "REGRESSION"
    assert point["exceed_count"] == 5


def test_case_hash_mismatch_is_invalid() -> None:
    raw = _raw([1.0] * 5)
    raw["cases"][0]["rounds"][0]["head"]["case_hash"] = "different"
    result = compare.compare_raw(raw)
    assert _point(result, "warm")["classification"] == "INVALID_COMPARISON"
    assert result["blocking"] is True


def test_missing_case_hash_is_invalid() -> None:
    raw = _raw([1.0] * 5)
    del raw["cases"][0]["rounds"][0]["head"]["case_hash"]
    result = compare.compare_raw(raw)
    assert _point(result, "warm")["classification"] == "INVALID_COMPARISON"
    assert result["blocking"] is True


@pytest.mark.parametrize(
    ("base_status", "head_status", "classification", "blocking"),
    [
        ("DATA_MISS", "DATA_MISS", "SKIPPED", False),
        ("DATA_MISS", "OK", "SKIPPED", False),
        ("OK", "DATA_MISS", "INVALID_COMPARISON", True),
    ],
)
def test_data_miss_status_semantics(
    base_status: str,
    head_status: str,
    classification: str,
    blocking: bool,
) -> None:
    raw = _raw([1.0] * 5)
    case = raw["cases"][0]["case"]
    for paired in raw["cases"][0]["rounds"]:
        paired["base"] = _response(case, status=base_status)
        paired["head"] = _response(case, status=head_status)
    result = compare.compare_raw(raw)
    assert {point["classification"] for point in result["points"]} == {classification}
    assert result["blocking"] is blocking


@pytest.mark.parametrize("missing_sides", [("base",), ("head",), ("base", "head")])
@pytest.mark.parametrize("failed_rounds", [1, 5])
def test_coverage_loss_after_successful_prewarm_is_blocking(missing_sides: tuple[str, ...], failed_rounds: int) -> None:
    raw = _raw([1.0] * 5)
    case = raw["cases"][0]["case"]
    raw["prewarm"] = [
        {"case_id": case["case_id"], "base": _response(case), "head": _response(case), "disposition": "COMPARE"}
    ]
    for paired in raw["cases"][0]["rounds"][:failed_rounds]:
        for side in missing_sides:
            paired[side] = _response(case, status="DATA_MISS")

    result = compare.compare_raw(raw)
    assert result["blocking"] is True
    for point in result["points"]:
        assert point["classification"] == "INVALID_COMPARISON"
        for round_number in range(1, failed_rounds + 1):
            for side in missing_sides:
                assert any(
                    f"round {round_number}:" in reason and f"{side} status is DATA_MISS" in reason
                    for reason in point["invalid_reasons"]
                )


def test_data_miss_skip_reason_includes_priming_failure() -> None:
    case = cases.expand_cases()[0]
    base = _response(case, status="DATA_MISS")
    head = _response(case, status="DATA_MISS")
    base["error"] = {"type": "PRIMING_FAILED", "message": "base prime missing"}
    head["error"] = {"type": "PRIMING_FAILED", "message": "head prime missing"}
    disposition, reason = compare.pair_disposition(case["case_id"], base, head)
    assert disposition == "SKIP"
    assert "PRIMING_FAILED base prime missing" in reason
    assert "PRIMING_FAILED head prime missing" in reason


def test_changed_skip_reason_is_reported_once() -> None:
    raw = _raw([1.0, 1.0])
    case = raw["cases"][0]["case"]
    for index, paired in enumerate(raw["cases"][0]["rounds"]):
        paired["base"] = _response(case, status="DATA_MISS")
        paired["head"] = _response(case, status="DATA_MISS")
        paired["base"]["error"]["message"] = f"reason {index}"
        paired["head"]["error"]["message"] = f"reason {index}"
    result = compare.compare_raw(raw)
    for metric in ("cold", "warm"):
        point = _point(result, metric)
        assert point["classification"] == "INVALID_COMPARISON"
        assert point["invalid_reasons"] == ["response status changed between measured rounds"]


def test_missing_metric_is_invalid_instead_of_crashing() -> None:
    raw = _raw([1.0] * 5)
    del raw["cases"][0]["rounds"][0]["head"]["warm"]
    result = compare.compare_raw(raw)
    assert _point(result, "warm")["classification"] == "INVALID_COMPARISON"
    assert result["blocking"] is True


@pytest.mark.parametrize("value", [True, "100.0"])
def test_non_numeric_metric_is_invalid(value: object) -> None:
    raw = _raw([1.0] * 5)
    raw["cases"][0]["rounds"][0]["head"]["cold_us"] = value
    result = compare.compare_raw(raw)
    assert _point(result, "cold")["classification"] == "INVALID_COMPARISON"
    assert result["blocking"] is True


def test_empty_and_incomplete_case_sets_are_blocking() -> None:
    empty = _raw([1.0] * 5)
    empty["cases"] = []
    empty_result = compare.compare_raw(empty)
    assert empty_result["blocking"] is True
    assert "no benchmark cases were recorded" in empty_result["run_errors"]

    incomplete = _raw([1.0] * 5)
    incomplete["configuration"]["expected_case_ids"].append("missing-case")
    incomplete_result = compare.compare_raw(incomplete)
    assert incomplete_result["blocking"] is True
    assert any("missing-case" in error for error in incomplete_result["run_errors"])

    duplicate = _raw([1.0] * 5)
    duplicate["cases"].append(duplicate["cases"][0])
    duplicate_result = compare.compare_raw(duplicate)
    assert duplicate_result["blocking"] is True
    assert "recorded case IDs are not unique" in duplicate_result["run_errors"]


def test_reported_ratio_and_delta_use_the_same_medians() -> None:
    raw = _raw([1.0] * 3)
    values = [(10.0, 12.0), (10.0, 12.0), (30.0, 20.0)]
    for paired, (base_us, head_us) in zip(raw["cases"][0]["rounds"], values, strict=True):
        paired["base"]["warm"]["call_median_us"] = base_us
        paired["head"]["warm"]["call_median_us"] = head_us
    point = _point(compare.compare_raw(raw), "warm")
    assert point["base_median_us"] == 10.0
    assert point["head_median_us"] == 12.0
    assert point["median_delta_us"] == 2.0
    assert point["median_ratio"] == pytest.approx(1.2)
    assert point["median_ratio"] == point["head_median_us"] / point["base_median_us"]


def test_prewarm_disposition_validates_status_and_metrics() -> None:
    case = cases.expand_cases()[0]
    assert (
        compare.prewarm_disposition(
            case["case_id"],
            _response(case, status="DATA_MISS"),
            _response(case, status="DATA_MISS"),
        )[0]
        == "SKIP"
    )
    assert (
        compare.prewarm_disposition(
            case["case_id"],
            _response(case),
            _response(case, status="DATA_MISS"),
        )[0]
        == "INVALID"
    )
    malformed = _response(case)
    del malformed["warm"]
    assert compare.prewarm_disposition(case["case_id"], _response(case), malformed)[0] == "INVALID"


def test_reports_write_json_csv_markdown_and_annotations(tmp_path: Path) -> None:
    result = compare.compare_raw(_raw([1.11] * 5))
    compare.write_outputs(result, tmp_path)
    assert json.loads((tmp_path / "comparison.json").read_text())["blocking"] is True
    assert "qwen3-32b" in (tmp_path / "comparison.csv").read_text()
    assert "Confirmed regressions" in (tmp_path / "summary.md").read_text()
    assert "REGRESSION" in (tmp_path / "annotations.txt").read_text()

    noisy_dir = tmp_path / "noisy"
    noisy = compare.compare_raw(_raw([1.11, 1.0, 1.0, 1.0, 1.0], cold_ratio=0.8))
    compare.write_outputs(noisy, noisy_dir)
    assert (noisy_dir / "annotations.txt").read_text() == ""


def test_report_summarizes_all_results_and_collapses_noise() -> None:
    result = compare.compare_raw(_raw([1.11, 1.0, 1.0, 1.0, 1.0], cold_ratio=0.8))
    summary = compare.render_markdown(result)
    assert (
        "**PASS** — 1 of 2 comparisons stable: 1 faster, 0 unchanged, 0 regressions; 1 noisy, 0 invalid, 0 skipped."
    ) in summary
    assert "<summary>⚠️ Noisy comparisons (1)</summary>" in summary
    assert "| case | cache | base | head | change | rounds |" in summary
    assert "<summary>Full matrix (2 cells)</summary>" in summary
    visible = summary.split("<details>", maxsplit=1)[0]
    assert "| status | model |" not in visible
    assert "Confirmed regressions" not in visible
    assert "</details>" in summary


def test_report_puts_regressions_first_and_full_matrix_in_details() -> None:
    raw = _raw([1.11] * 5)
    stable_case = next(case for case in cases.expand_cases() if case["model_id"] == "qwen3-235b-a22b")
    raw["configuration"]["expected_case_ids"].append(stable_case["case_id"])
    raw["cases"].append(
        {
            "case": stable_case,
            "rounds": [
                {
                    "round": index,
                    "base": _response(stable_case),
                    "head": _response(stable_case),
                }
                for index in range(1, 6)
            ],
        }
    )

    result = compare.compare_raw(raw)
    assert result["cells"][0]["model_id"] == "qwen3-32b"
    assert result["cells"][0]["metric"] == "warm"
    summary = compare.render_markdown(result)
    assert summary.index("### ❌ Confirmed regressions") < summary.index("<details>")
    assert "| `qwen3-32b/silicon/context/bs1-isl1024` | warm | 100.00 µs | 111.00 µs" in summary
    assert "<summary>Full matrix (4 cells)</summary>" in summary
    matrix = summary.split("<summary>Full matrix", maxsplit=1)[1]
    first_row = next(line for line in matrix.splitlines() if line.startswith("| ❌"))
    assert "qwen3-32b" in first_row
    assert "| regressions | noisy | invalid |" in summary


def test_report_keeps_blocking_errors_visible_and_collapses_skips() -> None:
    invalid_raw = _raw([1.0] * 5)
    invalid_raw["run_errors"].append("worker process failed")
    invalid_raw["cases"][0]["rounds"][0]["head"]["case_hash"] = "different"
    invalid_summary = compare.render_markdown(compare.compare_raw(invalid_raw))
    visible = invalid_summary.split("<details>", maxsplit=1)[0]
    assert "### ❌ Run errors" in visible
    assert "worker process failed" in visible
    assert "### ❌ Invalid comparisons" in visible
    assert "base and head case hashes differ" in visible
    assert "Noisy comparisons" not in invalid_summary
    assert "Skipped comparisons" not in invalid_summary

    skipped_raw = _raw([1.0] * 5)
    case = skipped_raw["cases"][0]["case"]
    for paired in skipped_raw["cases"][0]["rounds"]:
        paired["base"] = _response(case, status="DATA_MISS")
        paired["head"] = _response(case, status="DATA_MISS")
    skipped_summary = compare.render_markdown(compare.compare_raw(skipped_raw))
    assert "<summary>Skipped comparisons (2)</summary>" in skipped_summary
