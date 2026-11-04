# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown coverage and failed validation must never become finite forecasts."""

import csv

import pytest
from tools.perf_database import collection_cost_report as report

pytestmark = pytest.mark.unit


def sample(length=1, seconds=2, **extra):
    return {
        "system": "b300_sxm",
        "backend": "vllm",
        "operation": "attention_context",
        "case": {"input_len": str(length), "head_dim": "128", "dispatch": "native", "attn_dtype": "fp8"},
        "group_id": "group",
        "measurement_scope": "point",
        "collection_seconds": seconds,
        "setup_seconds": None,
        "forward_latency_ms": 999,
        "source_commit": "abc",
        "status": "passed",
        **extra,
    }


def test_setup_once_and_forward_not_added_twice():
    points = [sample(1, 2, setup_seconds=5), sample(2, 3, setup_seconds=5)]
    assert report.measured_cost(points) == 10
    assert report.measured_cost([*points, sample(seconds=11, measurement_scope="group")]) == 11


def test_failures_and_timeouts_cannot_fit():
    rows = [sample(1, 1), sample(3, 3, status="timeout"), sample(3, 2, status="failed")]
    assert report.predict(sample(2)["case"], rows) is None


def test_no_geometry_precision_dispatch_extrapolation():
    rows = [sample(1, 1), sample(3, 3)]
    assert report.predict(sample(2)["case"], rows) == (2, 2)
    assert report.predict(sample(4)["case"], rows) is None
    for key, value in (("head_dim", "256"), ("dispatch", "other"), ("attn_dtype", "bf16"), ("model_path", "new")):
        assert report.predict({**sample(2)["case"], key: value}, rows) is None


def test_heldout_twenty_percent_gate():
    rows = [sample(1, 1), sample(3, 3), sample(2, 2, role="holdout")]
    assert report.validation(rows)["passed"]
    rows[-1]["collection_seconds"] = 4
    assert not report.validation(rows)["passed"]
    assert not report.validation(rows[:2])["passed"]


def test_group_fifteen_percent_gate_and_missing_setup():
    rows = [
        sample(1, 2, setup_seconds=5),
        sample(2, 3, setup_seconds=5),
        sample(seconds=10, measurement_scope="group", role="holdout"),
    ]
    assert report.group_validation(rows)["passed"]
    rows[-1]["collection_seconds"] = 20
    assert not report.group_validation(rows)["passed"]
    assert not report.group_validation([sample(), rows[-1]])["passed"]


def test_exact_paired_groups_include_setup_growth_once():
    before = sample(
        seconds=10,
        measurement_scope="group",
        scenario="baseline",
        paired_group_id="p",
        group_member_cases=[sample(1)["case"]],
    )
    after = sample(
        seconds=16,
        measurement_scope="group",
        scenario="model",
        paired_group_id="p",
        group_member_cases=[sample(1)["case"], sample(2)["case"]],
    )
    added = {report.case_key(report.clean_case(sample(2)["case"]))}
    result = report.paired_groups([before, after], "model", added, set())
    assert result["complete"] and result["gpu_seconds_range"] == [6, 6]
    assert not report.paired_groups([after], "model", added, set())["complete"]
    after["status"] = "timeout"
    assert not report.paired_groups([before, after], "model", added, set())["complete"]


def test_missing_stratum_keeps_total_unknown(tmp_path):
    case = sample()["case"]
    for name, rows in (("baseline", [case]), ("model", [case, sample(2)["case"]])):
        directory = tmp_path / name
        directory.mkdir()
        with (directory / "vllm_sm103_attention_context.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=case)
            writer.writeheader()
            writer.writerows(rows)
    result = report.report_lane(tmp_path / "baseline", tmp_path / "model", [], "b300_sxm", "vllm")
    operation = result["attention_context"]
    assert operation["added_points"] == operation["uncovered_added_points"] == 1
    assert operation["extra_gpu_hours_range"] is None


def test_unrelated_source_revisions_do_not_form_interpolation_corners():
    assert report.predict(sample(2)["case"], [sample(1, 1), sample(3, 3, source_commit="different")]) is None
