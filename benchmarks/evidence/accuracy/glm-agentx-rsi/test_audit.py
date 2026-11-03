# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoring and GPU accounting guards for the research loop."""

import pytest

from audit import budget, compare_dashboard, metrics, write


def test_pool_requests_and_keep_small_positive_denominators():
    rows = [dict(silicon_ttft_ms=a, predicted_ttft_ms=p, silicon_tpot_ms=a, predicted_tpot_ms=p)
            for a, p in [(1, 2), (2, 2), (0.01, 1), (0, 1), (None, 1)]]
    value = metrics(rows)["tpot"]
    assert value["count"] == 3
    assert value["mape_percent"] == pytest.approx(10000 / 3)
    assert value["reference_below_0_1_ms_contribution_pp"] == 3300


def test_budget_counts_failures_and_reserved_active_limits():
    jobs = [{"job_id": 1, "reserved_gpus": 8, "charged_seconds": 3600},
            {"job_id": 2, "reserved_gpus": 1, "charged_seconds": 20}]
    assert budget(jobs)["charged_gpu_seconds"] == 28820
    with pytest.raises(ValueError, match="Duplicate"):
        budget(jobs + jobs)
    with pytest.raises(ValueError, match="exceeded"):
        budget(jobs, limit=10)


def test_frozen_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / "frozen.json"
    write(path, {"cohort": []})
    with pytest.raises(FileExistsError):
        write(path, {"cohort": ["changed"]})


def test_losing_a_request_cannot_establish_a_win():
    rows = [dict(request_id=str(i), status="matched", silicon_ttft_ms=1, predicted_ttft_ms=2,
                 silicon_tpot_ms=1, predicted_tpot_ms=2, ttft_ms_ape=100, tpot_ms_ape=100) for i in range(2)]
    score = compare_dashboard(rows, rows[:1])
    assert score["tpot"]["lost_count"] == 1
    assert not score["tpot"]["coverage_qualified"]
