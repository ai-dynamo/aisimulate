# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statistical independence and incomplete-coverage regression checks."""

from copy import deepcopy

import pytest
from analyze_e2e import analyze, summarize, trial_metrics


def fixture():
    plan = {"sampling_role": "pilot", "run_id": "test", "requested_trials": 10, "cohorts": []}
    progress = []
    for trial in range(10):
        request = {
            "request_id": f"request-{trial}",
            "valid": True,
            "ttft_ms": 100,
            "average_tpot_ms": 10,
            "exact_itl_available": True,
            "itl_ms": [10] * 31,
        }
        cohort = {
            "cohort_id": str(trial),
            "purpose": "example",
            "trial_index": trial,
            "trial_seed": trial,
            "comparison_role": "primary",
            "requests": [request],
        }
        plan["cohorts"].append(cohort)
        progress.append(cohort | {"valid": True, "output_tokens_per_second": 5})
    return plan, progress


def test_incomplete_pilot_cannot_freeze_main_budget():
    plan, progress = fixture()
    result = analyze(plan, progress[:-1])
    assert result["main_stage_budget"] is None
    assert result["client_coverage_complete"] is False
    assert result["missing_cohorts"] == ["9"]
    result = analyze(plan, progress)
    assert result["main_stage_budget"]["stage_trials"] == 20
    assert result["points"]["example"]["exact_itl_ms"]["independent_trials"] == 10


def test_token_count_does_not_create_independent_samples_or_weight_requests():
    _, progress = fixture()
    row = deepcopy(progress[0])
    row["requests"] *= 2
    row["requests"][1] = row["requests"][1] | {"itl_ms": [20]}
    assert trial_metrics(row)["exact_itl_ms"] == 15
    summary = summarize([10] * 20, main=True)
    assert summary["independent_trials"] == 20
    assert summary["mean_bootstrap_ci95"] == [10, 10]
    assert summary["relative_half_width"] == 0


def test_duplicate_trial_and_nonfinite_timing_fail_admission():
    plan, progress = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        analyze(plan, progress + progress[:1])
    progress[0]["requests"][0]["ttft_ms"] = float("nan")
    result = analyze(plan, progress)
    assert result["main_stage_budget"] is None
    assert result["points"]["example"]["ttft_ms"]["independent_trials"] == 9


def test_failed_setup_cannot_qualify_pilot():
    plan, progress = fixture()
    setup = {
        "cohort_id": "setup",
        "purpose": "prefix-warm-A",
        "trial_index": 0,
        "trial_seed": 0,
        "comparison_role": "setup",
        "requests": [{"request_id": "setup-request"}],
    }
    plan["cohorts"].append(setup)
    progress.append(deepcopy(setup) | {"valid": False})
    result = analyze(plan, progress)
    assert result["main_stage_budget"] is None
    assert result["invalid_client_cohorts"] == ["setup"]


def test_request_identity_and_independent_trial_identity_are_checked():
    plan, progress = fixture()
    wrong = deepcopy(progress)
    wrong[0]["requests"][0]["request_id"] = "different-request"
    with pytest.raises(ValueError, match="request identity"):
        analyze(plan, wrong)
    plan["cohorts"][1]["trial_seed"] = 0
    with pytest.raises(ValueError, match="duplicate independent trial"):
        analyze(plan, progress)
