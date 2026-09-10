# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unequal-duration cases distinguish MAPE from WAPE on the same supported set."""

import pytest

from compare_e2e import paired_summary
from compare_forward import error_summary
from compare_fpm_holdout import statistics_for
from compare_trace import independent_trial_summary, trace_summary


def intervals():
    return [
        {"status": "predicted", "observed_ms": 10.0, "predicted_ms": 20.0, "signed_error_percent": 100.0},
        {"status": "predicted", "observed_ms": 100.0, "predicted_ms": 110.0, "signed_error_percent": 10.0},
        {"status": "prediction_unavailable", "observed_ms": 1000.0},
    ]


@pytest.mark.parametrize("summarize", [error_summary, statistics_for, trace_summary])
def test_forward_metrics_share_support_and_preserve_missing_coverage(summarize):
    result = summarize(intervals())
    assert result["planned_points"] == 3 and result["predicted_points"] == 2
    assert result["mape_percent"] == pytest.approx(55)
    assert result["wape_percent"] == pytest.approx(100 * 20 / 110)


def test_http_metrics_share_supported_trials():
    rows = [
        {
            "purpose": "short",
            "trial_index": i,
            "status": row["status"],
            "observed": {"ttft_ms": row["observed_ms"]},
            "prediction": {"ttft_ms": row["predicted_ms"]} if row["status"] == "predicted" else {},
        }
        for i, row in enumerate(intervals())
    ]
    result = paired_summary(rows, final=False)["short"]
    assert result["coverage"] == {"planned_trials": 3, "predicted_trials": 2}
    assert result["metrics"]["ttft_ms"]["mape_percent"] == pytest.approx(55)
    assert result["metrics"]["ttft_ms"]["wape_percent"] == pytest.approx(100 * 20 / 110)


def test_trial_interval_mape_is_not_error_after_summing_latency():
    rows = [{"purpose": "short", "trial_index": 0, "trial_seed": 1, "intervals": intervals()[:2]}]
    result = independent_trial_summary(rows, final=False)["short"]
    assert result["interval_mape_percent"] == pytest.approx(55)
    assert result["interval_wape_percent"] == pytest.approx(100 * 20 / 110)
