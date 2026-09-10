# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from compare_forward import compare_cases, error_summary


def sample():
    return {
        "case_id": "decode-0",
        "phase": "generation",
        "batch_size": 2,
        "query": 1,
        "prefix": 128,
        "canonical_past_kv": 128,
        "native_inclusive_kv": 129,
        "rank_max_ms": [9, 10, 11],
        "median_ms": 10,
    }


def test_distinct_forward_axes_and_signed_error():
    def predict(metrics):
        return metrics["scheduled_requests"]["sum_decode_kv_tokens"] / 20

    op = compare_cases([sample()], predict, forward_model="op_level")[0]
    fpm = compare_cases([sample()], predict, forward_model="fpm")[0]
    assert op["predicted_ms"] == 12.9
    assert fpm["predicted_ms"] == 12.8
    assert op["signed_error_percent"] == pytest.approx(29)


def test_missing_prediction_does_not_improve_accuracy_or_coverage():
    def absent(_):
        raise ValueError("missing exact prefix bucket")

    missing = compare_cases([sample()], absent, forward_model="op_level")
    assert error_summary(missing) == {"planned_points": 1, "predicted_points": 0}
    measured = compare_cases([sample()], lambda _: 8, forward_model="op_level")
    summary = error_summary(missing + measured)
    assert summary["planned_points"] == 2 and summary["predicted_points"] == 1
    assert summary["mean_signed_error_percent"] == pytest.approx(-20)
    assert summary["wape_percent"] == 20


def test_incomplete_or_changed_observations_are_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        compare_cases([sample() | {"rank_max_ms": [10]}], lambda _: 10, forward_model="op_level")
    with pytest.raises(ValueError, match="differs"):
        compare_cases([sample() | {"median_ms": 11}], lambda _: 10, forward_model="op_level")
