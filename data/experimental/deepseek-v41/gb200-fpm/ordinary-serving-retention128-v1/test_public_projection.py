# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused privacy, coverage and interval-label guards; no model execution."""

import unittest

from export_public import (
    coordinate_region,
    paired_metrics,
    privacy_check,
    public_cohort,
    safe_hashes,
)
from render_report import interval_rows


class ProjectionGuards(unittest.TestCase):
    def test_frozen_uppercase_prefix_purpose_is_preserved(self):
        self.assertEqual(
            public_cohort({"purpose": "prefix-reuse-B", "trial_index": 29}),
            "trial-029/prefix-reuse-B",
        )

    def test_private_values_rejected(self):
        for value in ("/lustre/private/data", "12345678-1234-5678-1234-123456789abc"):
            with self.assertRaises(ValueError):
                privacy_check({"nested": [value]})
        with self.assertRaises(ValueError):
            safe_hashes({"private/path": "a" * 64})
        privacy_check({"cohort_id": "trial-000/short", "sha256": "a" * 64})

    def test_missing_prediction_stays_in_denominator(self):
        row = paired_metrics([1, 2, 3], [(1, 2), (9, 9)], metric="ttft_ms", purpose=None, unit="trial")
        self.assertEqual((row["observed_points"], row["predicted_points"]), (3, 2))
        self.assertEqual((row["mape_percent"], row["wape_percent"]), (50, 10))

    def test_coordinate_equality_is_separate_from_sparse_box_and_variance(self):
        points = {("prefill", 1, 32, 0), ("prefill", 1, 128, 0)}
        row = {
            "scheduled_requests": {
                "num_prefill_requests": 1,
                "num_decode_requests": 0,
                "sum_prefill_tokens": 64,
                "sum_prefill_kv_tokens": 0,
            },
            "native_scheduled_requests": {
                "var_prefill_length": 4,
                "var_decode_kv_tokens": 0,
            },
        }
        region = coordinate_region(row, points)
        self.assertEqual(region["region"], "unseen_inside_raw_phase_bounding_box")
        self.assertTrue(region["nonzero_native_length_variance"])
        row["scheduled_requests"]["sum_prefill_tokens"] = 32
        self.assertEqual(
            coordinate_region(row, points)["region"],
            "exact_aggregate_calibration_coordinate",
        )
        row["scheduled_requests"]["num_prefill_requests"] = 3
        self.assertEqual(coordinate_region(row, points)["outside_axes"], ["batch_size"])
        row["scheduled_requests"]["num_decode_requests"] = 1
        self.assertEqual(
            coordinate_region(row, points)["region"],
            "mixed_no_single_measured_coordinate",
        )

    def test_e2e_ratio_interval_is_not_mape_interval(self):
        point = {
            "independent_trials": 30,
            "bootstrap_seed": 94051000,
            "bootstrap_resamples": 5000,
            "observed_mean_bootstrap_ci95": [1, 2],
            "paired_ratio_error_percent_bootstrap_ci95": [3, 4],
        }
        rows = interval_rows(
            {"points": {"short": {"metrics": {"ttft_ms": point}}}},
            {"independent_trial_summary": {}},
            {"observation_statistics": {"points": {}}},
        )
        self.assertEqual(
            [r["estimand"] for r in rows],
            ["observed_mean", "paired_ratio_error_percent"],
        )
        self.assertEqual([r["low"] for r in rows], [1, 3])


if __name__ == "__main__":
    unittest.main()
