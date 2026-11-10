# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metric controls use authored scalar fixtures, never private captured prompts."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/prediction_regression/compare_mamba_cache_accuracy.py"
)
_SPEC = importlib.util.spec_from_file_location("cache_accuracy", _SCRIPT)
assert _SPEC and _SPEC.loader
metric = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metric)


def test_cache_score_uses_request_count_and_uncached_work_denominators():
    rows = [
        {
            "prompt_tokens": 100,
            "native": {"cached_tokens": 0},
            "before": {"cached_tokens": 20, "conditional_first_step_ape_pct": 10},
            "after": {"cached_tokens": 0, "conditional_first_step_ape_pct": 2},
        },
        {
            "prompt_tokens": 100,
            "native": {"cached_tokens": 80},
            "before": {"cached_tokens": 60, "conditional_first_step_ape_pct": 30},
            "after": {"cached_tokens": 80, "conditional_first_step_ape_pct": 4},
        },
    ]
    before = metric.aggregate(rows, "before")
    after = metric.aggregate(rows, "after")
    assert before["cache_exact_matches"] == 0
    assert before["cache_token_mae"] == 20
    assert before["uncached_prefill_token_wape_pct"] == pytest.approx(100 * 40 / 120)
    assert before["conditional_first_prefill_step_mape_pct"] == 20
    assert after["cache_exact_match_pct"] == 100
    assert after["cache_token_mae"] == 0


def test_client_timing_scores_every_profiling_request_and_rejects_missing_values():
    rows = [
        {
            "native_client": {"ttft_ms": 10, "tpot_visible_span_ms": 2},
            "before": {"replay_ttft_ms": 20, "replay_tpot_ms": 3},
        },
        {
            "native_client": {"ttft_ms": 100, "tpot_visible_span_ms": 4},
            "before": {"replay_ttft_ms": 110, "replay_tpot_ms": 4},
        },
        {"native": {"phase": "warmup"}},
    ]
    score = metric.client_timing(rows, "before")
    assert score["profiling_requests"] == 2
    assert score["ttft_mape_pct"] == 55
    assert score["tpot_mape_pct"] == 25
    rows[1]["before"]["replay_tpot_ms"] = None
    with pytest.raises(ValueError, match="do not silently change cohort"):
        metric.client_timing(rows, "before")


def test_reference_keeps_first_prefill_and_rejects_mixed_batch(tmp_path):
    def row(past, roles=None):
        return {
            "wall_time": 0.005,
            "scheduled_requests": {"past_kv_lengths": [past], "extend_lengths": [64]},
            "canonical_witness": {
                "request_roles": roles or ["prefill"],
                "request_labels": [
                    {
                        "x_request_id": "r",
                        "benchmark_phase": "profiling",
                        "source_trace_id": "s",
                    }
                ],
            },
        }

    path = tmp_path / "canonical.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in [row(128), row(192)]))
    reference = metric.first_native_prefills(path)
    assert len(reference) == 1
    assert reference["r"]["cached_tokens"] == 128
    assert reference["r"]["duration_ms"] == 5
    path.write_text(json.dumps(row(128, ["prefill", "decode"])))
    with pytest.raises(ValueError, match="singleton prefill"):
        metric.first_native_prefills(path)


def test_frozen_forest_uses_named_feature_order_and_log_target():
    forest = {
        "features": ["req_sum_past", "req_sum_extend"],
        "stores": {
            "pure_prefill": {
                "baseline": math.log(2),
                "trees": [
                    {
                        "left": [1, -1, -1],
                        "right": [2, -1, -1],
                        "feature": [0, -1, -1],
                        "threshold": [8, 0, 0],
                        "value": [0, math.log(3), math.log(5)],
                    }
                ],
            }
        },
    }
    assert metric.forest_prediction(forest, extend=100, past=8) == pytest.approx(6)
    assert metric.forest_prediction(forest, extend=1, past=9) == pytest.approx(10)


def test_repeat_control_excludes_only_host_cost_and_preserves_simulated_time():
    report = {
        "wall_time_ms": 123,
        "processed_tokens_per_s": 1,
        "processed_output_tokens_per_s": 2,
        "per_request": [{"ttft_ms": 12.3}],
        "duration_ms": 100,
    }
    canonical = metric.canonical_report(report)
    assert canonical == {"per_request": [{"ttft_ms": 12.3}], "duration_ms": 100}
    assert report["wall_time_ms"] == 123
