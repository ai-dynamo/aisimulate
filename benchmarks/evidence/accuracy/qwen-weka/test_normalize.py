# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic producer records cover omissions that aggregate metrics hide."""

import json

import pytest

from normalize import add_stream_observations, metric, reference_row, retain_cancelled_accounting


def metadata(identity="a", phase="profiling", **changes):
    return {
        "x_request_id": identity,
        "phase_kind": phase,
        "source_trace_id": "play",
        "source_outer_idx": 1,
        "request_start_ns": 2_000_000_000_000_000_000,
        "request_end_ns": 2_000_000_000_020_000_000,
        **changes,
    }


def test_token_units_are_integral_and_invalid_measurements_stay_unavailable():
    assert metric({"metrics": {"length": {"value": 2.0, "unit": "tokens"}}}, "length", "tokens") == 2
    assert metric({"metrics": {"length": {"value": 2.5, "unit": "tokens"}}}, "length", "tokens") is None
    with pytest.raises(ValueError, match="unit"):
        metric({"metrics": {"length": {"value": 2, "unit": "ms"}}}, "length", "tokens")


def test_raw_overflow_records_remain_and_stream_counts_conserve_tokens(tmp_path):
    namespaces = {"play": "weka:0123456789abcdef"}
    row = {"metadata": metadata(), "metrics": {"output_sequence_length": {"value": 2, "unit": "tokens"}}}
    refs = [reference_row(row, namespaces)]
    event = {
        "choices": [{"token_ids": [1, 2], "delta": None}],
        "usage": {"prompt_tokens_details": {"cached_tokens": 2}},
    }
    records = [
        {
            "metadata": metadata(),
            "start_perf_ns": 1000,
            "responses": [{"perf_ns": 2_001_000, "packets": [{"name": "data", "value": json.dumps(event)}]}],
        },
        {
            "metadata": metadata("overflow", context_overflow_skip=True),
            "start_perf_ns": 1000,
            "responses": [],
            "error": {"message": "overflow"},
        },
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(record) + "\n" for record in records))
    add_stream_observations(refs, raw, namespaces)
    assert len(refs) == 2 and refs[1]["error"] == "context_overflow"
    assert refs[0]["chunks"] == [{"at_ns": "2000000000002000000", "count": 2}]
    assert refs[0]["cached_tokens"] == 2


def accounting_log(completed=1, cancelled=1):
    return (
        "Phase warmup (warmup) complete | completed=1, cancelled=0, errors=0\n"
        f"Phase profiling (profiling) complete | completed={completed}, cancelled={cancelled}, errors=0\n"
        "Received CREDIT_PHASE_COMPLETE message, Phase complete: PhaseRecordsStats(phase_kind='profiling', "
        "start_ns=1000000000, sent_end_ns=601000000000, requests_end_ns=631000000000, "
        f"final_requests_completed={completed}, final_requests_cancelled={cancelled}, final_request_errors=0)\n"
    )


def test_unrecorded_cancellations_are_counted_without_invented_timings():
    references = [{"phase": "preparation"}, {"phase": "profile"}]
    ranges = {
        "warmup": {"start_ns": 0, "end_ns": 1_000_000_000},
        "profiling": {"start_ns": 1_000_000_000, "end_ns": 631_000_000_000},
    }
    result = retain_cancelled_accounting(references, accounting_log(), ranges)
    assert result["profile"]["cancelled_without_request_record"] == 1
    assert len(references) == 3 and references[-1]["error"] == "measurement_cancelled"
    assert references[-1]["start_ns"] is references[-1]["end_ns"] is None
    assert references[-1]["identity_valid"] is False
    with pytest.raises(ValueError, match="reconcile"):
        retain_cancelled_accounting([{"phase": "preparation"}, {"phase": "profile"}], accounting_log(2, 0), ranges)
