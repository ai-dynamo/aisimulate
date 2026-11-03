# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distinguish delivered counts, retained over-generation and completion overhead."""

from analyze_timing import planned_ids, request_metrics, span


def event(ms, tokens, rank=0):
    return {"at_ns": str(ms * 1000000), "output_tokens": tokens, "rank": rank}


def test_delivery_boundary_does_not_absorb_duplicate_terminal_frame():
    row = {
        "rid": "rsi-example",
        "status": "completed",
        "start_ns": "0",
        "end_ns": "16000000",
        "requested_output_tokens": 16,
        "events": [event(4, 4), event(11, 16), event(15, 16)],
    }
    result = request_metrics(row, [event(3, 4), event(9, 17)])
    assert result["ttft_ms"] == 4
    assert result["client"]["span_ms"] == 7
    assert result["server_ready"]["retained_excess_over_delivered"] == 1
    assert result["first_receipt_after_ready_ms"] == 1
    assert result["final_receipt_after_ready_ms"] == 2
    assert result["client"]["mean_interval_ms"] == 7 / 15
    assert "unavailable" in request_metrics(row, [event(3, 4, None), event(9, 17, None)])["server_ready"]


def test_one_token_undefined_interval_and_invalid_counter_accounting():
    assert span([event(2, 3)], 1)["mean_interval_ms"] is None
    assert span([event(2, 1)], 0)["unavailable"] == "invalid delivered output length"
    assert "unavailable" in span([event(2, 4), event(3, 2)], 4)
    assert "unavailable" in span([event(3, 2), event(2, 4)], 4)
    assert "unavailable" in span([event(2, 2)], 4)


def test_failed_request_is_retained_and_full_manifest_has_221_identities():
    result = request_metrics({"rid": "rsi-failed", "status": "failed", "requested_output_tokens": 16, "events": []}, [])
    assert result["status"] == "failed" and result["rid"] == "rsi-failed"
    assert "unavailable" in result
    assert len(planned_ids()) == 221
