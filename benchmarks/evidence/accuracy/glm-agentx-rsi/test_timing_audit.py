# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distinguish delivered counts, retained over-generation and completion overhead."""

import json

import pytest
from analyze_timing import (
    analyze_blocking,
    analyze_forward,
    analyze_frontend,
    blocking_comparison,
    forward_request,
    planned_ids,
    request_metrics,
    span,
)


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


def test_chat_mean_uses_total_length_without_claiming_per_token_alignment():
    row = {
        "rid": "rsi-chat",
        "role": "diagnostic_chat_frontend",
        "status": "completed",
        "start_ns": "0",
        "end_ns": "13000000",
        "output_tokens": 5,
        "requested_output_tokens": 5,
        "events": [{"at_ns": "2000000", "reasoning_content_chars": 2}, {"at_ns": "10000000", "content_chars": 4}],
    }
    result = request_metrics(row, [])
    assert result["visible_mean_interval_ms"] == 8 / 4
    assert "tool-only" in result["interval_scope"]
    assert "per-frame" in result["per_token_alignment_unavailable"]


def test_failed_request_is_retained_and_full_manifest_has_221_identities():
    result = request_metrics({"rid": "rsi-failed", "status": "failed", "requested_output_tokens": 16, "events": []}, [])
    assert result["status"] == "failed" and result["rid"] == "rsi-failed"
    assert "unavailable" in result
    assert len(planned_ids()) == 221


def test_frontend_pair_requires_cache_input_hash_and_preparation_receipts(tmp_path):
    directory = tmp_path / "instrumented-mtp"
    (directory / "client").mkdir(parents=True)
    runtime = directory / "resolved-server.json"
    runtime.write_text(json.dumps({"speculative_algorithm": "EAGLE", "page_size": 64, "disable_radix_cache": False}))
    rows = []
    for endpoint in ("native", "chat"):
        rid = "rsi-frontend-c8192-r0-" + endpoint
        rows += [
            {"rid": rid + "-warm", "status": "completed", "events": [{"meta": {"prompt_tokens": 128}}]},
            {
                "rid": rid,
                "status": "completed",
                "endpoint": endpoint,
                "prefix_tokens": 128,
                "cached_tokens": 64,
                "input_tokens": 160,
                "expected_input_tokens": 160,
                "input_sha256": "same",
                "start_ns": "0",
                "end_ns": "10000000",
                "output_tokens": 16,
                "requested_output_tokens": 16,
                "events": [event(4, 1), event(9, 16)],
            },
        ]
    source = directory / "client/requests.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stages = directory / "frontend-123.jsonl"
    stage = {
        "start_ns": "1000000",
        "end_ns": "3000000",
        "service_ms": 2,
        "input_tokens": 160,
        "input_sha256": "same",
        "template_cache_requested": False,
    }
    stages.write_text(json.dumps(stage) + "\n")
    result = analyze_frontend(tmp_path)["modes"]["instrumented-mtp"]
    assert result["qualified_pairs"] == 1 and len(result["unrecorded_or_unattempted"]) == 32
    rows[-1]["cached_tokens"] = None
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert analyze_frontend(tmp_path)["modes"]["instrumented-mtp"]["qualified_pairs"] == 0
    rows[-1]["cached_tokens"] = 128
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert analyze_frontend(tmp_path)["modes"]["instrumented-mtp"]["qualified_pairs"] == 0
    rows[-1]["cached_tokens"] = 64
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stage["input_sha256"] = "different"
    stages.write_text(json.dumps(stage) + "\n")
    assert analyze_frontend(tmp_path)["modes"]["instrumented-mtp"]["qualified_pairs"] == 0
    stage["input_sha256"] = "same"
    stages.write_text(json.dumps(stage) + "\n")
    runtime.unlink()
    assert analyze_frontend(tmp_path)["modes"]["instrumented-mtp"]["qualified_pairs"] == 0


@pytest.mark.parametrize("endpoint", ["tokenize", "chat"])
def test_blocking_audit_requires_cpu_exposure_and_consistent_clocks(endpoint):
    victim = {
        "output_tokens": 4,
        "dispatch": {"pid": 23},
        "events": [{"at_ns": "1200000000", "output_tokens": 1}, {"at_ns": "1201000000", "output_tokens": 4}],
    }
    server = [{"at_ns": "1000000000", "output_tokens": 1}, {"at_ns": "1030000000", "output_tokens": 4}]
    actor = {"start_ns": "800000000", "body_received_ns": "1300000000", "input_sha256": "matched"}
    if endpoint == "chat":
        actor = {"start_ns": "800000000", "end_ns": "1300000000", "input_sha256": None,
                 "expected_input_sha256": "matched"}
    stage = {"start_ns": "900000000", "end_ns": "1100000000", "input_sha256": "matched", "pid": 23}
    result = blocking_comparison(victim, actor, server, [stage])
    assert result["qualified"] and result["fully_buffered_during_cpu_stage"]
    assert result["server"]["mean_interval_ms"] == 10
    assert result["client"]["mean_interval_ms"] == pytest.approx(1 / 3)
    before = dict(stage, start_ns="850000000", end_ns="950000000")
    result = blocking_comparison(victim, actor, server, [before])
    assert result["qualified"] and result["cpu_finished_before_first_ready"]
    assert not result["fully_buffered_during_cpu_stage"]
    assert not blocking_comparison(victim, actor, server, [])["qualified"]
    assert not blocking_comparison(victim, actor, server, [stage, stage])["qualified"]
    assert not blocking_comparison(victim, actor, server, [dict(stage, pid=24)])["qualified"]
    late = [{"at_ns": "1400000000", "output_tokens": 4}]
    assert not blocking_comparison(victim, actor, late, [stage])["qualified"]


def test_blocking_startup_failure_retains_full_missing_accounting(tmp_path):
    value = analyze_blocking(tmp_path)
    assert not value["collector_completed"]
    for mode in value["modes"].values():
        assert mode["planned"] == 32 and mode["recorded"] == 0
        assert len(mode["unrecorded_or_unattempted"]) == 32
        assert mode["qualified_victims"] == 0
        assert not mode["collector_completed"]


def test_coding_control_manifest_preserves_all_unattempted_requests(tmp_path):
    for mode in ("unprofiled-mtp", "instrumented-mtp"):
        (tmp_path / mode).mkdir()
    result = analyze_blocking(tmp_path, coding=True)
    assert result["schema"] == "aisimulate.glm-rsi.coding-blocking-audit.v1"
    assert all(mode["planned"] == 17 and mode["recorded"] == 0
               and len(mode["unrecorded_or_unattempted"]) == 17 for mode in result["modes"].values())


def test_forward_audit_preserves_zero_interval_and_rejects_unknown_cache():
    row = {"rid": "rsi-profile-c393216-profiled", "status": "completed", "role": "profile",
           "context": 393216, "arm": "profiled", "input_tokens": 393216,
           "requested_output_tokens": 128, "start_ns": "0", "end_ns": "3000000",
           "events": [{"at_ns": "2000000", "output_tokens": 128,
                       "meta": {"prompt_tokens": 393216, "cached_tokens": 393152}}]}
    result = forward_request(row)
    assert result["qualified"] and result["client"]["mean_interval_ms"] == 0
    row["events"][0]["meta"].pop("cached_tokens")
    assert not forward_request(row)["qualified"]
    row["events"][0]["meta"]["cached_tokens"] = 393151
    assert not forward_request(row)["qualified"]


def test_forward_audit_empty_campaign_retains_both_missing_matrices(tmp_path):
    result = analyze_forward(tmp_path)
    assert result["modes"]["unprofiled-mtp"]["planned"] == 7
    assert result["modes"]["instrumented-mtp"]["planned"] == 10
    for mode in result["modes"].values():
        assert not mode["qualified"] and len(mode["missing"]) == mode["planned"]
