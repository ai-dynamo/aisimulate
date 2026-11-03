# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from audit_coding import bounds as coding_bounds
from audit_coding import validate_counts
from audit_matrix import errors, interval
from audit_shared_frontend import error as frontend_error
from audit_shared_frontend import latencies as frontend_latencies
from predict_matrix import client_ttft_ms, qualify_ready_event, replay_origin_ns
from response_gate_diagnostic import gate_time, summary


def test_submission_replay_preserves_original_client_ttft_boundary():
    origin = 4_000_000_000_000_000
    reference = {"start_ns": str(origin + 3_000_000)}
    client_arrival = {"first_token_ms": 100.0, "arrival_time_ms": 3.0, "ttft_ms": 97.0}
    submitted_arrival = {"first_token_ms": 100.0, "arrival_time_ms": 10.0, "ttft_ms": 90.0}
    assert client_ttft_ms(client_arrival, reference, origin) == 97.0
    assert client_ttft_ms(submitted_arrival, reference, origin) == 97.0
    assert submitted_arrival["ttft_ms"] != client_ttft_ms(submitted_arrival, reference, origin)


def test_recorded_native_tail_requires_explicit_conservation():
    event = {"schema": "aisimulate.glm-rsi.output-ready.v2", "generated_output_tokens": 130,
             "output_tokens": 128, "discarded_tail_tokens": 2, "finished_len": 128}
    qualify_ready_event(event)
    with pytest.raises(ValueError, match="conservation"):
        qualify_ready_event({**event, "discarded_tail_tokens": 0})
    with pytest.raises(ValueError, match="conservation"):
        qualify_ready_event({**event, "finished_len": None})
    with pytest.raises(ValueError, match="explicit"):
        qualify_ready_event({"output_tokens": 130})


def test_independent_audit_pools_requests_and_preserves_stream_denominator():
    # Unequal reference magnitudes must not turn into a ratio of means.
    result = errors([(100, 50), (100, 1000)])
    assert result["n"] == 2 and result["mape"] == 95
    assert result["mean_absolute_ms"] == 475
    assert interval([
        {"at_ns": "1000000000", "output_tokens": 3},
        {"at_ns": "1127000000", "output_tokens": 128},
    ]) == (1000000000, 1.0)


@pytest.mark.parametrize("pairs", [[], [(1, 0)], [(float("nan"), 1)], [(1, float("inf"))]])
def test_independent_audit_rejects_undefined_control_scores(pairs):
    with pytest.raises(ValueError):
        errors(pairs)


def test_submission_zero_based_origin_restores_client_transport_delay():
    client_origin = 4_000_000_000_000_000
    reference = {"start_ns": str(client_origin + 3_000_000)}
    measured = [
        {"start_ns": str(client_origin), "dispatch": {"at_ns": str(client_origin + 20_000_000)}},
        {"start_ns": str(client_origin + 3_000_000), "dispatch": {"at_ns": str(client_origin + 10_000_000)}},
    ]
    assert replay_origin_ns(measured, "client_start") == client_origin
    submission_origin = replay_origin_ns(measured, "server_submission")
    assert submission_origin == client_origin + 10_000_000
    assert client_ttft_ms({"first_token_ms": 90.0}, reference, submission_origin) == 97.0
    assert client_ttft_ms({"first_token_ms": 100.0}, reference, client_origin) == 97.0


def test_independent_coding_audit_preserves_tail_counts_and_units():
    value = {"schema": "aisimulate.glm-rsi.output-ready.v2", "generated_output_tokens": 18,
             "output_tokens": 16, "discarded_tail_tokens": 2, "finished_len": 16}
    validate_counts(value)
    with pytest.raises(ValueError, match="conservation"):
        validate_counts({**value, "discarded_tail_tokens": 0})
    assert coding_bounds([
        {"at_ns": "1000000000", "output_tokens": 3},
        {"at_ns": "1030000000", "output_tokens": 16},
    ]) == (1000000000, 1030000000, 2.0)


def test_delivery_gate_uses_only_work_known_when_output_becomes_ready():
    arrivals, free_at = [1.0, 5.0, 15.0], [11.0, 21.0, 31.0]
    assert gate_time(0.0, arrivals, free_at) == 0.0
    assert gate_time(3.0, arrivals, free_at) == 11.0
    assert gate_time(5.0, arrivals, free_at) == 21.0  # Explicit input-first tie.
    assert gate_time(7.0, arrivals, free_at) == 21.0
    assert gate_time(40.0, arrivals, free_at) == 40.0
    assert summary([(0.0, 2.0)])["mape"] == 100.0  # Defined zero prediction stays scored.


def test_native_shared_frontend_audit_keeps_zero_delivery_intervals():
    row = {"arrival_time_ms": 1, "first_token_ms": 11, "last_token_ms": 11,
           "terminal_time_ms": 11, "ttft_ms": 10, "itl_ms": 0,
           "output_length": 128, "requested_output_length": 128}
    assert frontend_latencies(row) == {"ttft": 10, "tpot": 0}
    assert frontend_error(0, 0.01) == 100
    assert frontend_error(0, 0) is None
    with pytest.raises(ValueError, match="causal"):
        frontend_latencies({**row, "terminal_time_ms": 10})
    with pytest.raises(ValueError, match="conserved"):
        frontend_latencies({**row, "requested_output_length": 129})
