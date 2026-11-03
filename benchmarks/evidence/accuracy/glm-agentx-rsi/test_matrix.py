# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from audit_matrix import errors, interval
from predict_matrix import client_ttft_ms, qualify_ready_event


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
