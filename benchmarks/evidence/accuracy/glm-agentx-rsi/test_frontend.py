# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check paired frontend boundaries, token qualification and private-field projection."""

import io
import json

import blocking
import frontend
import pytest


def test_tool_only_and_reasoning_chunks_remain_content_boundaries():
    value = frontend.project(
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "private thought",
                        "content": "private prose",
                        "tool_calls": [{"function": {"name": "private tool", "arguments": "private args"}}],
                    }
                }
            ]
        }
    )
    assert value == {"content_chars": 13, "reasoning_chars": 15, "tool_chars": 24}
    assert "private" not in json.dumps(value)
    assert not any(frontend.project({"choices": []}).values())


@pytest.mark.parametrize("endpoint", ["native", "chat"])
def test_equal_inputs_and_exact_output_lengths_are_required(monkeypatch, endpoint):
    def respond(wire, **kwargs):
        body = json.loads(wire.data)
        assert body["rid"] == "rsi-test"
        if endpoint == "native":
            payload = {
                "text": "private output",
                "meta_info": {"prompt_tokens": 3, "completion_tokens": 16, "cached_tokens": 0},
            }
        else:
            payload = {
                "choices": [{"delta": {"tool_calls": [{"function": {"arguments": "private args"}}]}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 16, "prompt_tokens_details": {"cached_tokens": 0}},
            }
        return io.BytesIO(("data: " + json.dumps(payload) + "\n\n").encode())

    monkeypatch.setattr(frontend.urllib.request, "urlopen", respond)
    row = frontend.measured(
        "http://test", "rsi-test", [1, 2, 3], [{"role": "user", "content": "private input"}], endpoint
    )
    assert row["status"] == "completed" and row["cached_tokens"] == 0
    assert "private" not in json.dumps(row)
    assert frontend.measured("http://test", "rsi-test", [1, 2], [], endpoint)["status"] == "failed"


def test_cpu_actor_retains_identity_failure_without_token_array(monkeypatch):
    monkeypatch.setattr(
        blocking.urllib.request,
        "urlopen",
        lambda *_a, **_k: io.BytesIO(json.dumps({"tokens": [123, 456], "count": 2}).encode()),
    )
    row = blocking.tokenize("http://localhost", "case", b"{}", 3, "wrong")
    assert row["status"] == "failed" and row["error"] == "ValueError"
    assert row["input_tokens"] == 2
    assert "tokens" not in row and "body" not in row


@pytest.mark.parametrize("qualified", [False, True])
def test_zero_cache_omission_requires_explicit_runtime_qualification(qualified):
    usage = {"prompt_tokens": 651, "completion_tokens": 16, "prompt_tokens_details": None}
    assert frontend.cache_receipt(usage, qualified)[0] == (0 if qualified else None)
    assert frontend.cache_receipt({}, qualified)[0] is None
    assert frontend.cache_receipt({**usage, "prompt_tokens_details": {"cached_tokens": None}}, qualified)[0] is None
    assert frontend.cache_receipt({**usage, "prompt_tokens_details": {"cached_tokens": 64}}, qualified)[0] == 64
