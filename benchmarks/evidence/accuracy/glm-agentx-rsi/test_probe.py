# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise numerical projection of native/OpenAI streams without a server."""

import io
import json
import threading
import urllib.error
from types import SimpleNamespace

import probe
import pytest


def test_native_stream_preserves_counts_and_drops_text(monkeypatch):
    wire = b'data: {"text":"private text","meta_info":{"completion_tokens":1}}\n\n'
    wire += b'data: {"text":"private text","meta_info":{"completion_tokens":4}}\n\n'
    wire += b"data: [DONE]\n\n"
    monkeypatch.setattr(probe, "call", lambda *_: io.BytesIO(wire))
    first = threading.Event()
    result = probe.request("http://test", "rsi-test", [3], 4, threading.Barrier(1), first)
    assert result["status"] == "completed" and first.is_set()
    assert [event["output_tokens"] for event in result["events"]] == [1, 4]
    assert "private text" not in json.dumps(result)
    assert probe.request("http://test", "rsi-test", [3], 5, threading.Barrier(1))["status"] == "incomplete"


def test_prepared_native_body_does_not_serialize_inside_measurement(monkeypatch):
    body = b'{"rid":"rsi-prepared","input_ids":[3],"stream":true}'
    wire = b'data: {"meta_info":{"completion_tokens":1}}\n\n'
    wire += b'data: {"meta_info":{"completion_tokens":4}}\n\n'
    captured = []

    def response(request, timeout):
        captured.append((request.full_url, request.data, timeout))
        return io.BytesIO(wire)

    monkeypatch.setattr(probe.urllib.request, "urlopen", response)
    monkeypatch.setattr(probe.json, "dumps", lambda *_a, **_k: pytest.fail("body was serialized during measurement"))
    result = probe.request("http://test", "rsi-prepared", [3], 4, threading.Barrier(1), encoded_body=body)
    assert result["status"] == "completed"
    assert captured == [("http://test/generate", body, 600)]
    assert [row["output_tokens"] for row in result["events"]] == [1, 4]


def test_chat_projection_retains_usage_and_reasoning_boundary_without_content(monkeypatch):
    def response(url, body):
        if "/flush_cache?timeout=60" in url:
            return io.BytesIO(b"true")
        payload = {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]}
        usage = {"choices": [], "usage": {"prompt_tokens": 8192, "completion_tokens": body["max_tokens"]}}
        return io.BytesIO(
            ("data: " + json.dumps(payload) + "\n\ndata: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode()
        )

    monkeypatch.setattr(probe, "call", response)
    output = io.StringIO()
    probe.chat_controls("http://test", [1, 2], SimpleNamespace(decode=lambda _: "private prompt"), output)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 32 and all(row["status"] == "completed" for row in rows)
    assert all(row["events"][0]["reasoning_content_chars"] == len("private reasoning") for row in rows)
    assert "private" not in output.getvalue()


def test_failed_chat_case_retains_every_concurrent_attempt(monkeypatch):
    def response(url, body):
        if "/flush_cache?timeout=60" in url:
            return io.BytesIO(b"true")
        if "-l14-" in body["rid"]:
            raise urllib.error.URLError("private failure body")
        return io.BytesIO(b'data: {"usage":{"prompt_tokens":8192,"completion_tokens":16},"choices":[]}\n\n')

    monkeypatch.setattr(probe, "call", response)
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="all attempted"):
        probe.chat_controls("http://test", [1, 2], SimpleNamespace(decode=lambda _: "private prompt"), output)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 15
    assert sum(row["status"] == "failed" for row in rows) == 14
    assert "private" not in output.getvalue()
