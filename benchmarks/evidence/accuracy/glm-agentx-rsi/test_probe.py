# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise numerical projection of native/OpenAI streams without a server."""

import io
import json
import threading
from types import SimpleNamespace

import probe


def test_native_stream_preserves_counts_and_drops_text(monkeypatch):
    wire = b'data: {"text":"private text","meta_info":{"completion_tokens":1}}\n\n'
    wire += b'data: {"text":"private text","meta_info":{"completion_tokens":4}}\n\n'
    wire += b'data: [DONE]\n\n'
    monkeypatch.setattr(probe, "call", lambda *_: io.BytesIO(wire))
    first = threading.Event()
    result = probe.request("http://test", "rsi-test", [3], 4, threading.Barrier(1), first)
    assert result["status"] == "completed" and first.is_set()
    assert [event["output_tokens"] for event in result["events"]] == [1, 4]
    assert "private text" not in json.dumps(result)
    assert probe.request("http://test", "rsi-test", [3], 5, threading.Barrier(1))["status"] == "incomplete"


def test_chat_projection_retains_usage_and_reasoning_boundary_without_content(monkeypatch):
    def response(url, body):
        if url.endswith("flush_cache"):
            return io.BytesIO(b"true")
        payload = {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]}
        usage = {"choices": [], "usage": {"prompt_tokens": 8192, "completion_tokens": body["max_tokens"]}}
        return io.BytesIO(("data: " + json.dumps(payload) + "\n\ndata: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode())

    monkeypatch.setattr(probe, "call", response)
    output = io.StringIO()
    probe.chat_controls("http://test", [1, 2], SimpleNamespace(decode=lambda _: "private prompt"), output)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 32 and all(row["status"] == "completed" for row in rows)
    assert all(row["events"][0]["reasoning_content_chars"] == len("private reasoning") for row in rows)
    assert "private" not in output.getvalue()
