# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runtime probe preserves token-count transitions without retaining content."""

import json
from types import SimpleNamespace

import rsi_timing


def test_probe_records_numerical_transitions_and_ignores_other_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_TIMING_ROOT", str(tmp_path))
    monkeypatch.setattr(rsi_timing, "_last", {})
    monkeypatch.setattr(rsi_timing, "_stream", None)
    request = SimpleNamespace(rid="rsi-test", output_ids=[11, 12])
    unrelated = SimpleNamespace(rid="private-health-request", output_ids=[99])
    try:
        rsi_timing.record([request, unrelated])
        rsi_timing.record([request])
        request.output_ids += [13]
        rsi_timing.record([request])
        rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
        assert [row["output_tokens"] for row in rows] == [2, 3]
        assert all(set(row) == {"rid", "at_ns", "output_tokens", "batch_requests", "rank"} for row in rows)
        assert rows[0]["rank"] is None
        assert rows[0]["batch_requests"] == 2
    finally:
        if rsi_timing._stream:
            rsi_timing._stream.close()
