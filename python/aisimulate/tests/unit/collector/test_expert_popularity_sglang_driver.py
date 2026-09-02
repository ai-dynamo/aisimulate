# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from collector.expert_popularity import sglang_driver

pytestmark = pytest.mark.unit


def test_recorder_generate_uses_configured_request_timeout(monkeypatch, tmp_path):
    calls = []

    def fake_post(base_url, endpoint, payload=None, timeout=1800):
        calls.append((base_url, endpoint, timeout))
        if endpoint == "/generate":
            raise RuntimeError("stop after observing generate timeout")
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(sglang_driver, "_post", fake_post)
    args = SimpleNamespace(base_url="http://server", request_timeout=37)

    with pytest.raises(RuntimeError, match="stop after observing"):
        sglang_driver._run_recorder_window(
            args=args,
            prepared=[{"input_ids": [1], "isl": 1, "request_index": 0}],
            raw_dir=tmp_path,
            responses_path=tmp_path / "responses.jsonl",
            repeat_index=0,
            shard_index=0,
            moe_layer_ids=[0],
        )

    assert calls == [
        ("http://server", "/start_expert_distribution_record", 1800),
        ("http://server", "/generate", 37),
    ]
