# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode CP stripes the persistent KV: per-rank bytes/token drop by dcp AND the
rank-local budget therefore holds dcp times as many tokens. Both halves must
move together; the token inverse used to see the full per-token size and left
the block count unchanged."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aisimulate_core.sdk import memory

pytestmark = pytest.mark.unit

_GIB = 1024**3
_PER_TOKEN = 100.0


class _StubModel:
    def __init__(self, dcp: int):
        self._dcp = dcp

    def _cp_kv_memory_divisor(self):
        return self._dcp

    def get_kvcache_bytes_per_sequence(self, seq_len):
        return _PER_TOKEN * seq_len

    def get_kvcache_batch_capacity(self, budget, max_batch_size):
        return int(budget // _PER_TOKEN)


def _estimate(monkeypatch, model) -> dict:
    class _StubBackend:
        def _get_memory_usage(self, *a, **k):
            return {"weights": 10.0, "activations": 1.0, "others": 1.0, "nccl": 1.0, "kvcache": 0.0}

    db = SimpleNamespace(version="0.14.1", system_spec={"gpu": {"mem_capacity": 100 * _GIB}})
    monkeypatch.setattr(memory, "get_model", lambda *a, **k: model)
    monkeypatch.setattr(memory, "get_backend", lambda backend: _StubBackend())
    monkeypatch.setattr(memory.perf_database, "get_database", lambda *a, **k: db)
    estimator = memory.KVCacheEstimator.from_request(
        "deepseek-ai/DeepSeek-V3",
        "h200_sxm",
        "vllm",
        max_num_tokens=8192,
        max_batch_size=256,
        dcp_size=getattr(model, "_dcp", 1),
    )
    return estimator.estimate(is_of_free=False, fraction=0.9, gpu_memory_capacity_bytes_override=None)


def test_dcp_stripes_per_token_bytes_and_multiplies_token_capacity(monkeypatch):
    base = _estimate(monkeypatch, _StubModel(dcp=1))
    striped = _estimate(monkeypatch, _StubModel(dcp=4))

    assert base["total_kv_size_bytes"] == striped["total_kv_size_bytes"]
    assert base["kv_size_per_token_bytes"] == int(_PER_TOKEN)
    assert striped["kv_size_per_token_bytes"] == int(_PER_TOKEN / 4)
    # A rank-local budget of B bytes holds tokens whose full KV is 4B bytes
    # (both sides floor to whole tokens, so allow the rounding slack).
    assert abs(striped["total_kv_size_tokens"] - 4 * base["total_kv_size_tokens"]) <= 4
    # Consistency: bytes / (bytes per token) == tokens on both sides.
    assert striped["total_kv_size_tokens"] == striped["total_kv_size_bytes"] // striped["kv_size_per_token_bytes"]


def test_model_without_divisor_hook_keeps_full_kv(monkeypatch):
    # Duck-typed model doubles (upstream memory tests) carry no CP hook.
    class _Plain:
        def get_kvcache_bytes_per_sequence(self, seq_len):
            return _PER_TOKEN * seq_len

        def get_kvcache_batch_capacity(self, budget, max_batch_size):
            return int(budget // _PER_TOKEN)

    out = _estimate(monkeypatch, _Plain())
    assert out["kv_size_per_token_bytes"] == int(_PER_TOKEN)
    assert out["total_kv_size_tokens"] == out["total_kv_size_bytes"] // int(_PER_TOKEN)
