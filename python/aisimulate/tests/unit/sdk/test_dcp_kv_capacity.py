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

    def get_kvcache_bytes_per_sequence(self, seq_len):
        return _PER_TOKEN * seq_len

    def get_kvcache_batch_capacity(self, budget, max_batch_size):
        return int(budget // _PER_TOKEN)

    def get_kvcache_rank_bytes_per_sequence(self, seq_len):
        return _PER_TOKEN * seq_len / self._dcp

    def get_kvcache_rank_batch_capacity(self, budget, max_batch_size):
        return int(budget * self._dcp // _PER_TOKEN)


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
        tp_size=8,
        moe_tp_size=8,
        moe_ep_size=1,
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


def test_model_without_rank_hooks_keeps_full_kv(monkeypatch):
    # Duck-typed model doubles (upstream memory tests) carry no rank hooks.
    class _Plain:
        def get_kvcache_bytes_per_sequence(self, seq_len):
            return _PER_TOKEN * seq_len

        def get_kvcache_batch_capacity(self, budget, max_batch_size):
            return int(budget // _PER_TOKEN)

    out = _estimate(monkeypatch, _Plain())
    assert out["kv_size_per_token_bytes"] == int(_PER_TOKEN)
    assert out["total_kv_size_tokens"] == out["total_kv_size_bytes"] // int(_PER_TOKEN)


def test_kimi_k3_dcp_reserves_rank_local_kda_state_before_striping():
    """Hybrid model: only the MLA layers' token-linear KV is striped by dcp; the
    per-request KDA state is rank-local and must be reserved whole. A budget
    smaller than one request's state admits nothing, however large dcp is."""
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models import get_model

    def _k3(dcp):
        cfg = ModelConfig(tp_size=8, attention_dp_size=1, moe_tp_size=8, moe_ep_size=1, dcp_size=dcp)
        return get_model("moonshotai/Kimi-K3", cfg, "sglang")

    plain, striped = _k3(1), _k3(4)
    state = striped._kda_state_bytes_per_request()
    per_token = plain.get_kvcache_bytes_per_sequence(1) - state
    assert state > 0 and per_token > 0

    # Per-rank bytes: state whole, token KV / 4.
    assert striped.get_kvcache_rank_bytes_per_sequence(0) == pytest.approx(state)
    assert striped.get_kvcache_rank_bytes_per_sequence(4096) == pytest.approx(state + 4096 * per_token / 4)

    # Inverse: half a request's state fits zero tokens (the old divisor
    # reported ~10k tokens here); state + N striped tokens fits exactly N.
    assert striped.get_kvcache_rank_batch_capacity(state / 2, max_batch_size=1) == 0
    assert striped.get_kvcache_rank_batch_capacity(state, max_batch_size=1) == 0
    assert striped.get_kvcache_rank_batch_capacity(state + 1000 * per_token / 4, max_batch_size=1) == 1000
    assert plain.get_kvcache_rank_batch_capacity(state + 1000 * per_token, max_batch_size=1) == 1000
