# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode CP stripes the persistent KV: per-rank bytes/token drop by dcp AND the
rank-local budget therefore holds dcp times as many tokens. Both halves must
move together; the token inverse used to see the full per-token size and left
the block count unchanged."""

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk import memory

pytestmark = pytest.mark.unit

_GIB = 1024**3
_PER_TOKEN = 100.0


def _install_stubs(monkeypatch, *, dcp: int):
    class _StubModel:
        def _cp_kv_memory_divisor(self):
            return dcp

        def get_kvcache_bytes_per_sequence(self, seq_len):
            return _PER_TOKEN * seq_len

        def get_kvcache_max_tokens(self, budget):
            return int(budget // _PER_TOKEN)

    class _StubBackend:
        def _get_memory_usage(self, *a, **k):
            return {"weights": 10.0, "activations": 1.0, "others": 1.0, "nccl": 1.0, "kvcache": 0.0}

    class _StubDB:
        def __init__(self):
            self.system_spec = {"gpu": {"mem_capacity": 100 * _GIB}}

    monkeypatch.setattr(memory, "get_model", lambda *a, **k: _StubModel())
    monkeypatch.setattr(memory, "get_backend", lambda backend: _StubBackend())
    monkeypatch.setattr(memory.perf_database, "get_database", lambda *a, **k: _StubDB())


def _estimate(monkeypatch, *, dcp: int) -> dict:
    _install_stubs(monkeypatch, dcp=dcp)
    estimator = memory.KVCacheEstimator.from_request(
        "deepseek-ai/DeepSeek-V3",
        "h200_sxm",
        "vllm",
        max_num_tokens=8192,
        max_batch_size=256,
        dcp_size=dcp,
    )
    return estimator.estimate(is_of_free=False, fraction=0.9, gpu_memory_capacity_bytes_override=None)


def test_dcp_stripes_per_token_bytes_and_multiplies_token_capacity(monkeypatch):
    base = _estimate(monkeypatch, dcp=1)
    striped = _estimate(monkeypatch, dcp=4)

    assert base["total_kv_size_bytes"] == striped["total_kv_size_bytes"]
    assert base["kv_size_per_token_bytes"] == int(_PER_TOKEN)
    assert striped["kv_size_per_token_bytes"] == int(_PER_TOKEN / 4)
    # A rank-local budget of B bytes holds tokens whose full KV is 4B bytes
    # (both sides floor to whole tokens, so allow the rounding slack).
    assert abs(striped["total_kv_size_tokens"] - 4 * base["total_kv_size_tokens"]) <= 4
    # Consistency: bytes / (bytes per token) == tokens on both sides.
    assert striped["total_kv_size_tokens"] == striped["total_kv_size_bytes"] // striped["kv_size_per_token_bytes"]


def test_dcp_one_keeps_the_model_inverse_untouched(monkeypatch):
    _install_stubs(monkeypatch, dcp=1)
    estimator = memory.KVCacheEstimator.from_request(
        "deepseek-ai/DeepSeek-V3", "h200_sxm", "vllm", max_num_tokens=8192, max_batch_size=256
    )
    inverse = estimator.breakdown["tokens_from_kv_bytes"]
    assert inverse.__self__.__class__.__name__ == "_StubModel"


def test_model_without_divisor_hook_keeps_full_kv(monkeypatch):
    class _Plain:
        def get_kvcache_bytes_per_sequence(self, seq_len):
            return _PER_TOKEN * seq_len

        def get_kvcache_max_tokens(self, budget):
            return int(budget // _PER_TOKEN)

    class _StubBackend:
        def _get_memory_usage(self, *a, **k):
            return {"weights": 10.0, "activations": 1.0, "others": 1.0, "nccl": 1.0, "kvcache": 0.0}

    class _StubDB:
        def __init__(self):
            self.system_spec = {"gpu": {"mem_capacity": 100 * _GIB}}

    monkeypatch.setattr(memory, "get_model", lambda *a, **k: _Plain())
    monkeypatch.setattr(memory, "get_backend", lambda backend: _StubBackend())
    monkeypatch.setattr(memory.perf_database, "get_database", lambda *a, **k: _StubDB())
    out = memory.KVCacheEstimator.from_request(
        "Qwen/Qwen3-32B", "h200_sxm", "vllm", max_num_tokens=8192, max_batch_size=256
    ).estimate(is_of_free=False, fraction=0.9, gpu_memory_capacity_bytes_override=None)
    assert out["kv_size_per_token_bytes"] == int(_PER_TOKEN)
    assert out["total_kv_size_tokens"] == out["total_kv_size_bytes"] // int(_PER_TOKEN)
