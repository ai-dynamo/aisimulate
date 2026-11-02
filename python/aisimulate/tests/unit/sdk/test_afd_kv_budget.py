# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit contracts for `estimate_afd_kv_budget_bytes`."""

from __future__ import annotations

from typing import ClassVar

import pytest

from aisimulate.sdk import afd_memory

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pure_model(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.afd_parallel.validate_afd_model",
        lambda model: {"num_experts": 128, "num_shared_experts": 0, "num_moe_layers": 48},
    )


_CAPACITY_GIB = 80.0
_WEIGHTS = {"prefill": 10.0, "decode": 10.0}


def _fake_database():
    class _Database:
        system_spec: ClassVar[dict] = {"gpu": {"mem_capacity": int(_CAPACITY_GIB * (1 << 30))}}

    return _Database()


def _patch_common(monkeypatch, *, per_token_gib: float, non_kv_gib: float):
    class _Backend:
        name = type("Name", (), {"value": "sglang"})()

        @staticmethod
        def get_default_free_gpu_memory_fraction(version):
            return 0.88

        @staticmethod
        def get_kv_cache_memory_check_params():
            return 0.0, 0.0

        @staticmethod
        def memory_fraction_of_free():
            return False

    def fake_pool_memory(
        backend,
        database,
        model,
        *,
        partition_weight_gb,
        phase,
        batch_size,
        isl,
        osl,
        prefix,
        max_seq_len,
        include_kvcache,
        kvcache_multiplier,
        prefill_tokens=None,
    ):
        del backend, database, model, partition_weight_gb, prefix, include_kvcache
        seq = max_seq_len if max_seq_len is not None else (max(isl, 1) if phase == "prefill" else isl + osl)
        multiplier = 1 if phase == "prefill" else kvcache_multiplier
        kvcache = batch_size * seq * multiplier * per_token_gib
        return {
            "weights": 10.0,
            "activations": non_kv_gib - 10.0,
            "kvcache": kvcache,
            "nccl": 0.0,
            "others": 0.0,
            "total": non_kv_gib + kvcache,
        }

    monkeypatch.setattr(afd_memory, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(afd_memory, "get_latest_database_version", lambda system, backend: "test")
    monkeypatch.setattr(afd_memory, "_build_pool", lambda *args, **kwargs: (_fake_database(), object()))
    monkeypatch.setattr(afd_memory, "_pool_memory", fake_pool_memory)


def _estimate(*, phase="both", max_seq_len=None):
    return afd_memory.estimate_afd_kv_budget_bytes(
        model_path="test/model",
        backend_name="sglang",
        backend_version="test",
        isl=1024,
        osl=128,
        prefix=0,
        a_pool={"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=18,
        num_microbatches=3,
        a_weight_gb=_WEIGHTS,
        phase=phase,
        max_seq_len=max_seq_len,
    )


def test_budget_is_the_capacity_fraction_minus_non_kv(monkeypatch):
    _patch_common(monkeypatch, per_token_gib=0.000_001, non_kv_gib=30.0)
    budget_bytes, per_token_bytes = _estimate()
    # of_total: budget = capacity * 0.88 - non_kv = 80*0.88 - 30 = 40.4 GiB.
    assert budget_bytes == pytest.approx(40.4 * (1 << 30))
    assert per_token_bytes == pytest.approx(0.000_001 * (1 << 30))


def test_budget_uses_the_worse_phase(monkeypatch):
    _patch_common(monkeypatch, per_token_gib=0.000_001, non_kv_gib=30.0)
    budget_bytes, _ = _estimate(phase="both")
    # Both phases share non-KV here, so the min matches either single phase.
    decode_budget, _ = _estimate(phase="decode")
    assert budget_bytes == pytest.approx(decode_budget)


def test_per_token_price_uses_unit_geometry(monkeypatch):
    # batch/multiplier must not leak into the per-token price: with unit
    # geometry kvcache = 1 * seq * 1 * per_token.
    _patch_common(monkeypatch, per_token_gib=0.000_002, non_kv_gib=30.0)
    _, per_token_bytes = _estimate(phase="decode")
    assert per_token_bytes == pytest.approx(0.000_002 * (1 << 30))


def test_non_positive_budget_fails_loudly(monkeypatch):
    _patch_common(monkeypatch, per_token_gib=0.000_001, non_kv_gib=75.0)
    with pytest.raises(afd_memory.AFDMemoryInfeasible, match="not positive"):
        _estimate(phase="decode")
