# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aisimulate.sweeper.config import Workload
from aisimulate.sweeper.kv_load import InfeasibleKVCapacity, resolve_kv_load
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)


@pytest.mark.parametrize("source", ["resolved", "custom", "legacy_custom"])
def test_capacity_cache_includes_resolved_root(tmp_path, monkeypatch, source):
    from aisimulate.sweeper import kv_load

    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
    calls = []

    def capacity(*args, systems_paths, **kwargs):
        calls.append(systems_paths)
        return 6400 if systems_paths == [str(roots[0])] else 12800

    monkeypatch.setattr(kv_load, "estimate_kv_tokens", capacity)
    kv_load._per_rank_capacity_tokens.cache_clear()
    sample = _sample("agg")
    parallel = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    capacities = []
    try:
        for root in [roots[0], roots[1], roots[0]]:
            if source == "resolved":
                sample["forward_pass_estimators"] = {"agg": {"config": {"systems_paths": [str(root)]}}}
                sample["agg_timing_model"] = {
                    "type": "external",
                    "provider": "aic",
                    "config": {"systems_paths": ["must-not-be-used"]},
                }
            else:
                config = {"systems_paths": [str(root)]} if source == "custom" else {"systems_path": str(root)}
                sample["agg_timing_model"] = {"type": "external", "provider": "aic", "config": config}
            result = resolve_kv_load(
                sample,
                workload=Workload(isl=100, osl=100, kv_load_ratio=1.0, request_count=1),
                parallel_config=parallel,
                ratio=1.0,
                backend_version="v",
            )
            capacities.append(result.role_capacity_tokens["agg"])
        assert capacities == [6400, 12800, 6400]
        assert calls == [[str(roots[0])], [str(roots[1])]]
    finally:
        kv_load._per_rank_capacity_tokens.cache_clear()


@pytest.mark.parametrize("custom_role", ["prefill", "decode"])
def test_mixed_timing_capacity_uses_each_roles_own_root(tmp_path, monkeypatch, custom_role):
    from aisimulate.sweeper import kv_load

    sample = _sample("disagg")
    roots = {role: tmp_path / role for role in ("prefill", "decode")}
    for root in roots.values():
        root.mkdir()
    canonical_role = "prefill" if custom_role == "decode" else "decode"
    sample["forward_pass_estimators"] = {canonical_role: {"config": {"systems_paths": [str(roots[canonical_role])]}}}
    sample[f"{custom_role}_timing_model"] = {
        "type": "external",
        "provider": "aic",
        "config": {"systems_paths": [str(roots[custom_role])]},
    }
    seen = []

    def capacity(*args, systems_paths, **kwargs):
        seen.append(systems_paths)
        return 6400

    monkeypatch.setattr(kv_load, "_per_rank_capacity_tokens", capacity)
    parallel = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    resolve_kv_load(
        sample,
        workload=Workload(isl=100, osl=100, kv_load_ratio=1.0, request_count=1),
        parallel_config=DisaggParallelConfig(prefill=parallel, decode=parallel),
        ratio=1.0,
        backend_version="v",
    )
    assert seen == [(str(roots["prefill"]),), (str(roots["decode"]),)]


def _sample(mode: str) -> dict:
    sample = {
        "deployment_mode": mode,
        "model_name": "m",
        "hardware_sku": "h",
        "backend": "vllm",
        "aic_nextn": None,
    }
    roles = ("agg",) if mode == "agg" else ("prefill", "decode")
    for role in roles:
        sample.update(
            {
                f"{role}_block_size": 64,
                f"{role}_max_num_batched_tokens": 8192,
                f"{role}_max_num_seqs": 256,
                f"{role}_gpu_memory_utilization": 0.9,
            }
        )
    return sample


def test_agg_capacity_scales_by_attention_dp_and_replicas(monkeypatch):
    shape = ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=2)
    config = ReplicaParallelConfig(shape=shape, replicas=3)
    monkeypatch.setattr(
        "aisimulate.sweeper.kv_load._per_rank_capacity_tokens",
        lambda *args, **kwargs: 10_000,
    )

    resolution = resolve_kv_load(
        _sample("agg"),
        workload=Workload(isl=100, osl=100, kv_load_ratio=0.5, num_request_ratio=10),
        parallel_config=config,
        ratio=0.5,
        backend_version="v",
    )

    # floor(10000 / 64) * 64 per rank, then x2 attention-DP ranks x3 replicas.
    assert resolution.role_capacity_tokens == {"agg": 59_904}
    assert resolution.concurrency_capacity == 399  # 59904 / (100 + 100/2)
    assert resolution.concurrency == 199


def test_disagg_load_uses_decode_capacity_but_validates_prefill(monkeypatch):
    prefill = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=2), replicas=1)
    decode = ReplicaParallelConfig(ParallelShape(tp=1, dp=4, moe_tp=1, moe_ep=4), replicas=2)
    config = DisaggParallelConfig(prefill=prefill, decode=decode)
    seen = []

    def fake_role(sample, *, role, config, backend_version):
        seen.append(role)
        return {"prefill": 100_000, "decode": 300_000}[role]

    monkeypatch.setattr("aisimulate.sweeper.kv_load._role_capacity_tokens", fake_role)
    resolution = resolve_kv_load(
        _sample("disagg"),
        workload=Workload(isl=1000, osl=1000, kv_load_ratio=1.0, num_request_ratio=10),
        parallel_config=config,
        ratio=1.0,
        backend_version="v",
    )

    assert seen == ["prefill", "decode"]
    assert resolution.concurrency_capacity == 200  # decode 300k / (1000 + 500)
    assert resolution.concurrency == 200


def test_disagg_kv_capacity_uses_role_hardware(monkeypatch):
    config = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
        decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    sample = _sample("disagg")
    sample.update(
        prefill_hardware_sku="prefill_sku",
        decode_hardware_sku="decode_sku",
    )
    seen = []

    def fake_capacity(*args, hardware_sku, **kwargs):
        seen.append(hardware_sku)
        return 100_000

    monkeypatch.setattr("aisimulate.sweeper.kv_load._per_rank_capacity_tokens", fake_capacity)

    resolve_kv_load(
        sample,
        workload=Workload(isl=1000, osl=1000, kv_load_ratio=1.0, num_request_ratio=10),
        parallel_config=config,
        ratio=1.0,
        backend_version="v",
    )

    assert seen == ["prefill_sku", "decode_sku"]


def test_zero_ratio_maps_to_one_request(monkeypatch):
    config = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    monkeypatch.setattr(
        "aisimulate.sweeper.kv_load._role_capacity_tokens",
        lambda *args, **kwargs: 10_000,
    )

    resolution = resolve_kv_load(
        _sample("agg"),
        workload=Workload(isl=100, osl=100, kv_load_ratio=0.0, num_request_ratio=10),
        parallel_config=config,
        ratio=0.0,
        backend_version="v",
    )

    assert resolution.concurrency == 1


def test_capacity_smaller_than_one_average_request_is_infeasible(monkeypatch):
    config = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    monkeypatch.setattr(
        "aisimulate.sweeper.kv_load._role_capacity_tokens",
        lambda *args, **kwargs: 100,
    )

    with pytest.raises(InfeasibleKVCapacity, match="cannot hold"):
        resolve_kv_load(
            _sample("agg"),
            workload=Workload(isl=100, osl=100, kv_load_ratio=1.0, num_request_ratio=10),
            parallel_config=config,
            ratio=1.0,
            backend_version="v",
        )


def test_block_size_must_be_positive():
    config = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    sample = _sample("agg")
    sample["agg_block_size"] = 0

    with pytest.raises(ValueError, match="agg_block_size must be greater than zero"):
        resolve_kv_load(
            sample,
            workload=Workload(isl=100, osl=100, kv_load_ratio=1.0, num_request_ratio=10),
            parallel_config=config,
            ratio=1.0,
            backend_version="v",
        )


def test_average_tokens_per_request_must_be_positive(monkeypatch):
    config = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    monkeypatch.setattr(
        "aisimulate.sweeper.kv_load._role_capacity_tokens",
        lambda *args, **kwargs: 10_000,
    )

    with pytest.raises(InfeasibleKVCapacity, match="positive average tokens per request"):
        resolve_kv_load(
            _sample("agg"),
            workload=SimpleNamespace(isl=0, osl=0),
            parallel_config=config,
            ratio=1.0,
            backend_version="v",
        )


@pytest.mark.parametrize("source", ["resolved", "custom"])
def test_capacity_uses_nonnull_version_and_resolved_controls(monkeypatch, source):
    from aisimulate.sweeper import kv_load

    sample = _sample("agg")
    sample.update(enable_eplb=True, wideep_num_slots=32, moe_backend="deepep_moe")
    identity = {"backend_version": None, "enable_eplb": False, "wideep_num_slots": 64, "moe_backend": None}
    if source == "resolved":
        sample["forward_pass_estimators"] = {"agg": {"config": identity}}
    else:
        sample["agg_timing_model"] = {"type": "external", "provider": "aic", "config": identity}
    seen = []

    def capacity(*args, **kwargs):
        seen.append(kwargs)
        return 6400

    monkeypatch.setattr(kv_load, "_per_rank_capacity_tokens", capacity)
    resolve_kv_load(
        sample,
        workload=Workload(isl=100, osl=100, request_count=1, kv_load_ratio=1.0),
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
        ratio=1,
        backend_version="pinned-version",
    )
    assert seen[0]["backend_version"] == "pinned-version"
    assert dict(seen[0]["model_controls"]) == {"wideep_num_slots": 64}


@pytest.mark.parametrize("invalid", [None, 7, "invalid", []])
def test_capacity_rejects_malformed_external_identity(invalid):
    sample = _sample("agg")
    sample["agg_timing_model"] = {"type": "external", "provider": "aic", "config": invalid}
    with pytest.raises(ValueError, match="external AIC timing config must be a mapping"):
        resolve_kv_load(
            sample,
            workload=Workload(isl=100, osl=100, request_count=1, kv_load_ratio=1.0),
            parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
            ratio=1,
            backend_version="v",
        )


def test_image_workloads_size_the_load_on_the_placeholders_the_runner_lays_out():
    """Native VL and analytical EPD both count the visual tokens, overrides included."""
    from aisimulate.sweeper.config import ImageWorkload

    sample = {
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "agg_block_size": 16,
        "agg_num_gpu_blocks": 6_250,
        "agg_vision": {"cache_mib": 100, "encoder_parallel": "tp"},
    }
    parallel = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)

    def concurrency(**bounds):
        return resolve_kv_load(
            sample,
            workload=Workload(
                isl=128,
                osl=4,
                kv_load_ratio=1.0,
                request_count=1,
                images=ImageWorkload(height=448, width=448, **bounds),
            ),
            parallel_config=parallel,
            ratio=1.0,
            backend_version="0.5.14",
        ).concurrency

    # 448x448 -> 196 visual tokens; capped at 65536 pixels the image shrinks to 256x256 -> 64.
    assert concurrency() == 100_000 // (128 + 196 + 2)
    assert concurrency(max_pixels=65536) == 100_000 // (128 + 64 + 2)
    # Analytical EPD (no vision tower on the language worker) sizes on the same placeholders.
    del sample["agg_vision"]
    assert concurrency() == 100_000 // (128 + 196 + 2)
