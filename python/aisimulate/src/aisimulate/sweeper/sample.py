# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unroll one backend selection into a self-contained deployment config."""

from __future__ import annotations

from typing import Any

from .config import SearchSpace
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig

# Pinned deployment/runtime scalars folded in so the selected sample stands alone.
_DEPLOYMENT_PINNED = (
    "model_name",
    "hardware_sku",
    "gpu_budget",
    "min_gpu_budget",
    "context_length",
    "startup_time",
    "aic_nextn",
)

# engine knobs per branch: searched batching + pinned scalars.
_AGG_SEARCHED = ("agg_max_num_batched_tokens", "agg_max_num_seqs")
_AGG_PINNED = (
    "agg_block_size",
    "agg_gpu_memory_utilization",
    "agg_enable_prefix_caching",
    "agg_kv_bytes_per_token",
    "agg_native_host_offload",
    "agg_num_gpu_blocks",
    "agg_timing_model",
    "agg_startup_time",
)
_PREFILL_SEARCHED = ("prefill_max_num_batched_tokens", "prefill_max_num_seqs")
_PREFILL_PINNED = (
    "prefill_block_size",
    "prefill_gpu_memory_utilization",
    "prefill_enable_prefix_caching",
    "prefill_kv_bytes_per_token",
    "prefill_native_host_offload",
    "prefill_num_gpu_blocks",
    "prefill_timing_model",
    "prefill_startup_time",
)
_DECODE_SEARCHED = ("decode_max_num_batched_tokens", "decode_max_num_seqs")
_DECODE_PINNED = (
    "decode_block_size",
    "decode_gpu_memory_utilization",
    "decode_enable_prefix_caching",
    "decode_kv_bytes_per_token",
    "decode_native_host_offload",
    "decode_num_gpu_blocks",
    "decode_timing_model",
    "decode_startup_time",
)


def _shape_fields(shape: ParallelShape) -> dict[str, Any]:
    return {
        "tp": shape.tp,
        "pp": shape.pp,
        "attention_dp": shape.dp,
        "moe_tp": shape.moe_tp,
        "moe_ep": shape.moe_ep,
        "strategy": shape.strategy,
    }


def _unroll_parallel(
    deployment_mode: str, parallel_config: ReplicaParallelConfig | DisaggParallelConfig
) -> dict[str, Any]:
    if deployment_mode == "agg":
        if not isinstance(parallel_config, ReplicaParallelConfig):
            raise TypeError("agg deployment_mode needs a ReplicaParallelConfig")
        out = _shape_fields(parallel_config.shape)
        out["replicas"] = parallel_config.replicas
        out["used_gpus"] = parallel_config.total_gpus
        return out
    if not isinstance(parallel_config, DisaggParallelConfig):
        raise TypeError("disagg deployment_mode needs a DisaggParallelConfig")
    out = {}
    for role, rc in (
        ("prefill", parallel_config.prefill),
        ("decode", parallel_config.decode),
    ):
        for key, value in _shape_fields(rc.shape).items():
            out[f"{role}_{key}"] = value
        out[f"{role}_replicas"] = rc.replicas
    out["used_gpus"] = parallel_config.total_gpus
    return out


def unroll_sample(
    *,
    search_space: SearchSpace,
    selection: dict[str, Any],
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig,
) -> dict[str, Any]:
    """Expand a backend selection and its projected parallel configuration."""
    mode = selection["deployment_mode"]
    sample: dict[str, Any] = {"deployment_mode": mode, "backend": selection["backend"]}

    for key in _DEPLOYMENT_PINNED:
        sample[key] = getattr(search_space, key)

    sample.update(_unroll_parallel(mode, parallel_config))

    # engine knobs for the active branch only
    if mode == "agg":
        searched, pinned = _AGG_SEARCHED, _AGG_PINNED
    else:
        searched = _PREFILL_SEARCHED + _DECODE_SEARCHED
        pinned = _PREFILL_PINNED + _DECODE_PINNED
    for key in searched:
        sample[key] = selection[key]
    for key in pinned:
        sample[key] = selection.get(key, getattr(search_space, key))
    if mode == "disagg":
        for key in (
            "kv_transfer_bytes_per_token",
            "kv_transfer_bandwidth",
            "kv_transfer_timing_mode",
        ):
            sample[key] = getattr(search_space, key)
    return sample
