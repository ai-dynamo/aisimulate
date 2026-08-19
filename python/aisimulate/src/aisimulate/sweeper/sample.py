# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unroll one backend selection into a self-contained deployment config."""

from __future__ import annotations

from typing import Any

from .config import SearchSpace
from .heterogeneous import DisaggBackendPair
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig

# Pinned deployment/runtime scalars folded in so the selected sample stands alone.
_DEPLOYMENT_PINNED = (
    "model_name",
    "hardware_sku",
    "gpu_budget",
    "min_gpu_budget",
    "context_length",
    "max_seq_len",
    "startup_time",
    "aic_nextn",
    "nextn_accepted",
    "enable_chunked_prefill",
    "enable_wideep",
    "enable_eplb",
    "wideep_num_slots",
    "moe_backend",
    "attention_backend",
    "gemm_quant_mode",
    "moe_quant_mode",
    "kvcache_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "free_gpu_memory_fraction",
    "prefill_rate_degradation",
    "decode_rate_degradation",
    "prefill_latency_correction",
    "decode_latency_correction",
    "ttft_correction_factor",
)

# engine knobs per branch: searched batching + pinned scalars.
_AGG_SEARCHED = ("agg_max_num_batched_tokens", "agg_max_num_seqs")
_AGG_PINNED = (
    "agg_block_size",
    "agg_gpu_memory_utilization",
    "agg_enable_prefix_caching",
)
_PREFILL_SEARCHED = ("prefill_max_num_batched_tokens", "prefill_max_num_seqs")
_PREFILL_PINNED = (
    "prefill_block_size",
    "prefill_gpu_memory_utilization",
    "prefill_enable_prefix_caching",
)
_DECODE_SEARCHED = ("decode_max_num_batched_tokens", "decode_max_num_seqs")
_DECODE_PINNED = (
    "decode_block_size",
    "decode_gpu_memory_utilization",
    "decode_enable_prefix_caching",
)


def _shape_fields(shape: ParallelShape) -> dict[str, Any]:
    return {
        "tp": shape.tp,
        "pp": shape.pp,
        "attention_dp": shape.dp,
        "moe_tp": shape.moe_tp,
        "moe_ep": shape.moe_ep,
        "cp": shape.cp,
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
    backend_pair: DisaggBackendPair | None = None,
) -> dict[str, Any]:
    """Expand a backend selection and its projected parallel configuration."""
    mode = selection["deployment_mode"]
    sample: dict[str, Any] = {"deployment_mode": mode, "backend": selection["backend"]}
    if backend_pair is not None:
        if mode != "disagg":
            raise ValueError("backend_pair is only valid for deployment_mode='disagg'")
        sample.update(
            prefill_backend=backend_pair.prefill,
            decode_backend=backend_pair.decode,
        )
        for role in ("prefill", "decode"):
            for name in ("model_name", "hardware_sku"):
                sample[f"{role}_{name}"] = search_space.role_value(role, name)

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
        if key in selection:
            sample[key] = selection[key]
            continue
        role, suffix = key.split("_max_", 1)
        alias = (
            f"{role}_context_tokens"
            if suffix == "num_batched_tokens"
            else f"{role}_batch_size"
        )
        sample[alias] = selection[alias]
        sample[key] = selection[alias]
    for key in pinned:
        sample[key] = getattr(search_space, key)
    return sample
