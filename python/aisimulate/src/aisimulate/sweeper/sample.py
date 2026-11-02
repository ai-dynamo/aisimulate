# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unroll one backend selection into a self-contained deployment config."""

from __future__ import annotations

from typing import Any

from .afd_parallel import AFDTopology
from .config import ENGINE_MODEL_CONTROL_FIELDS, SearchSpace
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
    "nextn_accepted",
    "enable_chunked_prefill",
    *ENGINE_MODEL_CONTROL_FIELDS,
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
    "agg_forward_model",
    "agg_fpm_parquet_path",
    "agg_startup_time",
    "agg_host_loop",
    "agg_frontend",
    "agg_frontend_by_transport",
    "agg_vision",
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
    "prefill_forward_model",
    "prefill_fpm_parquet_path",
    "prefill_startup_time",
    "prefill_host_loop",
    "prefill_frontend",
    "prefill_frontend_by_transport",
    "prefill_vision",
    "prefill_context_length",
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
    "decode_forward_model",
    "decode_fpm_parquet_path",
    "decode_startup_time",
    "decode_context_length",
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
    deployment_mode: str,
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig | AFDTopology,
) -> dict[str, Any]:
    if deployment_mode == "agg" and not isinstance(parallel_config, (ReplicaParallelConfig, AFDTopology)):
        raise TypeError("agg deployment_mode needs a ReplicaParallelConfig or AFDTopology")
    if deployment_mode == "disagg" and not isinstance(parallel_config, DisaggParallelConfig):
        raise TypeError("disagg deployment_mode needs a DisaggParallelConfig")
    roles = (
        (("aggregated", parallel_config),)
        if deployment_mode == "agg"
        else (("prefill", parallel_config.prefill), ("decode", parallel_config.decode))
    )
    out = {"used_gpus": parallel_config.total_gpus}
    for role, config in roles:
        prefix = "" if role == "aggregated" else f"{role}_"
        if isinstance(config, AFDTopology):
            out.setdefault("afd_roles", {})[role] = config.provenance()["topology"]
            out[f"{prefix}replicas"] = 1
        else:
            out.update({f"{prefix}{k}": v for k, v in _shape_fields(config.shape).items()})
            out[f"{prefix}replicas"] = config.replicas
    return out


def unroll_sample(
    *,
    search_space: SearchSpace,
    selection: dict[str, Any],
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig | AFDTopology,
) -> dict[str, Any]:
    """Expand a backend selection and its projected parallel configuration."""
    mode = selection["deployment_mode"]
    sample: dict[str, Any] = {"deployment_mode": mode, "backend": selection["backend"]}

    for key in _DEPLOYMENT_PINNED:
        sample[key] = getattr(search_space, key)
    if search_space.systems_paths is not None:
        sample["systems_paths"] = search_space.systems_paths
    if search_space.fpm_profile is not None:
        sample["fpm_profile"] = search_space.fpm_profile

    if mode == "disagg":
        sample["prefill_hardware_sku"] = search_space.hardware_sku_for("prefill")
        sample["decode_hardware_sku"] = search_space.hardware_sku_for("decode")

    sample.update(_unroll_parallel(mode, parallel_config))

    # engine knobs for the active branch only
    if mode == "agg":
        searched, pinned = _AGG_SEARCHED, _AGG_PINNED
    elif mode == "disagg":
        searched = _PREFILL_SEARCHED + _DECODE_SEARCHED
        pinned = _PREFILL_PINNED + _DECODE_PINNED
    else:
        raise ValueError(f"unsupported deployment mode {mode!r}")
    afd_roles = sample.get("afd_roles", {})
    searched = tuple(
        k for k in searched if ("aggregated" if k.startswith("agg_") else k.split("_", 1)[0]) not in afd_roles
    )
    pinned = tuple(k for k in pinned if ("aggregated" if k.startswith("agg_") else k.split("_", 1)[0]) not in afd_roles)
    for role in afd_roles:
        raw = search_space.afd_roles[role]
        sample.setdefault("afd_worker_settings", {})[role] = {
            k: raw[k] for k in ("context_length", "startup_seconds") if k in raw
        }
        if raw["afd"].get("chunked_prefill_size") is not None:
            sample["afd_roles"][role]["chunked_prefill_size"] = raw["afd"]["chunked_prefill_size"]
    if afd_roles:
        sample["estimator"] = {
            k: getattr(search_space, k)
            for k in (
                "database_mode",
                "systems_paths",
                "transfer_policy",
                "estimation_mode",
                "fallback_policy",
                "estimator_config",
            )
            if getattr(search_space, k) is not None
        }
        sample["quant"] = {
            k: getattr(search_space, k)
            for k in ("gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "kvcache_quant_mode", "comm_quant_mode")
            if getattr(search_space, k) is not None
        }
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
    if search_space.speculation is not None:
        sample["speculation"] = search_space.speculation.model_dump(mode="json")
    return sample
