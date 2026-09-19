# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unroll one backend selection into a self-contained deployment config."""

from __future__ import annotations

from typing import Any

from .afd_parallel import AFDParallelConfig
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
    "prefill_forward_model",
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
    "decode_forward_model",
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
    deployment_mode: str,
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig | AFDParallelConfig,
) -> dict[str, Any]:
    if deployment_mode in {"afd", "afd+pd"}:
        if not isinstance(parallel_config, AFDParallelConfig):
            raise TypeError(f"{deployment_mode} deployment_mode needs an AFDParallelConfig")
        if parallel_config.topology.adapter_topology != deployment_mode:
            raise TypeError(
                f"AFD topology advertises {parallel_config.topology.adapter_topology!r}, not {deployment_mode!r}"
            )
        out = {
            "afd": parallel_config.topology.provenance()["topology"],
            "afd_provenance": parallel_config.provenance(),
            "afd_phase": parallel_config.topology.phase.value,
            "afd_companion_role": parallel_config.companion_role,
            "used_gpus": parallel_config.total_gpus,
        }
        if parallel_config.companion is not None:
            role = parallel_config.companion_role
            assert role is not None
            for key, value in _shape_fields(parallel_config.companion.shape).items():
                out[f"{role}_{key}"] = value
            out[f"{role}_replicas"] = parallel_config.companion.replicas
        return out
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
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig | AFDParallelConfig,
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
    elif mode == "afd+pd":
        companion_role = sample["afd_companion_role"]
        searched = _DECODE_SEARCHED if companion_role == "decode" else _PREFILL_SEARCHED
        pinned = _DECODE_PINNED if companion_role == "decode" else _PREFILL_PINNED
    else:
        searched = ()
        pinned = ()
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
