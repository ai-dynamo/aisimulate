# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unroll one backend selection into a self-contained deployment config."""

from __future__ import annotations

from typing import Any

from .afd import AFDParallelConfig
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


def _unroll_afd(parallel_config: AFDParallelConfig) -> dict[str, Any]:
    topology = parallel_config.topology
    out: dict[str, Any] = {
        "afd_n_a_nodes": topology.n_a_nodes,
        "afd_n_f_nodes": topology.n_f_nodes,
        "afd_gpus_per_node": topology.gpus_per_node,
        "afd_tp_a": topology.tp_a,
        "afd_ffn_tp": topology.ffn_tp,
        "afd_f_moe_ep_size": topology.f_moe_ep_size,
        "afd_a_batch_size": topology.a_batch_size,
        "afd_total_batch_size": topology.total_batch_size,
        "afd_num_microbatches": topology.num_microbatches,
        "afd_pipeline_model": topology.pipeline_model.value,
        "afd_phase": topology.phase.value,
        "afd_combined_with_pd": topology.combined_with_pd,
        "afd_comm_overhead_factor": topology.comm_overhead_factor,
        "afd_boundary_on_attn": topology.boundary_on_attn,
        "afd_attention_workers": topology.attention_workers,
        "afd_ffn_workers": topology.ffn_workers,
        "afd_attention_gpus": topology.attention_gpus,
        "afd_ffn_gpus": topology.ffn_gpus,
        "afd_companion_gpus": (parallel_config.companion.total_gpus if parallel_config.companion is not None else 0),
        "afd_provenance": parallel_config.provenance(),
        "used_gpus": parallel_config.total_gpus,
    }
    if parallel_config.companion is not None:
        role = parallel_config.companion_role
        assert role is not None
        for key, value in _shape_fields(parallel_config.companion.shape).items():
            out[f"{role}_{key}"] = value
        out[f"{role}_replicas"] = parallel_config.companion.replicas
    return out


def _materialize_searched_knob(sample: dict[str, Any], selection: dict[str, Any], key: str) -> None:
    if key in selection:
        sample[key] = selection[key]
        return
    role, suffix = key.split("_max_", 1)
    alias = f"{role}_context_tokens" if suffix == "num_batched_tokens" else f"{role}_batch_size"
    sample[alias] = selection[alias]
    sample[key] = selection[alias]


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

    if isinstance(parallel_config, AFDParallelConfig):
        sample.update(_unroll_afd(parallel_config))
        if mode == "afd":
            return sample
        if mode != "afd+pd" or parallel_config.companion_role is None:
            raise TypeError("AFD parallel config does not match its deployment mode")
        role = parallel_config.companion_role
        searched = _PREFILL_SEARCHED if role == "prefill" else _DECODE_SEARCHED
        pinned = _PREFILL_PINNED if role == "prefill" else _DECODE_PINNED
        for key in searched:
            _materialize_searched_knob(sample, selection, key)
        for key in pinned:
            sample[key] = getattr(search_space, key)
        return sample

    sample.update(_unroll_parallel(mode, parallel_config))

    # engine knobs for the active branch only
    if mode == "agg":
        searched, pinned = _AGG_SEARCHED, _AGG_PINNED
    else:
        searched = _PREFILL_SEARCHED + _DECODE_SEARCHED
        pinned = _PREFILL_PINNED + _DECODE_PINNED
    for key in searched:
        _materialize_searched_knob(sample, selection, key)
    for key in pinned:
        sample[key] = getattr(search_space, key)
    return sample
