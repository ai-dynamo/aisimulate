# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an unrolled backend sample into a replay deployment specification."""

from __future__ import annotations

from typing import Any

from ..aic import estimate_kv_bytes_per_token, materialize_aic_num_gpu_blocks
from .replay import BackendDeploymentSpec, EncoderPoolSpec


def _role_prefix(role: str) -> str:
    """Field prefix in the unrolled sample for a role (empty for agg shape fields)."""
    return "" if role == "agg" else f"{role}_"


def _performance_model_metadata(sample: dict[str, Any], role: str, *, backend_version: str) -> dict[str, Any]:
    """Keep optional perf-model identity separate from runtime timing args."""
    prefix = _role_prefix(role)
    moe_tp = int(sample[f"{prefix}moe_tp"])
    moe_ep = int(sample[f"{prefix}moe_ep"])
    config: dict[str, Any] = {
        "backend": sample["backend"],
        "backend_version": backend_version,
        "system": sample["hardware_sku"],
        "model_path": sample["model_name"],
        "tp_size": int(sample[f"{prefix}tp"]),
        "attention_dp_size": int(sample[f"{prefix}attention_dp"]),
        "moe_tp_size": moe_tp if moe_tp * moe_ep > 1 else None,
        "moe_ep_size": moe_ep if moe_tp * moe_ep > 1 else None,
        "nextn": sample.get("aic_nextn"),
        "forward_model": (
            (sample.get(f"{role}_forward_model") or "op_level")
            if sample.get(f"{role}_timing_model") is None
            else "op_level"
        ),
    }
    return {"provider": "aic", "config": config}


def _engine_args_payload(sample: dict[str, Any], role: str, *, backend_version: str) -> dict[str, Any]:
    """Build the runner-neutral engine argument payload for one role."""
    prefix = _role_prefix(role)
    tp = int(sample[f"{prefix}tp"])
    attention_dp = int(sample[f"{prefix}attention_dp"])
    moe_tp = int(sample[f"{prefix}moe_tp"])
    moe_ep = int(sample[f"{prefix}moe_ep"])
    backend = sample["backend"]
    block_size = sample[f"{role}_block_size"]
    if block_size is None:
        block_size = {"vllm": 64, "sglang": 1, "trtllm": 32}[backend]
    memory_fraction = sample[f"{role}_gpu_memory_utilization"]
    if memory_fraction is None:
        memory_fraction = 0.88 if backend == "sglang" else 0.9
    memory_fraction_field = {
        "vllm": "gpu_memory_utilization",
        "sglang": "mem_fraction_static",
        "trtllm": "free_gpu_memory_fraction",
    }[backend]
    payload: dict[str, Any] = {
        "worker_type": "aggregated" if role == "agg" else role,
        "engine_type": backend,
        "aic_backend": backend,
        "aic_backend_version": backend_version,
        "aic_system": sample["hardware_sku"],
        "aic_model_path": sample["model_name"],
        "aic_tp_size": tp,
        "aic_attention_dp_size": attention_dp,
        "max_num_batched_tokens": int(sample[f"{role}_max_num_batched_tokens"]),
        "max_num_seqs": int(sample[f"{role}_max_num_seqs"]),
        "block_size": int(block_size),
        memory_fraction_field: float(memory_fraction),
        "enable_prefix_caching": bool(sample[f"{role}_enable_prefix_caching"]),
    }
    if backend == "vllm" and sample.get("context_length") is not None:
        payload["max_model_len"] = int(sample["context_length"])
    if moe_tp * moe_ep > 1:
        payload["aic_moe_tp_size"] = moe_tp
        payload["aic_moe_ep_size"] = moe_ep
    if sample.get("aic_nextn") is not None:
        payload["aic_nextn"] = int(sample["aic_nextn"])
    forward_model = sample.get(f"{role}_forward_model")
    if forward_model is not None and forward_model != "op_level":
        payload["aic_forward_model"] = str(forward_model)
    startup = sample.get(f"{role}_startup_time")
    if startup is None:
        startup = sample.get("startup_time")
    if startup is not None:
        payload["startup_time"] = float(startup)
    if sample.get(f"{role}_num_gpu_blocks") is not None:
        payload["num_gpu_blocks"] = int(sample[f"{role}_num_gpu_blocks"])
        payload.pop(memory_fraction_field, None)
    if sample.get(f"{role}_timing_model") is not None:
        payload["timing_model"] = dict(sample[f"{role}_timing_model"])
        if sample.get(f"{role}_num_gpu_blocks") is None:
            payload = materialize_aic_num_gpu_blocks(payload)
        for name in (
            "aic_backend_version",
            "aic_system",
            "aic_model_path",
            "aic_moe_tp_size",
            "aic_moe_ep_size",
            "aic_nextn",
            "aic_forward_model",
        ):
            payload.pop(name, None)
    host_offload = sample.get(f"{role}_native_host_offload")
    if host_offload is not None:
        configured_bytes = sample[f"{role}_kv_bytes_per_token"]
        payload["kv_cache_bytes_per_token"] = (
            estimate_kv_bytes_per_token(
                str(sample["model_name"]),
                tp_size=tp,
                pp_size=int(sample[f"{prefix}pp"]),
                moe_tp_size=moe_tp,
                moe_ep_size=moe_ep,
            )
            if configured_bytes == "auto"
            else int(configured_bytes)
        )
    transfer_geometry = sample.get("kv_transfer_bytes_per_token")
    if role in {"prefill", "decode"} and transfer_geometry is not None:
        payload["kv_transfer_bytes_per_token"] = (
            estimate_kv_bytes_per_token(
                str(sample["model_name"]),
                tp_size=int(sample["prefill_tp"]),
                pp_size=int(sample["prefill_pp"]),
                moe_tp_size=int(sample["prefill_moe_tp"]),
                moe_ep_size=int(sample["prefill_moe_ep"]),
            )
            if transfer_geometry == "auto"
            else int(transfer_geometry)
        )
    if host_offload is not None:
        payload["native_host_offload"] = dict(host_offload)
    if role in {"prefill", "decode"}:
        if sample.get("kv_transfer_bandwidth") is not None:
            payload["kv_transfer_bandwidth"] = float(sample["kv_transfer_bandwidth"])
        payload["kv_transfer_timing_mode"] = sample["kv_transfer_timing_mode"]
    return payload


def build_backend_deployment(
    sample: dict[str, Any],
    *,
    backend_version: str,
    encoder: EncoderPoolSpec | None = None,
) -> BackendDeploymentSpec:
    """Build the Dynamo-independent backend part of a :class:`ReplaySpec`."""
    mode = sample["deployment_mode"]
    common = {
        "encoder": encoder,
        "deployment_mode": mode,
        "backend": sample["backend"],
        "backend_version": backend_version,
        "parallel_config": {
            key: value
            for key, value in sample.items()
            if key
            in {
                "tp",
                "pp",
                "attention_dp",
                "moe_tp",
                "moe_ep",
                "strategy",
                "replicas",
                "prefill_tp",
                "prefill_pp",
                "prefill_attention_dp",
                "prefill_moe_tp",
                "prefill_moe_ep",
                "prefill_strategy",
                "prefill_replicas",
                "decode_tp",
                "decode_pp",
                "decode_attention_dp",
                "decode_moe_tp",
                "decode_moe_ep",
                "decode_strategy",
                "decode_replicas",
            }
        },
        "performance_model_metadata": {
            ("aggregated" if role == "agg" else role): _performance_model_metadata(
                sample, role, backend_version=backend_version
            )
            for role in (("agg",) if mode == "agg" else ("prefill", "decode"))
        },
    }
    if mode == "agg":
        return BackendDeploymentSpec(
            agg_engine_args=_engine_args_payload(sample, "agg", backend_version=backend_version),
            num_workers=int(sample["replicas"]),
            **common,
        )
    prefill_args = _engine_args_payload(sample, "prefill", backend_version=backend_version)
    decode_args = _engine_args_payload(sample, "decode", backend_version=backend_version)
    return BackendDeploymentSpec(
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        num_prefill_workers=int(sample["prefill_replicas"]),
        num_decode_workers=int(sample["decode_replicas"]),
        **common,
    )
