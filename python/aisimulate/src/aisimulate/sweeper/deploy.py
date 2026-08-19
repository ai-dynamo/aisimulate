# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an unrolled backend sample into a replay deployment specification."""

from __future__ import annotations

from typing import Any

from .replay import BackendDeploymentSpec, EngineRequestSpec


def _role_prefix(role: str) -> str:
    """Field prefix in the unrolled sample for a role (empty for agg shape fields)."""
    return "" if role == "agg" else f"{role}_"


def _engine_args_payload(
    sample: dict[str, Any],
    role: str,
    *,
    backend_version: str,
    engine_request: EngineRequestSpec | None = None,
) -> dict[str, Any]:
    """Build the runner-neutral engine argument payload for one role."""
    prefix = _role_prefix(role)
    tp = int(sample[f"{prefix}tp"])
    attention_dp = int(sample[f"{prefix}attention_dp"])
    moe_tp = int(sample[f"{prefix}moe_tp"])
    moe_ep = int(sample[f"{prefix}moe_ep"])
    backend = sample["backend"]
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
        "block_size": int(sample[f"{role}_block_size"]),
        memory_fraction_field: float(
            engine_request.memory_fraction_by_role[role]
            if engine_request is not None
            else sample[f"{role}_gpu_memory_utilization"]
        ),
        "enable_prefix_caching": bool(sample[f"{role}_enable_prefix_caching"]),
    }
    if moe_tp * moe_ep > 1:
        payload["aic_moe_tp_size"] = moe_tp
        payload["aic_moe_ep_size"] = moe_ep
    if sample.get("aic_nextn"):
        payload["aic_nextn"] = int(sample["aic_nextn"])
    if engine_request is not None:
        payload["max_model_len"] = engine_request.max_seq_len
        payload["enable_chunked_prefill"] = bool(
            engine_request.enable_chunked_prefill and role != "decode"
        )
        if engine_request.nextn_accepted is not None:
            payload["aic_nextn_accepted"] = engine_request.nextn_accepted
        for name in (
            "enable_wideep",
            "enable_eplb",
            "wideep_num_slots",
            "moe_backend",
            "attention_backend",
        ):
            value = getattr(engine_request, name)
            if value not in (None, False):
                payload[f"aic_{name}"] = value
        for name, argument in (
            ("gemm_quant_mode", "aic_gemm_dtype"),
            ("moe_quant_mode", "aic_moe_dtype"),
            ("kvcache_quant_mode", "aic_kv_cache_dtype"),
            ("fmha_quant_mode", "aic_fmha_dtype"),
            ("comm_quant_mode", "aic_comm_dtype"),
        ):
            value = getattr(engine_request, name)
            if value is not None:
                payload[argument] = value
    if sample.get("startup_time") is not None:
        payload["startup_time"] = float(sample["startup_time"])
    return payload


def build_backend_deployment(
    sample: dict[str, Any],
    *,
    backend_version: str,
    engine_request: EngineRequestSpec | None = None,
) -> BackendDeploymentSpec:
    """Build the Dynamo-independent backend part of a :class:`ReplaySpec`."""
    mode = sample["deployment_mode"]
    common = {
        "deployment_mode": mode,
        "backend": sample["backend"],
        "backend_version": backend_version,
        "engine_request": engine_request,
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
    }
    if mode == "agg":
        return BackendDeploymentSpec(
            agg_engine_args=_engine_args_payload(
                sample,
                "agg",
                backend_version=backend_version,
                engine_request=engine_request,
            ),
            num_workers=int(sample["replicas"]),
            **common,
        )
    return BackendDeploymentSpec(
        prefill_engine_args=_engine_args_payload(
            sample,
            "prefill",
            backend_version=backend_version,
            engine_request=engine_request,
        ),
        decode_engine_args=_engine_args_payload(
            sample,
            "decode",
            backend_version=backend_version,
            engine_request=engine_request,
        ),
        num_prefill_workers=int(sample["prefill_replicas"]),
        num_decode_workers=int(sample["decode_replicas"]),
        **common,
    )
