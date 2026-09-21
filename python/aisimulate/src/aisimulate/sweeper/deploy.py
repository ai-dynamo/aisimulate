# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an unrolled backend sample into a replay deployment specification."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..capacity import estimate_kv_bytes_per_token, materialize_aic_num_gpu_blocks
from ..config.common import ENGINE_MODEL_CONTROL_FIELDS, is_active_engine_model_control, omit_inactive_moe_controls
from ..config.engine import NgramSpeculationConfig
from .replay import BackendDeploymentSpec, EncoderPoolSpec, ForwardPassEstimatorSpec


def _role_prefix(role: str) -> str:
    """Field prefix in the unrolled sample for a role (empty for agg shape fields)."""
    return "" if role == "agg" else f"{role}_"


def _role_hardware_sku(sample: dict[str, Any], role: str) -> str:
    """Resolve a P/D override while preserving the shared-SKU fallback."""
    if role in {"prefill", "decode"}:
        return str(sample.get(f"{role}_hardware_sku") or sample["hardware_sku"])
    return str(sample["hardware_sku"])


def _performance_model_metadata(sample: dict[str, Any], role: str, *, backend_version: str) -> dict[str, Any]:
    """Keep optional perf-model identity separate from runtime timing args."""
    prefix = _role_prefix(role)
    moe_tp = int(sample[f"{prefix}moe_tp"])
    moe_ep = int(sample[f"{prefix}moe_ep"])
    config: dict[str, Any] = {
        "backend": sample["backend"],
        "backend_version": backend_version,
        "system": _role_hardware_sku(sample, role),
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
    if sample.get("speculation") is not None:
        config["speculation"] = NgramSpeculationConfig.model_validate(sample["speculation"]).cost_config()
    return {"provider": "aic", "config": config}


def _engine_args_payload(
    sample: dict[str, Any],
    role: str,
    *,
    backend_version: str,
    forward_pass_estimator: ForwardPassEstimatorSpec | None = None,
) -> dict[str, Any]:
    """Build the runner-neutral engine argument payload for one role."""
    if any(is_active_engine_model_control(name, sample.get(name)) for name in ENGINE_MODEL_CONTROL_FIELDS) and (
        forward_pass_estimator is None or sample.get(f"{role}_timing_model") is not None
    ):
        raise ValueError("engine model controls require a resolved canonical forward-pass estimator for every role")
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
        "aic_system": _role_hardware_sku(sample, role),
        "aic_model_path": sample["model_name"],
        "aic_tp_size": tp,
        "aic_attention_dp_size": attention_dp,
        "max_num_batched_tokens": int(sample[f"{role}_max_num_batched_tokens"]),
        "max_num_seqs": int(sample[f"{role}_max_num_seqs"]),
        "block_size": int(block_size),
        memory_fraction_field: float(memory_fraction),
        "enable_prefix_caching": bool(sample[f"{role}_enable_prefix_caching"]),
    }
    if sample.get("context_length") is not None:
        payload["max_model_len"] = int(sample["context_length"])
    if moe_tp * moe_ep > 1:
        payload["aic_moe_tp_size"] = moe_tp
        payload["aic_moe_ep_size"] = moe_ep
    if sample.get("speculation") is not None:
        payload["speculation"] = dict(sample["speculation"])
    if sample.get("aic_nextn"):
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
    if forward_pass_estimator is not None and sample.get(f"{role}_timing_model") is None:
        payload["timing_model"] = {
            "type": "external",
            "provider": "aic",
            "config": omit_inactive_moe_controls(forward_pass_estimator.config),
        }
        if memory_fraction_field in payload:
            payload["timing_model"]["config"][memory_fraction_field] = payload[memory_fraction_field]
        payload["tensor_parallel_size"] = tp
        payload["dp_size"] = attention_dp
        if forward_pass_estimator.performance_data_root:
            payload["systems_path"] = forward_pass_estimator.performance_data_root
        for key in tuple(payload):
            if key.startswith("aic_") and key != "aic_nextn":
                payload.pop(key)
    if sample.get("enable_chunked_prefill") is not None and role != "decode":
        payload["enable_chunked_prefill"] = sample["enable_chunked_prefill"]
    if sample.get("nextn_accepted") is not None:
        payload["aic_nextn_accepted"] = sample["nextn_accepted"]
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
                **({"kvcache_quant_mode": sample["kvcache_quant_mode"]} if sample.get("kvcache_quant_mode") else {}),
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
                **({"kvcache_quant_mode": sample["kvcache_quant_mode"]} if sample.get("kvcache_quant_mode") else {}),
            )
            if transfer_geometry == "auto"
            else int(transfer_geometry)
        )
    if host_offload is not None:
        payload["native_host_offload"] = dict(host_offload)
    if role in {"prefill", "decode"}:
        if sample.get("kv_transfer_bandwidth") is not None:
            payload["kv_transfer_bandwidth"] = float(sample["kv_transfer_bandwidth"])
        if sample.get("kv_transfer_timing_mode") is not None:
            payload["kv_transfer_timing_mode"] = sample["kv_transfer_timing_mode"]
    if role == "agg" and backend == "sglang":
        payload.update(_native_vl_engine_args(sample))
    return payload


def _native_vl_engine_args(sample: dict[str, Any]) -> dict[str, Any]:
    """Mirror the compiler's SGLang host-aware lowering for a sweeper candidate."""
    host_loop = bool(sample.get("agg_host_loop"))
    vision = sample.get("agg_vision")
    max_prefill = sample.get("agg_max_prefill_tokens")
    result: dict[str, Any] = {}
    sglang: dict[str, Any] = {}
    if host_loop or vision is not None or max_prefill is not None:
        sglang["chunked_prefill_size"] = int(sample["agg_max_num_batched_tokens"])
    if max_prefill is not None:
        sglang["max_prefill_tokens"] = int(max_prefill)
    if vision is not None:
        sglang["vlm_cache_bytes"] = int(vision.get("cache_mib", 100)) << 20
        result["vision"] = True
    if host_loop:
        sglang["host_loop"] = True
    if sglang:
        result["sglang"] = sglang
    if sample.get("agg_frontend") is not None:
        result["frontend"] = deepcopy(sample["agg_frontend"])
    return result


def build_backend_deployment(
    sample: dict[str, Any],
    *,
    backend_version: str,
    encoder: EncoderPoolSpec | None = None,
    forward_pass_estimators: Mapping[str, ForwardPassEstimatorSpec] | None = None,
) -> BackendDeploymentSpec:
    """Build the Dynamo-independent backend part of a :class:`ReplaySpec`."""
    forward_pass_estimators = dict(forward_pass_estimators or {})
    mode = sample["deployment_mode"]
    if encoder is not None and mode not in {"agg", "disagg"}:
        raise ValueError("analytical EPD supports only agg/disagg language deployments; AFD is unsupported")
    if mode in {"afd", "afd+pd"}:
        parallel_config = {
            "afd": sample["afd"],
            "afd_provenance": sample["afd_provenance"],
        }
        performance_model_metadata = {
            "afd": {
                "provider": "unresolved",
                "measurement_required": True,
                "config": sample["afd"],
                "provenance": sample["afd_provenance"],
            }
        }
        afd_common = {
            "deployment_mode": mode,
            "backend": sample["backend"],
            "backend_version": backend_version,
            "parallel_config": parallel_config,
            "performance_model_metadata": performance_model_metadata,
        }
        if mode == "afd":
            return BackendDeploymentSpec(**afd_common)

        companion_role = sample["afd_companion_role"]
        if companion_role not in {"prefill", "decode"}:
            raise ValueError("afd+pd requires one prefill or decode companion")
        prefix = f"{companion_role}_"
        for suffix in ("tp", "pp", "attention_dp", "moe_tp", "moe_ep", "strategy", "replicas"):
            parallel_config[f"{prefix}{suffix}"] = sample[f"{prefix}{suffix}"]
        performance_model_metadata[companion_role] = _performance_model_metadata(
            sample, companion_role, backend_version=backend_version
        )
        companion_args = _engine_args_payload(sample, companion_role, backend_version=backend_version)
        if companion_role == "prefill":
            return BackendDeploymentSpec(
                prefill_engine_args=companion_args,
                num_prefill_workers=int(sample["prefill_replicas"]),
                **afd_common,
            )
        return BackendDeploymentSpec(
            decode_engine_args=companion_args,
            num_decode_workers=int(sample["decode_replicas"]),
            **afd_common,
        )

    common = {
        "forward_pass_estimators": forward_pass_estimators,
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
                "prefill_hardware_sku",
                "prefill_tp",
                "prefill_pp",
                "prefill_attention_dp",
                "prefill_moe_tp",
                "prefill_moe_ep",
                "prefill_strategy",
                "prefill_replicas",
                "decode_hardware_sku",
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
    for role, estimator in forward_pass_estimators.items():
        common["performance_model_metadata"]["aggregated" if role == "agg" else role] = {
            "provider": "aic",
            "config": deepcopy(estimator.config),
            "selection": deepcopy(estimator.diagnostics),
        }
    if mode == "agg" and sample.get("agg_host_profile_digest") is not None:
        common["performance_model_metadata"]["aggregated"]["vl"] = {
            "host_profile_digest": sample["agg_host_profile_digest"],
        }
    if mode == "agg":
        return BackendDeploymentSpec(
            agg_engine_args=_engine_args_payload(
                sample,
                "agg",
                backend_version=backend_version,
                forward_pass_estimator=forward_pass_estimators.get("agg"),
            ),
            num_workers=int(sample["replicas"]),
            **common,
        )
    prefill_args = _engine_args_payload(
        sample,
        "prefill",
        backend_version=backend_version,
        forward_pass_estimator=forward_pass_estimators.get("prefill"),
    )
    decode_args = _engine_args_payload(
        sample, "decode", backend_version=backend_version, forward_pass_estimator=forward_pass_estimators.get("decode")
    )
    return BackendDeploymentSpec(
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        num_prefill_workers=int(sample["prefill_replicas"]),
        num_decode_workers=int(sample["decode_replicas"]),
        **common,
    )
