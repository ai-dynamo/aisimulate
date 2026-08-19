# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an unrolled backend sample into a replay deployment specification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .replay import (
    BackendDeploymentSpec,
    DisaggregatedCorrectionSpec,
    EngineRequestSpec,
    EstimatorSpec,
)


def _role_prefix(role: str) -> str:
    """Field prefix in the unrolled sample for a role (empty for agg shape fields)."""
    return "" if role == "agg" else f"{role}_"


def _engine_args_payload(
    sample: dict[str, Any],
    role: str,
    *,
    backend_version: str,
    estimator: EstimatorSpec | None = None,
    engine_request: EngineRequestSpec | None = None,
) -> dict[str, Any]:
    """Build the runner-neutral engine argument payload for one role."""
    prefix = _role_prefix(role)
    tp = int(sample[f"{prefix}tp"])
    attention_dp = int(sample[f"{prefix}attention_dp"])
    moe_tp = int(sample[f"{prefix}moe_tp"])
    moe_ep = int(sample[f"{prefix}moe_ep"])
    pp = int(sample[f"{prefix}pp"])
    cp = int(sample[f"{prefix}cp"])
    role_request = engine_request.for_role(role) if engine_request is not None else None
    backend = role_request.backend if role_request is not None else sample["backend"]
    resolved_backend_version = (
        role_request.backend_version if role_request is not None else backend_version
    )
    memory_fraction_field = {
        "vllm": "gpu_memory_utilization",
        "sglang": "mem_fraction_static",
        "trtllm": "free_gpu_memory_fraction",
    }[backend]
    payload: dict[str, Any] = {
        "worker_type": "aggregated" if role == "agg" else role,
        "engine_type": backend,
        "aic_backend": backend,
        "aic_backend_version": resolved_backend_version,
        "aic_system": sample.get(f"{role}_hardware_sku", sample["hardware_sku"]),
        "aic_model_path": sample.get(f"{role}_model_name", sample["model_name"]),
        "aic_tp_size": tp,
        "aic_pp_size": pp,
        "aic_attention_dp_size": attention_dp,
        "aic_cp_size": cp,
        "max_num_batched_tokens": int(sample[f"{role}_max_num_batched_tokens"]),
        "max_num_seqs": int(sample[f"{role}_max_num_seqs"]),
        "block_size": int(sample[f"{role}_block_size"]),
        memory_fraction_field: float(
            role_request.memory_fraction
            if role_request is not None
            else engine_request.memory_fraction_by_role[role]
            if engine_request is not None
            else sample[f"{role}_gpu_memory_utilization"]
        ),
        "enable_prefix_caching": bool(sample[f"{role}_enable_prefix_caching"]),
    }
    if moe_tp * moe_ep > 1:
        payload["aic_moe_tp_size"] = moe_tp
        payload["aic_moe_ep_size"] = moe_ep
    nextn = role_request.nextn if role_request is not None else sample.get("aic_nextn")
    if nextn:
        payload["aic_nextn"] = int(nextn)
    if engine_request is not None:
        controls = role_request or engine_request
        payload["max_model_len"] = controls.max_seq_len
        payload["enable_chunked_prefill"] = bool(
            controls.enable_chunked_prefill and role != "decode"
        )
        if controls.nextn_accepted is not None:
            payload["aic_nextn_accepted"] = controls.nextn_accepted
        for name in (
            "enable_wideep",
            "enable_eplb",
            "wideep_num_slots",
            "moe_backend",
            "attention_backend",
        ):
            value = getattr(controls, name)
            if value not in (None, False):
                payload[f"aic_{name}"] = value
        for name, argument in (
            ("gemm_quant_mode", "aic_gemm_dtype"),
            ("moe_quant_mode", "aic_moe_dtype"),
            ("kvcache_quant_mode", "aic_kv_cache_dtype"),
            ("fmha_quant_mode", "aic_fmha_dtype"),
            ("comm_quant_mode", "aic_comm_dtype"),
        ):
            value = getattr(controls, name)
            if value is not None:
                payload[argument] = value
    if sample.get("startup_time") is not None:
        payload["startup_time"] = float(sample["startup_time"])
    if estimator is not None:
        payload.update(
            {
                "aic_database_mode": estimator.database_mode,
                "aic_transfer_policy": list(estimator.transfer_policy),
                "aic_forward_model": estimator.forward_model,
                "systems_path": estimator.performance_data_root,
            }
        )
    return payload


def build_backend_deployment(
    sample: dict[str, Any],
    *,
    backend_version: str,
    estimator: EstimatorSpec | None = None,
    engine_request: EngineRequestSpec | None = None,
    role_estimators: Mapping[str, EstimatorSpec] | None = None,
) -> BackendDeploymentSpec:
    """Build the Dynamo-independent backend part of a :class:`ReplaySpec`."""
    mode = sample["deployment_mode"]
    if estimator is not None and (
        estimator.backend != sample["backend"]
        or estimator.backend_version != backend_version
    ):
        raise ValueError(
            "estimator identity does not match the sampled backend/version: "
            f"{estimator.backend}/{estimator.backend_version} != "
            f"{sample['backend']}/{backend_version}"
        )
    common = {
        "deployment_mode": mode,
        "backend": sample["backend"],
        "backend_version": backend_version,
        "estimator": estimator,
        "engine_request": engine_request,
        "role_estimators": dict(role_estimators or {}),
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
                "cp",
                "strategy",
                "replicas",
                "prefill_tp",
                "prefill_pp",
                "prefill_attention_dp",
                "prefill_moe_tp",
                "prefill_moe_ep",
                "prefill_cp",
                "prefill_strategy",
                "prefill_replicas",
                "decode_tp",
                "decode_pp",
                "decode_attention_dp",
                "decode_moe_tp",
                "decode_moe_ep",
                "decode_cp",
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
                estimator=estimator,
                engine_request=engine_request,
            ),
            num_workers=int(sample["replicas"]),
            **common,
        )
    heterogeneous = bool(role_estimators)
    if heterogeneous:
        expected_roles = {"prefill", "decode"}
        if set(role_estimators or {}) != expected_roles:
            raise ValueError(
                f"role_estimators must contain exactly {sorted(expected_roles)}"
            )
        if (
            engine_request is None
            or set(engine_request.role_requests) != expected_roles
        ):
            raise ValueError(
                "heterogeneous disagg requires explicit prefill/decode engine requests"
            )
        for role in sorted(expected_roles):
            role_estimator = role_estimators[role]
            role_request = engine_request.role_requests[role]
            sampled_backend = str(sample[f"{role}_backend"])
            if role_estimator.backend != sampled_backend:
                raise ValueError(
                    f"{role} estimator backend {role_estimator.backend!r} does not "
                    f"match sampled backend {sampled_backend!r}"
                )
            if (
                role_request.role != role
                or role_request.backend != role_estimator.backend
                or role_request.backend_version != role_estimator.backend_version
            ):
                raise ValueError(
                    f"{role} engine request identity does not match its estimator: "
                    f"{role_request.backend}/{role_request.backend_version} != "
                    f"{role_estimator.backend}/{role_estimator.backend_version}"
                )
        common.update(
            prefill_backend=str(sample["prefill_backend"]),
            prefill_backend_version=role_estimators["prefill"].backend_version,
            decode_backend=str(sample["decode_backend"]),
            decode_backend_version=role_estimators["decode"].backend_version,
            disaggregated_corrections=DisaggregatedCorrectionSpec(
                prefill_rate_degradation=float(sample["prefill_rate_degradation"]),
                decode_rate_degradation=float(sample["decode_rate_degradation"]),
                prefill_latency_correction=float(sample["prefill_latency_correction"]),
                decode_latency_correction=float(sample["decode_latency_correction"]),
                ttft_correction_factor=float(sample["ttft_correction_factor"]),
            ),
        )
    return BackendDeploymentSpec(
        prefill_engine_args=_engine_args_payload(
            sample,
            "prefill",
            backend_version=(
                role_estimators["prefill"].backend_version
                if heterogeneous
                else backend_version
            ),
            estimator=(role_estimators["prefill"] if heterogeneous else estimator),
            engine_request=engine_request,
        ),
        decode_engine_args=_engine_args_payload(
            sample,
            "decode",
            backend_version=(
                role_estimators["decode"].backend_version
                if heterogeneous
                else backend_version
            ),
            estimator=(role_estimators["decode"] if heterogeneous else estimator),
            engine_request=engine_request,
        ),
        num_prefill_workers=int(sample["prefill_replicas"]),
        num_decode_workers=int(sample["decode_replicas"]),
        **common,
    )
