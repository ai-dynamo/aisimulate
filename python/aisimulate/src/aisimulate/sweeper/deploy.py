# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an unrolled backend sample into a replay deployment specification."""

from __future__ import annotations

from typing import Any

from .replay import BackendDeploymentSpec, EstimatorSpec


def _role_prefix(role: str) -> str:
    """Field prefix in the unrolled sample for a role (empty for agg shape fields)."""
    return "" if role == "agg" else f"{role}_"


def _engine_args_payload(
    sample: dict[str, Any],
    role: str,
    *,
    backend_version: str,
    estimator: EstimatorSpec | None = None,
) -> dict[str, Any]:
    """Build the runner-neutral engine argument payload for one role."""
    prefix = _role_prefix(role)
    tp = int(sample[f"{prefix}tp"])
    attention_dp = int(sample[f"{prefix}attention_dp"])
    moe_tp = int(sample[f"{prefix}moe_tp"])
    moe_ep = int(sample[f"{prefix}moe_ep"])
    pp = int(sample[f"{prefix}pp"])
    cp = int(sample[f"{prefix}cp"])
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
        "aic_pp_size": pp,
        "aic_attention_dp_size": attention_dp,
        "aic_cp_size": cp,
        "max_num_batched_tokens": int(sample[f"{role}_max_num_batched_tokens"]),
        "max_num_seqs": int(sample[f"{role}_max_num_seqs"]),
        "block_size": int(sample[f"{role}_block_size"]),
        memory_fraction_field: float(sample[f"{role}_gpu_memory_utilization"]),
        "enable_prefix_caching": bool(sample[f"{role}_enable_prefix_caching"]),
    }
    if moe_tp * moe_ep > 1:
        payload["aic_moe_tp_size"] = moe_tp
        payload["aic_moe_ep_size"] = moe_ep
    if sample.get("aic_nextn") is not None:
        payload["aic_nextn"] = int(sample["aic_nextn"])
    if sample.get("startup_time") is not None:
        payload["startup_time"] = float(sample["startup_time"])
    if estimator is not None:
        payload.update(
            {
                "aic_database_mode": estimator.database_mode,
                "aic_transfer_policy": list(estimator.transfer_policy),
                "aic_forward_model": estimator.forward_model,
                "aic_engine_step_backend": estimator.engine_step_backend,
                "aic_systems_paths": list(estimator.systems_paths),
                "aic_performance_data_version": estimator.performance_data_version,
            }
        )
    return payload


def build_backend_deployment(
    sample: dict[str, Any],
    *,
    backend_version: str,
    estimator: EstimatorSpec | None = None,
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
            ),
            num_workers=int(sample["replicas"]),
            **common,
        )
    return BackendDeploymentSpec(
        prefill_engine_args=_engine_args_payload(
            sample,
            "prefill",
            backend_version=backend_version,
            estimator=estimator,
        ),
        decode_engine_args=_engine_args_payload(
            sample,
            "decode",
            backend_version=backend_version,
            estimator=estimator,
        ),
        num_prefill_workers=int(sample["prefill_replicas"]),
        num_decode_workers=int(sample["decode_replicas"]),
        **common,
    )
