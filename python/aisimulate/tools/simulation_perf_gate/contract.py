# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol-v4 input identity and numeric validation for native timing."""

import math
from pathlib import Path

MODEL_FIELDS = (
    "model",
    "system",
    "backend",
    "backend_version",
    "worker_type",
    "tp",
    "pp",
    "attention_dp",
    "moe_tp_size",
    "moe_ep_size",
    "attention_backend",
    "gemm_quant_mode",
    "moe_quant_mode",
    "kvcache_quant_mode",
    "fmha_quant_mode",
    "fpm_fmha_quant_mode",
    "comm_quant_mode",
    "kv_block_size",
)


def check_finite(value: object) -> None:
    """Check every number, including diagnostics, without encoding another JSON copy."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number in worker data")
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)


def check_models(response: dict, case: dict) -> None:
    """Require comparable real-model settings for every worker role."""
    engine = case["config"]["engine"]
    roles = {"prefill", "decode"} if engine["mode"] == "disaggregated" else {"aggregated"}
    identity = model_identity(response.get("model_identity"))
    provenance = response.get("model_provenance")
    if set(identity) != roles or not isinstance(provenance, dict) or set(provenance) != roles:
        raise ValueError("missing role-specific model provenance or identity")
    for role in roles:
        model = provenance[role]
        projected = fields(model, MODEL_FIELDS, f"/model_provenance/{role}")
        if projected != {key: identity[role][key] for key in MODEL_FIELDS}:
            raise ValueError(f"{role}: model provenance differs from identity")
        worker = engine["workers"][role]
        parallelism = worker["parallelism"]
        expected = {
            "provider": "aic",
            "estimation_mode": "op_level",
            "fallback_policy": "deny",
            "database_mode": "SILICON",
            "enable_shared_layer": True,
            "worker_type": role,
            "tp": parallelism["tensor"],
            "pp": parallelism["pipeline"],
            "attention_dp": parallelism["attention_data"],
            "kv_block_size": worker["kv_cache"]["block_size"],
            "model": engine["model"],
            "system": engine["hardware"],
            "backend": engine["backend"],
            "backend_version": engine["backend_version"],
        }
        if any(type(model.get(key)) is not type(value) or model[key] != value for key, value in expected.items()):
            raise ValueError(f"{role}: invalid real-model provenance")
        paths = model.get("systems_paths")
        if (
            identity[role]["systems_paths"] != ["package:aisimulate_core/systems"]
            or not isinstance(paths, list)
            or not paths
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths)
        ):
            raise ValueError(f"{role}: missing packaged model provenance")


def fields(value: dict, required: tuple[str, ...], path: str, optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    missing = sorted(set(required) - value.keys())
    if missing:
        raise ValueError(f"{path}: missing fields {', '.join(missing)}")
    return {**{key: value[key] for key in required}, **{key: value.get(key) for key in optional}}


def check_counts(value: dict, names: tuple[str, ...], path: str, *, nullable: bool = False) -> None:
    for name in names:
        count = value[name]
        if nullable and count is None:
            continue
        if type(count) is not int or count < 0:
            raise ValueError(f"{path}/{name}: invalid count")


def model_identity(value: dict) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError("missing model identity")
    result = {
        role: fields(model, (*MODEL_FIELDS, "systems_paths"), f"/model_identity/{role}")
        for role, model in value.items()
    }
    for role, model in result.items():
        for name in ("model", "system", "backend", "backend_version", "worker_type"):
            if not isinstance(model[name], str) or not model[name]:
                raise ValueError(f"/model_identity/{role}/{name}: invalid identity")
        check_counts(model, ("tp", "pp", "attention_dp", "kv_block_size"), f"/model_identity/{role}")
        check_counts(model, ("moe_tp_size", "moe_ep_size"), f"/model_identity/{role}", nullable=True)
    return result
