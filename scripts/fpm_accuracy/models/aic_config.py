# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Map Hugging Face configuration metadata to AISim's EngineConfig."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fpm_accuracy.exceptions import ConfigurationError
from fpm_accuracy.types.worker_config import WorkerConfigRecord, moe_mapping_from_expert_parallelism

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp8": "fp8",
    "fp8_static": "fp8_static",
    "fp8_block": "fp8_block",
    "nvfp4": "nvfp4",
    "int8": "int8",
    "int4": "int4",
    "w4afp8": "w4afp8",
    "w4a16_mxfp4": "w4a16_mxfp4",
    "w4a8_mxfp4_mxfp8": "w4a8_mxfp4_mxfp8",
}

AIC_ENGINE_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "model_name",
        "system_name",
        "systems_path",
        "backend",
        "backend_version",
        "tp_size",
        "pp_size",
        "attention_dp_size",
        "moe_tp_size",
        "moe_ep_size",
        "cp_size",
        "weight_dtype",
        "moe_dtype",
        "activation_dtype",
        "kv_cache_dtype",
        "kv_block_size",
        "nextn",
        "nextn_accept_rates",
        "extra",
    }
)

_REQUIRED_FIELDS = ("model_name", "system_name", "backend", "backend_version", "tp_size")
_POSITIVE_INTEGER_FIELDS = (
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "moe_tp_size",
    "moe_ep_size",
    "cp_size",
    "kv_block_size",
)


def map_worker_config_to_aic(
    record: WorkerConfigRecord,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map shared AISim engine fields; adapters enforce estimator capabilities."""

    unknown = set(overrides or {}) - AIC_ENGINE_CONFIG_FIELDS
    if unknown:
        raise ConfigurationError(f"unknown AISim EngineConfig override fields: {sorted(unknown)}")

    payload = record.config.model_dump(exclude_none=True, exclude_defaults=True)
    embedded = payload.get("aic_engine_config")
    embedded_is_complete = isinstance(embedded, Mapping) and all(name in embedded for name in _REQUIRED_FIELDS)
    if embedded_is_complete:
        config = dict(embedded)
    else:
        base = dict(payload)
        base.pop("aic_engine_config", None)
        config = base if _looks_like_engine_config(base) else _map_worker_schema_v1(base)
        if isinstance(embedded, Mapping):
            config.update(embedded)
    config.update(dict(overrides or {}))
    return _validate_engine_config(config, record.configuration_id)


def worker_role(record: WorkerConfigRecord) -> str:
    """Return the exact worker-role spelling required by AISim regression."""

    payload = record.config.model_dump(exclude_none=True, exclude_defaults=True)
    raw = _first(payload, "worker.role", "worker_role", "role", "worker_type")
    if raw is None:
        raise ConfigurationError(
            f"configuration {record.configuration_id!r} does not declare a worker role; "
            "add a Hugging Face override with prefill, decode, or aggregated"
        )
    normalized = str(raw).strip().lower()
    normalized = {"agg": "aggregated", "both": "aggregated"}.get(normalized, normalized)
    if normalized not in {"prefill", "decode", "aggregated"}:
        raise ConfigurationError(f"configuration {record.configuration_id!r} has unsupported worker role {raw!r}")
    return normalized


def _validate_engine_config(config: Mapping[str, Any], configuration_id: str) -> dict[str, Any]:
    result = dict(config)
    result.setdefault("schema_version", 1)
    result.setdefault("pp_size", 1)
    result.setdefault("attention_dp_size", 1)
    result.setdefault("extra", {})
    missing = [name for name in _REQUIRED_FIELDS if result.get(name) in (None, "")]
    if missing:
        raise ConfigurationError(
            f"configuration {configuration_id!r} cannot construct AISim EngineConfig; missing {', '.join(missing)}"
        )
    if result["schema_version"] != 1:
        raise ConfigurationError("AISim EngineConfig schema_version must be 1")
    if result["backend"] not in {"vllm", "sglang", "trtllm"}:
        raise ConfigurationError(f"unsupported AISim backend {result['backend']!r}")
    for name in ("weight_dtype", "moe_dtype", "activation_dtype", "kv_cache_dtype"):
        if name in result:
            result[name] = _normalize_dtype(result[name], field=name)
    for name in _POSITIVE_INTEGER_FIELDS:
        value = result.get(name)
        if value is not None and (isinstance(value, bool) or int(value) < 1):
            raise ConfigurationError(f"AISim EngineConfig {name} must be a positive integer")
        if value is not None:
            result[name] = int(value)
    nextn = result.get("nextn")
    if nextn is not None and (isinstance(nextn, bool) or int(nextn) < 0):
        raise ConfigurationError("AISim EngineConfig nextn must be a non-negative integer")
    if nextn is not None:
        result["nextn"] = int(nextn)
    rates = result.get("nextn_accept_rates")
    if rates is not None and any(not 0.0 <= float(rate) <= 1.0 for rate in rates):
        raise ConfigurationError("AISim EngineConfig nextn_accept_rates values must be in [0, 1]")
    if result.get("systems_path") is None:
        result.pop("systems_path", None)
    return result


def _map_worker_schema_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    weight_raw = _first(payload, "precision.weights", "precision.weight_dtype")
    kv_raw = _first(payload, "engine.kv_cache_dtype", "precision.kv_cache")
    expert_parallel = bool(_first(payload, "parallelism.expert_parallel_enabled") or False)
    tp_size = int(_first(payload, "parallelism.tensor_parallel_size", "parallelism.tp_size") or 1)
    model_kind = str(_first(payload, "model.kind") or "").lower()
    moe_tp_size, moe_ep_size = moe_mapping_from_expert_parallelism(model_kind, tp_size, expert_parallel)
    return {
        "schema_version": 1,
        "model_name": _first(payload, "model.id", "model.name"),
        "system_name": _normalize_system(_first(payload, "hardware.gpu_sku", "hardware.system_name")),
        "backend": str(_first(payload, "backend.name", "backend") or "").lower(),
        "backend_version": _first(payload, "backend.version", "software.backend_version"),
        "tp_size": tp_size,
        "pp_size": int(_first(payload, "parallelism.pipeline_parallel_size", "parallelism.pp_size") or 1),
        "attention_dp_size": int(
            _first(payload, "parallelism.attention_dp_size", "parallelism.attention_data_parallel_size") or 1
        ),
        "moe_tp_size": moe_tp_size,
        "moe_ep_size": moe_ep_size,
        "cp_size": _optional_int(_first(payload, "parallelism.context_parallel_size", "parallelism.cp_size")),
        "weight_dtype": _normalize_dtype(weight_raw),
        "moe_dtype": _normalize_dtype(_first(payload, "precision.experts", "precision.moe_weights")),
        "activation_dtype": _normalize_dtype(_first(payload, "precision.activations")),
        "kv_cache_dtype": _normalize_dtype(kv_raw, field="kv_cache_dtype"),
        "kv_block_size": _optional_int(_first(payload, "engine.kv_cache_block_size")),
        "nextn": _optional_int(_first(payload, "engine.nextn", "speculative.nextn")),
        "nextn_accept_rates": _first(payload, "engine.nextn_accept_rates", "speculative.nextn_accept_rates"),
        "extra": {},
    }


def _looks_like_engine_config(payload: Mapping[str, Any]) -> bool:
    return all(name in payload for name in ("model_name", "system_name", "backend", "tp_size"))


def _first(payload: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = payload
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return None


def _normalize_system(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "nvidia_h100_80gb": "h100_sxm",
        "nvidia_h100_sxm_80gb": "h100_sxm",
        "nvidia_h200_141gb": "h200_sxm",
        "nvidia_h200_sxm_141gb": "h200_sxm",
        "nvidia_b200": "b200_sxm",
    }.get(normalized, normalized)


def _normalize_dtype(value: Any, *, field: str | None = None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    # Match AISim's dynamo_ci KV-cache adapter; these are not weight aliases.
    if field == "kv_cache_dtype" and key in {"fp8_e4m3", "fp8_e5m2"}:
        return "fp8"
    normalized = _DTYPE_ALIASES.get(key)
    if normalized is None:
        raise ConfigurationError(f"unsupported AISim dtype {value!r}")
    return normalized


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


__all__ = ["AIC_ENGINE_CONFIG_FIELDS", "map_worker_config_to_aic", "worker_role"]
