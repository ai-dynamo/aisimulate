# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Typed worker-configuration schema used by forward-pass predictors."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

SUPPORTED_WORKER_CONFIG_SCHEMA_VERSIONS = frozenset({1})

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]

_INTEGER_STRING = re.compile(r"[+-]?\d+")
_FLOAT_STRING = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


class _WorkerConfigSection(BaseModel):
    """A versioned section with typed known fields and preserved extensions."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


class WorkerBackendConfig(_WorkerConfigSection):
    name: str | None = None
    version: str | None = None


class WorkerEngineConfig(_WorkerConfigSection):
    gpu_memory_utilization: FiniteFloat | None = None
    kv_cache_block_size: PositiveInt | None = None
    kv_cache_dtype: str | None = None
    max_model_length: PositiveInt | None = None
    max_num_batched_tokens: PositiveInt | None = None
    max_num_sequences: PositiveInt | None = None
    nextn: NonNegativeInt | None = None
    nextn_accept_rates: tuple[float, ...] | None = None
    prefix_caching_enabled: bool | None = None
    trust_remote_code: bool | None = None

    @field_validator("nextn_accept_rates", mode="before")
    @classmethod
    def _accept_json_array(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class WorkerHardwareConfig(_WorkerConfigSection):
    gpu_count: PositiveInt | None = None
    gpu_sku: str | None = None
    system_name: str | None = None


class WorkerModelConfig(_WorkerConfigSection):
    attention: str | None = None
    id: str | None = None
    kind: str | None = None
    name: str | None = None
    revision: str | None = None


class WorkerParallelismConfig(_WorkerConfigSection):
    attention_data_parallel_size: PositiveInt | None = None
    attention_dp_size: PositiveInt | None = None
    context_parallel_size: PositiveInt | None = None
    cp_size: PositiveInt | None = None
    expert_parallel_enabled: bool | None = None
    pipeline_parallel_size: PositiveInt | None = None
    pp_size: PositiveInt | None = None
    tensor_parallel_size: PositiveInt | None = None
    tp_size: PositiveInt | None = None


class WorkerPrecisionConfig(_WorkerConfigSection):
    activations: str | None = None
    experts: str | None = None
    kv_cache: str | None = None
    moe_weights: str | None = None
    weight_dtype: str | None = None
    weights: str | None = None


class WorkerSoftwareConfig(_WorkerConfigSection):
    backend_version: str | None = None
    dynamo_version: str | None = None


class WorkerRuntimeConfig(_WorkerConfigSection):
    role: str | None = None


class WorkerConfig(_WorkerConfigSection):
    """Schema-v1 worker configuration derived from an HF leaf manifest.

    The known fields mirror the current Dynamo worker identity consumed by
    AISim's ``EngineConfig`` adapter. Extra fields are retained because the
    source manifest contract is intentionally extensible.
    """

    backend: WorkerBackendConfig | str = Field(default_factory=WorkerBackendConfig)
    engine: WorkerEngineConfig = Field(default_factory=WorkerEngineConfig)
    hardware: WorkerHardwareConfig = Field(default_factory=WorkerHardwareConfig)
    model: WorkerModelConfig = Field(default_factory=WorkerModelConfig)
    parallelism: WorkerParallelismConfig = Field(default_factory=WorkerParallelismConfig)
    precision: WorkerPrecisionConfig = Field(default_factory=WorkerPrecisionConfig)
    software: WorkerSoftwareConfig = Field(default_factory=WorkerSoftwareConfig)
    worker: WorkerRuntimeConfig = Field(default_factory=WorkerRuntimeConfig)
    aic_engine_config: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class WorkerConfigRecord:
    configuration_id: str
    schema_version: int
    config: WorkerConfig


def moe_mapping_from_expert_parallelism(
    model_kind: str | None,
    tp_size: int | None,
    expert_parallel_enabled: bool | None,
) -> tuple[int | None, int | None]:
    """The ``(moe_tp_size, moe_ep_size)`` implied by ``expert_parallel_enabled``.

    ``tp_size`` is the ATTENTION mapping; the experts are sharded separately, and
    a backend that publishes only the flag still pins both widths. With expert
    parallel disabled the expert layers use the attention TP group; with it
    enabled those same ranks shard experts instead of tensor dimensions. Neither
    mode introduces ranks, so the width is ``tp_size`` either way.

    This is the single definition of that contract. ``models/aic_config.py`` maps
    it onto AISim's ``EngineConfig`` and :class:`WorkerConfigMetadata` falls back to
    it for display, so the label a reader sees describes the same decomposition
    the evaluation was actually run under. Do not fork the rule: a display-only
    copy that drifted would put a chart and its own numbers in disagreement.

    Returns ``(None, None)`` for anything not published as a mixture-of-experts
    model, where "how are the experts sharded" has no answer to give.
    """

    if (model_kind or "").strip().lower() != "moe":
        return None, None
    width = tp_size or 1
    if expert_parallel_enabled:
        return 1, width
    return width, 1


class WorkerConfigMetadata(BaseModel):
    """Normalized, presentation-safe identity and runtime metadata for one worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_id: str = Field(min_length=1)
    config_schema_version: PositiveInt
    model_id: str | None = None
    model_kind: str | None = None
    model_revision: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    dynamo_version: str | None = None
    gpu_sku: str | None = None
    gpu_count: PositiveInt | None = None
    tp_size: PositiveInt | None = None
    pp_size: PositiveInt | None = None
    attention_dp_size: PositiveInt | None = None
    moe_tp_size: PositiveInt | None = None
    moe_ep_size: PositiveInt | None = None
    cp_size: PositiveInt | None = None
    expert_parallel_enabled: bool | None = None
    weight_precision: str | None = None
    moe_precision: str | None = None
    activation_precision: str | None = None
    kv_cache_precision: str | None = None
    worker_role: str | None = None
    kv_cache_block_size: PositiveInt | None = None
    max_model_length: PositiveInt | None = None
    max_num_batched_tokens: PositiveInt | None = None
    max_num_sequences: PositiveInt | None = None
    gpu_memory_utilization: FiniteFloat | None = None
    prefix_caching_enabled: bool | None = None
    nextn: NonNegativeInt | None = None

    @classmethod
    def from_worker_config(cls, record: WorkerConfigRecord) -> WorkerConfigMetadata:
        """Resolve typed schema-v1 fields before consulting the embedded AISim fallback."""

        config = record.config
        embedded = config.aic_engine_config or {}
        flat = config.model_extra or {}
        backend = config.backend
        backend_name = backend if isinstance(backend, str) else backend.name
        backend_version = backend.version if isinstance(backend, WorkerBackendConfig) else None
        speculative = flat.get("speculative")
        speculative_nextn = (
            _optional_nonnegative_int(speculative.get("nextn")) if isinstance(speculative, Mapping) else None
        )

        # The expert mapping, resolved once so both widths come from one decision.
        # A published size always wins; the contract only fills a silence. In
        # practice the two agree — over the 35 configurations the dashboard
        # currently reads, all 20 that publish both sizes match the derivation
        # exactly — so this widens coverage rather than overriding anyone.
        model_kind = (
            config.model.kind or _optional_str(embedded.get("model_kind")) or _optional_str(flat.get("model_kind"))
        )
        tp_size = (
            config.parallelism.tensor_parallel_size
            or config.parallelism.tp_size
            or _optional_positive_int(embedded.get("tp_size"))
            or _optional_positive_int(flat.get("tp_size"))
        )
        expert_parallel_enabled = (
            config.parallelism.expert_parallel_enabled
            if config.parallelism.expert_parallel_enabled is not None
            else _first_optional_bool(embedded.get("expert_parallel_enabled"), flat.get("expert_parallel_enabled"))
        )
        derived_moe_tp_size, derived_moe_ep_size = moe_mapping_from_expert_parallelism(
            model_kind, tp_size, expert_parallel_enabled
        )

        return cls(
            configuration_id=record.configuration_id,
            config_schema_version=record.schema_version,
            model_id=(
                config.model.id
                or config.model.name
                or _optional_str(embedded.get("model_name"))
                or _optional_str(flat.get("model_name"))
            ),
            model_kind=model_kind,
            model_revision=(
                config.model.revision
                or _optional_str(embedded.get("model_revision"))
                or _optional_str(embedded.get("revision"))
                or _optional_str(flat.get("model_revision"))
                or _optional_str(flat.get("revision"))
            ),
            backend=backend_name or _optional_str(embedded.get("backend")) or _optional_str(flat.get("backend")),
            backend_version=(
                backend_version
                or config.software.backend_version
                or _optional_str(embedded.get("backend_version"))
                or _optional_str(flat.get("backend_version"))
            ),
            dynamo_version=(
                config.software.dynamo_version
                or _optional_str(embedded.get("dynamo_version"))
                or _optional_str(flat.get("dynamo_version"))
            ),
            gpu_sku=(
                config.hardware.gpu_sku
                or config.hardware.system_name
                or _optional_str(embedded.get("gpu_sku"))
                or _optional_str(embedded.get("system_name"))
                or _optional_str(flat.get("gpu_sku"))
                or _optional_str(flat.get("system_name"))
            ),
            gpu_count=(
                config.hardware.gpu_count
                or _optional_positive_int(embedded.get("gpu_count"))
                or _optional_positive_int(flat.get("gpu_count"))
            ),
            tp_size=tp_size,
            pp_size=(
                config.parallelism.pipeline_parallel_size
                or config.parallelism.pp_size
                or _optional_positive_int(embedded.get("pp_size"))
                or _optional_positive_int(flat.get("pp_size"))
            ),
            attention_dp_size=(
                config.parallelism.attention_dp_size
                or config.parallelism.attention_data_parallel_size
                or _optional_positive_int(embedded.get("attention_dp_size"))
                or _optional_positive_int(flat.get("attention_dp_size"))
            ),
            moe_tp_size=(
                _optional_positive_int(embedded.get("moe_tp_size"))
                or _optional_positive_int(flat.get("moe_tp_size"))
                or derived_moe_tp_size
            ),
            moe_ep_size=(
                _optional_positive_int(embedded.get("moe_ep_size"))
                or _optional_positive_int(flat.get("moe_ep_size"))
                or derived_moe_ep_size
            ),
            cp_size=(
                config.parallelism.context_parallel_size
                or config.parallelism.cp_size
                or _optional_positive_int(embedded.get("cp_size"))
                or _optional_positive_int(flat.get("cp_size"))
            ),
            expert_parallel_enabled=expert_parallel_enabled,
            weight_precision=(
                config.precision.weights
                or config.precision.weight_dtype
                or _optional_str(embedded.get("weight_dtype"))
                or _optional_str(flat.get("weight_dtype"))
            ),
            moe_precision=(
                config.precision.experts
                or config.precision.moe_weights
                or _optional_str(embedded.get("moe_dtype"))
                or _optional_str(flat.get("moe_dtype"))
            ),
            activation_precision=(
                config.precision.activations
                or _optional_str(embedded.get("activation_dtype"))
                or _optional_str(flat.get("activation_dtype"))
            ),
            kv_cache_precision=(
                config.engine.kv_cache_dtype
                or config.precision.kv_cache
                or _optional_str(embedded.get("kv_cache_dtype"))
                or _optional_str(flat.get("kv_cache_dtype"))
            ),
            worker_role=(
                config.worker.role
                or _optional_str(embedded.get("worker_role"))
                or _optional_str(embedded.get("role"))
                or _optional_str(flat.get("worker_role"))
                or _optional_str(flat.get("role"))
            ),
            kv_cache_block_size=(
                config.engine.kv_cache_block_size
                or _optional_positive_int(embedded.get("kv_cache_block_size"))
                or _optional_positive_int(embedded.get("kv_block_size"))
                or _optional_positive_int(flat.get("kv_cache_block_size"))
                or _optional_positive_int(flat.get("kv_block_size"))
            ),
            max_model_length=(
                config.engine.max_model_length
                or _optional_positive_int(embedded.get("max_model_length"))
                or _optional_positive_int(flat.get("max_model_length"))
            ),
            max_num_batched_tokens=(
                config.engine.max_num_batched_tokens
                or _optional_positive_int(embedded.get("max_num_batched_tokens"))
                or _optional_positive_int(flat.get("max_num_batched_tokens"))
            ),
            max_num_sequences=(
                config.engine.max_num_sequences
                or _optional_positive_int(embedded.get("max_num_sequences"))
                or _optional_positive_int(flat.get("max_num_sequences"))
            ),
            gpu_memory_utilization=(
                config.engine.gpu_memory_utilization
                if config.engine.gpu_memory_utilization is not None
                else _first_optional_float(embedded.get("gpu_memory_utilization"), flat.get("gpu_memory_utilization"))
            ),
            prefix_caching_enabled=(
                config.engine.prefix_caching_enabled
                if config.engine.prefix_caching_enabled is not None
                else _first_optional_bool(embedded.get("prefix_caching_enabled"), flat.get("prefix_caching_enabled"))
            ),
            nextn=(
                config.engine.nextn
                if config.engine.nextn is not None
                else _first_optional_nonnegative_int(embedded.get("nextn"), flat.get("nextn"), speculative_nextn)
            ),
        )


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    return result or None


def _optional_positive_int(value: Any) -> int | None:
    result = _optional_nonnegative_int(value)
    return result if result is not None and result > 0 else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and _INTEGER_STRING.fullmatch(value.strip()):
        try:
            result = int(value)
        except ValueError:
            return None
    else:
        return None
    return result if result >= 0 else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and _FLOAT_STRING.fullmatch(value.strip())):
        try:
            result = float(value)
        except (OverflowError, ValueError):
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _first_optional_bool(*values: Any) -> bool | None:
    for value in values:
        result = _optional_bool(value)
        if result is not None:
            return result
    return None


def _first_optional_float(*values: Any) -> float | None:
    for value in values:
        result = _optional_float(value)
        if result is not None:
            return result
    return None


def _first_optional_nonnegative_int(*values: Any) -> int | None:
    for value in values:
        result = _optional_nonnegative_int(value)
        if result is not None:
            return result
    return None
