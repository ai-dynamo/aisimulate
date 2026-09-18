# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Typed worker-configuration schema used by forward-pass predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

SUPPORTED_WORKER_CONFIG_SCHEMA_VERSIONS = frozenset({1})

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


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

    ``models/aic_config.py`` uses this mapping for AISim's ``EngineConfig``.

    Returns ``(None, None)`` for anything not published as a mixture-of-experts
    model, where "how are the experts sharded" has no answer to give.
    """

    if (model_kind or "").strip().lower() != "moe":
        return None, None
    width = tp_size or 1
    if expert_parallel_enabled:
        return 1, width
    return width, 1
