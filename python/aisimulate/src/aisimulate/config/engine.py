# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed engine input for prediction and recommendation."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import Choices, IntegerRange, NumericRange, StrictModel

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Fraction = Annotated[float, Field(gt=0, le=1)]
EngineMode = Literal["aggregated", "disaggregated"]
Backend = Literal["vllm", "sglang", "trtllm"]


class ParallelismPredictionConfig(StrictModel):
    replicas: PositiveInt = 1
    tensor: PositiveInt = 1
    pipeline: PositiveInt = 1
    attention_data: PositiveInt = 1
    moe_tensor: PositiveInt = 1
    moe_expert: PositiveInt = 1


class SchedulerPredictionConfig(StrictModel):
    max_batched_tokens: PositiveInt = 8192
    max_sequences: PositiveInt = 256


class KvCapacityPredictionConfig(StrictModel):
    type: Literal["default", "fixed"] = "default"
    memory_fraction: Fraction | None = None
    blocks: PositiveInt | None = None

    @model_validator(mode="after")
    def _validate_capacity(self) -> KvCapacityPredictionConfig:
        if self.type == "fixed":
            if self.blocks is None:
                raise ValueError("fixed KV capacity requires blocks")
            if self.memory_fraction is not None:
                raise ValueError("fixed KV capacity rejects memory_fraction")
        elif self.blocks is not None:
            raise ValueError("default KV capacity rejects blocks")
        return self


class KvCachePredictionConfig(StrictModel):
    block_size: PositiveInt | None = None
    prefix_caching: bool = True
    capacity: KvCapacityPredictionConfig = Field(default_factory=KvCapacityPredictionConfig)


class TimingConfig(StrictModel):
    type: Literal["default", "fixed", "polynomial"] = "default"
    prefill_ms: float | None = Field(default=None, ge=0.0)
    decode_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_timing(self) -> TimingConfig:
        if self.type == "fixed":
            if self.prefill_ms is None or self.decode_ms is None:
                raise ValueError("fixed timing requires prefill_ms and decode_ms")
        elif self.prefill_ms is not None or self.decode_ms is not None:
            raise ValueError(f"{self.type} timing rejects fixed timing values")
        return self


class WorkerPredictionConfig(StrictModel):
    parallelism: ParallelismPredictionConfig = Field(default_factory=ParallelismPredictionConfig)
    scheduler: SchedulerPredictionConfig = Field(default_factory=SchedulerPredictionConfig)
    kv_cache: KvCachePredictionConfig = Field(default_factory=KvCachePredictionConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    startup_seconds: float = Field(default=0.0, ge=0.0)


class WorkersPredictionConfig(StrictModel):
    aggregated: WorkerPredictionConfig | None = None
    prefill: WorkerPredictionConfig | None = None
    decode: WorkerPredictionConfig | None = None

    @model_validator(mode="after")
    def _apply_role_scheduler_defaults(self) -> WorkersPredictionConfig:
        for role, max_sequences in (
            ("aggregated", 256),
            ("prefill", 1),
            ("decode", 256),
        ):
            worker = getattr(self, role)
            if worker is None or "max_sequences" in worker.scheduler.model_fields_set:
                continue
            scheduler = worker.scheduler.model_copy(update={"max_sequences": max_sequences})
            setattr(self, role, worker.model_copy(update={"scheduler": scheduler}))
        return self


class KvTransferConfig(StrictModel):
    bytes_per_token: PositiveInt | Literal["auto"] = "auto"
    bandwidth_gb_per_second: PositiveFloat | None = None
    timing_mode: Literal["full_prompt", "destination_missing"] = "destination_missing"


class EnginePredictionConfig(StrictModel):
    mode: EngineMode = "aggregated"
    model: str
    hardware: str
    backend: Backend = "vllm"
    backend_version: str | None = None
    context_length: PositiveInt | Literal["max"] = "max"
    workers: WorkersPredictionConfig
    kv_transfer: KvTransferConfig | None = None

    @field_validator("model", "hardware")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must be nonempty")
        return value

    @model_validator(mode="after")
    def _validate_roles(self) -> EnginePredictionConfig:
        _validate_worker_roles(
            modes={self.mode},
            workers=self.workers,
            has_transfer=self.kv_transfer is not None,
        )
        if self.mode == "disaggregated" and self.backend == "trtllm":
            raise ValueError("TensorRT-LLM disaggregated mode is unsupported")
        return self


ParallelDomain = PositiveInt | Choices[PositiveInt] | IntegerRange


class ParallelismRecommendationConfig(StrictModel):
    preset: Literal["default", False] | list[ParallelismPredictionConfig] | dict[str, Any] = "default"
    replicas: ParallelDomain | None = None
    tensor: ParallelDomain | None = None
    pipeline: ParallelDomain | None = None
    attention_data: ParallelDomain | None = None
    moe_tensor: ParallelDomain | None = None
    moe_expert: ParallelDomain | None = None

    @model_validator(mode="after")
    def _validate_preset(self) -> ParallelismRecommendationConfig:
        if isinstance(self.preset, dict) and self.preset:
            raise ValueError("parallelism preset mapping must be empty to disable it")
        if isinstance(self.preset, list) and not self.preset:
            raise ValueError("parallelism preset list must be nonempty")
        independent = [
            name
            for name in (
                "replicas",
                "tensor",
                "pipeline",
                "attention_data",
                "moe_tensor",
                "moe_expert",
            )
            if getattr(self, name) is not None
        ]
        if self.preset not in (False, {}) and independent:
            raise ValueError(f"parallelism cannot combine preset with independent knobs {independent}")
        return self


class SchedulerRecommendationConfig(StrictModel):
    max_batched_tokens: PositiveInt | Choices[PositiveInt] | IntegerRange | None = None
    max_sequences: PositiveInt | Choices[PositiveInt] | IntegerRange | None = None


class KvCapacityRecommendationConfig(StrictModel):
    type: Literal["default", "fixed"] = "default"
    memory_fraction: Fraction | Choices[Fraction] | NumericRange | None = None
    blocks: PositiveInt | None = None

    @model_validator(mode="after")
    def _validate_capacity(self) -> KvCapacityRecommendationConfig:
        if isinstance(self.memory_fraction, NumericRange) and (
            self.memory_fraction.range.min <= 0 or self.memory_fraction.range.max > 1
        ):
            raise ValueError("memory_fraction range must stay within (0, 1]")
        if self.type == "fixed":
            if self.blocks is None:
                raise ValueError("fixed KV capacity requires blocks")
            if self.memory_fraction is not None:
                raise ValueError("fixed KV capacity rejects memory_fraction")
        elif self.blocks is not None:
            raise ValueError("default KV capacity rejects blocks")
        return self


class KvCacheRecommendationConfig(StrictModel):
    block_size: PositiveInt | Choices[PositiveInt] | IntegerRange | None = None
    prefix_caching: bool = True
    capacity: KvCapacityRecommendationConfig = Field(default_factory=KvCapacityRecommendationConfig)


class WorkerRecommendationConfig(StrictModel):
    parallelism: ParallelismRecommendationConfig = Field(default_factory=ParallelismRecommendationConfig)
    scheduler: SchedulerRecommendationConfig = Field(default_factory=SchedulerRecommendationConfig)
    kv_cache: KvCacheRecommendationConfig = Field(default_factory=KvCacheRecommendationConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    startup_seconds: float = Field(default=0.0, ge=0.0)


class WorkersRecommendationConfig(StrictModel):
    aggregated: WorkerRecommendationConfig | None = None
    prefill: WorkerRecommendationConfig | None = None
    decode: WorkerRecommendationConfig | None = None


class EngineRecommendationConfig(StrictModel):
    mode: EngineMode | Choices[EngineMode] = Field(
        default_factory=lambda: Choices[EngineMode](choices=["aggregated", "disaggregated"])
    )
    model: str
    hardware: str
    backend: Backend | Choices[Backend] = Field(default_factory=lambda: Choices[Backend](choices=["vllm", "sglang"]))
    backend_version: str | None = None
    context_length: PositiveInt | Literal["max"] = "max"
    workers: WorkersRecommendationConfig
    kv_transfer: KvTransferConfig | None = None

    @field_validator("model", "hardware")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must be nonempty")
        return value

    @model_validator(mode="after")
    def _validate_roles(self) -> EngineRecommendationConfig:
        modes = set(self.mode.choices) if isinstance(self.mode, Choices) else {self.mode}
        _validate_worker_roles(
            modes=modes,
            workers=self.workers,
            has_transfer=self.kv_transfer is not None,
        )
        return self


def _validate_worker_roles(*, modes: set[str], workers, has_transfer: bool) -> None:
    if "aggregated" in modes and workers.aggregated is None:
        raise ValueError("aggregated mode requires workers.aggregated")
    if "disaggregated" in modes and (workers.prefill is None or workers.decode is None):
        raise ValueError("disaggregated mode requires prefill and decode workers")
    if modes == {"aggregated"}:
        if workers.prefill is not None or workers.decode is not None:
            raise ValueError("aggregated mode rejects prefill/decode workers")
        if has_transfer:
            raise ValueError("aggregated mode rejects kv_transfer")
    if modes == {"disaggregated"} and workers.aggregated is not None:
        raise ValueError("disaggregated mode rejects aggregated workers")
