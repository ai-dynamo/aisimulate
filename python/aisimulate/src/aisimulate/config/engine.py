# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed engine input for prediction and recommendation."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import Choices, IntegerRange, NumericRange, StrictModel

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
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
    nextn: NonNegativeInt = Field(default=0, le=5)
    nextn_accepted: NonNegativeFloat | None = None
    enable_chunked_prefill: bool = False
    enable_wideep: bool = False
    enable_eplb: bool = False
    wideep_num_slots: PositiveInt | None = None
    moe_backend: Literal["deepep_moe", "megamoe"] | None = None
    attention_backend: Literal["flashinfer", "fa3"] | None = None
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    free_gpu_memory_fraction: Fraction | None = None
    workers: WorkersPredictionConfig
    kv_transfer: KvTransferConfig | None = None

    @field_validator("model", "hardware")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must be nonempty")
        return value

    @field_validator(
        "gemm_quant_mode",
        "moe_quant_mode",
        "kvcache_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
        mode="before",
    )
    @classmethod
    def _normalize_quant_mode(cls, value: Any) -> Any:
        return _normalize_engine_control_name(value)

    @field_validator("hardware")
    @classmethod
    def _reject_auto_hardware(cls, value: str) -> str:
        if value == "auto":
            raise ValueError("engine.hardware='auto' is recommendation-only")
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
        _validate_backend_block_sizes(backends={self.backend}, modes={self.mode}, workers=self.workers)
        _validate_engine_controls(self, backends={self.backend})
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

    @model_validator(mode="before")
    @classmethod
    def _validate_custom_preset_entries(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        preset = value.get("preset")
        if not isinstance(preset, list):
            return value
        required = {
            "replicas",
            "tensor",
            "pipeline",
            "attention_data",
            "moe_tensor",
            "moe_expert",
        }
        if not preset:
            raise ValueError("parallelism preset list must be nonempty")
        for index, entry in enumerate(preset):
            if not isinstance(entry, dict):
                raise ValueError(f"parallelism preset entry {index} must be a mapping")
            missing = required - set(entry)
            unknown = set(entry) - required
            if missing or unknown:
                raise ValueError(
                    "parallelism preset entries must cover exactly all knobs; "
                    f"missing={sorted(missing)}, unknown={sorted(unknown)}"
                )
        return value

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
    nextn: NonNegativeInt = Field(default=0, le=5)
    nextn_accepted: NonNegativeFloat | None = None
    enable_chunked_prefill: bool = False
    enable_wideep: bool = False
    enable_eplb: bool = False
    wideep_num_slots: PositiveInt | None = None
    moe_backend: Literal["deepep_moe", "megamoe"] | None = None
    attention_backend: Literal["flashinfer", "fa3"] | None = None
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    free_gpu_memory_fraction: Fraction | None = None
    workers: WorkersRecommendationConfig
    kv_transfer: KvTransferConfig | None = None

    @field_validator("model", "hardware")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must be nonempty")
        return value

    @field_validator(
        "gemm_quant_mode",
        "moe_quant_mode",
        "kvcache_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
        mode="before",
    )
    @classmethod
    def _normalize_quant_mode(cls, value: Any) -> Any:
        return _normalize_engine_control_name(value)

    @model_validator(mode="after")
    def _validate_roles(self) -> EngineRecommendationConfig:
        modes = set(self.mode.choices) if isinstance(self.mode, Choices) else {self.mode}
        _validate_worker_roles(
            modes=modes,
            workers=self.workers,
            has_transfer=self.kv_transfer is not None,
        )
        backends = set(self.backend.choices) if isinstance(self.backend, Choices) else {self.backend}
        _validate_backend_block_sizes(backends=backends, modes=modes, workers=self.workers)
        _validate_engine_controls(self, backends=backends)
        return self


def _normalize_engine_control_name(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("engine control names must be non-empty strings")
    return value.strip().lower()


def _validate_engine_controls(engine, *, backends: set[str]) -> None:
    if engine.nextn == 0:
        if engine.nextn_accepted is not None:
            raise ValueError("nextn_accepted requires nextn greater than zero")
    elif engine.nextn_accepted is None:
        raise ValueError(f"nextn={engine.nextn} requires explicit nextn_accepted")
    elif engine.nextn_accepted > engine.nextn:
        raise ValueError("nextn_accepted must be within [0, nextn]")
    if (engine.moe_backend is not None or engine.attention_backend is not None) and backends != {"sglang"}:
        raise ValueError("moe_backend and attention_backend require backend='sglang'")

    from aiconfigurator_core.sdk.common import (
        CommQuantMode,
        FMHAQuantMode,
        GEMMQuantMode,
        KVCacheQuantMode,
        MoEQuantMode,
    )

    for name, enum_type in (
        ("gemm_quant_mode", GEMMQuantMode),
        ("moe_quant_mode", MoEQuantMode),
        ("kvcache_quant_mode", KVCacheQuantMode),
        ("fmha_quant_mode", FMHAQuantMode),
        ("comm_quant_mode", CommQuantMode),
    ):
        value = getattr(engine, name)
        if value is not None and value not in enum_type.__members__:
            raise ValueError(f"{name} has unsupported value {value!r}")


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


def _validate_backend_block_sizes(*, backends: set[str], modes: set[str], workers) -> None:
    """Reject public domains with no backend-supported KV block size.

    The replay runtime accepts positive SGLang page sizes, while its vLLM-style
    schedulers (vLLM and TensorRT-LLM) require at least two tokens per block.
    Mixed backend domains may retain ``1`` because it is feasible for SGLang;
    the concrete prediction validation filters incompatible candidates.
    """

    if "sglang" in backends:
        return
    roles = []
    if "aggregated" in modes:
        roles.append("aggregated")
    if "disaggregated" in modes:
        roles.extend(("prefill", "decode"))
    for role in roles:
        worker = getattr(workers, role)
        if worker is None:
            continue
        value = worker.kv_cache.block_size
        if value is None:
            continue
        if isinstance(value, Choices):
            maximum = max(value.choices)
        elif isinstance(value, IntegerRange):
            maximum = value.range.max
        else:
            maximum = value
        if maximum < 2:
            raise ValueError(
                f"{role} KV block_size has no value supported by vLLM/TensorRT-LLM; "
                "those backends require block_size >= 2"
            )
