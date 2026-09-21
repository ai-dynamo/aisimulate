# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed engine input for prediction and recommendation."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from .common import (
    ENGINE_MODEL_CONTROL_FIELDS,
    Choices,
    IntegerRange,
    NumericRange,
    StrictModel,
    is_active_engine_model_control,
)

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
CudaGraphReservedBytes = Annotated[int, Field(strict=True, ge=0, le=1 << 53)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
EngineMode = Literal["aggregated", "disaggregated", "afd"]
Backend = Literal["vllm", "sglang", "trtllm"]
KvBytesPerToken = PositiveInt | Literal["auto"]
AFDPhase = Literal["prefill", "decode", "both"]
AFDPipelineModel = Literal["optimistic", "conservative", "serial"]
AFDExpertParallel = PositiveInt | Literal["n_f_nodes", "ffn_tp"]


class AFDTopologyPredictionConfig(StrictModel):
    """One concrete attention/FFN-disaggregated topology."""

    phase: AFDPhase
    combined_with_pd: bool = Field(strict=True)
    n_a_nodes: PositiveInt
    n_f_nodes: PositiveInt
    tp_a: PositiveInt
    a_batch_size: PositiveInt
    f_moe_ep_size: PositiveInt = 1
    num_microbatches: PositiveInt = 3
    pipeline_model: AFDPipelineModel = "optimistic"
    comm_overhead_factor: PositiveFloat = 1.0
    boundary_on_attn: bool = Field(default=True, strict=True)


class AFDSearchRecommendationConfig(StrictModel):
    """Finite AFD topology domain for a recommendation run."""

    phase: AFDPhase
    combined_with_pd: bool = Field(strict=True)
    # This is deliberately required. AFD batches must be memory-qualified for
    # the target model/hardware rather than inherited from an implicit default.
    a_batch_size: PositiveInt | Choices[PositiveInt] | IntegerRange
    tp_a: PositiveInt | Choices[PositiveInt] | IntegerRange | None = None
    f_moe_ep_size: AFDExpertParallel | Choices[AFDExpertParallel] | None = None
    num_microbatches: PositiveInt | Choices[PositiveInt] | IntegerRange = Field(
        default_factory=lambda: Choices[PositiveInt](choices=[2, 3, 4])
    )
    pipeline_model: AFDPipelineModel | Choices[AFDPipelineModel] = Field(
        default_factory=lambda: Choices[AFDPipelineModel](choices=["optimistic", "conservative"])
    )
    comm_overhead_factor: PositiveFloat = 1.0
    boundary_on_attn: bool = Field(default=True, strict=True)
    max_af_ratio: PositiveFloat = 4.0
    max_candidates: PositiveInt = 10_000


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
    prefill_schedule_interval: PositiveInt = Field(
        default=1, description="vLLM only: admit prefill once every N attention-DP group passes."
    )
    prefill_decode_interval: NonNegativeInt = Field(
        default=0,
        description=(
            "SGLang only: block new prefill and chunk continuation for N scheduler rounds after each EXTEND. "
            "Rounds may be idle and do not count output tokens; zero disables the interval."
        ),
    )


class KvCapacityPredictionConfig(StrictModel):
    type: Literal["default", "fixed"] = "default"
    memory_fraction: Fraction | None = None
    blocks: PositiveInt | None = None
    cuda_graph_reserved_bytes: CudaGraphReservedBytes = 0

    @model_validator(mode="after")
    def _validate_capacity(self) -> KvCapacityPredictionConfig:
        if self.type == "fixed":
            if self.blocks is None:
                raise ValueError("fixed KV capacity requires blocks")
            if self.memory_fraction is not None:
                raise ValueError("fixed KV capacity rejects memory_fraction")
            if self.cuda_graph_reserved_bytes != 0:
                raise ValueError("fixed KV capacity rejects cuda_graph_reserved_bytes")
        elif self.blocks is not None:
            raise ValueError("default KV capacity rejects blocks")
        return self


class HostOffloadConfig(StrictModel):
    num_host_blocks: PositiveInt
    d2h_bandwidth_gbps: NonNegativeFloat = 32.0
    h2d_bandwidth_gbps: NonNegativeFloat = 32.0


class G3OffloadConfig(StrictModel):
    scope: Literal["worker_local", "cluster_shared"]
    num_g3_blocks: PositiveInt
    latency_to_first_byte_ms: NonNegativeFloat = 0.1
    read_bandwidth_gbps: NonNegativeFloat = 10.0
    write_bandwidth_gbps: NonNegativeFloat = 10.0
    shared_read_bandwidth_gbps: NonNegativeFloat = 80.0
    shared_write_bandwidth_gbps: NonNegativeFloat = 80.0


class KvCachePredictionConfig(StrictModel):
    block_size: PositiveInt | None = None
    prefix_caching: bool = True
    bytes_per_token: KvBytesPerToken = "auto"
    capacity: KvCapacityPredictionConfig = Field(default_factory=KvCapacityPredictionConfig)
    host_offload: HostOffloadConfig | None = None
    g3_offload: G3OffloadConfig | None = None

    @model_validator(mode="after")
    def _validate_g3(self):
        if self.g3_offload is not None and self.host_offload is None:
            raise ValueError("g3_offload requires host_offload")
        return self


class NgramSpeculationConfig(StrictModel):
    """Prompt-lookup cost and explicit workload acceptance assumptions."""

    kind: Literal["ngram"]
    num_speculative_tokens: Annotated[int, Field(strict=True, ge=1, le=5)]
    acceptance_rates: list[Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]]
    seed: Annotated[int, Field(strict=True, ge=0, le=0xFFFF_FFFF_FFFF_FFFF)] = 42

    @model_validator(mode="after")
    def _validate_rates(self) -> NgramSpeculationConfig:
        if len(self.acceptance_rates) != self.num_speculative_tokens:
            raise ValueError("acceptance_rates must contain one conditional probability per speculative token")
        return self

    def cost_config(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": {"num_speculative_tokens": self.num_speculative_tokens}}


class TimingConfig(StrictModel):
    type: Literal["default", "fixed", "polynomial"] = "default"
    forward_model: Literal["op_level", "fpm"] = Field(default="op_level", exclude=True)
    fpm_parquet_path: str | None = None
    estimation_mode: Literal["auto", "op_level", "fpm_interpolation", "fpm_regression"] | None = None
    fallback_policy: Literal["deny", "allow"] | None = None
    estimator_config: dict[str, Any] | None = None
    systems_paths: list[str] | None = Field(default=None, min_length=1)
    database_mode: Literal["SILICON", "HYBRID", "EMPIRICAL", "SOL"] | None = None
    transfer_policy: str | list[str] | None = None

    @field_validator("database_mode", mode="before")
    @classmethod
    def _normalize_database_mode(cls, value):
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="before")
    @classmethod
    def _preserve_legacy_estimator_selection(cls, value):
        if isinstance(value, dict) and value.get("type", "default") == "default" and "forward_model" in value:
            value = dict(value)
            legacy = {"op_level": "op_level", "fpm": "fpm_interpolation"}.get(value["forward_model"])
            if legacy is not None and "estimation_mode" not in value:
                value["estimation_mode"] = legacy
                value.setdefault("fallback_policy", "deny")
        return value

    prefill_ms: float | None = Field(default=None, ge=0.0)
    decode_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_timing(self) -> TimingConfig:
        if self.estimation_mode == "fpm_interpolation":
            self.forward_model = "fpm"
        elif self.estimation_mode == "op_level":
            self.forward_model = "op_level"

        if self.type == "fixed":
            if self.prefill_ms is None or self.decode_ms is None:
                raise ValueError("fixed timing requires prefill_ms and decode_ms")
        elif self.prefill_ms is not None or self.decode_ms is not None:
            raise ValueError(f"{self.type} timing rejects fixed timing values")
        if self.type != "default" and self.forward_model != "op_level":
            raise ValueError(
                f"{self.type} timing rejects forward_model={self.forward_model!r}; "
                "forward_model applies to default timing only"
            )
        if self.fpm_parquet_path is not None:
            if not self.fpm_parquet_path:
                raise ValueError("fpm_parquet_path cannot be empty")
            if self.type != "default" or self.forward_model != "fpm":
                raise ValueError("fpm_parquet_path requires default timing with forward_model='fpm'")
        return self


class WorkerPredictionConfig(StrictModel):
    hardware: str | None = Field(default=None, min_length=1)
    parallelism: ParallelismPredictionConfig = Field(default_factory=ParallelismPredictionConfig)
    scheduler: SchedulerPredictionConfig = Field(default_factory=SchedulerPredictionConfig)
    kv_cache: KvCachePredictionConfig = Field(default_factory=KvCachePredictionConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    startup_seconds: float = Field(default=0.0, ge=0.0)


class EncoderPredictionConfig(StrictModel):
    """Dedicated analytical encoder pool, not an event-level language worker."""

    hardware: str | None = Field(default=None, min_length=1)
    backend_version: str | None = Field(default=None, min_length=1)
    tensor: PositiveInt = 1
    replicas: PositiveInt = 1
    batch_size: Annotated[int, Field(strict=True, gt=0, le=8)] = 1
    latency_correction: PositiveFloat = 1.0
    rate_degradation: Fraction = 0.9


class WorkersPredictionConfig(StrictModel):
    encoder: EncoderPredictionConfig | None = None
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
    bytes_per_token: KvBytesPerToken = "auto"
    bandwidth_gb_per_second: PositiveFloat | None = None
    timing_mode: Literal["full_prompt", "destination_missing"] = "destination_missing"


class EstimatorPolicyConfig(StrictModel):
    nextn: NonNegativeInt = Field(default=0, le=5)
    nextn_accepted: NonNegativeFloat | None = None
    enable_chunked_prefill: bool | None = Field(default=None, strict=True)
    enable_eplb: bool = Field(default=False, strict=True)
    wideep_num_slots: PositiveInt | None = None
    moe_backend: str | None = None
    attention_backend: str | None = None
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None

    @model_validator(mode="after")
    def _validate_engine_controls(self):
        if self.nextn and self.nextn_accepted is None:
            raise ValueError("nextn requires explicit nextn_accepted")
        if self.nextn_accepted is not None and (not self.nextn or self.nextn_accepted > self.nextn):
            raise ValueError("nextn_accepted requires nextn > 0 and must be within [0, nextn]")
        active = self.nextn or any(
            is_active_engine_model_control(name, getattr(self, name)) for name in ENGINE_MODEL_CONTROL_FIELDS
        )
        mode = getattr(self, "mode", "aggregated")
        modes = mode.choices if hasattr(mode, "choices") else [mode]
        if "afd" in modes and self.enable_chunked_prefill is not None:
            raise ValueError("enable_chunked_prefill is unsupported for AFD")
        workers = getattr(self, "workers", None)
        roles = [getattr(workers, role, None) for role in ("aggregated", "prefill", "decode")]
        if active and (
            "afd" in modes
            or getattr(workers, "encoder", None) is not None
            or any(worker is not None and worker.timing.type != "default" for worker in roles)
        ):
            raise ValueError("engine model controls require default timing in every regular language role")
        return self

    database_mode: Literal["SILICON", "HYBRID", "EMPIRICAL", "SOL"] = "SILICON"
    transfer_policy: str | list[str] | None = None
    systems_paths: list[str] | None = None
    estimation_mode: Literal["auto", "op_level", "fpm_interpolation", "fpm_regression"] = "auto"
    fallback_policy: Literal["deny", "allow"] = "deny"
    estimator_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("database_mode", mode="before")
    @classmethod
    def _normalize_database_mode(cls, value):
        return value.upper() if isinstance(value, str) else value

    @field_validator("systems_paths")
    @classmethod
    def _nonempty_system_roots(cls, value):
        if value is None:
            return value
        if not value or any(not path.strip() for path in value):
            raise ValueError("systems_paths must contain at least one nonempty root")
        return value

    @model_validator(mode="after")
    def _supported_estimator_policies(self):
        mode = getattr(self, "mode", "aggregated")
        modes = mode.choices if hasattr(mode, "choices") else [mode]
        workers = getattr(self, "workers", None)
        custom_policy = (
            self.database_mode != "SILICON"
            or self.transfer_policy is not None
            or self.systems_paths not in (None, ["default"])
            or self.estimation_mode != "auto"
            or self.fallback_policy != "deny"
            or bool(self.estimator_config)
            or getattr(self, "decoder_replay", False)
            or getattr(self, "enable_shared_layer", None) is not None
            or getattr(self, "strict_provenance", None) is not None
        )
        roles = [getattr(workers, role, None) for role in ("aggregated", "prefill", "decode")]
        unsupported_provider = "afd" in modes or getattr(workers, "encoder", None) is not None
        if unsupported_provider and any(
            worker is not None
            and (
                bool(worker.timing.estimator_config)
                or worker.timing.systems_paths is not None
                or worker.timing.database_mode is not None
                or worker.timing.transfer_policy is not None
                or worker.timing.fallback_policy == "allow"
                or worker.timing.estimation_mode not in {None, "op_level", "fpm_interpolation"}
            )
            for worker in roles
        ):
            raise ValueError("estimator policies require regular language workers with default timing")
        if custom_policy and (
            "afd" in modes
            or getattr(workers, "encoder", None) is not None
            or any(worker is not None and worker.timing.type != "default" for worker in roles)
        ):
            raise ValueError("estimator policies require regular language workers with default timing in every role")
        for worker in roles:
            if (
                worker is not None
                and worker.timing.type != "default"
                and any(
                    getattr(worker.timing, name) is not None
                    for name in (
                        "estimation_mode",
                        "fallback_policy",
                        "estimator_config",
                        "systems_paths",
                        "database_mode",
                        "transfer_policy",
                    )
                )
            ):
                raise ValueError("estimator settings require default timing")
        return self


class EnginePredictionConfig(EstimatorPolicyConfig):
    mode: EngineMode = "aggregated"
    model: str
    hardware: str
    backend: Backend = "vllm"
    backend_version: str | None = None
    decoder_replay: StrictBool = False
    enable_shared_layer: StrictBool | None = None
    strict_provenance: StrictBool | None = None
    context_length: PositiveInt | Literal["max"] = "max"
    speculation: NgramSpeculationConfig | None = None
    workers: WorkersPredictionConfig = Field(default_factory=WorkersPredictionConfig)
    kv_transfer: KvTransferConfig | None = None
    afd: AFDTopologyPredictionConfig | None = None

    @field_validator("model", "hardware")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must be nonempty")
        return value

    @field_validator("hardware")
    @classmethod
    def _reject_auto_hardware(cls, value: str) -> str:
        if value == "auto":
            raise ValueError("engine.hardware='auto' is recommendation-only")
        return value

    @model_validator(mode="after")
    def _validate_roles(self) -> EnginePredictionConfig:
        if self.decoder_replay:
            # Keep the pre-supervision configuration path lightweight. Import
            # the canonical model identity only when this runtime feature is
            # requested, rather than importing aisimulate_core for every CLI.
            from aisimulate_core.sdk.deepseek_v41 import MODEL_PATH as DEEPSEEK_V41_MODEL_PATH

            if self.model != DEEPSEEK_V41_MODEL_PATH or self.backend != "sglang":
                raise ValueError(f"decoder_replay requires model={DEEPSEEK_V41_MODEL_PATH!r} and backend='sglang'")
        _validate_worker_hardware(modes={self.mode}, workers=self.workers)
        if self.mode == "afd":
            _validate_prediction_afd(self)
        else:
            if self.afd is not None:
                raise ValueError("engine.afd requires engine.mode='afd'")
            _validate_worker_roles(
                modes={self.mode},
                workers=self.workers,
                has_transfer=self.kv_transfer is not None,
            )
        _validate_prediction_host_offload(self)
        _validate_backend_block_sizes(backends={self.backend}, modes={self.mode}, workers=self.workers)
        _validate_prediction_scheduler_backend(self)
        _validate_speculation(self, modes={self.mode}, backends={self.backend})
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
    bytes_per_token: KvBytesPerToken = "auto"
    capacity: KvCapacityRecommendationConfig = Field(default_factory=KvCapacityRecommendationConfig)
    host_offload: HostOffloadConfig | None = None


class WorkerRecommendationConfig(StrictModel):
    hardware: str | None = Field(default=None, min_length=1)
    parallelism: ParallelismRecommendationConfig = Field(default_factory=ParallelismRecommendationConfig)
    scheduler: SchedulerRecommendationConfig = Field(default_factory=SchedulerRecommendationConfig)
    kv_cache: KvCacheRecommendationConfig = Field(default_factory=KvCacheRecommendationConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    startup_seconds: float = Field(default=0.0, ge=0.0)


class EncoderRecommendationConfig(StrictModel):
    """Finite encoder domains; scalar values pin a single choice."""

    hardware: str | None = Field(default=None, min_length=1)
    backend_version: str | None = Field(default=None, min_length=1)
    tensor: PositiveInt | Choices[PositiveInt] = 1
    replicas: PositiveInt | Choices[PositiveInt] = 1
    batch_size: (
        Annotated[int, Field(strict=True, gt=0, le=8)] | Choices[Annotated[int, Field(strict=True, gt=0, le=8)]]
    ) = 1
    latency_correction: PositiveFloat = 1.0
    rate_degradation: Fraction = 0.9


class WorkersRecommendationConfig(StrictModel):
    encoder: EncoderRecommendationConfig | None = None
    aggregated: WorkerRecommendationConfig | None = None
    prefill: WorkerRecommendationConfig | None = None
    decode: WorkerRecommendationConfig | None = None


class EngineRecommendationConfig(EstimatorPolicyConfig):
    mode: EngineMode | Choices[EngineMode] = Field(
        default_factory=lambda: Choices[EngineMode](choices=["aggregated", "disaggregated"])
    )
    model: str
    hardware: str
    backend: Backend | Choices[Backend] = Field(default_factory=lambda: Choices[Backend](choices=["vllm", "sglang"]))
    backend_version: str | dict[str, str] | None = None
    context_length: PositiveInt | Literal["max"] = "max"
    speculation: NgramSpeculationConfig | None = None
    workers: WorkersRecommendationConfig = Field(default_factory=WorkersRecommendationConfig)
    kv_transfer: KvTransferConfig | None = None
    afd: AFDSearchRecommendationConfig | None = None

    @field_validator("model", "hardware")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must be nonempty")
        return value

    @model_validator(mode="after")
    def _validate_roles(self) -> EngineRecommendationConfig:
        modes = set(self.mode.choices) if isinstance(self.mode, Choices) else {self.mode}
        _validate_worker_hardware(modes=modes, workers=self.workers)
        if "afd" in modes:
            if modes != {"afd"}:
                raise ValueError("AFD recommendation mode cannot be mixed with aggregated/disaggregated modes")
            _validate_recommendation_afd(self)
        else:
            if self.afd is not None:
                raise ValueError("engine.afd requires engine.mode='afd'")
            _validate_worker_roles(
                modes=modes,
                workers=self.workers,
                has_transfer=self.kv_transfer is not None,
            )
        backends = set(self.backend.choices) if isinstance(self.backend, Choices) else {self.backend}
        if isinstance(self.backend_version, dict):
            unknown = sorted(set(self.backend_version) - backends)
            if unknown:
                raise ValueError(f"backend_version contains unconfigured backend(s): {unknown}")
        _validate_recommendation_host_offload(self)
        _validate_speculation(self, modes=modes, backends=backends)
        _validate_backend_block_sizes(backends=backends, modes=modes, workers=self.workers)
        return self


def _validate_speculation(engine, *, modes: set[str], backends: set[str]) -> None:
    if engine.speculation is None:
        return
    if engine.nextn:
        raise ValueError("speculation cannot be combined with nextn")
    if backends != {"vllm"} or "afd" in modes or engine.workers.encoder is not None:
        raise ValueError("ngram speculation requires vllm aggregated/disaggregated language workers")
    for role in ("aggregated", "prefill", "decode"):
        worker = getattr(engine.workers, role)
        if worker is None:
            continue
        if worker.kv_cache.host_offload is not None:
            raise ValueError("ngram speculation does not support host_offload")
        if worker.timing.forward_model != "op_level":
            raise ValueError("ngram speculation requires op_level timing")


def _validate_worker_hardware(*, modes: set[str], workers) -> None:
    for role in ("aggregated", "prefill", "decode"):
        worker = getattr(workers, role)
        if worker is None or worker.hardware is None:
            continue
        if role == "aggregated" or "disaggregated" not in modes or "afd" in modes:
            raise ValueError("worker hardware overrides require prefill/decode workers in disaggregated mode")
        hardware = worker.hardware.strip()
        if not hardware or hardware == "auto" or hardware != worker.hardware:
            raise ValueError(f"workers.{role}.hardware must be one concrete nonempty hardware identifier")


def _workers_with_host_offload(workers) -> list[tuple[str, Any]]:
    return [
        (role, worker)
        for role in ("aggregated", "prefill", "decode")
        if (worker := getattr(workers, role)) is not None and worker.kv_cache.host_offload is not None
    ]


def _configured_worker_roles(workers) -> set[str]:
    return {role for role in ("aggregated", "prefill", "decode") if getattr(workers, role) is not None}


def _validate_prediction_scheduler_backend(engine: EnginePredictionConfig) -> None:
    for role in ("aggregated", "prefill", "decode"):
        worker = getattr(engine.workers, role)
        if worker is None:
            continue
        for field, backend, default in (
            ("prefill_schedule_interval", "vllm", 1),
            ("prefill_decode_interval", "sglang", 0),
        ):
            if engine.backend != backend and getattr(worker.scheduler, field) != default:
                raise ValueError(f"workers.{role}.scheduler.{field} is supported only for backend={backend}")


def _validate_prediction_afd(engine: EnginePredictionConfig) -> None:
    if engine.workers.encoder is not None:
        raise ValueError("AFD does not support analytical EPD encoder pools")
    afd = engine.afd
    if afd is None:
        raise ValueError("engine.mode='afd' requires engine.afd")
    if engine.kv_transfer is not None:
        raise ValueError("AFD mode rejects kv_transfer; A/F transfers are modeled by engine.afd")
    configured = _configured_worker_roles(engine.workers)
    if afd.combined_with_pd:
        if afd.phase == "both":
            raise ValueError("AFD+P/D requires phase='prefill' or phase='decode'")
        companion = "decode" if afd.phase == "prefill" else "prefill"
        if configured != {companion}:
            raise ValueError(f"AFD phase={afd.phase!r} combined_with_pd=true requires only workers.{companion}")
    else:
        if afd.phase != "both":
            raise ValueError("pure AFD prediction requires phase='both'; use combined_with_pd=true for one phase")
        if configured:
            raise ValueError("pure AFD prediction rejects regular aggregated/prefill/decode workers")


def _validate_recommendation_afd(engine: EngineRecommendationConfig) -> None:
    if engine.workers.encoder is not None:
        raise ValueError("AFD does not support analytical EPD encoder pools")
    afd = engine.afd
    if afd is None:
        raise ValueError("engine.mode='afd' requires engine.afd")
    if engine.kv_transfer is not None:
        raise ValueError("AFD mode rejects kv_transfer; A/F transfers are modeled by engine.afd")
    configured = _configured_worker_roles(engine.workers)
    if afd.combined_with_pd:
        if afd.phase == "both":
            raise ValueError("AFD+P/D requires phase='prefill' or phase='decode'")
        companion = "decode" if afd.phase == "prefill" else "prefill"
        unexpected = configured - {companion}
        if unexpected:
            raise ValueError(f"AFD phase={afd.phase!r} combined_with_pd=true accepts only optional workers.{companion}")
    else:
        if afd.phase != "both":
            raise ValueError("pure AFD recommendation requires phase='both'; use combined_with_pd=true for one phase")
        if configured:
            raise ValueError("pure AFD recommendation rejects regular aggregated/prefill/decode workers")


def _validate_prediction_host_offload(engine: EnginePredictionConfig) -> None:
    configured = _workers_with_host_offload(engine.workers)
    if not configured:
        return
    if engine.mode != "aggregated" or [role for role, _ in configured] != ["aggregated"]:
        raise ValueError("host_offload is supported only for the aggregated worker")
    if engine.backend != "vllm":
        raise ValueError("host_offload is supported only for backend=vllm")
    worker = configured[0][1]
    if not worker.kv_cache.prefix_caching:
        raise ValueError("host_offload requires prefix_caching=true")
    if worker.parallelism.attention_data != 1:
        raise ValueError("host_offload requires attention_data=1")


def _validate_recommendation_host_offload(engine: EngineRecommendationConfig) -> None:
    configured = _workers_with_host_offload(engine.workers)
    if not configured:
        return
    if engine.mode != "aggregated" or [role for role, _ in configured] != ["aggregated"]:
        raise ValueError("host_offload recommendation requires concrete mode=aggregated")
    if engine.backend != "vllm":
        raise ValueError("host_offload recommendation requires concrete backend=vllm")
    worker = configured[0][1]
    if not worker.kv_cache.prefix_caching:
        raise ValueError("host_offload requires prefix_caching=true")
    parallel = worker.parallelism
    if parallel.preset not in (False, {}) or parallel.attention_data != 1:
        raise ValueError("host_offload recommendation requires fixed parallelism with attention_data=1")


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
