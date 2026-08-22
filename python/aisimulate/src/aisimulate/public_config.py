# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict public configuration models for ``aisimulate predict/recommend``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SyntheticSource(_StrictModel):
    type: Literal["synthetic"] = "synthetic"
    input_tokens: int = Field(default=1024, gt=0)
    output_tokens: int = Field(default=128, gt=0)


class SessionShape(_StrictModel):
    turns: int = Field(default=4, ge=2)
    shared_prefix_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    prefix_groups: int = Field(default=0, ge=0)
    inter_turn_delay_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_prefix_groups(self) -> SessionShape:
        if self.shared_prefix_ratio > 0.0 and self.prefix_groups == 0:
            raise ValueError(
                "prefix_groups must be positive when shared_prefix_ratio is positive"
            )
        return self


class SyntheticSessionSource(_StrictModel):
    type: Literal["synthetic-session"]
    new_input_tokens_per_turn: int = Field(default=1024, gt=0)
    output_tokens_per_turn: int = Field(default=128, gt=0)
    session: SessionShape


TraceFormat = Literal[
    "mooncake",
    "mooncake-delta",
    "agentic_mooncake",
    "applied_compute_agentic",
    "dynamo",
]


class TraceSource(_StrictModel):
    type: Literal["trace"]
    paths: list[str]
    format: TraceFormat = "mooncake"
    block_size: int | None = Field(default=None, gt=0)

    @field_validator("paths")
    @classmethod
    def _validate_paths(cls, paths: list[str]) -> list[str]:
        if not paths or any(not path for path in paths):
            raise ValueError("trace paths must contain at least one nonempty path")
        return paths

    @model_validator(mode="after")
    def _validate_path_count(self) -> TraceSource:
        if self.format != "dynamo" and len(self.paths) != 1:
            raise ValueError(f"trace format {self.format!r} requires exactly one path")
        return self


TrafficSource = Annotated[
    SyntheticSource | SyntheticSessionSource | TraceSource,
    Field(discriminator="type"),
]


class TrafficLoad(_StrictModel):
    type: Literal[
        "concurrency",
        "poisson",
        "constant_rate",
        "kv_capacity_fraction",
        "trace_timestamps",
    ] = "concurrency"
    concurrency: int | None = Field(default=None, gt=0)
    requests_per_second: float | None = Field(default=None, gt=0)
    sessions_per_second: float | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None, ge=0)
    fraction: float | None = Field(default=None, gt=0)
    speedup: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_fields_for_type(self) -> TrafficLoad:
        used = {
            name
            for name in (
                "concurrency",
                "requests_per_second",
                "sessions_per_second",
                "seed",
                "fraction",
                "speedup",
            )
            if getattr(self, name) is not None
        }
        allowed = {
            "concurrency": {"concurrency"},
            "poisson": {"requests_per_second", "sessions_per_second", "seed"},
            "constant_rate": {"requests_per_second", "sessions_per_second"},
            "kv_capacity_fraction": {"fraction"},
            "trace_timestamps": {"speedup"},
        }[self.type]
        unexpected = used - allowed
        if unexpected:
            raise ValueError(
                f"traffic.load.type={self.type!r} does not accept {sorted(unexpected)}"
            )
        required = {
            "concurrency": ("concurrency",),
            "poisson": (),
            "constant_rate": (),
            "kv_capacity_fraction": ("fraction",),
            "trace_timestamps": (),
        }[self.type]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"traffic.load.type={self.type!r} requires {', '.join(missing)}"
            )
        return self


class TrafficStop(_StrictModel):
    requests: int | None = Field(default=None, gt=0)
    requests_per_load_unit: float | None = Field(default=None, gt=0)
    sessions: int | None = Field(default=None, gt=0)
    sessions_per_load_unit: float | None = Field(default=None, gt=0)
    max_virtual_time_seconds: float | None = Field(default=None, gt=0)


class TrafficConfig(_StrictModel):
    source: TrafficSource
    load: TrafficLoad
    stop: TrafficStop | None = None

    @classmethod
    def default(cls) -> TrafficConfig:
        return cls(
            source=SyntheticSource(),
            load=TrafficLoad(type="concurrency", concurrency=10),
            stop=TrafficStop(requests=100),
        )

    @model_validator(mode="after")
    def _validate_source_load_stop(self) -> TrafficConfig:
        source = self.source
        load = self.load
        stop = self.stop
        if isinstance(source, TraceSource):
            if load.type not in {"trace_timestamps", "concurrency"}:
                raise ValueError(
                    "trace traffic requires trace_timestamps or concurrency load"
                )
            if stop is not None and any(
                getattr(stop, name) is not None
                for name in (
                    "requests",
                    "requests_per_load_unit",
                    "sessions",
                    "sessions_per_load_unit",
                )
            ):
                raise ValueError("trace traffic only accepts max_virtual_time_seconds")
            agentic = source.format == "agentic_mooncake"
            if agentic and load.type != "trace_timestamps":
                raise ValueError("agentic_mooncake requires trace_timestamps load")
            if source.format == "applied_compute_agentic" and load.type != "concurrency":
                raise ValueError("applied_compute_agentic requires concurrency load")
            if agentic and stop is not None and stop.max_virtual_time_seconds is not None:
                raise ValueError(
                    "agentic_mooncake does not support max_virtual_time_seconds"
                )
            return self

        if load.type in {"trace_timestamps"}:
            raise ValueError("synthetic traffic does not accept trace_timestamps load")
        rate = (
            load.requests_per_second
            if isinstance(source, SyntheticSource)
            else load.sessions_per_second
        )
        wrong_rate = (
            load.sessions_per_second
            if isinstance(source, SyntheticSource)
            else load.requests_per_second
        )
        if load.type in {"poisson", "constant_rate"} and rate is None:
            unit = (
                "requests_per_second"
                if isinstance(source, SyntheticSource)
                else "sessions_per_second"
            )
            raise ValueError(f"{load.type} synthetic traffic requires {unit}")
        if wrong_rate is not None:
            raise ValueError("traffic load rate unit does not match source type")
        if stop is None:
            raise ValueError("synthetic traffic requires a stop condition")
        if stop.max_virtual_time_seconds is not None:
            raise ValueError("max_virtual_time_seconds is trace-only")
        request_stop = stop.requests is not None or stop.requests_per_load_unit is not None
        session_stop = stop.sessions is not None or stop.sessions_per_load_unit is not None
        if isinstance(source, SyntheticSource):
            if not request_stop or session_stop:
                raise ValueError("synthetic request traffic requires a request stop")
        elif not session_stop or request_stop:
            raise ValueError("synthetic session traffic requires a session stop")
        if sum(
            value is not None
            for value in (
                stop.requests,
                stop.requests_per_load_unit,
                stop.sessions,
                stop.sessions_per_load_unit,
            )
        ) != 1:
            raise ValueError("traffic.stop requires exactly one count condition")
        return self


class ParallelismConfig(_StrictModel):
    replicas: int = Field(default=1, gt=0)
    tensor: int = Field(default=1, gt=0)
    pipeline: int = Field(default=1, gt=0)
    attention_data: int = Field(default=1, gt=0)
    moe_tensor: int = Field(default=1, gt=0)
    moe_expert: int = Field(default=1, gt=0)


class SchedulerConfig(_StrictModel):
    max_batched_tokens: int = Field(default=8192, gt=0)
    max_sequences: int = Field(default=256, gt=0)


class KvCapacityConfig(_StrictModel):
    type: Literal["default", "fixed"] = "default"
    memory_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    blocks: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_capacity(self) -> KvCapacityConfig:
        if self.type == "fixed":
            if self.blocks is None:
                raise ValueError("fixed KV capacity requires blocks")
            if self.memory_fraction is not None:
                raise ValueError("fixed KV capacity rejects memory_fraction")
        elif self.blocks is not None:
            raise ValueError("default KV capacity rejects blocks")
        return self


class KvCacheConfig(_StrictModel):
    block_size: int | None = Field(default=None, gt=0)
    prefix_caching: bool = True
    capacity: KvCapacityConfig = Field(default_factory=KvCapacityConfig)


class TimingConfig(_StrictModel):
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


class WorkerConfig(_StrictModel):
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    kv_cache: KvCacheConfig = Field(default_factory=KvCacheConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    startup_seconds: float = Field(default=0.0, ge=0.0)


class WorkersConfig(_StrictModel):
    aggregated: WorkerConfig | None = None
    prefill: WorkerConfig | None = None
    decode: WorkerConfig | None = None


class KvTransferConfig(_StrictModel):
    bytes_per_token: int | Literal["auto"] = "auto"
    bandwidth_gb_per_second: float | None = Field(default=None, gt=0.0)
    timing_mode: Literal["full_prompt", "destination_missing"] = (
        "destination_missing"
    )

    @field_validator("bytes_per_token")
    @classmethod
    def _validate_bytes_per_token(cls, value: int | str) -> int | str:
        if value != "auto" and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError("bytes_per_token must be 'auto' or a positive integer")
        return value


class EngineConfig(_StrictModel):
    mode: Literal["aggregated", "disaggregated"] = "aggregated"
    model: str
    hardware: str
    backend: Literal["vllm", "sglang", "trtllm"] = "vllm"
    backend_version: str | None = None
    context_length: int | Literal["max"] = "max"
    workers: WorkersConfig
    kv_transfer: KvTransferConfig | None = None

    @field_validator("model", "hardware")
    @classmethod
    def _validate_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must be nonempty")
        return value

    @field_validator("context_length")
    @classmethod
    def _validate_context_length(cls, value: int | str) -> int | str:
        if value != "max" and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError("context_length must be 'max' or a positive integer")
        return value

    @model_validator(mode="after")
    def _validate_roles(self) -> EngineConfig:
        workers = self.workers
        if self.mode == "aggregated":
            if workers.aggregated is None:
                raise ValueError("aggregated mode requires workers.aggregated")
            if workers.prefill is not None or workers.decode is not None:
                raise ValueError("aggregated mode rejects prefill/decode workers")
            if self.kv_transfer is not None:
                raise ValueError("aggregated mode rejects kv_transfer")
        else:
            if workers.prefill is None or workers.decode is None:
                raise ValueError("disaggregated mode requires prefill and decode workers")
            if workers.aggregated is not None:
                raise ValueError("disaggregated mode rejects aggregated workers")
            if self.backend == "trtllm":
                raise ValueError("TensorRT-LLM disaggregated mode is unsupported")
        return self


class PrefillLoadModel(_StrictModel):
    type: Literal["none", "aic"] = "none"


class RouterConfig(_StrictModel):
    policy: Literal["round_robin", "kv_router"] = "round_robin"
    prefill_load_model: PrefillLoadModel = Field(default_factory=PrefillLoadModel)
    overlap_score_credit: float | None = Field(default=None, ge=0.0)
    prefill_load_scale: float | None = Field(default=None, ge=0.0)
    temperature: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_policy(self) -> RouterConfig:
        kv_fields = (
            self.overlap_score_credit,
            self.prefill_load_scale,
            self.temperature,
        )
        if self.policy == "round_robin" and (
            self.prefill_load_model.type != "none"
            or any(value is not None for value in kv_fields)
        ):
            raise ValueError(
                "round_robin requires prefill_load_model.type='none' and "
                "rejects KV-router knobs"
            )
        return self


class PlannerConfig(_StrictModel):
    policy: Literal["disabled", "enabled"] = "disabled"
    target: Literal["throughput", "latency", "sla", "load"] = "throughput"
    enable_throughput_scaling: bool = True
    enable_load_scaling: bool = False
    throughput_adjustment_interval_seconds: int = Field(default=180, gt=0)
    load_adjustment_interval_seconds: int = Field(default=5, gt=0)
    max_num_fpm_samples: int = Field(default=64, gt=0)
    fpm_sample_bucket_size: int = Field(default=16, gt=0)
    load_scaling_down_sensitivity: int = Field(default=80, ge=0, le=100)
    load_min_observations: int = Field(default=5, gt=0)
    load_predictor: Literal["constant", "arima", "prophet", "kalman"] = "arima"
    load_predictor_log1p: bool = False
    prophet_window_size: int = Field(default=50, gt=0)
    kalman_q_level: float = Field(default=1.0, gt=0)
    kalman_q_trend: float = Field(default=0.1, gt=0)
    kalman_r: float = Field(default=10.0, gt=0)
    kalman_min_points: int = Field(default=5, gt=0)
    max_num_gpus: int = Field(default=8, gt=0)
    min_workers: int = Field(default=1, ge=0)
    prefill_min_workers: int | None = Field(default=None, gt=0)
    decode_min_workers: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_fields(self) -> PlannerConfig:
        root = math.isqrt(self.fpm_sample_bucket_size)
        if root * root != self.fpm_sample_bucket_size:
            raise ValueError("fpm_sample_bucket_size must be a perfect square")
        if (
            self.enable_load_scaling
            and self.load_adjustment_interval_seconds
            >= self.throughput_adjustment_interval_seconds
        ):
            raise ValueError(
                "load adjustment interval must be shorter than throughput interval"
            )
        return self


class SlaConfig(_StrictModel):
    ttft_ms: float | None = Field(default=None, gt=0)
    itl_ms: float | None = Field(default=None, gt=0)
    e2e_ms: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_form(self) -> SlaConfig:
        token_form = self.ttft_ms is not None or self.itl_ms is not None
        if token_form and (self.ttft_ms is None or self.itl_ms is None):
            raise ValueError("ttft_ms and itl_ms must be supplied together")
        if token_form and self.e2e_ms is not None:
            raise ValueError("e2e_ms is mutually exclusive with ttft_ms/itl_ms")
        return self


class EvaluationConfig(_StrictModel):
    sla: SlaConfig | None = None


class PredictionConfig(_StrictModel):
    traffic: TrafficConfig = Field(default_factory=TrafficConfig.default)
    engine: EngineConfig
    router: RouterConfig = Field(default_factory=RouterConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _validate_cross_component(self) -> PredictionConfig:
        source = self.traffic.source
        if isinstance(source, TraceSource) and source.format in {
            "mooncake-delta",
            "agentic_mooncake",
        }:
            if self.engine.mode != "aggregated":
                raise ValueError(f"{source.format} requires aggregated engine mode")
            if self.planner.policy != "disabled":
                raise ValueError(f"{source.format} requires planner.policy=disabled")
        if self.planner.policy == "enabled" and self.planner.enable_throughput_scaling:
            sla = self.evaluation.sla
            if (
                self.planner.target != "sla"
                or sla is None
                or sla.ttft_ms is None
                or sla.itl_ms is None
            ):
                raise ValueError(
                    "Planner throughput scaling requires target='sla' and "
                    "evaluation.sla.ttft_ms/itl_ms"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PredictionConfig:
        return cls.model_validate(_load_yaml(path))


class CandidateConstraints(_StrictModel):
    min_candidate_gpus: int | None = Field(default=None, gt=0)
    max_candidate_gpus: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> CandidateConstraints:
        if (
            self.min_candidate_gpus is not None
            and self.min_candidate_gpus > self.max_candidate_gpus
        ):
            raise ValueError("min_candidate_gpus cannot exceed max_candidate_gpus")
        return self


class OptimizationConfig(_StrictModel):
    target: Literal[
        "throughput",
        "throughput_per_gpu",
        "throughput_per_user",
        "goodput",
        "goodput_per_gpu",
        "ttft",
        "e2e_latency",
        "pareto",
    ] = "throughput"
    hardware: str | None = None
    constraints: CandidateConstraints = Field(default_factory=CandidateConstraints)

    @field_validator("hardware")
    @classmethod
    def _validate_hardware(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("optimization.hardware must be nonempty")
        return value


class OptimizerConfig(_StrictModel):
    algorithm: Literal["bayesian", "random"] = "bayesian"
    max_trials: int = Field(default=320, gt=0)
    parallelism: int = Field(default=16, gt=0)
    candidate_timeout_seconds: float = Field(default=600.0, gt=0)
    seed: int = Field(default=42, ge=0)


class RecommendationConfig(_StrictModel):
    """Strict top-level recommendation document.

    Component mappings retain domain objects until the recommendation compiler
    lowers them to the existing Sweeper model. ``validate_recommendation_tree``
    performs recursive unknown-field and domain-shape validation.
    """

    traffic: dict[str, Any] | None = None
    engine: dict[str, Any]
    router: dict[str, Any] | None = None
    planner: dict[str, Any] | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    optimization: OptimizationConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @model_validator(mode="after")
    def _validate_tree(self) -> RecommendationConfig:
        validate_recommendation_tree(self.model_dump(mode="python"))
        _validate_planner_preset_conflicts(self.planner)
        _validate_recommendation_router(self.router)
        hardware = self.engine.get("hardware")
        if hardware == "auto" and self.optimization.hardware is None:
            raise ValueError(
                "engine.hardware='auto' requires one optimization.hardware"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> RecommendationConfig:
        return cls.model_validate(_load_yaml(path))


def _load_yaml(path: str | Path) -> Any:
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read configuration {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed YAML in {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"configuration {source} must contain one YAML mapping")
    return data


_RECOMMENDATION_TREE: dict[str, Any] = {
    "traffic": {
        "source": {
            "type": None,
            "input_tokens": None,
            "output_tokens": None,
            "new_input_tokens_per_turn": None,
            "output_tokens_per_turn": None,
            "session": {
                "turns": None,
                "shared_prefix_ratio": None,
                "prefix_groups": None,
                "inter_turn_delay_ms": None,
            },
            "paths": None,
            "format": None,
            "block_size": None,
        },
        "load": {
            "type": None,
            "concurrency": None,
            "requests_per_second": None,
            "sessions_per_second": None,
            "seed": None,
            "fraction": None,
            "speedup": None,
        },
        "stop": {
            "requests": None,
            "requests_per_load_unit": None,
            "sessions": None,
            "sessions_per_load_unit": None,
            "max_virtual_time_seconds": None,
        },
    },
    "engine": {
        "mode": None,
        "model": None,
        "hardware": None,
        "backend": None,
        "backend_version": None,
        "context_length": None,
        "workers": {"*": {
            "parallelism": {
                "preset": None,
                "replicas": None,
                "tensor": None,
                "pipeline": None,
                "attention_data": None,
                "moe_tensor": None,
                "moe_expert": None,
            },
            "scheduler": {"max_batched_tokens": None, "max_sequences": None},
            "kv_cache": {
                "block_size": None,
                "prefix_caching": None,
                "capacity": {
                    "type": None,
                    "memory_fraction": None,
                    "blocks": None,
                },
            },
            "timing": {"type": None, "prefill_ms": None, "decode_ms": None},
            "startup_seconds": None,
        }},
        "kv_transfer": {
            "bytes_per_token": None,
            "bandwidth_gb_per_second": None,
            "timing_mode": None,
        },
    },
    "router": {
        "policy": None,
        "prefill_load_model": {"type": None},
        "overlap_score_credit": None,
        "prefill_load_scale": None,
        "temperature": None,
    },
    "planner": {
        "scaling_policy": {"preset": None},
        "fpm_sampling": {"preset": None},
        "load_sensitivity": {"preset": None},
        "load_predictor": {
            "preset": None,
            # ``type`` is the independent form of the concrete
            # ``planner.load_predictor`` knob when this sub-item's preset is off.
            "type": None,
        },
        "policy": None,
        "target": None,
        "enable_throughput_scaling": None,
        "enable_load_scaling": None,
        "throughput_adjustment_interval_seconds": None,
        "load_adjustment_interval_seconds": None,
        "max_num_fpm_samples": None,
        "fpm_sample_bucket_size": None,
        "load_scaling_down_sensitivity": None,
        "load_min_observations": None,
        "load_predictor_log1p": None,
        "prophet_window_size": None,
        "kalman_q_level": None,
        "kalman_q_trend": None,
        "kalman_r": None,
        "kalman_min_points": None,
        "max_num_gpus": None,
        "min_workers": None,
        "prefill_min_workers": None,
        "decode_min_workers": None,
    },
    "evaluation": {"sla": {"ttft_ms": None, "itl_ms": None, "e2e_ms": None}},
    "optimization": {
        "target": None,
        "hardware": None,
        "constraints": {
            "min_candidate_gpus": None,
            "max_candidate_gpus": None,
        },
    },
    "optimizer": {
        "algorithm": None,
        "max_trials": None,
        "parallelism": None,
        "candidate_timeout_seconds": None,
        "seed": None,
    },
}

_DOMAIN_PATHS = {
    "engine.mode",
    "engine.backend",
    "engine.workers.*.parallelism.replicas",
    "engine.workers.*.parallelism.tensor",
    "engine.workers.*.parallelism.pipeline",
    "engine.workers.*.parallelism.attention_data",
    "engine.workers.*.parallelism.moe_tensor",
    "engine.workers.*.parallelism.moe_expert",
    "engine.workers.*.scheduler.max_batched_tokens",
    "engine.workers.*.scheduler.max_sequences",
    "engine.workers.*.kv_cache.block_size",
    "engine.workers.*.kv_cache.capacity.memory_fraction",
    "router.policy",
    "router.prefill_load_model.type",
    "router.overlap_score_credit",
    "router.prefill_load_scale",
    "router.temperature",
    "planner.policy",
    "planner.enable_throughput_scaling",
    "planner.enable_load_scaling",
    "planner.throughput_adjustment_interval_seconds",
    "planner.load_adjustment_interval_seconds",
    "planner.max_num_fpm_samples",
    "planner.fpm_sample_bucket_size",
    "planner.load_scaling_down_sensitivity",
    "planner.load_min_observations",
    "planner.load_predictor",
    "planner.load_predictor.type",
    "planner.load_predictor_log1p",
    "planner.prophet_window_size",
    "planner.kalman_q_level",
    "planner.kalman_q_trend",
    "planner.kalman_r",
    "planner.kalman_min_points",
    "planner.min_workers",
    "planner.prefill_min_workers",
    "planner.decode_min_workers",
    "traffic.load.concurrency",
    "traffic.load.requests_per_second",
    "traffic.load.sessions_per_second",
    "traffic.load.fraction",
    "traffic.load.speedup",
}

_PRESET_PATHS = {
    "engine.workers.*.parallelism.preset",
    "planner.scaling_policy.preset",
    "planner.fpm_sampling.preset",
    "planner.load_sensitivity.preset",
    "planner.load_predictor.preset",
}


def _canonical_path(parts: list[str]) -> str:
    return ".".join("*" if part in {"aggregated", "prefill", "decode"} else part for part in parts)


def _is_domain(value: Any) -> bool:
    return isinstance(value, dict) and set(value) in ({"choices"}, {"range"})


def _validate_domain(value: dict[str, Any], path: str) -> None:
    if "choices" in value:
        choices = value["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{path}.choices must be a nonempty list")
        if len({repr(choice) for choice in choices}) != len(choices):
            raise ValueError(f"{path}.choices must contain unique values")
        return
    raw_range = value["range"]
    if not isinstance(raw_range, dict):
        raise ValueError(f"{path}.range must be a mapping")
    unknown = set(raw_range) - {"min", "max", "step", "scale"}
    if unknown:
        raise ValueError(f"{path}.range has unknown fields {sorted(unknown)}")
    if "min" not in raw_range or "max" not in raw_range:
        raise ValueError(f"{path}.range requires min and max")
    minimum, maximum = raw_range["min"], raw_range["max"]
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(minimum))
        or not math.isfinite(float(maximum))
        or minimum > maximum
    ):
        raise ValueError(f"{path}.range requires finite min <= max")
    scale = raw_range.get("scale", "linear")
    if scale not in {"linear", "log"}:
        raise ValueError(f"{path}.range.scale must be linear or log")
    step = raw_range.get("step")
    if scale == "log":
        if minimum <= 0 or step is not None:
            raise ValueError(f"{path} log range requires min > 0 and rejects step")
    elif step is not None and (
        isinstance(step, bool) or not isinstance(step, (int, float)) or step <= 0
    ):
        raise ValueError(f"{path}.range.step must be positive")


def _validate_preset(value: Any, path: str) -> None:
    if value in ("default", False) or value == {}:
        return
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{path} must be 'default', false, {{}}, or a nonempty mapping list"
        )
    for index, mapping in enumerate(value):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{path}[{index}] must be a nonempty mapping")
        if any(_is_domain(item) for item in mapping.values()):
            raise ValueError(f"{path}[{index}] cannot contain a search domain")


def _validate_planner_preset_conflicts(planner: dict[str, Any] | None) -> None:
    if not isinstance(planner, dict):
        return
    groups = {
        "scaling_policy": {
            "enable_throughput_scaling",
            "enable_load_scaling",
            "throughput_adjustment_interval_seconds",
            "load_adjustment_interval_seconds",
        },
        "fpm_sampling": {"max_num_fpm_samples", "fpm_sample_bucket_size"},
        "load_sensitivity": {
            "load_scaling_down_sensitivity",
            "load_min_observations",
        },
        "load_predictor": {
            "load_predictor_log1p",
            "prophet_window_size",
            "kalman_q_level",
            "kalman_q_trend",
            "kalman_r",
            "kalman_min_points",
        },
    }
    for group, knobs in groups.items():
        control = planner.get(group)
        if not isinstance(control, dict) or "preset" not in control:
            continue
        preset = control["preset"]
        active = preset not in (False, {})
        conflicts = sorted(knobs.intersection(planner))
        if group == "load_predictor" and "type" in control:
            conflicts.append("load_predictor.type")
        if active and conflicts:
            raise ValueError(
                f"planner.{group}.preset cannot be combined with independent knobs {conflicts}"
            )


def _validate_recommendation_router(router: dict[str, Any] | None) -> None:
    if not isinstance(router, dict):
        return
    policy = router.get("policy")
    if policy != "round_robin":
        return
    load_model = router.get("prefill_load_model")
    load_type = load_model.get("type") if isinstance(load_model, dict) else None
    kv_fields = [
        name
        for name in (
            "overlap_score_credit",
            "prefill_load_scale",
            "temperature",
        )
        if name in router
    ]
    if load_type not in (None, "none"):
        kv_fields.append("prefill_load_model.type")
    if kv_fields:
        raise ValueError(
            "router.policy=round_robin rejects KV-router fields "
            f"{sorted(kv_fields)}"
        )


def validate_recommendation_tree(data: dict[str, Any]) -> None:
    """Reject unknown nested fields and malformed domains/presets."""

    def visit(value: Any, schema: Any, parts: list[str]) -> None:
        path = ".".join(parts)
        canonical = _canonical_path(parts)
        if value is None:
            return
        if _is_domain(value):
            if canonical not in _DOMAIN_PATHS:
                raise ValueError(f"{path} is not sweepable")
            _validate_domain(value, path)
            return
        if schema is None:
            return
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        allowed = set(schema) - {"*"}
        wildcard = schema.get("*")
        unknown = set(value) - allowed
        if wildcard is None and unknown:
            raise ValueError(f"{path} has unknown fields {sorted(unknown)}")
        if wildcard is not None:
            invalid_roles = unknown - {"aggregated", "prefill", "decode"}
            if invalid_roles:
                raise ValueError(f"{path} has unknown fields {sorted(invalid_roles)}")
        for key, item in value.items():
            child = schema.get(key, wildcard)
            child_parts = [*parts, key]
            child_canonical = _canonical_path(child_parts)
            if child_canonical in _PRESET_PATHS:
                _validate_preset(item, ".".join(child_parts))
            else:
                visit(item, child, child_parts)

    for section in ("traffic", "engine", "router", "planner", "evaluation"):
        value = data.get(section)
        if value is not None:
            visit(value, _RECOMMENDATION_TREE[section], [section])


def known_config_path(path: str, *, command: str) -> bool:
    """Return whether a dot path names one public schema field."""

    parts = path.split(".")
    if not parts or any(not part or part.isdigit() for part in parts):
        return False
    if command == "predict" and parts[0] in {"optimization", "optimizer"}:
        return False
    schema: Any = _RECOMMENDATION_TREE
    for index, part in enumerate(parts):
        if not isinstance(schema, dict):
            return index == len(parts)
        child = schema.get(part)
        if child is None and "*" in schema and part in {
            "aggregated",
            "prefill",
            "decode",
        }:
            child = schema["*"]
        elif part not in schema:
            return False
        schema = child
    return True


def public_prediction_mapping(config: PredictionConfig) -> dict[str, Any]:
    """Return concrete public YAML with inactive component details omitted."""

    data = config.model_dump(mode="python", exclude_none=True)
    planner = data.get("planner")
    if isinstance(planner, dict) and planner.get("policy") == "disabled":
        data["planner"] = {"policy": "disabled"}
    router = data.get("router")
    if isinstance(router, dict) and router.get("policy") == "round_robin":
        data["router"] = {
            "policy": "round_robin",
            "prefill_load_model": {"type": "none"},
        }
    return data
