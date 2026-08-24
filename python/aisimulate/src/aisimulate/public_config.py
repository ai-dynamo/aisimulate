# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict public configuration models for ``aisimulate predict/recommend``."""

from __future__ import annotations

import math
from copy import deepcopy
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

    @model_validator(mode="after")
    def _apply_role_scheduler_defaults(self) -> WorkersConfig:
        for role, max_sequences in (
            ("aggregated", 256),
            ("prefill", 1),
            ("decode", 256),
        ):
            worker = getattr(self, role)
            if worker is None or "max_sequences" in worker.scheduler.model_fields_set:
                continue
            scheduler = worker.scheduler.model_copy(
                update={"max_sequences": max_sequences}
            )
            setattr(self, role, worker.model_copy(update={"scheduler": scheduler}))
        return self


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
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _validate_cross_component(self) -> PredictionConfig:
        source = self.traffic.source
        if (
            isinstance(source, TraceSource)
            and source.format in {"mooncake-delta", "agentic_mooncake"}
            and self.engine.mode != "aggregated"
        ):
            raise ValueError(f"{source.format} requires aggregated engine mode")
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

    Engine mappings retain domain objects until the recommendation compiler
    lowers them to the existing Sweeper model. Optional stack components are
    validated separately by their configuration adapters.
    """

    traffic: dict[str, Any] | None = None
    engine: dict[str, Any]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    optimization: OptimizationConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @model_validator(mode="after")
    def _validate_tree(self) -> RecommendationConfig:
        validate_recommendation_tree(self.model_dump(mode="python"))
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
    "traffic.load.concurrency",
    "traffic.load.requests_per_second",
    "traffic.load.sessions_per_second",
    "traffic.load.fraction",
    "traffic.load.speedup",
}

_PRESET_PATHS = {
    "engine.workers.*.parallelism.preset",
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

    for section in ("traffic", "engine", "evaluation"):
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
    if parts[0] not in _RECOMMENDATION_TREE:
        # Optional stack sections are validated by the adapter selected from
        # ``<stack>.<section>`` after all overrides have been applied.
        return True
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


def public_prediction_mapping(
    config: PredictionConfig,
    adapter_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return concrete public YAML with validated adapter sections included."""

    data = config.model_dump(mode="python", exclude_none=True)
    data.update(deepcopy(adapter_configs or {}))
    return data


PREDICTION_CORE_SECTIONS = frozenset({"traffic", "engine", "evaluation"})
RECOMMENDATION_CORE_SECTIONS = frozenset(
    {*PREDICTION_CORE_SECTIONS, "optimization", "optimizer"}
)


def split_config_sections(
    data: dict[str, Any], *, command: Literal["predict", "recommend"]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Split public core fields from present adapter-owned top-level sections."""

    core_names = (
        PREDICTION_CORE_SECTIONS
        if command == "predict"
        else RECOMMENDATION_CORE_SECTIONS
    )
    if command == "predict":
        forbidden = sorted(set(data).intersection({"optimization", "optimizer"}))
        if forbidden:
            raise ValueError(f"predict does not accept {forbidden}")
    core = {name: value for name, value in data.items() if name in core_names}
    adapters: dict[str, dict[str, Any]] = {}
    for section, value in data.items():
        if not isinstance(section, str) or not section or "." in section:
            raise ValueError(
                "top-level configuration keys must be nonempty section names "
                f"without dots; got {section!r}"
            )
        if section in core_names or section in {"optimization", "optimizer"}:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"adapter section {section!r} must be a mapping")
        adapters[section] = deepcopy(value)
    return core, adapters
