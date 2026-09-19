# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config primitives for concrete predictions and recommendation domains."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _absolute_systems_path(value: str) -> str:
    if not value.strip() or "," in value:
        raise ValueError("systems_path must name one nonempty local directory, not a comma-separated list")
    return str(Path(value).expanduser().resolve())


SystemsPath = Annotated[str, Field(strict=True, min_length=1), AfterValidator(_absolute_systems_path)]


def requested_backend_version(versions: str | dict[str, str] | None, backend: str) -> str | None:
    """Return the version pin for one backend; ``None`` means resolve latest."""
    return versions.get(backend) if isinstance(versions, dict) else versions


T = TypeVar("T")
PositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]


class Choices(StrictModel, Generic[T]):
    choices: list[T]

    @field_validator("choices")
    @classmethod
    def _validate_choices(cls, choices: list[T]) -> list[T]:
        if not choices:
            raise ValueError("choices must be a nonempty list")
        if len({repr(choice) for choice in choices}) != len(choices):
            raise ValueError("choices must contain unique values")
        return choices


class NumericRangeSpec(StrictModel):
    min: float = Field(strict=True, allow_inf_nan=False)
    max: float = Field(strict=True, allow_inf_nan=False)
    step: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def _validate_bounds(self) -> NumericRangeSpec:
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError("range bounds must be finite")
        if self.step is not None and not math.isfinite(self.step):
            raise ValueError("range step must be finite")
        if self.min > self.max:
            raise ValueError("range requires min <= max")
        if self.scale == "log":
            if self.min <= 0:
                raise ValueError("log range requires min > 0")
            if self.step is not None:
                raise ValueError("log range rejects step")
        return self


class NumericRange(StrictModel):
    range: NumericRangeSpec


class IntegerRangeSpec(StrictModel):
    min: int = Field(strict=True)
    max: int = Field(strict=True)
    step: int | None = Field(default=None, strict=True, gt=0)
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def _validate_bounds(self) -> IntegerRangeSpec:
        if self.min > self.max:
            raise ValueError("range requires min <= max")
        if self.scale == "linear" and self.step is None:
            raise ValueError("integer linear range requires step")
        if self.scale == "log":
            if self.min <= 0:
                raise ValueError("integer log range requires min > 0")
            if self.step is not None:
                raise ValueError("integer log range rejects step")
        return self


class IntegerRange(StrictModel):
    range: IntegerRangeSpec


class SlaConfig(StrictModel):
    ttft_ms: PositiveFiniteFloat | None = None
    itl_ms: PositiveFiniteFloat | None = None
    e2e_ms: PositiveFiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_form(self) -> SlaConfig:
        token_form = self.ttft_ms is not None or self.itl_ms is not None
        if token_form and self.e2e_ms is not None:
            raise ValueError("e2e_ms is mutually exclusive with ttft_ms/itl_ms")
        return self

    @property
    def has_bound(self) -> bool:
        return any(value is not None for value in (self.ttft_ms, self.itl_ms, self.e2e_ms))


class EvaluationConfig(StrictModel):
    sla: SlaConfig | None = None


class ResourceConfig(StrictModel):
    """Execution-host limits, independent of the simulated GPU configuration."""

    initialization_timeout_seconds: PositiveFiniteFloat = 60.0
    shutdown_timeout_seconds: PositiveFiniteFloat = 5.0
    memory_limit_gb: PositiveFiniteFloat | Literal["auto"] = "auto"
    cpu_limit: PositiveStrictInt | Literal["auto"] = "auto"
    reserve_memory_gb: float = Field(default=1.0, strict=True, ge=0, allow_inf_nan=False)
    reserve_memory_fraction: float = Field(default=0.0, strict=True, ge=0, lt=1, allow_inf_nan=False)
    available_memory_fraction: float = Field(default=0.9, strict=True, gt=0, le=1, allow_inf_nan=False)


class ExecutionConfig(StrictModel):
    resources: ResourceConfig = Field(default_factory=ResourceConfig)


class CandidateConstraints(StrictModel):
    min_candidate_gpus: PositiveStrictInt | None = None
    max_candidate_gpus: PositiveStrictInt = 32
    min_goodput_rps: PositiveFiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> CandidateConstraints:
        if self.min_candidate_gpus is not None and self.min_candidate_gpus > self.max_candidate_gpus:
            raise ValueError("min_candidate_gpus cannot exceed max_candidate_gpus")
        return self


class OptimizationConfig(StrictModel):
    target: Literal[
        "throughput",
        "throughput_per_gpu",
        "throughput_per_user",
        "goodput",
        "goodput_per_gpu",
        "min_gpus",
        "ttft",
        "e2e_latency",
        "pareto",
    ] = "throughput"
    hardware: str | None = None
    strict_sla: bool = Field(default=False, strict=True)
    constraints: CandidateConstraints = Field(default_factory=CandidateConstraints)

    @model_validator(mode="after")
    def _validate_min_goodput(self) -> OptimizationConfig:
        if self.constraints.min_goodput_rps is not None and self.target != "min_gpus":
            raise ValueError("min_goodput_rps is only supported with min_gpus")
        return self

    @field_validator("hardware")
    @classmethod
    def _validate_hardware(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or normalized == "auto":
            raise ValueError("optimization.hardware must be a concrete nonempty identifier")
        return normalized


class OptimizerConfig(StrictModel):
    algorithm: Literal["bayesian", "random"] = "bayesian"
    max_trials: PositiveStrictInt = 320
    parallelism: PositiveStrictInt = 16
    candidate_timeout_seconds: PositiveFiniteFloat = 600.0
    seed: NonNegativeStrictInt = 42


def load_yaml(path: str | Path) -> dict[str, Any]:
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


PREDICTION_CORE_SECTIONS = frozenset({"traffic", "engine", "evaluation", "execution"})
RECOMMENDATION_CORE_SECTIONS = frozenset({*PREDICTION_CORE_SECTIONS, "optimization", "optimizer"})


def split_config_sections(
    data: dict[str, Any], *, command: Literal["predict", "recommend"]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Split core fields from present adapter-owned top-level sections."""

    core_names = PREDICTION_CORE_SECTIONS if command == "predict" else RECOMMENDATION_CORE_SECTIONS
    if command == "predict":
        forbidden = sorted(set(data).intersection({"optimization", "optimizer"}))
        if forbidden:
            raise ValueError(f"predict does not accept {forbidden}")
    core = {name: value for name, value in data.items() if name in core_names}
    adapters: dict[str, dict[str, Any]] = {}
    for section, value in data.items():
        if not isinstance(section, str) or not section or "." in section:
            raise ValueError(
                f"top-level configuration keys must be nonempty section names without dots; got {section!r}"
            )
        if section in core_names or section in {"optimization", "optimizer"}:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"adapter section {section!r} must be a mapping")
        adapters[section] = deepcopy(value)
    return core, adapters


# Engine identity controls forwarded unchanged to the canonical Core constructor.
ENGINE_MODEL_CONTROL_FIELDS = (
    "enable_eplb",
    "wideep_num_slots",
    "moe_backend",
    "attention_backend",
    "gemm_quant_mode",
    "moe_quant_mode",
    "kvcache_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
)


def is_active_engine_model_control(name: str, value: Any) -> bool:
    """Distinguish inactive defaults without treating invalid numeric zero as False."""
    if value is None:
        return False
    if name == "enable_eplb":
        return value is not False
    if name == "moe_backend":
        return value != "default"
    return True


def omit_inactive_moe_controls(config: dict[str, Any]) -> dict[str, Any]:
    """Keep additive defaults out of timing payloads parsed by older runners."""
    result = dict(config)
    for name in ("moe_backend", "wideep_num_slots", "enable_eplb"):
        if not is_active_engine_model_control(name, result.get(name)):
            result.pop(name, None)
    return result
