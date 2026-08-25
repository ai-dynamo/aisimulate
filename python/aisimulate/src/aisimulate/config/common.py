# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config primitives for concrete predictions and recommendation domains."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


T = TypeVar("T")


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
    min: float
    max: float
    step: float | None = Field(default=None, gt=0)
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def _validate_bounds(self) -> NumericRangeSpec:
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError("range bounds must be finite")
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
    min: int
    max: int
    step: int = Field(gt=0)
    scale: Literal["linear"] = "linear"

    @model_validator(mode="after")
    def _validate_bounds(self) -> IntegerRangeSpec:
        if self.min > self.max:
            raise ValueError("range requires min <= max")
        return self


class IntegerRange(StrictModel):
    range: IntegerRangeSpec


class SlaConfig(StrictModel):
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


class EvaluationConfig(StrictModel):
    sla: SlaConfig | None = None


class CandidateConstraints(StrictModel):
    min_candidate_gpus: int | None = Field(default=None, gt=0)
    max_candidate_gpus: int = Field(default=32, gt=0)

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


class OptimizerConfig(StrictModel):
    algorithm: Literal["bayesian", "random"] = "bayesian"
    max_trials: int = Field(default=320, gt=0)
    parallelism: int = Field(default=16, gt=0)
    candidate_timeout_seconds: float = Field(default=600.0, gt=0)
    seed: int = Field(default=42, ge=0)


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


PREDICTION_CORE_SECTIONS = frozenset({"traffic", "engine", "evaluation"})
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
