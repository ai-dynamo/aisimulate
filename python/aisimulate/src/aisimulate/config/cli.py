# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level core envelopes for the two AISimulate commands."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .common import (
    EvaluationConfig,
    OptimizationConfig,
    OptimizerConfig,
    StrictModel,
    load_yaml,
)
from .engine import EnginePredictionConfig, EngineRecommendationConfig
from .traffic import (
    TraceSource,
    TrafficPredictionConfig,
    TrafficRecommendationConfig,
)


class CorePredictionConfig(StrictModel):
    traffic: TrafficPredictionConfig = Field(default_factory=TrafficPredictionConfig.default)
    engine: EnginePredictionConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _validate_cross_component(self) -> CorePredictionConfig:
        source = self.traffic.source
        if (
            isinstance(source, TraceSource)
            and source.format in {"mooncake-delta", "agentic_mooncake"}
            and self.engine.mode != "aggregated"
        ):
            raise ValueError(f"{source.format} requires aggregated engine mode")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> CorePredictionConfig:
        return cls.model_validate(load_yaml(path))


class CoreRecommendationConfig(StrictModel):
    traffic: TrafficRecommendationConfig | None = None
    engine: EngineRecommendationConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    optimization: OptimizationConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @model_validator(mode="after")
    def _validate_cross_component(self) -> CoreRecommendationConfig:
        if self.engine.hardware == "auto" and self.optimization.hardware is None:
            raise ValueError("engine.hardware='auto' requires one optimization.hardware")
        source = self.traffic.source if self.traffic is not None else None
        modes = set(self.engine.mode.choices) if hasattr(self.engine.mode, "choices") else {self.engine.mode}
        if (
            isinstance(source, TraceSource)
            and source.format in {"mooncake-delta", "agentic_mooncake"}
            and "disaggregated" in modes
        ):
            raise ValueError(f"{source.format} requires aggregated engine mode")
        sla = self.evaluation.sla
        if self.optimization.strict_sla and (sla is None or not sla.has_bound):
            raise ValueError("optimization.strict_sla requires at least one evaluation.sla bound")
        if self.optimization.target in {"goodput", "goodput_per_gpu"} and (sla is None or not sla.has_bound):
            raise ValueError(f"optimization target {self.optimization.target!r} requires an evaluation.sla bound")
        min_goodput_rps = self.optimization.constraints.min_goodput_rps
        if min_goodput_rps is not None and (sla is None or not sla.has_bound):
            raise ValueError("optimization.constraints.min_goodput_rps requires an evaluation.sla bound")
        if self.optimization.target == "min_gpus":
            if min_goodput_rps is None:
                raise ValueError("optimization target 'min_gpus' requires constraints.min_goodput_rps")
            if self.traffic is None or self.traffic.source.type != "synthetic":
                raise ValueError("optimization target 'min_gpus' requires fixed synthetic request-rate traffic")
            load = self.traffic.load
            offered_rps = load.requests_per_second
            if load.type not in {"constant_rate", "poisson"} or not isinstance(offered_rps, (int, float)):
                raise ValueError("optimization target 'min_gpus' requires one concrete requests_per_second value")
            if min_goodput_rps > float(offered_rps):
                raise ValueError("constraints.min_goodput_rps cannot exceed the offered requests_per_second")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> CoreRecommendationConfig:
        return cls.model_validate(load_yaml(path))


def prediction_mapping(
    config: CorePredictionConfig,
    adapter_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data = config.model_dump(mode="python", exclude_none=True)
    data.update(deepcopy(adapter_configs or {}))
    return data
