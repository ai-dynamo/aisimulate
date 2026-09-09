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
    SyntheticSource,
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
        _validate_epd(self.traffic, self.engine)
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
        _validate_epd(self.traffic, self.engine)
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
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> CoreRecommendationConfig:
        return cls.model_validate(load_yaml(path))


def _validate_epd(traffic, engine) -> None:
    encoder = engine.workers.encoder
    source = traffic.source if traffic is not None else None
    images = source.images if isinstance(source, SyntheticSource) else None
    if (encoder is None) != (images is None):
        raise ValueError("EPD requires both traffic.source.images and engine.workers.encoder")
    if encoder is None:
        return
    if traffic.load.type != "concurrency" or type(traffic.load.concurrency) is not int:
        raise ValueError("analytical EPD requires fixed synthetic concurrency, not rate or load search")
    for role in ("aggregated", "prefill", "decode"):
        worker = getattr(engine.workers, role)
        if worker is None:
            continue
        if worker.timing.type != "default" or worker.timing.forward_model != "op_level":
            raise ValueError("analytical EPD requires default op_level language timing")
        if worker.startup_seconds != 0:
            raise ValueError("analytical EPD requires static worker pools")


def prediction_mapping(
    config: CorePredictionConfig,
    adapter_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data = config.model_dump(mode="python", exclude_none=True)
    data.update(deepcopy(adapter_configs or {}))
    return data
