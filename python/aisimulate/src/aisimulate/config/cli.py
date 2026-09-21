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
    ExecutionConfig,
    OptimizationConfig,
    OptimizerConfig,
    StrictModel,
    load_yaml,
)
from .engine import EnginePredictionConfig, EngineRecommendationConfig, native_vl_worker
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
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def _validate_cross_component(self) -> CorePredictionConfig:
        _validate_epd(self.traffic, self.engine)
        source = self.traffic.source
        if self.engine.mode == "afd" and isinstance(source, SyntheticSource) and source.cached_prefix_tokens:
            raise ValueError("cached_prefix_tokens is unsupported for AFD")
        if self.engine.mode == "afd" and not isinstance(source, SyntheticSource):
            raise ValueError("AFD prediction requires fixed-length synthetic request traffic")
        if (
            isinstance(source, TraceSource)
            and source.format in {"mooncake-delta", "agentic_mooncake", "weka"}
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
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    optimization: OptimizationConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @model_validator(mode="after")
    def _validate_cross_component(self) -> CoreRecommendationConfig:
        _validate_epd(self.traffic, self.engine)
        if self.engine.hardware == "auto" and self.optimization.hardware is None:
            raise ValueError("engine.hardware='auto' requires one optimization.hardware")
        source = self.traffic.source if self.traffic is not None else None
        modes = set(self.engine.mode.choices) if hasattr(self.engine.mode, "choices") else {self.engine.mode}
        if "afd" in modes and isinstance(source, SyntheticSource) and source.cached_prefix_tokens:
            raise ValueError("cached_prefix_tokens is unsupported for AFD")
        if "afd" in modes and source is not None and not isinstance(source, SyntheticSource):
            raise ValueError("AFD recommendation requires fixed-length synthetic request traffic")
        if "afd" in modes and self.traffic is not None and self.traffic.load.type == "kv_capacity_fraction":
            raise ValueError("AFD recommendation requires an absolute traffic load, not kv_capacity_fraction")
        if (
            isinstance(source, TraceSource)
            and source.format in {"mooncake-delta", "agentic_mooncake", "weka"}
            and "disaggregated" in modes
        ):
            raise ValueError(f"{source.format} requires aggregated engine mode")
        sla = self.evaluation.sla
        if self.optimization.strict_sla and (sla is None or not sla.has_bound):
            raise ValueError("optimization.strict_sla requires at least one evaluation.sla bound")
        if self.optimization.target in {"goodput", "goodput_per_gpu", "min_gpus"} and (
            sla is None or not sla.has_bound
        ):
            raise ValueError(f"optimization target {self.optimization.target!r} requires an evaluation.sla bound")
        if self.optimization.target == "min_gpus":
            if self.traffic is None or not isinstance(source, SyntheticSource):
                raise ValueError("min_gpus requires fixed synthetic request-rate or concurrency traffic")
            load = self.traffic.load
            value = load.concurrency if load.type == "concurrency" else load.requests_per_second
            if load.type not in {"concurrency", "constant_rate", "poisson"} or type(value) not in (int, float):
                raise ValueError("min_gpus requires fixed synthetic request-rate or concurrency traffic")
            minimum = self.optimization.constraints.min_goodput_rps
            if load.type != "concurrency":
                if minimum is None:
                    raise ValueError("min_gpus with request-rate traffic requires constraints.min_goodput_rps")
                if minimum > value:
                    raise ValueError("min_goodput_rps cannot exceed the offered request rate")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> CoreRecommendationConfig:
        return cls.model_validate(load_yaml(path))


def _validate_epd(traffic, engine) -> None:
    encoder = engine.workers.encoder
    source = traffic.source if traffic is not None else None
    images = source.images if isinstance(source, SyntheticSource) else None
    native_vl = native_vl_worker(engine)
    if encoder is not None and native_vl is not None:
        raise ValueError(
            "engine.workers.encoder (analytical EPD) and workers.aggregated.host_loop (native VL replay) are exclusive"
        )
    if encoder is None and images is not None:
        if native_vl is None:
            raise ValueError(
                "image workloads require engine.workers.encoder (analytical EPD) or an aggregated SGLang worker "
                "with host_loop enabled (native VL replay)"
            )
        if native_vl.timing.type != "default":
            raise ValueError("native VL replay requires default timing")
        return
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
