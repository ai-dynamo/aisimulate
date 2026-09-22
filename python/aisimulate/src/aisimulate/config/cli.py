# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level core envelopes for the two AISimulate commands."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .common import (
    Choices,
    EvaluationConfig,
    ExecutionConfig,
    OptimizationConfig,
    OptimizerConfig,
    StrictModel,
    load_yaml,
)
from .engine import EnginePredictionConfig, EngineRecommendationConfig, require_native_vl_parallelism
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
        if isinstance(source, TraceSource) and source.format == "mooncake-delta" and self.engine.mode != "aggregated":
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
        if isinstance(source, TraceSource) and source.format == "mooncake-delta" and "disaggregated" in modes:
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


def _only(value: Any, expected: str) -> bool:
    """Whether a concrete or single-choice engine field resolves to `expected`."""
    return value == expected or (isinstance(value, Choices) and list(value.choices) == [expected])


def _validate_epd(traffic, engine) -> None:
    encoder = engine.workers.encoder
    source = traffic.source if traffic is not None else None
    images = source.images if isinstance(source, SyntheticSource) else None
    # The language worker that hosts the scheduler loop and the vision tower: aggregated or prefill.
    language, mode = (
        (engine.workers.aggregated, "aggregated")
        if engine.workers.aggregated is not None
        else (engine.workers.prefill, "disaggregated")
    )
    host_aware = language is not None and (language.host_loop or language.vision is not None)
    if encoder is not None and host_aware:
        if encoder.mode == "analytical":
            raise ValueError(
                "engine.workers.encoder (analytical EPD) and the language worker's host_loop or vision "
                "(native VL replay) are exclusive"
            )
        if language.vision is not None or language.frontend is not None or language.host_profile is not None:
            raise ValueError(
                "with a native encoder pool the language worker runs --language-only: it neither hosts the vision "
                "tower nor prices image frontend stages; only host_loop applies"
            )
    if encoder is None and images is not None:
        # Images without an encoder pool are encoded on the aggregated or the prefill SGLang worker.
        if language is None or not _only(engine.backend, "sglang") or not _only(engine.mode, mode):
            raise ValueError(
                "image workloads require engine.workers.encoder (analytical EPD) or an aggregated or prefill worker "
                f"with backend=sglang and concrete mode={mode} (native VL replay)"
            )
        if language.timing.type != "default":
            raise ValueError("native VL replay requires default timing")
        require_native_vl_parallelism(language)
        return
    if (encoder is None) != (images is None):
        raise ValueError("EPD requires both traffic.source.images and engine.workers.encoder")
    if encoder is None:
        return
    if encoder.mode == "native":
        if not _only(engine.backend, "sglang"):
            raise ValueError("native encoder replay requires backend=sglang")
        for role in ("aggregated", "prefill", "decode"):
            worker = getattr(engine.workers, role)
            if worker is not None and worker.startup_seconds != 0:
                raise ValueError("encoder pools require static worker pools")
        return
    if images.min_pixels is not None or images.max_pixels is not None:
        raise ValueError("images.min_pixels and max_pixels are honored only by native VL replay, not by analytical EPD")
    if traffic.load.type != "concurrency" or type(traffic.load.concurrency) is not int:
        raise ValueError("analytical EPD requires fixed synthetic concurrency, not rate or load search")
    for role in ("aggregated", "prefill", "decode"):
        worker = getattr(engine.workers, role)
        if worker is None:
            continue
        if worker.timing.type != "default" or worker.timing.forward_model != "op_level":
            raise ValueError("analytical EPD requires default op_level language timing")
        if worker.startup_seconds != 0:
            raise ValueError("encoder pools require static worker pools")


def prediction_mapping(
    config: CorePredictionConfig,
    adapter_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data = config.model_dump(mode="python", exclude_none=True)
    data.update(deepcopy(adapter_configs or {}))
    return data
