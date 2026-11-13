# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level core envelopes for the two AISimulate commands."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .ais_perf_model import anchor_paths, reject_repeated_keys, warn_old_keys
from .common import (
    Choices,
    EvaluationConfig,
    ExecutionConfig,
    OptimizationConfig,
    OptimizerConfig,
    StrictModel,
    load_yaml_with_text,
)
from .engine import EnginePredictionConfig, EngineRecommendationConfig, has_afd, require_native_vl_parallelism
from .traffic import (
    SyntheticSessionSource,
    SyntheticSource,
    TraceSource,
    TrafficPredictionConfig,
    TrafficRecommendationConfig,
)


def config_folder(path: str | os.PathLike[str]) -> str:
    """Return the folder that relative paths written in a config file start from.

    Symbolic links in the folder are resolved first, so ``..`` in such a path
    means the parent of the folder that really holds the file, whichever
    link the file was opened through.
    """
    # realpath walks the folder part the way the file is opened. Making the path
    # absolute first would drop "link/.." as text and name a folder the file is not in.
    return os.path.realpath(os.path.dirname(path) or os.curdir)


def _read_config_file(path: str | os.PathLike[str], kind: str) -> dict[str, Any]:
    """Read a config file and make the relative paths in ``ais_perf_model`` absolute.

    The file then loads the same way from any working directory. A key written
    twice inside ``ais_perf_model`` is an error. ``kind`` is ``predict`` or
    ``recommend``.
    """
    raw, text = load_yaml_with_text(path)
    reject_repeated_keys(text, kind, path)
    return anchor_paths(raw, config_folder(path), kind)


class CorePredictionConfig(StrictModel):
    traffic: TrafficPredictionConfig = Field(default_factory=TrafficPredictionConfig.default)
    engine: EnginePredictionConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def _validate_cross_component(self) -> CorePredictionConfig:
        _validate_epd(self.traffic, self.engine)
        source = self.traffic.source
        # The native AFD path serves trace and prefix-cached workloads.
        if isinstance(source, TraceSource) and source.format == "mooncake-delta" and self.engine.mode != "aggregated":
            raise ValueError(f"{source.format} requires aggregated engine mode")
        if (
            self.engine.detokenization is not None
            and self.engine.detokenization.active
            and self.traffic.load.agentic_profile is not None
        ):
            raise ValueError("engine.detokenization cannot be combined with traffic.load.agentic_profile")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> CorePredictionConfig:
        """Load a prediction config file.

        Relative paths inside ``engine.ais_perf_model`` start at the folder of
        the file, and a key written twice inside it is an error. Old estimator
        fields in the file give one deprecation warning, raised only after the
        file validated.
        """
        raw = _read_config_file(path, "predict")
        config = cls.model_validate(raw)
        warn_old_keys(raw, "predict", source=path)
        return config


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
        _validate_recommendation_context_lengths(self.traffic, self.engine)
        if self.engine.hardware == "auto" and self.optimization.hardware is None:
            raise ValueError("engine.hardware='auto' requires one optimization.hardware")
        source = self.traffic.source if self.traffic is not None else None
        modes = set(self.engine.mode.choices) if hasattr(self.engine.mode, "choices") else {self.engine.mode}
        if (
            has_afd(self.engine.workers)
            and self.traffic is not None
            and self.traffic.load.type == "kv_capacity_fraction"
        ):
            raise ValueError("AFD recommendation requires an absolute traffic load, not kv_capacity_fraction")
        if isinstance(source, TraceSource) and source.format == "mooncake-delta" and "disaggregated" in modes:
            raise ValueError(f"{source.format} requires aggregated engine mode")
        sla = self.evaluation.sla
        if (
            self.optimization.strict_sla
            and (sla is None or not sla.has_bound)
            and not self.optimization.constraints.metrics
        ):
            raise ValueError("optimization.strict_sla requires at least one evaluation.sla bound or metric constraint")
        if self.optimization.target in {"goodput", "goodput_per_gpu", "min_gpus"} and (
            sla is None or not sla.has_bound
        ):
            raise ValueError(f"optimization target {self.optimization.target!r} requires an evaluation.sla bound")
        if self.optimization.target == "min_gpus":
            load = self.traffic.load if self.traffic is not None else None
            if load is None or load.type == "kv_capacity_fraction":
                raise ValueError("min_gpus requires a fixed absolute traffic load")
            if load.type == "concurrency":
                value = load.concurrency
            elif load.type == "trace_timestamps":
                value = 1.0 if load.speedup is None else load.speedup
            else:
                value = load.requests_per_second if isinstance(source, SyntheticSource) else load.sessions_per_second
            if type(value) not in (int, float):
                raise ValueError("min_gpus requires a fixed absolute traffic load")
            minimum = self.optimization.constraints.min_goodput_rps
            open_loop = load.type in {"constant_rate", "poisson"} or (
                load.type == "trace_timestamps" and load.agentic_lanes is None
            )
            if open_loop and minimum is None:
                raise ValueError("min_gpus with open-loop traffic requires constraints.min_goodput_rps")
            if load.type in {"constant_rate", "poisson"}:
                # A session rate counts sessions; each session offers its turns as requests.
                turns = source.session.turns if isinstance(source, SyntheticSessionSource) else 1
                if minimum > value * turns:
                    raise ValueError("min_goodput_rps cannot exceed the offered request rate")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> CoreRecommendationConfig:
        """Load a recommendation config file.

        Relative paths inside ``engine.ais_perf_model`` start at the folder of
        the file, and a key written twice inside it is an error. Old estimator
        fields in the file give one deprecation warning, raised only after the
        file validated.
        """
        raw = _read_config_file(path, "recommend")
        config = cls.model_validate(raw)
        warn_old_keys(raw, "recommend", source=path)
        return config


def _only(value: Any, expected: str) -> bool:
    """Whether a concrete or single-choice engine field resolves to `expected`."""
    return value == expected or (isinstance(value, Choices) and list(value.choices) == [expected])


def _validate_epd(traffic, engine) -> None:
    encoder = engine.workers.encoder
    source = traffic.source if traffic is not None else None
    images = source.images if isinstance(source, SyntheticSource) else None
    if has_afd(engine.workers):
        if encoder is not None or images is not None:
            raise ValueError("AFD does not support encoder pools or image workloads")
        return
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
    if images is None and language is not None and language.vision is not None:
        role = "aggregated" if engine.workers.aggregated is not None else "prefill"
        raise ValueError(f"workers.{role}.vision requires traffic.source.images")
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
    if images.identity != "unique":
        raise ValueError(
            "encoder pools do not model image identity pools; traffic.source.images.identity must be unique"
        )
    if encoder.mode == "native":
        if not _only(engine.backend, "sglang"):
            raise ValueError("native encoder replay requires backend=sglang")
        for role in ("aggregated", "prefill", "decode"):
            worker = getattr(engine.workers, role)
            if worker is None:
                continue
            if worker.timing.type != "default":
                raise ValueError("native encoder replay requires default language timing")
            if worker.startup_seconds != 0:
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


def _validate_recommendation_context_lengths(traffic, engine) -> None:
    source = traffic.source if traffic is not None else SyntheticSource()
    if not isinstance(source, SyntheticSource):
        return
    isl = source.input_tokens
    osl = source.output_tokens
    limits = {
        "prefill": isl + 1,
        "decode": isl + osl,
    }
    for role, minimum in limits.items():
        worker = getattr(engine.workers, role)
        if worker is not None and worker.context_length is not None and worker.context_length < minimum:
            raise ValueError(
                f"engine.workers.{role}.context_length must be at least {minimum} for the configured workload"
            )


def prediction_mapping(
    config: CorePredictionConfig,
    adapter_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data = config.model_dump(mode="python", exclude_none=True)
    data.update(deepcopy(adapter_configs or {}))
    return data
