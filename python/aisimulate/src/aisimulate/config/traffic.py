# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed traffic input for prediction and recommendation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .common import Choices, IntegerRange, NumericRange, StrictModel

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Ratio = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class ImageInput(StrictModel):
    """Fixed image dimensions and count on every synthetic request."""

    height: PositiveInt
    width: PositiveInt
    count: PositiveInt = 1


class SyntheticSource(StrictModel):
    type: Literal["synthetic"] = "synthetic"
    input_tokens: PositiveInt = 1024
    output_tokens: PositiveInt = 128
    images: ImageInput | None = None


class SessionShape(StrictModel):
    turns: int = Field(default=4, strict=True, ge=2)
    shared_prefix_ratio: Ratio = 0.0
    prefix_groups: NonNegativeInt = 0
    inter_turn_delay_ms: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def _validate_prefix_groups(self) -> SessionShape:
        if self.shared_prefix_ratio > 0.0 and self.prefix_groups == 0:
            raise ValueError("prefix_groups must be positive when shared_prefix_ratio is positive")
        return self


class SyntheticSessionSource(StrictModel):
    type: Literal["synthetic-session"]
    new_input_tokens_per_turn: PositiveInt = 1024
    output_tokens_per_turn: PositiveInt = 128
    session: SessionShape


TraceFormat = Literal[
    "mooncake",
    "mooncake-delta",
    "agentic_mooncake",
    "applied_compute_agentic",
    "dynamo",
    "weka",
]
WekaNestedTimestampBasis = Literal["auto", "absolute", "relative"]


class TraceSource(StrictModel):
    type: Literal["trace"]
    paths: list[str]
    format: TraceFormat = "mooncake"
    block_size: PositiveInt | None = None
    nested_timestamp_basis: WekaNestedTimestampBasis | None = None

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
        if self.format not in {"dynamo", "weka"} and self.block_size is None:
            self.block_size = 512
        if self.nested_timestamp_basis is not None and self.format != "weka":
            raise ValueError("nested_timestamp_basis is only valid for trace format 'weka'")
        return self


TrafficSource = Annotated[
    SyntheticSource | SyntheticSessionSource | TraceSource,
    Field(discriminator="type"),
]


class TrafficStop(StrictModel):
    requests: PositiveInt | None = None
    requests_per_load_unit: PositiveFloat | None = None
    sessions: PositiveInt | None = None
    sessions_per_load_unit: PositiveFloat | None = None
    max_virtual_time_seconds: PositiveFloat | None = None


class TrafficPredictionLoad(StrictModel):
    type: Literal[
        "concurrency",
        "poisson",
        "constant_rate",
        "trace_timestamps",
    ] = "concurrency"
    concurrency: PositiveInt | None = None
    requests_per_second: PositiveFloat | None = None
    sessions_per_second: PositiveFloat | None = None
    seed: NonNegativeInt | None = None
    speedup: PositiveFloat | None = None
    agentic_lanes: PositiveInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_concurrency(cls, value):
        if isinstance(value, dict) and value.get("type", "concurrency") == "concurrency":
            value = dict(value)
            value.setdefault("concurrency", 10)
        return value

    @model_validator(mode="after")
    def _validate_fields_for_type(self) -> TrafficPredictionLoad:
        _validate_load_fields(self)
        return self


class TrafficRecommendationLoad(StrictModel):
    type: Literal[
        "concurrency",
        "poisson",
        "constant_rate",
        "kv_capacity_fraction",
        "trace_timestamps",
    ] = "concurrency"
    concurrency: PositiveInt | Choices[PositiveInt] | IntegerRange | None = None
    requests_per_second: PositiveFloat | Choices[PositiveFloat] | NumericRange | None = None
    sessions_per_second: PositiveFloat | Choices[PositiveFloat] | NumericRange | None = None
    seed: NonNegativeInt | None = None
    fraction: PositiveFloat | Choices[PositiveFloat] | NumericRange | None = None
    speedup: PositiveFloat | Choices[PositiveFloat] | NumericRange | None = None
    agentic_lanes: PositiveInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_concurrency(cls, value):
        if isinstance(value, dict) and value.get("type", "concurrency") == "concurrency":
            value = dict(value)
            value.setdefault("concurrency", 10)
        return value

    @model_validator(mode="after")
    def _validate_fields_for_type(self) -> TrafficRecommendationLoad:
        _validate_load_fields(self)
        for name in (
            "concurrency",
            "requests_per_second",
            "sessions_per_second",
            "fraction",
            "speedup",
            "agentic_lanes",
        ):
            value = getattr(self, name)
            if isinstance(value, (IntegerRange, NumericRange)) and value.range.min <= 0:
                raise ValueError(f"traffic.load.{name} range requires min > 0")
        return self


def _validate_load_fields(load) -> None:
    used = {
        name
        for name in (
            "concurrency",
            "requests_per_second",
            "sessions_per_second",
            "seed",
            "fraction",
            "speedup",
            "agentic_lanes",
        )
        if getattr(load, name, None) is not None
    }
    allowed = {
        "concurrency": {"concurrency"},
        "poisson": {"requests_per_second", "sessions_per_second", "seed"},
        "constant_rate": {"requests_per_second", "sessions_per_second"},
        "kv_capacity_fraction": {"fraction"},
        "trace_timestamps": {"speedup", "agentic_lanes"},
    }[load.type]
    unexpected = used - allowed
    if unexpected:
        raise ValueError(f"traffic.load.type={load.type!r} does not accept {sorted(unexpected)}")
    required = {
        "concurrency": ("concurrency",),
        "poisson": (),
        "constant_rate": (),
        "kv_capacity_fraction": ("fraction",),
        "trace_timestamps": (),
    }[load.type]
    missing = [name for name in required if getattr(load, name, None) is None]
    if missing:
        raise ValueError(f"traffic.load.type={load.type!r} requires {', '.join(missing)}")


class _TrafficConfigBase(StrictModel):
    source: TrafficSource
    stop: TrafficStop | None = None

    def _validate_source_load_stop(self, load) -> None:
        source = self.source
        stop = self.stop
        if isinstance(source, TraceSource):
            if load.type not in {"trace_timestamps", "concurrency"}:
                raise ValueError("trace traffic requires trace_timestamps or concurrency load")
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
            agentic = source.format in {"agentic_mooncake", "weka"}
            if agentic and load.type != "trace_timestamps":
                raise ValueError(f"{source.format} requires trace_timestamps load")
            if source.format == "applied_compute_agentic" and load.type != "concurrency":
                raise ValueError("applied_compute_agentic requires concurrency load")
            if agentic and stop is not None and stop.max_virtual_time_seconds is not None:
                raise ValueError(f"{source.format} does not support max_virtual_time_seconds")
            if load.agentic_lanes is not None and not agentic and source.format != "dynamo":
                raise ValueError("agentic_lanes requires weka, agentic_mooncake, or agentic dynamo input")
            return

        if load.type == "trace_timestamps":
            raise ValueError("synthetic traffic does not accept trace_timestamps load")
        if load.agentic_lanes is not None:
            raise ValueError("synthetic traffic does not accept agentic_lanes")
        rate = load.requests_per_second if isinstance(source, SyntheticSource) else load.sessions_per_second
        wrong_rate = load.sessions_per_second if isinstance(source, SyntheticSource) else load.requests_per_second
        if load.type in {"poisson", "constant_rate"} and rate is None:
            unit = "requests_per_second" if isinstance(source, SyntheticSource) else "sessions_per_second"
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
        if (
            sum(
                value is not None
                for value in (
                    stop.requests,
                    stop.requests_per_load_unit,
                    stop.sessions,
                    stop.sessions_per_load_unit,
                )
            )
            != 1
        ):
            raise ValueError("traffic.stop requires exactly one count condition")


class TrafficPredictionConfig(_TrafficConfigBase):
    load: TrafficPredictionLoad

    @classmethod
    def default(cls) -> TrafficPredictionConfig:
        return cls(
            source=SyntheticSource(),
            load=TrafficPredictionLoad(type="concurrency", concurrency=10),
            stop=TrafficStop(requests=100),
        )

    @model_validator(mode="after")
    def _validate_traffic(self) -> TrafficPredictionConfig:
        self._validate_source_load_stop(self.load)
        return self


class TrafficRecommendationConfig(_TrafficConfigBase):
    load: TrafficRecommendationLoad

    @model_validator(mode="after")
    def _validate_traffic(self) -> TrafficRecommendationConfig:
        self._validate_source_load_stop(self.load)
        return self
