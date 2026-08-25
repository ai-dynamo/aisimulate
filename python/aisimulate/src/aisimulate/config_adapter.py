# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified predict/recommend configuration-adapter ABI and discovery."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .sweeper.provider import API_VERSION as SWEEP_PROVIDER_API_VERSION
from .sweeper.provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    JSONValue,
    SweepContext,
)

CONFIG_ADAPTER_API_VERSION = 3
CONFIG_ADAPTER_ENTRY_POINT_GROUP = "aisimulate.config_adapters"


@dataclass(frozen=True)
class PredictionAdapterContext:
    """Concrete core configuration supplied to a component adapter."""

    engine: Mapping[str, JSONValue]
    traffic: Mapping[str, JSONValue]
    evaluation: Mapping[str, JSONValue]


@dataclass(frozen=True)
class RecommendationAdapterContext:
    """Recommendation core configuration supplied for adapter validation."""

    engine: Mapping[str, JSONValue]
    traffic: Mapping[str, JSONValue]
    evaluation: Mapping[str, JSONValue]
    optimization: Mapping[str, JSONValue]
    sweep: SweepContext


@runtime_checkable
class SimulationConfigAdapter(Protocol):
    """Optional component integration used by both public CLI commands."""

    name: str
    section: str
    config_adapter_api_version: int

    def compile_prediction(
        self,
        config: Mapping[str, JSONValue],
        context: PredictionAdapterContext,
    ) -> AdapterReplaySpec:
        """Validate and compile one concrete section."""

    def compile_recommendation(
        self,
        config: Mapping[str, JSONValue],
        context: RecommendationAdapterContext,
    ) -> AdapterSearchPlan:
        """Validate and compile one section's search plan."""

    def materialize_candidate(
        self,
        plan: AdapterSearchPlan,
        selection: Mapping[str, JSONValue],
        context: CandidateContext,
    ) -> AdapterReplaySpec:
        """Materialize one concrete section and its runtime hooks."""


@dataclass(frozen=True)
class CompiledSweepProvider:
    """Sweeper-v1 bridge around a precompiled config-adapter plan."""

    adapter: SimulationConfigAdapter
    plan: AdapterSearchPlan

    @property
    def name(self) -> str:
        return self.adapter.name

    api_version: int = SWEEP_PROVIDER_API_VERSION

    def generate_search_space(
        self,
        search_spec: Mapping[str, JSONValue],
        context: SweepContext,
    ) -> AdapterSearchPlan:
        del search_spec, context
        return deepcopy(self.plan)

    def materialize_replay(
        self,
        plan: AdapterSearchPlan,
        selection: Mapping[str, JSONValue],
        context: CandidateContext,
    ) -> AdapterReplaySpec:
        return self.adapter.materialize_candidate(plan, selection, context)


class ConfigAdapterResolutionError(RuntimeError):
    """A configured component adapter could not be resolved."""


def validate_config_adapter(adapter: Any, *, requested_name: str) -> SimulationConfigAdapter:
    if getattr(adapter, "name", None) != requested_name:
        raise ConfigAdapterResolutionError(
            f"config adapter {requested_name!r} returned name {getattr(adapter, 'name', None)!r}"
        )
    section = getattr(adapter, "section", None)
    expected_section = requested_name.rsplit(".", 1)[-1]
    if not isinstance(section, str) or not section or "." in section or section != expected_section:
        raise ConfigAdapterResolutionError(
            f"config adapter {requested_name!r} returned section {section!r}; expected {expected_section!r}"
        )
    config_version = getattr(adapter, "config_adapter_api_version", None)
    if type(config_version) is not int or config_version != CONFIG_ADAPTER_API_VERSION:
        raise ConfigAdapterResolutionError(
            f"config adapter {requested_name!r} uses config API version "
            f"{config_version!r}; "
            f"AISimulate requires {CONFIG_ADAPTER_API_VERSION}"
        )
    missing = [
        method
        for method in (
            "compile_prediction",
            "compile_recommendation",
            "materialize_candidate",
        )
        if not callable(getattr(adapter, method, None))
    ]
    if missing:
        raise ConfigAdapterResolutionError(
            f"config adapter {requested_name!r} is missing callable(s): " + ", ".join(missing)
        )
    return adapter


def resolve_config_adapters(
    names: Iterable[str],
    *,
    injected: Mapping[str, SimulationConfigAdapter] | None = None,
    entry_points: Iterable[importlib.metadata.EntryPoint] | None = None,
) -> dict[str, SimulationConfigAdapter]:
    """Resolve selected adapters without importing unrelated packages."""

    requested = list(dict.fromkeys(names))
    injected = injected or {}
    installed = (
        list(entry_points)
        if entry_points is not None
        else list(importlib.metadata.entry_points().select(group=CONFIG_ADAPTER_ENTRY_POINT_GROUP))
    )
    resolved: dict[str, SimulationConfigAdapter] = {}
    for name in requested:
        if name in injected:
            resolved[name] = validate_config_adapter(injected[name], requested_name=name)
            continue
        matches = [entry for entry in installed if entry.name == name]
        if not matches:
            available = sorted(set(injected) | {entry.name for entry in installed})
            raise ConfigAdapterResolutionError(
                f"config adapter {name!r} is unavailable; installed adapters: "
                f"{', '.join(available) if available else '<none>'}"
            )
        if len(matches) > 1:
            raise ConfigAdapterResolutionError(f"config adapter {name!r} has multiple installed providers")
        entry = matches[0]
        try:
            constructor = entry.load()
            adapter = constructor() if callable(constructor) else constructor
        except Exception as exc:
            raise ConfigAdapterResolutionError(
                f"failed to load config adapter {name!r} from {entry.value!r}: {type(exc).__name__}: {exc}"
            ) from exc
        resolved[name] = validate_config_adapter(adapter, requested_name=name)
    return resolved
