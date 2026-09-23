# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure native Dynamo routing through AISimulate's existing runtime."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field

from .config_adapter import PredictionAdapterContext, RecommendationAdapterContext
from .runner import EngineReplayRunner, EngineReplayRunnerFactory, InvalidRunnerError
from .sweeper.provider import AdapterReplaySpec, AdapterSearchPlan, CandidateContext, JSONValue, RuntimeHookSpec
from .sweeper.replay import HookCapability, ReplayOutputRequirements, ReplayReport, ReplaySpec, RunnerCapabilities

PROVIDER = "dynamo-policy.router"
HOOK_KIND = "placement_policy"
HOOK_API_VERSION = 1


class AffinityConfig(BaseModel):
    """Conversation grouping layered on the native KV-aware selector."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["session", "sibling_group"]
    ttl_seconds: float = Field(default=3600, strict=True, ge=1, le=31_536_000, allow_inf_nan=False)


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["kv_router"]
    affinity: AffinityConfig | None = None


def validate_router_config(config: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return RouterConfig.model_validate(config).model_dump(mode="json", exclude_none=True)


class DynamoPolicyConfigAdapter:
    name = PROVIDER
    section = "router"
    config_adapter_api_version = 3

    def compile_prediction(
        self, config: Mapping[str, JSONValue], context: PredictionAdapterContext
    ) -> AdapterReplaySpec:
        del context
        concrete = validate_router_config(config)
        return AdapterReplaySpec(
            config=concrete,
            runtime_hooks=(
                RuntimeHookSpec(
                    provider=PROVIDER,
                    kind=HOOK_KIND,
                    api_version=HOOK_API_VERSION,
                    config=dict(concrete),
                ),
            ),
        )

    def compile_recommendation(
        self, config: Mapping[str, JSONValue], context: RecommendationAdapterContext
    ) -> AdapterSearchPlan:
        del config, context
        raise ValueError("dynamo-policy supports offline predict only; routing recommendation is not supported")

    def materialize_candidate(
        self, plan: AdapterSearchPlan, selection: Mapping[str, JSONValue], context: CandidateContext
    ) -> AdapterReplaySpec:
        del plan, selection, context
        raise ValueError("dynamo-policy supports offline predict only; routing recommendation is not supported")


DYNAMO_REVISION = "d9eb42db1168131fdae318eef77255637e4d3495"
_INSTALL_HINT = "Reinstall or rebuild the aisimulate wheel; see docs/agentx-quickstart.md."


def _load_native() -> Any:
    """Validate the single wheel's Python/native versions and pinned policy."""
    try:
        native = importlib.import_module("aisimulate._runtime")
        contract = native.native_replay_contract()
        if not isinstance(contract, Mapping):
            raise TypeError("native_replay_contract must return a mapping")
        versions = {
            Version(importlib.metadata.version("aisimulate")),
            Version(contract["core_version"]),
            Version(contract["binding_version"]),
        }
    except (ImportError, AttributeError, KeyError, TypeError, InvalidVersion) as exc:
        raise InvalidRunnerError(
            f"Native Dynamo routing is unavailable or incompatible: {exc}. {_INSTALL_HINT}"
        ) from exc
    if len(versions) != 1:
        raise InvalidRunnerError(
            f"Mismatched Python/native package versions: {sorted(map(str, versions))}. {_INSTALL_HINT}"
        )
    if type(contract.get("api_version")) is not int or contract["api_version"] != 1:
        raise InvalidRunnerError(f"Dynamo routing requires native replay API version 1. {_INSTALL_HINT}")
    if contract.get("dynamo_revision") != DYNAMO_REVISION:
        raise InvalidRunnerError(f"Dynamo routing requires Dynamo revision {DYNAMO_REVISION}. {_INSTALL_HINT}")
    if not callable(getattr(native, "run_dynamo_replay_json", None)):
        raise InvalidRunnerError(f"Native runtime is missing run_dynamo_replay_json. {_INSTALL_HINT}")
    return native


def _capabilities() -> RunnerCapabilities:
    # Preserve canonical engine controls as that same implementation executes them.
    return replace(
        EngineReplayRunnerFactory().capabilities(),
        supported_backend_topologies=(("vllm", "agg"), ("vllm", "disagg"), ("sglang", "agg"), ("sglang", "disagg")),
        supported_hooks=(HookCapability(PROVIDER, HOOK_KIND, HOOK_API_VERSION),),
        supported_execution_modes=("offline",),
        supports_analytical_epd=False,
    )


@dataclass(frozen=True)
class DynamoPolicyRunnerFactory:
    trace_block_size: int = 512

    def __post_init__(self) -> None:
        _load_native()

    def capabilities(self) -> RunnerCapabilities:
        return _capabilities()

    def create(self, worker_id: int) -> DynamoPolicyReplayRunner:
        return DynamoPolicyReplayRunner(worker_id, self.trace_block_size, _load_native())


@dataclass
class _ConfiguredRuntime:
    native: Any
    router_config_json: str

    def run_replay_json(self, execution_spec_json: str) -> str:
        return self.native.run_dynamo_replay_json(execution_spec_json, self.router_config_json)


@dataclass
class DynamoPolicyReplayRunner:
    worker_id: int
    trace_block_size: int
    native: Any

    def run(self, spec: ReplaySpec, *, output_requirements: ReplayOutputRequirements | None = None) -> ReplayReport:
        capabilities = _capabilities()
        capabilities.require_compatible(spec)
        if set(spec.adapters) != {PROVIDER}:
            raise ValueError("dynamo-policy requires exactly one router section with policy: kv_router")
        adapter = spec.adapters[PROVIDER]
        if len(adapter.runtime_hooks) != 1:
            raise ValueError("dynamo-policy requires exactly one placement_policy runtime hook")
        hook = adapter.runtime_hooks[0]
        config = validate_router_config(adapter.config)
        if validate_router_config(hook.config) != config:
            raise ValueError("router configuration disagrees with its placement_policy runtime hook")
        runtime = _ConfiguredRuntime(self.native, json.dumps(config, allow_nan=False, separators=(",", ":")))
        runner = EngineReplayRunner(
            worker_id=self.worker_id,
            capabilities=capabilities,
            trace_block_size=self.trace_block_size,
            runtime=runtime,
        )
        try:
            return runner.run(spec, output_requirements=output_requirements)
        finally:
            runner.close()

    def close(self) -> None:
        # Each native invocation owns and tears down its selector, cache, and runtime.
        pass
