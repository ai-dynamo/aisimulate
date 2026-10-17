# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Configuration schema and field mapping adapted from ai-dynamo/dynamo,
# components/src/dynamo/router/simulation/{config.py,provider.py}, revision
# 6d6d36d7bfa63d80a0302df637c08f85523aa5b6 (Apache-2.0):
# https://github.com/ai-dynamo/dynamo/tree/6d6d36d7bfa63d80a0302df637c08f85523aa5b6/components/src/dynamo/router/simulation
# Modified for AISimulate-owned native execution and fixed-policy recommendation.

"""Built-in configuration for Dynamo policy inside the AISimulate replay loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config_adapter import PredictionAdapterContext, RecommendationAdapterContext
from .sweeper.provider import AdapterReplaySpec, AdapterSearchPlan, CandidateContext, JSONValue, RuntimeHookSpec
from .sweeper.replay import ReplaySpec

_NATIVE_HOOK = RuntimeHookSpec(provider="engine.router", kind="placement_policy", api_version=1)
_FiniteNonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class ConversationAffinityConfig(BaseModel):
    """Native conversation affinity whose lifetime follows simulation time."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["session", "sibling_group"]
    ttl_seconds: float = Field(default=3600, strict=True, ge=1, le=31_536_000, allow_inf_nan=False)


class _PrefillLoadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Custom AIS load estimators belong to the optional integration stack.
    type: Literal["none"] = "none"


class RouterConfig(BaseModel):
    """Concrete native policy; parameter search and custom load models are unsupported."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["round_robin", "kv_router"] = "round_robin"
    prefill_load_model: _PrefillLoadModel = Field(default_factory=_PrefillLoadModel)
    overlap_score_credit: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] | None = None
    prefill_load_scale: _FiniteNonnegative | None = None
    temperature: _FiniteNonnegative | None = None
    affinity: ConversationAffinityConfig | None = None

    @field_validator("overlap_score_credit", "prefill_load_scale", "temperature", mode="before")
    @classmethod
    def _reject_explicit_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("Router numeric knobs do not accept null")
        return value

    @model_validator(mode="after")
    def _validate_policy(self) -> RouterConfig:
        if self.policy == "round_robin":
            if self.affinity is not None:
                raise ValueError("router.affinity requires policy='kv_router'")
            knobs = {"overlap_score_credit", "prefill_load_scale", "temperature"} & self.model_fields_set
            if knobs:
                raise ValueError(f"round_robin rejects KV-router knobs {sorted(knobs)}")
        elif self.affinity is not None:
            for name, default in (("overlap_score_credit", 1.0), ("prefill_load_scale", 1.0), ("temperature", 0.0)):
                if getattr(self, name) not in (None, default):
                    raise ValueError("router.affinity requires default KV-router scoring parameters")
        return self


def _compile(config: Mapping[str, JSONValue]) -> AdapterReplaySpec:
    public = RouterConfig.model_validate(config)
    concrete = public.model_dump(mode="json", exclude_none=True)
    if public.policy == "round_robin":
        return AdapterReplaySpec(config=concrete)
    # Omit unspecified fields so the pinned production policy owns its defaults.
    router_config = {
        native: getattr(public, name)
        for name, native in (
            ("overlap_score_credit", "overlap_score_credit"),
            ("prefill_load_scale", "prefill_load_scale"),
            ("temperature", "router_temperature"),
        )
        if getattr(public, name) is not None
    }
    native: dict[str, JSONValue] = {"router_config": router_config}
    if public.affinity is not None:
        native["affinity"] = public.affinity.model_dump(mode="json")
    hook = RuntimeHookSpec(
        provider=_NATIVE_HOOK.provider,
        kind=_NATIVE_HOOK.kind,
        api_version=_NATIVE_HOOK.api_version,
        config={"provider": "dynamo_kv_router", "config": native},
    )
    return AdapterReplaySpec(config=concrete, runtime_hooks=(hook,))


class EngineRouterConfigAdapter:
    """Compile a fixed routing policy without importing any Dynamo Python package."""

    name = "engine.router"
    section = "router"
    config_adapter_api_version = 3

    def compile_prediction(
        self, config: Mapping[str, JSONValue], context: PredictionAdapterContext
    ) -> AdapterReplaySpec:
        del context
        return _compile(config)

    def compile_recommendation(
        self, config: Mapping[str, JSONValue], context: RecommendationAdapterContext
    ) -> AdapterSearchPlan:
        del context
        compiled = _compile(config)
        return AdapterSearchPlan(state=compiled.config, potential_runtime_hooks=compiled.runtime_hooks)

    def materialize_candidate(
        self, plan: AdapterSearchPlan, selection: Mapping[str, JSONValue], context: CandidateContext
    ) -> AdapterReplaySpec:
        del context
        if selection:
            raise ValueError("engine.router supports a fixed policy, not router parameter search")
        if not isinstance(plan.state, Mapping):
            raise ValueError("engine.router search plan requires a concrete configuration")
        return _compile(plan.state)


def native_placement(spec: ReplaySpec) -> dict[str, JSONValue]:
    """Validate the built-in adapter and lower it to the core placement descriptor."""

    unknown = set(spec.adapters) - {"engine.router"}
    if unknown:
        raise ValueError(f"EngineReplayRunner does not support component adapters {sorted(unknown)}")
    adapter = spec.adapters.get("engine.router")
    if adapter is None:
        return {"provider": "round_robin", "config": None}
    expected = _compile(adapter.config)
    if adapter.runtime_hooks != expected.runtime_hooks:
        raise ValueError("engine.router runtime hooks must match the configured routing policy")
    deployment = spec.backend_deployment
    if deployment.deployment_mode not in {"agg", "disagg"} or deployment.encoder is not None:
        raise ValueError("engine.router requires aggregated or disaggregated text replay")
    if not expected.runtime_hooks:
        return {"provider": "round_robin", "config": None}
    if spec.execution_mode != "offline":
        raise ValueError("engine.router requires offline execution")
    if adapter.config.get("affinity") is not None:
        _require_affinity_scope(spec)
    return expected.runtime_hooks[0].config


def _require_affinity_scope(spec: ReplaySpec) -> None:
    deployment = spec.backend_deployment
    if deployment.backend not in {"vllm", "sglang"}:
        raise ValueError("router.affinity requires backend=vllm or sglang")
    if any(hook.kind == "scaling_policy" for hook in spec.runtime_hooks):
        raise ValueError("router.affinity requires static worker pools")
    for args in (deployment.agg_engine_args, deployment.prefill_engine_args, deployment.decode_engine_args):
        if not args:
            continue
        rank = args.get("rank", args)
        if not isinstance(rank, Mapping):
            continue  # The engine descriptor validator reports malformed ranks.
        if any(rank.get(key) is not None for key in ("native_host_offload", "g3_offload")):
            raise ValueError("router.affinity requires device-only KV cache without host or G3 offload")
        if rank.get("kv_cache_groups"):
            raise ValueError("router.affinity does not support grouped KV caches")
        timing = rank.get("timing_model")
        identity = timing.get("config", {}) if isinstance(timing, Mapping) else {}
        if not isinstance(identity, Mapping):
            identity = {}
        if any(
            source.get(key) not in (None, 0)
            for source in (args, rank, identity)
            for key in ("aic_nextn", "nextn", "speculation")
        ):
            raise ValueError("router.affinity requires speculative decoding disabled")
        if any(source.get("startup_time") not in (None, 0) for source in (args, rank)):
            raise ValueError("router.affinity requires static worker pools without startup delay")
