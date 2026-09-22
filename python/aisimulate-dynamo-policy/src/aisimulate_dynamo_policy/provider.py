# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the public routing section without implementing a routing algorithm."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aisimulate.config_adapter import PredictionAdapterContext, RecommendationAdapterContext
from aisimulate.sweeper.provider import (
    AdapterReplaySpec,
    AdapterSearchPlan,
    CandidateContext,
    JSONValue,
    RuntimeHookSpec,
)

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
