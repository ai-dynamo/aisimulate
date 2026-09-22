# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse AISimulate's runner and substitute only its native replay composition."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from packaging.version import InvalidVersion, Version

from aisimulate.runner import EngineReplayRunner, EngineReplayRunnerFactory, InvalidRunnerError
from aisimulate.sweeper.replay import (
    HookCapability,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
)

from .provider import HOOK_API_VERSION, HOOK_KIND, PROVIDER, validate_router_config

DYNAMO_REVISION = "d9eb42db1168131fdae318eef77255637e4d3495"
_INSTALL_HINT = (
    "Install or rebuild matching aisimulate and aisimulate-dynamo-policy wheels from the same source checkout; "
    "see docs/agentx-quickstart.md."
)


def _contract(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise InvalidRunnerError(f"{name} returned invalid contract JSON. {_INSTALL_HINT}") from exc
    if not isinstance(value, Mapping):
        raise InvalidRunnerError(f"{name} did not return a native contract. {_INSTALL_HINT}")
    return value


def _load_native() -> Any:
    """Fail closed on mixed Python wheels, native ABIs, source trees, or Dynamo pins."""

    try:
        native = importlib.import_module("aisimulate_dynamo_policy._native")
        core = importlib.import_module("aisimulate._runtime")
        plugin_contract = _contract(native.native_contract(), "Dynamo policy plugin")
        core_contract = _contract(core.native_replay_contract(), "AISimulate runtime")
        core_version = Version(importlib.metadata.version("aisimulate"))
        plugin_version = Version(importlib.metadata.version("aisimulate-dynamo-policy"))
        native_core_version = Version(core_contract["core_version"])
        native_plugin_core_version = Version(plugin_contract["core_version"])
        native_plugin_version = Version(plugin_contract["plugin_version"])
    except (ImportError, AttributeError, KeyError, TypeError, InvalidVersion) as exc:
        raise InvalidRunnerError(
            f"Dynamo policy native integration is unavailable or incompatible: {exc}. {_INSTALL_HINT}"
        ) from exc
    versions = {core_version, plugin_version, native_core_version, native_plugin_core_version, native_plugin_version}
    if len(versions) != 1:
        raise InvalidRunnerError(
            f"Dynamo policy integration has mismatched Python/native package versions: {sorted(map(str, versions))}. "
            f"{_INSTALL_HINT}"
        )
    if any(
        type(contract.get("api_version")) is not int or contract["api_version"] != 1
        for contract in (core_contract, plugin_contract)
    ):
        raise InvalidRunnerError(f"Dynamo policy integration requires native replay API version 1. {_INSTALL_HINT}")
    source_hash = core_contract.get("core_source_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
        or source_hash != plugin_contract.get("core_source_sha256")
    ):
        raise InvalidRunnerError(
            f"Dynamo policy integration was built from different AISimulate core sources. {_INSTALL_HINT}"
        )
    if plugin_contract.get("dynamo_revision") != DYNAMO_REVISION:
        raise InvalidRunnerError(
            f"Dynamo policy integration requires Dynamo revision {DYNAMO_REVISION}; "
            f"found {plugin_contract.get('dynamo_revision')!r}. {_INSTALL_HINT}"
        )
    if not callable(getattr(native, "run_replay_json", None)):
        raise InvalidRunnerError(f"Dynamo policy integration is missing run_replay_json. {_INSTALL_HINT}")
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
        return self.native.run_replay_json(execution_spec_json, self.router_config_json)


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
