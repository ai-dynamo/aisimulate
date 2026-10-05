# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aisimulate.config_adapter import (
    ConfigAdapterResolutionError,
    resolve_config_adapters,
)
from aisimulate.stack import (
    DuplicateStackError,
    StackNotFoundError,
    resolve_runner_factory,
)
from aisimulate.sweeper.provider import AdapterReplaySpec, AdapterSearchPlan
from aisimulate.sweeper.replay import RunnerCapabilities


class _EntryPoint:
    def __init__(self, name, value, loaded):
        self.name = name
        self.value = value
        self.dist = "test-dist"
        self._loaded = loaded

    def load(self):
        return self._loaded


class _RunnerFactory:
    def capabilities(self):
        return RunnerCapabilities()

    def create(self, worker_id):
        del worker_id
        return object()


class _Adapter:
    name = "dynamo.router"
    section = "router"
    config_adapter_api_version = 3
    api_version = 1

    def compile_prediction(self, config, context):
        del config, context
        return AdapterReplaySpec()

    def compile_recommendation(self, search_spec, context):
        del search_spec, context
        return AdapterSearchPlan()

    def materialize_candidate(self, plan, selection, context):
        del plan, selection, context
        return AdapterReplaySpec()


def test_engine_stack_is_builtin() -> None:
    assert resolve_runner_factory("engine").capabilities() is not None


def test_optional_stack_is_loaded_lazily() -> None:
    entry = _EntryPoint("dynamo", "example:create", _RunnerFactory)

    assert isinstance(resolve_runner_factory("dynamo", entry_points=[entry]), _RunnerFactory)


def test_router_requires_the_selected_stack_plugin() -> None:
    class UnusedEntryPoint(_EntryPoint):
        def load(self):
            pytest.fail("engine must not load the external Dynamo integration")

    entry = UnusedEntryPoint("dynamo.router", "dynamo:adapter", _Adapter)
    with pytest.raises(ConfigAdapterResolutionError, match="'engine.router' is unavailable"):
        resolve_config_adapters(["engine.router"], entry_points=[entry])
    assert isinstance(
        resolve_config_adapters(
            ["dynamo.router"], entry_points=[_EntryPoint("dynamo.router", "dynamo:adapter", _Adapter)]
        )["dynamo.router"],
        _Adapter,
    )


def test_missing_and_duplicate_stacks_fail_explicitly() -> None:
    with pytest.raises(StackNotFoundError, match="installed stacks"):
        resolve_runner_factory("missing", entry_points=[])
    entries = [
        _EntryPoint("dynamo", "first:create", _RunnerFactory),
        _EntryPoint("dynamo", "second:create", _RunnerFactory),
    ]
    with pytest.raises(DuplicateStackError, match="multiple providers"):
        resolve_runner_factory("dynamo", entry_points=entries)


@pytest.mark.parametrize("version", [None, True, 2, 4])
def test_router_plugin_rejects_incompatible_config_api(version) -> None:
    class IncompatibleAdapter(_Adapter):
        config_adapter_api_version = version

    entry = _EntryPoint("dynamo.router", "dynamo:adapter", IncompatibleAdapter)
    with pytest.raises(ConfigAdapterResolutionError, match="config API version"):
        resolve_config_adapters(["dynamo.router"], entry_points=[entry])


def test_config_adapter_requires_predict_and_recommend_methods() -> None:
    entry = _EntryPoint("dynamo.router", "example:create", _Adapter)
    resolved = resolve_config_adapters(["dynamo.router"], entry_points=[entry])

    assert isinstance(resolved["dynamo.router"], _Adapter)

    class Incomplete:
        name = "dynamo.router"
        section = "router"
        config_adapter_api_version = 3
        api_version = 1

    broken = _EntryPoint("dynamo.router", "broken:create", Incomplete)
    with pytest.raises(ConfigAdapterResolutionError, match="missing callable"):
        resolve_config_adapters(["dynamo.router"], entry_points=[broken])
