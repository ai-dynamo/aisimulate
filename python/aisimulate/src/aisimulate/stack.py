# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery for optional AISimulate execution stacks."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping
from typing import Any

from .runner import EngineReplayRunnerFactory
from .sweeper.replay import RunnerFactory

RUNNER_FACTORY_ENTRY_POINT_GROUP = "aisimulate.runner_factories"


class StackResolutionError(RuntimeError):
    """A requested execution stack could not be resolved safely."""


class StackNotFoundError(StackResolutionError):
    """No built-in or installed stack has the requested name."""


class DuplicateStackError(StackResolutionError):
    """Multiple installed distributions registered the same stack name."""


class StackLoadError(StackResolutionError):
    """An installed stack entry point could not be loaded or constructed."""


def _validate_runner_factory(value: Any, *, stack: str) -> RunnerFactory:
    missing = [method for method in ("capabilities", "create") if not callable(getattr(value, method, None))]
    if missing:
        raise StackLoadError(
            f"stack {stack!r} returned an invalid RunnerFactory; missing callable(s): " + ", ".join(missing)
        )
    return value


def _installed_entry_points() -> list[importlib.metadata.EntryPoint]:
    return list(importlib.metadata.entry_points().select(group=RUNNER_FACTORY_ENTRY_POINT_GROUP))


def resolve_runner_factory(
    stack: str,
    *,
    builtins: Mapping[str, RunnerFactory] | None = None,
    entry_points: Iterable[importlib.metadata.EntryPoint] | None = None,
) -> RunnerFactory:
    """Resolve one stack without importing any unselected optional package."""

    builtin_factories: dict[str, RunnerFactory] = {
        "engine": EngineReplayRunnerFactory(),
        **dict(builtins or {}),
    }
    if stack in builtin_factories:
        return _validate_runner_factory(builtin_factories[stack], stack=stack)

    installed = list(entry_points) if entry_points is not None else _installed_entry_points()
    matches = [entry_point for entry_point in installed if entry_point.name == stack]
    if not matches:
        available = sorted(set(builtin_factories) | {entry.name for entry in installed})
        choices = ", ".join(available) if available else "<none>"
        if stack == "dynamo-policy":
            raise StackNotFoundError(
                "router configuration requires the optional 'aisimulate-dynamo-policy' package. "
                "Install matching aisimulate and aisimulate-dynamo-policy wheels with "
                "'python -m pip install <aisimulate.whl> <aisimulate_dynamo_policy.whl>'; "
                "see docs/agentx-quickstart.md for the paired source-build instructions. "
                "Routing will not fall back to round-robin."
            )
        raise StackNotFoundError(
            f"stack {stack!r} is unavailable; installed stacks: {choices}. "
            "Install the distribution that provides the requested stack."
        )
    if len(matches) > 1:
        providers = ", ".join(
            sorted(f"{entry.value} ({getattr(entry, 'dist', None) or 'unknown distribution'})" for entry in matches)
        )
        raise DuplicateStackError(
            f"stack {stack!r} has multiple providers in entry-point group "
            f"{RUNNER_FACTORY_ENTRY_POINT_GROUP!r}: {providers}"
        )

    entry = matches[0]
    try:
        factory_or_constructor = entry.load()
        factory = factory_or_constructor() if callable(factory_or_constructor) else factory_or_constructor
    except Exception as exc:
        raise StackLoadError(
            f"failed to load stack {stack!r} from {entry.value!r}: {type(exc).__name__}: {exc}"
        ) from exc
    return _validate_runner_factory(factory, stack=stack)
