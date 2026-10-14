# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public package surface for the unified AISimulate wheel."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aisimulate")
except PackageNotFoundError:
    __version__ = "0+unknown"

_LAZY_EXPORTS = {
    "BackendDeploymentSpec": ("aisimulate.sweeper.replay", "BackendDeploymentSpec"),
    "CorePredictionConfig": ("aisimulate.config", "CorePredictionConfig"),
    "CoreRecommendationConfig": ("aisimulate.config", "CoreRecommendationConfig"),
    "EngineReplayRunner": ("aisimulate.runner", "EngineReplayRunner"),
    "EngineReplayRunnerFactory": ("aisimulate.runner", "EngineReplayRunnerFactory"),
    "InvalidRunnerError": ("aisimulate.runner", "InvalidRunnerError"),
    "ReplayOutputRequirements": ("aisimulate.sweeper.replay", "ReplayOutputRequirements"),
    "ReplayReport": ("aisimulate.sweeper.replay", "ReplayReport"),
    "ReplaySpec": ("aisimulate.sweeper.replay", "ReplaySpec"),
    "Runner": ("aisimulate.sweeper.replay", "Runner"),
    "RunnerCapabilities": ("aisimulate.sweeper.replay", "RunnerCapabilities"),
    "RunnerFactory": ("aisimulate.sweeper.replay", "RunnerFactory"),
    "RunnerUnavailableError": ("aisimulate.runner", "RunnerUnavailableError"),
}


def __getattr__(name: str) -> object:
    """Load public Python APIs lazily so the native shim cannot form a cycle."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "BackendDeploymentSpec",
    "CorePredictionConfig",
    "CoreRecommendationConfig",
    "EngineReplayRunner",
    "EngineReplayRunnerFactory",
    "InvalidRunnerError",
    "ReplayOutputRequirements",
    "ReplayReport",
    "ReplaySpec",
    "Runner",
    "RunnerCapabilities",
    "RunnerFactory",
    "RunnerUnavailableError",
    "__version__",
]
