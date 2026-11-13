# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public package surface for the unified AISim wheel."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aisim")
except PackageNotFoundError:
    __version__ = "0+unknown"
else:
    try:
        version("aisimulate")
    except PackageNotFoundError:
        pass
    else:
        raise ImportError(
            "aisim and aisimulate are both installed and own overlapping files. "
            "Use a fresh environment, or repair this one with:\n"
            "  python -m pip uninstall -y aisimulate\n"
            "  python -m pip install --force-reinstall /path/to/your/selected/aisim.whl\n"
            "Uninstalling aisimulate can remove shared files, so reinstall aisim even if it is already installed. "
            "Update dependencies that still require aisimulate before migrating."
        )

_LAZY_EXPORTS = {
    "AFDCompanionPerformanceModel": ("aisim.runner", "AFDCompanionPerformanceModel"),
    "AFDCompanionTiming": ("aisim.runner", "AFDCompanionTiming"),
    "AICAFDCompanionPerformanceModel": ("aisim.runner", "AICAFDCompanionPerformanceModel"),
    "BackendDeploymentSpec": ("aisim.sweeper.replay", "BackendDeploymentSpec"),
    "CorePredictionConfig": ("aisim.config", "CorePredictionConfig"),
    "CoreRecommendationConfig": ("aisim.config", "CoreRecommendationConfig"),
    "EngineReplayRunner": ("aisim.runner", "EngineReplayRunner"),
    "EngineReplayRunnerFactory": ("aisim.runner", "EngineReplayRunnerFactory"),
    "InvalidRunnerError": ("aisim.runner", "InvalidRunnerError"),
    "ReplayOutputRequirements": ("aisim.sweeper.replay", "ReplayOutputRequirements"),
    "ReplayReport": ("aisim.sweeper.replay", "ReplayReport"),
    "ReplaySpec": ("aisim.sweeper.replay", "ReplaySpec"),
    "Runner": ("aisim.sweeper.replay", "Runner"),
    "RunnerCapabilities": ("aisim.sweeper.replay", "RunnerCapabilities"),
    "RunnerFactory": ("aisim.sweeper.replay", "RunnerFactory"),
    "RunnerUnavailableError": ("aisim.runner", "RunnerUnavailableError"),
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
    "AFDCompanionPerformanceModel",
    "AFDCompanionTiming",
    "AICAFDCompanionPerformanceModel",
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
