# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public package surface for the unified AISimulate wheel."""

from importlib.metadata import version

from .runner import (
    EngineReplayRunner,
    EngineReplayRunnerFactory,
    InvalidRunnerError,
    RunnerUnavailableError,
)
from .sweeper.replay import (
    BackendDeploymentSpec,
    EngineRequestSpec,
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RoleEngineRequestSpec,
    Runner,
    RunnerCapabilities,
    RunnerFactory,
)

__version__ = version("aisimulate")

__all__ = [
    "BackendDeploymentSpec",
    "EngineReplayRunner",
    "EngineReplayRunnerFactory",
    "EngineRequestSpec",
    "InvalidRunnerError",
    "ReplayOutputRequirements",
    "ReplayReport",
    "ReplaySpec",
    "RoleEngineRequestSpec",
    "Runner",
    "RunnerCapabilities",
    "RunnerFactory",
    "RunnerUnavailableError",
    "__version__",
]
