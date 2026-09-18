# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Package-specific exceptions with actionable user-facing messages."""


class FpmGymError(Exception):
    """Base class for expected AISim FPM Gym failures."""


class ConfigurationError(FpmGymError):
    """The evaluation or worker configuration is invalid."""


class DataError(FpmGymError):
    """The source data violates the evaluator's contract."""


class DependencyError(FpmGymError):
    """An optional runtime dependency is unavailable."""
