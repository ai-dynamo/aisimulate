# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Predictor-neutral domain types."""

from fpm_accuracy.types.forward_pass import (
    ForwardPassInput,
    ForwardPassIteration,
    ForwardPassMetric,
    WorkloadKind,
)
from fpm_accuracy.types.worker_config import WorkerConfig, WorkerConfigRecord

__all__ = [
    "ForwardPassInput",
    "ForwardPassIteration",
    "ForwardPassMetric",
    "WorkerConfig",
    "WorkerConfigRecord",
    "WorkloadKind",
]
