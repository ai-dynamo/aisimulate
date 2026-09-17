# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed CLI configuration owned by AISimulate core."""

from .cli import CorePredictionConfig, CoreRecommendationConfig
from .engine import (
    AFDSearchRecommendationConfig,
    AFDTopologyPredictionConfig,
    EnginePredictionConfig,
    EngineRecommendationConfig,
)
from .traffic import TrafficPredictionConfig, TrafficRecommendationConfig

__all__ = [
    "AFDSearchRecommendationConfig",
    "AFDTopologyPredictionConfig",
    "CorePredictionConfig",
    "CoreRecommendationConfig",
    "EnginePredictionConfig",
    "EngineRecommendationConfig",
    "TrafficPredictionConfig",
    "TrafficRecommendationConfig",
]
