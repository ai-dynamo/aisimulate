# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from AISim FPM Gym f934c030afc3a03cb04d8f3ff4709194f7445c98,
# src/aisim_fpm/dashboard/data.py. See ../README.md.
"""Only the measurement heatmap types needed by the public exporter."""

from pydantic import BaseModel, ConfigDict, Field


class DashboardModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HeatmapBin(DashboardModel):
    label: str
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)


class HeatmapCell(DashboardModel):
    x_index: int = Field(ge=0)
    y_index: int = Field(ge=0)
    measured_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    mape_pct: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class Heatmap(DashboardModel):
    x_label: str
    y_label: str
    x_bins: tuple[HeatmapBin, ...]
    y_bins: tuple[HeatmapBin, ...]
    cells: tuple[HeatmapCell, ...]
