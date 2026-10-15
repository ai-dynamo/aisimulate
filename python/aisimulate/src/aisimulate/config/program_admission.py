# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified defaults derived from ai-dynamo/dynamo, Apache-2.0:
# https://github.com/ai-dynamo/dynamo/blob/65a27b73b1aa35c0655e3eb0fbc50e327e2aa5d9/components/src/dynamo/thunderagent_router/router.py
"""Public configuration for the offline program admission behavioral model."""

from __future__ import annotations

from pydantic import Field, model_validator

from .common import StrictModel


class ProgramAdmissionConfig(StrictModel):
    capacity_tokens_per_worker: int = Field(strict=True, gt=0)
    admission_control: bool = Field(default=True, strict=True)
    pause_threshold: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    pause_target: float = Field(default=0.80, ge=0, le=1, allow_inf_nan=False)
    resume_hysteresis: float = Field(default=0.10, ge=0, le=1, allow_inf_nan=False)
    resume_timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    scheduler_interval_seconds: float = Field(default=5, gt=0, allow_inf_nan=False)
    acting_token_weight: float = Field(default=1, ge=0, le=1, allow_inf_nan=False)
    acting_decay_tau_seconds: float = Field(default=1, gt=0, allow_inf_nan=False)
    buffer_per_program: int = Field(default=100, strict=True, ge=0)
    soft_demote_threshold: float = Field(default=0.80, ge=0, le=1, allow_inf_nan=False)
    soft_demote_priority_jump: int = Field(default=-2, strict=True, ge=-(2**31), le=2**31 - 1)
    resume_priority_boost: int = Field(default=1, strict=True, ge=-(2**31), le=2**31 - 1)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> ProgramAdmissionConfig:
        if self.pause_target > self.pause_threshold:
            raise ValueError("pause_target must not exceed pause_threshold")
        if self.resume_hysteresis >= self.pause_threshold:
            raise ValueError("resume_hysteresis must be below pause_threshold")
        if self.soft_demote_threshold > self.pause_threshold:
            raise ValueError("soft_demote_threshold must not exceed pause_threshold")
        return self
