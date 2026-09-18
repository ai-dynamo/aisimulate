# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Strict, small overrides for ambiguous Hugging Face evidence bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fpm_accuracy.exceptions import ConfigurationError
from fpm_accuracy.hf.models import OrderingKind


class HfCaseOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_path: str = Field(min_length=1)
    snapshot_id: str | None = None
    truth_file_ids: tuple[str, ...] | None = None
    helper_file_ids: tuple[str, ...] | None = None
    fpm_artifact_ids: tuple[str, ...] | None = None
    worker_role: Literal["prefill", "decode", "aggregated"] | None = None
    ordering: OrderingKind | None = None

    @model_validator(mode="after")
    def _ids_are_unique(self) -> HfCaseOverride:
        for name in ("truth_file_ids", "helper_file_ids", "fpm_artifact_ids"):
            values = getattr(self, name)
            if values is not None and len(values) != len(set(values)):
                raise ValueError(f"{name} may not contain duplicate IDs")
        if all(
            getattr(self, name) is None
            for name in ("truth_file_ids", "helper_file_ids", "fpm_artifact_ids", "worker_role", "ordering")
        ):
            raise ValueError("an HF override must declare at least one binding or correction")
        return self


class HfOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    overrides: tuple[HfCaseOverride, ...] = ()

    @model_validator(mode="after")
    def _selectors_are_unique(self) -> HfOverrides:
        selectors = [(item.configuration_path, item.snapshot_id) for item in self.overrides]
        if len(selectors) != len(set(selectors)):
            raise ValueError("HF overrides may not repeat a configuration_path/snapshot_id selector")
        return self

    def find(self, configuration_path: str, snapshot_id: str) -> HfCaseOverride | None:
        exact = [
            item
            for item in self.overrides
            if item.configuration_path == configuration_path and item.snapshot_id == snapshot_id
        ]
        defaults = [
            item
            for item in self.overrides
            if item.configuration_path == configuration_path and item.snapshot_id is None
        ]
        return exact[0] if exact else (defaults[0] if defaults else None)


def load_overrides(path: Path | None) -> HfOverrides:
    if path is None:
        return HfOverrides(version=1)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read HF overrides {path}: {exc}") from exc
    try:
        return HfOverrides.model_validate(payload)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid HF overrides {path}: {exc}") from exc
