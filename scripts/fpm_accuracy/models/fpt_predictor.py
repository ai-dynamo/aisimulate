# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Predictor-neutral forward-pass-time interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fpm_accuracy.types.forward_pass import ForwardPassInput, ForwardPassIteration
from fpm_accuracy.types.worker_config import WorkerConfigRecord


@dataclass(frozen=True, slots=True)
class Prediction:
    value_ms: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PredictorContext:
    """Inputs needed to construct one predictor for one HF measurement case."""

    worker: WorkerConfigRecord
    worker_role: str
    options: Mapping[str, Any] = field(default_factory=dict)
    engine_config_overrides: Mapping[str, Any] = field(default_factory=dict)
    fpm_artifact: Any | None = None

    def __post_init__(self) -> None:
        if self.worker_role not in {"prefill", "decode", "aggregated"}:
            raise ValueError("worker_role must be prefill, decode, or aggregated")


class ForwardPassTimePredictor(ABC):
    """Interface every predictor implements. An ABC (not a Protocol) so a missing
    or misnamed method fails at instantiation rather than silently duck-typing."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def predict(self, features: ForwardPassInput) -> Prediction: ...

    @abstractmethod
    def tune(self, observations: Sequence[ForwardPassIteration]) -> None: ...

    @abstractmethod
    def diagnostics(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...


@runtime_checkable
class HasAicEngineConfig(Protocol):
    """Optional capability exposing an AISim adapter's effective engine config."""

    @property
    def engine_config(self) -> Mapping[str, Any] | None: ...


__all__ = [
    "ForwardPassTimePredictor",
    "HasAicEngineConfig",
    "Prediction",
    "PredictorContext",
]
