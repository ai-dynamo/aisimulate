# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline learned forward-pass model (``fpm5`` preset) for the Gym evaluation.

Unlike the online predictors, the learned model is fitted once before scoring.
Each workload kind is split by alternating observations in stream order: every
other observation trains the model and only the others are scored, so a scored
observation is never in the training set. This measures interpolation within
the measured range, not extrapolation to later traffic. Training observations
are reported as unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scripts.fpm_accuracy.exceptions import DependencyError
from scripts.fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, Prediction, PredictorContext
from scripts.fpm_accuracy.types.forward_pass import ForwardPassInput, ForwardPassIteration

FEATURE_PRESET = "fpm5"


def training_observation_ids(observations: Sequence[Any]) -> frozenset[str]:
    """Every other observation (the 1st, 3rd, ...) of each workload kind, in stream order."""

    by_kind: dict[str, list[str]] = {}
    for observation in observations:
        by_kind.setdefault(str(observation.workload_kind), []).append(observation.observation_id)
    return frozenset(observation_id for ids in by_kind.values() for observation_id in ids[0::2])


class LearnedFpm5Predictor(ForwardPassTimePredictor):
    """Rust inference over an artifact trained by ``aisimulate_core.sdk.fpm_learned``."""

    def __init__(self, model: Any, artifact: Mapping[str, Any]) -> None:
        self._model = model
        self._artifact = artifact

    @property
    def id(self) -> str:
        return f"learned-{FEATURE_PRESET}"

    @classmethod
    def fit(cls, context: PredictorContext, iterations: Sequence[ForwardPassIteration]) -> LearnedFpm5Predictor:
        try:
            from aisimulate_core.sdk import fpm_learned
            from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel
        except ImportError as exc:  # pragma: no cover - depends on the evaluated wheel
            raise DependencyError("This AISim revision does not provide the learned forward-pass model.") from exc
        if FEATURE_PRESET not in getattr(fpm_learned, "FEATURE_PRESETS", {}):
            raise DependencyError(f"This AISim revision has no {FEATURE_PRESET!r} feature preset.")
        try:
            import sklearn  # noqa: F401
        except ImportError as exc:
            raise DependencyError("scikit-learn is required to train the learned forward-pass model.") from exc
        try:
            artifact = fpm_learned.train(
                [iteration.tuning_payload() for iteration in iterations],
                context.worker_role,
                features=fpm_learned.FEATURE_PRESETS[FEATURE_PRESET],
            )
        except ValueError:
            # Too few training rows for any store: every scored point is unavailable.
            return cls(None, {"stores": {}, "metadata": {}})
        return cls(RustForwardPassPerfModel.from_learned(artifact), artifact)

    def predict(self, features: ForwardPassInput) -> Prediction:
        if self._model is None:
            return Prediction(None)
        return Prediction(self._model.estimate_forward_pass_time_ms(features.aic_payload()))

    def tune(self, observations: Sequence[ForwardPassIteration]) -> None:
        del observations  # fitted once; no online update

    def diagnostics(self) -> Mapping[str, Any]:
        return {
            "feature_preset": FEATURE_PRESET,
            "stores": sorted(self._artifact["stores"]),
            "train_rows": dict(self._artifact["metadata"].get("train_rows", {})),
        }

    def close(self) -> None:
        self._model = None
