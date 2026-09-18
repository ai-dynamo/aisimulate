# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""OOP adapters for AISim FPM and regression predictors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar, Literal

from fpm_accuracy.exceptions import ConfigurationError, DependencyError
from fpm_accuracy.models.aic_config import map_worker_config_to_aic
from fpm_accuracy.models.aic_fpm_database import PreparedAicFpmDatabase, prepare_aic_fpm_database
from fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, Prediction, PredictorContext
from fpm_accuracy.types.forward_pass import ForwardPassInput, ForwardPassIteration

AicMode = Literal["fpm", "regression"]


class _AicPredictor(ForwardPassTimePredictor):
    """Shared AISim payload, diagnostics, validation, and lifecycle behavior."""

    mode: ClassVar[AicMode]
    predictor_id: ClassVar[str]

    def __init__(
        self,
        model: Any,
        engine_config: Mapping[str, Any] | None,
        prepared_fpm: PreparedAicFpmDatabase | None = None,
    ) -> None:
        self._model = model
        self.engine_config = dict(engine_config) if engine_config is not None else None
        self._prepared_fpm = prepared_fpm

    @property
    def id(self) -> str:
        return self.predictor_id

    @classmethod
    def create(cls, context: PredictorContext) -> _AicPredictor:
        perf_model = _import_aisim_forward_pass_perf_model()
        options = dict(context.options)
        if cls.mode == "regression":
            if not callable(getattr(perf_model, "regression_store_diagnostics", None)):
                raise DependencyError("This AISim revision does not support worker-scoped regression.")
            config_type = _canonical_config_type()
            if config_type is not None:
                request = config_type.from_legacy_engine_config(
                    cls._native_engine_config(context), context.worker_role, options
                )
                model = perf_model.best_available(replace(request, estimation_mode="fpm_regression"))
            else:
                # The evaluator also runs against older released branch wheels.
                model = perf_model.from_regression(context.worker_role, options)
            try:
                store_diagnostics = getattr(model, "regression_store_diagnostics", None)
                if not callable(store_diagnostics):
                    raise AttributeError("regression_store_diagnostics is missing")
                # Also detect an updated Python wrapper with an older native binary.
                store_diagnostics()
            except AttributeError as exc:
                close = getattr(model, "close", None)
                if close is not None:
                    close()
                raise DependencyError(
                    "Regression requires AISim with regression_store_diagnostics(); rebuild or update AISim."
                ) from exc
            return cls(model, None)

        engine_config = cls._native_engine_config(context)
        prepared_fpm = None
        if cls.mode == "fpm":
            if context.fpm_artifact is None:
                raise ConfigurationError("AISim FPM prediction requires an explicit HF FPM artifact")
            prepared_fpm = prepare_aic_fpm_database(engine_config, context.fpm_artifact)
            engine_config["systems_path"] = str(prepared_fpm.systems_root)
            engine_config["forward_model"] = "fpm"
        try:
            config_type = _canonical_config_type()
            if config_type is not None:
                request = config_type.from_legacy_engine_config(engine_config, context.worker_role, options)
                model = perf_model.best_available(request)
            else:
                model = perf_model.from_native(engine_config, options)
        except Exception:
            if prepared_fpm is not None:
                prepared_fpm.close()
            raise
        return cls(model, engine_config, prepared_fpm)

    @classmethod
    def _native_engine_config(cls, context: PredictorContext) -> dict[str, Any]:
        return map_worker_config_to_aic(context.worker, context.engine_config_overrides)

    def predict(self, features: ForwardPassInput) -> Prediction:
        value = self._model.estimate_forward_pass_time_ms(features.aic_payload())
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"AISim returned invalid forward-pass prediction {value!r}")
        diagnostics = self.diagnostics()
        return Prediction(
            value_ms=value,
            metadata={"source": diagnostics.get("source"), "readiness": diagnostics.get("readiness")},
        )

    def tune(self, observations: Sequence[ForwardPassIteration]) -> None:
        if observations:
            # Each outer list is a sequential iteration; each inner list is its
            # attention-DP ranks. Flattening would reinterpret time as ranks.
            self._model.tune_with_fpms([observation.tuning_payload() for observation in observations])

    def diagnostics(self) -> Mapping[str, Any]:
        result = dict(self._model.diagnostics())
        result["mode"] = self.mode
        if self._prepared_fpm is not None:
            result.update(self._prepared_fpm.diagnostics)
        for key, method_name in (
            ("min_correction_factor", "get_min_correction_factor"),
            ("max_correction_factor", "get_max_correction_factor"),
            ("avg_correction_factor", "get_avg_correction_factor"),
        ):
            method = getattr(self._model, method_name, None)
            result[key] = method() if method is not None else None
        return result

    def close(self) -> None:
        try:
            close = getattr(self._model, "close", None)
            if close is not None:
                close()
        finally:
            if self._prepared_fpm is not None:
                self._prepared_fpm.close()
                self._prepared_fpm = None


class AicFpmPredictor(_AicPredictor):
    """AISim's whole-model predictor over one exact HF FPM parquet pair."""

    mode = "fpm"
    predictor_id = "aic-fpm"


class AicRegressionPredictor(_AicPredictor):
    """AISim's workload-inferred, online regression predictor."""

    mode = "regression"
    predictor_id = "regression"

    def diagnostics(self) -> Mapping[str, Any]:
        result = dict(super().diagnostics())
        result["regression_stores"] = self._model.regression_store_diagnostics()
        return result


def _import_aisim_forward_pass_perf_model() -> Any:
    """Import AISim only when a concrete predictor is constructed."""

    try:
        from aisimulate_core.sdk import RustForwardPassPerfModel
    except ImportError as exc:
        raise DependencyError(
            "AISim predictors require aisimulate. Install the sibling AISim checkout in this environment."
        ) from exc
    return RustForwardPassPerfModel


def _canonical_config_type() -> Any:
    from aisimulate_core import sdk

    return getattr(sdk, "ForwardPassPerfModelConfig", None)


__all__ = ["AicFpmPredictor", "AicRegressionPredictor"]
