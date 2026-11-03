# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal scenario transport for the production learned-model APIs.

Python prepares initial forests with the SDK trainer and buffers the first
admitted observations of a missing store. Native models own prediction, fitting
updates, calibration, mixed fallback and expert selection. Models are isolated
by case, worker and native workload store. Original-A composition keeps all of a
worker's stores in one native model so mixed append preserves its entire state.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scripts.fpm_accuracy.exceptions import DependencyError
from scripts.fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, Prediction
from scripts.fpm_accuracy.scenario_contract import (
    GBDT_FROZEN_RECIPE,
    GBDT_ONLINE_RECIPE,
    HYBRID_ONLINE_RECIPE,
    SELECTOR_ONLINE_RECIPE,
)
from scripts.fpm_accuracy.types.forward_pass import ForwardPassInput, ForwardPassIteration

_TRAINING_FIELDS = (
    "target",
    "max_iter",
    "learning_rate",
    "max_leaf_nodes",
    "min_samples_leaf",
    "min_store_rows",
    "random_state",
    "early_stopping",
)
_NO_TRAINING = {
    "no trainable iterations (all idle, zero wall_time, or role-incompatible)",
    "every store fell below min_store_rows; nothing to export",
}


def _canonical_normalize(perf_model, authored):
    try:
        return perf_model.normalize_config(authored)
    except ValueError as exc:
        message = str(exc).lower()
        old_schema = any(word in message for word in ("unknown field", "unknown variant", "unrecognized"))
        requested_api = any(
            word in message
            for word in (
                "fpm_learned",
                "adaptation",
                "causal_selector",
                "temporary_mixed_composition",
                "quality_gate",
                "hybrid",
            )
        )
        if old_schema and requested_api:
            raise DependencyError("The AISim runtime lacks the scenario's canonical learned-model controls.") from exc
        raise


def _contains_controls(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_controls(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _public_diagnostics(value: Any) -> Any:
    """Keep native evidence while omitting workstation and temporary paths."""
    if isinstance(value, dict):
        return {key: _public_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_diagnostics(item) for item in value]
    if isinstance(value, str) and (Path(value).is_absolute() or value.startswith("file://")):
        return "[local path omitted]"
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _bounded_native_diagnostics(value: Any) -> Any:
    """Retain native outcomes and identity without repeating full configurations.

    Configuration fingerprints cover every canonical field before path
    redaction; readable identity retains deployment dimensions. Full configs
    remain available through the adapter's private seed_configurations API.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "config" and isinstance(item, dict) and {"model", "system", "estimator_config"} <= item.keys():
                identity_fields = (
                    "model",
                    "system",
                    "backend",
                    "backend_version",
                    "worker_type",
                    "tp",
                    "pp",
                    "attention_dp",
                    "dcp",
                    "moe_tp_size",
                    "moe_ep_size",
                    "cp_size",
                    "gemm_quant_mode",
                    "moe_quant_mode",
                    "fmha_quant_mode",
                    "fpm_fmha_quant_mode",
                    "kvcache_quant_mode",
                    "comm_quant_mode",
                    "attention_backend",
                    "moe_backend",
                    "nextn",
                    "speculation",
                    "kv_block_size",
                    "decoder_replay",
                )
                result["config_sha256"] = _digest(item)
                result["deployment_identity"] = _public_diagnostics(
                    {name: item[name] for name in identity_fields if name in item}
                )
                result["estimator_config_sha256"] = _digest(item["estimator_config"])
                if item.get("fpm_profile") is not None:
                    result["fpm_profile_sha256"] = _digest(item["fpm_profile"])
            elif key == "retained_row_ids" and isinstance(item, list):
                result["retained_row_count"] = len(item)
                # Preserve the native membership hash, or bind the list if an
                # older diagnostic provider does not yet supply that field.
                if "retained_row_ids_sha256" not in value:
                    result["retained_row_ids_sha256"] = _digest(item)
            else:
                result[key] = _bounded_native_diagnostics(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_bounded_native_diagnostics(item) for item in value]
    return _public_diagnostics(value)


@dataclass
class _Scope:
    worker_id: str
    workload_kind: str
    model: Any = None
    config: dict | None = None
    stores: set[str] = field(default_factory=set)
    eligible_mixed: set[str] = field(default_factory=set)
    pending: dict[str, list] = field(default_factory=dict)
    initial_stores: dict[str, str] = field(default_factory=dict)
    offline_observations: int = 0
    submitted_observations: int = 0
    bootstrap_observations: int = 0
    feedback_observations: int = 0
    ignored_observations: int = 0
    store_additions: int = 0


class LearnedScenarioPredictor(ForwardPassTimePredictor):
    def __init__(
        self,
        method: str,
        perf_model: Any,
        request: Any,
        trainer: Any,
        observations: Sequence[ForwardPassIteration] = (),
        *,
        online=False,
        cold_start=False,
        mixed=None,
    ):
        if mixed is not None:
            request = mixed.configure_request(request)
        self._method, self._perf_model, self._request, self._trainer = method, perf_model, request, trainer
        self._online, self._cold_start, self._mixed = online, cold_start, mixed
        self._recipe = {
            "gbdt": GBDT_ONLINE_RECIPE if online else GBDT_FROZEN_RECIPE,
            "hybrid": HYBRID_ONLINE_RECIPE,
            "selector": SELECTOR_ONLINE_RECIPE,
        }[method]
        self._adaptation = (
            "causal_selector" if method == "selector" else ("online" if method == "gbdt" and online else "frozen")
        )
        self._scopes: dict[tuple[str, str], _Scope] = {}
        self._artifacts: dict[str, Path] = {}
        self._checkpoints: dict[str, Path] = {}
        self._temporary = TemporaryDirectory(prefix="aisim-scenario-learned-")
        self._seed_index = 0
        try:
            self._preflight()
            grouped: dict[tuple[str, str], list] = {}
            for observation in observations:
                rows = observation.tuning_payload()
                key = self._key(rows)
                if key is not None:
                    grouped.setdefault(key, []).append(rows)
            for key, rows in grouped.items():
                scope = self._scope(key)
                scope.offline_observations = len(rows)
                fitted = self._fit(scope, rows)
                if not fitted and self._method == "gbdt" and online and mixed is None:
                    # B+C may finish a below-threshold B store from admitted C.
                    scope.pending[key[1]] = rows
        except Exception:
            self.close()
            raise

    @property
    def id(self):
        return self._method

    @property
    def _minimum(self):
        return self._recipe["training"]["min_store_rows"]

    def _preflight(self):
        required = ["best_available", "normalize_config", "diagnostics", "tune_with_fpms"]
        required += (
            ["learned_expert_artifacts", "learned_selector_diagnostics"]
            if self._method == "selector"
            else ["export_learned_artifact"]
        )
        if self._adaptation != "frozen":
            required.append("learned_online_diagnostics")
        for name in required:
            if not callable(getattr(self._perf_model, name, None)):
                raise DependencyError(f"Learned scenarios require the native {name} API.")
        if list(self._trainer.FEATURE_PRESETS["fpm5"]) != self._recipe["features"]:
            raise DependencyError("The runtime feature ABI differs from the recorded Gym fpm5 recipe.")
        if "identity" not in inspect.signature(self._trainer.train).parameters:
            raise DependencyError("The SDK trainer must support complete canonical deployment identity.")
        if self._adaptation != "frozen":
            if "online_checkpoint_dir" not in inspect.signature(self._trainer.train).parameters:
                raise DependencyError("Online scenarios require the SDK training-checkpoint exporter.")
            try:
                import sklearn
            except ImportError as exc:
                raise DependencyError("scikit-learn is required to prepare learned-model seeds.") from exc
            if sklearn.__version__ not in self._trainer.ONLINE_SUPPORTED_SKLEARN_VERSIONS:
                raise DependencyError(f"Online checkpoint export does not support scikit-learn {sklearn.__version__}.")
        if (
            self._mixed is not None
            and self._adaptation != "frozen"
            and not callable(getattr(self._perf_model, "append_mixed_learned_checkpoint", None))
        ):
            raise DependencyError("A+C requires native append_mixed_learned_checkpoint.")
        self._identity = self._configuration(None)

    def _configuration(self, source: Path | None):
        estimator = deepcopy(self._request.estimator_config)
        learned = {"adaptation": self._adaptation}
        if self._adaptation != "frozen":
            learned.update(deepcopy(self._recipe["online"]))
            if learned.pop("recent_sampling", "fifo") != "fifo":
                raise ValueError("Scenario online history must use the native FIFO policy.")
            learned["adaptation"] = self._adaptation
        if source is not None:
            learned["artifact_path" if self._adaptation == "frozen" else "checkpoint_path"] = str(source)
        estimator["fpm_learned"] = learned
        if self._method in {"hybrid", "selector"}:
            estimator["correction"] = deepcopy(self._recipe["correction"])
        else:
            estimator.setdefault("correction", {})["enabled"] = False
        if self._method == "selector":
            estimator["selector"] = deepcopy(self._recipe["selector"])
        authored = replace(
            self._request,
            estimation_mode="auto" if source is None else "fpm_learned",
            fallback_policy="deny",
            estimator_config=estimator,
        )
        config = _canonical_normalize(self._perf_model, authored)
        actual = config.get("estimator_config", {})
        # Rust omits default-valued learned and selector sections. With a
        # supplied source learned controls are always serialized and checked.
        if (
            config.get("estimation_mode") != authored.estimation_mode
            or config.get("fallback_policy") != "deny"
            or not _contains_controls(actual.get("correction"), estimator["correction"])
            or ("fpm_learned" in actual and not _contains_controls(actual["fpm_learned"], learned))
            or (source is not None and "fpm_learned" not in actual)
            or (
                self._method == "selector"
                and "selector" in actual
                and not _contains_controls(actual["selector"], estimator["selector"])
            )
        ):
            raise DependencyError("The AISim runtime did not preserve the requested learned-model controls.")
        return config

    def _key(self, rows):
        kind = self._trainer.classify_workload(rows, self._request.worker_type)
        if kind is None:
            return None
        if self._mixed is not None:
            kind = "original_a"
        elif self._method in {"hybrid", "selector"} and not self._cold_start:
            # Offline hybrid/selector seeds retain their native per-worker
            # correction and selection pools across workload stores.
            kind = "offline"
        return str(rows[0]["worker_id"]), kind

    def _scope(self, key):
        if key not in self._scopes:
            self._scopes[key] = _Scope(*key)
        return self._scopes[key]

    def _fit(self, scope: _Scope, rows: Sequence[list], *, append_kind: str | None = None) -> bool:
        directory = Path(self._temporary.name)
        seed = directory / f"seed-{self._seed_index}"
        self._seed_index += 1
        parameters = {key: self._recipe["training"][key] for key in _TRAINING_FIELDS}
        try:
            artifact = self._trainer.train(
                rows,
                self._request.worker_type,
                features=self._recipe["features"],
                identity=self._identity,
                **({"online_checkpoint_dir": seed} if self._adaptation != "frozen" else {}),
                **parameters,
            )
        except ValueError as exc:
            if append_kind is None and str(exc) in _NO_TRAINING:
                return False
            raise
        if append_kind is not None and (
            set(artifact["stores"]) != {append_kind} or artifact["metadata"]["train_rows"][append_kind] != self._minimum
        ):
            raise ValueError("A missing mixed class requires exactly its first admitted single-store seed.")
        if self._adaptation == "frozen":
            seed = seed.with_suffix(".json")
            seed.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False))
            artifact_path = seed
        else:
            artifact_path = seed / "model.json"
        config = self._configuration(seed)
        if append_kind is not None:
            if scope.model.append_mixed_learned_checkpoint(str(seed)) != 1:
                raise ValueError("Native mixed append must add exactly one absent store.")
            scope.store_additions += 1
        else:
            if self._mixed is not None:
                config = self._mixed.bind(
                    self._perf_model,
                    self._trainer,
                    config,
                    rows,
                    artifact_path,
                    self._minimum,
                    directory,
                    quality_gate=self._method == "gbdt" and self._online,
                )
            model = self._perf_model.best_available(config)
            scope.model, scope.config = model, config
            if self._mixed is not None:
                scope.eligible_mixed = {
                    route["workload_kind"]
                    for route in model.mixed_composition_diagnostics()["routes"]
                    if route["fallback_eligible"]
                }
        # Publish state immediately after native mutation: evidence failure must
        # never cause a label or an appended checkpoint to be submitted twice.
        scope.stores.update(artifact["stores"])
        initial = self._forest(scope.model)
        for kind in artifact["stores"]:
            if kind in initial["stores"]:
                scope.initial_stores[kind] = _digest(initial["stores"][kind])
        self._artifacts[hashlib.sha256(artifact_path.read_bytes()).hexdigest()] = artifact_path
        if self._adaptation != "frozen":
            manifest = seed / "manifest.json"
            self._checkpoints[hashlib.sha256(manifest.read_bytes()).hexdigest()] = seed
        return True

    def _forest(self, model):
        if self._method == "selector":
            return model.learned_expert_artifacts()["stable"]
        return model.export_learned_artifact()

    def predict(self, features: ForwardPassInput) -> Prediction:
        rows = features.aic_payload()
        if any("wall_time" in row for row in rows):
            raise ValueError("Scenario prediction input cannot contain its measured target.")
        scope = self._scopes.get(self._key(rows))
        if scope is None or scope.model is None:
            return Prediction(None)
        metadata = {}
        try:
            if self._mixed is not None:
                result = scope.model.estimate_forward_pass_with_route(rows)
                value = result["latency_ms"]
                metadata["temporary_mixed_composition"] = result["temporary_mixed_composition"]
            else:
                value = scope.model.estimate_forward_pass_time_ms(rows)
        except Exception as exc:
            from scripts.fpm_accuracy.models.scenario_interpolation import is_interpolation_support_miss

            if not is_interpolation_support_miss(exc):
                raise
            return Prediction(None, {"reason": "unsupported_interpolation_query"})
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Native scenario model returned an invalid latency.")
        return Prediction(value, metadata)

    def tune(self, observations: Sequence[ForwardPassIteration]) -> None:
        if not self._online:
            raise ValueError("A frozen A/B evaluation cannot admit test labels.")
        for observation in observations:
            rows = observation.tuning_payload()
            key = self._key(rows)
            if key is None:
                continue
            scope = self._scope(key)
            scope.submitted_observations += 1
            kind = self._trainer.classify_workload(rows, self._request.worker_type)
            if self._mixed is not None and scope.model is not None:
                # A missing mixed store still calibrates its prior. Once ready,
                # native feedback updates both experts using pre-update losses.
                error = None
                try:
                    scope.model.tune_with_fpms([rows])
                    scope.feedback_observations += 1
                except Exception as exc:
                    error = exc
                if (
                    error is None
                    and self._adaptation != "frozen"
                    and kind in scope.eligible_mixed
                    and kind not in scope.stores
                ):
                    self._bootstrap(scope, rows, kind, append=True)
                if error is not None:
                    raise error
            elif scope.model is not None:
                scope.model.tune_with_fpms([rows])
                scope.feedback_observations += 1
            elif self._mixed is None and (self._method == "gbdt" or self._cold_start):
                self._bootstrap(scope, rows, kind, append=False)
            else:
                scope.ignored_observations += 1

    def _bootstrap(self, scope, rows, kind, *, append):
        pending = scope.pending.setdefault(kind, [])
        pending.append(rows)
        scope.bootstrap_observations += 1
        if len(pending) >= self._minimum:
            try:
                ready = self._fit(scope, pending, append_kind=kind if append else None)
                if not ready:
                    raise ValueError("Admitted bootstrap rows did not produce a trainable native store.")
                del scope.pending[kind]
            except Exception:
                if kind not in scope.stores:
                    pending.pop()
                else:
                    scope.pending.pop(kind, None)
                raise

    def diagnostics(self) -> Mapping[str, Any]:
        scopes = []
        for scope in self._scopes.values():
            model = scope.model
            final_stores = self._forest(model)["stores"] if model is not None else {}
            final_hashes = {kind: _digest(store) for kind, store in final_stores.items()}
            if self._adaptation in {"frozen", "causal_selector"} and final_hashes != scope.initial_stores:
                raise ValueError("Native calibration or selection changed a frozen forest.")
            native = dict(model.diagnostics()) if model is not None else None
            if native is not None and self._mixed is not None:
                # The complete mixed route/counter evidence is exported once
                # below, rather than again inside the general diagnostics.
                native.pop("temporary_mixed_composition", None)
            scopes.append(
                {
                    "worker_id": scope.worker_id,
                    "workload_kind": scope.workload_kind,
                    "ready": model is not None,
                    "stores": sorted(scope.stores),
                    "offline_observations": scope.offline_observations,
                    "submitted_observations": scope.submitted_observations,
                    "bootstrap_observations": scope.bootstrap_observations,
                    "feedback_observations": scope.feedback_observations,
                    "ignored_observations": scope.ignored_observations,
                    "store_additions": scope.store_additions,
                    "pending_bootstrap": {kind: len(rows) for kind, rows in scope.pending.items()},
                    "initial_store_sha256": scope.initial_stores,
                    "final_store_sha256": final_hashes,
                    "native": _bounded_native_diagnostics(native),
                    "online": _bounded_native_diagnostics(model.learned_online_diagnostics())
                    if model is not None and self._adaptation != "frozen"
                    else None,
                    "selector": _bounded_native_diagnostics(model.learned_selector_diagnostics())
                    if model is not None and self._method == "selector"
                    else None,
                    "mixed_composition": _bounded_native_diagnostics(model.mixed_composition_diagnostics())
                    if model is not None and self._mixed
                    else None,
                }
            )
        return {
            "method": self.id,
            "model_scope": "case_worker_and_native_workload_store",
            "adaptation": self._adaptation,
            "recipe": self._recipe,
            "scopes": scopes,
        }

    def seed_artifact_paths(self):
        return dict(self._artifacts)

    def seed_checkpoint_paths(self):
        return dict(self._checkpoints)

    def original_a_manifest_paths(self):
        return dict(self._mixed.manifests) if self._mixed is not None else {}

    def seed_configurations(self):
        return {
            f"{worker}/{kind}": deepcopy(scope.config)
            for (worker, kind), scope in self._scopes.items()
            if scope.config is not None
        }

    def close(self):
        scopes, self._scopes = self._scopes, {}
        errors = []
        try:
            for scope in scopes.values():
                if scope.model is not None:
                    try:
                        scope.model.close()
                    except Exception as exc:
                        errors.append(exc)
        finally:
            self._temporary.cleanup()
            if self._mixed is not None:
                self._mixed.close()
        if errors:
            raise ExceptionGroup("Failed to close native scenario models", errors)
