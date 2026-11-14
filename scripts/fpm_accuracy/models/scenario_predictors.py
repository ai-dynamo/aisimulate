# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native predictors and measured self-benchmark seeds for scenario evaluation.

All model construction goes through the canonical ``best_available`` API.
Fitting policy (offline, frozen, or predict-before-update) belongs to the
scenario evaluator; this module never reads a prediction's latency target.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, replace
from itertools import pairwise
from operator import attrgetter
from typing import Any

from scripts.fpm_accuracy.contract import strict_json
from scripts.fpm_accuracy.exceptions import ConfigurationError, DataError, DependencyError
from scripts.fpm_accuracy.hf.models import FpmArtifact, MeasurementCase
from scripts.fpm_accuracy.models import aic_predictors
from scripts.fpm_accuracy.models.aic_config import map_worker_config_to_aic
from scripts.fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, Prediction, PredictorContext
from scripts.fpm_accuracy.types.forward_pass import (
    ForwardPassInput,
    ForwardPassIteration,
    ForwardPassMetric,
    RequestMetrics,
)

# Field names are identical for every request record. Comparing their values in
# this fixed name order equals comparing sorted (name, value) pairs, without
# allocating dictionaries and name/value tuples for every DP rank.
_REQUEST_METRIC_ORDER_KEY = attrgetter(*sorted(field.name for field in fields(RequestMetrics)))


def create_scenario_predictor(
    method: str,
    context: PredictorContext,
    *,
    training_iterations: Sequence[ForwardPassIteration] = (),
    online: bool = False,
    cold_start: bool = False,
    mixed_fallback: bool = False,
    scenario: str | None = None,
) -> ForwardPassTimePredictor:
    """Transport scenario policy to canonical native estimators.

    The caller alone selects A/B/C observations and five-second admission times.
    Original-A interpolation is enabled only for aggregated A and A+C cells.
    """
    if method not in {"warmup", "nowarmup", "linear", "spline", "gbdt", "hybrid", "selector"}:
        raise ValueError(f"unknown scenario predictor {method!r}")
    if method in {"hybrid", "selector"} and not online:
        raise ValueError("Hybrid and selector require an online scenario.")
    if cold_start and (method not in {"hybrid", "selector"} or not online or training_iterations):
        raise ValueError("Cold initialization requires an online hybrid/selector without offline training.")
    if mixed_fallback and (
        method not in {"gbdt", "hybrid", "selector"}
        or scenario not in {"a", "ac"}
        or cold_start
        or not training_iterations
        or online != (scenario == "ac")
        or (method != "gbdt" and scenario == "a")
    ):
        raise ValueError("Original-A fallback requires GBDT A or GBDT/hybrid/selector A+C.")
    perf_model = aic_predictors._import_aisim_forward_pass_perf_model()
    config_type = aic_predictors._canonical_config_type()
    if config_type is None or not callable(getattr(perf_model, "normalize_config", None)):
        raise DependencyError("Scenario comparison requires the canonical native best_available API.")
    if method in {"warmup", "nowarmup"}:
        if training_iterations:
            raise ValueError("Interpolation consumes its FPM artifact, not telemetry training iterations.")
        return _create_interpolation(context, perf_model, config_type)
    engine_config = map_worker_config_to_aic(context.worker, context.engine_config_overrides)
    mixed = None
    try:
        if mixed_fallback and context.worker_role == "aggregated":
            from scripts.fpm_accuracy.models.mixed_scenario import MixedOriginalA

            mixed = MixedOriginalA(engine_config, context.fpm_artifact, scenario)
            normalization = seed_target_normalization_evidence(training_iterations)
            if normalization is not None:
                mixed.provenance["seed_target_normalization"] = normalization
                if normalization["replaced_source_rows"]:
                    mixed.provenance["grouping"]["target_scope"] = "native_normalized_aggregate_latency"
        request = config_type.from_legacy_engine_config(engine_config, context.worker_role, dict(context.options))
        if method in {"gbdt", "hybrid", "selector"}:
            from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor

            return LearnedScenarioPredictor(
                method,
                perf_model,
                request,
                _import_gbdt_trainer(),
                training_iterations,
                online=online,
                cold_start=cold_start,
                mixed=mixed,
            )
    except Exception:
        if mixed is not None:
            mixed.close()
        raise
    if method == "linear":
        config = aic_predictors._gym_regression_config(perf_model, request, context.options, engine_config)
    else:
        estimator = deepcopy(request.estimator_config)
        fit = estimator["fpm_regression"]["fit"]
        fit.pop("linear", None)
        fit["kind"] = "spline"
        config = perf_model.normalize_config(
            replace(request, estimation_mode="fpm_regression", fallback_policy="deny", estimator_config=estimator)
        )
    predictor = _WorkerScenarioRegression(method, perf_model, config)
    try:
        predictor.tune(training_iterations)
    except Exception:
        predictor.close()
        raise
    return predictor


def _create_interpolation(context: PredictorContext, perf_model: Any, config_type: Any) -> ForwardPassTimePredictor:
    """Use the canonical measured-text route for the selected warmup library."""
    from scripts.fpm_accuracy.models.aic_fpm_database import prepare_aic_fpm_database
    from scripts.fpm_accuracy.models.scenario_interpolation import configure_scenario_interpolation

    if context.worker.config.parallelism.decode_context_parallel_size != 1:
        raise DependencyError("Native FPM does not support decode context parallelism (dcp > 1).")
    if context.fpm_artifact is None:
        raise ConfigurationError("Scenario interpolation requires an explicit HF FPM artifact.")
    engine_config = map_worker_config_to_aic(context.worker, context.engine_config_overrides)
    prepared = prepare_aic_fpm_database(engine_config, context.fpm_artifact)
    engine_config["systems_path"] = str(prepared.systems_root)
    engine_config["forward_model"] = "fpm"
    try:
        request = config_type.from_legacy_engine_config(engine_config, context.worker_role, dict(context.options))
        request = configure_scenario_interpolation(request, context.fpm_artifact, prepared)
        model = perf_model.best_available(request)
    except Exception:
        prepared.close()
        raise
    return _ScenarioInterpolation(model, engine_config, prepared)


class _ScenarioInterpolation(aic_predictors.AicFpmPredictor):
    def fpm_seed_target_normalization(self) -> Mapping[str, Any]:
        operation = getattr(self._model, "fpm_seed_target_normalization", None)
        if not callable(operation):
            raise DependencyError("Self-benchmark fake-KV targets require native seed target normalization.")
        return operation()

    def predict(self, features: ForwardPassInput) -> Prediction:
        from aisim_core.sdk.errors import PerfDataNotAvailableError
        from scripts.fpm_accuracy.models.scenario_interpolation import is_interpolation_support_miss

        try:
            return super().predict(features)
        except PerfDataNotAvailableError as exc:
            if not is_interpolation_support_miss(exc):
                raise
            return Prediction(None, metadata={"reason": "unsupported_interpolation_query"})

    def diagnostics(self) -> Mapping[str, Any]:
        from scripts.fpm_accuracy.models.learned_scenario import _public_diagnostics

        result = dict(super().diagnostics())
        controls = (
            result.get("provenance", {}).get("config", {}).get("estimator_config", {}).get("fpm_interpolation", {})
        )
        if controls.get("method") in {"sol", "direct"}:
            result["interpolation"] = {
                "method": controls["method"],
                "timing_only": controls["method"] == "direct",
                "text_only": controls.get("text_only", False),
                "metadata_recovered": "fpm_metadata_recovery" in result,
            }
        return _public_diagnostics(result)


class _WorkerScenarioRegression(ForwardPassTimePredictor):
    """Separate mutable native state for every observed worker in one case.

    Workers are discovered as they arrive. No future targets or workload
    inventory are inspected to choose a worker's estimator or seed data.
    """

    def __init__(self, method: str, perf_model: Any, config: dict[str, Any]) -> None:
        self._method, self._perf_model, self._config = method, perf_model, config
        self._workers: dict[str, Any] = {}
        # Fail missing-native/configuration checks during construction, before
        # the evaluator would classify an error as a failed prediction.
        self._unassigned = self._perf_model.best_available(self._config)

    @property
    def id(self) -> str:
        return self._method

    def _worker(self, worker_id: str) -> Any:
        if worker_id not in self._workers:
            if self._unassigned is not None:
                self._workers[worker_id], self._unassigned = self._unassigned, None
            else:
                self._workers[worker_id] = self._perf_model.best_available(self._config)
        return self._workers[worker_id]

    def predict(self, features: ForwardPassInput) -> Prediction:
        worker_id = str(features.rank_payloads[0]["worker_id"])
        value = self._worker(worker_id).estimate_forward_pass_time_ms(features.aic_payload())
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Native scenario predictor returned an invalid latency.")
        return Prediction(value)

    def tune(self, observations: Sequence[ForwardPassIteration]) -> None:
        for observation in observations:
            self._worker(observation.ranks[0].worker_id).tune_with_fpms([observation.tuning_payload()])

    def diagnostics(self) -> Mapping[str, Any]:
        from scripts.fpm_accuracy.models.learned_scenario import _public_diagnostics

        return _public_diagnostics(
            {
                "method": self.id,
                "model_scope": "worker_id_within_case",
                "config": self._config,
                "workers": {worker: model.diagnostics() for worker, model in self._workers.items()},
            }
        )

    def close(self) -> None:
        models = list(self._workers.values())
        if self._unassigned is not None:
            models.append(self._unassigned)
        self._workers, self._unassigned = {}, None
        errors = []
        for model in models:
            close = getattr(model, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("Failed to close scenario models", errors)


def _import_gbdt_trainer() -> Any:
    try:
        from aisim_core.sdk import fpm_learned
    except ImportError as exc:
        raise DependencyError("The evaluated AISim version does not provide the PR #280 GBDT trainer.") from exc
    if "fpm5" not in fpm_learned.FEATURE_PRESETS:
        raise DependencyError("The evaluated GBDT trainer does not provide the Gym fpm5 feature preset.")
    return fpm_learned


def order_offline_regression_seeds(
    iterations: Sequence[ForwardPassIteration],
) -> tuple[ForwardPassIteration, ...]:
    """Remove profile serialization order from A/A+C spline initialization.

    Only the evaluator's offline artifact branch calls this transport helper.
    Neither targets nor source row/counter IDs enter the ordering. Each worker
    receives the same fixed shuffle of its sorted workload coordinates. Duplicate
    feature keys are ambiguous profile inputs, not an invitation to break ties
    with labels or the source row order.
    """
    from scripts.fpm_accuracy.scenario_contract import OFFLINE_REGRESSION_SEED_ORDER

    groups = defaultdict(list)
    for iteration in iterations:
        key = (
            iteration.configuration_id,
            tuple(
                (
                    rank.version,
                    rank.dp_rank,
                    str(rank.workload_kind),
                    _REQUEST_METRIC_ORDER_KEY(rank.scheduled),
                    _REQUEST_METRIC_ORDER_KEY(rank.queued),
                )
                for rank in sorted(iteration.ranks, key=lambda rank: rank.dp_rank)
            ),
        )
        groups[iteration.ranks[0].worker_id].append((key, iteration))
    result = []
    for worker in sorted(groups):
        decorated = groups[worker]
        decorated.sort(key=lambda item: item[0])
        if any(left[0] == right[0] for left, right in pairwise(decorated)):
            raise DataError("Offline spline seeds contain duplicate workload feature keys for one worker.")
        random.Random(OFFLINE_REGRESSION_SEED_ORDER["seed"]).shuffle(decorated)
        result.extend(iteration for _, iteration in decorated)
    return tuple(result)


def self_benchmark_iterations(case: MeasurementCase, artifact: FpmArtifact) -> tuple[ForwardPassIteration, ...]:
    """Transport qualified self-benchmark targets into the existing tuning API.

    Schema-6 balanced_v1 collection repeats the table's counts and token sums
    on every attention-DP rank, despite the iteration_totals coordinate name.
    Its collector validates each rank against those unchanged coordinates.
    Individual rank latencies are not recorded in the table: rank zero carries
    the qualified aggregate target and other ranks use the unobserved zero
    sentinel, so the canonical iteration target remains that aggregate value.
    This does not assert that rank zero was the slowest measured rank. Unknown
    partition policies and schema-7 execution identities remain unsupported.
    Genuine labels stay unchanged. Marked fake-KV targets use the canonical
    native loader's effective labels, with both raw and effective provenance;
    these corrected labels are not presented as additional measurements.
    """
    import pyarrow.parquet as pq

    if artifact not in case.fpm_artifacts:
        raise DataError("Self-benchmark artifact is not a member of the selected measurement case.")
    metadata_bytes = artifact.local_metadata_path.read_bytes()
    if hashlib.sha256(metadata_bytes).hexdigest() != artifact.metadata_sha256:
        raise DataError("Self-benchmark metadata changed after dataset validation.")
    if hashlib.sha256(artifact.local_path.read_bytes()).hexdigest() != artifact.sha256:
        raise DataError("Self-benchmark parquet changed after dataset validation.")
    metadata = strict_json(metadata_bytes.decode("utf-8"))
    if metadata.get("schema_version") != 6:
        raise DependencyError("Self-benchmark regression seeding requires Gym-supported schema-6 execution identity.")
    if metadata.get("parquet_sha256") != artifact.sha256:
        raise DataError("Self-benchmark metadata does not bind the selected parquet hash.")
    from scripts.fpm_accuracy.models.aic_fpm_database import _recover_reconstructed_metadata

    # Complete the same validated schema-6 contract as native staging before
    # interpreting DP coordinates. Original metadata bytes/hashes stay pinned.
    metadata, _ = _recover_reconstructed_metadata(metadata, artifact)
    selector = metadata["configuration_selector"]
    dp = selector.get("dp", 1)
    if isinstance(dp, bool) or not isinstance(dp, int) or dp < 1:
        raise DataError("Self-benchmark metadata has an invalid attention DP size.")
    if dp > 1 and metadata.get("coordinate_system") != "iteration_totals_balanced_v1":
        raise DependencyError("Multi-rank self-benchmark seeding requires iteration_totals_balanced_v1 coordinates.")
    rows = pq.read_table(artifact.local_path).to_pylist()
    if len(rows) != artifact.row_count or len(rows) != metadata["row_count"]:
        raise DataError("Self-benchmark row count does not match the pinned manifest.")
    effective_targets = _effective_seed_targets(case, artifact, rows)
    workers = sorted({item.iteration.ranks[0].worker_id for item in case.observations})
    result = []
    for index, row in enumerate(rows):
        if dp > 1 and row.get("partition_policy") != "balanced_v1":
            raise DependencyError(f"Self-benchmark row {index} needs the balanced_v1 rank partition policy.")
        if _count(row, "dp", index, positive=True) != dp:
            raise DataError(f"Self-benchmark row {index} attention DP differs from its metadata.")
        phase = row.get("workload_kind")
        if phase not in {"prefill", "decode"}:
            raise DataError(f"Self-benchmark row {index} has an invalid workload kind.")
        if case.worker_role != "aggregated" and phase != case.worker_role:
            continue
        batch = _count(row, "batch_size", index, positive=True)
        tokens = _count(row, "total_prefill_tokens", index)
        kv = _count(row, "total_kv_read_tokens", index)
        latency = row.get("latency_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, int | float)
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise DataError(f"Self-benchmark row {index} has an invalid measured latency.")
        if (phase == "prefill" and tokens == 0) or (phase == "decode" and tokens != 0):
            raise DataError(f"Self-benchmark row {index} has inconsistent workload coordinates.")
        scheduled = (
            RequestMetrics(num_prefill_requests=batch, sum_prefill_tokens=tokens, sum_prefill_kv_tokens=kv)
            if phase == "prefill"
            else RequestMetrics(num_decode_requests=batch, sum_decode_kv_tokens=kv)
        )
        provenance = (
            {
                "source_metadata": {
                    "coordinate_system": "iteration_totals_balanced_v1",
                    "partition_policy": "balanced_v1",
                    "rank_workload_reconstruction": "replicated_table_coordinates",
                    "target_scope": "recorded_aggregate_latency",
                    "aggregate_label_carrier_dp_rank": 0,
                    "individual_rank_latencies_recorded": False,
                }
            }
            if dp > 1
            else {}
        )
        if index in effective_targets:
            target = effective_targets[index]
            latency = target["effective_latency_ms"]
            provenance["self_benchmark_target"] = target
            if target["replaced"] and "source_metadata" in provenance:
                provenance["source_metadata"]["target_scope"] = "native_normalized_aggregate_latency"
        for worker in workers:
            result.append(
                ForwardPassIteration(
                    ranks=tuple(
                        ForwardPassMetric(
                            fpm_id=index,
                            configuration_id=case.configuration_id,
                            version=1,
                            worker_id=worker,
                            dp_rank=rank,
                            counter_id=index,
                            wall_time_s=latency / 1000 if rank == 0 else 0.0,
                            scheduled=scheduled,
                            queued=RequestMetrics(),
                            capture_id=f"self-benchmark:{artifact.artifact_id}",
                            extra_metadata=provenance,
                        )
                        for rank in range(dp)
                    )
                )
            )
    if not result:
        raise DependencyError("Self-benchmark artifact has no compatible measured rows for this case.")
    return tuple(result)


def _effective_seed_targets(
    case: MeasurementCase, artifact: FpmArtifact, rows: Sequence[Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Transport the native loader's correction; never estimate labels in Python.

    Profiles without fake markers in the selected worker phase keep the existing
    path and incur no model construction. Otherwise one preparation call returns
    every marked source row for validation, independent of logical-worker count
    or the later number of queries.
    """
    marked = {index for index, row in enumerate(rows) if row.get("kv_seed_regime") == "fake_fallback"}
    if not marked or (
        case.worker_role != "aggregated"
        and not any(rows[index].get("workload_kind") == case.worker_role for index in marked)
    ):
        return {}
    context = PredictorContext(
        worker=case.configuration.worker_config_record,
        worker_role=case.worker_role,
        fpm_artifact=artifact,
    )
    predictor = create_scenario_predictor("warmup", context)
    try:
        operation = getattr(predictor, "fpm_seed_target_normalization", None)
        if not callable(operation):
            raise DependencyError("Self-benchmark fake-KV targets require native seed target normalization.")
        report = operation()
    finally:
        predictor.close()
    if report.get("row_count") != len(rows) or not isinstance(report.get("replacement_enabled"), bool):
        raise DataError("Native seed normalization does not describe the selected profile.")
    if report.get("policy") != "same_cell_batch_real_kv_extrapolation_v1":
        raise DependencyError("Unsupported native self-benchmark target normalization policy.")
    targets = {}
    for record in report.get("rows", ()):
        index = record.get("row_index")
        if type(index) is not int or index not in marked or index in targets:
            raise DataError("Native seed normalization has invalid or duplicate source row indices.")
        row = rows[index]
        if any(
            record.get(field) != row.get(field)
            for field in ("workload_kind", "batch_size", "total_prefill_tokens", "total_kv_read_tokens")
        ) or record.get("raw_latency_ms") != row.get("latency_ms"):
            raise DataError("Native seed normalization does not match the pinned source row.")
        effective = record.get("effective_latency_ms")
        if (
            isinstance(effective, bool)
            or not isinstance(effective, int | float)
            or not math.isfinite(effective)
            or effective <= 0
            or not isinstance(record.get("replaced"), bool)
        ):
            raise DataError("Native seed normalization returned an invalid effective target.")
        if not record["replaced"] and effective != record["raw_latency_ms"]:
            raise DataError("Native seed normalization changed a retained original target.")
        if record["replaced"] and not report["replacement_enabled"]:
            raise DataError("Native seed normalization replaced targets while replacement was disabled.")
        targets[index] = {
            **record,
            "policy": report["policy"],
            "replacement_enabled": report["replacement_enabled"],
            "artifact_id": artifact.artifact_id,
            "source_parquet_sha256": artifact.sha256,
            "source_metadata_sha256": artifact.metadata_sha256,
            "target_scope": "native_normalized_aggregate_latency"
            if record["replaced"]
            else "recorded_aggregate_latency",
        }
    if set(targets) != marked or report.get("replaced_count") != sum(r["replaced"] for r in targets.values()):
        raise DataError("Native seed normalization omitted marked rows or has inconsistent counts.")
    if report.get("kept_count") != sum(not r["replaced"] for r in targets.values()):
        raise DataError("Native seed normalization has inconsistent retained-target counts.")
    return targets


def seed_target_normalization_evidence(iterations: Sequence[ForwardPassIteration]) -> dict | None:
    """Hash-bind effective source targets separately from replicated seed rows."""
    unique = {}
    seeded = 0
    for iteration in iterations:
        record = iteration.ranks[0].extra_metadata.get("self_benchmark_target")
        if record is None:
            continue
        seeded += 1
        key = (record["source_parquet_sha256"], record["row_index"])
        if key in unique and unique[key] != record:
            raise ValueError("Replicated self-benchmark targets disagree on normalization provenance.")
        unique[key] = record
    if not unique:
        return None
    records = [unique[key] for key in sorted(unique)]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "policies": sorted({r["policy"] for r in records}),
        "source_parquet_sha256": sorted({r["source_parquet_sha256"] for r in records}),
        "source_metadata_sha256": sorted({r["source_metadata_sha256"] for r in records}),
        "marked_source_rows": len(records),
        "replaced_source_rows": sum(r["replaced"] for r in records),
        "kept_source_rows": sum(not r["replaced"] for r in records),
        "replicated_seed_rows": seeded,
        "replacement_enabled": all(r["replacement_enabled"] for r in records),
        "effective_targets_sha256": hashlib.sha256(payload).hexdigest(),
        "target_scope": "native_profile_seed_targets_not_additional_measurements",
    }


def _count(row: Mapping[str, Any], field: str, index: int, *, positive: bool = False) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise DataError(f"Self-benchmark row {index} has an invalid {field}.")
    return value
