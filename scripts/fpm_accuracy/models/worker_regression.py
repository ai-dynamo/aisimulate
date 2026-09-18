# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Case-local ownership of standalone AISim regression predictors by worker."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, Prediction, PredictorContext
from fpm_accuracy.types.forward_pass import ForwardPassInput, ForwardPassIteration, WorkloadKind, classify_workload

WorkerRole = Literal["prefill", "decode", "aggregated"]
REGRESSION_ROLE_POLICY = "offline_all_observations_all_active_ranks_v1"
REGRESSION_BUCKET_POLICY = "worker_role_full_rank_workload_v1"
AGGREGATED_BUCKETS = ("pure_decode", "contains_locally_mixed", "cross_rank_aggregated", "pure_prefill")


def infer_worker_roles(iterations: Iterable[ForwardPassIteration]) -> dict[str, WorkerRole]:
    """Infer regression roles from the full offline scheduled-work inventory.

    This examines future workloads, never latency targets, and does not claim
    to recover configured deployment roles. Idle ranks do not contribute.
    """
    kinds_by_worker: dict[str, set[WorkloadKind]] = {}
    for iteration in iterations:
        kinds = kinds_by_worker.setdefault(iteration.ranks[0].worker_id, set())
        kinds.update(rank.workload_kind for rank in iteration.ranks if rank.workload_kind is not WorkloadKind.EMPTY)
    result: dict[str, WorkerRole] = {}
    for worker_id, kinds in kinds_by_worker.items():
        if not kinds:
            raise ValueError(f"worker {worker_id!r} has no active scheduled work")
        result[worker_id] = (
            "prefill"
            if kinds == {WorkloadKind.PREFILL}
            else "decode"
            if kinds == {WorkloadKind.DECODE}
            else "aggregated"
        )
    return result


def regression_buckets(role: str) -> tuple[str, ...]:
    if role == "prefill":
        return ("pure_prefill",)
    if role == "decode":
        return ("pure_decode",)
    return AGGREGATED_BUCKETS


def regression_bucket(features: ForwardPassInput, role: str) -> str:
    """Report AISim's expected store; AISim itself selects the training store."""
    if role != "aggregated":
        return regression_buckets(role)[0]
    kinds = {
        classify_workload(
            sum_prefill_tokens=rank["scheduled_requests"]["sum_prefill_tokens"],
            num_decode_requests=rank["scheduled_requests"]["num_decode_requests"],
            sum_decode_kv_tokens=rank["scheduled_requests"]["sum_decode_kv_tokens"],
        )
        for rank in features.rank_payloads
    }
    kinds.discard(WorkloadKind.EMPTY)
    if WorkloadKind.MIXED in kinds:
        return "contains_locally_mixed"
    if WorkloadKind.PREFILL in kinds and WorkloadKind.DECODE in kinds:
        return "cross_rank_aggregated"
    if kinds == {WorkloadKind.DECODE}:
        return "pure_decode"
    if kinds == {WorkloadKind.PREFILL}:
        return "pure_prefill"
    raise ValueError("regression prediction has no active scheduled work")


def regression_metadata(
    features: ForwardPassInput,
    roles: Mapping[str, str],
    *,
    prior_observations: int = 0,
    min_observations: int = 5,
) -> dict[str, Any]:
    worker_id = str(features.rank_payloads[0]["worker_id"])
    role = roles[worker_id]
    return {
        "worker_id": worker_id,
        "regression_worker_role": role,
        "regression_bucket": regression_bucket(features, role),
        "regression_prior_observations": prior_observations,
        "regression_warmup": prior_observations < min_observations,
    }


class WorkerRegressionPredictor(ForwardPassTimePredictor):
    """Dispatch each worker to its own ordinary factory-created predictor."""

    def __init__(
        self,
        context: PredictorContext,
        roles: Mapping[str, WorkerRole],
        factory: Callable[[str, PredictorContext], ForwardPassTimePredictor],
    ) -> None:
        self.roles = dict(roles)
        self.options = {"max_observations": 64, **context.options}
        self.min_observations = int(self.options.get("min_observations", 5))
        self._children: dict[str, ForwardPassTimePredictor] = {}
        self._observations: dict[tuple[str, str], int] = {}
        try:
            for worker_id, role in self.roles.items():
                self._children[worker_id] = factory(
                    "regression", replace(context, worker_role=role, options=self.options)
                )
                stores = self._children[worker_id].diagnostics()["regression_stores"]
                available = {store["workload_kind"] for store in stores}
                missing = set(regression_buckets(role)) - available
                if missing:
                    raise ValueError(
                        f"worker {worker_id!r} regression stores missing {sorted(missing)}; "
                        f"available: {sorted(available)}"
                    )
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"worker cleanup also failed: {cleanup_error}")
            raise

    @property
    def id(self) -> str:
        return "regression"

    def routing_metadata(self, features: ForwardPassInput) -> dict[str, Any]:
        metadata = regression_metadata(features, self.roles, min_observations=self.min_observations)
        key = (metadata["worker_id"], metadata["regression_bucket"])
        prior = self._observations.get(key, 0)
        metadata.update(regression_prior_observations=prior, regression_warmup=prior < self.min_observations)
        return metadata

    def predict(self, features: ForwardPassInput) -> Prediction:
        metadata = self.routing_metadata(features)
        child = self._children[metadata["worker_id"]]
        stores = child.diagnostics()["regression_stores"]
        selected = next(store for store in stores if store["workload_kind"] == metadata["regression_bucket"])
        metadata.update(
            regression_store_ready=selected["ready"],
            regression_retained_observations=selected["retained_observations"],
            readiness="ready" if selected["ready"] else "insufficient_data",
        )
        prediction = child.predict(features)
        return Prediction(prediction.value_ms, {**prediction.metadata, **metadata})

    def tune(self, observations: Sequence[ForwardPassIteration]) -> None:
        for observation in observations:
            metadata = self.routing_metadata(observation.prediction_input())
            self._children[metadata["worker_id"]].tune([observation])
            key = (metadata["worker_id"], metadata["regression_bucket"])
            self._observations[key] = self._observations.get(key, 0) + 1

    def diagnostics(self) -> Mapping[str, Any]:
        return {
            "mode": "regression",
            "model_scope": "worker_id_within_case",
            "worker_role_policy": REGRESSION_ROLE_POLICY,
            "bucket_policy": REGRESSION_BUCKET_POLICY,
            "worker_roles": self.roles,
            "max_observations_per_store": self.options["max_observations"],
            "min_observations": self.min_observations,
            "spatial_bucket_count": self.options.get("bucket_count", 16),
            "worker_diagnostics": {worker_id: child.diagnostics() for worker_id, child in self._children.items()},
        }

    def close(self) -> None:
        children, self._children = self._children, {}
        errors: list[Exception] = []
        for child in children.values():
            try:
                child.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("failed to close worker regression predictors", errors)
