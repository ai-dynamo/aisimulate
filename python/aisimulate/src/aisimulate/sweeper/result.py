# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-versioned, machine-readable Sweeper result contracts.

The contract deliberately separates the complete candidate ledger from derived
views.  Exhaustive and optimizer-guided execution can therefore emit the same
payload without making a top-N list or Pareto frontier the source of truth.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .config import Candidate, SmartSearchConfig
from .recommend import LoadRecommendation, LoadTarget
from .replay import canonical_json, validate_json_value

RESULT_SCHEMA_VERSION = "1.1"


class SearchStrategy(str, Enum):
    """How the candidate space was traversed."""

    EXHAUSTIVE = "exhaustive"
    OPTIMIZER_GUIDED = "optimizer_guided"


class CandidateRetention(str, Enum):
    """Which candidate records are embedded in a :class:`SweepResult`."""

    ALL = "all"
    FEASIBLE = "feasible"
    VIEWS = "views"


class CandidateStatus(str, Enum):
    """Terminal status of one unique candidate attempt."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class ReasonCategory(str, Enum):
    """Stable, actionable categories for non-feasible candidates."""

    GPU_BUDGET = "gpu_budget"
    KV_CAPACITY = "kv_capacity"
    BACKEND_TOPOLOGY = "backend_topology"
    STRICT_SLA = "strict_sla"
    RUNTIME_TIMEOUT = "runtime_timeout"
    CANDIDATE_MATERIALIZATION = "candidate_materialization"
    REPLAY_RUNTIME = "replay_runtime"
    RUNNER_CONTRACT = "runner_contract"
    INVALID_METRICS = "invalid_metrics"
    UNKNOWN = "unknown"


class OperationProvenance(BaseModel):
    """Provenance for one operation-level estimate used by a candidate."""

    model_config = ConfigDict(extra="forbid")

    operation: str
    source: str
    version: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CandidateProvenance(BaseModel):
    """Inputs and evidence needed to explain one candidate estimate."""

    model_config = ConfigDict(extra="forbid")

    model: str
    hardware: str
    backend: str | None = None
    backend_version: str | None = None
    performance_data: list[dict[str, JsonValue]] = Field(default_factory=list)
    topology: dict[str, JsonValue] = Field(default_factory=dict)
    workload: dict[str, JsonValue] = Field(default_factory=dict)
    objective: dict[str, JsonValue] = Field(default_factory=dict)
    sla: dict[str, JsonValue] | None = None
    power: dict[str, JsonValue] = Field(default_factory=dict)
    operations: list[OperationProvenance] = Field(default_factory=list)
    runner_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_json(self) -> CandidateProvenance:
        validate_json_value(self.model_dump(mode="python"), path="candidate provenance")
        return self


class CandidateRecord(BaseModel):
    """One row in the complete candidate ledger."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^candidate-[0-9]{6,}$")
    status: CandidateStatus
    config: dict[str, JsonValue]
    used_gpus: int | None = Field(default=None, ge=0)
    score: float | None = Field(default=None, allow_inf_nan=False)
    metrics: dict[str, float] = Field(default_factory=dict)
    objectives: dict[str, float] | None = None
    reason_category: ReasonCategory | None = None
    reason: str | None = None
    provenance: CandidateProvenance

    @model_validator(mode="after")
    def _validate_status_payload(self) -> CandidateRecord:
        validate_json_value(self.config, path=f"candidate {self.candidate_id} config")
        non_finite_metrics = [name for name, value in self.metrics.items() if not math.isfinite(value)]
        non_finite_objectives = [name for name, value in (self.objectives or {}).items() if not math.isfinite(value)]
        if non_finite_metrics or non_finite_objectives:
            raise ValueError(
                "candidate metrics and objectives must be finite; got "
                f"metrics={non_finite_metrics}, objectives={non_finite_objectives}"
            )
        if self.status is CandidateStatus.FEASIBLE:
            if self.reason_category is not None or self.reason is not None:
                raise ValueError("a feasible candidate cannot carry a rejection reason")
            if self.used_gpus is None or self.score is None:
                raise ValueError("a feasible candidate requires used_gpus and score")
        elif self.reason_category is None or not self.reason:
            raise ValueError("a non-feasible candidate requires a reason category and detail")
        return self

    def as_candidate(self) -> Candidate:
        """Return the compatibility candidate for a feasible record."""

        if self.status is not CandidateStatus.FEASIBLE:
            raise ValueError(f"candidate {self.candidate_id} is not feasible")
        assert self.used_gpus is not None and self.score is not None
        return Candidate(
            config=self.config,
            used_gpus=self.used_gpus,
            score=self.score,
            metrics=self.metrics,
            objectives=self.objectives,
        )


class SweepCounts(BaseModel):
    """Counts for the full run, independent of candidate retention."""

    model_config = ConfigDict(extra="forbid")

    evaluated: int = Field(ge=0)
    feasible: int = Field(ge=0)
    infeasible: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    timed_out: int = Field(ge=0)
    failed: int = Field(ge=0)
    cache_hits: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_total(self) -> SweepCounts:
        # Unsupported suggestions fail capability gating before evaluation.
        expected = self.feasible + self.infeasible + self.timed_out + self.failed
        if self.evaluated != expected:
            raise ValueError("evaluated must equal feasible + infeasible + timed_out + failed")
        return self


class ResultViews(BaseModel):
    """Stable candidate-ID views derived from the candidate ledger."""

    model_config = ConfigDict(extra="forbid")

    top_n: list[str] = Field(default_factory=list)
    pareto_front: list[str] = Field(default_factory=list)


class LoadRecommendationRecord(BaseModel):
    """One retained candidate's load-sizing result.

    The record references the canonical candidate ledger rather than embedding a
    second copy of the candidate. Uncapped fields always describe the true minimum;
    deployed fields describe the optional GPU-capped deployment.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^candidate-[0-9]{6,}$")
    capacity_per_replica: float = Field(gt=0, allow_inf_nan=False)
    capacity_per_gpu: float = Field(gt=0, allow_inf_nan=False)
    replicas_needed: int = Field(ge=1)
    total_gpus_needed: int = Field(ge=1)
    deployed_replicas: int = Field(ge=1)
    deployed_gpus: int = Field(ge=1)
    supported_load: float = Field(gt=0, allow_inf_nan=False)
    load_served_pct: float = Field(ge=0, le=100, allow_inf_nan=False)
    limiting_role: str = Field(min_length=1)
    partial: bool

    @classmethod
    def from_recommendation(
        cls,
        candidate_id: str,
        recommendation: LoadRecommendation,
    ) -> LoadRecommendationRecord:
        """Reference a standalone sizing result from the canonical ledger."""

        return cls(
            candidate_id=candidate_id,
            capacity_per_replica=recommendation.capacity_per_replica,
            capacity_per_gpu=recommendation.capacity_per_gpu,
            replicas_needed=recommendation.replicas_needed,
            total_gpus_needed=recommendation.total_gpus_needed,
            deployed_replicas=recommendation.deployed_replicas,
            deployed_gpus=recommendation.deployed_gpus,
            supported_load=recommendation.supported_load,
            load_served_pct=recommendation.load_served_pct,
            limiting_role=recommendation.limiting_role,
            partial=recommendation.partial,
        )


class LoadRecommendationFailureRecord(BaseModel):
    """One actionable sizing rejection, optionally linked to a retained candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str | None = Field(
        default=None,
        pattern=r"^candidate-[0-9]{6,}$",
    )
    reason: str = Field(min_length=1)


class LoadRecommendationView(BaseModel):
    """Target, ranked deployments, or reasons that no deployment was feasible."""

    model_config = ConfigDict(extra="forbid")

    target: LoadTarget
    recommendations: list[LoadRecommendationRecord] = Field(default_factory=list)
    no_feasible_reasons: list[LoadRecommendationFailureRecord] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _validate_outcome(self) -> LoadRecommendationView:
        candidate_ids = [item.candidate_id for item in self.recommendations]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("load recommendation candidate IDs must be unique")
        failure_ids = [
            item.candidate_id
            for item in self.no_feasible_reasons
            if item.candidate_id is not None
        ]
        if len(failure_ids) != len(set(failure_ids)):
            raise ValueError("load recommendation failure candidate IDs must be unique")
        if self.recommendations and self.no_feasible_reasons:
            raise ValueError("a successful load recommendation cannot carry no-feasible reasons")
        if not self.recommendations and not self.no_feasible_reasons:
            raise ValueError("an empty load recommendation requires no-feasible reasons")
        for item in self.recommendations:
            if item.deployed_replicas > item.replicas_needed:
                raise ValueError("deployed_replicas cannot exceed replicas_needed")
            expected_replicas = max(
                1,
                math.ceil(self.target.value / item.capacity_per_replica),
            )
            if item.replicas_needed != expected_replicas:
                raise ValueError(
                    "replicas_needed must be the uncapped minimum for the target"
                )
            expected_supported_load = item.capacity_per_replica * item.deployed_replicas
            if not math.isclose(item.supported_load, expected_supported_load):
                raise ValueError("supported_load must equal capacity_per_replica * deployed_replicas")
            expected_served_pct = min(100.0, item.supported_load / self.target.value * 100.0)
            if not math.isclose(item.load_served_pct, expected_served_pct):
                raise ValueError("load_served_pct must match supported_load and target")
            if item.partial != (item.load_served_pct < 100.0):
                raise ValueError("partial must indicate load_served_pct below 100")
            if item.partial:
                if not self.target.allow_partial:
                    raise ValueError(
                        "a partial recommendation requires target.allow_partial=true"
                    )
                if self.target.max_gpus is None:
                    raise ValueError("a partial recommendation requires target.max_gpus")
                if item.deployed_replicas >= item.replicas_needed:
                    raise ValueError(
                        "a partial recommendation must deploy fewer than replicas_needed"
                    )
            elif item.deployed_replicas != item.replicas_needed:
                raise ValueError(
                    "a full recommendation must deploy the uncapped replica count"
                )
            if (
                self.target.max_gpus is not None
                and item.deployed_gpus > self.target.max_gpus
            ):
                raise ValueError("deployed_gpus cannot exceed target.max_gpus")
        return self


class SweepRunProvenance(BaseModel):
    """Run-wide reproducibility information."""

    model_config = ConfigDict(extra="forbid")

    search_strategy: SearchStrategy
    implementation: str = "aisimulate.sweeper"
    implementation_version: str | None = None
    run_id: str
    created_at: datetime
    input_fingerprint: str
    config: dict[str, JsonValue]

    @model_validator(mode="after")
    def _require_json_config(self) -> SweepRunProvenance:
        validate_json_value(self.config, path="result provenance config")
        return self


class SweepResult(BaseModel):
    """Canonical result shared by exhaustive and optimizer-guided Sweeper runs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"] = RESULT_SCHEMA_VERSION
    candidate_retention: CandidateRetention = CandidateRetention.ALL
    counts: SweepCounts
    candidates: list[CandidateRecord]
    views: ResultViews
    load_recommendation: LoadRecommendationView | None = None
    provenance: SweepRunProvenance

    @model_validator(mode="after")
    def _validate_identity_and_views(self) -> SweepResult:
        if self.schema_version == "1.0" and self.load_recommendation is not None:
            raise ValueError("load_recommendation requires schema_version='1.1'")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        retained_ids = set(candidate_ids)
        recommendation_ids = {
            item.candidate_id
            for item in (
                self.load_recommendation.recommendations
                if self.load_recommendation is not None
                else []
            )
        }
        failure_ids = {
            item.candidate_id
            for item in (
                self.load_recommendation.no_feasible_reasons
                if self.load_recommendation is not None
                else []
            )
            if item.candidate_id is not None
        }
        referenced_ids = (
            set(self.views.top_n)
            | set(self.views.pareto_front)
            | recommendation_ids
            | failure_ids
        )
        missing = referenced_ids - retained_ids
        if missing:
            raise ValueError(f"result views reference candidates not retained: {sorted(missing)}")
        if self.views.top_n and self.views.pareto_front:
            raise ValueError("a result cannot carry scalar top-N and Pareto views together")
        for name, view in (
            ("top_n", self.views.top_n),
            ("pareto_front", self.views.pareto_front),
        ):
            if len(view) != len(set(view)):
                raise ValueError(f"{name} candidate IDs must be unique")
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        if self.load_recommendation is not None:
            for item in self.load_recommendation.recommendations:
                used_gpus = by_id[item.candidate_id].used_gpus
                if used_gpus is None or used_gpus < 1:
                    raise ValueError("a load recommendation requires a positive candidate GPU count")
                if item.total_gpus_needed != item.replicas_needed * used_gpus:
                    raise ValueError("total_gpus_needed must match replicas_needed and candidate GPUs")
                if item.deployed_gpus != item.deployed_replicas * used_gpus:
                    raise ValueError("deployed_gpus must match deployed_replicas and candidate GPUs")
                expected_capacity_per_gpu = item.capacity_per_replica / used_gpus
                if not math.isclose(item.capacity_per_gpu, expected_capacity_per_gpu):
                    raise ValueError("capacity_per_gpu must match capacity_per_replica and candidate GPUs")
        non_feasible = [
            candidate_id
            for candidate_id in (self.views.top_n + self.views.pareto_front + list(recommendation_ids))
            if by_id[candidate_id].status is not CandidateStatus.FEASIBLE
        ]
        if non_feasible:
            raise ValueError(f"result views reference non-feasible candidates: {non_feasible}")
        status_counts = dict.fromkeys(CandidateStatus, 0)
        for candidate in self.candidates:
            status_counts[candidate.status] += 1
        if self.candidate_retention is CandidateRetention.ALL:
            if status_counts[CandidateStatus.FEASIBLE] != self.counts.feasible:
                raise ValueError("retained feasible candidates do not match counts")
            if status_counts[CandidateStatus.INFEASIBLE] != self.counts.infeasible:
                raise ValueError("retained infeasible candidates do not match counts")
            if status_counts[CandidateStatus.UNSUPPORTED] != self.counts.unsupported:
                raise ValueError("retained unsupported candidates do not match counts")
            if status_counts[CandidateStatus.TIMED_OUT] != self.counts.timed_out:
                raise ValueError("retained timed-out candidates do not match counts")
            if status_counts[CandidateStatus.FAILED] != self.counts.failed:
                raise ValueError("retained failed candidates do not match counts")
        elif self.candidate_retention is CandidateRetention.FEASIBLE:
            if any(
                candidate.status is not CandidateStatus.FEASIBLE
                for candidate in self.candidates
            ):
                raise ValueError(
                    "candidate_retention='feasible' can retain only feasible candidates"
                )
            if len(self.candidates) != self.counts.feasible:
                raise ValueError(
                    "candidate_retention='feasible' must retain every feasible candidate"
                )
        elif retained_ids != referenced_ids:
            extras = sorted(retained_ids - referenced_ids)
            raise ValueError(
                "candidate_retention='views' retained unreferenced candidates: "
                f"{extras}"
            )
        return self

    @property
    def feasible_candidates(self) -> list[Candidate]:
        """All retained feasible candidates, in evaluation order."""

        return [record.as_candidate() for record in self.candidates if record.status is CandidateStatus.FEASIBLE]

    @property
    def selected_candidates(self) -> list[Candidate]:
        """The scalar top-N or Pareto view as compatibility candidates."""

        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        selected_ids = self.views.pareto_front or self.views.top_n
        return [by_id[candidate_id].as_candidate() for candidate_id in selected_ids]

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the lossless canonical representation using strict JSON."""

        payload = self.model_dump(
            mode="json",
            exclude={"load_recommendation"} if self.schema_version == "1.0" else None,
        )
        validate_json_value(payload, path="sweep result")
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> SweepResult:
        """Parse and validate a lossless canonical JSON result."""

        return cls.model_validate_json(payload)

    def to_csv(self) -> str:
        """Return the documented one-row-per-candidate flattened CSV view.

        Nested values remain canonical JSON cells.  CSV is intended for analysis;
        the JSON envelope remains the lossless interchange format because CSV does
        not encode empty-run provenance or all run-wide structure.
        """

        output = io.StringIO(newline="")
        legacy_fieldnames = [
            "schema_version",
            "candidate_id",
            "status",
            "reason_category",
            "reason",
            "used_gpus",
            "score",
            "config_json",
            "metrics_json",
            "objectives_json",
            "provenance_json",
            "is_top_n",
            "is_pareto",
        ]
        load_fieldnames = [
            "load_target_kind",
            "load_target_value",
            "load_target_max_gpus",
            "load_target_allow_partial",
            "recommendation_capacity_per_replica",
            "recommendation_capacity_per_gpu",
            "recommendation_replicas_needed",
            "recommendation_total_gpus_needed",
            "recommendation_deployed_replicas",
            "recommendation_deployed_gpus",
            "recommendation_supported_load",
            "recommendation_load_served_pct",
            "recommendation_limiting_role",
            "recommendation_partial",
            "recommendation_failure_reason",
            "recommendation_no_feasible_reasons_json",
        ]
        fieldnames = legacy_fieldnames + (
            load_fieldnames if self.schema_version == "1.1" else []
        )
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        top_n = set(self.views.top_n)
        pareto = set(self.views.pareto_front)
        load_view = self.load_recommendation
        recommendations = (
            {recommendation.candidate_id: recommendation for recommendation in load_view.recommendations}
            if load_view is not None
            else {}
        )
        recommendation_failures = (
            {
                failure.candidate_id: failure
                for failure in load_view.no_feasible_reasons
                if failure.candidate_id is not None
            }
            if load_view is not None
            else {}
        )
        for candidate in self.candidates:
            recommendation = recommendations.get(candidate.candidate_id)
            recommendation_failure = recommendation_failures.get(
                candidate.candidate_id
            )
            writer.writerow(
                {
                    "schema_version": self.schema_version,
                    "candidate_id": candidate.candidate_id,
                    "status": candidate.status.value,
                    "reason_category": (
                        candidate.reason_category.value if candidate.reason_category is not None else ""
                    ),
                    "reason": candidate.reason or "",
                    "used_gpus": "" if candidate.used_gpus is None else candidate.used_gpus,
                    "score": "" if candidate.score is None else candidate.score,
                    "config_json": canonical_json(candidate.config),
                    "metrics_json": canonical_json(candidate.metrics),
                    "objectives_json": canonical_json(candidate.objectives),
                    "provenance_json": canonical_json(candidate.provenance),
                    "is_top_n": candidate.candidate_id in top_n,
                    "is_pareto": candidate.candidate_id in pareto,
                    "load_target_kind": load_view.target.kind if load_view is not None else "",
                    "load_target_value": load_view.target.value if load_view is not None else "",
                    "load_target_max_gpus": (
                        load_view.target.max_gpus
                        if load_view is not None and load_view.target.max_gpus is not None
                        else ""
                    ),
                    "load_target_allow_partial": (load_view.target.allow_partial if load_view is not None else ""),
                    "recommendation_capacity_per_replica": (
                        recommendation.capacity_per_replica if recommendation is not None else ""
                    ),
                    "recommendation_capacity_per_gpu": (
                        recommendation.capacity_per_gpu if recommendation is not None else ""
                    ),
                    "recommendation_replicas_needed": (
                        recommendation.replicas_needed if recommendation is not None else ""
                    ),
                    "recommendation_total_gpus_needed": (
                        recommendation.total_gpus_needed if recommendation is not None else ""
                    ),
                    "recommendation_deployed_replicas": (
                        recommendation.deployed_replicas if recommendation is not None else ""
                    ),
                    "recommendation_deployed_gpus": (
                        recommendation.deployed_gpus if recommendation is not None else ""
                    ),
                    "recommendation_supported_load": (
                        recommendation.supported_load if recommendation is not None else ""
                    ),
                    "recommendation_load_served_pct": (
                        recommendation.load_served_pct if recommendation is not None else ""
                    ),
                    "recommendation_limiting_role": (
                        recommendation.limiting_role if recommendation is not None else ""
                    ),
                    "recommendation_partial": (recommendation.partial if recommendation is not None else ""),
                    "recommendation_failure_reason": (
                        recommendation_failure.reason
                        if recommendation_failure is not None
                        else ""
                    ),
                    "recommendation_no_feasible_reasons_json": (
                        canonical_json(load_view.no_feasible_reasons)
                        if load_view is not None and load_view.no_feasible_reasons
                        else ""
                    ),
                }
            )
        return output.getvalue()


def make_run_provenance(
    config: SmartSearchConfig,
    *,
    search_strategy: SearchStrategy,
    run_id: str,
    implementation_version: str | None = None,
    created_at: datetime | None = None,
) -> SweepRunProvenance:
    """Build deterministic-input run provenance for either execution strategy."""

    config_payload = config.model_dump(mode="json")
    fingerprint = hashlib.sha256(canonical_json(config_payload).encode()).hexdigest()
    if implementation_version is None:
        try:
            implementation_version = version("aisimulate")
        except PackageNotFoundError:
            implementation_version = None
    return SweepRunProvenance(
        search_strategy=search_strategy,
        implementation_version=implementation_version,
        run_id=run_id,
        created_at=created_at or datetime.now(UTC),
        input_fingerprint=f"sha256:{fingerprint}",
        config=config_payload,
    )


def make_candidate_provenance(
    config: SmartSearchConfig,
    candidate_config: dict[str, JsonValue],
    *,
    metrics: dict[str, float] | None = None,
    runner_metadata: dict[str, JsonValue] | None = None,
) -> CandidateProvenance:
    """Normalize core inputs plus optional runner evidence into one provenance record."""

    runner_metadata = runner_metadata or {}
    topology_fields = {
        key: value
        for key, value in candidate_config.items()
        if key
        in {
            "deployment_mode",
            "tp",
            "pp",
            "attention_dp",
            "moe_tp",
            "moe_ep",
            "strategy",
            "replicas",
            "prefill_tp",
            "prefill_pp",
            "prefill_attention_dp",
            "prefill_moe_tp",
            "prefill_moe_ep",
            "prefill_strategy",
            "prefill_replicas",
            "decode_tp",
            "decode_pp",
            "decode_attention_dp",
            "decode_moe_tp",
            "decode_moe_ep",
            "decode_strategy",
            "decode_replicas",
        }
    }
    raw_performance_data = runner_metadata.get("performance_data", [])
    performance_data = (
        raw_performance_data
        if isinstance(raw_performance_data, list) and all(isinstance(item, dict) for item in raw_performance_data)
        else []
    )
    if not performance_data and candidate_config.get("backend_version") is not None:
        performance_data = [
            {
                "source": "aiconfigurator_performance_database",
                "hardware": config.search_space.hardware_sku,
                "backend": candidate_config.get("backend"),
                "version": candidate_config["backend_version"],
            }
        ]
    raw_operations = runner_metadata.get("operations", [])
    operations: list[OperationProvenance] = []
    if isinstance(raw_operations, list):
        for item in raw_operations:
            if isinstance(item, dict) and "operation" in item and "source" in item:
                item_metadata = item.get("metadata")
                metadata = dict(item_metadata) if isinstance(item_metadata, dict) else {}
                metadata.update(
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"operation", "source", "version", "metadata"}
                    }
                )
                operations.append(
                    OperationProvenance(
                        operation=str(item["operation"]),
                        source=str(item["source"]),
                        version=(str(item["version"]) if item.get("version") is not None else None),
                        metadata=metadata,
                    )
                )
    power = {key: value for key, value in (metrics or {}).items() if "power" in key or "energy" in key}
    raw_power = runner_metadata.get("power")
    if isinstance(raw_power, dict):
        power.update(raw_power)
    goal_payload = config.goal.model_dump(mode="json")
    return CandidateProvenance(
        model=config.search_space.model_name,
        hardware=config.search_space.hardware_sku,
        backend=(str(candidate_config["backend"]) if candidate_config.get("backend") is not None else None),
        backend_version=(
            str(candidate_config["backend_version"]) if candidate_config.get("backend_version") is not None else None
        ),
        performance_data=performance_data,
        topology=topology_fields,
        workload=config.workload.model_dump(mode="json"),
        objective=goal_payload,
        sla=goal_payload.get("sla"),
        power=power,
        operations=operations,
        runner_metadata=runner_metadata,
    )


def retain_candidate_records(
    records: list[CandidateRecord],
    *,
    retention: CandidateRetention,
    views: ResultViews,
    additional_view_ids: Iterable[str] = (),
) -> list[CandidateRecord]:
    """Apply the requested payload-retention policy without changing run counts."""

    if retention is CandidateRetention.ALL:
        return records
    if retention is CandidateRetention.FEASIBLE:
        return [record for record in records if record.status is CandidateStatus.FEASIBLE]
    selected = set(views.top_n) | set(views.pareto_front) | set(additional_view_ids)
    return [record for record in records if record.candidate_id in selected]
