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
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, field_validator, model_validator

from ..power import POWER_FIELDS, normalize_power_summary, power_unavailable_reason
from .config import Candidate, SmartSearchConfig
from .replay import ReplaySpec, canonical_json, validate_json_value

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
    RESOURCE_LIMITED = "resource_limited"


class ReasonCategory(str, Enum):
    """Stable, actionable categories for non-feasible candidates."""

    GPU_BUDGET = "gpu_budget"
    KV_CAPACITY = "kv_capacity"
    SLA_CONSTRAINT = "sla_constraint"
    LOAD_CONSTRAINT = "load_constraint"
    BACKEND_TOPOLOGY = "backend_topology"
    RUNTIME_TIMEOUT = "runtime_timeout"
    RESOURCE_LIMIT = "resource_limit"
    CANDIDATE_MATERIALIZATION = "candidate_materialization"
    REPLAY_RUNTIME = "replay_runtime"
    RUNNER_CONTRACT = "runner_contract"
    INVALID_METRICS = "invalid_metrics"
    NO_SAMPLES = "no_samples"
    PARALLEL_PROJECTION = "parallel_projection"
    ADAPTER_CONSTRAINT = "adapter_constraint"
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
    prediction_config: dict[str, JsonValue] | None = None
    used_gpus: int | None = Field(default=None, ge=0)
    score: float | None = Field(default=None, allow_inf_nan=False)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    objectives: dict[str, float] | None = None
    reason_category: ReasonCategory | None = None
    reason: str | None = None
    provenance: CandidateProvenance

    @field_validator("metrics", mode="before")
    @classmethod
    def _normalize_power_fields(cls, metrics):
        if isinstance(metrics, dict) and metrics:
            return {**metrics, **normalize_power_summary(metrics)}
        return metrics

    @model_validator(mode="after")
    def _validate_status_payload(self) -> CandidateRecord:
        validate_json_value(self.config, path=f"candidate {self.candidate_id} config")
        validate_json_value(
            self.prediction_config,
            path=f"candidate {self.candidate_id} prediction config",
        )
        non_finite_metrics = [
            name
            for name, value in self.metrics.items()
            if not (value is None and name in POWER_FIELDS) and (value is None or not math.isfinite(value))
        ]
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
            prediction_config=self.prediction_config,
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
    resource_limited: int = Field(default=0, ge=0)
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

    _execution_resources: dict = PrivateAttr(default_factory=dict)

    @property
    def execution_resources(self) -> dict:
        """Host-specific supervision evidence, excluded from portable result identity."""
        return deepcopy(self._execution_resources)

    schema_version: Literal["1.1"] = RESULT_SCHEMA_VERSION
    candidate_retention: CandidateRetention = CandidateRetention.ALL
    counts: SweepCounts
    candidates: list[CandidateRecord]
    views: ResultViews
    provenance: SweepRunProvenance

    @model_validator(mode="before")
    @classmethod
    def _upgrade_previous_schema(cls, value):
        if isinstance(value, Mapping) and value.get("schema_version") == "1.0":
            # Version 1.0 had no host-limited outcome. Convert explicitly so
            # serializing a loaded legacy result never mislabels new fields.
            counts = value.get("counts") or {}
            candidates = value.get("candidates") or []
            if isinstance(counts, Mapping) and counts.get("resource_limited", 0):
                raise ValueError("schema 1.0 cannot contain resource-limited counts")
            if any(isinstance(item, Mapping) and item.get("status") == "resource_limited" for item in candidates):
                raise ValueError("schema 1.0 cannot contain resource-limited candidates")
            return {**value, "schema_version": RESULT_SCHEMA_VERSION}
        return value

    @model_validator(mode="after")
    def _validate_identity_and_views(self) -> SweepResult:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        retained_ids = set(candidate_ids)
        missing = (set(self.views.top_n) | set(self.views.pareto_front)) - retained_ids
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
        non_feasible = [
            candidate_id
            for candidate_id in self.views.top_n + self.views.pareto_front
            if by_id[candidate_id].status is not CandidateStatus.FEASIBLE
        ]
        if non_feasible:
            raise ValueError(f"result views reference non-feasible candidates: {non_feasible}")
        if self.candidate_retention is CandidateRetention.ALL:
            status_counts = dict.fromkeys(CandidateStatus, 0)
            for candidate in self.candidates:
                status_counts[candidate.status] += 1
            if status_counts[CandidateStatus.FEASIBLE] != self.counts.feasible:
                raise ValueError("retained feasible candidates do not match counts")
            if status_counts[CandidateStatus.INFEASIBLE] != self.counts.infeasible:
                raise ValueError("retained infeasible candidates do not match counts")
            if status_counts[CandidateStatus.UNSUPPORTED] != self.counts.unsupported:
                raise ValueError("retained unsupported candidates do not match counts")
            if status_counts[CandidateStatus.TIMED_OUT] != self.counts.timed_out:
                raise ValueError("retained timed-out candidates do not match counts")
            if status_counts[CandidateStatus.RESOURCE_LIMITED] != self.counts.resource_limited:
                raise ValueError("retained resource-limited candidates do not match counts")
            if status_counts[CandidateStatus.FAILED] != self.counts.failed:
                raise ValueError("retained failed candidates do not match counts")
        return self

    @property
    def feasible_candidates(self) -> list[Candidate]:
        """All retained feasible candidates, in evaluation order."""

        return [record.as_candidate() for record in self.candidates if record.status is CandidateStatus.FEASIBLE]

    @property
    def selected_candidate_ids(self) -> list[str]:
        """Candidate IDs in the active scalar or Pareto view."""

        return list(self.views.pareto_front or self.views.top_n)

    @property
    def selected_candidates(self) -> list[Candidate]:
        """The scalar top-N or Pareto view as candidate objects."""

        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        return [by_id[candidate_id].as_candidate() for candidate_id in self.selected_candidate_ids]

    def with_selected_prediction_configs(
        self,
        selections: list[tuple[str, Mapping[str, JsonValue]]],
    ) -> SweepResult:
        """Return a result whose selected view maps one-to-one to concrete configs.

        ``selections`` must be an order-preserving subset of the current view. It
        is used by output producers after their final adapter canonicalization and
        deduplication so every selected candidate ID corresponds to one artifact.
        """

        selected_ids = [candidate_id for candidate_id, _ in selections]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected prediction candidate IDs must be unique")
        selected_set = set(selected_ids)
        missing = selected_set - set(self.selected_candidate_ids)
        if missing:
            raise ValueError(f"selected prediction candidate IDs are not in the current view: {sorted(missing)}")
        expected_order = [candidate_id for candidate_id in self.selected_candidate_ids if candidate_id in selected_set]
        if selected_ids != expected_order:
            raise ValueError("selected prediction candidate IDs must preserve the current view order")
        configs_by_id = {candidate_id: deepcopy(dict(config)) for candidate_id, config in selections}
        for candidate_id, config in configs_by_id.items():
            validate_json_value(config, path=f"selected prediction config {candidate_id}")
        candidates = [
            candidate.model_copy(update={"prediction_config": configs_by_id[candidate.candidate_id]})
            if candidate.candidate_id in configs_by_id
            else candidate
            for candidate in self.candidates
        ]
        views = ResultViews(
            pareto_front=selected_ids if self.views.pareto_front else [],
            top_n=selected_ids if self.views.top_n else [],
        )
        payload = self.model_dump(mode="python")
        payload.update(candidates=candidates, views=views)
        result = type(self).model_validate(payload)
        result._execution_resources = deepcopy(self._execution_resources)
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the lossless canonical representation using strict JSON."""

        payload = self.model_dump(mode="json")
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
        fieldnames = [
            "schema_version",
            "candidate_id",
            "status",
            "reason_category",
            "reason",
            "used_gpus",
            "score",
            "power_w",
            "power_coverage",
            "power_source",
            "config_json",
            "prediction_config_json",
            "metrics_json",
            "objectives_json",
            "provenance_json",
            "is_top_n",
            "is_pareto",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        top_n = set(self.views.top_n)
        pareto = set(self.views.pareto_front)
        for candidate in self.candidates:
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
                    "power_w": candidate.metrics.get("power_w", ""),
                    "power_coverage": candidate.metrics.get("power_coverage", ""),
                    "power_source": candidate.provenance.power.get("source", ""),
                    "config_json": canonical_json(candidate.config),
                    "prediction_config_json": canonical_json(candidate.prediction_config),
                    "metrics_json": canonical_json(candidate.metrics),
                    "objectives_json": canonical_json(candidate.objectives),
                    "provenance_json": canonical_json(candidate.provenance),
                    "is_top_n": candidate.candidate_id in top_n,
                    "is_pareto": candidate.candidate_id in pareto,
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
    candidate_config: dict[str, JsonValue],
    *,
    replay_spec: ReplaySpec | None = None,
    metrics: dict[str, float | None] | None = None,
    runner_metadata: dict[str, JsonValue] | None = None,
) -> CandidateProvenance:
    """Normalize a materialized replay plus optional runner evidence.

    Pre-materialization rejections have no concrete replay specification, so
    their candidate provenance contains only the concrete fields known at the
    rejection point. The complete input search domain remains in run provenance.
    """

    runner_metadata = runner_metadata or {}
    deployment = replay_spec.backend_deployment if replay_spec is not None else None
    topology_fields = (
        {
            "deployment_mode": deployment.deployment_mode,
            **deepcopy(deployment.parallel_config),
        }
        if deployment is not None
        else {
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
                "prefill_hardware_sku",
                "prefill_pp",
                "prefill_attention_dp",
                "prefill_moe_tp",
                "prefill_moe_ep",
                "prefill_strategy",
                "prefill_replicas",
                "decode_tp",
                "decode_hardware_sku",
                "decode_pp",
                "decode_attention_dp",
                "decode_moe_tp",
                "decode_moe_ep",
                "decode_strategy",
                "decode_replicas",
            }
        }
    )
    raw_performance_data = runner_metadata.get("performance_data", [])
    encoder = candidate_config.get("encoder")
    if isinstance(encoder, dict):
        topology_fields["encoder"] = deepcopy(encoder)
        topology_fields["language_gpus"] = candidate_config.get("language_gpus")
        topology_fields["total_gpus"] = candidate_config.get("used_gpus")
    performance_data: list[dict[str, JsonValue]] = (
        deepcopy(raw_performance_data)
        if isinstance(raw_performance_data, list) and all(isinstance(item, dict) for item in raw_performance_data)
        else []
    )
    if deployment is not None:
        for role, raw_metadata in deployment.performance_model_metadata.items():
            if isinstance(raw_metadata, dict):
                performance_data.append(
                    {
                        "role": role,
                        "source": "backend_deployment",
                        **deepcopy(raw_metadata),
                    }
                )
    identity_config: dict[str, JsonValue] = {}
    if isinstance(encoder, dict):
        performance_data.append({"role": "encoder", "provider": "aic", "database_mode": "SILICON", **deepcopy(encoder)})
    if deployment is not None:
        for raw_metadata in deployment.performance_model_metadata.values():
            if isinstance(raw_metadata, dict) and isinstance(raw_metadata.get("config"), dict):
                identity_config = raw_metadata["config"]
                break
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
    if power:
        power.update(
            {
                "source": "runner_reported",
                "scope": "unspecified",
                "publication_status": "reported",
            }
        )
    raw_power = runner_metadata.get("power")
    if isinstance(raw_power, dict):
        power.update(raw_power)
    if metrics is not None:
        normalized_power = normalize_power_summary(metrics)
        power.update(normalized_power)
        if normalized_power["power_w"] is None:
            power["publication_status"] = (
                "withheld" if normalized_power["power_coverage"] is not None else "unavailable"
            )
            power["unavailable_reason"] = power_unavailable_reason(normalized_power)
    workload: dict[str, JsonValue] = {}
    goal_payload: dict[str, JsonValue] = {}
    if replay_spec is not None:
        workload = deepcopy(replay_spec.workload)
        if replay_spec.concurrency is not None:
            workload["concurrency"] = replay_spec.concurrency
        goal_payload = deepcopy(replay_spec.goal)
    return CandidateProvenance(
        model=str(identity_config.get("model_path", candidate_config.get("model_name", "unknown"))),
        hardware=str(identity_config.get("system", candidate_config.get("hardware_sku", "unknown"))),
        backend=(
            deployment.backend
            if deployment is not None
            else (str(candidate_config["backend"]) if candidate_config.get("backend") is not None else None)
        ),
        backend_version=(
            deployment.backend_version
            if deployment is not None
            else (
                str(candidate_config["backend_version"])
                if candidate_config.get("backend_version") is not None
                else None
            )
        ),
        performance_data=performance_data,
        topology=topology_fields,
        workload=workload,
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
) -> list[CandidateRecord]:
    """Apply the requested payload-retention policy without changing run counts."""

    if retention is CandidateRetention.ALL:
        return records
    if retention is CandidateRetention.FEASIBLE:
        return [record for record in records if record.status is CandidateStatus.FEASIBLE]
    selected = set(views.top_n) | set(views.pareto_front)
    return [record for record in records if record.candidate_id in selected]
