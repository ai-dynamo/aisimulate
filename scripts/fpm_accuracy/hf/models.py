# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Typed Hugging Face source evidence and in-memory measurement cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from fpm_accuracy.types.forward_pass import (
    ForwardPassInput,
    ForwardPassIteration,
    RequestMetrics,
    WorkloadKind,
)
from fpm_accuracy.types.worker_config import WorkerConfigRecord


class CaseStatus(StrEnum):
    READY = "ready"
    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    NO_MEASUREMENTS = "no_measurements"
    SUPPORTING_EVIDENCE_ONLY = "supporting_evidence_only"


class OrderingKind(StrEnum):
    CHRONOLOGICAL = "chronological"
    INTENTIONAL_SWEEP = "intentional_sweep"
    FILE_ORDER_FALLBACK = "file_order_fallback"


class MeasurementState(StrEnum):
    MEASUREMENT_UNAVAILABLE = "measurement_unavailable"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class MeasurementIssue:
    state: MeasurementState
    reason: str
    source_file_id: str
    source_path: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class MeasurementFile:
    measurement_file_id: str
    path: str
    sha256: str
    role: str
    local_path: Path
    provenance_url: str
    source_path: str | None = None
    source_sha256: str | None = None
    derived: bool = False
    representation: str | None = None
    iteration_count: int | None = None
    rank_record_count: int | None = None
    grouping: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class FpmArtifact:
    artifact_id: str
    path: str
    metadata_path: str
    sha256: str
    metadata_sha256: str
    role: str
    phases: tuple[str, ...]
    row_count: int
    local_path: Path
    local_metadata_path: Path
    provenance_url: str
    metadata_provenance_url: str


@dataclass(frozen=True, slots=True)
class MeasurementReference:
    artifact_id: str
    protocol_id: str | None
    evidence_format_id: str | None
    manifest_path: str
    manifest_sha256: str
    file_count: int
    provenance_url: str


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    configuration_id: str
    configuration_path: str
    snapshot_id: str
    snapshot_status: str
    manifest_path: str
    manifest_sha256: str
    manifest_provenance_url: str
    model_id: str
    model_revision: str | None
    system: str
    gpu_family: str
    framework: str
    framework_version: str
    parallelism: str
    parallel_strategy: str
    fpm_artifacts: tuple[FpmArtifact, ...]
    measurements: MeasurementReference
    worker_config_record: WorkerConfigRecord
    aisim_commit: str | None


@dataclass(frozen=True, slots=True)
class MeasurementObservation:
    """One measured point. The predictor-facing view has no latency field."""

    observation_id: str
    configuration_id: str
    order: int
    source_file_id: str
    source_path: str
    source_row: int
    iteration: ForwardPassIteration
    event_time: str | None = None

    @property
    def scheduled(self) -> RequestMetrics:
        return self.iteration.representative_rank.scheduled

    @property
    def queued(self) -> RequestMetrics:
        return self.iteration.representative_rank.queued

    @property
    def actual_ms(self) -> float:
        value = self.iteration.observed_time_ms
        if value is None:  # Measured observations are constructed with positive latency.
            raise RuntimeError(f"measurement {self.observation_id!r} has no positive latency")
        return value

    @property
    def workload_kind(self) -> WorkloadKind:
        return self.iteration.workload_kind

    def prediction_input(self) -> ForwardPassInput:
        return self.iteration.prediction_input()

    def tuning_payload(self) -> list[dict[str, object]]:
        return self.iteration.tuning_payload()


@dataclass(frozen=True, slots=True)
class MeasurementCase:
    case_id: str
    measurement_membership_sha256: str
    configuration: ConfigurationSnapshot
    protocol_id: str | None
    status: CaseStatus
    observations: tuple[MeasurementObservation, ...]
    truth_files: tuple[MeasurementFile, ...]
    helper_files: tuple[MeasurementFile, ...]
    fpm_artifacts: tuple[FpmArtifact, ...]
    worker_role: Literal["prefill", "decode", "aggregated"]
    ordering: OrderingKind
    override_applied: bool
    override_sha256: str | None
    override_effects: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    issues: tuple[MeasurementIssue, ...] = ()
    parser_policy_id: str | None = None

    @property
    def configuration_id(self) -> str:
        return self.configuration.configuration_id

    @property
    def provenance_urls(self) -> tuple[str, ...]:
        values = [
            self.configuration.manifest_provenance_url,
            self.configuration.measurements.provenance_url,
            *(file.provenance_url for file in self.truth_files),
            *(file.provenance_url for file in self.helper_files),
            *(artifact.provenance_url for artifact in self.fpm_artifacts),
            *(artifact.metadata_provenance_url for artifact in self.fpm_artifacts),
        ]
        return tuple(dict.fromkeys(values))
