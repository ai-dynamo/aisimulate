# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned contracts for self-service model/GPU support onboarding."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from aisimulate.config.common import StrictModel

_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


class SupportIdentity(StrictModel):
    """Exact identity of one requested support cell."""

    model: str
    model_revision: str
    tokenizer_revision: str
    chat_template_revision: str
    model_kind: Literal["auto", "dense", "moe"] = "auto"
    framework: Literal["vllm", "sglang", "trtllm"] = "vllm"
    framework_version: str
    gpu: str
    gpu_count: int = Field(strict=True, gt=0)
    node_count: int = Field(default=1, strict=True, gt=0)
    gpus_per_node: int = Field(strict=True, gt=0)
    interconnect: str
    sm: int | None = Field(default=None, strict=True, gt=0)
    engine_profile: str = "default"
    serving_mode: Literal["aggregated"] = "aggregated"
    aisimulate_revision: str

    @field_validator(
        "model",
        "model_revision",
        "tokenizer_revision",
        "chat_template_revision",
        "framework_version",
        "gpu",
        "interconnect",
        "engine_profile",
        "aisimulate_revision",
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identity values must be nonempty")
        return normalized

    @field_validator(
        "model_revision",
        "tokenizer_revision",
        "chat_template_revision",
        "aisimulate_revision",
    )
    @classmethod
    def _immutable_revision(cls, value: str) -> str:
        if value.lower() in {"main", "master", "latest", "head"}:
            raise ValueError(f"identity revision {value!r} is mutable")
        return value

    @model_validator(mode="after")
    def _validate_topology(self) -> SupportIdentity:
        if self.node_count * self.gpus_per_node != self.gpu_count:
            raise ValueError("node_count * gpus_per_node must equal gpu_count for an exact support cell")
        return self


class SloSpec(StrictModel):
    ttft_ms: float = Field(gt=0, allow_inf_nan=False)
    tpot_ms: float = Field(gt=0, allow_inf_nan=False)


class WorkloadSpec(StrictModel):
    """One versioned workload used by recommendation and E2E validation."""

    id: str
    kind: Literal["synthetic", "trace"]
    input_tokens: int | None = Field(default=None, strict=True, gt=0)
    output_tokens: int | None = Field(default=None, strict=True, gt=0)
    trace_path: str | None = None
    trace_digest: str | None = None
    trace_format: (
        Literal[
            "mooncake",
            "mooncake-delta",
            "agentic_mooncake",
            "applied_compute_agentic",
            "dynamo",
        ]
        | None
    ) = None
    concurrency: int = Field(default=10, strict=True, gt=0)
    request_count: int = Field(default=100, strict=True, gt=0)
    cache_state: Literal["cold", "warm", "mixed"] = "cold"
    slo: SloSpec

    @field_validator("id")
    @classmethod
    def _nonempty_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("workload id must be nonempty")
        return normalized

    @field_validator("trace_digest")
    @classmethod
    def _valid_digest(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("trace_digest must be a 64-character SHA-256 digest")
        return value.lower().removeprefix("sha256:") if value else None

    @model_validator(mode="after")
    def _validate_shape(self) -> WorkloadSpec:
        if self.kind == "synthetic":
            if self.input_tokens is None or self.output_tokens is None:
                raise ValueError("synthetic workloads require input_tokens and output_tokens")
            if any(value is not None for value in (self.trace_path, self.trace_digest, self.trace_format)):
                raise ValueError("synthetic workloads reject trace fields")
        else:
            if not self.trace_path or not self.trace_digest or not self.trace_format:
                raise ValueError("trace workloads require trace_path, trace_digest, and trace_format")
            if self.input_tokens is not None or self.output_tokens is not None:
                raise ValueError("trace workloads reject fixed input/output token counts")
        return self


class SearchProfile(StrictModel):
    version: Literal["mvp-v1"] = "mvp-v1"
    max_candidates: int = Field(default=16, strict=True, ge=4, le=16)
    objective: Literal[
        "throughput",
        "throughput_per_gpu",
        "throughput_per_user",
        "goodput",
        "goodput_per_gpu",
        "ttft",
        "e2e_latency",
        "pareto",
    ] = "throughput"
    context_length: int = Field(default=16384, strict=True, gt=0)
    seed: int = Field(default=42, strict=True, ge=0)


class FPMProfile(StrictModel):
    required: bool = True
    backend: Literal["vllm"] = "vllm"
    parallel_preset: Literal["auto", "tp", "tep", "dep", "pure_tp"] = "auto"


class ValidationPolicy(StrictModel):
    version: Literal["mvp-v1"] = "mvp-v1"
    max_mape: float = Field(default=0.20, gt=0, lt=1, allow_inf_nan=False)
    min_e2e_pairs: Literal[8] = 8
    recommendation_uplift_min: float | None = Field(
        default=None,
        gt=1,
        allow_inf_nan=False,
    )


class ExecutionProfile(StrictModel):
    provider: Literal["brev"] = "brev"
    reuse_existing: Literal[True] = True
    instance: str | None = None


class SupportRequest(StrictModel):
    schema_version: Literal["aisimulate-support-request/v1"] = "aisimulate-support-request/v1"
    identity: SupportIdentity
    workloads: list[WorkloadSpec]
    search: SearchProfile = Field(default_factory=SearchProfile)
    fpm: FPMProfile = Field(default_factory=FPMProfile)
    validation: ValidationPolicy = Field(default_factory=ValidationPolicy)
    execution: ExecutionProfile = Field(default_factory=ExecutionProfile)

    @model_validator(mode="after")
    def _validate_mvp_workloads(self) -> SupportRequest:
        if len(self.workloads) != 2:
            raise ValueError("MVP requires exactly two workloads")
        ids = [workload.id for workload in self.workloads]
        if len(set(ids)) != len(ids):
            raise ValueError("workload ids must be unique")
        synthetic = [workload for workload in self.workloads if workload.kind == "synthetic"]
        traces = [workload for workload in self.workloads if workload.kind == "trace"]
        if len(synthetic) != 1 or len(traces) != 1:
            raise ValueError("MVP requires one synthetic workload and one versioned trace")
        fixed = synthetic[0]
        if (fixed.input_tokens, fixed.output_tokens) != (8192, 1024):
            raise ValueError("MVP synthetic workload must use fixed 8K input / 1K output")
        if self.search.context_length < 9216:
            raise ValueError("search.context_length must cover the fixed 8K/1K workload")
        return self


class TopologyCandidate(StrictModel):
    id: str
    replicas: int = Field(strict=True, gt=0)
    tensor: int = Field(strict=True, gt=0)
    pipeline: Literal[1, 2, 4, 8] = 1
    attention_data: int = Field(strict=True, gt=0)
    moe_tensor: int = Field(strict=True, gt=0)
    moe_expert: int = Field(strict=True, gt=0)
    total_gpus: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def _validate_gpu_count(self) -> TopologyCandidate:
        expected = self.replicas * self.tensor * self.attention_data * self.pipeline
        if self.total_gpus != expected:
            raise ValueError(f"total_gpus must equal replicas * tensor * attention_data * pipeline ({expected})")
        return self

    def as_parallelism_preset(self) -> dict[str, int]:
        return {
            "replicas": self.replicas,
            "tensor": self.tensor,
            "pipeline": self.pipeline,
            "attention_data": self.attention_data,
            "moe_tensor": self.moe_tensor,
            "moe_expert": self.moe_expert,
        }


class ExistingSupport(StrictModel):
    status: Literal["supported", "unsupported", "unknown"]
    exact_match: bool | None = None
    aggregated: bool | None = None
    disaggregated: bool | None = None
    detail: str | None = None


class EvidenceRecord(StrictModel):
    phase: Literal["e2e", "fpm_prefill", "fpm_decode"]
    workload_id: str | None = None
    config_role: Literal["baseline", "top1", "top2", "top3"]
    candidate_id: str
    metric: Literal["ttft_ms", "tpot_ms", "output_throughput_tok_s", "forward_pass_ms"]
    predicted: float = Field(gt=0, allow_inf_nan=False)
    measured: float = Field(gt=0, allow_inf_nan=False)
    gpu_count: int = Field(strict=True, gt=0)
    source_run_id: str
    slo_compliant: bool | None = None
    held_out: bool = False

    @model_validator(mode="after")
    def _validate_phase_fields(self) -> EvidenceRecord:
        if self.phase == "e2e":
            if self.workload_id is None:
                raise ValueError("E2E evidence requires workload_id")
            if self.metric == "forward_pass_ms":
                raise ValueError("E2E evidence rejects forward_pass_ms")
            if self.slo_compliant is None:
                raise ValueError("E2E evidence requires slo_compliant")
            if self.held_out:
                raise ValueError("E2E evidence is not an FPM holdout")
        else:
            if self.workload_id is not None:
                raise ValueError("FPM evidence is workload-independent")
            if self.metric != "forward_pass_ms":
                raise ValueError("FPM evidence requires forward_pass_ms")
            if not self.held_out:
                raise ValueError("FPM evidence must be held out from calibration")
            if self.slo_compliant is not None:
                raise ValueError("FPM evidence rejects slo_compliant")
        if not self.candidate_id or not self.source_run_id:
            raise ValueError("candidate_id and source_run_id must be nonempty")
        return self


class EvidenceBundle(StrictModel):
    schema_version: Literal["aisimulate-support-evidence/v1"] = "aisimulate-support-evidence/v1"
    support_cell_id: str
    records: list[EvidenceRecord]


class GateResult(StrictModel):
    status: Literal["pass", "failed", "blocked"]
    details: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ValidationResult(StrictModel):
    schema_version: Literal["aisimulate-support-validation/v1"] = "aisimulate-support-validation/v1"
    support_cell_id: str
    status: Literal["pass", "failed", "blocked"]
    gates: dict[str, GateResult]
    errors: list[str] = Field(default_factory=list)
