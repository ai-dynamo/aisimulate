# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared onboarding inputs; validity does not establish simulation readiness."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Literal

from packaging.version import Version
from pydantic import Field, field_validator, model_validator

from aisimulate.config.common import PositiveFiniteFloat, PositiveStrictInt, StrictModel, load_yaml


class SupportIdentity(StrictModel):
    model: str
    model_revision: str
    model_kind: Literal["dense", "moe"]
    framework: Literal["vllm"] = "vllm"
    framework_version: str
    gpu: str
    gpu_count: PositiveStrictInt
    node_count: PositiveStrictInt = 1
    gpus_per_node: PositiveStrictInt
    interconnect: str
    sm: PositiveStrictInt | None = None
    tokenizer_revision: str | None = None
    chat_template_revision: str | None = None
    aisimulate_revision: str | None = None

    @field_validator(
        "model",
        "model_revision",
        "framework_version",
        "gpu",
        "interconnect",
        "tokenizer_revision",
        "chat_template_revision",
        "aisimulate_revision",
    )
    @classmethod
    def _nonempty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("identity values must be nonempty and contain no NUL characters")
        return value

    @field_validator(
        "model_revision", "framework_version", "tokenizer_revision", "chat_template_revision", "aisimulate_revision"
    )
    @classmethod
    def _pinned_revision(cls, value: str | None) -> str | None:
        if value is not None and value.lower() in {"main", "master", "latest", "head"}:
            raise ValueError("declare a pinned revision or version rather than a mutable default")
        return value

    @field_validator("gpu")
    @classmethod
    def _system_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError("gpu must be a packaged system name, not a path")
        return value

    @model_validator(mode="after")
    def _allocation(self) -> SupportIdentity:
        if self.node_count * self.gpus_per_node != self.gpu_count:
            raise ValueError("node_count * gpus_per_node must equal gpu_count")
        return self


class SloSpec(StrictModel):
    ttft_ms: PositiveFiniteFloat = 1000.0
    tpot_ms: PositiveFiniteFloat = 100.0


class WorkloadSpec(StrictModel):
    """One fixed synthetic shape; request_count is not a timing sample count."""

    input_tokens: PositiveStrictInt = 1024
    output_tokens: PositiveStrictInt = 128
    concurrency: PositiveStrictInt = 1
    request_count: PositiveStrictInt = 4
    slo: SloSpec = Field(default_factory=SloSpec)

    @model_validator(mode="after")
    def _request_count(self) -> WorkloadSpec:
        if self.request_count < self.concurrency:
            raise ValueError("request_count must be at least concurrency")
        return self


class SearchProfile(StrictModel):
    tensor_parallel: PositiveStrictInt = 1
    context_length: PositiveStrictInt = 16384
    max_candidates: int = Field(default=1, strict=True, ge=1, le=2)
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
    seed: int = Field(default=42, strict=True, ge=0)


_DNS_LABEL = r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?"
_DNS_SUBDOMAIN = rf"{_DNS_LABEL}(?:\.{_DNS_LABEL})*"


class FPMDeployment(StrictModel):
    """Deployment-only inputs included in the collector's frozen-plan identity."""

    dynamo_version: str | None = None
    image: str | None = Field(default=None, pattern=r"^[^\s\x00]+$")
    namespace: str | None = Field(default=None, pattern=rf"^{_DNS_LABEL}$")
    model_cache: str | None = None
    transport: Literal["nvlink", "ib", "efa"] | None = None
    image_pull_secret: str | None = Field(default=None, pattern=rf"^{_DNS_SUBDOMAIN}$", max_length=253)

    @field_validator("dynamo_version")
    @classmethod
    def _release_version(cls, value: str | None) -> str | None:
        if value is not None:
            Version(value)
        return value

    @field_validator("model_cache")
    @classmethod
    def _model_cache(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parts = value.split(":")
        if len(parts) > 3 or len(parts[0]) > 253 or not re.fullmatch(_DNS_SUBDOMAIN, parts[0]):
            raise ValueError("model_cache must be NAME[:MOUNT[:SUBPATH]] with a valid PVC name")
        if len(parts) > 1 and parts[1] and not PurePosixPath(parts[1]).is_absolute():
            raise ValueError("model_cache MOUNT must be an absolute container path")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("model_cache must contain no NUL characters or newlines")
        return value


class SupportRequest(StrictModel):
    schema_version: Literal["aisimulate-support-request/v1"] = "aisimulate-support-request/v1"
    identity: SupportIdentity
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    search: SearchProfile = Field(default_factory=SearchProfile)

    @model_validator(mode="after")
    def _shape(self) -> SupportRequest:
        if self.search.tensor_parallel > self.identity.gpus_per_node:
            raise ValueError("search.tensor_parallel must fit within gpus_per_node")
        if self.workload.input_tokens + self.workload.output_tokens > self.search.context_length:
            raise ValueError("search.context_length must cover the input and output tokens")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SupportRequest:
        return cls.model_validate(load_yaml(path))
