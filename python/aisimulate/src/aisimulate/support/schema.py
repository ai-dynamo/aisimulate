# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared onboarding inputs; validity does not establish simulation readiness."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from packaging.version import Version
from pydantic import Field, field_validator, model_validator

from aisimulate.config.common import PositiveFiniteFloat, PositiveStrictInt, StrictModel, load_yaml
from aisimulate.fpm_profile import FpmModelProfile


class SupportIdentity(StrictModel):
    model: str
    model_revision: str
    model_kind: Literal["dense", "moe"]
    framework: Literal["vllm"] = "vllm"
    framework_version: str
    gpu: str
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
    attention_data_parallel: PositiveStrictInt | None = None
    moe_tensor_parallel: PositiveStrictInt | None = None
    moe_expert_parallel: PositiveStrictInt | None = None
    context_length: PositiveStrictInt = 16384
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
    fpm_profile: FpmModelProfile | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_allocation(cls, values: Any) -> Any:
        if isinstance(values, Mapping):
            obsolete = []
            for section, fields in (
                ("identity", ("gpu_count", "node_count", "gpus_per_node")),
                ("search", ("max_candidates",)),
            ):
                fields_in_request = values.get(section)
                if isinstance(fields_in_request, Mapping):
                    obsolete.extend(f"{section}.{name}" for name in fields if name in fields_in_request)
            if obsolete:
                raise ValueError(
                    "Legacy onboarding fields are no longer supported: " + ", ".join(obsolete) + ". "
                    "Copy the request, remove these fields, and regenerate the plan in a new output directory. "
                    "Set deployment replicas and GPU budgets in ordinary predict/recommend configs."
                )
        return values

    def parallelism(self) -> dict[str, int]:
        search = self.search
        return {
            "replicas": 1,
            "tensor": search.tensor_parallel,
            "pipeline": 1,
            "attention_data": search.attention_data_parallel or 1,
            "moe_tensor": search.moe_tensor_parallel
            or (search.tensor_parallel if self.identity.model_kind == "moe" else 1),
            "moe_expert": search.moe_expert_parallel or 1,
        }

    @property
    def worker_gpus(self) -> int:
        """Minimum GPUs required for the selected collection worker, not available capacity."""
        parallel = self.parallelism()
        return parallel["tensor"] * parallel["attention_data"]

    @property
    def parallel_preset(self) -> str:
        parallel = self.parallelism()
        if self.identity.model_kind == "dense":
            return "tp"
        if parallel["attention_data"] > 1:
            return "dep"
        return "tep" if parallel["moe_expert"] > 1 else "pure_tp"

    def profile_deployment(self):
        if self.fpm_profile is None:
            return None
        parallel = self.parallelism()
        return self.fpm_profile.select(
            model=self.identity.model,
            system=self.identity.gpu,
            backend=self.identity.framework,
            backend_version=self.identity.framework_version,
            tp_size=parallel["tensor"],
            pp_size=parallel["pipeline"],
            attention_dp_size=parallel["attention_data"],
            moe_tp_size=parallel["moe_tensor"],
            moe_ep_size=parallel["moe_expert"],
        )

    def scheduler_limits(self) -> dict[str, int]:
        """Rank-local limits shared by generated configs and collection bounds."""
        deployment = self.profile_deployment()
        if deployment is None:
            return {"max_batched_tokens": 8192, "max_sequences": 256}
        return {
            "max_batched_tokens": min(8192, deployment.resources.max_num_tokens),
            "max_sequences": min(self.workload.concurrency, deployment.resources.max_batch_size),
        }

    @model_validator(mode="after")
    def _shape(self) -> SupportRequest:
        parallel = self.parallelism()
        tp, dp, mtp, ep = (parallel[name] for name in ("tensor", "attention_data", "moe_tensor", "moe_expert"))
        if self.identity.model_kind == "dense":
            valid = dp == mtp == ep == 1
        else:
            valid = (
                (dp == 1 and mtp == tp and ep == 1) or (tp == mtp == 1 and ep == dp) or (dp == mtp == 1 and ep == tp)
            )
        if not valid:
            raise ValueError("onboarding requires a complete TP, DEP, or TEP topology; set the attention and MoE sizes")
        if self.workload.input_tokens + self.workload.output_tokens > self.search.context_length:
            raise ValueError("search.context_length must cover the input and output tokens")
        if self.fpm_profile is not None:
            self.profile_deployment()
            if self.identity.model_revision != self.fpm_profile.model_revision:
                raise ValueError("identity.model_revision must match fpm_profile.model_revision")
            if (self.fpm_profile.num_experts > 0) != (self.identity.model_kind == "moe"):
                raise ValueError("identity.model_kind must match fpm_profile.num_experts")
            if self.search.context_length > self.fpm_profile.context_length:
                raise ValueError("search.context_length exceeds fpm_profile.context_length")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SupportRequest:
        return cls.model_validate(load_yaml(path))
