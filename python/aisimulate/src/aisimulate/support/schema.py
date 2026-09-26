# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared onboarding inputs; validity does not establish simulation readiness."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from packaging.version import Version
from pydantic import Field, SerializerFunctionWrapHandler, field_validator, model_serializer, model_validator

from aisimulate.config.common import PositiveFiniteFloat, PositiveStrictInt, StrictModel, load_yaml
from aisimulate.fpm_profile import FpmModelProfile

AGENTX_REFERENCE_CONTEXT = 256000


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
    # Historical field location retained for saved requests. This is a runtime
    # limit, independent of the optional synthetic validation workload.
    context_length: PositiveStrictInt = AGENTX_REFERENCE_CONTEXT
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


class CollectionSpec(StrictModel):
    """AISimulate runtime and capture limits for Dynamo's native grid generation."""

    max_num_tokens: PositiveStrictInt | None = None
    max_batch_size: PositiveStrictInt | None = None
    # Missing policy preserves saved requests' explicit-2048 collector behavior.
    # Fresh onboarding sets runtime explicitly at the CLI input boundary.
    prefill_cudagraph_policy: Literal["runtime", "explicit"] = "explicit"
    max_prefill_cudagraph_size: PositiveStrictInt | None = None
    gpu_memory_utilization: float | None = Field(default=None, strict=True, gt=0, le=1, allow_inf_nan=False)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        values = handler(self)
        # Preserve old request hashes and generated-plan verification. An
        # explicitly reviewed policy is always serialized, including explicit.
        if "prefill_cudagraph_policy" not in self.model_fields_set:
            values.pop("prefill_cudagraph_policy", None)
        return values

    @model_validator(mode="after")
    def _capture_policy(self) -> CollectionSpec:
        if self.prefill_cudagraph_policy == "runtime" and self.max_prefill_cudagraph_size is not None:
            raise ValueError("runtime prefill_cudagraph_policy rejects max_prefill_cudagraph_size; use explicit")
        return self

    @property
    def memory_fraction(self) -> float:
        """vLLM's fraction of total GPU memory, shared with simulation admission."""
        return self.gpu_memory_utilization if self.gpu_memory_utilization is not None else 0.9

    @field_validator("max_num_tokens")
    @classmethod
    def _collector_token_minimum(cls, value: int | None) -> int | None:
        if value is not None and value < 2:
            raise ValueError("FPM collection requires max_num_tokens >= 2")
        return value


_DNS_LABEL = r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?"
_DNS_SUBDOMAIN = rf"{_DNS_LABEL}(?:\.{_DNS_LABEL})*"


class FPMDeployment(StrictModel):
    """Deployment-only inputs included in the collector's frozen-plan identity."""

    executor: Literal["kubernetes", "slurm"] = "kubernetes"
    dynamo_version: str | None = None
    image: str | None = Field(default=None, pattern=r"^[^\s\x00]+$")
    container_mount: list[str] = Field(default_factory=list)
    cpus_per_task: int | None = Field(default=None, strict=True, gt=0)
    cpu_bind: Literal["cores", "none"] | None = None
    namespace: str | None = Field(default=None, pattern=rf"^{_DNS_LABEL}$")
    model_cache: str | None = None
    transport: Literal["nvlink", "ib", "efa"] | None = None
    image_pull_secret: str | None = Field(default=None, pattern=rf"^{_DNS_SUBDOMAIN}$", max_length=253)

    @model_validator(mode="after")
    def _executor_options(self) -> FPMDeployment:
        if self.executor == "slurm":
            incompatible = [
                "--" + name.replace("_", "-")
                for name in ("namespace", "model_cache", "image_pull_secret")
                if getattr(self, name) is not None
            ]
            if incompatible:
                raise ValueError("--executor slurm rejects Kubernetes options: " + ", ".join(incompatible))
            if self.image is None:
                raise ValueError("--executor slurm requires --image for the Pyxis container")
        elif self.container_mount:
            raise ValueError("--container-mount requires --executor slurm; use --model-cache for Kubernetes")
        elif self.cpus_per_task is not None or self.cpu_bind is not None:
            raise ValueError("--cpus-per-task and --cpu-bind require --executor slurm")
        return self

    @field_validator("container_mount")
    @classmethod
    def _container_mounts(cls, values: list[str]) -> list[str]:
        if any(
            not value.strip() or any(ord(char) < 32 or ord(char) == 127 or char == "," for char in value)
            for value in values
        ):
            raise ValueError("container mounts must be nonempty and contain no control characters or commas")
        return values

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
    schema_version: Literal["aisimulate-support-request/v2"] = "aisimulate-support-request/v2"
    identity: SupportIdentity
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    search: SearchProfile = Field(default_factory=SearchProfile)
    collection: CollectionSpec = Field(default_factory=CollectionSpec)
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
            if values.get("schema_version") == "aisimulate-support-request/v1":
                search = values.get("search", {})
                values = {**values, "schema_version": "aisimulate-support-request/v2"}
                if isinstance(search, Mapping):
                    search = dict(search)
                    search.setdefault("context_length", 16384)
                    values["search"] = search
            # Preserve explicit legacy limits and profile resource bounds. Do
            # not infer collection limits from a saved validation workload.
            profile = values.get("fpm_profile")
            search = values.get("search", {})
            if profile is not None and isinstance(search, Mapping) and "context_length" not in search:
                context = (
                    profile.get("context_length")
                    if isinstance(profile, Mapping)
                    else getattr(profile, "context_length", None)
                )
                if type(context) is int and context > 0:
                    values = {**values, "search": {**search, "context_length": min(context, AGENTX_REFERENCE_CONTEXT)}}
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
        limits = {
            "max_batched_tokens": self.collection.max_num_tokens
            or (deployment.resources.max_num_tokens if deployment is not None else 8192),
            "max_sequences": self.collection.max_batch_size
            or (deployment.resources.max_batch_size if deployment is not None else 256),
        }
        if limits["max_batched_tokens"] < limits["max_sequences"]:
            raise ValueError(
                f"resolved collection max_num_tokens ({limits['max_batched_tokens']}) must be at least "
                f"max_batch_size ({limits['max_sequences']}); edit the collection or profile bounds"
            )
        return limits

    def collection_settings(self) -> dict[str, Any]:
        """Resolved runtime inputs and the source of each proposed bound."""
        scheduler = self.scheduler_limits()
        profile = self.fpm_profile is not None
        settings: dict[str, Any] = {
            "context_length": self.search.context_length,
            **scheduler,
            "max_prefill_cudagraph_size": self.collection.max_prefill_cudagraph_size or 2048,
            "sources": {
                "context_length": (
                    "reviewed runtime limit; model/profile context capped at the 256000-token AgentX reference "
                    "unless explicitly overridden"
                ),
                "max_batched_tokens": "user collection override"
                if self.collection.max_num_tokens is not None
                else "reviewed profile resource bound"
                if profile
                else "initial vLLM collection policy (8192); review for the target runtime",
                "max_sequences": "user collection override"
                if self.collection.max_batch_size is not None
                else "reviewed profile resource bound"
                if profile
                else "initial vLLM collection policy (256); independent of validation concurrency",
                "max_prefill_cudagraph_size": "user collection override"
                if self.collection.max_prefill_cudagraph_size is not None
                else "collector default (2048); review against the target runtime CUDA graph configuration",
            },
        }
        if "prefill_cudagraph_policy" in self.collection.model_fields_set:
            policy = self.collection.prefill_cudagraph_policy
            settings["prefill_cudagraph_policy"] = policy
            settings["sources"]["prefill_cudagraph_policy"] = (
                "reviewed runtime selection; CUDA graph sizes resolve in the initialized vLLM engine"
                if policy == "runtime"
                else "reviewed explicit capture override; match the intended serving configuration"
            )
            if policy == "runtime":
                settings["max_prefill_cudagraph_size"] = None
                settings["sources"]["max_prefill_cudagraph_size"] = (
                    "deferred to the pinned runtime; onboarding does not prescribe CUDA graph sizes"
                )
        if self.collection.gpu_memory_utilization is not None:
            settings["gpu_memory_utilization"] = self.collection.gpu_memory_utilization
            settings["sources"]["gpu_memory_utilization"] = (
                "reviewed fraction of total GPU memory; initial policy is 0.90; runtime fit remains unverified"
            )
        deployment = self.profile_deployment()
        runtime = deployment.resources.runtime_memory if deployment is not None else None
        if runtime is not None:
            settings["sources"]["context_length"] = (
                f"reviewed context within the observed runtime memory limit of {runtime.max_model_len} tokens"
            )
            settings["sources"]["gpu_memory_utilization"] = (
                f"recorded runtime memory fraction {runtime.gpu_memory_utilization}; reused without scaling capacity"
            )
        return settings

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
            deployment = self.profile_deployment()
            scheduler = self.scheduler_limits()
            deployment.resources.validate_envelope(
                max_num_tokens=scheduler["max_batched_tokens"], max_batch_size=scheduler["max_sequences"]
            )
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
