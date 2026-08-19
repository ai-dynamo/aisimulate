# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve legacy engine/request controls before the search reaches replay."""

from __future__ import annotations

from dataclasses import dataclass

from aiconfigurator_core.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)
from aiconfigurator_core.sdk.models import get_model_family

from .config import SmartSearchConfig, Workload
from .estimator import resolve_systems_paths
from .heterogeneous import (
    DisaggBackendPair,
    DisaggRole,
    RoleEstimatorSpecs,
    RoleFailureCategory,
    RoleSearchError,
)
from .kv_estimate import memory_fraction_kind
from .model_hw import resolve_model_hardware
from .replay import EngineRequestSpec, RoleEngineRequestSpec

_DEEPSEEK_V4_MEGAMOE_MODELS = frozenset(
    {
        "deepseek-ai/DeepSeek-V4-Pro",
        "sgl-project/DeepSeek-V4-Pro-FP8",
    }
)


@dataclass(frozen=True)
class EngineControlTemplate:
    """Backend-resolved controls which do not depend on a sampled shape."""

    backend: str
    max_seq_len: int
    model_family: str
    is_moe: bool
    memory_fraction_kind: str


def _validate_quant_mode(name: str, value: str | None, enum_type: type) -> None:
    if value is not None and value not in enum_type.__members__:
        allowed = ", ".join(enum_type.__members__)
        raise ValueError(f"{name} has unsupported value {value!r}; allowed: {allowed}")


def _required_workload_tokens(workload: Workload) -> int | None:
    if workload.is_trace_based:
        return None
    assert workload.isl is not None and workload.osl is not None
    return workload.isl + workload.osl


def resolve_engine_controls(
    config: SmartSearchConfig,
) -> dict[str, EngineControlTemplate]:
    """Validate selected model/backend combinations and resolve sequence capacity.

    This runs before provider preparation and before any replay work. Controls that
    cannot affect the selected model/backend are rejected instead of being silently
    accepted and lost during materialization.
    """

    ss = config.search_space
    _validate_quant_mode("gemm_quant_mode", ss.gemm_quant_mode, GEMMQuantMode)
    _validate_quant_mode("moe_quant_mode", ss.moe_quant_mode, MoEQuantMode)
    _validate_quant_mode("kvcache_quant_mode", ss.kvcache_quant_mode, KVCacheQuantMode)
    _validate_quant_mode("fmha_quant_mode", ss.fmha_quant_mode, FMHAQuantMode)
    _validate_quant_mode("comm_quant_mode", ss.comm_quant_mode, CommQuantMode)

    model_family = get_model_family(ss.model_name)
    systems_paths = list(resolve_systems_paths(ss.systems_paths))
    required_tokens = _required_workload_tokens(config.workload)
    resolved: dict[str, EngineControlTemplate] = {}
    for backend in dict.fromkeys(ss.backend):
        model_hw = resolve_model_hardware(
            ss.model_name,
            ss.hardware_sku,
            backend=backend,
            systems_paths=systems_paths,
        )
        max_seq_len = ss.max_seq_len or ss.context_length or model_hw.max_context
        if max_seq_len is None:
            raise ValueError(
                f"max_seq_len is required because model {ss.model_name!r} exposes "
                "no maximum context length"
            )
        if required_tokens is not None and max_seq_len < required_tokens:
            raise ValueError(
                f"max_seq_len={max_seq_len} cannot hold the synthetic request "
                f"isl+osl={required_tokens}"
            )
        if model_hw.max_context is not None and max_seq_len > model_hw.max_context:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds model {ss.model_name!r} "
                f"maximum context {model_hw.max_context}"
            )

        moe_controls = (
            ss.enable_wideep,
            ss.enable_eplb,
            ss.wideep_num_slots is not None,
            ss.moe_backend is not None,
            ss.moe_quant_mode is not None,
        )
        if any(moe_controls) and not model_hw.is_moe:
            raise ValueError(
                f"MoE controls require an MoE model; {ss.model_name!r} is dense"
            )
        if ss.attention_backend is not None:
            if not model_hw.mla:
                raise ValueError(
                    "attention_backend is only supported for MLA models; "
                    f"{ss.model_name!r} is not MLA"
                )
            if backend != "sglang":
                raise ValueError(
                    "attention_backend is supported only with backend='sglang'; "
                    f"got {backend!r}"
                )
        if ss.moe_backend is not None and backend != "sglang":
            raise ValueError(
                f"moe_backend={ss.moe_backend!r} is supported only with "
                f"backend='sglang'; got {backend!r}"
            )
        if ss.moe_backend == "megamoe":
            if model_family != "DEEPSEEKV4":
                raise ValueError(
                    "moe_backend='megamoe' is supported only for DeepSeek-V4 models"
                )
            if ss.model_name not in _DEEPSEEK_V4_MEGAMOE_MODELS:
                raise ValueError(
                    "moe_backend='megamoe' has packaged performance data only for "
                    f"DeepSeek-V4-Pro; got {ss.model_name!r}"
                )
            if model_hw.sm_version < 100:
                raise ValueError(
                    "moe_backend='megamoe' requires a Blackwell-class system; "
                    f"got {ss.hardware_sku!r}"
                )

        resolved[backend] = EngineControlTemplate(
            backend=backend,
            max_seq_len=max_seq_len,
            model_family=model_family,
            is_moe=model_hw.is_moe,
            memory_fraction_kind=memory_fraction_kind(backend),
        )
    return resolved


def materialize_engine_request(
    template: EngineControlTemplate,
    *,
    config: SmartSearchConfig,
    sample: dict[str, object],
) -> EngineRequestSpec:
    """Bind backend-resolved controls to one candidate's active roles."""

    ss = config.search_space
    roles = (
        ("agg",)
        if sample["deployment_mode"] == "agg"
        else ("prefill", "decode")
    )
    memory_by_role = {
        role: float(
            ss.free_gpu_memory_fraction
            if ss.free_gpu_memory_fraction is not None
            else sample[f"{role}_gpu_memory_utilization"]
        )
        for role in roles
    }
    return EngineRequestSpec(
        cached_prefix_tokens=config.workload.cached_prefix_tokens,
        context_tokens={
            role: int(sample[f"{role}_max_num_batched_tokens"]) for role in roles
        },
        enable_chunked_prefill=ss.enable_chunked_prefill,
        enable_wideep=ss.enable_wideep,
        enable_eplb=ss.enable_eplb,
        wideep_num_slots=ss.wideep_num_slots,
        moe_backend=ss.moe_backend,
        attention_backend=ss.attention_backend,
        gemm_quant_mode=ss.gemm_quant_mode,
        moe_quant_mode=ss.moe_quant_mode,
        kvcache_quant_mode=ss.kvcache_quant_mode,
        fmha_quant_mode=ss.fmha_quant_mode,
        comm_quant_mode=ss.comm_quant_mode,
        nextn=ss.aic_nextn or 0,
        nextn_accepted=ss.nextn_accepted,
        memory_fraction_kind=template.memory_fraction_kind,
        memory_fraction_by_role=memory_by_role,
        max_seq_len=template.max_seq_len,
        model_family=template.model_family,
    )


_ROLE_ENGINE_FIELDS = (
    "model_name",
    "hardware_sku",
    "max_seq_len",
    "enable_chunked_prefill",
    "enable_wideep",
    "enable_eplb",
    "wideep_num_slots",
    "moe_backend",
    "attention_backend",
    "gemm_quant_mode",
    "moe_quant_mode",
    "kvcache_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "free_gpu_memory_fraction",
    "systems_paths",
)


def _role_config(
    config: SmartSearchConfig,
    role: DisaggRole,
    backend: str,
) -> SmartSearchConfig:
    ss = config.search_space
    updates = {name: ss.role_value(role.value, name) for name in _ROLE_ENGINE_FIELDS}
    updates["backend"] = [backend]
    role_space = ss.model_copy(update=updates)
    return config.model_copy(update={"search_space": role_space})


def resolve_role_engine_controls(
    config: SmartSearchConfig,
    estimator_specs: dict[str, RoleEstimatorSpecs],
) -> dict[str, dict[str, EngineControlTemplate]]:
    """Resolve engine controls independently for every searched P/D pair."""

    resolved: dict[str, dict[str, EngineControlTemplate]] = {}
    for pair_label, estimators in estimator_specs.items():
        role_templates: dict[str, EngineControlTemplate] = {}
        for role in (DisaggRole.PREFILL, DisaggRole.DECODE):
            estimator = estimators.estimator_for(role)
            try:
                role_templates[role.value] = resolve_engine_controls(
                    _role_config(config, role, estimator.backend)
                )[estimator.backend]
            except ValueError as exc:
                raise RoleSearchError(
                    role,
                    RoleFailureCategory.ENGINE_CONTROLS,
                    str(exc),
                    provenance={
                        "backend_pair": pair_label,
                        "model_name": estimator.model_path,
                        "hardware_sku": estimator.system,
                        "backend": estimator.backend,
                    },
                ) from exc
        resolved[pair_label] = role_templates
    return resolved


def materialize_role_engine_request(
    pair: DisaggBackendPair,
    templates: dict[str, EngineControlTemplate],
    estimators: RoleEstimatorSpecs,
    *,
    config: SmartSearchConfig,
    sample: dict[str, object],
) -> EngineRequestSpec:
    """Bind inherited role controls to one heterogeneous candidate."""

    ss = config.search_space
    role_requests: dict[str, RoleEngineRequestSpec] = {}
    for role in (DisaggRole.PREFILL, DisaggRole.DECODE):
        name = role.value
        template = templates[name]
        estimator = estimators.estimator_for(role)
        memory_fraction = ss.role_value(name, "free_gpu_memory_fraction")
        role_requests[name] = RoleEngineRequestSpec(
            role=name,
            backend=pair.backend_for(role),
            backend_version=estimator.backend_version,
            cached_prefix_tokens=config.workload.cached_prefix_tokens,
            context_tokens=int(sample[f"{name}_max_num_batched_tokens"]),
            enable_chunked_prefill=(
                role is DisaggRole.PREFILL
                and bool(ss.role_value(name, "enable_chunked_prefill"))
            ),
            enable_wideep=bool(ss.role_value(name, "enable_wideep")),
            enable_eplb=bool(ss.role_value(name, "enable_eplb")),
            wideep_num_slots=ss.role_value(name, "wideep_num_slots"),
            moe_backend=ss.role_value(name, "moe_backend"),
            attention_backend=ss.role_value(name, "attention_backend"),
            gemm_quant_mode=ss.role_value(name, "gemm_quant_mode"),
            moe_quant_mode=ss.role_value(name, "moe_quant_mode"),
            kvcache_quant_mode=ss.role_value(name, "kvcache_quant_mode"),
            fmha_quant_mode=ss.role_value(name, "fmha_quant_mode"),
            comm_quant_mode=ss.role_value(name, "comm_quant_mode"),
            nextn=ss.aic_nextn or 0,
            nextn_accepted=ss.nextn_accepted,
            memory_fraction_kind=template.memory_fraction_kind,
            memory_fraction=float(
                memory_fraction
                if memory_fraction is not None
                else sample[f"{name}_gpu_memory_utilization"]
            ),
            max_seq_len=template.max_seq_len,
            model_family=template.model_family,
        )
    return EngineRequestSpec(
        cached_prefix_tokens=config.workload.cached_prefix_tokens,
        context_tokens={
            role: request.context_tokens for role, request in role_requests.items()
        },
        nextn=ss.aic_nextn or 0,
        nextn_accepted=ss.nextn_accepted,
        role_requests=role_requests,
    )
