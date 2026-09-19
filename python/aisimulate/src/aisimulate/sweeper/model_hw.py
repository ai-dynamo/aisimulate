# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a (model, hardware SKU) into the facts the parallel enumeration needs,
and bound the parallel configs to shapes that actually hold the model.

It directly reuses AIConfigurator:

- ``check_is_moe``                  -> is_moe
- ``_estimate_model_weight_bytes``  -> model weight size (-> wideEP heuristic)
- ``load_system_spec``              -> the SKU's VRAM / GPUs-per-node

Validity is **KV-cache based**: :func:`parallel_configs_for` enumerates shapes
from 1 GPU/worker and keeps a shape iff its estimated KV capacity exceeds the
workload's ``max_seq_len`` (:mod:`aisimulate.sweeper.kv_estimate`). That is per-shape (TEP /
DEP / TP differ at the same GPU count) and uses the real quantized weights — it
replaces the old BF16 min-GPU weight floor entirely.

With an explicit FPM profile, enumeration uses only its declared deployment
tuples and rank-local resource bounds. It does not resolve an analytical model
or infer another topology's resources from its GPU count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aisimulate.generator.naive import _estimate_model_weight_bytes
from aisimulate_core.sdk import perf_database
from aisimulate_core.sdk.models import check_is_moe
from aisimulate_core.sdk.utils import get_model_config_from_model_path

from .kv_estimate import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_NUM_TOKENS,
    DEFAULT_MEMORY_FRACTION,
    feasible_shape_tokens,
)
from .parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
    enumerate_disagg_configs,
    enumerate_parallel_configs,
)

# GQA+MoE architectures. Pure expert-TP is no longer gated on this list; it is
# scanned for every MoE model and then filtered by backend/KV feasibility.
_GQA_MOE_ARCHITECTURES = frozenset({"Qwen3MoeForCausalLM"})


class NoViableParallelConfig(ValueError):
    """No parallel config can hold the model+sequence within the GPU budget."""


@dataclass(frozen=True)
class ModelHardware:
    """Per-(model, hardware, backend) facts that bound the parallel search."""

    model_name: str
    hardware_sku: str
    backend: str
    is_moe: bool
    mla: bool  # non-GQA MoE marker retained for reporting/model facts
    enable_wideep: bool
    weight_bytes: int
    vram_per_gpu: int
    gpus_per_node: int
    max_context: int | None  # model's max context length (the default max_seq_len)
    num_experts: int = 0


def resolve_model_hardware(
    model_name: str,
    hardware_sku: str,
    *,
    backend: str,
    systems_paths: list[str] | None = None,
) -> ModelHardware:
    """Read the model weights + SKU spec (via AIC) to derive is_moe / mla / wideep
    and the model's max context length."""
    model_config = get_model_config_from_model_path(model_name)
    is_moe = check_is_moe(model_name)
    architecture = model_config.get("architecture", "")
    allow_pure_tp = is_moe and architecture in _GQA_MOE_ARCHITECTURES
    mla = is_moe and not allow_pure_tp
    max_context = model_config.get("context")
    num_experts = int(model_config.get("num_experts") or model_config.get("n_routed_experts") or 0)

    system_spec = perf_database.load_system_spec(
        hardware_sku, **({"systems_paths": systems_paths} if systems_paths is not None else {})
    )
    if not system_spec:
        raise ValueError(
            f"unknown hardware_sku {hardware_sku!r}: no system config found on AIConfigurator Core's systems path"
        )
    vram_per_gpu = int(system_spec["gpu"]["mem_capacity"])
    gpus_per_node = int(system_spec["node"]["num_gpus_per_node"])
    weight_bytes = _estimate_model_weight_bytes(model_name)

    # Large MoE (a node can't hold ~2x the weights) auto-enables multi-node wideEP.
    enable_wideep = is_moe and gpus_per_node * vram_per_gpu < 2 * weight_bytes

    return ModelHardware(
        model_name=model_name,
        hardware_sku=hardware_sku,
        backend=backend,
        is_moe=is_moe,
        mla=mla,
        enable_wideep=enable_wideep,
        weight_bytes=weight_bytes,
        vram_per_gpu=vram_per_gpu,
        gpus_per_node=gpus_per_node,
        max_context=int(max_context) if max_context else None,
        num_experts=num_experts,
    )


def parallel_configs_for(
    model_name: str,
    hardware_sku: str,
    *,
    gpu_budget: int,
    deployment_mode: str,
    backend: str,
    backend_version: str | None = None,
    max_seq_len: int | None = None,
    min_gpu_budget: int | None = None,
    max_num_tokens: int = DEFAULT_MAX_NUM_TOKENS,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    role_runtime: dict[str, tuple[int, int, float] | tuple[int, int, float, int | None]] | None = None,
    systems_paths: list[str] | None = None,
    fpm_profile: dict[str, Any] | None = None,
    model_controls: dict[str, str | int | bool] | None = None,
    nextn: int = 0,
    candidate_configs: list[ReplicaParallelConfig] | list[DisaggParallelConfig] | None = None,
) -> list[ReplicaParallelConfig] | list[DisaggParallelConfig]:
    """Resolve the model/hardware, then enumerate the parallel configs that fit
    the GPU budget and can hold a ``max_seq_len``-token sequence.

    Validity is **KV-cache based**: shapes are enumerated from 1 GPU/worker and a
    shape is kept iff its estimated KV capacity exceeds ``max_seq_len`` (the
    accurate, per-shape check; see :mod:`aisimulate.sweeper.kv_estimate`). ``max_num_tokens`` /
    ``max_batch_size`` / ``memory_fraction`` are the runtime knobs the estimate
    reserves around the KV budget.

    ``max_seq_len`` defaults to the model's max context length (the engine's
    ``max_model_len`` -> the longest sequence any request can occupy); pass a
    smaller value only to tune for a workload known to be shorter.

    ``deployment_mode`` is ``"agg"`` (-> ``list[ReplicaParallelConfig]``) or
    ``"disagg"`` (-> ``list[DisaggParallelConfig]``). Raises
    :class:`NoViableParallelConfig` when no shape can hold the sequence within the
    budget.
    """
    profile = None
    if fpm_profile is not None:
        from aisimulate_core.sdk.fpm_profile import load_fpm_profile

        profile = load_fpm_profile(fpm_profile)
        if profile.model != model_name:
            raise ValueError("FPM profile model identity does not match the requested model")
        seq_len = max_seq_len if max_seq_len is not None else profile.context_length
        if seq_len > profile.context_length:
            raise ValueError("max_seq_len exceeds FPM profile context_length")
        configs = (
            candidate_configs
            if candidate_configs is not None
            else _profile_parallel_configs(
                profile,
                hardware_sku=hardware_sku,
                backend=backend,
                backend_version=backend_version,
                gpu_budget=gpu_budget,
                min_gpu_budget=min_gpu_budget,
                deployment_mode=deployment_mode,
            )
        )
    else:
        mh = resolve_model_hardware(model_name, hardware_sku, backend=backend, systems_paths=systems_paths)
        seq_len = max_seq_len if max_seq_len is not None else mh.max_context
    if seq_len is None:
        raise ValueError(f"max_seq_len is required: {model_name} config exposes no max context length")

    # Enumerate from 1 GPU/worker; the KV estimate is the sole feasibility filter.
    # MoE tensor-parallel (moe_ep == 1) is enabled for every MoE model, MLA
    # included: real deployments (e.g. InferenceX GLM-5, reported as EP=1) run it,
    # so the search must be able to find it rather than have it filtered out here.
    if profile is None:
        common = dict(
            is_moe=mh.is_moe,
            backend=backend,
            gpu_budget=gpu_budget,
            min_gpu_budget=min_gpu_budget,
            enable_wideep=mh.enable_wideep,
            allow_moe_pure_tp=True,
        )
        if deployment_mode == "disagg":
            configs = enumerate_disagg_configs(**common)
        elif deployment_mode == "agg":
            configs = enumerate_parallel_configs(**common)
        else:
            raise ValueError(f"deployment_mode must be 'agg' or 'disagg', got {deployment_mode!r}")

    # KV-cache validity: keep configs whose every role-shape holds a max_seq_len sequence.
    def feasible_for(role: str, shapes):
        runtime = (role_runtime or {}).get(role, (max_num_tokens, max_batch_size, memory_fraction))
        if len(runtime) == 3:
            role_tokens, role_batch, role_memory = runtime
            fixed_tokens = None
        elif len(runtime) == 4:
            role_tokens, role_batch, role_memory, fixed_tokens = runtime
        else:
            raise ValueError(
                "role_runtime values must be (tokens, batch, memory) or (tokens, batch, memory, fixed_tokens)"
            )
        grouped_shapes = set()
        if profile is not None:
            for shape in dict.fromkeys(shapes):
                deployment = profile.select(
                    model=model_name,
                    system=hardware_sku,
                    backend=backend,
                    backend_version=backend_version,
                    tp_size=shape.tp,
                    pp_size=shape.pp,
                    attention_dp_size=shape.dp,
                    moe_tp_size=shape.moe_tp,
                    moe_ep_size=shape.moe_ep,
                )
                deployment.resources.validate_envelope(max_num_tokens=role_tokens, max_batch_size=role_batch)
                if deployment.resources.cache_layout == "grouped":
                    grouped_shapes.add(shape)
        grouped_feasible = set()
        if grouped_shapes:
            from .kv_estimate import estimate_grouped_cache_budget

            if deployment_mode != "agg" or nextn:
                raise ValueError("grouped FPM cache requires aggregated replay without speculation")
            if fixed_tokens is not None:
                raise ValueError("grouped FPM cache cannot use a fixed scalar token capacity")
            for shape in grouped_shapes:
                try:
                    budget = estimate_grouped_cache_budget(
                        shape,
                        model_name=model_name,
                        hardware_sku=hardware_sku,
                        backend=backend,
                        backend_version=backend_version,
                        systems_paths=systems_paths,
                        fpm_profile=fpm_profile,
                        context_length=seq_len,
                        max_num_tokens=role_tokens,
                        max_batch_size=role_batch,
                        memory_fraction=role_memory,
                        model_controls=model_controls,
                    )
                except ValueError as exc:
                    if "no KV budget" in str(exc):
                        continue
                    raise
                if budget["total_kv_size_bytes"] >= budget["request_peak_cache_bytes"]:
                    grouped_feasible.add(shape)
            shapes = [shape for shape in shapes if shape not in grouped_shapes]
        if fixed_tokens is not None:
            return {shape: fixed_tokens for shape in dict.fromkeys(shapes) if fixed_tokens > seq_len}
        linear_feasible = feasible_shape_tokens(
            shapes,
            model_name=model_name,
            hardware_sku=hardware_sku,
            backend=backend,
            backend_version=backend_version,
            systems_paths=systems_paths,
            max_seq_len=seq_len,
            max_num_tokens=role_tokens,
            max_batch_size=role_batch,
            memory_fraction=role_memory,
            **({"fpm_profile": fpm_profile} if fpm_profile is not None else {}),
            **({"model_controls": model_controls} if model_controls else {}),
            **({"nextn": nextn} if nextn else {}),
        )
        return set(linear_feasible) | grouped_feasible

    if deployment_mode == "agg":
        feasible = feasible_for("agg", [c.shape for c in configs])
        kept = [c for c in configs if c.shape in feasible]
    else:
        prefill_feasible = feasible_for("prefill", [c.prefill.shape for c in configs])
        decode_feasible = feasible_for("decode", [c.decode.shape for c in configs])
        kept = [c for c in configs if c.prefill.shape in prefill_feasible and c.decode.shape in decode_feasible]
    if not kept:
        raise NoViableParallelConfig(
            f"{model_name} on {hardware_sku}: no parallel config holds a {seq_len}-token "
            f"sequence within {gpu_budget} GPUs ({backend} KV-cache estimate)"
        )
    return kept


def _profile_parallel_configs(
    profile,
    *,
    hardware_sku: str,
    backend: str,
    backend_version: str | None,
    gpu_budget: int,
    min_gpu_budget: int | None,
    deployment_mode: str,
    decode_hardware_sku: str | None = None,
):
    """Enumerate only declared deployment shapes; replicas do not change a cell."""

    def replicas_for(system):
        deployments = [
            deployment
            for deployment in profile.deployments
            if deployment.system == system
            and deployment.backend == backend
            and deployment.backend_version == backend_version
        ]
        if not deployments:
            raise ValueError(f"no FPM deployment profile for {system}/{backend}/{backend_version}")
        replicas = []
        for deployment in deployments:
            shape = ParallelShape(
                tp=deployment.tp,
                pp=deployment.pp,
                dp=deployment.dp,
                moe_tp=deployment.moe_tp,
                moe_ep=deployment.moe_ep,
            )
            replicas.extend(
                ReplicaParallelConfig(shape=shape, replicas=count)
                for count in range(1, gpu_budget // shape.gpus_per_worker + 1)
            )
        return replicas

    replicas = replicas_for(hardware_sku)
    if deployment_mode == "agg":
        configs = replicas
    elif deployment_mode == "disagg":
        decode_replicas = replicas_for(decode_hardware_sku) if decode_hardware_sku else replicas
        configs = [
            DisaggParallelConfig(prefill=prefill, decode=decode)
            for prefill in replicas
            for decode in decode_replicas
            if prefill.total_gpus + decode.total_gpus <= gpu_budget
        ]
    else:
        raise ValueError("FPM profiles support only agg or disagg deployment modes")
    return [config for config in configs if min_gpu_budget is None or config.total_gpus >= min_gpu_budget]
