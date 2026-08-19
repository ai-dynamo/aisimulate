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
"""

from __future__ import annotations

from dataclasses import dataclass

from aiconfigurator.generator.naive import _estimate_model_weight_bytes
from aiconfigurator_core.sdk import perf_database
from aiconfigurator_core.sdk.models import check_is_moe
from aiconfigurator_core.sdk.models.base import _MODEL_REGISTRY
from aiconfigurator_core.sdk.models.helpers import _architecture_to_model_family
from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

from .kv_estimate import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_NUM_TOKENS,
    DEFAULT_MEMORY_FRACTION,
    feasible_shape_tokens,
)
from .parallel_enum import (
    DisaggParallelConfig,
    ReplicaParallelConfig,
    RoleParallelCandidates,
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
    model_family: str
    default_gpus_per_worker: tuple[int, ...]
    default_pp_candidates: tuple[int, ...]
    default_cp_candidates: tuple[int, ...]


def resolve_model_hardware(
    model_name: str, hardware_sku: str, *, backend: str
) -> ModelHardware:
    """Read the model weights + SKU spec (via AIC) to derive is_moe / mla / wideep
    and the model's max context length."""
    model_config = get_model_config_from_model_path(model_name)
    is_moe = check_is_moe(model_name)
    architecture = model_config.get("architecture", "")
    allow_pure_tp = is_moe and architecture in _GQA_MOE_ARCHITECTURES
    mla = is_moe and not allow_pure_tp
    max_context = model_config.get("context")
    model_family = _architecture_to_model_family(str(model_config.get("architecture", "")))

    system_spec = perf_database.load_system_spec(hardware_sku)
    if not system_spec:
        raise ValueError(
            f"unknown hardware_sku {hardware_sku!r}: no system config found on "
            "AIConfigurator Core's systems path"
        )
    vram_per_gpu = int(system_spec["gpu"]["mem_capacity"])
    gpus_per_node = int(system_spec["node"]["num_gpus_per_node"])
    weight_bytes = _estimate_model_weight_bytes(model_name)

    # Large MoE (a node can't hold ~2x the weights) auto-enables multi-node wideEP.
    enable_wideep = is_moe and gpus_per_node * vram_per_gpu < 2 * weight_bytes
    model_cls = _MODEL_REGISTRY.get(model_family)
    default_cp_candidates = (
        (1, 2, 4, 8)
        if model_cls is not None and model_cls.supports_cp(backend)
        else (1,)
    )
    architecture = str(model_config.get("architecture", ""))
    default_pp_candidates = (
        (1, 2)
        if architecture in {"DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM"}
        and hardware_sku in {"gb200", "gb300"}
        else (1,)
    )
    if is_moe and enable_wideep and backend in {"trtllm", "sglang"}:
        # Wide-EP augments the fused single-node ladder; it does not replace it.
        default_gpus_per_worker = (1, 2, 4, 8, 16, 32, 64)
    elif hardware_sku in {"gb200", "gb300"}:
        default_gpus_per_worker = (1, 2, 4, 8, 16)
    else:
        default_gpus_per_worker = (1, 2, 4, 8)

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
        model_family=model_family,
        default_gpus_per_worker=default_gpus_per_worker,
        default_pp_candidates=default_pp_candidates,
        default_cp_candidates=default_cp_candidates,
    )


def _resolved_role_candidates(
    configured: RoleParallelCandidates | None,
    facts: ModelHardware,
    *,
    role: str,
) -> RoleParallelCandidates:
    """Apply capability-derived PP/CP defaults without overriding explicit lists."""

    defaults = RoleParallelCandidates()
    if configured is not None:
        default_cp = facts.default_cp_candidates if role != "decode" else (1,)
        cp_candidates = (
            tuple(value for value in configured.cp if value in default_cp)
            if configured.cp
            else default_cp
        )
        return RoleParallelCandidates(
            gpus_per_worker=(
                configured.gpus_per_worker or facts.default_gpus_per_worker
            ),
            tp=configured.tp or defaults.tp,
            pp=configured.pp or facts.default_pp_candidates,
            attention_dp=configured.attention_dp or defaults.attention_dp,
            moe_tp=configured.moe_tp or defaults.moe_tp,
            moe_ep=configured.moe_ep or defaults.moe_ep,
            cp=cp_candidates,
            workers=configured.workers,
        )
    return RoleParallelCandidates(
        gpus_per_worker=facts.default_gpus_per_worker,
        tp=defaults.tp,
        pp=facts.default_pp_candidates,
        attention_dp=defaults.attention_dp,
        moe_tp=defaults.moe_tp,
        moe_ep=defaults.moe_ep,
        cp=facts.default_cp_candidates if role != "decode" else (1,),
    )


def parallel_configs_for(
    model_name: str,
    hardware_sku: str,
    *,
    gpu_budget: int,
    deployment_mode: str,
    backend: str,
    max_seq_len: int | None = None,
    min_gpu_budget: int | None = None,
    max_num_tokens: int = DEFAULT_MAX_NUM_TOKENS,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    agg_candidates: RoleParallelCandidates | None = None,
    prefill_candidates: RoleParallelCandidates | None = None,
    decode_candidates: RoleParallelCandidates | None = None,
    num_gpu_per_replica: tuple[int, ...] | None = (
        1,
        2,
        4,
        8,
        *range(16, 129, 8),
    ),
    max_gpu_per_replica: int | None = 128,
    max_prefill_workers: int | None = 32,
    max_decode_workers: int | None = 32,
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
    mh = resolve_model_hardware(model_name, hardware_sku, backend=backend)
    seq_len = max_seq_len if max_seq_len is not None else mh.max_context
    if seq_len is None:
        raise ValueError(
            f"max_seq_len is required: {model_name} config exposes no max context length"
        )

    # Enumerate from 1 GPU/worker; the KV estimate is the sole feasibility filter.
    # MoE tensor-parallel (moe_ep == 1) is enabled for every MoE model, MLA
    # included: real deployments (e.g. InferenceX GLM-5, reported as EP=1) run it,
    # so the search must be able to find it rather than have it filtered out here.
    common = dict(
        is_moe=mh.is_moe,
        backend=backend,
        gpu_budget=gpu_budget,
        min_gpu_budget=min_gpu_budget,
        enable_wideep=mh.enable_wideep,
        allow_moe_pure_tp=True,
    )
    if deployment_mode == "disagg":
        configs = enumerate_disagg_configs(
            **common,
            prefill_candidates=_resolved_role_candidates(
                prefill_candidates, mh, role="prefill"
            ),
            decode_candidates=_resolved_role_candidates(
                decode_candidates, mh, role="decode"
            ),
            num_gpu_per_replica=num_gpu_per_replica,
            max_gpu_per_replica=max_gpu_per_replica,
            max_prefill_workers=max_prefill_workers,
            max_decode_workers=max_decode_workers,
        )
    elif deployment_mode == "agg":
        role = _resolved_role_candidates(agg_candidates, mh, role="agg")
        configs = enumerate_parallel_configs(
            **common,
            gpus_per_worker_candidates=role.gpus_per_worker,
            tp_candidates=role.tp,
            pp_candidates=role.pp,
            attention_dp_candidates=role.attention_dp,
            moe_tp_candidates=role.moe_tp,
            moe_ep_candidates=role.moe_ep,
            cp_candidates=role.cp,
            worker_candidates=role.workers,
        )
    else:
        raise ValueError(
            f"deployment_mode must be 'agg' or 'disagg', got {deployment_mode!r}"
        )

    # KV-cache validity: keep configs whose every role-shape holds a max_seq_len sequence.
    if deployment_mode == "agg":
        shapes = [c.shape for c in configs]
    else:
        shapes = [c.prefill.shape for c in configs] + [c.decode.shape for c in configs]
    feasible = feasible_shape_tokens(
        shapes,
        model_name=model_name,
        hardware_sku=hardware_sku,
        backend=backend,
        max_seq_len=seq_len,
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        memory_fraction=memory_fraction,
    )
    if deployment_mode == "agg":
        kept = [c for c in configs if c.shape in feasible]
    else:
        kept = [
            c
            for c in configs
            if c.prefill.shape in feasible and c.decode.shape in feasible
        ]
    if not kept:
        raise NoViableParallelConfig(
            f"{model_name} on {hardware_sku}: no parallel config holds a {seq_len}-token "
            f"sequence within {gpu_budget} GPUs ({backend} KV-cache estimate)"
        )
    return kept
