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
from aiconfigurator_core.sdk.models import check_is_moe, supports_context_parallelism
from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

from .kv_estimate import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_NUM_TOKENS,
    DEFAULT_MEMORY_FRACTION,
    feasible_shape_tokens,
)
from .parallel_enum import (
    DisaggParallelConfig,
    PreparationBudget,
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
    num_experts: int = 0
    num_heads: int = 0
    num_layers: int = 0


def resolve_model_hardware(model_name: str, hardware_sku: str, *, backend: str) -> ModelHardware:
    """Read the model weights + SKU spec (via AIC) to derive is_moe / mla / wideep
    and the model's max context length."""
    model_config = get_model_config_from_model_path(model_name)
    is_moe = check_is_moe(model_name)
    architecture = model_config.get("architecture", "")
    allow_pure_tp = is_moe and architecture in _GQA_MOE_ARCHITECTURES
    mla = is_moe and not allow_pure_tp
    max_context = model_config.get("context")
    num_experts = int(model_config.get("num_experts") or model_config.get("n_routed_experts") or 0)

    system_spec = perf_database.load_system_spec(hardware_sku)
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
        num_heads=int(model_config.get("n") or 0),
        num_layers=int(model_config.get("layers") or 0),
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
    agg_candidates: RoleParallelCandidates | None = None,
    prefill_candidates: RoleParallelCandidates | None = None,
    decode_candidates: RoleParallelCandidates | None = None,
    num_gpu_per_replica: tuple[int, ...] | None = None,
    max_gpu_per_replica: int | None = None,
    max_prefill_workers: int | None = None,
    max_decode_workers: int | None = None,
    preparation: PreparationBudget | None = None,
    role_runtime: dict[str, tuple[int, int, float] | tuple[int, int, float, int | None]] | None = None,
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
    preparation = preparation or PreparationBudget()
    mh = resolve_model_hardware(model_name, hardware_sku, backend=backend)
    seq_len = max_seq_len if max_seq_len is not None else mh.max_context
    if seq_len is None:
        raise ValueError(f"max_seq_len is required: {model_name} config exposes no max context length")

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
        preparation=preparation,
        stage=f"{deployment_mode}.{backend}",
    )
    if deployment_mode == "disagg":
        configs = enumerate_disagg_configs(
            **common,
            prefill_candidates=prefill_candidates,
            decode_candidates=decode_candidates,
            num_gpu_per_replica=num_gpu_per_replica,
            max_gpu_per_replica=max_gpu_per_replica,
            max_prefill_workers=max_prefill_workers,
            max_decode_workers=max_decode_workers,
        )
    elif deployment_mode == "agg":
        role = agg_candidates or RoleParallelCandidates()
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
        filtered = [
            c
            for c in configs
            if (max_gpu_per_replica is None or c.total_gpus <= max_gpu_per_replica)
            and (num_gpu_per_replica is None or c.total_gpus in num_gpu_per_replica)
        ]
        preparation.record(f"{deployment_mode}.{backend}.replica", "considered", len(configs))
        preparation.record(f"{deployment_mode}.{backend}.replica", "accepted", len(filtered))
        preparation.record(f"{deployment_mode}.{backend}.replica", "rejected", len(configs) - len(filtered))
        configs = filtered
    else:
        raise ValueError(f"deployment_mode must be 'agg' or 'disagg', got {deployment_mode!r}")

    cp_supported: bool | None = None

    def supported(config):
        nonlocal cp_supported
        roles = (
            (("agg", config),) if deployment_mode == "agg" else (("prefill", config.prefill), ("decode", config.decode))
        )
        for role_name, role_config in roles:
            shape = role_config.shape
            if (mh.num_heads and mh.num_heads % shape.tp) or (mh.num_layers and mh.num_layers % shape.pp):
                preparation.record(f"{deployment_mode}.{backend}.capability", "model_divisibility")
                return False
            if role_config.shape.cp > 1:
                if cp_supported is None:
                    cp_supported = supports_context_parallelism(model_name, backend)
                if role_name == "decode" or not cp_supported:
                    preparation.record(f"{deployment_mode}.{backend}.capability", "context_parallelism")
                    return False
        return True

    configs = [c for c in configs if supported(c)]

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
        if fixed_tokens is not None:
            return {shape: fixed_tokens for shape in dict.fromkeys(shapes) if fixed_tokens > seq_len}
        return feasible_shape_tokens(
            shapes,
            model_name=model_name,
            hardware_sku=hardware_sku,
            backend=backend,
            backend_version=backend_version,
            max_seq_len=seq_len,
            max_num_tokens=role_tokens,
            max_batch_size=role_batch,
            memory_fraction=role_memory,
        )

    if deployment_mode == "agg":
        feasible = feasible_for("agg", [c.shape for c in configs])
        kept = [c for c in configs if c.shape in feasible]
    else:
        prefill_feasible = feasible_for("prefill", [c.prefill.shape for c in configs])
        decode_feasible = feasible_for("decode", [c.decode.shape for c in configs])
        kept = [c for c in configs if c.prefill.shape in prefill_feasible and c.decode.shape in decode_feasible]
    preparation.record(f"{deployment_mode}.{backend}.kv", "considered", len(configs))
    preparation.record(f"{deployment_mode}.{backend}.kv", "accepted", len(kept))
    preparation.record(f"{deployment_mode}.{backend}.kv", "kv_infeasible", len(configs) - len(kept))
    if not kept:
        raise NoViableParallelConfig(
            f"{model_name} on {hardware_sku}: no parallel config holds a {seq_len}-token "
            f"sequence within {gpu_budget} GPUs ({backend} KV-cache estimate)"
        )
    return kept
