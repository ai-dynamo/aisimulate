# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD memory preflight and KV budget from native plan residency/buffers.

The shared backend computes per-pool activations, KV and fixed overhead with
empty partition ops; native physical weights replace the aggregate weights.
Decode verifies nextn+1 tokens, while prefill activations remain unscaled.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from aisimulate.sdk.backends.factory import get_backend
from aisimulate.sdk.config_builders import apply_nextn, build_model_config
from aisimulate.sdk.models import (
    get_model,
    resolve_context_fmha_by_data,
    resolve_dsv4_moe_arch,
)
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.memory import kv_cache_budget_bytes
from aisimulate_core.sdk.models.afd import _resolve_afd_execution_quantization
from aisimulate_core.sdk.perf_database import get_database_view, get_latest_database_version


class AFDMemoryInfeasible(RuntimeError):
    """One AFD pool does not fit in GPU memory (the OUT_OF_MEMORY category)."""


def _pool_memory(
    backend,
    database,
    model,
    *,
    partition_weight_gb: float,
    phase: str,
    batch_size: int,
    isl: int,
    osl: int,
    prefix: int,
    max_seq_len: int | None,
    include_kvcache: bool,
    kvcache_multiplier: int,
    prefill_tokens: int | None = None,
) -> dict[str, float]:
    if phase == "prefill":
        effective_max_seq_len = max_seq_len if max_seq_len is not None else isl
        num_tokens = 0
        multiplier = 1
        if prefill_tokens is not None:
            # The request/KV window remains complete even when only a chunk
            # is processed, or ISL is a placeholder for runtime trace traffic.
            isl = prefill_tokens
    else:
        effective_max_seq_len = max_seq_len if max_seq_len is not None else isl + osl
        num_tokens = batch_size
        multiplier = kvcache_multiplier
    memory = backend.get_partition_memory_usage(
        model,
        database,
        partition_ops=[],
        batch_size=batch_size,
        beam_width=1,
        isl=isl,
        osl=osl,
        num_tokens=num_tokens,
        prefix=prefix,
        max_seq_len=effective_max_seq_len,
        include_kvcache=include_kvcache,
        kvcache_multiplier=multiplier,
    )
    memory = dict(memory)
    memory["weights"] = partition_weight_gb
    memory["total"] = (
        memory["weights"]
        + memory.get("activations", 0.0)
        + memory.get("kvcache", 0.0)
        + memory.get("nccl", 0.0)
        + memory.get("others", 0.0)
    )
    return memory


def _build_pool(
    backend,
    backend_name: str,
    backend_version: str | None,
    a_version: str,
    model_path: str,
    gpus_per_node: int,
    phase: str,
    pool: dict[str, Any],
    *,
    is_a_pool: bool,
    quant_overrides: Mapping[str, str] | None = None,
    nextn: int = 0,
    database_mode: str = "SILICON",
    enable_shared_layer: bool | None = None,
    strict_provenance: bool = False,
    systems_paths: list[str] | None = None,
):
    """Resolve the pool metadata used by both memory checks and KV budgeting."""
    systems_paths = pool.get("systems_paths", systems_paths)
    system = pool["system"]
    version = backend_version or (
        a_version
        if is_a_pool
        else get_latest_database_version(system=system, backend=backend_name, systems_paths=systems_paths)
    )
    if version is None:
        raise AFDMemoryInfeasible(f"no perf database found for system={system!r}, backend={backend_name!r}")
    database = get_database_view(
        system,
        backend_name,
        version,
        systems_paths=systems_paths,
        database_mode=database_mode,
        # SOL needs hardware metadata only; measured modes retain the
        # missing-directory gate used by the native pool constructors.
        allow_missing_data=database_mode == "SOL",
        shared_layer=enable_shared_layer,
        strict_provenance=strict_provenance,
    )
    if database is None:
        raise AFDMemoryInfeasible(
            f"failed to load the perf database for system={system!r}, backend={backend_name!r}, version={version!r}"
        )
    if is_a_pool:
        tp = pool["a_tp_size"]
        model_config = build_model_config(tp, 1, 1, tp, 1, **(quant_overrides or {}))
        # The A replica runs context attention whenever the phase covers
        # prefill ("prefill" or "both") — the F rank never touches FMHA.
        resolve_context_fmha_by_data(
            model_config, model_path, database, backend_name, is_context_role=phase != "decode"
        )
    else:
        f_pool_width = pool["n_f_nodes"] * gpus_per_node
        model_config = build_model_config(
            f_pool_width, 1, 1, f_pool_width // pool["f_moe_ep_size"], pool["f_moe_ep_size"], **(quant_overrides or {})
        )
    resolve_dsv4_moe_arch(model_config, model_path, system_name=system, backend_name=backend_name)
    try:
        _resolve_afd_execution_quantization(model_config, model_path, database.system_spec, backend_name=backend_name)
    except (ValueError, RuntimeError) as exc:
        label = "A" if is_a_pool else "F"
        raise InvalidEngineConfigurationError(
            f"AFD {label} pool on {system!r}, requested precision={quant_overrides!r}: {exc}"
        ) from exc
    apply_nextn(model_config, nextn)
    # Memory needs model-specific KV metadata, not an unused F attention
    # graph validated against the expert-pool width.
    reference = dataclasses.replace(
        model_config,
        tp_size=model_config.tp_size if is_a_pool else 1,
        moe_tp_size=1,
        moe_ep_size=model_config.tp_size if is_a_pool else 1,
    )
    model = get_model(model_path, reference, backend.name.value)
    model.config = dataclasses.replace(
        model.config,
        tp_size=model_config.tp_size,
        moe_tp_size=model_config.moe_tp_size,
        moe_ep_size=model_config.moe_ep_size,
    )
    return database, model


def _prepare_a_pool(
    *,
    model_path: str,
    backend_name: str,
    backend_version: str | None,
    gpus_per_node: int,
    phase: str,
    a_pool: dict[str, Any],
    quant_overrides: Mapping[str, str] | None = None,
    nextn: int = 0,
    database_mode: str = "SILICON",
    enable_shared_layer: bool | None = None,
    strict_provenance: bool = False,
    systems_paths: list[str] | None = None,
):
    backend = get_backend(backend_name)
    systems_paths = a_pool.get("systems_paths", systems_paths)
    a_version = backend_version or get_latest_database_version(
        system=a_pool["system"], backend=backend_name, systems_paths=systems_paths
    )
    if a_version is None:
        raise AFDMemoryInfeasible(f"no perf database found for system={a_pool['system']!r}, backend={backend_name!r}")
    database, model = _build_pool(
        backend,
        backend_name,
        backend_version,
        a_version,
        model_path,
        gpus_per_node,
        phase,
        a_pool,
        is_a_pool=True,
        quant_overrides=quant_overrides,
        nextn=nextn,
        database_mode=database_mode,
        enable_shared_layer=enable_shared_layer,
        strict_provenance=strict_provenance,
        systems_paths=systems_paths,
    )
    return backend, a_version, database, model


def _pool_kv_budget(backend, capacity_gb: float, fraction: float | None, memory: Mapping[str, float]) -> float:
    reserved, tolerance = backend.get_kv_cache_memory_check_params()
    return kv_cache_budget_bytes(
        capacity=capacity_gb,
        non_kv=memory["total"] - memory.get("kvcache", 0.0),
        fraction=fraction,
        of_free=backend.memory_fraction_of_free(),
        reserved=reserved,
        tolerance=tolerance,
    )


def check_afd_memory_fit(
    *,
    model_path: str,
    backend_name: str,
    backend_version: str | None,
    isl: int,
    osl: int,
    prefix: int,
    nextn: int,
    a_pool: dict[str, Any],
    f_pool: dict[str, Any],
    gpus_per_node: int,
    f_gpus_per_node: int | None = None,
    batch_size_per_a_replica: int,
    num_microbatches: int,
    a_weight_gb: Mapping[str, float],
    f_weight_gb: Mapping[str, float],
    phase: str = "both",
    max_seq_len: int | None = None,
    prefill_tokens: int | None = None,
    a_workspace_gb: Mapping[str, float] | None = None,
    f_workspace_gb: Mapping[str, float] | None = None,
    f_activation_gb: Mapping[str, float] | None = None,
    a_activation_gb: Mapping[str, float] | None = None,
    quant_overrides: Mapping[str, str] | None = None,
    database_mode: str = "SILICON",
    enable_shared_layer: bool | None = None,
    strict_provenance: bool = False,
    systems_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Raise :class:`AFDMemoryInfeasible` when either pool does not fit.

    ``a_pool`` / ``f_pool`` carry ``system`` + the pool's parallel identity
    (``a_tp_size``/``n_a_nodes`` for A; ``f_moe_ep_size``/``n_f_nodes`` for F).
    ``a_weight_gb`` / ``f_weight_gb`` map each phase to the native plan's
    physical resident weight sum in GiB per GPU.
    ``prefill_tokens`` bounds new tokens per sequence in a prefill pass;
    it does not shorten the declared KV reservation.
    """

    from aisimulate.sweeper.afd_parallel import validate_afd_model
    from aisimulate_core._native import resolve_afd_parallelism, validate_afd_attention_tp

    validate_afd_attention_tp(a_pool["a_tp_size"])
    # Reject unsupported geometry before loading model metadata or databases.
    parallelism = (
        a_pool["a_tp_size"],
        a_pool["n_a_nodes"],
        f_pool["n_f_nodes"],
        gpus_per_node,
        f_pool.get("f_moe_ep_size"),
    )
    resolve_afd_parallelism(*parallelism, None, f_gpus_per_node=f_gpus_per_node)
    info = validate_afd_model(model_path)
    geometry = json.loads(resolve_afd_parallelism(*parallelism, info["num_experts"], f_gpus_per_node=f_gpus_per_node))
    f_pool = {**f_pool, "f_moe_ep_size": geometry["f_moe_ep_size"]}

    backend, a_version, a_database, a_model = _prepare_a_pool(
        model_path=model_path,
        backend_name=backend_name,
        backend_version=backend_version,
        gpus_per_node=gpus_per_node,
        phase=phase,
        a_pool=a_pool,
        quant_overrides=quant_overrides,
        nextn=nextn,
        database_mode=database_mode,
        enable_shared_layer=enable_shared_layer,
        strict_provenance=strict_provenance,
        systems_paths=systems_paths,
    )
    f_database, f_model = _build_pool(
        backend,
        backend_name,
        backend_version,
        a_version,
        model_path,
        f_gpus_per_node or gpus_per_node,
        phase,
        f_pool,
        is_a_pool=False,
        quant_overrides=quant_overrides,
        nextn=nextn,
        database_mode=database_mode,
        enable_shared_layer=enable_shared_layer,
        strict_provenance=strict_provenance,
        systems_paths=systems_paths,
    )

    # Geometry is validated by Rust before calling this memory adapter.
    a_micro_batch_size = (batch_size_per_a_replica + num_microbatches - 1) // num_microbatches
    # The F pool serves every A replica (b_micro_total).
    a_replica_count = a_pool["n_a_nodes"] * gpus_per_node // a_pool["a_tp_size"]
    b_micro_total = a_replica_count * a_micro_batch_size

    check_phases = ("prefill", "decode") if phase == "both" else (phase,)
    free_fraction = backend.get_default_free_gpu_memory_fraction(a_version)

    diagnostics = {}
    for check_phase in check_phases:
        diagnostics[check_phase] = {}
        for pool, database, model, weights, is_a_pool in (
            (a_pool, a_database, a_model, a_weight_gb, True),
            (f_pool, f_database, f_model, f_weight_gb, False),
        ):
            memory = _pool_memory(
                backend,
                database,
                model,
                partition_weight_gb=float(weights[check_phase]),
                phase=check_phase,
                batch_size=a_micro_batch_size if is_a_pool else b_micro_total,
                isl=isl,
                osl=osl,
                prefix=prefix,
                max_seq_len=max_seq_len,
                include_kvcache=is_a_pool,
                kvcache_multiplier=num_microbatches,
                prefill_tokens=prefill_tokens,
            )
            workspace = a_workspace_gb if is_a_pool else f_workspace_gb
            activations = a_activation_gb if is_a_pool else f_activation_gb
            extra = float((workspace or {}).get(check_phase, 0.0))
            peak = float((activations or {}).get(check_phase, 0.0))
            delta = max(0.0, peak - memory["activations"])
            memory["activations"] += delta
            if extra:
                memory["afd_workspace"] = extra
            memory["total"] += extra + delta
            mem_capacity_gb = int(database.system_spec["gpu"]["mem_capacity"]) / (1 << 30)
            label = "A" if is_a_pool else "F"
            diagnostics[check_phase]["attention" if is_a_pool else "ffn"] = {
                "system": pool["system"],
                "capacity_gib_per_gpu": mem_capacity_gb,
                "components_gib_per_gpu": {key: float(value) for key, value in memory.items()},
            }
            if memory["total"] >= mem_capacity_gb:
                raise AFDMemoryInfeasible(
                    f"out_of_memory: the model {model_path!r} does not fit in GPU memory on {pool['system']!r} "
                    f"with this AFD topology ({label} pool, phase={check_phase}): "
                    f"total {memory['total']:.2f} GB >= capacity {mem_capacity_gb:.2f} GB; "
                    "lower batch_size_per_a_replica or use a system with more VRAM per GPU"
                )
            if is_a_pool and free_fraction is not None:
                # Use the backend's canonical capacity/fraction/reserve contract.
                kv_gib = memory.get("kvcache", 0.0)
                kv_budget = _pool_kv_budget(backend, mem_capacity_gb, free_fraction, memory)
                if kv_gib > kv_budget:
                    raise AFDMemoryInfeasible(
                        f"out_of_memory: the {label}-pool KV cache ({kv_gib:.2f} GB) exceeds "
                        f"the budget {kv_budget:.2f} GB on {pool['system']!r}; lower "
                        "batch_size_per_a_replica, or use a system with more VRAM per GPU"
                    )

    return diagnostics


def estimate_afd_kv_budget_bytes(
    *,
    model_path: str,
    backend_name: str,
    backend_version: str | None,
    isl: int,
    osl: int,
    prefix: int,
    a_pool: dict[str, Any],
    gpus_per_node: int,
    batch_size_per_a_replica: int,
    num_microbatches: int,
    a_weight_gb: Mapping[str, float],
    phase: str = "both",
    max_seq_len: int | None = None,
    prefill_tokens: int | None = None,
    a_workspace_gb: Mapping[str, float] | None = None,
    a_activation_gb: Mapping[str, float] | None = None,
    quant_overrides: Mapping[str, str] | None = None,
    nextn: int = 0,
    database_mode: str = "SILICON",
    enable_shared_layer: bool | None = None,
    strict_provenance: bool = False,
    systems_paths: list[str] | None = None,
) -> tuple[float, float]:
    """Return ``(kv_budget_bytes_per_gpu, kv_bytes_per_token_per_gpu)`` for the A pool.

    Shares the fit gate's memory model so the admission capacity derived from
    this estimate and the OOM check cannot drift. The budget is the minimum
    over the covered phases (the decode phase is retract-relevant: its KV
    footprint grows with generation; a prefill-only pool budgets on prefill).
    The per-token price is measured at unit geometry (one sequence, no
    microbatch multiplier) and is a phase-independent model property.
    """

    from aisimulate.sweeper.afd_parallel import validate_afd_model
    from aisimulate_core._native import validate_afd_attention_tp

    validate_afd_attention_tp(a_pool["a_tp_size"])
    validate_afd_model(model_path)
    backend, a_version, database, model = _prepare_a_pool(
        model_path=model_path,
        backend_name=backend_name,
        backend_version=backend_version,
        gpus_per_node=gpus_per_node,
        phase=phase,
        a_pool=a_pool,
        quant_overrides=quant_overrides,
        nextn=nextn,
        database_mode=database_mode,
        enable_shared_layer=enable_shared_layer,
        strict_provenance=strict_provenance,
        systems_paths=systems_paths,
    )

    # Geometry is validated by Rust before calling this memory adapter.
    a_micro_batch_size = (batch_size_per_a_replica + num_microbatches - 1) // num_microbatches
    check_phases = ("prefill", "decode") if phase == "both" else (phase,)
    free_fraction = backend.get_default_free_gpu_memory_fraction(a_version)
    mem_capacity_gb = int(database.system_spec["gpu"]["mem_capacity"]) / (1 << 30)

    budget_gib: float | None = None
    for check_phase in check_phases:
        memory = _pool_memory(
            backend,
            database,
            model,
            partition_weight_gb=float(a_weight_gb[check_phase]),
            phase=check_phase,
            batch_size=a_micro_batch_size,
            isl=isl,
            osl=osl,
            prefix=prefix,
            max_seq_len=max_seq_len,
            include_kvcache=True,
            kvcache_multiplier=num_microbatches,
            prefill_tokens=prefill_tokens,
        )
        extra = float((a_workspace_gb or {}).get(check_phase, 0.0))
        peak = float((a_activation_gb or {}).get(check_phase, 0.0))
        delta = max(0.0, peak - memory["activations"])
        memory["activations"] += delta
        if extra:
            memory["afd_workspace"] = extra
        memory["total"] += extra + delta
        phase_budget = _pool_kv_budget(backend, mem_capacity_gb, free_fraction, memory)
        budget_gib = phase_budget if budget_gib is None else min(budget_gib, phase_budget)

    effective_seq = max_seq_len if max_seq_len is not None else isl + osl
    unit = _pool_memory(
        backend,
        database,
        model,
        partition_weight_gb=0.0,
        phase="decode",
        batch_size=1,
        isl=isl,
        osl=osl,
        prefix=prefix,
        max_seq_len=max_seq_len,
        include_kvcache=True,
        kvcache_multiplier=1,
    )
    kv_per_token_gib = unit.get("kvcache", 0.0) / effective_seq
    if budget_gib is None or budget_gib <= 0:
        raise AFDMemoryInfeasible(
            f"the A-pool KV budget on {a_pool['system']!r} is not positive ({budget_gib}); "
            "use a system with more VRAM per GPU"
        )
    if kv_per_token_gib <= 0:
        raise AFDMemoryInfeasible("the A-pool KV per-token estimate is not positive")
    gib = float(1 << 30)
    return budget_gib * gib, kv_per_token_gib * gib
