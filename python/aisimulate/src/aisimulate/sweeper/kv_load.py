# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve candidate-relative synthetic load from AIC KV-cache capacity."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

from .config import ENGINE_MODEL_CONTROL_FIELDS, Workload
from .forward_pass_estimator import resolve_systems_paths
from .kv_estimate import estimate_kv_tokens
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig


class InfeasibleKVCapacity(ValueError):
    """A candidate's selected shape and batching leave no usable KV capacity."""


@dataclass(frozen=True)
class KVLoadResolution:
    """The requested normalized load and the concrete closed-loop concurrency."""

    ratio: float
    concurrency: int
    concurrency_capacity: int
    role_capacity_tokens: dict[str, int]


@cache
def _per_rank_capacity_tokens(
    shape: ParallelShape,
    *,
    model_name: str,
    hardware_sku: str,
    backend: str,
    backend_version: str,
    systems_paths: tuple[str, ...],
    max_num_tokens: int,
    max_batch_size: int,
    memory_fraction: float,
    nextn: int,
    model_controls: tuple[tuple[str, str | int | bool], ...] = (),
) -> int:
    tokens = estimate_kv_tokens(
        shape,
        model_name=model_name,
        hardware_sku=hardware_sku,
        backend=backend,
        backend_version=backend_version,
        systems_paths=list(systems_paths),
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        memory_fraction=memory_fraction,
        nextn=nextn,
        **({"model_controls": dict(model_controls)} if model_controls else {}),
    )
    if tokens is None:
        raise InfeasibleKVCapacity(
            f"no KV budget for backend={backend}, shape={shape}, "
            f"max_num_batched_tokens={max_num_tokens}, max_num_seqs={max_batch_size}"
        )
    return tokens


def _role_capacity_tokens(
    sample: dict[str, Any],
    *,
    role: str,
    config: ReplicaParallelConfig,
    backend_version: str,
) -> int:
    """Aggregate scheduler-visible KV tokens across attention-DP ranks and replicas."""
    block_size = int(sample[f"{role}_block_size"])
    if block_size <= 0:
        raise ValueError(f"{role}_block_size must be greater than zero, got {block_size}")
    fixed_blocks = sample.get(f"{role}_num_gpu_blocks")
    if fixed_blocks is not None:
        per_rank_tokens = int(fixed_blocks) * block_size
    else:
        resolved = sample.get("forward_pass_estimators", {}).get(role, {}).get("config")
        if resolved is None:
            timing = sample.get(f"{role}_timing_model")
            if isinstance(timing, dict) and timing.get("type") == "external" and timing.get("provider") == "aic":
                resolved = timing.get("config")
                if not isinstance(resolved, dict):
                    raise ValueError(f"{role} external AIC timing config must be a mapping")
            else:
                resolved = {}
        roots = resolved.get("systems_paths")
        if not roots and resolved.get("systems_path"):
            roots = [resolved["systems_path"]]
        per_rank_tokens = _per_rank_capacity_tokens(
            config.shape,
            model_name=str(resolved.get("model", resolved.get("model_path", sample["model_name"]))),
            hardware_sku=str(resolved.get("system", sample.get(f"{role}_hardware_sku") or sample["hardware_sku"])),
            backend=str(resolved.get("backend", sample["backend"])),
            backend_version=resolved.get("backend_version") or backend_version,
            systems_paths=resolve_systems_paths(roots),
            max_num_tokens=int(sample[f"{role}_max_num_batched_tokens"]),
            max_batch_size=int(sample[f"{role}_max_num_seqs"]),
            memory_fraction=float(sample[f"{role}_gpu_memory_utilization"]),
            nextn=int(resolved.get("nextn", sample.get("aic_nextn")) or 0),
            model_controls=tuple(
                (name, resolved.get(name, sample.get(name)))
                for name in ENGINE_MODEL_CONTROL_FIELDS
                if resolved.get(name, sample.get(name)) is not None
                and not (name == "enable_eplb" and resolved.get(name, sample.get(name)) is False)
            ),
        )
    # Dynamo's AIC estimator returns per-rank blocks. Offline replay models one
    # engine-wide KV pool, so attention-DP ranks contribute independent capacity;
    # tensor/expert parallel ranks shard the same sequences and are not multipliers.
    per_rank_usable_tokens = (per_rank_tokens // block_size) * block_size
    return per_rank_usable_tokens * config.shape.dp * config.replicas


def resolve_kv_load(
    sample: dict[str, Any],
    *,
    workload: Workload,
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig,
    ratio: float,
    backend_version: str,
) -> KVLoadResolution:
    """Map a normalized KV load to candidate-specific closed-loop concurrency.

    ``ratio=1`` is the estimated steady-state KV occupancy where each in-flight
    request holds ``isl + floor(osl / 2)`` tokens on average. ``ratio=0`` is the
    minimum useful replay load and therefore maps to one request.
    """
    if workload.isl is None or workload.osl is None:
        raise ValueError("kv_load_ratio requires a synthetic workload with isl and osl")

    if isinstance(parallel_config, DisaggParallelConfig):
        role_configs = {
            "prefill": parallel_config.prefill,
            "decode": parallel_config.decode,
        }
        load_role = "decode"
    elif isinstance(parallel_config, ReplicaParallelConfig):
        role_configs = {"agg": parallel_config}
        load_role = "agg"
    else:
        raise TypeError(f"unsupported parallel config for KV load: {type(parallel_config).__name__}")

    capacities = {
        role: _role_capacity_tokens(sample, role=role, config=config, backend_version=backend_version)
        for role, config in role_configs.items()
    }
    isl = int(workload.isl)
    images = getattr(workload, "images", None)
    if images is not None:
        # Visual placeholders occupy KV like text; size the load on the effective prompt.
        from aisimulate_core.sdk.backends.base_backend import BaseBackend
        from aisimulate_core.sdk.config import RuntimeConfig

        isl = BaseBackend.effective_prefill_isl(
            str(sample["model_name"]),
            RuntimeConfig(
                isl=isl, image_height=images.height, image_width=images.width, num_images_per_request=images.count
            ),
        )
    expected_tokens_per_request = isl + int(workload.osl) // 2
    if expected_tokens_per_request <= 0:
        raise InfeasibleKVCapacity(
            f"kv_load_ratio requires positive average tokens per request, got isl={workload.isl}, osl={workload.osl}"
        )
    concurrency_capacity = capacities[load_role] // expected_tokens_per_request
    if concurrency_capacity < 1:
        raise InfeasibleKVCapacity(
            f"{load_role} KV capacity {capacities[load_role]} tokens cannot hold the "
            f"estimated {expected_tokens_per_request} tokens per in-flight request"
        )
    concurrency = max(1, int(float(ratio) * concurrency_capacity))
    return KVLoadResolution(
        ratio=float(ratio),
        concurrency=concurrency,
        concurrency_capacity=concurrency_capacity,
        role_capacity_tokens=capacities,
    )
