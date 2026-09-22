# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile an exact FastAFD AGG profile into a model operation graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk.fastafd_profile import FastAFDMoEStageProfile

_PROFILE_PRECISION_ALIASES = {
    "w4a8_mxfp4_mxfp8_trtllm": "w4a8_mxfp4_mxfp8",
}


def apply_fastafd_moe_profile(
    model: Any,
    *,
    model_path: str,
    system: str,
    backend: str,
    nextn: int,
    profile_path: str | Path,
    profile_backend: str,
) -> dict[str, Any]:
    """Replace the decode MoE span with exact measured profile points."""

    config = model.config
    if backend != "sglang":
        raise ValueError("FastAFD MoE profiles require backend='sglang'")
    if config.pp_size != 1:
        raise ValueError("FastAFD MoE profiles require pp_size=1")
    if config.moe_tp_size != 1:
        raise ValueError("FastAFD MoE profiles require moe_tp_size=1")
    if profile_backend not in {"megamoe", "deepep_deepgemm"}:
        raise ValueError("fastafd_moe_backend must be 'megamoe' or 'deepep_deepgemm'")

    profile = FastAFDMoEStageProfile.load(profile_path)
    topology = f"ep{config.moe_ep_size}"
    moe_precision = _PROFILE_PRECISION_ALIASES.get(
        config.moe_quant_mode.value.name,
        config.moe_quant_mode.value.name,
    )
    common = {
        "model_path": model_path,
        "system": system,
        "stage": "agg",
        "topology": topology,
        "mtp_nextn": nextn,
        "microbatches": 1,
        "routed_topk": model._topk,
        "moe_precision": moe_precision,
        "moe_backend": profile_backend,
    }
    matches = [
        entry for entry in profile.entries if all(getattr(entry.key, field) == value for field, value in common.items())
    ]
    if not matches:
        raise ValueError(f"no FastAFD AGG measurements match {common}")
    measured_layer_counts = {entry.key.moe_layers for entry in matches}
    if len(measured_layer_counts) != 1:
        raise ValueError("FastAFD AGG measurements disagree on moe_layers")
    measured_layers = measured_layer_counts.pop()
    if measured_layers > model._num_layers:
        raise ValueError("FastAFD moe_layers exceeds the model layer count")

    points = sorted(
        (
            entry.key.logical_batch_per_source_rank * (nextn + 1),
            entry.latency_ms,
        )
        for entry in matches
    )
    if len({tokens for tokens, _ in points}) != len(points):
        raise ValueError("FastAFD AGG measurements contain duplicate runtime token counts")

    stage = ops.FastAfdMoeStage(
        "generation_fastafd_moe_stage",
        points,
        profile.profile_sha256,
    )
    _replace_generation_moe(model, stage, measured_layers=measured_layers)
    return {
        "provider": "fastafd",
        "profile_path": str(profile.source),
        "profile_sha256": profile.profile_sha256,
        "moe_backend": profile_backend,
        "stage": "agg",
        "topology": topology,
        "points": len(points),
        "moe_layers": measured_layers,
    }


def _replace_generation_moe(model: Any, stage: Any, *, measured_layers: int) -> None:
    generation_ops = list(model.generation_ops)
    residual_layers = model._num_layers - measured_layers

    overlap_indexes = [
        index
        for index, op in enumerate(generation_ops)
        if isinstance(op, ops.OverlapOp) and op._name == "generation_moe_overlap"
    ]
    if overlap_indexes:
        if len(overlap_indexes) != 1:
            raise ValueError("FastAFD requires exactly one generation_moe_overlap")
        index = overlap_indexes[0]
        overlap = generation_ops[index]
        routers = [op for op in overlap._group_a if op._name == "generation_router_gemm"]
        if len(routers) != 1:
            raise ValueError("FastAFD requires one generation router outside the measured stage")
        replacement = [routers[0], stage]
        if residual_layers:
            ratio = residual_layers / model._num_layers
            routed = _scaled_ops(
                [op for op in overlap._group_a if op._name != "generation_router_gemm"],
                ratio,
            )
            shared = _scaled_ops(overlap._group_b, ratio)
            replacement.append(
                ops.OverlapOp(
                    "generation_dense_ffn_approximation",
                    group_a=routed,
                    group_b=shared,
                )
            )
        model.generation_ops = generation_ops[:index] + replacement + generation_ops[index + 1 :]
        return

    router_indexes = [index for index, op in enumerate(generation_ops) if op._name == "generation_router_gemm"]
    if len(router_indexes) != 1:
        raise ValueError("FastAFD requires exactly one generation_router_gemm")
    router_index = router_indexes[0]
    end = router_index + 1
    while end < len(generation_ops) and generation_ops[end]._name.startswith("generation_moe"):
        end += 1
    if end == router_index + 1:
        raise ValueError("FastAFD could not identify the generation MoE span")
    model.generation_ops = generation_ops[: router_index + 1] + [stage] + generation_ops[end:]


def _scaled_ops(items: list[Any], ratio: float) -> list[Any]:
    for op in items:
        op._scale_factor *= ratio
    return items


__all__ = ["apply_fastafd_moe_profile"]
