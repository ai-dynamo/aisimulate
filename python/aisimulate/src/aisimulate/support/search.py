# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic bounded search profile for the support MVP."""

from __future__ import annotations

from typing import Any

from .errors import SupportWorkflowError
from .identity import candidate_id
from .schema import SupportRequest, TopologyCandidate, WorkloadSpec

_WIDTHS = (1, 2, 4, 8)


def _resolve_model_kind(request: SupportRequest) -> str:
    configured = request.identity.model_kind
    if configured != "auto":
        return configured
    try:
        from aiconfigurator.sdk.models import check_is_moe

        return "moe" if check_is_moe(request.identity.model) else "dense"
    except Exception as exc:
        raise SupportWorkflowError(
            "could not determine whether the model is dense or MoE; set identity.model_kind "
            "to 'dense' or 'moe' in the support request"
        ) from exc


def _candidate(
    *,
    replicas: int,
    tensor: int,
    pipeline: int = 1,
    attention_data: int,
    moe_tensor: int,
    moe_expert: int,
) -> TopologyCandidate:
    total_gpus = replicas * tensor * attention_data * pipeline
    payload = {
        "replicas": replicas,
        "tensor": tensor,
        "pipeline": pipeline,
        "attention_data": attention_data,
        "moe_tensor": moe_tensor,
        "moe_expert": moe_expert,
        "total_gpus": total_gpus,
    }
    return TopologyCandidate(id=candidate_id(payload), **payload)


def bounded_topologies(request: SupportRequest) -> list[TopologyCandidate]:
    """Return at most 16 stable topology candidates without a Cartesian expansion."""

    budget = request.identity.gpu_count
    model_kind = _resolve_model_kind(request)
    candidates: dict[str, TopologyCandidate] = {}
    for width in (value for value in _WIDTHS if value <= budget):
        replicas = (1, budget // width)
        shapes = [(width, 1, 1, 1)]
        if model_kind == "moe":
            shapes = [(width, 1, width, 1)]
            if width > 1:
                shapes.extend(
                    (
                        (width, 1, 1, width),
                        (1, width, 1, width),
                    )
                )
        for tensor, attention_data, moe_tensor, moe_expert in shapes:
            for replica_count in replicas:
                item = _candidate(
                    replicas=replica_count,
                    tensor=tensor,
                    attention_data=attention_data,
                    moe_tensor=moe_tensor,
                    moe_expert=moe_expert,
                )
                if item.total_gpus <= budget:
                    candidates[item.id] = item

    # Curated large-model fallbacks: TP8 plus the minimum PP degrees needed when
    # a model cannot fit at PP1. These are deliberately not crossed with DP/EP.
    if budget >= 16:
        tensor = min(8, max(value for value in _WIDTHS if value <= budget))
        for pipeline in (2, 4, 8):
            worker_gpus = tensor * pipeline
            if worker_gpus > budget:
                continue
            moe_tensor = tensor if model_kind == "moe" else 1
            for replicas in dict.fromkeys((1, budget // worker_gpus)):
                item = _candidate(
                    replicas=replicas,
                    tensor=tensor,
                    pipeline=pipeline,
                    attention_data=1,
                    moe_tensor=moe_tensor,
                    moe_expert=1,
                )
                if item.total_gpus <= budget:
                    candidates[item.id] = item

    if not candidates:
        raise SupportWorkflowError(f"the mvp-v1 profile produced no topology within the {budget}-GPU budget")

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            item.total_gpus != 1,
            item.total_gpus != budget,
            item.total_gpus,
            item.tensor,
            item.pipeline,
            item.attention_data,
            item.moe_tensor,
            item.moe_expert,
            item.replicas,
        ),
    )
    return ordered[: request.search.max_candidates]


def _traffic(workload: WorkloadSpec) -> dict[str, Any]:
    if workload.kind == "synthetic":
        return {
            "source": {
                "type": "synthetic",
                "input_tokens": workload.input_tokens,
                "output_tokens": workload.output_tokens,
            },
            "load": {"type": "concurrency", "concurrency": workload.concurrency},
            "stop": {"requests": workload.request_count},
        }
    load: dict[str, Any]
    if workload.trace_format in {"agentic_mooncake", "dynamo", "mooncake", "mooncake-delta"}:
        load = {"type": "trace_timestamps", "speedup": 1.0}
    else:
        load = {"type": "concurrency", "concurrency": workload.concurrency}
    return {
        "source": {
            "type": "trace",
            "paths": [workload.trace_path],
            "format": workload.trace_format,
        },
        "load": load,
    }


def recommendation_config(
    request: SupportRequest,
    workload: WorkloadSpec,
    topologies: list[TopologyCandidate],
    *,
    systems_path: str,
) -> dict[str, Any]:
    """Build a public `aisimulate recommend` config over only the bounded candidates."""

    return {
        "traffic": _traffic(workload),
        "engine": {
            "mode": request.identity.serving_mode,
            "model": request.identity.model,
            "hardware": request.identity.gpu,
            "backend": request.identity.framework,
            "backend_version": request.identity.framework_version,
            "forward_model": "fpm",
            "systems_path": systems_path,
            "context_length": request.search.context_length,
            "workers": {
                "aggregated": {
                    "parallelism": {"preset": [candidate.as_parallelism_preset() for candidate in topologies]}
                }
            },
        },
        "evaluation": {
            "sla": {
                "ttft_ms": workload.slo.ttft_ms,
                "itl_ms": workload.slo.tpot_ms,
            }
        },
        "optimization": {
            "target": request.search.objective,
            "constraints": {
                "min_candidate_gpus": 1,
                "max_candidate_gpus": request.identity.gpu_count,
            },
        },
        "optimizer": {
            "algorithm": "random",
            "max_trials": len(topologies),
            "parallelism": min(4, len(topologies)),
            "candidate_timeout_seconds": 600.0,
            "seed": request.search.seed,
        },
    }
