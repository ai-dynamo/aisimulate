# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, runtime-independent CUDA graph model features."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any

MODEL_CONFIG_ALIASES = {
    "MiniMaxAI/MiniMax-M3-MXFP8": "MiniMaxAI/MiniMax-M3",
}

MODEL_ARCHITECTURE_FEATURES = (
    "model_hidden_size",
    "model_num_hidden_layers",
    "model_num_attention_heads",
    "model_num_key_value_heads",
    "model_head_dim",
    "model_dense_intermediate_size",
    "model_moe_intermediate_size",
    "model_num_experts",
    "model_experts_per_token",
    "model_num_shared_experts",
    "model_vocab_size",
)

NUMERIC_FEATURES = (
    # Complete graph-shape summary.
    "cuda_graph_capture_count",
    "cuda_graph_active_capture_count",
    "cuda_graph_capture_size_sum",
    "cuda_graph_active_capture_size_sum",
    "cuda_graph_capture_size_squared_sum",
    "cuda_graph_capture_size_p50",
    "cuda_graph_capture_size_p90",
    "cuda_graph_largest_capture_size",
    "cuda_graph_full_count",
    "cuda_graph_full_largest_capture_size",
    "cuda_graph_piecewise_count",
    "cuda_graph_piecewise_largest_capture_size",
    # Scheduling limits.
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "speculative_tokens",
    # Raw topology.
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "dcp_size",
    "pcp_size",
    "moe_tp_size",
    "moe_ep_size",
    # Model architecture.
    *MODEL_ARCHITECTURE_FEATURES,
    # Per-rank architecture and graph/topology interactions.
    "layers_per_pipeline_rank",
    "attention_width_per_rank",
    "kv_width_per_rank",
    "dense_width_per_rank",
    "moe_width_per_rank",
    "experts_per_rank",
    "active_expert_width_per_rank",
    "parallel_world_size",
    "capture_attention_elements",
    "capture_kv_elements",
    "capture_dense_elements",
    "capture_moe_active_elements",
    "capture_moe_capacity_elements",
    "peak_attention_elements",
    "peak_moe_active_elements",
)

CATEGORICAL_FEATURES = (
    "gpu_family",
    "vllm_family",
    "quantization",
    "compute_dtype",
    "kv_cache_dtype",
    "cuda_graph_mode",
    "parallel_mode",
    "attention_backend",
    "compilation_mode",
    "compilation_backend",
    "moe_backend",
    "linear_backend",
    "flashinfer_autotune",
    "speculative_method",
)


def _model_config_id(model_id: str) -> str:
    return MODEL_CONFIG_ALIASES.get(model_id, model_id)


def _model_config_bytes(model_id: str) -> tuple[str, bytes] | None:
    config_id = _model_config_id(model_id)
    resource = files("aiconfigurator_core") / "model_configs" / f"{config_id.replace('/', '--')}_config.json"
    if not resource.is_file():
        return None
    return config_id, resource.read_bytes()


def model_architecture_features(model_id: str | None) -> dict[str, Any]:
    """Resolve stable architecture features from the packaged model config."""
    empty = {
        "model_architecture": None,
        "model_architecture_config_id": None,
        "model_architecture_config_sha256": None,
        **dict.fromkeys(MODEL_ARCHITECTURE_FEATURES),
    }
    if not model_id or (resolved := _model_config_bytes(model_id)) is None:
        return empty

    config_id, content = resolved
    raw = json.loads(content)
    config = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    architectures = config.get("architectures") or raw.get("architectures") or []
    num_experts = config.get("num_local_experts") or config.get("n_routed_experts") or config.get("num_experts") or 0
    dense_intermediate = config.get("intermediate_size") or 0
    moe_intermediate = (
        config.get("moe_intermediate_size")
        or config.get("routed_expert_hidden_size")
        or (dense_intermediate if num_experts else 0)
    )
    return {
        "model_architecture": architectures[0] if architectures else None,
        "model_architecture_config_id": config_id,
        "model_architecture_config_sha256": hashlib.sha256(content).hexdigest(),
        "model_hidden_size": config.get("hidden_size"),
        "model_num_hidden_layers": config.get("num_hidden_layers"),
        "model_num_attention_heads": config.get("num_attention_heads"),
        "model_num_key_value_heads": config.get("num_key_value_heads"),
        "model_head_dim": config.get("head_dim")
        or (
            config.get("hidden_size") // config.get("num_attention_heads")
            if config.get("hidden_size") and config.get("num_attention_heads")
            else None
        ),
        "model_dense_intermediate_size": dense_intermediate,
        "model_moe_intermediate_size": moe_intermediate,
        "model_num_experts": num_experts,
        "model_experts_per_token": config.get("num_experts_per_tok")
        or config.get("num_experts_per_token")
        or config.get("top_k")
        or 0,
        "model_num_shared_experts": config.get("num_shared_experts") or config.get("n_shared_experts") or 0,
        "model_vocab_size": config.get("vocab_size"),
    }


def _quantile(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    return values[math.ceil((len(values) - 1) * quantile)]


def _capture_sizes(value: object) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        return []
    return sorted({int(item) for item in value})


def graph_shape_features(row: Mapping[str, Any]) -> dict[str, int]:
    """Summarize the complete capture distribution and effective graph sets."""
    sizes = _capture_sizes(row.get("cuda_graph_capture_sizes"))
    speculative_width = max(1, int(row.get("speculative_tokens") or 0) + 1)
    active = [size for size in sizes if speculative_width == 1 or size % speculative_width == 0]
    if not active:
        active = sizes

    max_num_seqs = int(row.get("max_num_seqs") or 0)
    full_limit = max_num_seqs * speculative_width
    full = [size for size in active if not full_limit or size <= full_limit]
    graph_mode = str(row.get("cuda_graph_mode") or "").upper()
    if "FULL" not in graph_mode:
        full = []
    piecewise = active if "PIECEWISE" in graph_mode else []

    return {
        "cuda_graph_capture_count": len(sizes),
        "cuda_graph_active_capture_count": len(active),
        "cuda_graph_capture_size_sum": sum(sizes),
        "cuda_graph_active_capture_size_sum": sum(active),
        "cuda_graph_capture_size_squared_sum": sum(size * size for size in sizes),
        "cuda_graph_capture_size_p50": _quantile(sizes, 0.50),
        "cuda_graph_capture_size_p90": _quantile(sizes, 0.90),
        "cuda_graph_largest_capture_size": max(sizes, default=0),
        "cuda_graph_full_count": len(full),
        "cuda_graph_full_largest_capture_size": max(full, default=0),
        "cuda_graph_piecewise_count": len(piecewise),
        "cuda_graph_piecewise_largest_capture_size": max(piecewise, default=0),
    }


def _positive(value: object, default: float = 1.0) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parallel_mode(row: Mapping[str, Any]) -> str:
    tp = _positive(row.get("tp_size"))
    attention_dp = _positive(row.get("attention_dp_size"))
    moe_ep = _positive(row.get("moe_ep_size"))
    if attention_dp > 1 and moe_ep > 1:
        return "dep"
    if tp > 1 and moe_ep > 1:
        return "tep"
    if tp > 1:
        return "tp"
    return "single"


def _backend_family(version: object) -> str:
    text = str(version or "")
    match = re.match(r"(\d+)\.(\d+)", text)
    return ".".join(match.groups()) if match else text.split("+", maxsplit=1)[0]


def derived_feature_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the runtime-independent graph, architecture, and topology features."""
    result = dict(row)
    result.update(graph_shape_features(row))
    architecture = model_architecture_features(str(row.get("model_id") or ""))
    for field, value in architecture.items():
        if result.get(field) is None:
            result[field] = value

    tp = _positive(result.get("tp_size"))
    pp = _positive(result.get("pp_size"))
    attention_dp = _positive(result.get("attention_dp_size"))
    dcp = _positive(result.get("dcp_size"))
    pcp = _positive(result.get("pcp_size"))
    moe_tp = _positive(result.get("moe_tp_size"))
    moe_ep = _positive(result.get("moe_ep_size"))
    layers = _positive(result.get("model_num_hidden_layers"), 0.0)
    hidden = _positive(result.get("model_hidden_size"), 0.0)
    kv_heads = _positive(result.get("model_num_key_value_heads"), 0.0)
    head_dim = _positive(result.get("model_head_dim"), 0.0)
    dense_intermediate = _positive(result.get("model_dense_intermediate_size"), 0.0)
    moe_intermediate = _positive(result.get("model_moe_intermediate_size"), 0.0)
    experts = _positive(result.get("model_num_experts"), 0.0)
    top_k = _positive(result.get("model_experts_per_token"), 0.0)

    layers_per_rank = layers / pp
    attention_width = hidden / tp
    kv_width = kv_heads * head_dim / min(tp, max(kv_heads, 1.0))
    dense_width = dense_intermediate / tp
    moe_width = moe_intermediate / moe_tp
    experts_per_rank = experts / moe_ep
    active_expert_width = top_k * moe_width
    capture_sum = float(result["cuda_graph_active_capture_size_sum"])
    capture_max = float(result["cuda_graph_largest_capture_size"])

    result.update(
        {
            "gpu_family": str(result.get("system") or "").removesuffix("_sxm"),
            "vllm_family": _backend_family(result.get("backend_version")),
            "parallel_mode": _parallel_mode(result),
            "layers_per_pipeline_rank": layers_per_rank,
            "attention_width_per_rank": attention_width,
            "kv_width_per_rank": kv_width,
            "dense_width_per_rank": dense_width,
            "moe_width_per_rank": moe_width,
            "experts_per_rank": experts_per_rank,
            "active_expert_width_per_rank": active_expert_width,
            "parallel_world_size": tp * pp * attention_dp * dcp * pcp,
            "capture_attention_elements": capture_sum * layers_per_rank * attention_width,
            "capture_kv_elements": capture_sum * layers_per_rank * kv_width,
            "capture_dense_elements": capture_sum * layers_per_rank * dense_width,
            "capture_moe_active_elements": capture_sum * layers_per_rank * active_expert_width,
            "capture_moe_capacity_elements": capture_sum * layers_per_rank * moe_width * experts_per_rank,
            "peak_attention_elements": capture_max * attention_width,
            "peak_moe_active_elements": capture_max * active_expert_width,
        }
    )
    return result


__all__ = [
    "CATEGORICAL_FEATURES",
    "MODEL_ARCHITECTURE_FEATURES",
    "NUMERIC_FEATURES",
    "derived_feature_row",
    "graph_shape_features",
    "model_architecture_features",
]
