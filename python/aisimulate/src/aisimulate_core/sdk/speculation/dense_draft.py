# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/dense_draft.py

"""Dense (GQA + gated-MLP) draft-block geometry shared by block-drafting
schemes whose draft is a stack of full-width dense decoder layers.

Ground truth: the Qwen3-8B community drafts
``deepseek-ai/dspark_qwen3_8b_block7`` and ``z-lab/Qwen3-8B-DFlash-b16``
are byte-identical in block geometry (5 full-attention Qwen3 layers,
hidden 4096 / inter 12288 / 32:8 heads / head_dim 128) and differ only in
the sampling heads. Both checkpoints' safetensors sizes close against this
geometry to <0.1%.

Op shapes mirror ``LLAMAModel`` exactly so draft compute prices off the
same perf tables as the target.
"""

from __future__ import annotations

from dataclasses import dataclass

from aisimulate_core.sdk.speculation.base import positive_integer


@dataclass(frozen=True)
class DenseDraftGeometry:
    """Geometry of one dense draft decoder stack, parsed from the draft
    checkpoint's HF ``config.json``."""

    num_layers: int
    hidden_size: int
    inter_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    # Nested speculators layer configs omit vocabulary; the owning scheme resolves it.
    vocab_size: int | None
    sliding_window: int | None
    use_qk_norm: bool

    @classmethod
    def from_hf_config(cls, cfg: dict) -> DenseDraftGeometry:
        required = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads")
        missing = [k for k in required if cfg.get(k) is None]
        if missing:
            raise ValueError(f"dense draft config lacks required keys: {missing}")
        dimensions = {key: positive_integer(cfg[key], key) for key in required}
        num_heads = dimensions["num_attention_heads"]
        hidden_size = dimensions["hidden_size"]
        if "head_dim" not in cfg and hidden_size % num_heads:
            raise ValueError("dense draft hidden_size must be divisible by num_attention_heads when head_dim is absent")
        window = None if cfg.get("use_sliding_window") is False else cfg.get("sliding_window")
        return cls(
            num_layers=dimensions["num_hidden_layers"],
            hidden_size=hidden_size,
            inter_size=dimensions["intermediate_size"],
            num_heads=num_heads,
            num_kv_heads=positive_integer(cfg.get("num_key_value_heads", num_heads), "num_key_value_heads"),
            head_dim=positive_integer(cfg.get("head_dim", hidden_size // num_heads), "head_dim"),
            vocab_size=positive_integer(cfg["vocab_size"], "vocab_size") if "vocab_size" in cfg else None,
            sliding_window=positive_integer(window, "sliding_window") if window is not None else None,
            # Qwen3 layers carry q/k norms; plain llama layers do not.
            use_qk_norm=cfg.get("model_type") == "qwen3",
        )


def dense_block_ops(geom: DenseDraftGeometry, model, prefix: str, *, is_context: bool) -> list:
    """One dense draft decoder-block stack (count = geom.num_layers),
    op-for-op the LLAMAModel layer graph (minus embed/logits, which the
    calling scheme owns — they differ per scheme)."""
    import aisimulate_core.sdk.operations as ops

    cfg = model.config
    tp_size = cfg.tp_size
    if geom.num_heads % tp_size:
        raise ValueError(f"draft num_attention_heads {geom.num_heads} must be divisible by tp_size {tp_size}")
    if geom.inter_size % tp_size:
        raise ValueError(f"draft intermediate_size {geom.inter_size} must be divisible by tp_size {tp_size}")
    h = geom.hidden_size
    n = float(geom.num_layers)
    kv_per_gpu = max(1, geom.num_kv_heads // tp_size)
    attn_args = dict(head_size=geom.head_dim, use_qk_norm=geom.use_qk_norm, window_size=geom.sliding_window or 0)
    attn = (
        ops.ContextAttention(
            f"{prefix}_attention",
            n,
            geom.num_heads // tp_size,
            kv_per_gpu,
            cfg.kvcache_quant_mode,
            cfg.fmha_quant_mode,
            cp_size=1,
            **attn_args,
        )
        if is_context
        else ops.GenerationAttention(
            f"{prefix}_attention",
            n,
            geom.num_heads // tp_size,
            kv_per_gpu,
            cfg.kvcache_quant_mode,
            **attn_args,
        )
    )
    return [
        ops.ElementWise(f"{prefix}_add_norm_1", n, 2 * h, 2 * h, 0.8),
        ops.GEMM(
            f"{prefix}_qkv_gemm",
            n,
            geom.num_heads * geom.head_dim // tp_size + geom.head_dim * kv_per_gpu * 2,
            h,
            cfg.gemm_quant_mode,
        ),
        attn,
        ops.GEMM(
            f"{prefix}_proj_gemm",
            n,
            h,
            geom.num_heads * geom.head_dim // tp_size,
            cfg.gemm_quant_mode,
            low_precision_input=True,
        ),
        ops.ElementWise(f"{prefix}_add_norm_2", n, 2 * h, 2 * h, 0.8),
        ops.GEMM(f"{prefix}_gate_ffn1_gemm", n, 2 * geom.inter_size // tp_size, h, cfg.gemm_quant_mode),
        ops.ElementWise(f"{prefix}_act_gate", n, 2 * geom.inter_size // tp_size, geom.inter_size // tp_size, 0.8),
        ops.GEMM(
            f"{prefix}_ffn2_gemm",
            n,
            h,
            geom.inter_size // tp_size,
            cfg.gemm_quant_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce(f"{prefix}_ar_1", n, h, tp_size),
        ops.CustomAllReduce(f"{prefix}_ar_2", n, h, tp_size),
    ]


def dense_kv_bytes_per_sequence(geom: DenseDraftGeometry, model, seq_len: int) -> float:
    """Per-sequence draft KV bytes (per GPU): K+V entries per token per
    layer, sharded over TP like the target's GQA cache."""
    kv_per_gpu = max(1, geom.num_kv_heads // model.config.tp_size)
    tokens = min(max(seq_len, 0), geom.sliding_window) if geom.sliding_window else max(seq_len, 0)
    entry_bytes = 2 * kv_per_gpu * geom.head_dim * model.config.kvcache_quant_mode.value.memory
    return float(geom.num_layers * tokens * entry_bytes)
