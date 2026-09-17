# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Gemma 4 topology adapted and modified for performance modeling from:
# https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/gemma4.py
# https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/gemma4_mm.py
# Copyright contributors to the vLLM project.
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/gemma4/modeling_gemma4.py
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Both upstream projects are licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
# Kimi topology is a modified adaptation (Apache-2.0), copyright contributors
# to the vLLM project:
# https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25_vit.py

"""Generic ViT encoder op builder for multimodal VL models.

Provides :func:`build_encoder_ops`, a module-level function that constructs the
full list of encoder ops for any ViT-based vision encoder. Model classes call
this function in their ``__init__`` instead of duplicating op construction logic.

Op structure
------------
For a ViT with depth D and projector_dims with P (in, out) pairs::

  _patch_embedding_ops  →  2 ops, each with count=1:
    encoder_patch_embed_gemm  GEMM equivalent of the strided Conv3D patch projection
    encoder_position_embed    ElementWise position interpolation/addition

  _vit_transformer_ops  →  10 ops, each with count=depth:
    encoder_add_norm_1    ElementWise
    encoder_qkv_gemm      GEMM
    encoder_attention     EncoderAttention   (non-causal, MHA, no KV cache)
    encoder_proj_gemm     GEMM  (low_precision_input=True)
    encoder_ar_1          CustomAllReduce
    encoder_add_norm_2    ElementWise
    encoder_gate_gemm     GEMM  (only if gated_mlp; SwiGLU gate projection)
    encoder_ffn1_gemm     GEMM
    encoder_act           ElementWise
    encoder_ffn2_gemm     GEMM  (low_precision_input=True)
    encoder_ar_2          CustomAllReduce
    encoder_rope_apply    ElementWise  (only if partial_rotary_factor > 0;
                                        replaces the attention-internal RoPE term)

  _projector_ops  →  P GEMMs + (P-1) activations + optional norms / AR
                     (or 0 ops if projector_dims is empty):
    encoder_merger_norm          ElementWise pre-pixel-shuffle LayerNorm
    encoder_projector_fc{i}_gemm  GEMM
    encoder_projector_fc{i}_act   ElementWise  (omitted for final layer)
    encoder_projector_ar          CustomAllReduce

  Encoder DP (enable_encoder_dp, default; vLLM mm_encoder_tp_mode="data" /
  SGLang --mm-enable-dp-encoder) builds all of the above with tp=1 — full
  replica per rank, visuals ceil-sharded across the tp_size ranks at query
  time in BaseBackend._run_encoder_phase — and appends for tp_size > 1:
    encoder_dp_all_gather         NCCL all_gather of post-merge embeddings

TP parallelism for projector layers
------------------------------------
The ViT transformer ends with a CustomAllReduce so every projector layer
receives a full (un-sharded) first-layer input.  For a two-layer projector
(the common case for PatchMerger-style architectures):

  - Layer 0: row-parallel   (M = out // tp, K = in      — shards the output)
  - Layer 1: column-parallel (M = out,       K = in // tp — input is sharded
                               from the previous layer, output is reduced by AR)

For P = 1 the single layer is row-parallel (M = out // tp, K = in) followed by
the AllReduce.  For P > 2 intermediate layers also receive sharded inputs; callers
are responsible for choosing a projector_dims layout that is TP-correct.
Replicated projectors retain full dimensions for every layer and omit AllReduce.
"""

from __future__ import annotations

import dataclasses

import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk import common


def _patch_embedding_ops(enc_cfg: common.VisionEncoderConfig) -> list:
    """Build the input patch projection and position-embedding operations.

    Qwen's strided Conv3D has kernel and stride equal to
    ``(temporal_patch_size, patch_size, patch_size)``. Patches do not overlap,
    so each output patch is semantically the same linear projection represented
    by the GEMM below.
    """
    patch_volume = enc_cfg.in_channels * enc_cfg.temporal_patch_size * enc_cfg.patch_size * enc_cfg.patch_size
    vit_gemm_mode = common.GEMMQuantMode.bfloat16
    return [
        ops.GEMM("encoder_patch_embed_gemm", 1, enc_cfg.hidden_size, patch_volume, vit_gemm_mode),
        # Reads the patch embedding and interpolated position table, then writes their sum.
        ops.ElementWise("encoder_position_embed", 1, 2 * enc_cfg.hidden_size, enc_cfg.hidden_size, 0.8),
    ]


def _vit_transformer_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int) -> list:
    """Build the 10 ViT transformer block ops (each repeated enc_cfg.depth times),
    plus the optional encoder_rope_apply elementwise op.

    Raises ValueError if num_heads or intermediate_size is not divisible by tp_size.
    """
    depth = enc_cfg.depth
    h_vit = enc_cfg.hidden_size
    n_vit = enc_cfg.num_heads
    inter_vit = enc_cfg.intermediate_size
    qkv_hidden = enc_cfg.qkv_hidden_size or h_vit
    if qkv_hidden % n_vit != 0:
        raise ValueError(f"ViT qkv_hidden_size ({qkv_hidden}) must be divisible by num_heads ({n_vit})")
    head_size_vit = qkv_hidden // n_vit

    if tp_size > 1:
        if n_vit % tp_size != 0:
            raise ValueError(f"ViT num_heads ({n_vit}) must be divisible by tp_size ({tp_size})")
        if inter_vit % tp_size != 0:
            raise ValueError(f"ViT intermediate_size ({inter_vit}) must be divisible by tp_size ({tp_size})")

    # ViT always runs in bfloat16 regardless of LLM quantization settings
    vit_gemm_mode = common.GEMMQuantMode.bfloat16
    vit_fmha_mode = common.FMHAQuantMode.bfloat16

    result = [
        ops.ElementWise("encoder_add_norm_1", depth, 2 * h_vit, 2 * h_vit, 0.8),
        ops.GEMM(
            "encoder_qkv_gemm",
            depth,
            3 * qkv_hidden // tp_size,
            h_vit,
            vit_gemm_mode,
        ),
        ops.EncoderAttention(
            "encoder_attention",
            depth,
            n_vit // tp_size,
            head_size_vit,
            fmha_quant_mode=vit_fmha_mode,
            partial_rotary_factor=0.0,
        ),
        ops.GEMM(
            "encoder_proj_gemm",
            depth,
            h_vit,
            qkv_hidden // tp_size,
            vit_gemm_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce("encoder_ar_1", depth, h_vit, tp_size),
        ops.ElementWise("encoder_add_norm_2", depth, 2 * h_vit, 2 * h_vit, 0.8),
        # SwiGLU-style FFN (enc_cfg.gated_mlp) has a separate gate projection
        # (hidden -> intermediate) alongside the up projection below; plain
        # up/down FFNs omit it.
        *(
            [ops.GEMM("encoder_gate_gemm", depth, inter_vit // tp_size, h_vit, vit_gemm_mode)]
            if enc_cfg.gated_mlp
            else []
        ),
        ops.GEMM(
            "encoder_ffn1_gemm",
            depth,
            inter_vit // tp_size,
            h_vit,
            vit_gemm_mode,
        ),
        ops.ElementWise(
            "encoder_act",
            depth,
            inter_vit // tp_size,
            inter_vit // tp_size,
            0.8,
        ),
        ops.GEMM(
            "encoder_ffn2_gemm",
            depth,
            h_vit,
            inter_vit // tp_size,
            vit_gemm_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce("encoder_ar_2", depth, h_vit, tp_size),
    ]

    if enc_cfg.partial_rotary_factor > 0:
        # the attention op's internal RoPE term is disabled in exchange (partial_rotary_factor=0.0).
        # partial_rotary_factor gates the op but does not shrink it: the eager kernel
        # duplicates the half-dim cos/sin table to full head_dim and rotates all of Q/K.
        rope_dim = 6 * qkv_hidden // tp_size
        result.append(ops.ElementWise("encoder_rope_apply", depth, rope_dim, rope_dim, 0.8))

    if enc_cfg.final_norm:
        result.append(ops.ElementWise("encoder_final_norm", 1, h_vit, h_vit, 0.8))
    if enc_cfg.pool_temporal:
        # Kimi's sd2_tpool merger performs temporal mean pooling and spatial
        # 2x2 rearrangement before PatchMerger. This is a memory-bound layout
        # operation; projector GEMMs run on the pooled output-token geometry.
        # Sources: Transformers commit cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55
        # and vLLM commit d2906091bfc579cebefe3d8e8fb9077397ce9882.
        result.append(ops.ElementWise("encoder_patch_merge_pool", 1, h_vit, h_vit, 0.8))

    return result


def _projector_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int) -> list:
    """Build the projector MLP ops from enc_cfg.projector_dims.

    TP layout per layer:
      - Non-final layers: row-parallel (M = out // tp, K = in; output sharded) + activation
      - Final layer: column-parallel if P > 1 (M = out, K = in // tp; input sharded)
                     row-parallel if P == 1 (M = out // tp, K = in; full input)
      - Ends with a CustomAllReduce over the final output dimension unless
        projector_replicated=True (full dimensions and no projector collectives).

    Returns [] if projector_dims is empty.
    """
    dims = enc_cfg.projector_dims
    if not dims:
        return []

    n_inst = enc_cfg.projector_n_instances
    vit_gemm_mode = common.GEMMQuantMode.bfloat16
    n_layers = len(dims)
    # Kimi PatchMerger uses ReplicatedLinear for both projections, including
    # under encoder TP. Modified adaptation of vLLM (Apache-2.0), copyright
    # contributors to the vLLM project:
    # https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25_vit.py
    if enc_cfg.projector_replicated:
        tp_size = 1

    result = []
    # The final merger normalizes hidden_size before pixel shuffle. Deepstack
    # mergers normalize merger_dim after shuffle, but the inverse token-count
    # change preserves the same total hidden_size traffic per pre-merge patch.
    if enc_cfg.projector_pre_norm:
        result.append(
            ops.ElementWise(
                "encoder_merger_norm",
                n_inst,
                enc_cfg.hidden_size,
                enc_cfg.hidden_size,
                0.8,
            )
        )
    for i, (in_d, out_d) in enumerate(dims):
        is_last = i == n_layers - 1
        # Final layer in a multi-layer projector takes sharded input from the previous
        # row-parallel layer (column-parallel style). Single-layer and non-final layers
        # always receive a full (non-sharded) input (row-parallel style).
        col_parallel = is_last and n_layers > 1
        if col_parallel:
            m, k = out_d, in_d // tp_size
        else:
            m, k = out_d // tp_size, in_d
        result.append(ops.GEMM(f"encoder_projector_fc{i}_gemm", n_inst, m, k, vit_gemm_mode))
        if not is_last:
            result.append(
                ops.ElementWise(
                    f"encoder_projector_fc{i}_act",
                    n_inst,
                    out_d // tp_size,
                    out_d // tp_size,
                    0.8,
                )
            )

    if not enc_cfg.projector_replicated:
        result.append(ops.CustomAllReduce("encoder_projector_ar", n_inst, dims[-1][1], tp_size))
    if enc_cfg.projector_post_norm:
        result.append(ops.ElementWise("encoder_projector_post_norm", n_inst, dims[-1][1], dims[-1][1], 0.8))
    return result


def build_encoder_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int, enable_encoder_dp: bool = True) -> list:
    """Build the complete list of encoder ops for a ViT-based vision encoder.

    Combines the patch/position input ops, ViT transformer ops (10 ops x depth
    repetitions), and projector ops (one GEMM per layer, activations between
    layers, merger norm, and AllReduce; or 0 if no projector is configured).

    Args:
        enc_cfg: VisionEncoderConfig populated with ViT and projector parameters.
        tp_size: Worker tensor-parallel degree — the DP degree under encoder DP,
                 else the ViT weight-sharding degree (must evenly divide
                 num_heads and intermediate_size when tp_size > 1).
        enable_encoder_dp: Encoder data parallelism over the TP group (default
                 True) — see module docstring.

    Returns:
        Flat list of operation objects ready to assign to model.encoder_ops.
    """
    if not enable_encoder_dp:
        return _patch_embedding_ops(enc_cfg) + _vit_transformer_ops(enc_cfg, tp_size) + _projector_ops(enc_cfg, tp_size)

    # DP: full-replica ops (tp=1); the per-layer AllReduces degenerate to no-ops.
    result = _patch_embedding_ops(enc_cfg) + _vit_transformer_ops(enc_cfg, 1) + _projector_ops(enc_cfg, 1)
    if tp_size > 1:
        result.append(
            ops.NCCL(
                "encoder_dp_all_gather",
                1,
                "all_gather",
                num_elements_per_token=enc_cfg.out_hidden_size * enc_cfg.projector_n_instances * tp_size,
                num_gpus=tp_size,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    return result


def _gemma4_vision_transformer_ops(enc_cfg: common.Gemma4VisionEncoderConfig, tp_size: int) -> list:
    """Build the Gemma 4 vision tower without Qwen3-VL merger assumptions.

    The graph follows ``Gemma4VisionModel``: learned patch + 2-D position
    embeddings, 27 non-causal transformer blocks with Q/K/V normalization,
    full x/y RoPE and a gated MLP, followed by position-aware average pooling,
    standardization, RMS normalization, and one language-space projection.

    Sources: Hugging Face Transformers commit
    cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55 and vLLM commit
    d2906091bfc579cebefe3d8e8fb9077397ce9882.
    """
    depth = enc_cfg.depth
    h_vit = enc_cfg.hidden_size
    n_vit = enc_cfg.num_heads
    n_kv = enc_cfg.num_key_value_heads
    head_dim = enc_cfg.head_dim
    inter_vit = enc_cfg.intermediate_size
    pool = enc_cfg.pooling_kernel_size

    if min(depth, h_vit, n_vit, n_kv, head_dim, inter_vit, enc_cfg.patch_size, pool) <= 0:
        raise ValueError("Gemma 4 vision encoder dimensions must all be positive")
    if n_vit != n_kv:
        raise ValueError("Gemma 4 vision encoder requires full MHA (num_heads == num_key_value_heads)")
    if n_vit * head_dim != h_vit:
        raise ValueError("Gemma 4 vision num_heads * head_dim must equal hidden_size")
    for field, value in (
        ("num_heads", n_vit),
        ("num_key_value_heads", n_kv),
        ("intermediate_size", inter_vit),
    ):
        if value % tp_size != 0:
            raise ValueError(f"Gemma 4 vision {field} ({value}) must be divisible by tp_size ({tp_size})")

    vit_gemm_mode = common.GEMMQuantMode.bfloat16
    vit_fmha_mode = common.FMHAQuantMode.bfloat16
    qkv_width = (n_vit + 2 * n_kv) * head_dim // tp_size
    attn_width = n_vit * head_dim // tp_size
    inter_per_tp = inter_vit // tp_size

    result = [
        # Pixel patches are already flattened by the processor to 3 * patch².
        ops.GEMM(
            "encoder_patch_embed_gemm",
            1,
            h_vit,
            3 * enc_cfg.patch_size**2,
            vit_gemm_mode,
        ),
        # Two independent learned tables (x and y).  A scale factor of two
        # accounts for both lookups in latency and resident weights.
        ops.Embedding(
            "encoder_position_embedding",
            2,
            enc_cfg.position_embedding_size,
            h_vit,
            0.3,
        ),
        ops.ElementWise("encoder_patch_embed_add", 1, 3 * h_vit, h_vit, 0.8),
        ops.ElementWise("encoder_input_norm", depth, h_vit, h_vit, 0.8),
        ops.GEMM("encoder_qkv_gemm", depth, qkv_width, h_vit, vit_gemm_mode),
        # Per-head Q/K/V RMSNorm plus full two-dimensional RoPE on Q and K.
        # RoPE is explicit here rather than the Qwen-specific partial-RoPE
        # path inside EncoderAttention.
        ops.ElementWise("encoder_qkv_norm_rope_2d", depth, 3 * attn_width, 3 * attn_width, 0.8),
        ops.EncoderAttention(
            "encoder_attention",
            depth,
            n_vit // tp_size,
            head_dim,
            fmha_quant_mode=vit_fmha_mode,
            partial_rotary_factor=0.0,
        ),
        ops.GEMM(
            "encoder_proj_gemm",
            depth,
            h_vit,
            attn_width,
            vit_gemm_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce("encoder_ar_1", depth, h_vit, tp_size),
        ops.ElementWise("encoder_post_attn_norm_residual", depth, 3 * h_vit, h_vit, 0.8),
        ops.ElementWise("encoder_pre_ffn_norm", depth, h_vit, h_vit, 0.8),
        # Gemma 4's vision MLP is gated: separate gate/up projections are
        # represented as one fused GEMM with a 2*intermediate output.
        ops.GEMM("encoder_ffn_gate_up_gemm", depth, 2 * inter_per_tp, h_vit, vit_gemm_mode),
        ops.ElementWise("encoder_ffn_act_mul", depth, 2 * inter_per_tp, inter_per_tp, 0.8),
        ops.GEMM(
            "encoder_ffn_down_gemm",
            depth,
            h_vit,
            inter_per_tp,
            vit_gemm_mode,
            low_precision_input=True,
        ),
        ops.CustomAllReduce("encoder_ar_2", depth, h_vit, tp_size),
        ops.ElementWise("encoder_post_ffn_norm_residual", depth, 3 * h_vit, h_vit, 0.8),
        # Query x is the pre-pooling patch count. scale_num_tokens converts it
        # to pooled tokens while dim_in accounts for all pool² source patches.
        ops.ElementWise(
            "encoder_gemma4_pool_avg",
            1,
            pool**2 * h_vit,
            h_vit,
            0.8,
            scale_num_tokens=pool**2,
        ),
        # sqrt(hidden) scaling and optional checkpoint standardization execute
        # on the pooled soft-token stream.
        ops.ElementWise(
            "encoder_gemma4_pool_postprocess",
            1,
            (3 if enc_cfg.standardize else 1) * h_vit,
            h_vit,
            0.8,
        ),
        ops.ElementWise("encoder_projector_pre_norm", 1, h_vit, h_vit, 0.8),
        # Gemma4MultimodalEmbedder uses a ReplicatedLinear in vLLM and is
        # absent from the Hugging Face vision TP plan.  Keep the complete
        # 1152 -> language-hidden projection on every rank; there is no Qwen
        # PatchMerger-style projector sharding or projector AllReduce.
        ops.GEMM(
            "encoder_projector_fc0_gemm",
            1,
            enc_cfg.out_hidden_size,
            h_vit,
            vit_gemm_mode,
        ),
    ]
    return result


def build_gemma4_vision_encoder_ops(
    enc_cfg: common.Gemma4VisionEncoderConfig,
    tp_size: int,
    enable_encoder_dp: bool = True,
) -> list:
    """Build Gemma 4 vision ops under encoder-DP or legacy encoder-TP.

    Encoder-DP keeps one complete vision tower per TP rank, shards whole images
    across those replicas, then all-gathers the projected soft-token embeddings.
    Encoder-TP shards attention and gated-MLP weights with per-block
    all-reduces.  Gemma's language projection remains replicated, matching the
    engine contract.
    """
    if not isinstance(enc_cfg, common.Gemma4VisionEncoderConfig):
        raise TypeError(
            f"build_gemma4_vision_encoder_ops requires Gemma4VisionEncoderConfig, got {type(enc_cfg).__name__}"
        )

    if not enable_encoder_dp:
        return _gemma4_vision_transformer_ops(enc_cfg, tp_size)

    result = _gemma4_vision_transformer_ops(enc_cfg, 1)
    if tp_size > 1:
        result.append(
            ops.NCCL(
                "encoder_dp_all_gather",
                1,
                "all_gather",
                num_elements_per_token=enc_cfg.out_hidden_size * tp_size,
                num_gpus=tp_size,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    return result


@dataclasses.dataclass
class EncoderOnlyModel:
    """Vision-encoder-only model for a disaggregated encode (EPD) worker.

    Mirrors an encoder-only instance (e.g. SGLang ``--encoder-only``); only
    ViT-side rules govern its tensor parallelism.  Duck-types the slice of
    ``BaseModel`` the encoder phase reads: ``encoder_ops``,
    ``encoder_config`` and ``config``.  The empty LM op lists satisfy the
    compiled engine's spec-build contract (read unconditionally before any
    ad-hoc op-list evaluation).
    """

    encoder_ops: list
    encoder_config: common.VisionEncoderConfig
    config: object
    context_ops: list = dataclasses.field(default_factory=list)
    generation_ops: list = dataclasses.field(default_factory=list)


def _llama4_projector_ops(enc_cfg: common.VisionEncoderConfig, tp_size: int) -> list:
    """Build Llama 4's pixel-shuffle adaptor and multimodal connector.

    The checkpoint contains two distinct modules and therefore two distinct
    communication boundaries:

    * ``vision_adapter.mlp`` is column-parallel then row-parallel.  Its second
      GEMM reduces the 4096-wide output across TP ranks.
    * ``multi_modal_projector.linear_1`` is column-parallel with gathered
      output, projecting the 4096-wide vision embedding to the 5120-wide text
      embedding consumed by the hybrid-MoE backbone.
    """
    dims = enc_cfg.projector_dims
    if len(dims) != 3:
        raise ValueError(f"Llama 4 expects exactly three projector dimensions, got {dims!r}")
    (merge_in, adapter_hidden), (adapter_in, adapter_out), (connector_in, connector_out) = dims
    if adapter_hidden != adapter_in or adapter_out != connector_in:
        raise ValueError(f"Llama 4 projector dimensions are not composable: {dims!r}")
    for dim_name, dim in (
        ("adapter_hidden", adapter_hidden),
        ("adapter_in", adapter_in),
        ("connector_out", connector_out),
    ):
        if dim % tp_size != 0:
            raise ValueError(f"Llama 4 {dim_name} ({dim}) must be divisible by tp_size ({tp_size})")

    q = common.GEMMQuantMode.bfloat16
    result = [
        # pixel_shuffle turns four 1408-wide patch vectors into one 5632-wide
        # vector before the MLP; it is a real memory-movement stage even though
        # it has no trainable weights.
        ops.ElementWise("encoder_projector_pixel_shuffle", 1, merge_in, merge_in, 0.8),
        # vLLM ColumnParallelLinear: output features are sharded.
        ops.GEMM("encoder_projector_adapter_fc0_gemm", 1, adapter_hidden // tp_size, merge_in, q),
        ops.ElementWise(
            "encoder_projector_adapter_fc0_act",
            1,
            adapter_hidden // tp_size,
            adapter_hidden // tp_size,
            0.8,
        ),
        # vLLM RowParallelLinear: input features are sharded; output is reduced.
        ops.GEMM("encoder_projector_adapter_fc1_gemm", 1, adapter_out, adapter_in // tp_size, q),
        ops.CustomAllReduce("encoder_projector_adapter_ar", 1, adapter_out, tp_size),
        ops.ElementWise("encoder_projector_adapter_fc1_act", 1, adapter_out, adapter_out, 0.8),
        # The final connector is ColumnParallelLinear(gather_output=True).
        ops.GEMM("encoder_projector_mm_gemm", 1, connector_out // tp_size, connector_in, q),
    ]
    if tp_size > 1:
        result.append(
            ops.NCCL(
                "encoder_projector_mm_all_gather",
                1,
                "all_gather",
                num_elements_per_token=connector_out,
                num_gpus=tp_size,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    return result


def build_llama4_encoder_ops(
    enc_cfg: common.VisionEncoderConfig,
    tp_size: int,
    enable_encoder_dp: bool = True,
) -> list:
    """Build the checkpoint-faithful Llama 4 image-tower operation graph.

    Unlike Qwen3-VL's shared ViT path, Llama 4 has a learned patch-embedding
    linear, a per-tile CLS token, a pixel-shuffle adaptor, and a separate
    vision-to-text connector.  The backend supplies their different sequence
    lengths: raw patches for patch embedding, raw patches + CLS for the ViT,
    and post-shuffle image tokens for adaptor/connector operations.

    The tensor-parallel sharding topology is adapted (modified) from vLLM v0.8.5, commit
    ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d, model_executor/models/mllama4.py.
    Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
    All rights reserved. Licensed under the Apache License, Version 2.0.
    """
    if enc_cfg.image_size <= 0:
        raise ValueError("Llama 4 vision_config.image_size must be positive")
    if not enc_cfg.has_cls_token:
        raise ValueError("Llama 4 vision encoder requires a CLS token")
    if enc_cfg.image_size % enc_cfg.patch_size != 0:
        raise ValueError(
            f"Llama 4 image_size ({enc_cfg.image_size}) must be divisible by patch_size ({enc_cfg.patch_size})"
        )
    merge_stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
    if enc_cfg.image_size % merge_stride != 0:
        raise ValueError(
            f"Llama 4 image_size ({enc_cfg.image_size}) must be divisible by the merged-patch stride ({merge_stride})"
        )

    encoder_tp = 1 if enable_encoder_dp else tp_size
    if enc_cfg.hidden_size % encoder_tp != 0:
        raise ValueError(
            f"Llama 4 vision hidden_size ({enc_cfg.hidden_size}) must be divisible by tp_size ({encoder_tp})"
        )
    patch_input = enc_cfg.in_channels * enc_cfg.patch_size**2
    q = common.GEMMQuantMode.bfloat16
    result = [
        ops.GEMM("encoder_patch_embedding_gemm", 1, enc_cfg.hidden_size // encoder_tp, patch_input, q),
    ]
    if encoder_tp > 1:
        # vLLM's patch-embedding ColumnParallelLinear uses gather_output=True,
        # so every TP rank sees the full hidden vector before the ViT blocks.
        result.append(
            ops.NCCL(
                "encoder_patch_embedding_all_gather",
                1,
                "all_gather",
                num_elements_per_token=enc_cfg.hidden_size,
                num_gpus=encoder_tp,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    result.extend(
        [
            # Class/position additions plus the pre-transformer LayerNorm.
            ops.ElementWise(
                "encoder_class_position_norm",
                1,
                3 * enc_cfg.hidden_size,
                3 * enc_cfg.hidden_size,
                0.8,
            ),
            *_vit_transformer_ops(enc_cfg, encoder_tp),
            ops.ElementWise("encoder_post_norm", 1, 2 * enc_cfg.hidden_size, 2 * enc_cfg.hidden_size, 0.8),
            *_llama4_projector_ops(enc_cfg, encoder_tp),
        ]
    )

    if enable_encoder_dp and tp_size > 1:
        result.append(
            ops.NCCL(
                "encoder_dp_all_gather",
                1,
                "all_gather",
                num_elements_per_token=enc_cfg.out_hidden_size * tp_size,
                num_gpus=tp_size,
                comm_quant_mode=common.CommQuantMode.half,
            )
        )
    return result


def build_kimi_k3_encoder_ops(
    enc_cfg: common.VisionEncoderConfig,
    tp_size: int,
    enable_encoder_dp: bool = True,
) -> list:
    """Build the complete MoonViT3D + PatchMergerV2 graph for Kimi K3."""
    if enc_cfg.encoder_type != "kimi_k3_moonvit3d_patchmergerv2":
        raise ValueError(f"Expected Kimi K3 encoder config, got {enc_cfg.encoder_type!r}")
    if not (enc_cfg.final_norm and enc_cfg.pool_temporal and enc_cfg.projector_post_norm):
        raise ValueError("Kimi K3 requires complete MoonViT3D and PatchMergerV2 operation semantics")
    if not enc_cfg.projector_replicated or enc_cfg.projector_pre_norm:
        raise ValueError("Kimi K3 PatchMergerV2 requires replicated projections and only a final normalization")
    if enc_cfg.video_attention_type != "spatial_temporal":
        raise ValueError("Kimi K3 requires spatial_temporal vision attention")
    if enc_cfg.qkv_hidden_size <= 0 or enc_cfg.max_temporal_patches <= 0:
        raise ValueError("Kimi K3 requires explicit QKV and temporal-position geometry")
    return build_encoder_ops(enc_cfg, tp_size, enable_encoder_dp)
