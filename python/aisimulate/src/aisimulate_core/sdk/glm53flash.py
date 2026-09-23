# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text geometry, adapted from the pinned Z.AI MIT configuration.

Source: zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a,
config.json. Copyright (c) 2026 Z.AI Co., Ltd; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

MODEL_REVISIONS = {
    "zai-org/GLM-5.3-Flash": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    "nvidia/GLM-5.3-Flash-NVFP4": "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
}
BACKEND_REVISIONS = {
    "vllm": "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
    "sglang": "94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
}


@dataclass(frozen=True)
class Glm53FlashConfig:
    """Native layer schedule and shape, shared by SOL, FPM and Ops producers."""

    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    hidden_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    linear_num_heads: int
    linear_head_dim: int
    conv_kernel: int
    gate_lower_bound: float
    hc_mult: int
    hc_sinkhorn_iters: int
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    swiglu_limit: float

    @classmethod
    def from_text_config(cls, config: dict) -> Glm53FlashConfig:
        linear = config["linear_attn_config"]
        values = {key: config[key] for key in cls.__dataclass_fields__ if key in config}
        values.update(
            layer_types=tuple(config["layer_types"]),
            mlp_layer_types=tuple(config["mlp_layer_types"]),
            linear_num_heads=linear["num_heads"],
            linear_head_dim=linear["head_dim"],
            conv_kernel=linear["short_conv_kernel_size"],
            gate_lower_bound=linear["gate_lower_bound"],
        )
        result = cls(**values)
        if len(result.layer_types) != config["num_hidden_layers"] or len(result.mlp_layer_types) != len(
            result.layer_types
        ):
            raise ValueError("GLM-5.3-Flash requires one attention and FFN type per layer")
        if set(result.layer_types) - {"linear_attention", "deepseek_sparse_attention"}:
            raise ValueError("unknown GLM-5.3-Flash attention layer type")
        if set(result.mlp_layer_types) - {"dense", "sparse"}:
            raise ValueError("unknown GLM-5.3-Flash FFN layer type")
        if tuple(i for i, kind in enumerate(result.layer_types) if kind == "linear_attention") != tuple(
            linear["kda_layers"]
        ):
            raise ValueError("GLM-5.3-Flash KDA layer indices disagree with layer_types")
        if tuple(i for i, kind in enumerate(result.layer_types) if kind == "deepseek_sparse_attention") != tuple(
            linear["full_attn_layers"]
        ):
            raise ValueError("GLM-5.3-Flash sparse layer indices disagree with layer_types")
        if config["qk_rope_head_dim"] != 0 or not config["mla_use_nope"]:
            raise ValueError("GLM-5.3-Flash baseline requires NoPE sparse MLA")
        if not config["index_kpool_compress"] or not config["index_kpool_always_select_tail"]:
            raise ValueError("GLM-5.3-Flash requires compressed IndexPool and retained tail")
        if config["moe_router_dtype"] != "float32" or config["scoring_func"] != "sigmoid":
            raise ValueError("GLM-5.3-Flash requires the native FP32 sigmoid router")
        if not config["mhc"] or result.gate_lower_bound != -5.0 or result.conv_kernel != 4:
            raise ValueError("GLM-5.3-Flash requires the pinned mHC and KDA contract")
        return result

    def to_dict(self) -> dict:
        return asdict(self)
