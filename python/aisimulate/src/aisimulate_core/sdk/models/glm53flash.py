# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text-only, pure TP graph.

Architecture adapted from Z.AI MIT configuration eb9eb208eb0d988989d07a6a12d0fdeb5f52574a.
Execution grouping follows the pinned vLLM/SGLang files listed in
model_configs/GLM53FLASH_PROVENANCE.md and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json

import aisimulate_core._native as core
import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk import common
from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS, Glm53FlashConfig
from aisimulate_core.sdk.models.base import BaseModel, register_model


def _native(kind: str, **values):
    return core.op_from_spec_json(json.dumps({kind: values}))


@register_model("GLM53FLASH")
class Glm53FlashModel(BaseModel):
    """34 KDA + 11 NoPE sparse MLA layers; MTP and vision are outside this graph."""

    @classmethod
    def create(cls, info: dict, model_config, backend_name: str) -> BaseModel:
        return cls(info, model_config, backend_name)

    @property
    def activation_hidden_size(self) -> int:
        return self._hidden_size

    def __init__(self, info: dict, model_config, backend_name: str):
        super().__init__(
            info["model_path"],
            info["model_family"],
            info["architecture"],
            info["layers"],
            info["n"],
            info["n_kv"],
            info["d"],
            info["hidden_size"],
            info["inter_size"],
            info["vocab"],
            info["context"],
            model_config,
            info["extra_params"],
        )
        d = self.extra_params
        if not isinstance(d, Glm53FlashConfig):
            raise TypeError("GLM-5.3-Flash requires its shared text descriptor")
        if backend_name not in {"vllm", "sglang"}:
            raise NotImplementedError("GLM-5.3-Flash currently supports vLLM and SGLang")
        tp = model_config.tp_size
        mtp, ep = model_config.resolve_moe_parallelism()
        if (
            tp not in {1, 2, 4}
            or mtp != tp
            or ep != 1
            or model_config.pp_size != 1
            or model_config.cp_size != 1
            or model_config.attention_dp_size != 1
        ):
            raise NotImplementedError("GLM-5.3-Flash baseline requires TP1/2/4, MoE TP=TP, DP=PP=CP=EP=1")
        if self._nextn or model_config.decoder_replay or self._num_layers != len(d.layer_types):
            raise NotImplementedError(
                "GLM-5.3-Flash text baseline requires all45 layers, nextn=0 and no decoder replay"
            )
        if model_config.moe_backend in {"megamoe", "wideep"}:
            raise NotImplementedError("GLM-5.3-Flash baseline uses tensor-parallel local experts")
        self.raw_config = info["raw_config"]
        self.text_config = self.raw_config["text_config"]
        self.execution_profile = "full"
        self.input_modality = "text"
        self.model_revision = MODEL_REVISIONS.get(info["model_path"], "")
        quant = self.raw_config["quantization_config"]
        self.checkpoint_format = "nvfp4" if quant.get("quant_algo") == "NVFP4" else "fp8"
        expected = self.checkpoint_format if self.checkpoint_format == "nvfp4" else "fp8_block"
        if model_config.gemm_quant_mode.name != expected or model_config.moe_quant_mode.name != expected:
            raise NotImplementedError("GLM-5.3-Flash requires its checkpoint's native FP8/NVFP4 precision partition")
        if model_config.kvcache_quant_mode.name != "fp8":
            raise NotImplementedError("GLM-5.3-Flash formal baseline requires FP8 sparse KV and FP32 KDA state")
        h = d.hidden_size
        self._cache_specs = []
        identity = dict(backend=backend_name, checkpoint_format=self.checkpoint_format)

        def mhc(name, role):
            return _native(
                "Glm53Mhc",
                name=name,
                role=role,
                hidden_size=h,
                hc_mult=d.hc_mult,
                sinkhorn_iters=d.hc_sinkhorn_iters,
                **identity,
            )

        def attention(layer, context):
            kda = d.layer_types[layer] == "linear_attention"
            spec = dict(
                name=f"attention_{layer}",
                is_context=context,
                layer_kind="kda" if kda else "sparse_mla",
                hidden_size=h,
                tp_size=tp,
                num_heads=(d.linear_num_heads if kda else d.num_attention_heads) // tp,
                head_dim=d.linear_head_dim if kda else d.qk_nope_head_dim,
                q_lora_rank=d.q_lora_rank,
                kv_lora_rank=d.kv_lora_rank,
                value_head_dim=d.v_head_dim,
                index_n_heads=d.index_n_heads,
                index_head_dim=d.index_head_dim,
                index_topk=d.index_topk,
                index_pool=d.index_kpool,
                conv_kernel=d.conv_kernel,
                gate_lower_bound=d.gate_lower_bound,
                projection_quant_mode="bfloat16"
                if kda or backend_name == "vllm" or self.checkpoint_format == "nvfp4"
                else "fp8_block",
                kv_cache_dtype="fp8",
                **identity,
            )
            if context:
                self._cache_specs.append(json.dumps(spec))
            return _native("Glm53Attention", **spec)

        def allreduce(name):
            return ops.NCCL(name, 1, "all_reduce", h, tp, common.CommQuantMode.half)

        def mlp(layer, phase):
            dense = d.mlp_layer_types[layer] == "dense"
            width = (d.intermediate_size if dense else d.moe_intermediate_size * d.n_shared_experts) // tp
            # The NVFP4 checkpoint excludes shared experts. The FP8 checkpoint
            # quantizes them; attention exceptions are handled independently.
            q = (
                model_config.gemm_quant_mode
                if dense or self.checkpoint_format == "fp8"
                else common.GEMMQuantMode.bfloat16
            )
            label = "dense" if dense else "shared"
            result = [
                ops.GEMM(f"{phase}_{label}_gate_up_{layer}", 1, 2 * width, h, q),
                ops.ElementWise(f"{phase}_{label}_swiglu_clamp10_{layer}", 1, 2 * width, width, 0.8),
                ops.GEMM(f"{phase}_{label}_down_{layer}", 1, h, width, q),
            ]
            if not dense:
                result += [
                    _native(
                        "Glm53Router",
                        name=f"router_{layer}",
                        hidden_size=h,
                        num_experts=d.n_routed_experts,
                        topk=d.num_experts_per_tok,
                        **identity,
                    ),
                    ops.MoE(
                        f"{phase}_moe_{layer}",
                        1,
                        h,
                        d.moe_intermediate_size,
                        d.num_experts_per_tok,
                        d.n_routed_experts,
                        tp,
                        1,
                        model_config.moe_quant_mode,
                        "power_law_1.01"
                        if model_config.workload_distribution == "power_law"
                        else model_config.workload_distribution,
                        1,
                    ),
                ]
            module = _native(
                "Glm53Ffn",
                name=f"ffn_{layer}",
                is_context=phase == "context",
                is_dense=dense,
                hidden_size=h,
                intermediate_size=d.intermediate_size if dense else d.moe_intermediate_size,
                num_experts=d.n_routed_experts,
                topk=d.num_experts_per_tok,
                tp_size=tp,
                n_shared_experts=d.n_shared_experts,
                swiglu_limit=d.swiglu_limit,
                scoring_func=d.scoring_func,
                routed_scaling_factor=d.routed_scaling_factor,
                n_group=d.n_group,
                topk_group=d.topk_group,
                norm_topk_prob=d.norm_topk_prob,
                gemm_quant_mode=model_config.gemm_quant_mode.name,
                shared_quant_mode=(
                    model_config.gemm_quant_mode if self.checkpoint_format == "fp8" else common.GEMMQuantMode.bfloat16
                ).name,
                moe_quant_mode=model_config.moe_quant_mode.name,
                children=[json.loads(op._spec_json()) for op in result],
                **identity,
            )
            return [module, allreduce(f"{phase}_ffn_allreduce_{layer}")]

        for context in (True, False):
            phase = "context" if context else "generation"
            target = self.context_ops if context else self.generation_ops
            target += [
                ops.Embedding(f"{phase}_embedding", 1, self._vocab_size // tp, h, 0.3),
                allreduce(f"{phase}_embedding_allreduce"),
                mhc("mhc_expand", "expand"),
            ]
            for layer in range(len(d.layer_types)):
                if backend_name == "vllm":
                    target.append(
                        mhc("mhc_pre_attn_0", "pre") if layer == 0 else mhc(f"mhc_fused_attn_{layer}", "fused_post_pre")
                    )
                else:
                    target.append(mhc(f"mhc_pre_attn_{layer}", "pre"))
                target += [attention(layer, context), allreduce(f"{phase}_attention_allreduce_{layer}")]
                if backend_name == "vllm":
                    target.append(mhc(f"mhc_fused_ffn_{layer}", "fused_post_pre"))
                else:
                    target += [mhc(f"mhc_post_attn_{layer}", "post"), mhc(f"mhc_pre_ffn_{layer}", "pre")]
                target += mlp(layer, phase)
                if backend_name == "sglang" or layer == len(d.layer_types) - 1:
                    target.append(mhc(f"mhc_post_ffn_{layer}", "post"))
            target += [
                mhc("mhc_contract", "contract"),
                ops.ElementWise(f"{phase}_final_norm", 1, h, h, 0.8),
                ops.GEMM(f"{phase}_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
            ]
        self._resident_weight_bytes = float(sum(op.get_weights() for op in self.context_ops))
        # Generic GEMM/MoE FP8 weight inventory omits block scales; NVFP4's
        # mapping already includes its one FP8 scale per16 packed weights.
        if self.checkpoint_format == "fp8":
            dense_elements = d.mlp_layer_types.count("dense") * 3 * h * d.intermediate_size // tp
            moe_elements = (
                d.mlp_layer_types.count("sparse")
                * 3
                * h
                * d.moe_intermediate_size
                * (d.n_routed_experts + d.n_shared_experts)
                // tp
            )
            self._resident_weight_bytes += (dense_elements + moe_elements) * 4 / (128 * 128)
        # Final RMSNorm is represented by a memory-only ElementWise op.
        self._resident_weight_bytes += h * 2 + d.mlp_layer_types.count("sparse") * d.n_routed_experts * 4

    def get_resident_weights_bytes(self) -> float:
        return self._resident_weight_bytes

    def get_additional_activation_bytes(self, num_tokens: int) -> float:
        d = self.extra_params
        # Reusable expanded BF16 residual buffers and FP32 mHC coefficients.
        # Native paged-cache/graph/kernel workspace still needs GPU admission.
        return float(num_tokens * (4 * d.hc_mult * d.hidden_size + 8 * (d.hc_mult + 2) * d.hc_mult))

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        return sum(core.glm53_cache_bytes(spec, seq_len) for spec in self._cache_specs)

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)

    def get_kvcache_batch_capacity(self, kv_budget_bytes: float, max_batch_size: int) -> int:
        # Reserve one fixed recurrent/tail state for every active scheduler
        # slot before budgeting the global token pool's conservative slope.
        fixed = self.get_kvcache_bytes_per_sequence(0)
        slope = (self.get_kvcache_bytes_per_sequence(4) - fixed) / 4
        return max(0, int((kv_budget_bytes - max_batch_size * fixed) // slope))
