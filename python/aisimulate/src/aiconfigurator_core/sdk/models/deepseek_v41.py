# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 text AR, with independently scoped prefill decoder stages.

Architecture source: deepseek-ai/DeepSeek-V4.1-Flash at
fb2764a5cf321eaa5070ca8f9e892818f477c16d. See sdk/deepseek_v41.py and
THIRD_PARTY_NOTICES.md. Vision and DSpark are deliberately separate capabilities.
"""

from __future__ import annotations

import json

import aiconfigurator_core._aiconfigurator_core as core
import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.deepseek_v41 import DeepSeekV41Config, resolve_execution_profile
from aiconfigurator_core.sdk.models.base import BaseModel, register_model


def _native(kind: str, **values):
    return core.op_from_spec_json(json.dumps({kind: values}))


def _spec(op):
    return json.loads(op._spec_json())


@register_model("DEEPSEEKV41")
class DeepSeekV41Model(BaseModel):
    """40-layer text backbone; resident weights do not depend on replay mode."""

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
        if not isinstance(d, DeepSeekV41Config):
            raise TypeError("DeepSeekV41Model requires the shared DeepSeekV41Config descriptor")
        if self._nextn:
            raise NotImplementedError(
                "DeepSeek-V4.1 text AR requires nextn=0; DSpark has a separate execution contract"
            )
        if self._num_layers != d.num_hidden_layers:
            raise ValueError("DeepSeek-V4.1 layer overrides cannot preserve cross-layer KV ownership")
        if model_config.moe_backend == "megamoe":
            raise NotImplementedError(
                "DeepSeek-V4.1 SOL/Hybrid uses decomposed MoE; MegaMoE requires separate measurement"
            )
        if model_config.attention_dp_size != 1 or model_config.pp_size != 1:
            raise NotImplementedError(
                "DeepSeek-V4.1 text baseline requires attention_dp_size=1 and pp_size=1; "
                "DP Engram collectives and PP cache ownership need separate contracts"
            )
        tp = model_config.tp_size
        mtp, ep = model_config.resolve_moe_parallelism()
        if tp * model_config.attention_dp_size != mtp * ep:
            raise ValueError("attention TP * DP must equal MoE TP * EP for DeepSeek-V4.1")
        if d.n_routed_experts % ep or d.index_n_heads % tp or d.o_groups % tp:
            raise ValueError("DeepSeek-V4.1 EP must divide experts; TP must divide index heads and output groups")
        self._topk, self._num_experts, self._moe_inter_size = info["topk"], info["num_experts"], info["moe_inter_size"]
        self.raw_config = info["raw_config"]
        self.text_config = self.raw_config["text_config"]
        self.execution_profile = resolve_execution_profile(model_config.decoder_replay, backend_name).value
        self.engram_residency = "hbm_tp_sharded"
        h = self._hidden_size
        distribution = (
            "power_law_1.01"
            if model_config.workload_distribution == "power_law"
            else model_config.workload_distribution
        )

        def attention(layer: int, context: bool):
            return _native(
                "Dsv41Attention",
                name="context_attention" if context else "generation_attention",
                is_context=context,
                role=d.layer_role(layer),
                compress_ratio=d.compress_ratios[layer],
                hidden_size=h,
                num_heads=d.num_attention_heads // tp,
                head_dim=d.head_dim,
                q_lora_rank=d.q_lora_rank,
                o_lora_rank=d.o_lora_rank,
                o_groups=max(1, d.o_groups // tp),
                index_n_heads=d.index_n_heads // tp,
                index_head_dim=d.index_head_dim,
                index_topk=d.index_topk,
                window_size=d.sliding_window,
                candidate_limit=d.candidate_topk_blocks * d.candidate_block_size
                if layer > d.candidate_source_layer_id
                else 0,
                is_candidate_source=layer == d.candidate_source_layer_id,
                bounded_prefill=context and model_config.decoder_replay and layer >= d.decoder_start_layer,
                gemm_quant_mode=model_config.gemm_quant_mode.name,
                fmha_quant_mode=model_config.fmha_quant_mode.name,
            )

        def stage(layer: int, context: bool):
            phase = "context" if context else "generation"
            children = []
            if layer in d.engram_layer_ids:
                engram_index = d.engram_layer_ids.index(layer)
                children.append(
                    _native(
                        "Dsv41Engram",
                        name=f"{phase}_engram",
                        num_embeddings=d.engram_num_embeddings[engram_index],
                        head_dim=d.engram_head_dim,
                        hash_columns=(d.engram_max_ngram_size - 1) * d.engram_n_heads,
                        hidden_size=h,
                        hc_mult=d.hc_mult,
                        tp_size=tp,
                    )
                )
                children.append(
                    ops.NCCL(
                        f"{phase}_engram_allreduce",
                        1,
                        "all_reduce",
                        (d.engram_max_ngram_size - 1) * d.engram_n_heads * d.engram_head_dim,
                        tp,
                        common.CommQuantMode.half,
                    )
                )
            children.extend(
                [
                    _native(
                        "Dsv41Mhc",
                        name=f"{phase}_mhc",
                        hidden_size=h,
                        hc_mult=d.hc_mult,
                        sinkhorn_iters=d.hc_sinkhorn_iters,
                    ),
                    ops.ElementWise(f"{phase}_attn_norm", 1, h, h, 0.8),
                    attention(layer, context),
                    ops.NCCL(f"{phase}_attention_allreduce", 1, "all_reduce", h, tp, common.CommQuantMode.half),
                    ops.ElementWise(f"{phase}_ffn_norm", 1, h, h, 0.8),
                ]
            )
            local_inter = self._moe_inter_size * d.n_shared_experts // tp
            shared = [
                _native(
                    "Dsv41Linear",
                    name=f"{phase}_shared_gate_up_gemm",
                    n=2 * local_inter,
                    k=h,
                    quant_mode=model_config.gemm_quant_mode.name,
                ),
                ops.ElementWise(f"{phase}_shared_act_gate", 1, 2 * local_inter, local_inter, 0.8),
                _native(
                    "Dsv41Linear",
                    name=f"{phase}_shared_ffn2_gemm",
                    n=h,
                    k=local_inter,
                    quant_mode=model_config.gemm_quant_mode.name,
                ),
            ]
            routed = [ops.GEMM(f"{phase}_router_gemm", 1, self._num_experts, h, common.GEMMQuantMode.bfloat16)]
            for pre in (True, False):
                routed.append(
                    ops.MoEDispatch(
                        f"{phase}_moe_{'pre' if pre else 'post'}_dispatch",
                        1,
                        h,
                        self._topk,
                        self._num_experts,
                        mtp,
                        ep,
                        model_config.attention_dp_size,
                        pre,
                        quant_mode=model_config.moe_quant_mode,
                        backend=backend_name,
                        is_context=context,
                        attn_ar_modeled=True,
                    )
                )
                if pre:
                    routed.append(
                        ops.MoE(
                            f"{phase}_moe",
                            1,
                            h,
                            self._moe_inter_size,
                            self._topk,
                            self._num_experts,
                            mtp,
                            ep,
                            model_config.moe_quant_mode,
                            distribution,
                            model_config.attention_dp_size,
                        )
                    )
            # The post-expert reduction consumes both routed and shared
            # partials, and therefore follows the optional compute overlap.
            combine = routed.pop()
            if ep == 1:
                # The text TP baseline uses an explicit NCCL collective,
                # consistently with attention/Engram and its measured comm
                # table. Serving validation disables custom all-reduce.
                combine = ops.NCCL(f"{phase}_moe_post_dispatch", 1, "all_reduce", h, mtp, common.CommQuantMode.half)
            # SGLang's qualified TP eager path executes forward_normal on
            # one stream. Dual-stream shared/routed work requires capture or
            # graph/SBO dispatch (sglang@1aa0e962 deepseek_v2.py:885-960,
            # 1107-1126,1191-1222). Other backend/EP modes remain assumptions.
            if context or (ep == 1 and backend_name == "sglang"):
                children.extend(shared + routed)
            else:
                children.append(ops.OverlapOp(f"{phase}_moe_overlap", group_a=routed, group_b=shared))
            children.append(combine)
            return _native(
                "Dsv41Stage",
                name=f"{phase}_v41_layer_{layer}",
                is_context=context,
                decoder_replay=model_config.decoder_replay,
                bounded=layer >= d.decoder_start_layer,
                window_size=d.sliding_window,
                children=[_spec(op) for op in children],
            )

        for context in (True, False):
            phase = "context" if context else "generation"
            target = self.context_ops if context else self.generation_ops
            target.append(ops.Embedding(f"{phase}_embedding", 1, self._vocab_size // tp, h, 0.3))
            target.append(ops.NCCL(f"{phase}_embedding_allreduce", 1, "all_reduce", h, tp, common.CommQuantMode.half))
            target.extend(stage(i, context) for i in range(d.num_hidden_layers))
            target.extend(
                [
                    ops.GEMM(f"{phase}_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                    ops.P2P(f"{phase}_p2p", model_config.pp_size - 1, h, model_config.pp_size),
                ]
            )
        self._resident_weight_bytes = float(sum(op.get_weights() for op in self.context_ops))
        # Native MXFP4 scales are one UE8M0 byte per 32 weights. The generic
        # legacy MoE op stores only packed-weight bytes, so preserve the scale
        # inventory here independently of either phase's execution count.
        if model_config.moe_quant_mode.name.startswith("w4") and "mxfp4" in model_config.moe_quant_mode.name:
            expert_elements = d.num_hidden_layers * 3 * h * self._moe_inter_size * self._num_experts / (mtp * ep)
            self._resident_weight_bytes += expert_elements / 32

    def get_additional_activation_bytes(self, num_tokens: int) -> float:
        d = self.extra_params
        # Two expanded residual buffers plus the largest Engram lookup,
        # projection and hash workspace; layer-local buffers are reusable.
        residual = 2 * d.hc_mult * d.hidden_size * 2
        hash_columns = (d.engram_max_ngram_size - 1) * d.engram_n_heads
        engram = 2 * (hash_columns * d.engram_head_dim + (d.hc_mult + 1) * d.hidden_size)
        statistics = 2 * (d.hc_mult + 2) * d.hc_mult * 4
        hashes = hash_columns * len(d.engram_layer_ids) * 8
        return float(num_tokens * (residual + engram + statistics + hashes))

    def get_resident_weights_bytes(self) -> float:
        return self._resident_weight_bytes

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        return self.extra_params.kvcache_bytes(seq_len)

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)

    def get_kvcache_batch_capacity(self, kv_budget_bytes: float, max_batch_size: int) -> int:
        d = self.extra_params
        fixed = d.num_hidden_layers * d.sliding_window * d.head_dim
        fixed += sum(
            2 * d.compress_ratios[i] * d.head_dim * 4 for i in d.kv_source_layer_ids if d.compress_ratios[i] > 1
        )
        slope = sum(
            (d.compressed_entry_bytes + d.index_entry_bytes) / d.compress_ratios[i] for i in d.kv_source_layer_ids
        )
        # Reserve complete ring/state buffers for every scheduler slot. Charging
        # the asymptotic global slope also covers odd ratio-two publication tails.
        return max(0, int((kv_budget_bytes - max_batch_size * fixed) // slope))
