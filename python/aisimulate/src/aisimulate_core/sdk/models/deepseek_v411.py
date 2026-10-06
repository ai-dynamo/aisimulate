# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 text AR, `dsv411` decomposition (coexists with ``DEEPSEEKV41``).

Selected by ``ModelConfig.dsv41_family == "dsv411"`` (Rust-owned switch). Differences
from the ``DEEPSEEKV41`` model: the index scoring is its own measured component,
KV byte layouts and the index-scoring precision are explicit per-backend runtime
facts, stages sum sequentially (cross-module overlap is the whole-forward model's
concern), only the ``full`` execution profile exists, and both runtimes serve the
same physical compressed-KV row (fp8_ds_mla, 584 bytes).

Architecture source: deepseek-ai/DeepSeek-V4.1-Flash at
fb2764a5cf321eaa5070ca8f9e892818f477c16d (config.json). Runtime facts measured on
sglang v0.5.21 and vLLM v0.30.0 (H20, 2026-10-02); see collector/*/README.dsv411.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import aisimulate_core._native as core
import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk import common
from aisimulate_core.sdk.deepseek_v41 import DeepSeekV41Config
from aisimulate_core.sdk.models.base import BaseModel, register_model

FAMILY = "DEEPSEEKV411"
EXECUTION_PROFILE = "full"


@dataclass(frozen=True)
class Dsv411RuntimeFacts:
    """Per-backend physical facts the operators carry explicitly (bytes, precisions, sharding)."""

    window_entry_bytes: float
    main_entry_bytes: float
    index_entry_bytes: float
    fmha_quant_mode: str
    index_scoring_quant_mode: str
    index_skip_within_topk: bool
    engram_sharding: str

    @classmethod
    def for_backend(cls, backend_name: str) -> Dsv411RuntimeFacts:
        if backend_name == "sglang":
            # deepseek_v4_memory_pool.py (584-byte FlashMLA row), fp4 index K (68 bytes),
            # SM90 scoring unpacks FP4 K inside BF16 kernels; row-sharded engram + all-reduce.
            return cls(584.0, 584.0, 68.0, "fp8", "bfloat16", False, "row")
        if backend_name == "vllm":
            # attention.py:947-972 (fp8_ds_mla, 584 B, alignment 576), indexer K cache fp8
            # (132 B, :1009-1028), DeepGEMM fp8 mqa_logits, scoring skipped when
            # max_seq_len // cr <= index_topk in eager prefill only (:1195-1217; under CUDA graphs the
            # prepared index queries are always scored); head-sharded engram + all-gather.
            return cls(584.0, 584.0, 132.0, "fp8", "fp8", True, "head")
        raise NotImplementedError(f"dsv411 has no measured runtime facts for backend {backend_name!r}")


def _native(kind: str, **values):
    return core.op_from_spec_json(json.dumps({kind: values}))


def _spec(op):
    return json.loads(op._spec_json())


@register_model(FAMILY)
class DeepSeekV411Model(BaseModel):
    """40-layer text backbone, dsv411 decomposition."""

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
            raise TypeError("DeepSeekV411Model requires the DeepSeekV41Config descriptor")
        if self._nextn:
            raise NotImplementedError("dsv411 text AR requires nextn=0 (DSpark has a separate contract)")
        if self._num_layers != d.num_hidden_layers:
            raise ValueError("dsv411 layer overrides cannot preserve cross-layer KV ownership")
        if getattr(model_config, "decoder_replay", False):
            raise NotImplementedError(
                "dsv411 measures the full execution profile only; decoder_replay is DEEPSEEKV41-only"
            )
        if model_config.moe_backend == "megamoe":
            raise NotImplementedError("dsv411 uses decomposed MoE; MegaMoE requires separate measurement")
        if model_config.attention_dp_size != 1 or model_config.pp_size != 1 or model_config.cp_size != 1:
            raise NotImplementedError("dsv411 requires attention_dp_size=1, pp_size=1 and cp_size=1")
        tp = model_config.tp_size
        mtp, ep = model_config.resolve_moe_parallelism()
        if tp * model_config.attention_dp_size != mtp * ep:
            raise ValueError("attention TP * DP must equal MoE TP * EP for dsv411")
        if d.n_routed_experts % ep or d.o_groups % tp or d.num_attention_heads % tp:
            raise ValueError("dsv411 EP must divide experts; TP must divide heads and output groups")
        facts = Dsv411RuntimeFacts.for_backend(backend_name)
        self.runtime_facts = facts
        self._topk, self._num_experts, self._moe_inter_size = info["topk"], info["num_experts"], info["moe_inter_size"]
        self.raw_config = info["raw_config"]
        self.text_config = self.raw_config["text_config"]
        self.execution_profile = EXECUTION_PROFILE
        self.engram_residency = "hbm_tp_sharded"
        h = self._hidden_size
        gemm = model_config.gemm_quant_mode.name
        distribution = (
            "power_law_1.01"
            if model_config.workload_distribution == "power_law"
            else model_config.workload_distribution
        )
        kv_layout = dict(
            window_entry_bytes=facts.window_entry_bytes,
            main_entry_bytes=facts.main_entry_bytes,
            index_entry_bytes=facts.index_entry_bytes,
        )

        def attention_core(layer: int, context: bool):
            return _native(
                "Dsv411AttentionCore",
                # the engine's mixed-step path identifies attention-shaped ops by these names
                name="context_attention" if context else "generation_attention",
                is_context=context,
                role=d.layer_role(layer),
                compress_ratio=d.compress_ratios[layer],
                tp_size=tp,
                hidden_size=h,
                num_heads=d.num_attention_heads // tp,
                head_dim=d.head_dim,
                q_lora_rank=d.q_lora_rank,
                o_lora_rank=d.o_lora_rank,
                o_groups=max(1, d.o_groups // tp),
                window_size=d.sliding_window,
                index_topk=d.index_topk,
                index_head_dim=d.index_head_dim,
                kv_layout=kv_layout,
                gemm_quant_mode=gemm,
                fmha_quant_mode=facts.fmha_quant_mode,
            )

        def indexer(layer: int, context: bool):
            phase = "context" if context else "generation"
            return _native(
                "Dsv411Indexer",
                name=f"{phase}_indexer",
                is_context=context,
                compress_ratio=d.compress_ratios[layer],
                tp_size=tp,
                hidden_size=h,
                q_lora_rank=d.q_lora_rank,
                # both runtimes replicate every index head per rank (sglang dsv41_sparse.py:203-224,
                # vLLM attention.py:1089-1102)
                index_n_heads=d.index_n_heads,
                index_head_dim=d.index_head_dim,
                index_topk=d.index_topk,
                is_candidate_source=layer == d.candidate_source_layer_id,
                candidate_limit=d.candidate_topk_blocks * d.candidate_block_size
                if layer > d.candidate_source_layer_id
                else 0,
                index_entry_bytes=facts.index_entry_bytes,
                scoring_quant_mode=facts.index_scoring_quant_mode,
                skip_within_topk=facts.index_skip_within_topk,
                gemm_quant_mode=gemm,
            )

        def stage(layer: int, context: bool):
            phase = "context" if context else "generation"
            role = d.layer_role(layer)
            children = []
            if layer in d.engram_layer_ids:
                index = d.engram_layer_ids.index(layer)
                children.append(
                    _native(
                        "Dsv411Engram",
                        name=f"{phase}_engram",
                        is_context=context,
                        num_embeddings=d.engram_num_embeddings[index],
                        head_dim=d.engram_head_dim,
                        hash_columns=(d.engram_max_ngram_size - 1) * d.engram_n_heads,
                        hidden_size=h,
                        hc_mult=d.hc_mult,
                        tp_size=tp,
                        sharding=facts.engram_sharding,
                        gemm_quant_mode=gemm,
                    )
                )
                # row sharding reduces the gathered rows; head sharding gathers columns —
                # same payload per rank, modeled with the measured all-reduce table.
                children.append(
                    ops.NCCL(
                        f"{phase}_engram_collective",
                        1,
                        "all_reduce",
                        (d.engram_max_ngram_size - 1) * d.engram_n_heads * d.engram_head_dim,
                        tp,
                        common.CommQuantMode.half,
                    )
                )
            children.append(
                _native(
                    "Dsv411Mhc",
                    name=f"{phase}_mhc",
                    is_context=context,
                    hidden_size=h,
                    hc_mult=d.hc_mult,
                    sinkhorn_iters=d.hc_sinkhorn_iters,
                    tp_size=tp,
                )
            )
            if role in ("full", "reindex"):
                children.append(indexer(layer, context))
            children.append(attention_core(layer, context))
            children.append(ops.NCCL(f"{phase}_attention_allreduce", 1, "all_reduce", h, tp, common.CommQuantMode.half))
            local_inter = self._moe_inter_size * d.n_shared_experts // tp
            children.extend(
                [
                    _native(
                        "Dsv411SharedLinear",
                        name=f"{phase}_shared_gate_up",
                        is_context=context,
                        n=2 * local_inter,
                        k=h,
                        tp_size=tp,
                        quant_mode=gemm,
                    ),
                    ops.ElementWise(f"{phase}_shared_act_gate", 1, 2 * local_inter, local_inter, 0.8),
                    _native(
                        "Dsv411SharedLinear",
                        name=f"{phase}_shared_down",
                        is_context=context,
                        n=h,
                        k=local_inter,
                        tp_size=tp,
                        quant_mode=gemm,
                    ),
                    ops.GEMM(f"{phase}_router_gemm", 1, self._num_experts, h, common.GEMMQuantMode.bfloat16),
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
                        moe_kernel_source=model_config.moe_kernel_source,
                    ),
                ]
            )
            combine = ops.MoEDispatch(
                f"{phase}_moe_post_dispatch",
                1,
                h,
                self._topk,
                self._num_experts,
                mtp,
                ep,
                model_config.attention_dp_size,
                False,
                quant_mode=model_config.moe_quant_mode,
                backend=backend_name,
                is_context=context,
                attn_ar_modeled=True,
            )
            if ep == 1:
                combine = ops.NCCL(f"{phase}_moe_post_dispatch", 1, "all_reduce", h, mtp, common.CommQuantMode.half)
            children.append(combine)
            return _native(
                "Dsv411Stage",
                name=f"{phase}_v411_layer_{layer}",
                is_context=context,
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
        if model_config.moe_quant_mode.name.startswith("w4") and "mxfp4" in model_config.moe_quant_mode.name:
            expert_elements = d.num_hidden_layers * 3 * h * self._moe_inter_size * self._num_experts / (mtp * ep)
            self._resident_weight_bytes += expert_elements / 32

    # --- memory contract: physical rows measured on both runtimes (584 window/main, index per backend)
    def _entry_bytes(self) -> tuple[float, float, float]:
        f = self.runtime_facts
        return f.window_entry_bytes, f.main_entry_bytes, f.index_entry_bytes

    def get_additional_activation_bytes(self, num_tokens: int) -> float:
        d = self.extra_params
        residual = 2 * d.hc_mult * d.hidden_size * 2
        hash_columns = (d.engram_max_ngram_size - 1) * d.engram_n_heads
        engram = 2 * (hash_columns * d.engram_head_dim + (d.hc_mult + 1) * d.hidden_size)
        statistics = 2 * (d.hc_mult + 2) * d.hc_mult * 4
        hashes = hash_columns * len(d.engram_layer_ids) * 8
        return float(num_tokens * (residual + engram + statistics + hashes))

    def get_resident_weights_bytes(self) -> float:
        return self._resident_weight_bytes

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        d = self.extra_params
        window_entry, main_entry, index_entry = self._entry_bytes()
        total = d.num_hidden_layers * min(seq_len, d.sliding_window) * window_entry
        for layer in d.kv_source_layer_ids:
            ratio = d.compress_ratios[layer]
            published = seq_len // ratio
            total += published * main_entry
            if ratio > 1:
                total += 2 * ratio * d.head_dim * 4  # FP32 pooling state
        for layer in d.index_source_layer_ids:
            if layer in d.kv_source_layer_ids:
                total += (seq_len // d.compress_ratios[layer]) * index_entry
        return float(total)

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        return self._binary_search_kvcache_max_tokens(kv_budget_bytes)

    def get_kvcache_batch_capacity(self, kv_budget_bytes: float, max_batch_size: int) -> int:
        d = self.extra_params
        window_entry, main_entry, index_entry = self._entry_bytes()
        fixed = d.num_hidden_layers * d.sliding_window * window_entry
        fixed += sum(
            2 * d.compress_ratios[i] * d.head_dim * 4 for i in d.kv_source_layer_ids if d.compress_ratios[i] > 1
        )
        slope = sum((main_entry + index_entry) / d.compress_ratios[i] for i in d.kv_source_layer_ids)
        return max(0, int((kv_budget_bytes - max_batch_size * fixed) // slope))
