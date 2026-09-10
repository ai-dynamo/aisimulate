# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared DeepSeek-V4.1 text architecture and execution contract.

Structural source: deepseek-ai/DeepSeek-V4.1-Flash at
fb2764a5cf321eaa5070ca8f9e892818f477c16d (config.json and inference/model.py;
MIT, Copyright (c) 2023 DeepSeek). See THIRD_PARTY_NOTICES.md.
This module describes performance-model inputs; it does not execute model code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum

MODEL_PATH = "deepseek-ai/DeepSeek-V4.1-Flash"
MODEL_REVISION = "fb2764a5cf321eaa5070ca8f9e892818f477c16d"


class DeepSeekV41ExecutionProfile(StrEnum):
    FULL = "full"
    DECODER_BOUNDED = "decoder_bounded"


def resolve_execution_profile(decoder_replay: bool, backend_name: str) -> DeepSeekV41ExecutionProfile:
    """Only SGLang has a verified bounded-decoder execution contract."""
    if not isinstance(decoder_replay, bool):
        raise ValueError("decoder_replay must be a boolean")
    if decoder_replay and backend_name != "sglang":
        raise NotImplementedError(f"DeepSeek-V4.1 decoder replay is not verified for {backend_name}")
    return DeepSeekV41ExecutionProfile.DECODER_BOUNDED if decoder_replay else DeepSeekV41ExecutionProfile.FULL


@dataclass(frozen=True)
class DeepSeekV41Config:
    """Backbone geometry; the separate three DSpark layers are not AR layers."""

    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_n_heads: int
    engram_head_dim: int

    @classmethod
    def from_text_config(cls, text_config: dict) -> DeepSeekV41Config:
        values = {field.name: text_config[field.name] for field in fields(cls)}
        for name in (
            "compress_ratios",
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "engram_layer_ids",
            "engram_num_embeddings",
        ):
            values[name] = tuple(values[name])
        values["compress_ratios"] = values["compress_ratios"][: values["num_hidden_layers"]]
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        layers = self.num_hidden_layers
        if len(self.compress_ratios) != layers or set(self.compress_ratios) - {0, 1, 2}:
            raise ValueError("DeepSeek-V4.1 requires one backbone compression ratio (0, 1, or 2) per layer")
        for ids in (self.kv_source_layer_ids, self.index_source_layer_ids, self.engram_layer_ids):
            if tuple(sorted(set(ids))) != ids or any(i < 0 or i >= layers for i in ids):
                raise ValueError("DeepSeek-V4.1 source/layer IDs must be unique, ascending backbone indices")
        if not set(self.kv_source_layer_ids) <= set(self.index_source_layer_ids):
            raise ValueError("DeepSeek-V4.1 KV owners must also own an indexer")
        if self.candidate_source_layer_id not in self.kv_source_layer_ids:
            raise ValueError("DeepSeek-V4.1 candidate source must own compressed KV")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("DeepSeek-V4.1 Engram layer/table counts differ")
        source = None
        for layer, ratio in enumerate(self.compress_ratios):
            if layer in self.kv_source_layer_ids:
                source = layer
            if ratio and (source is None or self.compress_ratios[source] != ratio):
                raise ValueError(f"DeepSeek-V4.1 layer {layer} has no compatible preceding KV owner")

    @property
    def decoder_start_layer(self) -> int:
        """The first layer eligible for bounded replay; layer 20 builds global KV."""
        return max(self.kv_source_layer_ids) + 1

    def layer_role(self, layer: int) -> str:
        if not self.compress_ratios[layer]:
            return "swa"
        if layer in self.kv_source_layer_ids:
            return "full"
        return "reindex" if layer in self.index_source_layer_ids else "reuse"

    @property
    def compressed_entry_bytes(self) -> float:
        return self.head_dim / 2 + (self.head_dim + 15) // 16

    @property
    def index_entry_bytes(self) -> float:
        return self.index_head_dim / 2 + (self.index_head_dim + 31) // 32

    def kvcache_bytes(self, sequence_length: int) -> float:
        """Unique global pools plus each layer's FP8 window and FP32 pooling state."""
        length = max(0, sequence_length)
        if not length:
            return 0.0
        window = self.num_hidden_layers * min(length, self.sliding_window) * self.head_dim
        global_cache = 0.0
        state = 0.0
        for owner in self.kv_source_layer_ids:
            ratio = self.compress_ratios[owner]
            global_cache += (length // ratio) * (self.compressed_entry_bytes + self.index_entry_bytes)
            if ratio > 1:
                state += 2 * ratio * self.head_dim * 4
        return window + global_cache + state

    def engram_table_bytes(self, tp_size: int) -> float:
        """GPU-resident row-sharded FP8 tables including one UE8M0 scale per 32."""
        row_bytes = self.engram_head_dim + (self.engram_head_dim + 31) // 32
        return float(sum((n + tp_size - 1) // tp_size for n in self.engram_num_embeddings) * row_bytes)


@dataclass(frozen=True)
class V41RequestWorkload:
    """One actual forward's uncached tokens and original absolute KV position."""

    query_tokens: int
    prefix_tokens: int = 0

    def __post_init__(self) -> None:
        if self.query_tokens < 0 or self.prefix_tokens < 0:
            raise ValueError("DeepSeek-V4.1 token counts must be nonnegative")


@dataclass(frozen=True)
class V41StageWorkload:
    query_tokens: int
    prefix_tokens: int


def stage_workloads(
    descriptor: DeepSeekV41Config,
    profile: DeepSeekV41ExecutionProfile | str,
    requests: tuple[V41RequestWorkload, ...],
    layer: int,
) -> tuple[V41StageWorkload, ...]:
    """Bound per request, retaining its absolute attention position.

    Inputs describe an actual extend call, not the entire logical prompt.
    Cached prefixes are not new inputs and do not fill a short extend's tail.
    """
    profile = DeepSeekV41ExecutionProfile(profile)
    bounded = profile == DeepSeekV41ExecutionProfile.DECODER_BOUNDED and layer >= descriptor.decoder_start_layer
    return tuple(
        V41StageWorkload(
            min(r.query_tokens, descriptor.sliding_window) if bounded else r.query_tokens,
            r.prefix_tokens + (max(r.query_tokens - descriptor.sliding_window, 0) if bounded else 0),
        )
        for r in requests
    )
