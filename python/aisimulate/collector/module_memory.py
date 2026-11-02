# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visible allocation estimates for the single-layer MLA/DSA/MSA collectors.

Native workspaces and allocator overhead remain runtime measurements, not a
fit guarantee. Callers apply these estimates only while planning cases.
"""

import json
import os
from functools import cache
from pathlib import Path

from collector.helper import _resolve_local_model_path


@cache
def module_model_config(model_path: str) -> dict:
    local = Path(model_path) / "config.json"
    packaged = Path(__file__).resolve().parents[1] / "src/aisimulate_core/model_configs"
    path = local if local.is_file() else packaged / f"{model_path.replace('/', '--')}_config.json"
    if not path.is_file():
        path = Path(_resolve_local_model_path(model_path)) / "config.json"
    with path.open() as stream:
        config = json.load(stream)
    return config.get("text_config", config)


def validate_module_position(model_path: str, full_length: int) -> None:
    """Fail before constructing a module whose RoPE table cannot index the case."""
    maximum = module_model_config(model_path).get("max_position_embeddings")
    if maximum is not None and full_length > int(maximum):
        raise ValueError(f"{model_path}: required sequence length {full_length} exceeds model limit {maximum}")


def module_footprint_bytes(case, *, backend: str, attention_type: str, is_context: bool, page_size: int) -> int:
    """Count the collector's paged caches, history inputs and attention activations.

    The positional case is the ungrouped module getter contract: sequence, batch,
    heads, KV dtype, compute dtype, GEMM type, model, then optional attention type
    and prefix. TRT-LLM owns one layer; SGLang replaces this helper's single-layer
    cache estimate with its full runner pool before applying memory admission.
    """
    sequence, batch, heads, kv_dtype, _, _, model_path, *rest = case
    prefix = int(rest[-1]) if is_context and rest and isinstance(rest[-1], int) else 0
    config = module_model_config(model_path)
    query_tokens = batch * (sequence if is_context else 1)
    full_length = sequence + prefix + (0 if is_context else 1)
    # Cache managers reserve a further position; round each request separately.
    padded = ((full_length + 1 + page_size - 1) // page_size) * page_size
    cache_tokens = batch * padded
    elem = 1 if kv_dtype == "fp8" else 2
    hidden = int(config["hidden_size"])
    activations = query_tokens * 2 * hidden * 2  # hidden input and output
    if attention_type == "msa":
        dim = int(config.get("head_dim", hidden // int(config["num_attention_heads"])))
        kv_heads = max(1, heads * int(config["num_key_value_heads"]) // int(config["num_attention_heads"]))
        sparse = config["sparse_attention_config"]
        index_dim = int(sparse["sparse_index_dim"])
        if backend == "trtllm":
            # create_kv_cache_and_metadata reserves 2x (main + side-cache)
            # tokens, with two spare pages per request. Keep that native policy.
            cache_tokens = 2 * batch * (padded + 2 * page_size)
        cache = cache_tokens * (2 * kv_heads * dim * elem + index_dim * 2)
        activations += query_tokens * (heads + 2 * kv_heads) * dim * 2
        topk = int(sparse["sparse_topk_blocks"])
        if backend == "vllm":
            # collect_msa_module reserves top-k by computed tokens, not cached KV.
            reserved = max(131072, query_tokens) if is_context else query_tokens
            activations += ((reserved + 3) // 4) * 4 * kv_heads * topk * 4
        elif backend == "trtllm":
            # TRT-LLM c76f4a85 (1.3.0rc29), attention/backends/sparse/minimax_m3:
            # msa_indexer.py:103-155 returns FP32 [index_heads, K/128, Q].
            index_heads = int(sparse["sparse_num_index_heads"])
            activations += index_heads * ((full_length + 127) // 128) * query_tokens * 4
            activations += query_tokens * kv_heads * topk * 4
            if page_size == 32:
                # Triton reference keeps the full Q x index_heads x K FP32 slab
                # and masked copy (triton_backend.py:328-371); per-query K/V
                # expansions have a separate 512MiB chunk budget (:216-291).
                activations += 2 * query_tokens * index_heads * full_length * 4
                chunk_budget = max(1, int(os.environ.get("TRTLLM_MINIMAX_M3_SPARSE_CHUNK_BYTES", 512 * 2**20)))
                # Native chunking admits at least one query, even when its
                # K/V expansion alone is larger than the requested budget.
                one_query = full_length * max(
                    2 * kv_heads * dim * 4 + 2 * heads * 4,
                    2 * index_dim * 4 + index_heads * 4,
                )
                activations += max(chunk_budget, one_query)
    else:
        rank, rope = int(config["kv_lora_rank"]), int(config["qk_rope_head_dim"])
        nope, value = int(config["qk_nope_head_dim"]), int(config["v_head_dim"])
        if backend == "vllm":
            # _create_kv_cache_and_metadata retains at least 8192 pages.
            cache_tokens = max(cache_tokens + page_size, 8192 * page_size)
        entry = (rank + rope) * elem
        if attention_type == "dsa":
            index_dim = int(config["index_head_dim"])
            entry += index_dim + 4 * ((index_dim + 127) // 128)
            activations += query_tokens * int(config["index_topk"]) * 4
            if backend == "trtllm" and is_context:
                # TRT-LLM c76f4a85, sparse/dsa/indexer.py:1116-1169 and
                # :1571-1607 tiles prefill logits and gathers FP8 index K.
                # Include the native env override and minimum one-query tile.
                history_tokens = batch * full_length
                element_budget = int(os.environ.get("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", 1 << 31))
                tile = max(1, min(query_tokens, element_budget // max(1, history_tokens)))
                activations += tile * history_tokens * 4
                activations += history_tokens * (index_dim + 4)
        cache = cache_tokens * entry
        if backend == "vllm":
            # Synthetic cached K/V sources stay BF16 even for FP8 destinations.
            history = prefix if is_context else max(0, sequence - 1)
            cache += batch * history * (rank + rope) * 2
        activations += query_tokens * heads * (2 * (nope + rope) + value) * 2
    return cache + activations


def standalone_mla_footprint_bytes(case, *, backend: str, projected: bool = True) -> int:
    """Visible tensors in each standalone collect_mla.py, including cached sources.

    Both adapters validate the fixed 512-latent / 128-nope / 64-RoPE geometry
    before returning cases. TRT-LLM builds full-head context K/V; SGLang uses
    one latent head and either 128 projected or 512 latent query dimensions.
    """
    sequence, batch, _, dtype, heads, _, tp, page, _, _, is_context = case
    heads //= tp
    elem = 1 if "fp8" in str(dtype).lower() or "float8" in str(dtype).lower() else 2
    total_length = sequence + (0 if is_context else 1)
    cache_tokens = batch * ((total_length + 2 + page - 1) // page) * page
    memory = cache_tokens * 576 * elem
    query_tokens = batch * (sequence if is_context else 1)
    if backend == "trtllm":
        # Context: compressed KV, RoPE, projected Q, packed KV, concatenated K,
        # latent-cache concatenation and output; all activations stay BF16.
        entry = 2 * 576 + heads * (192 + 256 + 192 + 128) if is_context else 2 * 576 + heads * (576 + 64 + 512)
        memory += query_tokens * entry * 2
        if not is_context and elem == 1:
            memory += query_tokens * heads * 576  # quant_q_buffer
    else:
        dim = 128 if is_context and projected else 512
        memory += query_tokens * ((heads + 1) * (dim + 64) + heads * dim) * 2
        if not is_context:
            memory += batch * sequence * 576 * 2  # cached BF16 source K/RoPE
        if not projected:
            memory += query_tokens * (heads + 1) * (dim + 64) * 2  # concatenated Q/K
        memory += batch * total_length * 20  # request map, token indices, positions
        if projected:
            memory += total_length * 64 * 4  # float32 rotary cache
    return memory
