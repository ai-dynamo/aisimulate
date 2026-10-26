# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.31 native Triton attention qualification with packed KV storage.

Contracts at vllm-project/vllm db9527a46873454610df6dbedf79a36d6bf1a7f6:
* vllm/v1/worker/gpu_model_runner.py:2426-2446: common metadata population.
* vllm/v1/worker/block_table.py:201-229: positions map to block slots.
* vllm/v1/attention/backends/utils.py:209-344: native layout negotiation.
* vllm/v1/kv_cache_interface.py:296-420: native packed cache views/strides.
* vllm/v1/attention/backends/triton_attn.py:615-630,764-805: packed K/V.

Only the non-windowed native Triton path is qualified here. Other selected
backends raise, with no fallback. Getters retain all cases; campaign filters
select the reviewed B200 cases. Measurements retain KV update plus forward.
"""

__compat__ = "vllm==0.31.0"

import sys
from itertools import accumulate

import torch
from collector.vllm import collect_attn_v1
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
from vllm.v1.kv_cache_interface import KVCacheTensor, compute_layout_strides, create_kv_cache_views


def get_context_attention_test_cases(if_unit_test=False):
    return collect_attn_v1.get_context_attention_test_cases(if_unit_test)


def get_generation_attention_test_cases():
    return collect_attn_v1.get_generation_attention_test_cases()


def validate_backend(backend_name):
    if backend_name != "TRITON_ATTN":
        raise NotImplementedError(f"vLLM 0.31 attention qualification does not cover selected backend {backend_name}")


def prepare_cache(batch, config, backend, spec, num_blocks, device, is_context_phase):
    """Populate one layer's native cache and a disjoint paged batch.

    The serving field/slot population sites are cited in the module docstring.
    Use production layout functions instead of assuming NHD/HND strides.
    No private metadata caches from older vLLM versions are populated.
    """
    block_size = spec.block_size
    block_counts = [(length + block_size - 1) // block_size for length in batch.seq_lens]
    used_blocks = sum(block_counts)
    if num_blocks <= used_blocks:
        raise ValueError("KV allocation must include the null block and every sequence block")
    block_ids = torch.randperm(used_blocks, device=device) + 1
    block_table = torch.zeros((batch.batch_size, max(block_counts)), dtype=torch.int32, device=device)
    query_slots, history_slots, positions = [], [], []
    cursor = 0
    for index, (seq_len, query_len, count) in enumerate(
        zip(batch.seq_lens, batch.query_lens, block_counts, strict=True)
    ):
        block_table[index, :count] = block_ids[cursor : cursor + count]
        cursor += count
        token_positions = torch.arange(seq_len, dtype=torch.int64, device=device)
        slots = block_table[index, token_positions // block_size].to(torch.int64) * block_size
        slots = slots + token_positions % block_size
        history_len = seq_len - query_len
        query_slots.append(slots[history_len:])
        history_slots.append(slots[:history_len])
        positions.append(token_positions[history_len:])

    query_start_cpu = torch.tensor([0, *accumulate(batch.query_lens)], dtype=torch.int32, device="cpu")
    seq_lens_cpu = torch.tensor(batch.seq_lens, dtype=torch.int32, device="cpu")
    common = CommonAttentionMetadata(
        query_start_loc=query_start_cpu.to(device),
        query_start_loc_cpu=query_start_cpu,
        seq_lens=seq_lens_cpu.to(device),
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        num_reqs=batch.batch_size,
        num_actual_tokens=sum(batch.query_lens),
        max_query_len=max(batch.query_lens),
        max_seq_len=max(batch.seq_lens),
        block_table_tensor=block_table,
        slot_mapping=torch.cat(query_slots),
        causal=True,
        is_prefilling=torch.tensor([is_context_phase] * batch.batch_size, device=device),
        positions=torch.cat(positions),
    )
    supported = [[layout.name for layout in get_supported_kv_cache_layouts([backend])]]
    layout = resolve_kv_cache_layout(config, supported, [spec])
    strides = compute_layout_strides(spec, num_blocks, 1, layout)
    size = num_blocks * spec.page_size_bytes
    placement = KVCacheTensor(size=size, layers=["placeholder"], layer_stride=strides[0], block_stride=strides[1])
    raw = torch.zeros(size, dtype=torch.int8, device=device)
    cache = create_kv_cache_views(raw, spec, num_blocks, layout, placement)[0]
    return common, cache, torch.cat(history_slots)


def run_attention_torch(
    batch_size,
    input_len,
    num_heads,
    num_kv_heads,
    head_dim,
    use_fp8_kv_cache,
    is_context_phase,
    window_size=0,
    *,
    perf_filename,
    device="cuda:0",
):
    if window_size:
        raise NotImplementedError("vLLM 0.31 Triton qualification covers non-windowed attention only")
    return collect_attn_v1.run_attention_torch(
        batch_size,
        input_len,
        num_heads,
        num_kv_heads,
        head_dim,
        use_fp8_kv_cache,
        is_context_phase,
        window_size,
        perf_filename=perf_filename,
        device=device,
        _runtime=sys.modules[__name__],
    )
