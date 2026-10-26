# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.31 single-layer cache/metadata construction.

Contracts at db9527a46873454610df6dbedf79a36d6bf1a7f6:
vllm/v1/worker/gpu_model_runner.py:2426-2446 (metadata),
vllm/v1/worker/block_table.py:201-229 (slots),
vllm/v1/attention/backends/utils.py:209-344 (layout negotiation),
vllm/v1/kv_cache_interface.py:296-420 (native views/byte strides).
Not a replacement for shared physical cache allocations required by DSV4.
"""

from itertools import accumulate

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
from vllm.v1.kv_cache_interface import KVCacheTensor, compute_layout_strides, create_kv_cache_views


def create_common_metadata(batch, block_size, num_blocks, device, is_context_phase, *, randomize_blocks=True):
    """Build public serving metadata and disjoint history/query slot mappings.

    The serving field/slot population sites are cited in the module docstring.
    No private metadata caches from older vLLM versions are populated.
    """
    block_counts = [(length + block_size - 1) // block_size for length in batch.seq_lens]
    used_blocks = sum(block_counts)
    if num_blocks <= used_blocks:
        raise ValueError("KV allocation must include the null block and every sequence block")
    block_ids = (
        torch.randperm(used_blocks, device=device) if randomize_blocks else torch.arange(used_blocks, device=device)
    ) + 1
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
    return common, torch.cat(history_slots)


def allocate_cache(config, backend, spec, num_blocks, device):
    supported = [[layout.name for layout in get_supported_kv_cache_layouts([backend])]]
    layout = resolve_kv_cache_layout(config, supported, [spec])
    if layout.name not in supported[0]:
        raise ValueError(f"Resolved cache layout {layout.name} is unsupported by {backend}")
    strides = compute_layout_strides(spec, num_blocks, 1, layout)
    size = num_blocks * spec.page_size_bytes
    placement = KVCacheTensor(size=size, layers=["placeholder"], layer_stride=strides[0], block_stride=strides[1])
    raw = torch.zeros(size, dtype=torch.int8, device=device)
    cache = create_kv_cache_views(raw, spec, num_blocks, layout, placement)[0]
    return cache
