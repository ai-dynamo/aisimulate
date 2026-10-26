# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSV4 0.31 native shared allocation; legacy case and timing contracts.

Contracts at vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6:
vllm/v1/core/kv_cache_utils.py:1669-1830 (shared packed groups),
vllm/v1/worker/utils.py:259-318,395-452,650-675 (builders/allocation/binding),
vllm/v1/worker/block_table.py:201-229 (group slot mapping),
vllm/models/deepseek_v4/compressor.py:157-168 (shared state pages),
vllm/utils/deep_gemm.py:662-750 (standalone paged MQA input contract).
Calls native allocation helpers; does not reproduce their implementations.
"""

__compat__ = "vllm==0.31.0"

import math
import sys
from itertools import pairwise

import torch
from collector.vllm import collect_dsv4_attn_v1 as legacy
from collector.vllm.runtime_031 import create_common_metadata
from collector.vllm.utils import BatchSpec
from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    bind_kv_cache_to_layers,
    prepare_kernel_block_sizes,
)


def _make_common_metadata(*, batch_size, seq_len, is_context, device, query_len=None):
    query_len = (seq_len if is_context else 1) if query_len is None else query_len
    batch = BatchSpec(seq_lens=[seq_len] * batch_size, query_lens=[query_len] * batch_size)
    # This initial metadata supplies positions/lengths. Each native cache group
    # gets its own physical page table below, including compressor state groups.
    return create_common_metadata(
        batch, 64, 1 + batch_size * math.ceil(seq_len / 64), device, is_context, randomize_blocks=False
    )[0]


def _group_common_metadata(common, batch, block_size, kernel_block_size, num_blocks, block_offset, device):
    grouped, _ = create_common_metadata(batch, block_size, num_blocks, device, False, randomize_blocks=False)
    grouped.is_prefilling = common.is_prefilling
    # Native groups alias the same backing allocation. Disjoint physical IDs
    # prevent compressor state writes from overwriting the latent/indexer cache.
    table = grouped.block_table_tensor
    grouped.block_table_tensor = torch.where(table > 0, table + block_offset, table)
    grouped.slot_mapping += block_offset * block_size
    if block_size % kernel_block_size:
        raise ValueError("Native kernel block size must divide the allocation block size")
    ratio = block_size // kernel_block_size
    if ratio != 1:
        table = grouped.block_table_tensor
        offsets = torch.arange(ratio, device=device, dtype=table.dtype)
        grouped.block_table_tensor = (table.unsqueeze(-1) * ratio + offsets).reshape(batch.batch_size, -1)
    return grouped


def _build_metadata_and_bind_caches(attn_module, vllm_config, common, *, device):
    static_ctx = vllm_config.compilation_config.static_forward_context
    layers = [attn_module, attn_module.swa_cache_layer]
    if attn_module.indexer is not None:
        layers.append(attn_module.indexer.k_cache)
    compressors = [attn_module.compressor]
    if attn_module.indexer is not None:
        compressors.append(attn_module.indexer.compressor)
    layers.extend(compressor.state_cache for compressor in compressors if compressor is not None)
    specs = {}
    for layer in layers:
        spec = static_ctx[layer.prefix].get_kv_cache_spec(vllm_config)
        if spec is not None:
            specs[layer.prefix] = spec
    backends = {name: static_ctx[name].get_attn_backend() for name in specs}
    supported = [layout.name for layout in get_supported_kv_cache_layouts(backends.values())]
    layout = resolve_kv_cache_layout(vllm_config, [supported], specs.values())
    if layout.name not in supported:
        raise ValueError(f"Resolved DSV4 cache layout {layout.name} is unsupported")
    groups = get_kv_cache_groups(vllm_config, specs)
    seq_lens = common.seq_lens_cpu_upper_bound.tolist()
    starts = common.query_start_loc_cpu.tolist()
    batch = BatchSpec(seq_lens=seq_lens, query_lens=[b - a for a, b in pairwise(starts)])
    counts = [sum(math.ceil(length / group.kv_cache_spec.block_size) for length in seq_lens) for group in groups]
    num_blocks = 1 + sum(counts)
    previous_override = vllm_config.cache_config.num_gpu_blocks_override
    try:
        vllm_config.cache_config.num_gpu_blocks_override = num_blocks
        cache_config = get_kv_cache_config_from_groups(vllm_config, groups, available_memory=0)
    finally:
        vllm_config.cache_config.num_gpu_blocks_override = previous_override
    if cache_config.num_blocks < num_blocks:
        raise ValueError("Native DSV4 allocation cannot hold the planned disjoint page tables")

    attention_groups = []
    for group_id, group in enumerate(groups):
        group_specs = group.kv_cache_spec
        attention_groups.append(
            [
                AttentionGroup(
                    backends[name],
                    [name],
                    group_specs.kv_cache_specs[name]
                    if isinstance(group_specs, UniformTypeKVCacheSpecs)
                    else group_specs,
                    group_id,
                )
                for name in group.layer_names
            ]
        )
    kernel_sizes = prepare_kernel_block_sizes(cache_config, attention_groups)
    caches = allocate_kv_cache(cache_config, torch.device(device), layout, kernel_sizes)
    bind_kv_cache_to_layers(caches, static_ctx, kv_cache_groups=groups)
    metadata = {}
    offset = 0
    for group, attention_group, kernel_size, count in zip(groups, attention_groups, kernel_sizes, counts, strict=True):
        grouped = _group_common_metadata(
            common, batch, group.kv_cache_spec.block_size, kernel_size, num_blocks, offset, device
        )
        offset += count
        for attn in attention_group:
            attn.create_metadata_builders(vllm_config, torch.device(device), kernel_size)
            built = attn.get_metadata_builder().build(0, grouped)
            metadata.update((name, built) for name in attn.layer_names)
    return metadata


def get_dsv4_csa_context_test_cases():
    return legacy.get_dsv4_csa_context_test_cases()


def get_dsv4_hca_context_test_cases():
    return legacy.get_dsv4_hca_context_test_cases()


def get_dsv4_csa_generation_test_cases():
    return legacy.get_dsv4_csa_generation_test_cases()


def get_dsv4_hca_generation_test_cases():
    return legacy.get_dsv4_hca_generation_test_cases()


def get_dsv4_paged_mqa_logits_test_cases():
    return legacy.get_dsv4_paged_mqa_logits_test_cases()


def get_dsv4_hca_attn_test_cases():
    return legacy.get_dsv4_hca_attn_test_cases()


def run_dsv4_attn_worker(*args, **kwargs):
    return legacy.run_dsv4_attn_worker(*args, **kwargs, _runtime=sys.modules[__name__])


def run_dsv4_sparse_kernel_worker(*args, **kwargs):
    return legacy.run_dsv4_sparse_kernel_worker(*args, **kwargs, _runtime=sys.modules[__name__])
