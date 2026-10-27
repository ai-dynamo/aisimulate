# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.31 MLA/DSA module cache contract; preserve native model dispatch.

Serving contracts at db9527a46873454610df6dbedf79a36d6bf1a7f6:
* model_executor/models/deepseek_v2.py:999-1237: model constructor/forward.
* model_executor/layers/attention/mla_attention.py:733-814: cache binding/writer.
* model_executor/models/deepseek_v2.py:647-663: indexer cache binding/spec.
* model_executor/layers/sparse_attn_indexer.py:449: indexer cache producer.
Paths above are relative to vllm/. The model, weights, selectors, prefill dtype
probe, forward context and graph timing are shared with the legacy collector.
Only the removed common-metadata fields and native cache allocation/binding
contract are replaced. HiSparse/offload is outside this single-GPU harness.
"""

__compat__ = "vllm==0.31.0"

import sys

import torch
from collector.vllm import collect_mla_module
from collector.vllm.runtime_031 import allocate_cache
from collector.vllm.runtime_031 import create_common_metadata as _create_common_metadata
from vllm.v1.worker.utils import AttentionGroup


def get_mla_context_module_test_cases():
    return collect_mla_module.get_mla_context_module_test_cases()


def get_mla_generation_module_test_cases():
    return collect_mla_module.get_mla_generation_module_test_cases()


def create_common_metadata(*args):
    # Preserve ordered pages and avoid CUDA RNG after native module construction.
    return _create_common_metadata(*args, randomize_blocks=False)


def create_metadata_builder(backend, spec, layer_names, config, device):
    # Serving factory supplies required backend arguments, including the DSA
    # indexer's aligned block-table width (vllm/v1/worker/utils.py:269-312
    # at db9527a46873454610df6dbedf79a36d6bf1a7f6).
    group = AttentionGroup(backend=backend, layer_names=layer_names, kv_cache_spec=spec, kv_cache_group_id=0)
    group.create_metadata_builders(config, torch.device(device))
    return group.get_metadata_builder()


def bind_cache(config, layer, spec, num_blocks, device):
    cache = allocate_cache(config, layer.get_attn_backend(), spec, num_blocks, device)
    layer.bind_kv_cache(cache)
    return layer.kv_cache


def populate_mla_cache(config, layer, spec, num_blocks, common, kv_contexts, rope_contexts, device):
    if layer.hisparse_cache is not None:
        raise NotImplementedError("MLA qualification does not cover HiSparse/offloaded cache state")
    cache = bind_cache(config, layer, spec, num_blocks, device)
    # Reproduce serving's cache producer on persisted tokens, outside timing.
    # Use the same physical page table as the current-token slot mapping.
    for index, (kv, rope) in enumerate(zip(kv_contexts, rope_contexts, strict=True)):
        count = kv.shape[0]
        if count:
            positions = torch.arange(count, dtype=torch.int64, device=device)
            blocks = common.block_table_tensor[index, positions // spec.block_size].to(torch.int64)
            slots = blocks * spec.block_size + positions % spec.block_size
            layer.update_kv_cache(kv, rope.squeeze(1), cache, slots, None, spec.cache_dtype_str, layer._k_scale)
    return cache


def run_mla_module_worker(*args, **kwargs):
    return collect_mla_module.run_mla_module_worker(*args, **kwargs, _runtime=sys.modules[__name__])
