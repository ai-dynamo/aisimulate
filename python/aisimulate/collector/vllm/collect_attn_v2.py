# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.31 native dense attention qualification with packed KV storage.

Contracts at vllm-project/vllm db9527a46873454610df6dbedf79a36d6bf1a7f6:
* vllm/v1/worker/gpu_model_runner.py:2426-2446: common metadata population.
* vllm/v1/worker/block_table.py:201-229: positions map to block slots.
* vllm/v1/attention/backends/utils.py:209-344: native layout negotiation.
* vllm/v1/kv_cache_interface.py:296-420: native packed cache views/strides.
* vllm/v1/attention/backends/triton_attn.py:615-630,764-805: packed K/V.

The native selector remains authoritative for FlashInfer, FlashAttention and
Triton. Other selected backends raise, with no fallback. Getters retain all
cases; GPU smoke and numerical checks gate each campaign scope. Measurements
retain KV update plus forward.

Additional contracts at the same vLLM SHA:
* flashinfer.py:882-895,1326-1439: per-layer parameters and public metadata.
* flashinfer.py:1918-1925,1990-2009: native FP8 query-input capability.
* flashinfer.py:2058-2066: metadata selects FI native versus TRTLLM paths.
* flash_attn.py:687-718,1147-1220: builder, FA version and query capability.
(All paths above are under vllm/v1/attention/backends/.)
"""

__compat__ = "vllm==0.31.0"

import sys

from collector.vllm import collect_attn_v1
from collector.vllm.runtime_031 import allocate_cache, create_common_metadata


def get_context_attention_test_cases(if_unit_test=False):
    return collect_attn_v1.get_context_attention_test_cases(if_unit_test)


def get_generation_attention_test_cases():
    return collect_attn_v1.get_generation_attention_test_cases()


def validate_backend(backend_name):
    if backend_name not in ("TRITON_ATTN", "FLASHINFER", "FLASH_ATTN"):
        raise NotImplementedError(f"vLLM 0.31 attention qualification does not cover selected backend {backend_name}")


def prepare_cache(batch, config, backend, spec, num_blocks, device, is_context_phase):
    common, history_slots = create_common_metadata(batch, spec.block_size, num_blocks, device, is_context_phase)
    cache = allocate_cache(config, backend, spec, num_blocks, device)
    return common, cache, history_slots


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
    # Native Triton maps sliding_window to (window_size - 1, 0):
    # vllm/v1/attention/backends/triton_attn.py:475-480,676 at the SHA above.
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
