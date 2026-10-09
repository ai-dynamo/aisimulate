# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified: CPU-only sizing of DeepSeek V4 cache groups for AISimulate's state cache.
# Adapted from https://github.com/vllm-project/vllm/tree/c5c116138267ac738bc262495326d28ffff834a2
# Original paths: vllm/models/deepseek_v4/attention.py, vllm/models/deepseek_v4/compressor.py,
# vllm/v1/attention/backends/mla/sparse_swa.py, vllm/v1/kv_cache_interface.py,
# vllm/v1/core/kv_cache_utils.py, vllm/v1/core/single_type_kv_cache_manager.py.

"""Resolve DeepSeek V4 cache geometry as vLLM allocates it, without GPU weights.

vLLM gives every DeepSeek V4 cache group the same pool row: the bytes per
block of the widest group. A request holds one row per 256 tokens for the
compressed KV and indexer K, plus the rows its sliding-window groups (SWA KV
and compressor states) keep resident. AISimulate's state cache represents
the first part as token bytes and the second as one fixed per-request state.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .common import indexer_cache_entry_bytes

VLLM_REVISION = "c5c116138267ac738bc262495326d28ffff834a2"
# FlashMLA and the indexer support only 256-token kernel blocks, and the
# compressor state blocks below are sized for them.
BLOCK_SIZE = 256
_SWA_BLOCK_SIZE = 64
# fp8_ds_mla: 448 FP8 NoPE bytes + 128 BF16 RoPE bytes + 8 scale bytes.
_ENTRY_BYTES = 584
_PAGE_ALIGNMENT = 576
_STATE_BLOCK_SIZE = {4: 4, 128: 8}
_FP32_BYTES = 4

Bucket = tuple[int, int | None]  # (block size, sliding window or None)


def _cdiv(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _page(slots: int, bytes_per_slot: int) -> int:
    return _cdiv(slots * bytes_per_slot, _PAGE_ALIGNMENT) * _PAGE_ALIGNMENT


def _approximate_gcd(values: list[int], lower_bound: int) -> int:
    """vLLM's padding-minimizing repeat count; ties prefer the larger value."""
    return min(range(lower_bound, max(values) + 1), key=lambda d: (sum(-value % d for value in values), -d))


def _layer_specs(geometry: Any, indexer_entry_bytes: int) -> list[tuple[Bucket, int, int]]:
    """Return (bucket, slots per page, bytes per slot) per cache layer."""
    specs = []
    for ratio in geometry.compress_ratios:
        specs.append(((_SWA_BLOCK_SIZE, geometry.sliding_window), _SWA_BLOCK_SIZE, _ENTRY_BYTES))
        if not ratio:
            continue
        coff = 2 if ratio == 4 else 1
        state = (_STATE_BLOCK_SIZE[ratio], coff * ratio)
        caches = [(_ENTRY_BYTES, geometry.head_dim)]
        if ratio == 4:
            caches.append((indexer_entry_bytes, geometry.index_head_dim))
        for entry_bytes, dim in caches:
            specs.append(((BLOCK_SIZE, None), BLOCK_SIZE // ratio, entry_bytes))
            specs.append((state, state[0], 2 * coff * dim * _FP32_BYTES))
    return specs


def _cache_groups(specs: list[tuple[Bucket, int, int]]) -> list[dict[str, Any]]:
    """Split buckets into vLLM's packed groups (block-outermost layout).

    This ports the parts of vLLM's packed grouping that apply to DeepSeek V4 at
    the pinned commit. Its Mamba/circular-buffer state-bucket split, EAGLE group
    annotation and block-stride rounding are no-ops for these specs.
    """
    buckets: dict[Bucket, Counter[int]] = {}
    for bucket, slots, slot_bytes in specs:
        buckets.setdefault(bucket, Counter())[_page(slots, slot_bytes)] += 1
    # Balanced buckets hold the same number of layers per page size.
    repeats_by_bucket = {
        bucket: max(pages.values()) for bucket, pages in buckets.items() if len(set(pages.values())) == 1
    }
    min_repeats = max((n for bucket, n in repeats_by_bucket.items() if len(buckets[bucket]) > 1), default=0)
    if not min_repeats:
        raise ValueError("DeepSeek V4 sizing expects compressed layers with compressor states")
    repeats = _approximate_gcd(list(repeats_by_bucket.values()), min_repeats)
    groups = []
    for (block_size, window), pages in buckets.items():
        count = (
            _cdiv(repeats_by_bucket[(block_size, window)], repeats) if (block_size, window) in repeats_by_bucket else 1
        )
        for index in range(count):
            layers = {page: len(range(index, total, count)) for page, total in pages.items()}
            groups.append(
                {
                    "block_size": block_size,
                    "sliding_window": window,
                    "layers": sum(layers.values()),
                    "bytes_per_block": sum(page * n for page, n in layers.items()),
                }
            )
    return groups


def deepseek_v4_state_cache(
    geometry: Any, *, indexer_cache_dtype: str, num_speculative_tokens: int = 0
) -> dict[str, Any]:
    """Per-rank token bytes, fixed state bytes and pool row size for one model.

    Draft tokens extend each sliding window by the tokens scheduled per step.
    """
    if geometry.head_dim != 512 or geometry.qk_rope_head_dim != 64 or geometry.index_head_dim != 128:
        raise ValueError("DeepSeek V4 sizing follows vLLM's fp8_ds_mla layout: head_dim 512, RoPE 64, indexer 128")
    if set(geometry.compress_ratios) - {0, *_STATE_BLOCK_SIZE}:
        raise ValueError("DeepSeek V4 sizing supports compress ratios 0, 4 and 128")
    indexer = "fp8" if indexer_cache_dtype == "auto" else indexer_cache_dtype
    dim = geometry.index_head_dim
    if indexer == "fp8":
        indexer_bytes = indexer_cache_entry_bytes(dim)
    elif indexer == "mxfp4":
        # Two values per byte plus one E8M0 scale per 32 values.
        indexer_bytes = dim // 2 + dim // 32
    else:
        raise ValueError("indexer_cache_dtype must be auto, fp8 or mxfp4")
    specs = _layer_specs(geometry, indexer_bytes)
    groups = _cache_groups(specs)
    row = max(group["bytes_per_block"] for group in groups)
    # W window tokens plus k draft tokens can span cdiv(W + k, B) + 1 blocks of B tokens.
    fixed_blocks = sum(
        _cdiv(group["sliding_window"] + num_speculative_tokens, group["block_size"]) + 1
        for group in groups
        if group["sliding_window"]
    )
    raw_state = sum(window * slot_bytes for (_, window), _, slot_bytes in specs if window)
    state = fixed_blocks * row
    return {
        "source": "inferred",
        "block_size": BLOCK_SIZE,
        "kv_bytes_per_token": _cdiv(row, BLOCK_SIZE),
        "bytes_per_request": state,
        "raw_bytes_per_request": raw_state,
        "padding_bytes_per_request": state - raw_state,
        "bytes_per_block": row,
        "fixed_blocks": fixed_blocks,
        "cache_groups": groups,
        "kv_cache_dtype": "fp8_ds_mla",
        "indexer_cache_dtype": indexer,
        "num_speculative_tokens": num_speculative_tokens,
        "backend_revision": VLLM_REVISION,
    }
