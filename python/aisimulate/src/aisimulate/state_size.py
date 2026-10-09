# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve CLI state-cache policy around the shared memory estimation API."""

from typing import Any

from .capacity import estimate_state_cache
from .config.engine import EnginePredictionConfig, StateCacheConfig, WorkerPredictionConfig


def resolve_state_size(engine: EnginePredictionConfig, worker: WorkerPredictionConfig) -> dict[str, Any]:
    cache = worker.kv_cache
    sizing = cache.state_cache
    if sizing is None:
        return {"source": "disabled", "bytes_per_request": None}
    if sizing.bytes_per_request is not None:
        result = {
            "source": "overridden",
            "bytes_per_request": sizing.bytes_per_request,
            "block_size": cache.block_size,
            "kv_bytes_per_token": cache.bytes_per_token,
        }
    else:
        result = estimate_state_cache(
            engine.model,
            backend=engine.backend,
            tp_size=worker.parallelism.tensor,
            pp_size=worker.parallelism.pipeline,
            block_size=cache.block_size,
            kv_bytes_per_token=None if cache.bytes_per_token == "auto" else cache.bytes_per_token,
            kvcache_quant_mode=engine.kvcache_quant_mode,
            mamba_cache_dtype=sizing.mamba_cache_dtype,
            indexer_cache_dtype=sizing.indexer_cache_dtype,
            num_speculative_tokens=engine.speculation.num_speculative_tokens if engine.speculation else engine.nextn,
        )
    block_size = result["block_size"]
    bytes_per_token = result["kv_bytes_per_token"]
    if cache.prefix_match_unit is not None and block_size % cache.prefix_match_unit:
        raise ValueError("prefix_match_unit must be a positive divisor of resolved block_size")
    size = StateCacheConfig(bytes_per_request=result["bytes_per_request"])
    blocks = size.state_blocks(block_size, bytes_per_token)
    block_bytes = block_size * bytes_per_token
    capacity = cache.capacity.blocks
    if capacity is None:
        capacity = cache.capacity.bytes // block_bytes
    if not 0 < capacity <= (1 << 64) - 1:
        raise ValueError("fixed KV capacity must fit at least one block within u64")
    if capacity < blocks + 1:
        raise ValueError("state_cache capacity must fit one token block and one request state")
    return {**result, "state_blocks": blocks, "allocated_bytes_per_request": blocks * block_bytes}
