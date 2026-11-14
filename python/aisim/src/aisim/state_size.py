# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve CLI state-cache policy around the shared memory estimation API."""

from typing import Any

from .capacity import estimate_state_cache
from .config.engine import EnginePredictionConfig, StateCacheConfig, WorkerPredictionConfig, recorded_state_pool


def resolve_state_size(engine: EnginePredictionConfig, worker: WorkerPredictionConfig, role: str) -> dict[str, Any]:
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
        if sizing.sliding_window_lookahead_tokens is not None:
            # A resolved DeepSeek V4 state keeps the window it was resolved with.
            result["sliding_window_lookahead_tokens"] = sizing.sliding_window_lookahead_tokens
    else:
        if sizing.sliding_window_lookahead_tokens is not None:
            raise ValueError("state_cache.sliding_window_lookahead_tokens requires bytes_per_request")
        result = estimate_state_cache(
            engine.model,
            backend=engine.backend,
            tp_size=worker.parallelism.tensor,
            pp_size=worker.parallelism.pipeline,
            block_size=cache.block_size,
            kv_bytes_per_token=None if cache.bytes_per_token == "auto" else cache.bytes_per_token,
            kvcache_quant_mode=worker.timing.kvcache_quant_mode or engine.kvcache_quant_mode,
            mamba_cache_dtype=sizing.mamba_cache_dtype,
            indexer_cache_dtype=sizing.indexer_cache_dtype,
            num_speculative_tokens=engine.speculation.num_speculative_tokens if engine.speculation else engine.nextn,
        )
    if sizing.retention_interval is not None:
        if "sliding_window_lookahead_tokens" not in result:
            raise ValueError("state_cache.retention_interval requires inferred DeepSeek V4 geometry")
        if sizing.retention_interval % result["block_size"]:
            raise ValueError("state_cache.retention_interval must be a multiple of the resolved block_size")
    if "sliding_window_lookahead_tokens" in result and cache.prefix_match_unit is not None:
        raise ValueError("DeepSeek V4 state resumes at block boundaries; leave prefix_match_unit unset")
    block_size = result["block_size"]
    bytes_per_token = result["kv_bytes_per_token"]
    if cache.prefix_match_unit is not None and block_size % cache.prefix_match_unit:
        raise ValueError("prefix_match_unit must be a positive divisor of resolved block_size")
    size = StateCacheConfig(bytes_per_request=result["bytes_per_request"])
    blocks = size.state_blocks(block_size, bytes_per_token)
    block_bytes = block_size * bytes_per_token
    if cache.capacity.type != "fixed":
        capacity = runtime_memory_blocks(engine, worker, role, block_bytes)
    elif (capacity := cache.capacity.blocks) is None:
        capacity = cache.capacity.bytes // block_bytes
    if not 0 < capacity <= (1 << 64) - 1:
        raise ValueError("fixed KV capacity must fit at least one block within u64")
    if capacity < blocks + 1:
        raise ValueError("state_cache capacity must fit one token block and one request state")
    return {**result, "state_blocks": blocks, "allocated_bytes_per_request": blocks * block_bytes}


def runtime_memory_blocks(
    engine: EnginePredictionConfig, worker: WorkerPredictionConfig, role: str, block_bytes: int
) -> int:
    """Whole blocks of the cache pool the FPM profile recorded for this worker (``resources.runtime_memory``)."""
    return recorded_state_pool(engine, worker, role).kv_cache_bytes // block_bytes
