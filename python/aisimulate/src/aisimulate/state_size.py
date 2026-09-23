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
        result = {"source": "overridden", "bytes_per_request": sizing.bytes_per_request}
    else:
        result = estimate_state_cache(
            engine.model,
            backend=engine.backend,
            tp_size=worker.parallelism.tensor,
            pp_size=worker.parallelism.pipeline,
            block_size=cache.block_size,
            kv_bytes_per_token=cache.bytes_per_token,
            model_dtype=sizing.model_dtype,
            mamba_cache_dtype=sizing.mamba_cache_dtype,
            mamba_ssm_cache_dtype=sizing.mamba_ssm_cache_dtype,
            num_speculative_tokens=engine.speculation.num_speculative_tokens if engine.speculation else 0,
            layout=sizing.layout,
            allow_hf_config_download=True,
        )
    size = StateCacheConfig(bytes_per_request=result["bytes_per_request"])
    blocks = size.state_blocks(cache.block_size, cache.bytes_per_token)
    block_bytes = cache.block_size * cache.bytes_per_token
    capacity = cache.capacity.blocks
    if capacity is None:
        capacity = cache.capacity.bytes // block_bytes
    if capacity < blocks + 1:
        raise ValueError("state_cache capacity must fit one token block and one request state")
    return {**result, "state_blocks": blocks, "allocated_bytes_per_request": blocks * block_bytes}
