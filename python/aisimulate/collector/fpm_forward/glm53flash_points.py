# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GLM-5.3-Flash FPM point list, generated from the onboarding limits and the accepted profile.

The GLM-5.3-Flash producer (``runtime/glm53flash/glm53flash_prefix_scheduler.py``) measures an explicit point list
only, and onboarding passes grid limits only. The planner freezes this list into the collection plan, so the plan
hash covers it. The list keeps the shape of Dynamo's generic grid where the producer's rules allow it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .config import _powers_of_two_up_to

if TYPE_CHECKING:
    from aisimulate_core.sdk.fpm_profile import FpmDeploymentProfile

    from .config import FPMCollectionOptions

GLM_ARCHITECTURE = "Glm5NextForConditionalGeneration"
GLM_POINTS_ORIGIN = "glm53flash_grid_v1"
GLM_SMOKE_MANIFEST = "glm53flash_grid_v1_smoke"
# The producer's CHUNK_ALIGN, MAX_BATCH, MAX_NEW and MAX_CONTEXT. The producer module imports vLLM and Dynamo, so the
# planner keeps copies.
CHUNK_ALIGN = 4
MAX_BATCH = 32
MAX_NEW = 8192
MAX_CONTEXT = 131072
# The smallest power of two above the producer's minimum decode context of 3.
MIN_CONTEXT = 4


def _manifest(payload: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    # The canonical form of a user point file, so the runner stages and checks this list like one.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return payload, canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def glm_measured_context(options: FPMCollectionOptions) -> int:
    """The deepest per-request context that the producer measures.

    vLLM needs seven positions beyond it, the producer's ``CONTEXT_HEADROOM``.
    """
    return min(MAX_CONTEXT, options.vllm_max_model_len - 7)


def glm_point_manifest(
    *,
    hash_block_size: int,
    token_budget: int,
    prefill_batch_limit: int,
    decode_batch_limit: int,
    measured_context: int,
    kv_cache_bytes: int,
    page_size_bytes: int,
    state_groups: int,
) -> tuple[dict[str, Any], str, str]:
    """Return the point list, its canonical JSON and the SHA-256 of that JSON.

    Every point meets the producer's rules. Prefixes are whole hash blocks. New-token and prefix lengths are multiples
    of 4. Prefix plus new tokens stay within the measured context. The new tokens of a batch fit the token budget, so
    each request's suffix runs in one step. The producer divides a total evenly, so every request of a point gets the
    same prefix, the same new tokens and the same decode context.
    """
    if token_budget % CHUNK_ALIGN or hash_block_size % CHUNK_ALIGN:
        raise ValueError("GLM-5.3-Flash collection requires a token budget that is a multiple of 4")
    budget = min(token_budget, MAX_NEW)
    measured = min(measured_context, MAX_CONTEXT)
    unit = hash_block_size
    usable = kv_cache_bytes // page_size_bytes - 1

    def fits(batch_size: int, tokens: int) -> bool:
        # A conservative stand-in for the producer's check through the native cache managers: ceil((tokens + 1) / unit)
        # blocks of the request's attention, and one block in each state group.
        return batch_size * (-(-(tokens + 1) // unit) + state_groups) <= usable

    def batch_sizes(limit: int) -> list[int]:
        return sorted({*_powers_of_two_up_to(limit), limit})

    new_tokens = {budget, *range(unit, budget + 1, unit)}
    new_tokens.update(value for value in _powers_of_two_up_to(budget) if value >= CHUNK_ALIGN)
    prefill = []
    for new in sorted(new_tokens):
        blocks = (measured - new) // unit
        rungs = sorted(rung for rung in {0, 1, blocks, *_powers_of_two_up_to(blocks)} if 0 <= rung <= blocks)
        for batch in batch_sizes(min(MAX_BATCH, prefill_batch_limit, budget // new)):
            prefill.extend(
                {"batch_size": batch, "total_prefill_tokens": batch * new, "total_kv_read_tokens": batch * rung * unit}
                for rung in rungs
                if fits(batch, rung * unit + new)
            )
    contexts = {measured, *(value for value in _powers_of_two_up_to(measured) if value >= MIN_CONTEXT)}
    decode = [
        {"batch_size": batch, "total_kv_read_tokens": batch * context}
        for context in sorted(contexts)
        for batch in batch_sizes(min(MAX_BATCH, decode_batch_limit))
        if fits(batch, context)
    ]
    return _manifest({"schema_version": 3, "prefill": prefill, "decode": decode})


def glm_plan_manifest(options: FPMCollectionOptions, deployments: Sequence[FpmDeploymentProfile]) -> tuple[str, str]:
    """Return the canonical JSON and the SHA-256 of the point list for a plan's options and profile deployments.

    The planner resolves ``options.max_prefill_isl`` first. The cells of a plan share one list, so the deployments must
    share one hash block, and the capacity rule takes the smallest capacity, the largest page and the most state groups
    among them.
    """
    memories = [deployment.resources.runtime_memory for deployment in deployments]
    hash_blocks = {None if memory is None else memory.hash_block_size_tokens for memory in memories}
    if len(hash_blocks) != 1 or None in hash_blocks:
        raise ValueError("GLM-5.3-Flash collection requires the accepted profile's hash_block_size_tokens")
    prefill_batch_limit = options.max_prefill_batch_size or options.max_num_seqs
    decode_batch_limit = options.max_decode_batch_size or options.max_num_seqs
    if prefill_batch_limit is None or decode_batch_limit is None:
        raise ValueError("GLM-5.3-Flash collection requires --fpm-max-num-seqs or the per-phase batch limits")
    _, canonical, digest = glm_point_manifest(
        hash_block_size=hash_blocks.pop(),
        token_budget=options.max_prefill_isl,
        prefill_batch_limit=prefill_batch_limit,
        decode_batch_limit=decode_batch_limit,
        measured_context=glm_measured_context(options),
        kv_cache_bytes=min(memory.kv_cache_bytes for memory in memories),
        page_size_bytes=max(
            group.page_size_bytes for deployment in deployments for group in deployment.resources.cache_groups
        ),
        state_groups=max(
            sum(group.kind == "state" for group in deployment.resources.cache_groups) for deployment in deployments
        ),
    )
    return canonical, digest


def glm_smoke_subset(payload: dict[str, Any], hash_block_size: int) -> tuple[dict[str, Any], str, str]:
    """Return the four smoke points of a generated list, with their canonical JSON and its SHA-256.

    They are the 64-token prefill of one request without a prefix and after one hash block of prefix, and the decode
    of one and of two requests at the smallest context of the list above one hash block.
    """
    prefill = [
        point
        for point in payload["prefill"]
        if point["batch_size"] == 1
        and point["total_prefill_tokens"] == 64
        and point["total_kv_read_tokens"] in (0, hash_block_size)
    ]
    contexts = [point["total_kv_read_tokens"] // point["batch_size"] for point in payload["decode"]]
    context = min((value for value in contexts if value > hash_block_size), default=None)
    decode = [
        point
        for point in payload["decode"]
        if context is not None
        and point["batch_size"] in (1, 2)
        and point["total_kv_read_tokens"] == point["batch_size"] * context
    ]
    if len(prefill) != 2 or len(decode) != 2:
        raise ValueError("GLM-5.3-Flash smoke subset requires the generated list to hold its four points")
    return _manifest({"schema_version": payload["schema_version"], "prefill": prefill, "decode": decode})
