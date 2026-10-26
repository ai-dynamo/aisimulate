# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Kimi K3 and DeepSeek V4 token/state cache geometry without loading GPU weights."""

from __future__ import annotations

from typing import Any


def estimate_state_cache(
    model_path: str,
    *,
    backend: str = "vllm",
    tp_size: int = 1,
    pp_size: int = 1,
    block_size: int | None = None,
    kv_bytes_per_token: int | None = None,
    kvcache_quant_mode: str | None = None,
    mamba_cache_dtype: str = "auto",
    indexer_cache_dtype: str = "auto",
    num_speculative_tokens: int = 0,
) -> dict[str, Any]:
    """Resolve the physical block size and ONE state copy per rank.

    DeepSeek V4: see ``deepseek_v4``. Block size and token bytes follow
    vLLM's allocation; ``indexer_cache_dtype`` selects its FP8 (``auto``) or
    MXFP4 indexer, and draft tokens widen each sliding window.

    Kimi K3: see ``kimi_k3``. Uses the existing Kimi model for target-only
    token KV geometry. Draft KV and extra state copies belong to their runtime
    pools, not this one-copy size. Model configuration loading follows the
    SDK's normal model loader.

    ``block_size`` is the requested page granularity (default 64).
    It must satisfy the attention kernel's alignment. Automatic sizing follows
    the pinned vLLM KDA none/align layout and Triton MLA's 16-token minimum;
    supply a larger granularity for kernels requiring it (e.g. 128 for CUTLASS).
    The returned block size grows to fit a recurrent state. An explicit token
    byte rate overrides the model's rate, including caller-normalized geometry.

    State bytes include per-layer page padding, before pool-block rounding.
    ``num_speculative_tokens`` is metadata: it does not change a KDA copy's shape.
    """
    from ..models import _architecture_to_model_family, _get_model_info
    from .deepseek_v4 import BLOCK_SIZE, deepseek_v4_state_cache
    from .kimi_k3 import kimi_k3_state_cache

    info = _get_model_info(model_path)
    family = _architecture_to_model_family(info["architecture"])
    if family not in ("KIMIK3", "DEEPSEEKV4"):
        raise ValueError("automatic state sizing supports Kimi K3 and DeepSeek V4; supply manual state geometry")
    if backend != "vllm" or pp_size != 1:
        raise ValueError("automatic state sizing requires vllm and PP=1")
    for name, value, minimum in (
        ("tp_size", tp_size, 1),
        ("pp_size", pp_size, 1),
        ("num_speculative_tokens", num_speculative_tokens, 0),
    ):
        _integer(name, value, minimum)
    if family == "DEEPSEEKV4":
        if block_size not in (None, BLOCK_SIZE):
            raise ValueError(f"vLLM's DeepSeek V4 kernels and compressor states use {BLOCK_SIZE}-token blocks")
        if kv_bytes_per_token is not None:
            raise ValueError("DeepSeek V4 token bytes follow vLLM's cache groups; leave bytes_per_token as auto")
        if kvcache_quant_mode not in (None, "fp8"):
            raise ValueError("vLLM's DeepSeek V4 FlashMLA backend stores fp8_ds_mla KV; use kvcache_quant_mode fp8")
        if mamba_cache_dtype != "auto":
            raise ValueError("mamba_cache_dtype applies to Kimi K3 only")
        # KV is replicated on every TP rank, so per-rank geometry does not depend on TP.
        return deepseek_v4_state_cache(
            info["extra_params"], indexer_cache_dtype=indexer_cache_dtype, num_speculative_tokens=num_speculative_tokens
        )
    if indexer_cache_dtype != "auto":
        raise ValueError("indexer_cache_dtype applies to DeepSeek V4 only")
    return kimi_k3_state_cache(
        model_path,
        info,
        backend=backend,
        tp_size=tp_size,
        block_size=block_size,
        kv_bytes_per_token=kv_bytes_per_token,
        kvcache_quant_mode=kvcache_quant_mode,
        mamba_cache_dtype=mamba_cache_dtype,
        num_speculative_tokens=num_speculative_tokens,
    )


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or not minimum <= value <= (1 << 64) - 1:
        raise ValueError(f"{name} must be an integer from {minimum} through u64 max")
