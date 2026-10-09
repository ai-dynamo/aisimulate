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
# Modified: CPU-only sizing, strict support checks and AISimulate diagnostics.
# Adapted from https://github.com/vllm-project/vllm/tree/a474da28131f61684849b31e29af0eebaaedc383
# Original paths: vllm/model_executor/layers/mamba/mamba_utils.py,
# vllm/model_executor/models/kimi_linear.py, vllm/model_executor/layers/kda.py,
# vllm/platforms/interface.py, vllm/v1/kv_cache_interface.py,
# vllm/v1/attention/backends/mla/triton_mla.py.

"""Resolve Kimi K3 and DeepSeek V4 token/state cache geometry without loading GPU weights."""

from __future__ import annotations

from typing import Any

VLLM_REVISION = "a474da28131f61684849b31e29af0eebaaedc383"
_WIDTH = {"float16": 2, "bfloat16": 2, "float32": 4}


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

    DeepSeek V4: see ``deepseek_v4_state``. Block size and token bytes follow
    vLLM's allocation; ``indexer_cache_dtype`` selects its FP8 (``auto``) or
    MXFP4 indexer, and draft tokens widen each sliding window.

    Kimi K3:

    Uses the existing Kimi model for target-only token KV geometry. Draft KV and
    extra state copies belong to their runtime pools, not this one-copy size.
    Model configuration loading follows the SDK's normal model loader.

    ``block_size`` is the requested page granularity (default 64).
    It must satisfy the attention kernel's alignment. Automatic sizing follows
    the pinned vLLM KDA none/align layout and Triton MLA's 16-token minimum;
    supply a larger granularity for kernels requiring it (e.g. 128 for CUTLASS).
    The returned block size grows to fit a recurrent state. An explicit token
    byte rate overrides the model's rate, including caller-normalized geometry.

    State bytes include per-layer page padding, before pool-block rounding.
    ``num_speculative_tokens`` is metadata: it does not change a KDA copy's shape.
    """
    from .config_builders import build_model_config
    from .deepseek_v4_state import BLOCK_SIZE, deepseek_v4_state_cache
    from .models import _architecture_to_model_family, _get_model_info, get_model

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
    if block_size is None:
        block_size = 64
    _integer("block_size", block_size, 2)
    if block_size % 16:
        raise ValueError("K3 automatic block_size requires a multiple of 16 tokens")
    if mamba_cache_dtype not in ("auto", "float16", "float32"):
        raise ValueError("mamba_cache_dtype must be auto, float16 or float32")
    try:
        config = build_model_config(
            tp_size=tp_size,
            pp_size=1,
            attention_dp_size=1,
            moe_tp_size=tp_size,
            moe_ep_size=1,
            kvcache_quant_mode=kvcache_quant_mode,
        )
    except KeyError as error:
        raise ValueError(f"unsupported kvcache_quant_mode: {kvcache_quant_mode!r}") from error
    config.language_only = True
    model = get_model(model_path, config, backend)
    geometry = model.extra_params
    attention_layers = geometry.layer_types.count("full_attention")
    state_layers = geometry.layer_types.count("linear_attention")
    if not attention_layers or not state_layers:
        raise ValueError("K3 state sizing requires both MLA and KDA layers")
    if kv_bytes_per_token is None:
        kv_bytes_per_token = int(model.get_kvcache_elements_per_token() * model.config.kvcache_quant_mode.value.memory)
    _integer("kv_bytes_per_token", kv_bytes_per_token, 1)
    if kv_bytes_per_token % attention_layers:
        raise ValueError("kv_bytes_per_token must divide evenly across MLA layers")

    conv_dtype = mamba_cache_dtype
    if conv_dtype == "auto":
        raw = info["raw_config"]
        text = raw.get("text_config", raw)
        conv_dtype = text.get("dtype", text.get("torch_dtype"))
        if conv_dtype not in ("float16", "bfloat16"):
            raise ValueError("set mamba_cache_dtype explicitly for an unresolved model dtype")
    heads = geometry.kda_num_heads // tp_size
    dim = geometry.kda_head_dim
    # One state, with no legacy five-slot budget multiplier.
    raw_page = heads * (3 * dim * (geometry.kda_conv_kernel - 1) * _WIDTH[conv_dtype] + dim * dim * 4)
    requested_page = block_size * (kv_bytes_per_token // attention_layers)
    block_size *= (raw_page + requested_page - 1) // requested_page
    _integer("block_bytes", block_size * kv_bytes_per_token, 1)
    page = block_size * (kv_bytes_per_token // attention_layers)
    state_bytes = state_layers * page
    _integer("bytes_per_request", state_bytes, 1)
    return {
        "source": "inferred",
        "block_size": block_size,
        "kv_bytes_per_token": kv_bytes_per_token,
        "bytes_per_request": state_bytes,
        "raw_bytes_per_request": state_layers * raw_page,
        "padding_bytes_per_request": state_layers * (page - raw_page),
        "raw_bytes_per_layer": raw_page,
        "padded_bytes_per_layer": page,
        "conv_dtype": conv_dtype,
        "ssm_dtype": "float32",
        "num_speculative_tokens": num_speculative_tokens,
        "backend_revision": VLLM_REVISION,
    }


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or not minimum <= value <= (1 << 64) - 1:
        raise ValueError(f"{name} must be an integer from {minimum} through u64 max")
