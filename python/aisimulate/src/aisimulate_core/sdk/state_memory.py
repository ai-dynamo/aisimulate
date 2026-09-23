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
# vllm/model_executor/models/kimi_linear.py,
# vllm/model_executor/layers/kda.py, vllm/transformers_utils/configs/kimi_linear.py,
# vllm/platforms/interface.py, vllm/v1/kv_cache_interface.py.

"""Config-driven per-request state memory estimation, shared by all consumers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VLLM_REVISION = "a474da28131f61684849b31e29af0eebaaedc383"
KDA_LAYOUT = "vllm-kda-a474da28"
_WIDTH = {"float16": 2, "bfloat16": 2, "float32": 4}


def _unsupported(reason: str) -> ValueError:
    return ValueError(f"state_cache inference: {reason}; set bytes_per_request explicitly for this layout")


def _positive(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _unsupported(f"missing or invalid {name}")
    return value


def estimate_state_cache(
    model_path: str | None = None,
    *,
    backend: str,
    block_size: int,
    kv_bytes_per_token: int,
    model_config: Mapping[str, Any] | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    model_dtype: str = "auto",
    mamba_cache_dtype: str = "auto",
    mamba_ssm_cache_dtype: str = "auto",
    num_speculative_tokens: int = 0,
    layout: str = "auto",
    allow_hf_config_download: bool = False,
) -> dict[str, Any]:
    """Estimate bytes for ONE recurrent state per request per simulated rank.

    Supply exactly one of ``model_path`` (local/cached HF configuration or a
    model ID) and ``model_config`` (an already loaded HF config mapping).
    Backend, TP/PP, state dtypes and speculation describe the engine layout.
    ``block_size`` and ``kv_bytes_per_token`` are its explicit, resolved
    per-rank token geometry, including any caller-authored normalization.

    The result contains ``bytes_per_request`` after backend page padding but
    BEFORE simulator block rounding, raw/padding bytes, and resolved layout,
    dtype and revision diagnostics. ``vllm_revision`` is retained as a legacy
    diagnostic alias of ``backend_revision`` for the vLLM implementation.

    This memory API is independent of KV-pool capacity, hardware timing data,
    CLI configuration, and GPU vLLM runtime. It never reserves a pool or adds
    checkpoint copies. Network config loading is opt-in, as in estimate_kv_cache.

    Currently qualified: Kimi-K3/KimiLinear with the pinned vLLM KDA layout, TP with divisible
    heads, PP=1. Unsupported backend/model/layout combinations raise ValueError;
    unknown models never silently become zero state. Caller policy owns manual
    overrides, disabling state, and pool-capacity checks.
    """
    if backend != "vllm":
        raise _unsupported(f"unsupported backend {backend!r}")
    for name, value, minimum in (
        ("tp_size", tp_size, 1),
        ("pp_size", pp_size, 1),
        ("block_size", block_size, 2),
        ("kv_bytes_per_token", kv_bytes_per_token, 1),
        ("num_speculative_tokens", num_speculative_tokens, 0),
    ):
        _integer(name, value, minimum=minimum)
    if block_size * kv_bytes_per_token > (1 << 64) - 1:
        raise ValueError("state cache block byte size overflows u64")
    for name, value, allowed in (
        ("model_dtype", model_dtype, ("auto", "float16", "bfloat16", "float32")),
        ("mamba_cache_dtype", mamba_cache_dtype, ("auto", "float16", "float32")),
        ("mamba_ssm_cache_dtype", mamba_ssm_cache_dtype, ("auto", "float16", "float32")),
    ):
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    if not isinstance(layout, str) or layout not in {"auto", KDA_LAYOUT}:
        raise _unsupported(f"unknown layout {layout!r}")
    if pp_size != 1:
        raise _unsupported("only PP=1 is supported (stage sizes need not be equal)")
    if not isinstance(allow_hf_config_download, bool):
        raise ValueError("allow_hf_config_download must be a bool")
    if (model_path is None) == (model_config is None):
        raise ValueError("provide exactly one of model_path and model_config")
    if model_config is not None:
        if not isinstance(model_config, Mapping):
            raise ValueError("model_config must be an HF config mapping")
        raw = model_config
    else:
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a nonempty string")
        raw = _load_state_config(model_path, allow_hf_config_download=allow_hf_config_download)
        if not isinstance(raw, Mapping):
            raise _unsupported(f"cannot load model config {model_path!r}")
    config = raw.get("text_config", raw)
    if not isinstance(config, Mapping) or config.get("model_type") != "kimi_linear":
        raise _unsupported("unknown model geometry; only Kimi-K3/KimiLinear KDA is supported")
    recurrent, attention = _kda_layer_counts(config)
    if num_speculative_tokens:
        raise _unsupported("speculative KDA execution is not qualified for the pinned layout")
    tp = tp_size
    authored_model_dtype = model_dtype
    if model_dtype == "auto":
        model_dtype = config.get("dtype", config.get("torch_dtype", raw.get("dtype", raw.get("torch_dtype"))))
        # vLLM's float32 auto downcast depends on device capabilities. Do not guess.
        if model_dtype not in ("float16", "bfloat16"):
            raise _unsupported("set model_dtype for missing, float32 or unsupported model dtype")
    conv_dtype = model_dtype if mamba_cache_dtype == "auto" else mamba_cache_dtype
    # The pinned layer reads HF dtype, while the platform calculator reads
    # resolved model dtype. Require an explicit cache dtype when they differ.
    hf_dtype = config.get("dtype", config.get("torch_dtype", raw.get("dtype", raw.get("torch_dtype"))))
    if mamba_cache_dtype == "auto" and authored_model_dtype != "auto" and authored_model_dtype != hf_dtype:
        raise _unsupported("KDA model_dtype changes require an explicit mamba_cache_dtype")
    if mamba_ssm_cache_dtype not in {"auto", "float32"}:
        raise _unsupported("the pinned KDA layout always uses float32 recurrent state")
    ssm_dtype = "float32"
    linear = config["linear_attn_config"]
    heads = _positive(linear, "num_heads")
    dim = _positive(linear, "head_dim")
    kernel = _positive(linear, "short_conv_kernel_size")
    if heads % tp:
        raise _unsupported("KDA heads must be divisible by TP")
    if linear.get("num_k_heads", heads) != heads or linear.get("head_k_dim", dim) != dim:
        raise _unsupported("asymmetric KDA heads are not supported by the pinned model layout")
    # Three distinct conv tensors and ONE FP32 matrix. Snapshot copies are
    # managed by the engine, not multiplied into this working-state size.
    conv = 3 * (heads // tp) * dim * (kernel - 1) * _WIDTH[conv_dtype]
    temporal = (heads // tp) * dim * dim * _WIDTH[ssm_dtype]
    raw_page = conv + temporal
    # Uniform full-attention layers: the explicit shared token geometry gives
    # the final attention page per layer. Never enlarge the caller's block size.
    if kv_bytes_per_token % attention:
        raise _unsupported("bytes_per_token must divide evenly across full-attention layers")
    padded_page = block_size * (kv_bytes_per_token // attention)
    if padded_page < raw_page:
        raise _unsupported("attention page is smaller than recurrent state; supply resolved vLLM block geometry")
    bytes_per_request = recurrent * padded_page
    _integer("bytes_per_request", bytes_per_request, minimum=1)
    return {
        "source": "inferred",
        "backend": backend,
        "backend_revision": VLLM_REVISION,
        "bytes_per_request": bytes_per_request,
        "raw_bytes_per_request": recurrent * raw_page,
        "padding_bytes_per_request": recurrent * (padded_page - raw_page),
        "layout": KDA_LAYOUT,
        "vllm_revision": VLLM_REVISION,
        "recurrent_layers_per_rank": recurrent,
        "tensor_parallel": tp,
        "pipeline_parallel": 1,
        "conv_dtype": conv_dtype,
        "ssm_dtype": ssm_dtype,
        "num_speculative_tokens": num_speculative_tokens,
        "raw_bytes_per_layer": raw_page,
        "padded_bytes_per_layer": padded_page,
    }


def _integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= (1 << 64) - 1:
        raise ValueError(f"{name} must be an integer from {minimum} through u64 max")


def _kda_layer_counts(config: Mapping[str, Any]) -> tuple[int, int]:
    count = _positive(config, "num_hidden_layers")
    linear = config.get("linear_attn_config")
    if not isinstance(linear, Mapping):
        raise _unsupported("missing linear_attn_config for KDA")
    groups = []
    for name in ("kda_layers", "full_attn_layers"):
        ids = linear.get(name)
        if (
            not isinstance(ids, list)
            or not ids
            or any(isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= count for i in ids)
        ):
            raise _unsupported(f"{name} must contain valid 1-based layer IDs")
        if len(set(ids)) != len(ids):
            raise _unsupported(f"duplicate IDs in {name}")
        groups.append(set(ids))
    if groups[0] & groups[1] or groups[0] | groups[1] != set(range(1, count + 1)):
        raise _unsupported("KDA and full-attention layer IDs must partition every model layer")
    return len(groups[0]), len(groups[1])


def _load_state_config(model_path: str, *, allow_hf_config_download: bool) -> Mapping[str, Any] | None:
    """Use the shared HF loaders without requiring membership in the FPM model set."""
    from pathlib import Path

    from .utils import (
        _download_hf_config,
        _get_model_config_path,
        _load_local_config,
        _load_pre_downloaded_hf_config,
    )

    if Path(model_path).is_dir():
        return _load_local_config(model_path)
    cached_path = _get_model_config_path() / f"{model_path.replace('/', '--')}_config.json"
    if cached_path.is_file():
        return _load_pre_downloaded_hf_config(model_path)
    if allow_hf_config_download:
        return _download_hf_config(model_path)
    return None
