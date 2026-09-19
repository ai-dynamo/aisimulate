# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
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
# vllm/model_executor/models/{config,kimi_linear,qwen3_next,qwen3_5}.py,
# vllm/model_executor/layers/kda.py, vllm/transformers_utils/configs/kimi_linear.py,
# vllm/transformers_utils/configs/{qwen3_next,qwen3_5,qwen3_5_moe}.py,
# vllm/platforms/interface.py, vllm/v1/kv_cache_interface.py.

"""One working state per simulated rank, independent of pool-capacity estimation."""

from typing import Any

from .config.engine import EnginePredictionConfig, StateCacheConfig, WorkerPredictionConfig

VLLM_REVISION = "a474da28131f61684849b31e29af0eebaaedc383"
GDN_LAYOUT = "vllm-gdn-a474da28"
KDA_LAYOUT = "vllm-kda-a474da28"
_GDN_TYPES = {"qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"}
_WIDTH = {"float16": 2, "bfloat16": 2, "float32": 4}


def _unsupported(reason: str) -> ValueError:
    return ValueError(f"state_cache inference: {reason}; set bytes_per_request explicitly for this layout")


def _positive(config: dict, name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _unsupported(f"missing or invalid {name}")
    return value


def resolve_state_size(engine: EnginePredictionConfig, worker: WorkerPredictionConfig) -> dict[str, Any]:
    cache = worker.kv_cache
    sizing = cache.state_cache
    if sizing is None:
        return {"source": "disabled", "bytes_per_request": None}
    if sizing.bytes_per_request is not None:
        result = {"source": "overridden", "bytes_per_request": sizing.bytes_per_request}
    else:
        result = _infer(engine, worker)
    size = StateCacheConfig(bytes_per_request=result["bytes_per_request"])
    blocks = size.state_blocks(cache.block_size, cache.bytes_per_token)
    block_bytes = cache.block_size * cache.bytes_per_token
    capacity = cache.capacity.blocks
    if capacity is None:
        capacity = cache.capacity.bytes // block_bytes
    if capacity < blocks + 1:
        raise ValueError("state_cache capacity must fit one token block and one request state")
    return {**result, "state_blocks": blocks, "allocated_bytes_per_request": blocks * block_bytes}


def _infer(engine: EnginePredictionConfig, worker: WorkerPredictionConfig) -> dict[str, Any]:
    from aisimulate_core.sdk.memory import NaiveKVCacheEstimator

    cache = worker.kv_cache
    sizing = cache.state_cache
    assert sizing is not None
    if sizing.layout not in {"auto", GDN_LAYOUT, KDA_LAYOUT}:
        raise _unsupported(f"unknown layout {sizing.layout!r}")
    if worker.parallelism.pipeline != 1:
        raise _unsupported("only PP=1 is supported (stage sizes need not be equal)")
    raw = NaiveKVCacheEstimator._load_config(engine.model, allow_hf_config_download=True)
    if not isinstance(raw, dict):
        raise _unsupported(f"cannot load model config {engine.model!r}")
    config = raw.get("text_config", raw)
    if not isinstance(config, dict) or config.get("model_type") not in _GDN_TYPES | {"kimi_linear"}:
        raise _unsupported("unknown model geometry")
    is_kda = config["model_type"] == "kimi_linear"
    layout = KDA_LAYOUT if is_kda else GDN_LAYOUT
    if sizing.layout not in {"auto", layout}:
        raise _unsupported(f"layout {sizing.layout!r} does not match model geometry ({layout})")
    if is_kda:
        recurrent, attention = _kda_layer_counts(config)
        if engine.speculation is not None:
            raise _unsupported("speculative KDA execution is not qualified for the pinned layout")
    else:
        count = _positive(config, "num_hidden_layers")
        layers = config.get("layer_types")
        if layers is None:
            interval = (
                4
                if config["model_type"] == "qwen3_next"
                else _positive({"full_attention_interval": 4, **config}, "full_attention_interval")
            )
            layers = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]
        if (
            not isinstance(layers, list)
            or len(layers) != count
            or any(layer not in ("linear_attention", "full_attention") for layer in layers)
        ):
            raise _unsupported("invalid layer_types")
        recurrent = layers.count("linear_attention")
        attention = layers.count("full_attention")
        if not recurrent or not attention:
            raise _unsupported("expected hybrid full-attention/GDN layers")
    tp = worker.parallelism.tensor
    model_dtype = sizing.model_dtype
    if model_dtype == "auto":
        model_dtype = config.get("dtype", config.get("torch_dtype", raw.get("dtype", raw.get("torch_dtype"))))
        # vLLM's float32 auto downcast depends on device capabilities. Do not guess.
        if model_dtype not in ("float16", "bfloat16"):
            raise _unsupported("set model_dtype for missing, float32 or unsupported model dtype")
    conv_dtype = model_dtype if sizing.mamba_cache_dtype == "auto" else sizing.mamba_cache_dtype
    num_spec = engine.speculation.num_speculative_tokens if engine.speculation else 0
    if is_kda:
        # The pinned layer reads HF dtype, while the platform calculator reads
        # resolved model dtype. Require an explicit cache dtype when they differ.
        hf_dtype = config.get("dtype", config.get("torch_dtype", raw.get("dtype", raw.get("torch_dtype"))))
        if sizing.mamba_cache_dtype == "auto" and sizing.model_dtype != "auto" and sizing.model_dtype != hf_dtype:
            raise _unsupported("KDA model_dtype changes require an explicit mamba_cache_dtype")
        if sizing.mamba_ssm_cache_dtype not in {"auto", "float32"}:
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
    else:
        kh = _positive(config, "linear_num_key_heads")
        vh = _positive(config, "linear_num_value_heads")
        kd = _positive(config, "linear_key_head_dim")
        vd = _positive(config, "linear_value_head_dim")
        kernel = _positive(config, "linear_conv_kernel_dim")
        if kh % tp or vh % tp:
            raise _unsupported("GDN heads must be divisible by TP")
        ssm_dtype = sizing.mamba_ssm_cache_dtype
        if ssm_dtype == "auto" and config["model_type"] in {"qwen3_5_text", "qwen3_5_moe_text"}:
            # vLLM's model-specific config hook runs before the shared calculator.
            ssm_dtype = config.get("mamba_ssm_dtype", "auto")
            if ssm_dtype is None:
                ssm_dtype = "auto"
        if ssm_dtype == "auto":
            ssm_dtype = conv_dtype
        if not isinstance(ssm_dtype, str) or ssm_dtype not in _WIDTH:
            raise _unsupported("unsupported model mamba_ssm_dtype; set mamba_ssm_cache_dtype")
        conv = (2 * kh * kd + vh * vd) // tp * (kernel - 1 + num_spec) * _WIDTH[conv_dtype]
        temporal = vh // tp * vd * kd * _WIDTH[ssm_dtype]
    raw_page = conv + temporal
    # Uniform full-attention layers: the explicit shared token geometry gives
    # the final attention page per layer. Never enlarge the caller's block size.
    if cache.bytes_per_token % attention:
        raise _unsupported("bytes_per_token must divide evenly across full-attention layers")
    padded_page = cache.block_size * (cache.bytes_per_token // attention)
    if padded_page < raw_page:
        raise _unsupported("attention page is smaller than recurrent state; supply resolved vLLM block geometry")
    return {
        "source": "inferred",
        "bytes_per_request": recurrent * padded_page,
        "layout": layout,
        "vllm_revision": VLLM_REVISION,
        "recurrent_layers_per_rank": recurrent,
        "tensor_parallel": tp,
        "pipeline_parallel": 1,
        "conv_dtype": conv_dtype,
        "ssm_dtype": ssm_dtype,
        "num_speculative_tokens": num_spec,
        "raw_bytes_per_layer": raw_page,
        "padded_bytes_per_layer": padded_page,
    }


def _kda_layer_counts(config: dict) -> tuple[int, int]:
    count = _positive(config, "num_hidden_layers")
    linear = config.get("linear_attn_config")
    if not isinstance(linear, dict):
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
