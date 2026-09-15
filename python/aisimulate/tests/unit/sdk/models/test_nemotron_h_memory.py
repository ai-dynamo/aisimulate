# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemotronH hybrid KV memory: attention-only per-token KV plus a constant Mamba2 state."""

import pytest

from aiconfigurator_core.sdk import common, config, models
from aiconfigurator_core.sdk.memory import NaiveKVCacheEstimator

pytestmark = pytest.mark.unit

# 88 layers = 40 'M' + 40 'E' + 8 '*'; 2 KV heads x 128 head_dim;
# Mamba2: 128 heads x 64 head_dim, d_state 128, conv_kernel 4, 8 groups;
# checkpoint declares mamba_ssm_cache_dtype=float32, model dtype bf16.
SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
SUPER_ATTN_LAYERS, SUPER_MAMBA_LAYERS = 8, 40


def _model(model_path=SUPER, tp_size=1, **kwargs):
    model_config = config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=tp_size,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        **kwargs,
    )
    return models.get_model(model_path, model_config, "vllm")


def _super_attn_bytes_per_token(tp_size):
    return SUPER_ATTN_LAYERS * 2 * -(-2 // tp_size) * 128  # fp8 KV: 1 byte/elem


def _super_mamba_state_bytes(tp_size, ssm_dtype_bytes):
    nheads, n_groups = 128 // tp_size, max(8 // tp_size, 1)
    ssm = nheads * 64 * 128 * ssm_dtype_bytes
    conv = (nheads * 64 + 2 * n_groups * 128) * (4 - 1) * 2  # conv state in model dtype
    return SUPER_MAMBA_LAYERS * (ssm + conv)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_per_token_kv_counts_attention_layers_only(tp_size):
    model = _model(tp_size=tp_size)
    per_token = _super_attn_bytes_per_token(tp_size)
    static = model.get_kvcache_static_bytes_per_sequence()

    assert model.get_kvcache_elements_per_token() == per_token
    assert model.get_kvcache_bytes_per_sequence(1) == static + per_token
    assert model.get_kvcache_bytes_per_sequence(4096) == static + 4096 * per_token


@pytest.mark.parametrize(
    ("override", "ssm_dtype_bytes"),
    [(None, 4), ("float32", 4), ("float16", 2), ("bfloat16", 2), ("auto", 2)],
)
def test_mamba_state_bytes_match_closed_form(override, ssm_dtype_bytes):
    # None -> checkpoint's float32; an explicit ModelConfig override wins.
    model = _model(mamba_ssm_cache_dtype=override)
    assert model.get_kvcache_static_bytes_per_sequence() == _super_mamba_state_bytes(1, ssm_dtype_bytes)


def test_mamba_state_is_tp_sharded():
    assert _model(tp_size=2).get_kvcache_static_bytes_per_sequence() == _super_mamba_state_bytes(2, 4)


def test_mamba_state_fp16_is_about_82_mib():
    assert _model(mamba_ssm_cache_dtype="float16").get_kvcache_static_bytes_per_sequence() == 86_343_680


def test_max_tokens_reserves_static_state_first():
    model = _model()
    static = model.get_kvcache_static_bytes_per_sequence()
    per_token = _super_attn_bytes_per_token(1)
    budget = 64 * 2**30

    assert model.get_kvcache_max_tokens(static + 100 * per_token) == 100
    assert model.get_kvcache_max_tokens(static) == 0
    assert model.get_kvcache_max_tokens(budget) == int((budget - static) // per_token)


def test_unknown_mamba_ssm_cache_dtype_raises():
    with pytest.raises(ValueError, match="mamba_ssm_cache_dtype"):
        _model(mamba_ssm_cache_dtype="fp32").get_kvcache_static_bytes_per_sequence()


def test_checkpoint_without_ssm_dtype_uses_model_dtype():
    # Nemotron-H-56B: 118 layers = 54 'M' + 10 '*' + 54 '-', no MoE, no
    # mamba_ssm_cache_dtype key -> "auto" -> bf16. Mamba2: 256 heads x 64, d_state 256, conv 4, 8 groups.
    model = _model("nvidia/Nemotron-H-56B-Base-8K")
    ssm = 256 * 64 * 256 * 2
    conv = (256 * 64 + 2 * 8 * 256) * 3 * 2
    assert model.get_kvcache_static_bytes_per_sequence() == 54 * (ssm + conv)
    assert model.get_kvcache_elements_per_token() == 10 * 2 * 8 * model._head_size


def test_non_hybrid_model_is_unchanged():
    model = models.get_model(
        "Qwen/Qwen3-32B",
        config.ModelConfig(tp_size=1, pp_size=1, kvcache_quant_mode=common.KVCacheQuantMode.fp8),
        "vllm",
    )
    per_token = model.get_kvcache_bytes_per_sequence(1)

    assert model.get_kvcache_static_bytes_per_sequence() == 0
    assert per_token == model.get_kvcache_elements_per_token()
    assert model.get_kvcache_bytes_per_sequence(4096) == 4096 * per_token
    assert model.get_kvcache_max_tokens(1000 * per_token + 1) == 1000


def test_naive_estimator_counts_attention_layers_only():
    estimator = NaiveKVCacheEstimator.from_model_path(SUPER, tp_size=1, pp_size=1, allow_hf_config_download=False)
    # bf16 model dtype on the HF-config-only path: 2 bytes/elem.
    assert estimator.kv_bytes_per_token() == SUPER_ATTN_LAYERS * 2 * 2 * 128 * 2
