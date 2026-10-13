# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA (Kimi Delta Attention) models on vLLM: the decode batch is capped at 256.

vLLM 0.29.0 / SM90 faults at the replay of the 512-batch decode CUDA graph
for Kimi-K3 (64/256/384/448/480/496 pass, 512 fails, graphs off at 512
passes); owner decision 2026-09-25: cap max_batch_size at 256 for models
declaring linear_attn_config.kda_layers. The fact enters the rules as
ModelConfig.has_kda from both render paths (module_bridge, naive).
"""
import pytest

from aisimulate.generator.rendering.rule_engine import apply_rule_plugins
from aisimulate.generator.utils import model_has_kda

pytestmark = pytest.mark.unit


def _pv(has_kda, batch=512, preserve=False):
    return {
        "SlaConfig": {"isl": 4000, "osl": 500},
        "DynConfig": {"mode": "agg"},
        "ModelConfig": {"is_moe": True, "has_kda": has_kda},
        "params": {"agg": {"max_batch_size": batch, "tokens_per_block": 32,
                           **({"preserve_engine_limits": True} if preserve else {})}},
    }


def test_kda_model_decode_batch_capped_at_256_and_budgets_follow():
    pv = _pv(True)
    apply_rule_plugins(pv, backend="vllm")
    agg = pv["params"]["agg"]
    assert agg["max_batch_size"] == 256
    # token budgets derive from the CAPPED batch (256 + 4000 + 1500 = 5756 -> 5760)
    assert agg["max_num_tokens"] == 5760


def test_non_kda_model_keeps_the_512_floor():
    pv = _pv(False, batch=256)
    apply_rule_plugins(pv, backend="vllm")
    assert pv["params"]["agg"]["max_batch_size"] == 512  # the pre-existing floor, untouched
    pv = _pv(None, batch=256)  # fact absent (older callers): no cap
    apply_rule_plugins(pv, backend="vllm")
    assert pv["params"]["agg"]["max_batch_size"] == 512


def test_small_requested_batch_is_not_raised_by_the_cap():
    pv = _pv(True, batch=64)
    apply_rule_plugins(pv, backend="vllm")
    # the 512 floor lifts 64 -> 512, the KDA cap brings it to 256 (never above the cap)
    assert pv["params"]["agg"]["max_batch_size"] == 256


def test_preserve_engine_limits_disables_the_cap():
    pv = _pv(True, batch=512, preserve=True)
    apply_rule_plugins(pv, backend="vllm")
    assert pv["params"]["agg"]["max_batch_size"] == 512


def test_model_has_kda_reads_the_bundled_config():
    assert model_has_kda("moonshotai/Kimi-K3") is True
    assert model_has_kda("Qwen/Qwen3-8B") is False
    assert model_has_kda("org/does-not-exist-anywhere") is False
