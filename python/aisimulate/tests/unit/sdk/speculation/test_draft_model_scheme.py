# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_draft_model_scheme.py

"""Standalone draft-model scheme (classic two-model speculative decoding).

The draft is instantiated through the model registry, so its op graph,
weights and KV accounting come from the draft model itself — validated
here with the real Qwen/Qwen3-0.6B drafting for Qwen/Qwen3-8B."""

from __future__ import annotations

import pytest

from aisimulate_core.sdk import common, models
from aisimulate_core.sdk import config as sdk_config
from aisimulate_core.sdk.speculation import SpeculationConfig
from aisimulate_core.sdk.speculation.draft_model import DraftModelScheme

pytestmark = pytest.mark.unit


def _q8b(**params):
    cfg = sdk_config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        speculation=SpeculationConfig(kind="draft_model", params=params, draft_model_path="Qwen/Qwen3-0.6B"),
    )
    return models.get_model("Qwen/Qwen3-8B", cfg, "vllm")


class TestDraftModelScheme:
    def test_param_resolution(self):
        model = _q8b(num_speculative_tokens=3)
        scheme = model.spec_scheme
        assert isinstance(scheme, DraftModelScheme)
        assert scheme.verify_width() == 4
        assert scheme.draft_model_path == "Qwen/Qwen3-0.6B"

    def test_k_required(self):
        with pytest.raises(ValueError, match="num_speculative_tokens"):
            _q8b()

    def test_draft_ops_are_the_full_small_model_times_k(self):
        model = _q8b(num_speculative_tokens=3)
        specs = model.spec_scheme.build_draft_generation_ops(model)
        by_name = {s.op._name: s for s in specs}
        # Full model graph: embedding + layers + logits head all present.
        assert "generation_embedding" in by_name
        assert "generation_logits_gemm" in by_name
        # Three full forwards, each retaining the checkpoint's 28 layers.
        qkv = [s.op for s in specs if s.op._name == "generation_qkv_gemm"]
        assert len(qkv) == 3
        assert all(op._scale_factor == 28 for op in qkv)
        # Draft geometry is the 0.6B's, not the target's: qkv n at h=1024,
        # 16 q heads + 2x8 kv heads, head_dim 128.
        assert by_name["generation_qkv_gemm"].op._k == 1024
        assert all(s.tokens_per_request == 1 for s in specs)

    def test_weights_are_the_draft_checkpoint(self):
        model = _q8b(num_speculative_tokens=3)
        w = model.spec_scheme.draft_weights_bytes(model)
        # Qwen3-0.6B bf16 ~= 1.50 GB; K-scaling must not inflate weights.
        assert w == pytest.approx(1.503e9, rel=0.02)
        model5 = _q8b(num_speculative_tokens=5)
        w5 = model5.spec_scheme.draft_weights_bytes(model5)
        assert w5 == pytest.approx(w)

    def test_kv_uses_draft_models_own_accounting(self):
        model = _q8b(num_speculative_tokens=3)
        kv = model.spec_scheme.draft_kv_bytes_per_sequence(model, 1_000)
        # 28 layers x 2(K+V) x 8 kv heads x 128 head_dim x 2 B (bf16)
        assert kv == pytest.approx(28 * 1_000 * 2 * 8 * 128 * 2)

    def test_context_ops_prefill_the_draft(self):
        model = _q8b(num_speculative_tokens=3)
        specs = model.spec_scheme.build_draft_context_ops(model)
        names = [s.op._name for s in specs]
        assert "context_attention" in names

    def test_unsupported_backend_rejected(self):
        cfg = sdk_config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            speculation=SpeculationConfig(
                kind="draft_model",
                params={"num_speculative_tokens": 3},
                draft_model_path="Qwen/Qwen3-0.6B",
            ),
        )
        with pytest.raises(ValueError, match="backends"):
            models.get_model("Qwen/Qwen3-8B", cfg, "trtllm")


def test_target_layer_override_does_not_change_independent_draft():
    baseline = _q8b(num_speculative_tokens=3)
    import dataclasses

    cfg = dataclasses.replace(baseline.config, overwrite_num_layers=2)
    target = models.get_model("Qwen/Qwen3-8B", cfg, "vllm")
    independent = models.get_model(
        "Qwen/Qwen3-0.6B",
        dataclasses.replace(cfg, speculation=None, overwrite_num_layers=0),
        "vllm",
    )
    scheme = target.spec_scheme
    assert target._num_layers == 2
    assert scheme._draft_model._num_layers == independent._num_layers == 28
    assert scheme.draft_weights_bytes(target) == sum(op.get_weights() for op in independent.generation_ops)
    assert scheme.draft_weights_bytes(target) == baseline.spec_scheme.draft_weights_bytes(baseline)
    for seq_len in (1, 1000, 8192):
        assert scheme.draft_kv_bytes_per_sequence(target, seq_len) == independent.get_kvcache_bytes_per_sequence(
            seq_len
        )


@pytest.mark.parametrize("draft_tp", [None, 1, 2])
def test_moe_draft_parallelism_resolves_after_draft_tp_override(draft_tp):
    cfg = sdk_config.ModelConfig(
        tp_size=2,
        pp_size=1,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=4,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        speculation=SpeculationConfig(
            kind="draft_model",
            params={"num_speculative_tokens": 3, "draft_tp_size": draft_tp},
            draft_model_path="Qwen/Qwen3-30B-A3B",
        ),
    )
    target = models.get_model("Qwen/Qwen3-32B", cfg, "vllm")
    draft = target.spec_scheme._draft_model
    assert draft.config.tp_size == (draft_tp or 2)
    assert draft.config.moe_tp_size == draft.config.tp_size
    assert draft.config.moe_ep_size == 2
    assert draft.config.attn_width == draft.config.moe_tp_size * draft.config.moe_ep_size
    assert (cfg.moe_tp_size, cfg.moe_ep_size) == (1, 4)
