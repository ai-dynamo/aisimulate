# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer dense hybrid sliding-window/global-attention contracts."""

from collections import Counter

import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.models.base import _MODEL_REGISTRY
from aiconfigurator.sdk.models.muse_glimmer import MuseGlimmerModel
from aiconfigurator.sdk.task_v2 import Task
from aiconfigurator.sdk.utils import _parse_hf_config_json

pytestmark = pytest.mark.unit

LAYER_TYPES = ("sliding_attention", "sliding_attention", "sliding_attention", "full_attention") * 13


def _model_config(*, tp_size=1, cp_size=1):
    return config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        attention_dp_size=1,
        cp_size=cp_size,
        cp_style="allgather" if cp_size > 1 else "none",
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )


def _hf_config(*, layer_types=LAYER_TYPES, sliding_window=2048):
    return {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": len(layer_types),
            "hidden_size": 6656,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 19968,
            "vocab_size": 202048,
            "max_position_embeddings": 131072,
            "layer_types": list(layer_types),
            "sliding_window": sliding_window,
        },
    }


def _build(*, tp_size=1, cp_size=1):
    return get_model(
        "meta-models/Muse-Glimmer-30B",
        _model_config(tp_size=tp_size, cp_size=cp_size),
        "sglang",
    )


def _expected_phase_names(phase):
    expected = Counter(
        {
            f"{phase}_embedding": 1,
            f"{phase}_embedding_ar": 1,
            f"{phase}_attention": 2,
            f"{phase}_ar_1": 1,
            f"{phase}_ar_2": 1,
            f"{phase}_logits_gemm": 1,
            f"{phase}_p2p": 1,
        }
    )
    for kind in ("swa", "global"):
        for suffix in (
            "attn_norm",
            "qkv_gemm",
            "gate_gemm",
            "gate_mul",
            "proj_gemm",
            "ffn_norm",
            "mlp_gate_up_gemm",
            "mlp_act",
            "mlp_down_gemm",
        ):
            expected[f"{phase}_{kind}_{suffix}"] = 1
    return expected


def test_muse_config_parses_and_registers_dedicated_family():
    parsed = _parse_hf_config_json(_hf_config())

    assert common.ARCHITECTURE_TO_MODEL_FAMILY[parsed["architecture"]] == "MUSEGLIMMER"
    assert common.MULTIMODAL_TEXT_CONFIG_KEY[parsed["architecture"]] == "text_config"
    assert _MODEL_REGISTRY["MUSEGLIMMER"] is MuseGlimmerModel
    assert parsed["layers"] == 52
    assert (parsed["n"], parsed["n_kv"], parsed["d"]) == (32, 2, 128)
    assert (parsed["hidden_size"], parsed["inter_size"], parsed["vocab"]) == (6656, 19968, 202048)
    assert (parsed["topk"], parsed["num_experts"]) == (0, 0)
    assert parsed["extra_params"] == common.MuseGlimmerConfig(LAYER_TYPES, 2048)


@pytest.mark.parametrize(
    ("layer_types", "sliding_window", "message"),
    [
        (("sliding_attention",), 2048, "layer_types length"),
        (("sliding_attention", "linear_attention"), 2048, "must contain only"),
        (LAYER_TYPES, 0, "positive sliding_window"),
    ],
)
def test_muse_config_rejects_invalid_hybrid_layout(layer_types, sliding_window, message):
    document = _hf_config(layer_types=layer_types, sliding_window=sliding_window)
    if len(layer_types) == 1:
        document["text_config"]["num_hidden_layers"] = 52
    with pytest.raises(ValueError, match=message):
        _parse_hf_config_json(document)


@pytest.mark.parametrize(("phase", "attribute"), [("context", "context_ops"), ("generation", "generation_ops")])
def test_phase_graph_has_exact_operation_multiset(phase, attribute):
    phase_ops = getattr(_build(), attribute)

    assert len(phase_ops) == 26
    assert Counter(op._name for op in phase_ops) == _expected_phase_names(phase)


def test_attention_windows_scales_and_qk_norm_are_exact():
    model = _build()
    context = [op for op in model.context_ops if isinstance(op, ops.ContextAttention)]
    generation = [op for op in model.generation_ops if isinstance(op, ops.GenerationAttention)]

    assert len(context) == 2
    assert len(generation) == 2
    assert sorted((op._window_size, op._scale_factor, op._use_qk_norm) for op in context) == [
        (0, 13.0, True),
        (2048, 39.0, True),
    ]
    assert sorted((op._window_size, op._scale_factor, op._use_qk_norm) for op in generation) == [
        (0, 13.0, True),
        (2048, 39.0, True),
    ]


def test_gate_gemms_have_exact_count_names_and_shapes():
    model = _build()
    gates = [
        op
        for op in model.context_ops + model.generation_ops
        if isinstance(op, ops.GEMM) and op._name.endswith("_gate_gemm")
    ]
    expected_names = {
        "context_swa_gate_gemm",
        "context_global_gate_gemm",
        "generation_swa_gate_gemm",
        "generation_global_gate_gemm",
    }

    assert len(gates) == 4
    assert Counter(op._name for op in gates) == Counter(expected_names)
    assert {(op._n, op._k) for op in gates} == {(4096, 6656)}


def test_allreduces_have_exact_count_names_and_scales():
    model = _build(tp_size=8)
    allreduces = [op for op in model.context_ops + model.generation_ops if isinstance(op, ops.CustomAllReduce)]
    expected_scales = {
        "context_embedding_ar": 1.0,
        "context_ar_1": 52.0,
        "context_ar_2": 52.0,
        "generation_embedding_ar": 1.0,
        "generation_ar_1": 52.0,
        "generation_ar_2": 52.0,
    }

    assert len(allreduces) == 6
    assert Counter(op._name for op in allreduces) == Counter(expected_scales.keys())
    assert {op._name: op._scale_factor for op in allreduces} == expected_scales


def test_embedding_weights_are_sharded_across_tensor_parallel_ranks():
    model = _build(tp_size=8)
    expected_bytes_per_rank = (202048 // 8) * 6656 * common.GEMMQuantMode.bfloat16.value.memory

    for phase_ops, name in (
        (model.context_ops, "context_embedding"),
        (model.generation_ops, "generation_embedding"),
    ):
        embeddings = [op for op in phase_ops if isinstance(op, ops.Embedding) and op._name == name]
        assert len(embeddings) == 1
        assert embeddings[0].get_weights() == expected_bytes_per_rank


def test_kv_cache_caps_only_sliding_window_layers():
    model = _build()

    assert model.get_kvcache_elements_per_token() == 2 * 52 * 2 * 128
    assert model.get_kvcache_bytes_per_sequence(1024) == float(52 * 1024 * 1024)
    expected = 39 * 1024 * 2048 + 13 * 1024 * 8192
    assert model.get_kvcache_bytes_per_sequence(8192) == float(expected)
    assert model.get_kvcache_max_tokens(float(expected)) == 8192


def test_context_parallelism_emits_one_uniform_gather_and_splits_allreduces():
    model = _build(cp_size=2)
    attention = [op for op in model.context_ops if isinstance(op, ops.ContextAttention)]
    gathers = [op for op in model.context_ops if isinstance(op, ops.NCCL) and op._name == "context_cp_all_gather"]
    allreduces = [op for op in model.context_ops if isinstance(op, ops.CustomAllReduce)]

    assert len(attention) == 2
    assert all(op._cp_size == 2 for op in attention)
    assert len(gathers) == 1
    assert gathers[0]._scale_factor == 52
    assert len(allreduces) == 3
    assert Counter(op._name for op in allreduces) == Counter(
        {"context_embedding_ar": 1, "context_ar_1": 1, "context_ar_2": 1}
    )
    assert all(op._seq_split == 2 for op in allreduces)


def test_sglang_default_sweep_keeps_tensor_and_context_parallelism_separate():
    task = Task(
        serving_mode="agg",
        model_path="meta-models/Muse-Glimmer-30B",
        system_name="b300_sxm",
        backend_name="sglang",
        backend_version="0.5.14",
        total_gpus=32,
    )

    parallel = list(task.iter_parallel("agg"))
    assert [8, 1, 1, 1, 1, 1] in parallel
    assert [1, 1, 1, 1, 1, 8] in parallel
    assert all(cp == 1 or (tp == 1 and dp == 1) for tp, _pp, dp, _moe_tp, _moe_ep, cp in parallel)


def test_muse_uses_dense_activation_tier_on_every_backend():
    from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
    from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
    from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend

    assert TRTLLMBackend.ACTIVATION_COEFFICIENTS["MUSEGLIMMER"] == TRTLLMBackend.ACTIVATION_COEFFICIENTS["LLAMA"]
    assert SGLANGBackend.ACTIVATION_COEFFICIENTS["MUSEGLIMMER"] == SGLANGBackend.ACTIVATION_COEFFICIENTS["LLAMA"]
    assert VLLMBackend.ACTIVATION_COEFFICIENTS["MUSEGLIMMER"] == TRTLLMBackend.ACTIVATION_COEFFICIENTS["MUSEGLIMMER"]
