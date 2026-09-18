# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 carries the ``attention_backend`` lane override into its attention ops.

AIC-1715: ``ModelConfig.attention_backend`` is the user-facing knob that heads the
attention lane precedence order. Since the pyo3 op unification, models are built
WITHOUT a database handle (pure shape graphs); ops no longer store the override
directly (``ContextAttention``/``GenerationAttention`` have no Python
``__init__``). The knob rides ``model.config.attention_backend`` — read
generically for every model family, not per-op — until
``engine.py::_resolve_attention_lane_orders`` resolves it (with a database) at
spec-build time and sets each op's ``_lane_order``. This file checks both
halves: the knob survives model construction, and resolution heads the walk
with it.
"""

import json

import pytest

from aisimulate.sdk import common
from aisimulate.sdk import config as sdk_config
from aisimulate.sdk.models.qwen35 import Qwen35Model

pytestmark = pytest.mark.unit


def _qwen35_config():
    """Two-layer hybrid (one GDN + one full-attention) dense Qwen3.5."""
    return common.Qwen35Config(
        layer_types=("linear_attention", "full_attention"),
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


def _build_model(attention_backend):
    model_config = sdk_config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        attention_backend=attention_backend,
    )
    return Qwen35Model(
        "Qwen/Qwen3.5-Test",
        "QWEN35",
        "Qwen35ForCausalLM",
        2,  # num_layers
        32,  # num_heads
        8,  # num_kv_heads
        128,  # head_size
        4096,  # hidden_size
        12288,  # inter_size
        151936,  # vocab_size
        32768,  # context_length
        model_config,
        _qwen35_config(),
        backend_name="sglang",
    )


def _attention_ops(model):
    from aisimulate.sdk.operations.attention import ContextAttention, GenerationAttention

    ctx = [op for op in model.context_ops if isinstance(op, ContextAttention)]
    gen = [op for op in model.generation_ops if isinstance(op, GenerationAttention)]
    assert ctx and gen, "the full-attention layer must produce both attention ops"
    return ctx, gen


def test_attention_backend_reaches_both_attention_ops():
    """The knob survives model construction and heads BOTH ops' resolved lane
    order once ``build_engine_spec_json`` resolves it against a real database
    (context AND generation, not just one side) — the modern equivalent of the
    retired per-op ``_attention_backend`` attribute check. A database is
    required: resolution reads backend/version/sm_version/systems_root off it;
    without one, a named override is rejected rather than silently discarded."""
    from aisimulate.sdk.perf_database import get_database
    from aisimulate_core.sdk.engine import _resolve_attention_lane_orders

    model = _build_model("trtllm_mha")
    assert model.config.attention_backend == "trtllm_mha"
    ctx_ops, gen_ops = _attention_ops(model)

    database = get_database("b200_sxm", "sglang", "0.5.14")
    _resolve_attention_lane_orders(ctx_ops, database, model.config.attention_backend)
    _resolve_attention_lane_orders(gen_ops, database, model.config.attention_backend)

    assert all(op._lane_order[0] == "trtllm_mha" for op in ctx_ops)
    assert all(op._lane_order[0] == "trtllm_mha" for op in gen_ops)


def test_attention_backend_defaults_to_no_override():
    """Unset knob means no override: the framework-default lane heads the order
    (a ``None`` database still resolves the always-valid ``["default"]``)."""
    from aisimulate_core.sdk.engine import _resolve_attention_lane_orders

    model = _build_model(None)
    assert model.config.attention_backend is None
    ctx_ops, gen_ops = _attention_ops(model)

    _resolve_attention_lane_orders(ctx_ops, None, model.config.attention_backend)
    _resolve_attention_lane_orders(gen_ops, None, model.config.attention_backend)

    assert all(op._lane_order == ["default"] for op in ctx_ops)
    assert all(op._lane_order == ["default"] for op in gen_ops)


def test_model_config_attention_backend_defaults_to_none():
    """``ModelConfig`` must not force a lane: the default is "no override"."""
    assert sdk_config.ModelConfig().attention_backend is None


# ---------------------------------------------------------------------------
# AIC-1762: per-architecture attention-backend default (final-review finding
# I1). ``model.architecture`` is threaded through ``_resolve_attention_lane_
# orders`` exactly like ``model.config.attention_backend`` above -- read once
# off the model, not per-op -- so this is the model-construction half of the
# same proof. Verified against the REAL shipped SGLang 0.5.17 databases (not
# synthetic fixtures): Qwen3.8-Max's own architecture string resolves
# trtllm_mha on GB300 and B200; the 397B sibling architecture (undeclared in
# attention_lane_defaults.yaml's architectures: section) stays on the
# inherited triton default, byte-identical to the pre-AIC-1762 walk.
# ---------------------------------------------------------------------------


def test_architecture_default_reaches_both_attention_ops_without_override():
    from aisimulate.sdk import models
    from aisimulate.sdk.perf_database import get_database
    from aisimulate_core.sdk.engine import _resolve_attention_lane_orders

    def _real_model_config():
        return sdk_config.ModelConfig(
            tp_size=8,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        )

    max_model = models.get_model("Qwen/Qwen3.8-2.4T-A95B", _real_model_config(), "sglang")
    condgen_model = models.get_model("Qwen/Qwen3.5-397B-A17B", _real_model_config(), "sglang")
    assert max_model.architecture == "Qwen3_5MoeForCausalLM"
    assert condgen_model.architecture == "Qwen3_5MoeForConditionalGeneration"

    database = get_database("gb300", "sglang", "0.5.17")

    max_ctx, max_gen = _attention_ops(max_model)
    condgen_ctx, condgen_gen = _attention_ops(condgen_model)

    _resolve_attention_lane_orders(max_ctx, database, None, max_model.architecture)
    _resolve_attention_lane_orders(max_gen, database, None, max_model.architecture)
    _resolve_attention_lane_orders(condgen_ctx, database, None, condgen_model.architecture)
    _resolve_attention_lane_orders(condgen_gen, database, None, condgen_model.architecture)

    assert all(op._lane_order[0] == "trtllm_mha" for op in max_ctx), [op._lane_order for op in max_ctx]
    assert all(op._lane_order[0] == "trtllm_mha" for op in max_gen), [op._lane_order for op in max_gen]
    assert all(op._lane_order[0] == "triton" for op in condgen_ctx), [op._lane_order for op in condgen_ctx]
    assert all(op._lane_order[0] == "triton" for op in condgen_gen), [op._lane_order for op in condgen_gen]


@pytest.mark.parametrize(
    ("attention_backend", "expected_lane"),
    [(None, "trtllm_mha"), ("default", "trtllm_mha"), ("triton", "triton")],
    ids=["unset", "default", "named-override"],
)
def test_architecture_default_is_serialized_by_build_engine_spec_json(attention_backend, expected_lane):
    """Exercise architecture and override propagation through the public spec builder."""
    from aisimulate.sdk import models
    from aisimulate.sdk.perf_database import get_database
    from aisimulate_core.sdk.engine import build_engine_spec_json

    model_config = sdk_config.ModelConfig(
        tp_size=16,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=16,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        attention_backend=attention_backend,
    )
    model_path = "Qwen/Qwen3.8-2.4T-A95B-FP8"
    model = models.get_model(model_path, model_config, "sglang")
    database = get_database("b200_sxm", "sglang", "0.5.17")
    assert database.system_spec["gpu"]["sm_version"] == 100

    spec = json.loads(
        build_engine_spec_json(
            model,
            model_path=model_path,
            system="b200_sxm",
            backend="sglang",
            backend_version="0.5.17",
            kv_block_size=None,
            systems_path=None,
            nextn=0,
            database=database,
        )
    )
    context_orders = [op["ContextAttention"]["lane_order"] for op in spec["context_ops"] if "ContextAttention" in op]
    generation_orders = [
        op["GenerationAttention"]["lane_order"] for op in spec["generation_ops"] if "GenerationAttention" in op
    ]

    assert context_orders and generation_orders
    assert all(order[0] == expected_lane for order in context_orders)
    assert all(order[0] == expected_lane for order in generation_orders)


def test_architecture_default_yields_to_an_explicit_override():
    """Explicit override still wins outright, even for Max's own architecture
    default -- explicit intent stays first-class (owner design, same
    precedence rule ``resolve_attention_lane_tiers`` documents)."""
    from aisimulate.sdk import models
    from aisimulate.sdk.perf_database import get_database
    from aisimulate_core.sdk.engine import _resolve_attention_lane_orders

    model_config = sdk_config.ModelConfig(
        tp_size=8,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        attention_backend="triton",
    )
    max_model = models.get_model("Qwen/Qwen3.8-2.4T-A95B", model_config, "sglang")
    database = get_database("gb300", "sglang", "0.5.17")

    max_ctx, max_gen = _attention_ops(max_model)
    _resolve_attention_lane_orders(max_ctx, database, model_config.attention_backend, max_model.architecture)
    _resolve_attention_lane_orders(max_gen, database, model_config.attention_backend, max_model.architecture)

    assert all(op._lane_order[0] == "triton" for op in max_ctx), [op._lane_order for op in max_ctx]
    assert all(op._lane_order[0] == "triton" for op in max_gen), [op._lane_order for op in max_gen]
