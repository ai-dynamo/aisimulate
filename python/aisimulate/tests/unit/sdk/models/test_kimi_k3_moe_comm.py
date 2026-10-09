# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 per-MoE-layer TP collectives: attention AR plus the latent combine."""

import json

import pytest

from aisimulate.sdk import config
from aisimulate.sdk.models import get_model
from aisimulate.sdk.operations import CustomAllReduce

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("is_context", [True, False])
def test_kimi_k3_moe_layer_does_not_recharge_attention_allreduce(backend, is_context):
    """Pre-dispatch is free (the attention AR is priced separately); the
    combine stays one all-reduce in the 3584-wide latent space."""
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.perf_database import get_database_view

    model = get_model(
        "moonshotai/Kimi-K3",
        config.ModelConfig(tp_size=8, attention_dp_size=1, moe_tp_size=8, moe_ep_size=1),
        backend,
    )
    db = get_database_view("gb300", backend, "current", allow_missing_data=True, database_mode="SOL")
    phase = "context" if is_context else "generation"
    ops = {op._name: op for op in (model.context_ops if is_context else model.generation_ops)}
    assert {f"{phase}_kda_ar", f"{phase}_mla_ar"} <= ops.keys()

    def cost(op):
        return float(_evaluate_single_op(db, op, is_context=is_context, batch_size=1, s=64, prefix=0, x=64))

    pre = ops[f"{phase}_moe_pre_dispatch"]
    assert json.loads(pre._spec_json())["MoeDispatch"]["attn_ar_modeled"]
    assert cost(pre) == 0.0

    spec = json.loads(ops[f"{phase}_moe_post_dispatch"]._spec_json())["MoeDispatch"]
    assert spec["hidden_size"] == 3584
    reference = CustomAllReduce("reference", spec["scale_factor"], spec["hidden_size"], 8)
    assert cost(ops[f"{phase}_moe_post_dispatch"]) == pytest.approx(cost(reference), rel=1e-12)
    assert cost(reference) > 0
