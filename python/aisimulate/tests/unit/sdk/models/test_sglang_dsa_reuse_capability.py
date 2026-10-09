# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real GLM FP8 defaults must not borrow another precision's reuse capability."""

import json

import pytest

from aisimulate.sdk import common, config, engine, models
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("system", ["b300_sxm", "gb200", "gb300"])
@pytest.mark.parametrize("prefill", [True, False])
def test_native_glm_fp8_query_survives_bf16_only_reuse_rows(system, prefill):
    # These bundled tables carry FP8-block full rows and BF16-only skip rows.
    # The old file-wide capability probe attempted a nonexistent FP8 skip
    # query. A matching skip slice is required before enabling amortization.
    request = ForwardPassPerfModelConfig(
        model="zai-org/GLM-5.2-FP8",
        system=system,
        backend="sglang",
        backend_version="0.5.14",
        worker_type="aggregated",
        tp=8,
        moe_tp_size=1,
        moe_ep_size=8,
        kvcache_quant_mode="fp8",
        estimation_mode="op_level",
    )
    model = RustForwardPassPerfModel.best_available(request)
    try:
        rows = model.static_phase_diagnostics(batch_size=1, context_length=1024, prefill=prefill)
    finally:
        model.close()
    phase = "context" if prefill else "generation"
    attention = next(row for row in rows if row["name"] == f"{phase}_attention")
    assert attention["source"] == "silicon"
    assert attention["latency_ms"] > 0


@pytest.mark.parametrize("quant", [common.GEMMQuantMode.fp8, common.GEMMQuantMode.nvfp4])
@pytest.mark.parametrize("phase", ["context", "generation"])
def test_explicit_attention_quant_override_is_not_replaced_by_checkpoint_default(quant, phase):
    # The SDK permits hypothetical precision studies. Missing silicon for an
    # explicit mode must not be hidden by changing the requested projection
    # dtype back to BF16 or the checkpoint's FP8-block default.
    model = models.get_model(
        "zai-org/GLM-5.2-FP8",
        config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8, gemm_quant_mode=quant),
        backend_name="sglang",
    )
    specs = json.loads(engine._ops_json(getattr(model, f"{phase}_ops")))
    attention = next(fields for spec in specs for tag, fields in spec.items() if tag == f"Dsa{phase.capitalize()}")
    assert attention["gemm_quant_mode"] == quant.name
