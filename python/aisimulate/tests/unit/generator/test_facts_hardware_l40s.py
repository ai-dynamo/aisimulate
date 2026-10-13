# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ada (sm89) hardware facts: the l40s profile, and the quantization-keyed trtllm MoE backend.

Without the profile resolve_facts() raised "Unknown hardware profile 'l40s'" and the pipeline swallowed
it (no model default reached l40s renders). On SM89 trtllm 1.3.0rc29 serves NVFP4 MoE only through
MARLIN, which rejects bf16/fp8 MoE layers, so the profile keys the choice by quantization family
(op-probe harness L40 campaign, 2026-10-04, findings sm89_trtllm_rc29_probe_2026_10_04)."""

import pytest

from aisimulate.generator.facts.apply import apply_moe_backend, quant_family_of
from aisimulate.generator.facts.request_resolution import hardware_key_for_system
from aisimulate.generator.facts.resolve import _FACTS_DIR, load_backend_version_matrix, resolve_facts

pytestmark = pytest.mark.unit


def _latest_dynamo() -> str:
    matrix = load_backend_version_matrix(str(_FACTS_DIR / "runtimes" / "dynamo.yaml"))
    return next(v for v, backends in matrix.items() if "trtllm" in backends)


def _l40s():
    return resolve_facts(model_profile_id=None, hardware="l40s", transport="nvlink",
                         dynamo_version=_latest_dynamo(), backend="trtllm").hardware


def test_l40s_is_its_own_profile_and_no_phantom_l40_system():
    assert hardware_key_for_system("l40s") == "l40s"
    assert hardware_key_for_system("l40") == "l40"  # not an SDK system: falls through, resolve_facts raises clearly
    hw = _l40s()
    assert hw["node_selector"]["nvidia.com/gpu.product"] == "NVIDIA-L40S"
    assert hw["moe_backend"] == {"trtllm": {"nvfp4": "MARLIN"}}


@pytest.mark.parametrize("quant, family", [
    (None, None),
    ({"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8"}, "nvfp4"),          # modelopt hf_quant_config
    ({"quant_method": "mxfp4"}, "mxfp4"),                                       # gpt-oss
    ({"quant_method": "fp8", "weight_block_size": [128, 128]}, "fp8_block"),    # DeepSeek-V3 class
    ({"quant_method": "fp8"}, "fp8"),
    ({"quant_algo": "FP8"}, "fp8"),
    # modelopt MIXED_PRECISION: the experts' algo decides (nvidia/Qwen3.8-2.4T-A95B-NVFP4 layout)
    ({"quant_algo": "MIXED_PRECISION", "quant_method": "modelopt", "quantized_layers": {
        "model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"}, "model.layers.0.linear_attn.out_proj": {"quant_algo": "FP8"},
        "model.layers.0.mlp.experts": {"quant_algo": "NVFP4"}}}, "nvfp4"),
    ({"quant_algo": "MIXED_PRECISION", "quant_method": "modelopt", "quantized_layers": {
        "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}, "model.layers.0.mlp.down_proj": {"quant_algo": "FP8"}}}, "fp8"),
])
def test_quant_family_of(quant, family):
    assert quant_family_of(quant) == family


def test_l40s_fills_marlin_for_nvfp4_moe_only():
    hw = _l40s()
    ctx = {"is_moe": True, "moe_config": {}}
    apply_moe_backend(ctx, hw, backend="trtllm", quantization={"quant_algo": "NVFP4"})
    assert ctx["moe_config"]["backend"] == "MARLIN"
    for quant in (None, {"quant_method": "fp8", "weight_block_size": [128, 128]}, {"quant_method": "mxfp4"}):
        ctx = {"is_moe": True, "moe_config": {}}
        apply_moe_backend(ctx, hw, backend="trtllm", quantization=quant)
        assert "backend" not in ctx["moe_config"], quant   # the template default (CUTLASS) applies
    dense = {"is_moe": False, "moe_config": {}}
    apply_moe_backend(dense, hw, backend="trtllm", quantization={"quant_algo": "NVFP4"})
    assert "backend" not in dense["moe_config"]


def test_user_moe_backend_param_beats_the_hardware_fill_on_trtllm():
    """`--generator-set params.agg.moe_backend=...` used to be silently dropped on trtllm: the hardware fact
    filled moe_config.backend regardless and the template never read the generic param."""
    hw = {"moe_backend": {"trtllm": "CUTLASS"}}
    ctx = {"is_moe": True, "moe_config": {}, "agg_moe_backend": "marlin", "moe_backend": "marlin"}
    apply_moe_backend(ctx, hw, backend="trtllm")
    assert "backend" not in ctx["moe_config"]
    ctx = {"is_moe": True, "moe_config": {}}
    apply_moe_backend(ctx, hw, backend="trtllm")
    assert ctx["moe_config"]["backend"] == "CUTLASS"
