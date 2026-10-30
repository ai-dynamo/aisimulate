# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer checkpoint precision for Qwen3.5/3.6/3.8 and Gemma 4 (shared checkpoint_precision module)."""

import json
from importlib.resources import files

import pytest

from aisimulate_core.sdk import common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models import get_model

pytestmark = pytest.mark.unit

GEMM = common.GEMMQuantMode
MOE = common.MoEQuantMode
QWEN36_MOE = "nvidia--Qwen3.6-35B-A3B-NVFP4"
GEMMA_DENSE = "nvidia--Gemma-4-31B-IT-NVFP4"
GEMMA_MOE = "nvidia--Gemma-4-26B-A4B-NVFP4"


def _load(name):
    root = files("aisimulate_core") / "model_configs"
    config = json.loads((root / f"{name}_config.json").read_text())
    hf_quant = json.loads((root / f"{name}_hf_quant_config.json").read_text())
    return config, hf_quant


def _write(tmp_path, config, hf_quant):
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(hf_quant))
    return str(tmp_path)


def _layer_types(config):
    return config.get("text_config", config)["layer_types"]


def _listings(config, hf_quant):
    """Every ``quantized_layers`` listing of a checkpoint (config.json and hf_quant_config.json)."""
    sections = [hf_quant.get("quantization", {}), config.get("quantization_config") or {}]
    return [section["quantized_layers"] for section in sections if "quantized_layers" in section]


def _buckets(model, name):
    """``{mode: summed layer count}`` of the context ops called ``name``."""
    buckets = {}
    for op in model.context_ops:
        if op._name == name:
            buckets[op._quant_mode] = buckets.get(op._quant_mode, 0) + op._scale_factor
    return buckets


def _qwen36_split(tmp_path, nvfp4_linear_layers, fp8_moe_layers):
    """Qwen3.6-35B-A3B with some GDN in_proj layers NVFP4 and some routed-expert layers FP8."""
    config, hf_quant = _load(QWEN36_MOE)
    linear = [i for i, kind in enumerate(_layer_types(config)) if kind == "linear_attention"][:nvfp4_linear_layers]
    moe = list(range(len(_layer_types(config))))[:fp8_moe_layers]
    for listing in _listings(config, hf_quant):
        for layer in linear:
            for module in ("in_proj_qkv", "in_proj_z"):
                listing[f"model.language_model.layers.{layer}.linear_attn.{module}"] = {"quant_algo": "NVFP4"}
        for layer in moe:
            listing[f"model.language_model.layers.{layer}.mlp.experts"] = {"quant_algo": "FP8"}
    return _write(tmp_path, config, hf_quant), _layer_types(config)


@pytest.mark.parametrize(("system", "nvfp4"), [("gb200", GEMM.nvfp4), ("h200_sxm", GEMM.nvfp4_wo)])
def test_qwen_prices_each_gdn_and_expert_layer_at_its_checkpoint_precision(tmp_path, system, nvfp4):
    path, layer_types = _qwen36_split(tmp_path, nvfp4_linear_layers=10, fp8_moe_layers=8)
    model = get_model(path, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system=system), "vllm")
    linear = layer_types.count("linear_attention")

    assert _buckets(model, "context_gdn_in_proj_gemm") == {nvfp4: 10, GEMM.fp8_static: linear - 10}
    # in_proj_b/in_proj_a are unlisted in this targets-only export, so vLLM loads in_proj_ba unquantized.
    assert _buckets(model, "context_gdn_in_proj_ba_gemm") == {GEMM.bfloat16: linear}
    assert _buckets(model, "context_gdn_out_proj_gemm") == {GEMM.fp8_static: linear}
    # The first eight layers' experts are FP8; the rest keep the checkpoint's W4A16_NVFP4.
    experts = {**_buckets(model, "context_gdn_moe")}
    for mode, count in _buckets(model, "context_full_moe").items():
        experts[mode] = experts.get(mode, 0) + count
    assert experts == {MOE.fp8: 8, MOE.w4a16_nvfp4: len(layer_types) - 8}
    dispatch = {op._quant_mode for op in model.context_ops if "moe_pre_dispatch" in op._name}
    assert dispatch == {MOE.fp8, MOE.w4a16_nvfp4}


def test_qwen_explicit_modes_bypass_checkpoint_precision(tmp_path):
    path, _ = _qwen36_split(tmp_path, nvfp4_linear_layers=10, fp8_moe_layers=8)
    config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, gemm_quant_mode=GEMM.bfloat16, moe_quant_mode=MOE.fp8)
    model = get_model(path, config, "vllm")
    for name in ("context_gdn_in_proj_gemm", "context_qkv_gemm", "context_gdn_shared_gate_up_gemm"):
        assert set(_buckets(model, name)) == {GEMM.bfloat16}, name
    assert set(_buckets(model, "context_gdn_moe")) == {MOE.fp8}


def test_qwen38_without_per_layer_metadata_keeps_one_op_per_projection():
    model = get_model("Qwen/Qwen3.8-2.4T-A95B-FP8", ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8), "vllm")
    for name in ("context_gdn_in_proj_gemm", "context_qkv_gemm", "context_gdn_moe"):
        buckets = _buckets(model, name)
        assert len(buckets) == 1, (name, buckets)


@pytest.mark.parametrize(("system", "nvfp4"), [("gb200", GEMM.nvfp4), ("h200_sxm", GEMM.nvfp4_wo)])
def test_gemma_prices_attention_per_layer_from_exclusions(tmp_path, system, nvfp4):
    config, hf_quant = _load(GEMMA_DENSE)
    layer_types = _layer_types(config)
    excluded = range(20)
    ignore = [f"model.language_model.layers.{layer}.self_attn*" for layer in excluded]
    ignore += [p for p in config["quantization_config"]["ignore"] if "self_attn" not in p]
    config["quantization_config"]["ignore"] = ignore
    hf_quant["quantization"]["exclude_modules"] = ignore
    model = get_model(
        _write(tmp_path, config, hf_quant), ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system=system), "vllm"
    )

    swa_excluded = layer_types[:20].count("sliding_attention")
    global_excluded = layer_types[:20].count("full_attention")
    assert _buckets(model, "context_swa_qkv_gemm") == {
        GEMM.bfloat16: swa_excluded,
        nvfp4: layer_types.count("sliding_attention") - swa_excluded,
    }
    assert _buckets(model, "context_global_proj_gemm") == {
        GEMM.bfloat16: global_excluded,
        nvfp4: layer_types.count("full_attention") - global_excluded,
    }
    # The shared dense MLP is quantized in every layer.
    assert _buckets(model, "context_swa_shared_mlp_gate_up_gemm") == {nvfp4: layer_types.count("sliding_attention")}


def test_gemma_global_layers_without_v_proj_take_their_listed_precision(tmp_path):
    config, hf_quant = _load(GEMMA_MOE)
    text = config.get("text_config", config)
    assert text["attention_k_eq_v"]
    listing = {}
    for layer, kind in enumerate(text["layer_types"]):
        projections = (
            ("q_proj", "k_proj", "v_proj", "o_proj") if kind == "sliding_attention" else ("q_proj", "k_proj", "o_proj")
        )
        for projection in projections:
            listing[f"model.language_model.layers.{layer}.self_attn.{projection}"] = {"quant_algo": "FP8"}
        listing[f"model.language_model.layers.{layer}.experts.0.gate_proj"] = {"quant_algo": "NVFP4"}
        listing[f"model.language_model.layers.{layer}.experts.0.down_proj"] = {"quant_algo": "NVFP4"}
    section = {"quant_algo": "MIXED_PRECISION", "quantized_layers": listing}
    config["quantization_config"] = {"quant_method": "modelopt", **section}
    hf_quant["quantization"] = section
    model = get_model(_write(tmp_path, config, hf_quant), ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")

    # Global layers reuse K as V: q_proj and k_proj alone carry the fused QKV precision.
    assert set(_buckets(model, "context_global_qkv_gemm")) == {GEMM.fp8_static}
    assert set(_buckets(model, "context_swa_qkv_gemm")) == {GEMM.fp8_static}
    # Unlisted in a targets-only export, the shared dense MLP runs unquantized.
    assert set(_buckets(model, "context_swa_shared_mlp_gate_up_gemm")) == {GEMM.bfloat16}
    assert set(_buckets(model, "context_swa_moe")) == {MOE.nvfp4}


@pytest.mark.parametrize("system", ["h200_sxm", "gb200"])
@pytest.mark.parametrize("family", ["qwen", "gemma"])
def test_canonical_api_prices_per_layer_precision(monkeypatch, tmp_path, system, family):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    if family == "qwen":
        path, _ = _qwen36_split(tmp_path, nvfp4_linear_layers=10, fp8_moe_layers=8)
    else:
        path = "nvidia/Gemma-4-26B-A4B-NVFP4"
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=path,
            system=system,
            backend="vllm",
            backend_version="0.30.0",
            worker_type="decode",
            tp=1,
            moe_tp_size=1,
            moe_ep_size=1,
            database_mode="SOL",
            estimation_mode="op_level",
        )
    )
    try:
        rows = model.static_phase_diagnostics(batch_size=8, context_length=1024, prefill=False)
    finally:
        model.close()
    assert rows and all(row["latency_ms"] >= 0 for row in rows)
