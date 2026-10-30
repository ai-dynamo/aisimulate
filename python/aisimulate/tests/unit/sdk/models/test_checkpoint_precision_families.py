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
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def _qwen36_layers(tmp_path, experts_by_layer, shared_by_layer):
    """Qwen3.6-35B-A3B listing with per-layer routed-expert and shared-expert algorithms (None: unlisted)."""
    config, hf_quant = _load(QWEN36_MOE)
    for listing in _listings(config, hf_quant):
        for layer in range(len(_layer_types(config))):
            base = f"model.language_model.layers.{layer}.mlp"
            for key, algo in ((f"{base}.experts", experts_by_layer(layer)),) + tuple(
                (f"{base}.shared_expert.{p}", shared_by_layer(layer)) for p in ("gate_proj", "up_proj", "down_proj")
            ):
                if algo is None:
                    listing.pop(key, None)
                else:
                    listing[key] = {"quant_algo": algo}
    return _write(tmp_path, config, hf_quant), _layer_types(config)


def _decode_rows(path, system="gb200"):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

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
        rows = model.static_phase_diagnostics(batch_size=64, context_length=8192, prefill=False)
    finally:
        model.close()
    latencies = {}
    for row in rows:
        latencies[row["name"]] = latencies.get(row["name"], 0.0) + row["latency_ms"]
    return latencies


def test_qwen_decode_overlap_is_priced_per_precision_signature(monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    # Group A: slow BF16 shared expert (unlisted in a targets-only export) next to NVFP4 experts.
    # Group B: slow BF16 routed experts next to an NVFP4 shared expert. The overlap's slower branch
    # flips between the groups, so a max over pooled precisions would undercount.
    split = 20

    def group_a(layer):
        return layer < split

    mixed, kinds = _qwen36_layers(
        tmp_path / "mixed",
        lambda layer: "NVFP4" if group_a(layer) else None,
        lambda layer: None if group_a(layer) else "NVFP4",
    )
    all_a, _ = _qwen36_layers(tmp_path / "a", lambda layer: "NVFP4", lambda layer: None)
    all_b, _ = _qwen36_layers(tmp_path / "b", lambda layer: None, lambda layer: "NVFP4")
    mixed_rows, a_rows, b_rows = _decode_rows(mixed), _decode_rows(all_a), _decode_rows(all_b)
    for prefix, kind in (("gdn", "linear_attention"), ("full", "full_attention")):
        name = f"generation_{prefix}_moe_overlap"
        layers = [layer for layer, k in enumerate(kinds) if k == kind]
        share_a = sum(group_a(layer) for layer in layers) / len(layers)
        expected = share_a * a_rows[name] + (1 - share_a) * b_rows[name]
        assert mixed_rows[name] == pytest.approx(expected, rel=1e-9), prefix


def test_qwen_routed_experts_are_bucketed_per_layer_kind(tmp_path):
    config, _ = _load(QWEN36_MOE)
    kinds_of = _layer_types(config)
    path, kinds = _qwen36_layers(
        tmp_path,
        lambda layer: "FP8" if kinds_of[layer] == "linear_attention" else "NVFP4",
        lambda layer: "NVFP4",
    )
    model = get_model(path, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="gb200"), "vllm")
    assert _buckets(model, "context_gdn_moe") == {MOE.fp8: kinds.count("linear_attention")}
    assert _buckets(model, "context_full_moe") == {MOE.nvfp4: kinds.count("full_attention")}


def test_listed_descendants_declare_their_module_precision(tmp_path):
    from aisimulate_core.sdk.models.checkpoint_precision import CheckpointPrecision, decoder_paths

    paths = decoder_paths("mixer")
    listing = {
        "backbone.layers.0.mixer.in_proj.weight": {"quant_algo": "FP8"},
        "backbone.layers.0.mixer.experts.0.up_proj": {"quant_algo": "NVFP4"},
        "backbone.layers.0.mixer.experts.1.up_proj": {"quant_algo": "NVFP4"},
        "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"},
        "backbone.layers.0.mixer.out_proj.weight": {"quant_algo": "NVFP4"},
        "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "NVFP4"},
        "backbone.layers.1.mixer.experts.1.up_proj": {"quant_algo": "FP8"},
    }
    checkpoint = CheckpointPrecision(
        {"quantization_config": {"quant_algo": "MIXED_PRECISION", "quantized_layers": listing}}
    )
    assert checkpoint.module(paths(0, "in_proj")) == (False, {"fp8"})
    assert checkpoint.module(paths(0, "experts")) == (False, {"nvfp4"})
    # An exact listing wins over its descendants.
    assert checkpoint.module(paths(0, "out_proj")) == (False, {"fp8"})
    # A module is not covered by an ancestor's listing.
    assert checkpoint.module(paths(0, "in_proj.weight.extra")) == (False, set())
    # Descendants that disagree are a conflict, as for any module.
    assert checkpoint.module(paths(1, "experts"))[1] == {"nvfp4", "fp8"}


def test_descendant_listed_projection_is_priced_through_model_construction(tmp_path):
    from aisimulate_core.sdk.models.nemotron_h import _projection_modes

    listing = {f"backbone.layers.0.mixer.{p}.weight": {"quant_algo": "FP8"} for p in ("in_proj", "out_proj")}
    raw = {"hf_quant_config": {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": listing}}}
    modes = _projection_modes(raw, "M", GEMM.nvfp4)
    assert modes == {"in_proj": {GEMM.fp8_static: 1}, "out_proj": {GEMM.fp8_static: 1}}


def test_qwen_decode_builds_one_overlap_per_precision_signature(tmp_path):
    split = 20
    path, kinds = _qwen36_layers(
        tmp_path,
        lambda layer: "NVFP4" if layer < split else None,
        lambda layer: None if layer < split else "NVFP4",
    )
    model = get_model(path, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="gb200"), "vllm")
    for prefix, kind in (("gdn", "linear_attention"), ("full", "full_attention")):
        groups = {}
        for op in model.generation_ops:
            if op._name != f"generation_{prefix}_moe_overlap":
                continue
            (moe,) = [o for o in op._group_a if o._name.endswith("_moe")]
            (gate_up,) = [o for o in op._group_b if o._name.endswith("_shared_gate_up_gemm")]
            key = (moe._quant_mode, gate_up._quant_mode)
            groups[key] = groups.get(key, 0) + moe._scale_factor
        layers = [layer for layer, k in enumerate(kinds) if k == kind]
        in_a = sum(layer < split for layer in layers)
        scale = model._mtp_scale_factor
        # Each overlap keeps its own layers' routed and shared precision together.
        assert groups == pytest.approx(
            {(MOE.nvfp4, GEMM.bfloat16): in_a * scale, (MOE.bfloat16, GEMM.nvfp4): (len(layers) - in_a) * scale}
        ), prefix


def _qkv_modes(quantization_config, default):
    from aisimulate_core.sdk.models.checkpoint_precision import CheckpointPrecision, decoder_paths, projection_modes

    checkpoint = CheckpointPrecision({"quantization_config": quantization_config})
    qkv = {"full": {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")}}
    return projection_modes(checkpoint, ["full"], qkv, decoder_paths(), default)


def test_fused_modules_compare_their_effective_precision():
    # Targets-only: Q excluded, K/V unlisted. All three load BF16, so the fused QKV is BF16.
    targets_only = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.layers.0.mlp.down_proj": {"quant_algo": "NVFP4"}},
        "ignore": ["model.layers.0.self_attn.q_proj"],
    }
    assert _qkv_modes(targets_only, GEMM.nvfp4) == {"qkv": {GEMM.bfloat16: 1}}
    # With a base lane the export is not targets-only, so an unlisted module keeps the default.
    # Q listed FP8 resolves to the same fp8_static lane as the K/V default: the fused QKV is valid.
    listed = {
        "quant_method": "fp8",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"model.layers.0.self_attn.q_proj": "FP8"},
    }
    assert _qkv_modes(listed, GEMM.fp8_static) == {"qkv": {GEMM.fp8_static: 1}}
    # A real disagreement is still rejected.
    with pytest.raises(ValueError, match="disagree on precision"):
        _qkv_modes(listed, GEMM.nvfp4)


def _block_fp8_qwen(tmp_path):
    """Qwen3.6-35B-A3B geometry with a uniform compressed-tensors block-FP8 config group."""
    config, _ = _load(QWEN36_MOE)
    weights = {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]}
    group = {"targets": ["Linear"], "weights": weights, "input_activations": {"num_bits": 8, "type": "float"}}
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": group},
        "ignore": ["lm_head", "re:.*mlp.gate$", "re:.*shared_expert_gate$"],
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path), _layer_types(config)


@pytest.mark.parametrize("canonical", [False, True])
def test_block_fp8_config_group_keeps_block_lanes(monkeypatch, tmp_path, canonical):
    path, kinds = _block_fp8_qwen(tmp_path)
    if canonical:
        from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, engine

        built = []
        monkeypatch.setattr(engine, "get_model", lambda *args: built.append(get_model(*args)) or built[-1])
        monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
        RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model=path,
                system="h200_sxm",
                backend="vllm",
                backend_version="0.30.0",
                worker_type="decode",
                tp=1,
                moe_tp_size=1,
                moe_ep_size=1,
                database_mode="SOL",
                estimation_mode="op_level",
            )
        ).close()
        (model,) = built
    else:
        model = get_model(path, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="h200_sxm"), "vllm")
    linear = kinds.count("linear_attention")
    for name in ("context_gdn_in_proj_gemm", "context_gdn_out_proj_gemm", "context_gdn_shared_gate_up_gemm"):
        assert _buckets(model, name) == {GEMM.fp8_block: linear}, name
    assert set(_buckets(model, "context_gdn_moe")) == {MOE.fp8_block}
    assert set(_buckets(model, "context_full_moe")) == {MOE.fp8_block}
