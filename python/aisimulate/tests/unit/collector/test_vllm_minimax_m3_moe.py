# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 0.31 native MoE argument and mixed-precision prefix contracts."""

import ast
from pathlib import Path

import pytest

from .test_getter_deduplication import _install_vllm_stubs, _load_collector, _stub_module

pytestmark = pytest.mark.unit

MODELS = ("MiniMaxAI/MiniMax-M3", "nvidia/MiniMax-M3-NVFP4")


@pytest.fixture
def collector(monkeypatch):
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    monkeypatch.setattr(module, "vllm_version", "0.31.0")
    return module


@pytest.mark.parametrize("model", MODELS)
def test_native_packed_activation_and_default_beta(collector, model):
    runtime = collector._resolve_moe_runtime_config(model, {})
    assert runtime["activation"] == "swigluoai_uninterleave"
    assert (runtime["swiglu_limit"], runtime["swiglu_alpha"], runtime["swiglu_beta"]) == (7.0, 1.702, 1.0)
    assert runtime["scoring_func"] == "sigmoid" and runtime["renormalize"]
    assert runtime["use_routing_bias"] and runtime["router_logits_float32"]
    assert runtime["routed_scaling_factor"] == 2.0
    assert not runtime["apply_routed_scale_to_output"]


@pytest.mark.parametrize("version", ["0.24.0", "0.27.1", "0.30.0"])
def test_older_versions_keep_prior_constructor(collector, monkeypatch, version):
    monkeypatch.setattr(collector, "vllm_version", version)
    runtime = collector._resolve_moe_runtime_config(MODELS[1], {})
    assert runtime["activation"] == "swigluoai" and runtime["swiglu_limit"] is None
    assert "swiglu_alpha" not in runtime and "swiglu_beta" not in runtime
    assert not collector._uses_minimax_m3_031(MODELS[1])


def test_gpt_oss_keeps_interleaved_layout(collector):
    module_config = collector.get_moe_quantization_module_config("vllm", "w4a16_mxfp4", model_name="openai/gpt-oss-20b")
    runtime = collector._resolve_moe_runtime_config("openai/gpt-oss-20b", module_config)
    assert runtime["activation"] == "swigluoai"
    assert "swiglu_alpha" not in runtime and "swiglu_beta" not in runtime


def test_explicit_activation_parameters_preserved(collector, monkeypatch):
    original = collector._load_model_moe_config
    monkeypatch.setattr(
        collector,
        "_load_model_moe_config",
        lambda name: {
            **original(name),
            "swiglu_alpha": 1.6,
            "swiglu_beta": 0.7,
            "swiglu_limit": 6.0,
        },
    )
    runtime = collector._resolve_moe_runtime_config(MODELS[1], {})
    assert (runtime["swiglu_limit"], runtime["swiglu_alpha"], runtime["swiglu_beta"]) == (6.0, 1.6, 0.7)


def test_checkpoint_prefix_reaches_real_nvfp4_experts(collector):
    checkpoint, prefix = collector._minimax_m3_nvfp4_checkpoint(MODELS[1])
    assert prefix == "language_model.model.layers.3.block_sparse_moe.experts"
    assert (checkpoint["quant_method"], checkpoint["quant_algo"]) == ("modelopt", "MIXED_PRECISION")
    entries = {key: value for key, value in checkpoint["quantized_layers"].items() if key.startswith(prefix + ".")}
    assert len(entries) == 128 * 3
    assert all(value == {"quant_algo": "NVFP4", "group_size": 16} for value in entries.values())


@pytest.mark.parametrize("defect", ["no_experts", "wrong_algo", "wrong_group", "wrong_checkpoint"])
def test_incompatible_checkpoint_raises_without_backend_fallback(collector, monkeypatch, defect):
    checkpoint = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "language_model.model.layers.3.block_sparse_moe.experts.0.w1": {"quant_algo": "NVFP4", "group_size": 16},
        },
    }
    entry = next(iter(checkpoint["quantized_layers"].values()))
    if defect == "no_experts":
        checkpoint["quantized_layers"] = {}
    elif defect == "wrong_algo":
        entry["quant_algo"] = "MXFP8"
    elif defect == "wrong_group":
        entry["group_size"] = 32
    else:
        checkpoint["quant_method"] = "compressed-tensors"
    monkeypatch.setattr(collector, "_load_checkpoint_quantization_config", lambda _: checkpoint)
    with pytest.raises(ValueError, match="checkpoint-native"):
        collector._minimax_m3_nvfp4_checkpoint(MODELS[1])


def test_quantization_branch_uses_native_mixed_config_and_actual_prefix(collector, monkeypatch):
    seen = []

    class NativeMixedConfig:
        @classmethod
        def from_config(cls, config):
            seen.append(config)
            return cls()

    _stub_module(
        monkeypatch, "vllm.model_executor.layers.quantization.modelopt", ModelOptMixedPrecisionConfig=NativeMixedConfig
    )
    tree = ast.parse(Path(collector.__file__).read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_moe_torch")
    quant_branch = next(
        node for node in run.body if isinstance(node, ast.If) and ast.unparse(node.test) == "moe_type == 'bfloat16'"
    )
    namespace = {**vars(collector), "moe_type": "nvfp4", "model_name": MODELS[1], "moe_prefix": "collector_moe"}
    exec(compile(ast.Module(body=[quant_branch], type_ignores=[]), collector.__file__, "exec"), namespace)
    assert isinstance(namespace["quant_config"], NativeMixedConfig)
    assert namespace["moe_prefix"] == "language_model.model.layers.3.block_sparse_moe.experts"
    assert seen[0]["quant_algo"] == "MIXED_PRECISION"
    factory = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FusedMoE"
    )
    assert next(keyword for keyword in factory.keywords if keyword.arg == "prefix").value.id == "moe_prefix"
    runtime = collector._resolve_moe_runtime_config(MODELS[1], {})
    expansion = next(
        keyword.value for keyword in factory.keywords if keyword.arg is None and isinstance(keyword.value, ast.DictComp)
    )
    kwargs = eval(compile(ast.Expression(expansion), collector.__file__, "eval"), {"runtime_config": runtime})
    assert kwargs == {"swiglu_alpha": 1.702, "swiglu_beta": 1.0}


def test_repair_preserves_moe_case_arguments(collector, monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(collector, "get_sm_version", lambda: 100)
    current = collector.get_moe_test_cases()
    monkeypatch.setattr(collector, "_uses_minimax_m3_031", lambda _: False)
    previous = collector.get_moe_test_cases()
    assert current == previous
