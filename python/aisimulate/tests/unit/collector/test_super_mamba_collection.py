# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path

import pytest
from collector.case_generator import get_common_moe_test_cases, moe_model_allows_quantization
from collector.vllm.collect_mamba2 import get_mamba2_test_cases

pytestmark = pytest.mark.unit
MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"


def test_super_latent_moe_is_not_filtered_out(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    assert moe_model_allows_quantization("vllm", MODEL, "nvfp4")
    cases = get_common_moe_test_cases(backend="vllm")
    assert any(
        (c.hidden_size, c.inter_size, c.topk, c.num_experts, c.tp, c.ep) == (1024, 2688, 22, 512, 2, 1) for c in cases
    )
    assert all(c.hidden_size == 1024 for c in cases)


def test_mamba_collection_covers_tp_and_mtp_without_combining_shards(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    cases = get_mamba2_test_cases()
    decode = [case for case in cases if case[0] == "generation"]
    assert {(c[8], c[9]) for c in decode} == {(tp, width) for tp in (1, 2, 4, 8) for width in (1, 2, 3, 4)}
    assert all(c[4] == 128 // c[8] and c[6] == 8 // c[8] for c in cases)
    assert all(c[13:] == ["float16", True, 5] for c in cases)


def moe_helpers():
    # Extract only metadata helpers so the contract runs without GPU dependencies.
    path = Path(__file__).resolve().parents[3] / "collector/vllm/collect_moe.py"
    names = {
        "_model_config_path",
        "_load_model_moe_config",
        "_load_checkpoint_quantization_config",
        "_modelopt_mixed_expert_prefix",
    }
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"Path": Path, "json": json, "_MODEL_CONFIG_ROOT": path.parent / "missing"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_mixed_checkpoint_uses_language_config_and_companion_quantization(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "llm_config": {"model_type": "nemotron_h", "mlp_hidden_act": "relu2"},
                "quantization_config": {"quant_method": "modelopt_mixed"},
            }
        )
    )
    layers = {"model.layers.1.mixer.experts.0.up_proj": {"quant_algo": "NVFP4"}}
    quant = {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": layers}}
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant))
    helpers = moe_helpers()
    assert helpers["_load_model_moe_config"](str(tmp_path))["mlp_hidden_act"] == "relu2"
    actual = helpers["_load_checkpoint_quantization_config"](str(tmp_path))
    assert actual == quant
    assert helpers["_modelopt_mixed_expert_prefix"](actual) == "model.layers.1.mixer.experts"
    layers["model.layers.1.mixer.experts.0.down_proj"] = {"quant_algo": "FP8"}
    with pytest.raises(ValueError, match="uniformly NVFP4"):
        helpers["_modelopt_mixed_expert_prefix"](quant)
