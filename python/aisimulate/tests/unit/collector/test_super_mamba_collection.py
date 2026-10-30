# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path

import pytest
from collector.case_generator import get_common_moe_test_cases, moe_model_allows_quantization
from collector.vllm import collect_mamba2
from collector.vllm.collect_mamba2 import _decode_query_start_loc, get_mamba2_test_cases

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


def test_super_runtime_pin_keeps_the_legacy_data_family():
    from collector.framework_manifest import require_collector_runtime
    from collector.op_catalog import load_family_map
    from collector.vllm.registry import REGISTRY

    runtime = require_collector_runtime("vllm", "0.31.0", requested_ops={"mamba2", "moe"}, model_path=MODEL)
    assert runtime.version == "0.31.0"
    assert runtime.source_commit == "db9527a46873454610df6dbedf79a36d6bf1a7f6"
    assert load_family_map()["mamba2_perf"] == "linear_attention"
    assert next(entry for entry in REGISTRY if entry.op == "mamba2").unverified


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
        "_uses_modelopt_nvfp4_checkpoint",
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
    assert helpers["_modelopt_mixed_expert_prefix"](actual, "nemotron_h") == "model.layers.1.mixer.experts"
    layers["model.layers.1.mixer.experts.0.down_proj"] = {"quant_algo": "FP8"}
    with pytest.raises(ValueError, match="uniformly NVFP4"):
        helpers["_modelopt_mixed_expert_prefix"](quant, "nemotron_h")


@pytest.mark.parametrize("model_type", ["deepseek_v4", "qwen3_5_moe", "minimax_m3"])
def test_other_model_families_keep_the_existing_moe_quantization_path(model_type):
    quant = {"quant_method": "modelopt_mixed"}
    assert moe_helpers()["_modelopt_mixed_expert_prefix"](quant, model_type) is None


def test_mixed_collection_requires_per_layer_metadata():
    with pytest.raises(ValueError, match="complete quantized_layers"):
        moe_helpers()["_modelopt_mixed_expert_prefix"]({"quant_method": "modelopt_mixed"}, "nemotron_h")


def test_uniform_nemotron_nvfp4_uses_modelopt_and_missing_metadata_fails(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "nemotron_h"}))
    helpers = moe_helpers()
    with pytest.raises(ValueError, match="complete checkpoint quantization metadata"):
        helpers["_uses_modelopt_nvfp4_checkpoint"](str(tmp_path))
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "kv_cache_quant_algo": "FP8",
                    "exclude_modules": [],
                }
            }
        )
    )
    assert helpers["_uses_modelopt_nvfp4_checkpoint"](str(tmp_path))


def test_decode_passes_cumulative_lengths_only_for_speculative_widths():
    starts = object()
    assert _decode_query_start_loc(1, starts) is None
    assert _decode_query_start_loc(2, starts) is starts
    assert _decode_query_start_loc(4, starts) is starts
    # Both decode kernels take the serving value, never the raw varlen offsets.
    tree = ast.parse(Path(collect_mamba2.__file__).read_text())
    passed = {
        (keyword.arg, ast.unparse(keyword.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg in ("query_start_loc", "cu_seqlens")
    }
    assert passed == {("query_start_loc", "decode_starts"), ("cu_seqlens", "decode_starts")}
