# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks execute real getters; they do not claim native GPU qualification."""

import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.attention_memory import filter_attention_cases
from collector.case_generator import get_mla_module_sweep_spec
from collector.module_memory import module_footprint_bytes, module_model_config, validate_module_position

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[3] / "collector"


def _function(backend, name, namespace):
    path = COLLECTOR / backend / "collect_mla_module.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_dense_mla_full_history_anchors_preserve_sparse_query_chunks(backend):
    namespace = {
        "get_mla_module_sweep_spec": get_mla_module_sweep_spec,
        "_get_precision_combos": lambda *_args: [("bfloat16", "bfloat16", "bfloat16")],
    }
    if backend == "vllm":
        _function(backend, "_dsa_context_budgets", namespace)
        _function(backend, "_dsa_context_point_allowed", namespace)
    getter = _function(backend, "get_context_test_cases", namespace)
    mla = getter("mla")
    assert {24576, 32768, 40960, 65536, 131072, 262144, 524288, 1048576} <= {case[0] for case in mla if case[1] == 1}
    sweep = get_mla_module_sweep_spec(backend)
    old_shapes = {
        (s, b, h)
        for s in sweep.context_sequence_lengths
        for b in sweep.context_batch_sizes
        for h in sweep.inner_sweep_head_counts
        if b * s <= sweep.context_max_tokens
        and (
            backend == "vllm"
            or s < sweep.context_large_sequence_min
            or b <= sweep.context_large_sequence_max_batch_size
        )
    }
    assert old_shapes <= {tuple(case[:3]) for case in mla}
    assert max(case[0] for case in getter("dsa")) <= max(sweep.context_sequence_lengths)
    assert all(case[0] * case[1] <= sweep.context_mla_max_tokens for case in mla)


def _cuda(gib):
    return SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda _device: SimpleNamespace(total_memory=gib * 2**30),
    )


def test_native_msa_pool_budget_keeps_large_case_on_blackwell(capsys):
    case = [131072, 256, 64, "bfloat16", "bfloat16", "bfloat16", "MiniMaxAI/MiniMax-M3"]

    def footprint(row):
        return module_footprint_bytes(row, backend="trtllm", attention_type="msa", is_context=False, page_size=128)

    assert 80 * 2**30 < footprint(case) < 180 * 2**30 * 0.9
    assert filter_attention_cases([case], footprint, _cuda(80), "msa_generation_module") == []
    assert filter_attention_cases([case], footprint, _cuda(180), "msa_generation_module") == [case]
    assert "dropped 1/1" in capsys.readouterr().out


def test_model_position_limit_accepts_boundary_and_raises_above():
    validate_module_position("nvidia/GLM-5.2-NVFP4", 1048576)
    with pytest.raises(ValueError, match="exceeds model limit"):
        validate_module_position("nvidia/GLM-5.2-NVFP4", 1048577)
    with pytest.raises(ValueError, match="exceeds model limit"):
        validate_module_position("deepseek-ai/DeepSeek-V3.2", 262144)


def test_position_validation_accepts_a_local_model_directory(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
    validate_module_position(str(tmp_path), 8192)
    with pytest.raises(ValueError, match="exceeds model limit"):
        validate_module_position(str(tmp_path), 8193)


def test_position_validation_uses_existing_resolver_for_unbundled_model(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
    requested = []

    def resolve(model):
        requested.append(model)
        return str(tmp_path)

    monkeypatch.setattr("collector.module_memory._resolve_local_model_path", resolve)
    validate_module_position("fixture/unbundled-model", 8192)
    assert requested == ["fixture/unbundled-model"]


@pytest.mark.parametrize("prefix", [0, 524288])
@pytest.mark.parametrize("batch,query,expected", [(32, 128, 131072), (16, 16384, 262144)])
def test_vllm_msa_scheduler_capacity_counts_query_tokens(tmp_path, monkeypatch, prefix, batch, query, expected):
    """Execute construction through the scheduler call, before GPU allocation."""
    config = module_model_config("MiniMaxAI/MiniMax-M3")
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(current_platform=SimpleNamespace(is_rocm=lambda: False))
    )
    captured = {}

    class ConfigCaptured(Exception):
        pass

    def create_config(**kwargs):
        captured.update(kwargs)
        raise ConfigCaptured

    namespace = {
        "os": os,
        "json": json,
        "_resolve_local_model_path": lambda _: str(tmp_path),
        "_ceil_div": lambda a, b: (a + b - 1) // b,
        "create_vllm_config": create_config,
    }
    path = COLLECTOR / "vllm/collect_msa_module.py"
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_create_msa_attention_module"
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(ConfigCaptured):
        namespace["_create_msa_attention_module"](
            str(tmp_path), 64, "bfloat16", "bfloat16", prefix + query, batch, True, query_len=query
        )
    assert captured["max_num_batched_tokens"] == expected
    assert captured["max_model_len"] == max(prefix + query, 4096)
    assert captured["num_gpu_blocks"] == 1 + ((prefix + query + 128) // 128) * batch


def test_trtllm_dsa_footprint_includes_native_logits_budget(monkeypatch):
    case = [4096, 1, 8, "fp8", "bfloat16", "bfloat16", "nvidia/GLM-5.2-NVFP4", "dsa", 262144]

    def footprint():
        return module_footprint_bytes(case, backend="trtllm", attention_type="dsa", is_context=True, page_size=32)

    monkeypatch.setenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", str(1 << 31))
    default = footprint()
    monkeypatch.setenv("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", str(1 << 26))
    small = footprint()
    assert default - small > 3 * 2**30


def test_trtllm_msa_reference_workspace_is_budgeted():
    case = [128, 1, 64, "bfloat16", "bfloat16", "bfloat16", "MiniMaxAI/MiniMax-M3", 524288]
    optimized = module_footprint_bytes(case, backend="trtllm", attention_type="msa", is_context=True, page_size=128)
    reference = module_footprint_bytes(case, backend="trtllm", attention_type="msa", is_context=True, page_size=32)
    assert reference > optimized + 2 * 2**30
