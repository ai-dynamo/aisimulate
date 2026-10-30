# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the per-worker attention-layer caches of the TRT-LLM MLA/DSA and
MSA module collectors (same pattern as collect_dsv4_attn._cached_dsv4_attention_module).
The modules import tensorrt_llm at module scope, so the helpers are AST-extracted."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[4] / "collector" / "trtllm"


def _load(filename: str, helper: str, builder: str):
    tree = ast.parse((COLLECTOR / filename).read_text())
    selected = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == helper)
        or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "_LAYER_CACHE")
    ]
    assert len(selected) == 2
    builds, evictions = [], []
    namespace = {
        builder: lambda **kwargs: (builds.append(kwargs) or ("module", "config", "arch", len(builds))),
        "gc": SimpleNamespace(collect=lambda: evictions.append("gc")),
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: evictions.append("empty_cache"))),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace[helper], builds, evictions


def test_mla_layer_is_built_once_per_key_and_evicted_on_a_new_key():
    cached, builds, evictions = _load("collect_mla_module.py", "_cached_attention_layer", "create_attention_layer")
    first = cached("deepseek-ai/DeepSeek-V3.2", 128, True, "fp8_block", "cuda:0")
    assert cached("deepseek-ai/DeepSeek-V3.2", 128, True, "fp8_block", "cuda:0") is first
    assert len(builds) == 1 and evictions == []
    assert builds[0] == {
        "model_path": "deepseek-ai/DeepSeek-V3.2",
        "num_heads": 128,
        "use_fp8_kv_cache": True,
        "gemm_type": "fp8_block",
        "device": "cuda:0",
    }
    second = cached("deepseek-ai/DeepSeek-V3.2", 16, True, "fp8_block", "cuda:0")
    assert second is not first and len(builds) == 2
    assert evictions == ["gc", "empty_cache"], "the previous layer is released before the next build"
    third = cached("deepseek-ai/DeepSeek-V3.2", 128, True, "fp8_block", "cuda:0")
    assert third is not first, "one entry at a time"


def test_msa_layer_cache_keys_on_model_heads_gemm_and_device():
    cached, builds, evictions = _load(
        "collect_msa_module.py", "_cached_msa_attention_layer", "create_msa_attention_layer"
    )
    a = cached("MiniMaxAI/MiniMax-M3", 64, "bfloat16", "cuda:0")
    assert cached("MiniMaxAI/MiniMax-M3", 64, "bfloat16", "cuda:0") is a and len(builds) == 1
    expected = {"model_path": "MiniMaxAI/MiniMax-M3", "num_heads": 64, "gemm_type": "bfloat16", "device": "cuda:0"}
    assert builds[0] == expected
    cached("MiniMaxAI/MiniMax-M3", 64, "fp8_block", "cuda:0")
    assert len(builds) == 2 and evictions == ["gc", "empty_cache"]


@pytest.mark.parametrize(
    "filename,runner,helper",
    [
        ("collect_mla_module.py", "run_mla_module", "_cached_attention_layer"),
        ("collect_msa_module.py", "run_msa_module", "_cached_msa_attention_layer"),
    ],
)
def test_runner_builds_its_layer_through_the_cache(filename, runner, helper):
    tree = ast.parse((COLLECTOR / filename).read_text())
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == runner)
    calls = [
        n.func.id
        for n in ast.walk(body)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.startswith(("create_", "_cached_"))
    ]
    assert helper in calls
    assert not any(c in ("create_attention_layer", "create_msa_attention_layer") for c in calls)
