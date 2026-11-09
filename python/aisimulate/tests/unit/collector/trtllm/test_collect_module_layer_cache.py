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


def test_mla_projections_are_tuned_at_each_count_under_autotune():
    """Serving's warmup pass forwards every projection under autotune() at its token counts."""
    import contextlib
    import weakref

    tree = ast.parse((COLLECTOR / "collect_mla_module.py").read_text())
    selected = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_autotune_projections")
        or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "_TUNED_NUM_TOKENS")
    ]
    assert len(selected) == 2
    events = []

    class Linear:
        def __init__(self, name, in_features):
            self.name, self.in_features, self.dtype = name, in_features, "bf16"

        def forward(self, x):
            events.append((self.name, x, tuning[-1]))

    class Module:
        def __init__(self, children):
            self.children = children

        def modules(self):
            return [self, *self.children]

    tuning = [False]

    @contextlib.contextmanager
    def autotune():
        tuning.append(True)
        yield
        tuning.append(False)

    torch = SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        randn=lambda shape, dtype, device: (shape, dtype, device),
        bfloat16="bf16",
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    namespace = {
        "weakref": weakref,
        "time": __import__("time"),
        "torch": torch,
        "Linear": Linear,
        "autotune": autotune,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "collect_mla_module.py", "exec"), namespace)
    module = Module([Linear("q_b_proj", 1536), object(), Linear("o_proj", 16384)])
    namespace["_autotune_projections"](module, (8192, 1), "cuda:0")
    assert events == [
        ("q_b_proj", ((8192, 1536), "bf16", "cuda:0"), True),
        ("o_proj", ((8192, 16384), "bf16", "cuda:0"), True),
        ("q_b_proj", ((1, 1536), "bf16", "cuda:0"), True),
        ("o_proj", ((1, 16384), "bf16", "cuda:0"), True),
    ]
    assert namespace["_TUNED_NUM_TOKENS"][module] == 8192
    namespace["_autotune_projections"](module, (32768,), "cuda:0")
    assert namespace["_TUNED_NUM_TOKENS"][module] == 32768
