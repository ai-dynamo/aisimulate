# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TRT-LLM dense-attention collector: rows and the serving contracts of its run function.

AST checks only: run_attention_torch cannot run on CUDA-free CI hosts.
"""

import ast
from pathlib import Path

import pytest
from collector.op_plans import plan

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_function():
    source_path = REPO_ROOT / "collector" / "trtllm" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch")


@pytest.mark.parametrize("op", ["attention_context", "attention_generation"])
def test_every_row_runs_on_the_default_trtllm_backend(op, monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    rows = plan(op, "trtllm", 90).rows
    assert {row[9] for row in rows} == {"TRTLLM"}
    # Gemma 4's head_dim 512 global layers run there too, as serving runs them.
    assert any(row[4] == 512 for row in rows)


def test_the_run_function_builds_only_the_trtllm_backend():
    source = ast.unparse(_run_function())
    assert "FLASHINFER" not in source and "flashinfer" not in source


def test_dense_attention_uses_full_kv_cache():
    assignments = [
        node
        for node in ast.walk(_run_function())
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "kv_cache_type" for target in node.targets)
    ]

    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Attribute)
    assert value.attr == "SELF"


def test_out_scale_mirrors_serving_use_quantize_output():
    # out_scale mirrors serving's Attention._use_quantize_output(): only a model
    # with fp8 weights supplies o_proj.inv_input_scale. The fp8-model case is the
    # context use_fp8_context_fmha case; the fp8_kv_cache cases of both phases are
    # an unquantized model with an fp8 KV cache (out_scale None, bf16 output, the
    # deployment the serving gates probe). No SM condition.
    guards = [
        node
        for node in ast.walk(_run_function())
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "out_scale" for t in n.targets)
            for n in ast.walk(node)
        )
    ]
    assert len(guards) == 1, "expected exactly one conditional out_scale assignment"
    test_names = {n.id for n in ast.walk(guards[0].test) if isinstance(n, ast.Name)}
    assert test_names == {"use_fp8_context_fmha"}
    assert not any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "get_sm_version" for n in ast.walk(guards[0])
    ), "out_scale must not be SM-conditioned"
