# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free contracts of the TRT-LLM dense and encoder attention collectors.

Both modules import tensorrt_llm at module scope, so the checks read their AST.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCES = {
    "dense": REPO_ROOT / "collector" / "trtllm" / "collect_attn.py",
    "encoder": REPO_ROOT / "collector" / "trtllm" / "collect_attn_encoder.py",
}


def _tree(name: str) -> ast.Module:
    path = SOURCES[name]
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_rows_never_depend_on_environment_variables(name):
    # Rule 5: rows depend only on the pin and the plan. The TRT-LLM build reads no
    # TRTLLM_ENABLE_XQA_JIT, and an A/B override of the KV page would make rows
    # depend on the environment.
    tree = _tree(name)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "os" not in imported
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"environ", "getenv", "putenv"}


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_imports_target_one_framework_layout(name):
    # One release per collector file: no fallback imports for another TRT-LLM layout.
    tree = _tree(name)
    assert not [node for node in tree.body if isinstance(node, ast.Try)]
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(module and module.startswith("tensorrt_llm._torch.attention_backend") for module in modules)


def test_dense_metadata_keeps_paged_context_fmha_off():
    # Serving derives use_paged_context_fmha from chunked prefill, block reuse and draft
    # tokens (TrtllmAttentionMetadata.__post_init__); the generator renders all three off,
    # and paged context would switch an fp8-KV context to the fp8 FMHA.
    calls = [
        node
        for node in ast.walk(_tree("dense"))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "AttentionRuntimeFeatures"
    ]
    assert len(calls) == 2
    for call in calls:
        flags = {keyword.arg: keyword.value for keyword in call.keywords}
        for flag in ("chunked_prefill", "cache_reuse"):
            assert isinstance(flags[flag], ast.Constant) and flags[flag].value is False
        if "has_speculative_draft_tokens" in flags:
            assert flags["has_speculative_draft_tokens"].value is False


def test_dense_kv_page_is_the_framework_default():
    function = next(
        node
        for node in _tree("dense").body
        if isinstance(node, ast.FunctionDef) and node.name == "_deployment_tokens_per_block"
    )
    source = ast.unparse(function)
    assert "KvCacheConfig().tokens_per_block" in source
    assert not [node for node in ast.walk(function) if isinstance(node, ast.Try)]
