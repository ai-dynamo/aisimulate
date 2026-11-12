# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AST-only guards for the TRT-LLM MLA/DSA module collector.

The trtllm ``create_attention_layer`` must resolve the model id to a local,
auto_map-stripped config dir (``helper._resolve_local_model_path``) before any
transformers / TRT-LLM config load — otherwise the raw HF id routes through the
``trust_remote_code`` path whose ``_remote_code.lock`` serialized the 8 parallel
workers and produced mass file-lock Timeouts on the ``*_module`` ops. AST-only so
the check runs on CUDA-free CI (``create_attention_layer`` cannot be exec'd
without the framework).
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "collector" / "trtllm" / "collect_mla_module.py"


def _function(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_resolver_is_imported_from_helper():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "collector.helper"
        for alias in node.names
    }
    assert "_resolve_local_model_path" in imported, (
        "collect_mla_module must import _resolve_local_model_path from collector.helper"
    )


def test_create_attention_layer_resolves_model_path_before_config_load():
    fn = _function("create_attention_layer")

    # The reassignment `model_path = _resolve_local_model_path(model_path)`.
    resolve_line = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "model_path" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "_resolve_local_model_path"
        ):
            resolve_line = node.lineno
            break
    assert resolve_line is not None, (
        "create_attention_layer must reassign model_path via _resolve_local_model_path(model_path)"
    )

    # It must run before the first transformers config read (get_config_dict),
    # so the local auto_map-stripped path — not the raw HF id — is loaded.
    config_line = min(
        (
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get_config_dict"
        ),
        default=None,
    )
    assert config_line is not None, "expected a get_config_dict(...) call in create_attention_layer"
    assert resolve_line < config_line, (
        "model_path must be resolved to a local config BEFORE get_config_dict(), "
        "or the trust_remote_code lock path is still taken"
    )


def test_dsa_layers_are_tuned_like_dense_mla_layers():
    """The DSA layer's tunable ops are Linear projections too (the indexer adds wq_b), so
    the layer build and the per-point retune call _autotune_projections for DSA as well:
    no `sparse_attention_config` condition may gate a call."""
    for name in ("create_attention_layer", "run_mla_module"):
        fn = _function(name)
        calls = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_autotune_projections"
        ]
        assert calls, f"{name} must tune the projections"
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and "sparse_attention_config" in ast.unparse(node.test):
                gated = [
                    inner
                    for inner in ast.walk(node)
                    if isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == "_autotune_projections"
                ]
                assert not gated, f"{name} gates _autotune_projections on sparse_attention_config"


def test_dsa_sparse_config_is_built_by_the_framework():
    """ModelConfig.from_pretrained builds the DSA config for DeepseekV32ForCausalLM and
    GlmMoeDsaForCausalLM checkpoints, and loads glm_moe_dsa configs itself, as serving does
    when no sparse_attention_config is set: the collector hands it none and patches no
    config registry."""
    fn = _function("create_attention_layer")
    call = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "from_pretrained"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["sparse_attention_config"], ast.Constant)
    assert keywords["sparse_attention_config"].value is None
    source = SOURCE.read_text(encoding="utf-8")
    assert "DeepSeekSparseAttentionConfig(" not in source
    assert "_CONFIG_REGISTRY" not in source


def _call_lines(fn, name):
    return [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == name
    ]


def test_the_history_index_k_is_written_once_the_metadata_is_prepared():
    """The DSA indexer scores the cached history, which nothing else writes: the collector
    writes it after prepare() has resolved the block table, before the manager is returned
    for timing."""
    fn = _function("create_kv_cache_and_metadata")
    (prepare,) = _call_lines(fn, "prepare")
    (write,) = _call_lines(fn, "_write_indexer_history")
    assert prepare < write


def test_index_scores_get_their_magnitude_before_the_weights_are_post_processed():
    """Dummy weights leave the index scores near 1e-6 and the NVFP4 calibration scalars unset;
    the fix-up runs on the initialized layer, before post_load_weights derives from it."""
    fn = _function("create_attention_layer")
    dummy = min(_call_lines(fn, "initialize_dummy_weights"))  # the layer's; the next norm's follows
    (magnitude,) = _call_lines(fn, "_give_index_scores_magnitude")
    (post_load,) = _call_lines(fn, "post_load_weights")
    assert dummy < magnitude < post_load
