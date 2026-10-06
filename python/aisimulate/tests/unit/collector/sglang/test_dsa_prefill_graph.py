# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang prefill uses the native capture buckets, including padded shapes."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/sglang/dsa_prefill_graph.py"


def select_bucket(backend, sizes, tokens):
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "graph_token_bucket"
    )
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    runner = SimpleNamespace(
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend=backend, bs=sizes))
        )
    )
    return ns["graph_token_bucket"](runner, tokens)


@pytest.mark.parametrize(
    ("tokens", "expected"), [(1, 1), (128, 128), (129, 256), (1057, 2048), (2048, 2048), (2049, None)]
)
def test_prefill_capture_bucket_boundaries(tokens, expected):
    assert select_bucket("tc_piecewise", [2048, 128, 1, 256], tokens) == expected


def test_disabled_and_unsupported_backends_are_distinct():
    assert select_bucket("disabled", [1, 128], 1) is None
    assert select_bucket("tc_piecewise", [], 1) is None
    with pytest.raises(ValueError, match="Unsupported"):
        select_bucket("breakable", [1, 128], 1)


def test_cleanup_removes_only_new_owned_native_compile_hooks():
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_release_owned_compile_hooks"
    )
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)

    def hook_for(module):
        def hook():
            return module

        return hook

    owned, foreign = object(), object()
    hooks = {1: hook_for(owned), 2: hook_for(owned), 3: hook_for(foreign), 4: object(), 5: lambda: None}
    # A constructor may fail after registering its hook but before returning a
    # runner. Cleanup requires only the owned module and original registry.
    ns["_release_owned_compile_hooks"](hooks, {1}, owned)
    assert set(hooks) == {1, 3, 4, 5}
