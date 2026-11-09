# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vLLM MSA collector lays each cache out in the layout serving resolves."""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_msa_module.py"


class _Tensor:
    """The slice of a torch tensor _layer_kv_cache touches, over numpy."""

    def __init__(self, array):
        self.array = array

    def permute(self, *dims):
        return _Tensor(self.array.transpose(dims))

    def view(self, _dtype):
        return self


def _layer_kv_cache(monkeypatch):
    """Load only _layer_kv_cache: the module itself imports vllm and torch."""
    tree = ast.parse(SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_layer_kv_cache")
    fake_torch = SimpleNamespace(uint8=np.uint8, zeros=lambda shape, dtype, device: _Tensor(np.zeros(shape, dtype)))
    interface = types.ModuleType("vllm.v1.kv_cache_interface")
    interface.compute_layer_kv_cache_shape_bytes = lambda spec, num_blocks: (num_blocks, spec.heads, spec.tokens, 512)
    for name in ("vllm", "vllm.v1"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", interface)
    namespace = {"torch": fake_torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), namespace)
    return namespace["_layer_kv_cache"]


@pytest.mark.parametrize(
    "layer_view_order,strides",
    [
        # LBHNC (SM100 family): each head's 128 tokens are one contiguous run.
        ((0, 1, 2, 3), (4 * 128 * 512, 128 * 512, 512, 1)),
        # LBNHC (elsewhere): a token's heads sit side by side inside the block.
        ((0, 2, 1, 3), (128 * 4 * 512, 512, 4 * 512, 1)),
    ],
)
def test_cache_keeps_the_logical_view_over_the_resolved_physical_order(monkeypatch, layer_view_order, strides):
    layer_kv_cache = _layer_kv_cache(monkeypatch)
    spec = SimpleNamespace(heads=4, tokens=128, dtype="bfloat16")

    cache = layer_kv_cache(spec, 3, SimpleNamespace(layer_view_order=layer_view_order), "cuda:0").array

    assert cache.shape == (3, 4, 128, 512)
    assert cache.strides == strides
