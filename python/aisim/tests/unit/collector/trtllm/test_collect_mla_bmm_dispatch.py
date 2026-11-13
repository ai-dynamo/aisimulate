# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free checks that the TRT-LLM mla_bmm collector makes serving's calls.

Serving's fp8_block_scaling_bmm_out and MLA._bmm_bf16_out (_torch/attention/mla.py)
pick the op per SM, and the SM89/90/120 fp8 branch writes into a fresh buffer that it
copies into the output view. The module imports tensorrt_llm at module scope, so the
helpers are AST-extracted and run against stubs.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector" / "trtllm" / "collect_mla_bmm.py"


def _load(names: set[str], namespace: dict) -> dict:
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "sm,path",
    [
        (89, "fp8_block_scaling_bmm_out"),
        (90, "fp8_block_scaling_bmm_out"),
        (120, "fp8_block_scaling_bmm_out"),
        (100, "torch_bmm_dequant_bf16"),
        (103, "torch_bmm_dequant_bf16"),
        (107, "torch_bmm_dequant_bf16"),
    ],
)
def test_fp8_path_follows_serving_wrapper(sm, path):
    ns = _load({"_fp8_bmm_path"}, {"get_sm_version": lambda: sm})
    assert ns["_fp8_bmm_path"]() == path


@pytest.mark.parametrize("sm", [80, 86, 121])
def test_fp8_path_raises_where_serving_raises(sm):
    ns = _load({"_fp8_bmm_path"}, {"get_sm_version": lambda: sm})
    with pytest.raises(RuntimeError, match="NotImplementedError"):
        ns["_fp8_bmm_path"]()


class _Tensor:
    def __init__(self, name):
        self.name = name
        self.shape = (2, 3, 4)
        self.dtype = "bf16"
        self.device = "cuda:0"
        self.copied_from = None

    def contiguous(self):
        return _Tensor(f"{self.name}.contiguous")

    def new_empty(self, shape, dtype, device):
        assert (shape, dtype, device) == (self.shape, self.dtype, self.device)
        return _Tensor(f"{self.name}.fresh")

    def copy_(self, other):
        self.copied_from = other.name


@pytest.mark.parametrize("sm,lhs", [(90, "a"), (100, "a"), (103, "a"), (120, "a.contiguous"), (121, "a.contiguous")])
def test_bf16_bmm_densifies_the_lhs_on_sm120(sm, lhs):
    calls = []
    torch = SimpleNamespace(ops=SimpleNamespace(trtllm=SimpleNamespace(bmm_out=lambda a, b, out: calls.append(a.name))))
    ns = _load({"_bmm_bf16_out"}, {"get_sm_version": lambda: sm, "torch": torch})
    ns["_bmm_bf16_out"](_Tensor("a"), _Tensor("b"), _Tensor("out"))
    assert calls == [lhs]


@pytest.mark.parametrize("sm", [90, 120])
def test_fp8_bmm_writes_a_fresh_buffer_then_copies(sm):
    calls = []

    def bmm(mat1_fp8, mat2_fp8, mat1_scale, mat2_scale, output):
        calls.append((mat1_fp8, mat1_scale, output.name))

    torch = SimpleNamespace(ops=SimpleNamespace(trtllm=SimpleNamespace(fp8_block_scaling_bmm_out=bmm)))
    ns = _load(
        {"_fp8_block_scaling_bmm_out"},
        {"torch": torch, "_quantize_fp8_activation": lambda x: (f"{x.name}.fp8", f"{x.name}.scale")},
    )
    out = _Tensor("out")
    ns["_fp8_block_scaling_bmm_out"](_Tensor("q_nope"), "w", "w_scale", out)
    assert calls == [("q_nope.fp8", "q_nope.scale", "out.fresh")]
    assert out.copied_from == "out.fresh"
