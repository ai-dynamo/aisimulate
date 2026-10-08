# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.30 ropes q_pe alone (key=None) in the DSV3.2 dense-MHA FP8 branch; both
rope impls reject None. The collector wraps the module's rotary forward so that call
ropes the query against a throwaway key and returns (query, None)."""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_mla_module.py"


def _load():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rope_tolerating_missing_key")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["_rope_tolerating_missing_key"]


class _Q:
    def __init__(self, tag):
        self.tag = tag

    def clone(self):
        return _Q(self.tag + "'")


def test_missing_key_is_replaced_by_a_throwaway_copy_and_dropped():
    calls = []

    def forward(positions, query, key=None, offsets=None):
        assert key is not None  # the real impls' contract
        calls.append((positions, query.tag, key.tag, offsets))
        return (_Q(query.tag + "+rope"), _Q(key.tag + "+rope"))

    wrapped = _load()(forward)
    q, k = wrapped("pos", _Q("q"))
    assert (q.tag, k) == ("q+rope", None)
    assert calls == [("pos", "q", "q'", None)]


def test_present_key_passes_through_unchanged():
    def forward(positions, query, key=None, offsets=None):
        return (query, key)

    wrapped = _load()(forward)
    assert wrapped("pos", "q", "k", "off") == ("q", "k")


def test_shim_forwards_only_the_arguments_it_received():
    # vLLM 0.30 forward_cuda(positions, query, key=None): no offsets parameter. The first shim
    # passed offsets=None positionally -> TypeError "takes from 3 to 4 positional arguments but
    # 5 were given" (508 GLM-5 cases, pipelines 72387095/72387091).
    def forward_cuda(positions, query, key=None):
        return (query, key)

    wrapped = _load()(forward_cuda)
    assert wrapped("pos", "q", "k") == ("q", "k")
    q, k = wrapped("pos", _Q("q"))
    assert (q.tag, k) == ("q", None)


def test_shim_is_idempotent():
    # Installed per case on a possibly reused module: re-wrapping stacked closures until
    # RecursionError (2,286 cases per run at e598d380).
    def forward(positions, query, key=None):
        return (query, key)

    shim = _load()
    once = shim(forward)
    assert shim(once) is once and shim(shim(once)) is once
    assert getattr(once, "_aisim_rope_shim", False) is True


def test_fp8_block_guard_present_before_layer_construction():
    text = (Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_gemm.py").read_text()
    guard = text.index("FIXME(kernel-limit): vLLM fp8_block GEMM needs n and k multiples of 16")
    assert guard < text.index("def create_gemm():")
