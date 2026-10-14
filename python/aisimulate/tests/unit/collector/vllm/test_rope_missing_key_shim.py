# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.30/0.31 rope q_pe alone (key=None) in the DSV3.2 dense-MHA FP8 branch
(models/deepseek_v32/attention.py:534 @0.30.0). Only DeepseekScalingRotaryEmbedding
rejects that; the collector wraps THAT class's forward so the call ropes the query
against a throwaway key that shares the query's strides and returns (query, None).
Rotary classes that accept key=None (GLM-5.x's RotaryEmbedding) stay on serving's call:
wrapping them handed vLLM's own rope kernel — one head stride for query and key — a
contiguous key with the wrong head stride, i.e. out-of-bounds in-place writes (the
GLM-5.2 fp8-KV dsa_context illegal-address / stall family, 2026-10-14)."""

import ast
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_mla_module.py"
_FUNCTIONS = (
    "_strided_throwaway_copy",
    "_rope_tolerating_missing_key",
    "_rope_module_rejects_missing_key",
    "_install_rope_missing_key_shims",
)


def _load_all():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _FUNCTIONS]
    assert {f.name for f in fns} == set(_FUNCTIONS)
    ns: dict = {}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def _load():
    return _load_all()["_rope_tolerating_missing_key"]


def _contiguous_strides(shape):
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return tuple(reversed(strides))


class _Q:
    """Just enough of torch.Tensor for the shim: shape/stride and the four methods it calls."""

    def __init__(self, tag, shape=(4, 8, 64), stride=None, storage_numel=None):
        self.tag = tag
        self.shape = tuple(shape)
        self._stride = tuple(stride) if stride is not None else _contiguous_strides(self.shape)
        self.storage_numel = storage_numel if storage_numel is not None else math.prod(self.shape)
        self.made_by = "fresh"

    def stride(self):
        return self._stride

    def clone(self):
        q = _Q(self.tag + "'", self.shape)
        q.made_by = "clone"
        return q

    def new_empty(self, numel):
        return _Q("empty", (numel,), storage_numel=numel)

    def as_strided(self, shape, stride):
        q = _Q("view", shape, stride, storage_numel=self.storage_numel)
        q.made_by = "as_strided"
        return q

    def copy_(self, src):
        self.tag = src.tag + "'"
        return self


def _q_pe_slice(tokens=3, heads=8, qk_head_dim=256, rope_dim=64):
    # q_pe = q.split([nope, rope], dim=-1)[1]: a last-dim slice of [tokens, heads, qk_head_dim].
    return _Q("q", (tokens, heads, rope_dim), (heads * qk_head_dim, qk_head_dim, 1))


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


def test_throwaway_key_shares_the_query_strides_not_a_contiguous_clone():
    # vLLM's rope kernel addresses the key with the QUERY's head stride
    # (csrc/libtorch_stable/pos_encoding_kernels.cu:57,68,164-165 @0.30.0). A contiguous clone of
    # the q_pe slice (head stride 64 vs 256) is written past its end by 192*(heads-1) elements.
    seen = []

    def forward(positions, query, key=None):
        seen.append(key)
        return (query, key)

    query = _q_pe_slice(tokens=3, heads=8)
    _load()(forward)("pos", query)
    (key,) = seen
    assert key.made_by == "as_strided"
    assert key.shape == query.shape
    assert key.stride() == query.stride() == (8 * 256, 256, 1)
    # Storage covers the last addressable element of the strided layout, nothing less.
    assert key.storage_numel == 1 + 2 * (8 * 256) + 7 * 256 + 63
    assert key.tag == "q'"  # the data was copied from the query


def test_throwaway_copy_of_an_empty_query_falls_back_to_clone():
    copy = _load_all()["_strided_throwaway_copy"]
    key = copy(_Q("q", (0, 8, 64), (8 * 256, 256, 1)))
    assert key.made_by == "clone"


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


# --- which rotary classes get the shim -------------------------------------------------


class RotaryEmbedding:
    """vLLM's base class: forward_cuda -> ops.rotary_embedding, key may be None."""

    def forward_cuda(self, positions, query, key=None):
        return (query, key)

    def forward(self, positions, query, key=None):
        return self.forward_cuda(positions, query, key)


class Llama3RotaryEmbedding(RotaryEmbedding):
    pass


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """DeepSeek-V3.2's yarn rope: FlashInfer op key.view / native assert key is not None."""

    def forward_cuda(self, positions, query, key=None):
        assert key is not None, "the real impls reject key=None"
        return (query, key)


class DeepseekV4ScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """Overrides forward_cuda back onto ops.rotary_embedding: accepts key=None again."""

    def forward_cuda(self, positions, query, key=None):
        return (query, key)


class Linear:
    def forward(self, x):
        return x


class _Attn:
    def __init__(self, **modules):
        self._modules = modules

    def named_modules(self):
        return list(self._modules.items())


def test_rejects_missing_key_only_for_deepseek_scaling_rope_itself():
    rejects = _load_all()["_rope_module_rejects_missing_key"]
    assert rejects(DeepseekScalingRotaryEmbedding()) is True
    assert rejects(RotaryEmbedding()) is False  # GLM-5.x: rope_type "default"
    assert rejects(Llama3RotaryEmbedding()) is False
    assert rejects(DeepseekV4ScalingRotaryEmbedding()) is False
    assert rejects(Linear()) is False


def test_installer_shims_deepseek_scaling_rope_and_leaves_glm_rope_on_servings_call():
    ns = _load_all()
    glm_rope = RotaryEmbedding()
    dsv32_rope = DeepseekScalingRotaryEmbedding()
    attn = _Attn(
        **{
            "rotary_emb": glm_rope,
            "indexer_rope_emb": Llama3RotaryEmbedding(),
            "mla_attn.rotary_emb": dsv32_rope,
            "q_b_proj": Linear(),
        }
    )
    patched, left = ns["_install_rope_missing_key_shims"](attn)
    assert patched == ["DeepseekScalingRotaryEmbedding"]
    assert left == ["RotaryEmbedding", "Llama3RotaryEmbedding"]

    # GLM's rope is untouched: serving's `rotary_emb(positions, q_pe)` reaches the class's own
    # forward with key=None, exactly as in vLLM.
    assert "forward" not in vars(glm_rope)
    assert glm_rope.forward("pos", "q") == ("q", None)

    # DeepSeek-V3.2's rope now accepts the same call and returns (query, None).
    assert getattr(dsv32_rope.forward, "_aisim_rope_shim", False) is True
    q, k = dsv32_rope.forward("pos", _q_pe_slice())
    assert (q.tag, k) == ("q", None)


def test_installer_is_idempotent_across_cases():
    ns = _load_all()
    dsv32_rope = DeepseekScalingRotaryEmbedding()
    attn = _Attn(rotary_emb=dsv32_rope)
    ns["_install_rope_missing_key_shims"](attn)
    once = dsv32_rope.forward
    ns["_install_rope_missing_key_shims"](attn)
    assert dsv32_rope.forward is once


def test_run_mla_module_installs_shims_through_the_class_filter_only():
    text = SOURCE.read_text(encoding="utf-8")
    assert "_install_rope_missing_key_shims(attn_module)" in text
    # The old unconditional wrap of every *RotaryEmbedding submodule must not come back.
    assert "_m.forward = _rope_tolerating_missing_key(_m.forward)" not in text


def test_fp8_block_guard_present_before_layer_construction():
    text = (Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_gemm.py").read_text()
    guard = text.index("FIXME(kernel-limit): vLLM fp8_block GEMM needs n and k multiples of 16")
    assert guard < text.index("def create_gemm():")
