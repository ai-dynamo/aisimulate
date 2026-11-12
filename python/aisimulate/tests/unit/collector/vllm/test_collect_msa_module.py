# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vLLM MSA collector lays each cache out in the layout serving resolves and
times the indexer on written history and index projections of its own."""

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


# numpy stand-ins for the torch dtypes the collector tells apart.
_FP8, _BF16 = np.float16, np.float64


class _Array(np.ndarray):
    """The slice of a torch tensor the input-preparation helpers touch, over numpy."""

    device = "cuda:0"

    def copy_(self, other):
        self[...] = other
        return self

    def to(self, *args, dtype=None, device=None):
        dtype = next((arg for arg in args if not isinstance(arg, str)), dtype)
        return self if dtype is None else self.astype(dtype)


def _array(shape, dtype, value=0.0):
    return np.full(shape, value, dtype=dtype).view(_Array)


class _Generator:
    def __init__(self, device):
        assert device == "cpu"

    def manual_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        return self


def _helpers(*names, **constants):
    """Load the named helpers over numpy: the module itself imports vllm and torch."""
    from contextlib import nullcontext

    tree = ast.parse(SRC.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    fake_torch = SimpleNamespace(
        Tensor=_Array,
        Generator=_Generator,
        no_grad=nullcontext,
        device=lambda value: value,
        uint8=np.uint8,
        float32=np.float32,
        randn=lambda shape, generator, dtype, device: (
            generator.rng.standard_normal(tuple(shape)).astype(dtype).view(_Array)
        ),
        randint=lambda low, high, shape, generator, dtype, device: (
            generator.rng.integers(low, high, tuple(shape)).astype(dtype).view(_Array)
        ),
        ones=lambda shape, dtype, device: _array(tuple(shape), dtype, 1.0),
        empty=lambda shape, dtype, device: _array(tuple(shape), dtype),
    )
    namespace = {"torch": fake_torch, **constants}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("num_blocks", [3, 5, 12])
def test_index_key_history_fills_the_cache_with_one_repeated_block_of_keys(num_blocks):
    # 5 distinct blocks of 4 tokens, drawn 2 blocks at a time.
    helpers = _helpers("_write_index_key_history", _HISTORY_TOKENS=20, _SAMPLE_CHUNK_ELEMENTS=64, _HISTORY_KEYS={})
    cache = _array((num_blocks, 1, 4, 8), np.float32)

    helpers["_write_index_key_history"](cache)

    assert np.all(cache != 0) and 0.5 < cache.std() < 1.5
    distinct = min(num_blocks, 5)
    assert len({block.tobytes() for block in cache[:distinct]}) == distinct
    assert len({row.tobytes() for row in cache[:distinct].reshape(-1, 8)}) == distinct * 4
    for block in range(distinct, num_blocks):
        assert np.array_equal(cache[block], cache[block % 5])


def test_index_key_history_is_seeded():
    first, second, third = (_array((6, 1, 4, 8), np.float32) for _ in range(3))
    helpers = _helpers(
        "_write_index_key_history", _HISTORY_TOKENS=1 << 18, _SAMPLE_CHUNK_ELEMENTS=1 << 20, _HISTORY_KEYS={}
    )
    helpers["_write_index_key_history"](first)
    helpers["_write_index_key_history"](second)  # the worker's kept block
    again = _helpers(
        "_write_index_key_history", _HISTORY_TOKENS=1 << 18, _SAMPLE_CHUNK_ELEMENTS=1 << 20, _HISTORY_KEYS={}
    )
    again["_write_index_key_history"](third)  # another worker samples the same keys

    assert np.array_equal(first, second) and np.array_equal(first, third)
    assert len(helpers["_HISTORY_KEYS"]) == 1


def _projection(weight_dtype, *, packed=False):
    """A fused [q | k | v | index_q | index_k] projection of 2 + 1 + 1 + 2 + 1 heads of 2 rows."""
    width = 4 if packed else 8
    projection = SimpleNamespace(weight=_array((14, width), weight_dtype))
    if packed:
        projection.weight_scale = _array((14, 1), _FP8)
    return SimpleNamespace(qkv_proj=projection, index_q_size=4, idx_head_dim=2)


@pytest.mark.parametrize("weight_dtype", [_BF16, _FP8])
def test_index_projection_rows_are_seeded_and_differ_between_heads(weight_dtype):
    helpers = _helpers("_give_index_projection_values")
    module, again = _projection(weight_dtype), _projection(weight_dtype)
    module.qkv_proj.weight[:8] = 0.01

    helpers["_give_index_projection_values"](module)
    helpers["_give_index_projection_values"](again)

    weight = module.qkv_proj.weight
    assert weight.dtype == weight_dtype
    # q, k and v keep the dummy fill; the index_q and index_k rows are the last six.
    assert np.all(weight[:8] == weight_dtype(0.01))
    assert np.all(weight[8:] != 0) and np.abs(weight[8:]).max() < 0.2
    assert len({row.tobytes() for row in weight[8:]}) == 6
    assert np.array_equal(weight[8:], again.qkv_proj.weight[8:])


def test_packed_nvfp4_index_rows_take_seeded_bytes_and_a_unit_block_scale():
    helpers = _helpers("_give_index_projection_values")
    module = _projection(np.uint8, packed=True)

    helpers["_give_index_projection_values"](module)

    weight, scale = module.qkv_proj.weight, module.qkv_proj.weight_scale
    assert weight.dtype == np.uint8 and scale.dtype == _FP8
    assert not weight[:8].any() and not scale[:8].any()
    assert len({row.tobytes() for row in weight[8:]}) == 6
    assert np.all(scale[8:] == 1.0)


@pytest.mark.parametrize("meta", [False, True])
def test_move_keeps_the_rotary_table_when_to_empty_reallocates_buffers(meta):
    helpers = _helpers("_move_to_device")
    table = np.arange(6, dtype=np.float32).reshape(3, 2)

    class Module:
        def __init__(self):
            self.rotary = SimpleNamespace(cos_sin_cache=table.copy().view(_Array))
            self.moves = []

        def parameters(self):
            return [SimpleNamespace(is_meta=meta)]

        def modules(self):
            return [self, self.rotary]

        def to(self, device):
            self.moves.append(("to", device))
            return self

        def to_empty(self, *, device):
            self.moves.append(("to_empty", device))
            self.rotary.cos_sin_cache = _array((3, 2), np.float32, np.nan)
            return self

    module = Module()

    assert helpers["_move_to_device"](module, "cuda:0") is module
    assert module.moves == [("to_empty" if meta else "to", "cuda:0")]
    assert np.array_equal(module.rotary.cos_sin_cache, table)


def _function(name):
    tree = ast.parse(SRC.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _call_lines(function, name):
    return [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == name
    ]


def test_dummy_fill_skips_the_rotary_table_and_seeds_the_index_projection_before_processing():
    build = _function("_create_msa_attention_module")
    fill = next(
        node for node in ast.walk(build) if isinstance(node, ast.For) and "named_buffers" in ast.unparse(node.iter)
    )
    # The table is passed over before any branch can overwrite it.
    skip = fill.body[1]
    assert ast.unparse(skip.test) == "name.endswith('cos_sin_cache')"
    assert isinstance(skip.body[-1], ast.Continue)

    (moved,) = _call_lines(build, "_move_to_device")
    (seeded,) = _call_lines(build, "_give_index_projection_values")
    (processed,) = _call_lines(build, "process_weights_after_loading")
    assert moved < fill.lineno < seeded < processed


def test_history_is_written_after_the_dtype_check_and_before_the_first_forward():
    run = _function("run_msa_module")
    (written,) = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_write_index_key_history"
    ]
    assert ast.unparse(written.args[0]) == "index_kv_cache"
    raises = [node for node in ast.walk(run) if isinstance(node, ast.Raise)]
    dtype_check = next(node.lineno for node in raises if "precision mismatch" in ast.unparse(node))
    assert dtype_check < written.lineno < min(_call_lines(run, "forward"))
