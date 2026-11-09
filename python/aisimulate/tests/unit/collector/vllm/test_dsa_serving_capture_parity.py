# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DSA module's capture-dependent choices follow serving's (vLLM 0.32.0), without vLLM or CUDA.

Serving runs DeepseekV32Attention on breakable CUDA graphs: a prefill step up to the
capture ceiling is graph-replayed around an eager _sparse_indexer_and_attn, a larger one
runs eagerly. The collector times every step as one graph, so _dsa_serving_capture_parity
pins _should_prepare_mqa_query to serving's capture state for the step and lets the
indexer take serving's dense-MHA early return (sparse_attn_indexer.py:542-558).
"""

import ast
import enum
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/vllm/collect_mla_module.py"


def _load_function(name):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"contextmanager": contextmanager}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class _ForwardContext:
    def __init__(self):
        self.cudagraph_runtime_mode = CUDAGraphMode.NONE
        self.attn_metadata = None


@pytest.fixture
def forward_context(monkeypatch):
    context = _ForwardContext()
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.CUDAGraphMode = CUDAGraphMode
    forward = types.ModuleType("vllm.forward_context")
    forward.get_forward_context = lambda: context
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward)
    return context


class FakeDeepseekV32Attention:
    """The v0.32.0 surface the parity reads: the two hooks, _use_sparse_mha, skip_topk."""

    def __init__(self, layer_name="model.layers.0.self_attn.attn"):
        self.skip_topk = False
        self._dense_mha_metadata_layer_name = layer_name
        self.indexer_runs = []

    def _use_sparse_mha(self, attn_metadata):
        return attn_metadata.sparse_mha

    def _should_prepare_mqa_query(self, attn_metadata, cudagraph_runtime_mode):
        return "class"

    def _sparse_indexer_and_attn(self, *args, **kwargs):
        # The real method calls the indexer only when skip_topk is False (attention.py:481).
        self.indexer_runs.append(not self.skip_topk)
        return args, kwargs


def _mla_metadata(*, dense, decode_tokens=0):
    prefill = types.SimpleNamespace(use_dense_mha=dense)
    return types.SimpleNamespace(prefill=prefill, num_decode_tokens=decode_tokens)


@pytest.mark.parametrize(
    ("sparse_mha", "mode", "captured", "expected"),
    [
        (True, CUDAGraphMode.NONE, False, False),  # eager dense/masked-MHA prefill skips BMM + quant
        (True, CUDAGraphMode.NONE, True, True),  # serving captures this step: BMM + quant run
        (True, CUDAGraphMode.FULL, False, True),
        (False, CUDAGraphMode.NONE, False, True),  # sparse MQA always prepares the MQA query
    ],
)
def test_mqa_query_preparation_follows_serving_capture(forward_context, sparse_mha, mode, captured, expected):
    module = FakeDeepseekV32Attention()
    parity = _load_function("_dsa_serving_capture_parity")
    with parity(module, serving_captures_forward=captured):
        metadata = types.SimpleNamespace(sparse_mha=sparse_mha)
        assert module._should_prepare_mqa_query(metadata, mode) is expected
        assert module._should_prepare_mqa_query(None, mode) is True
    # The class's own method is back once the point ends.
    assert module._should_prepare_mqa_query(None, mode) == "class"
    assert "_sparse_indexer_and_attn" not in vars(module)


def test_dense_mha_prefill_skips_indexer_scoring_as_serving_does(forward_context):
    module = FakeDeepseekV32Attention()
    forward_context.attn_metadata = {"model.layers.0.self_attn.attn": _mla_metadata(dense=True)}
    with _load_function("_dsa_serving_capture_parity")(module, serving_captures_forward=True):
        assert module._sparse_indexer_and_attn(1, out=2) == ((1,), {"out": 2})
    assert module.indexer_runs == [False]
    assert module.skip_topk is False


@pytest.mark.parametrize(
    ("metadata", "mode", "layer_name"),
    [
        (_mla_metadata(dense=False), CUDAGraphMode.NONE, "model.layers.0.self_attn.attn"),  # sparse prefill
        (_mla_metadata(dense=True, decode_tokens=4), CUDAGraphMode.NONE, "model.layers.0.self_attn.attn"),
        (types.SimpleNamespace(prefill=None, num_decode_tokens=2), CUDAGraphMode.NONE, "model.layers.0.self_attn.attn"),
        (_mla_metadata(dense=True), CUDAGraphMode.FULL, "model.layers.0.self_attn.attn"),
        (_mla_metadata(dense=True), CUDAGraphMode.NONE, ""),  # layer opted out (MTP, reuse, PCP)
    ],
)
def test_indexer_runs_wherever_serving_runs_it(forward_context, metadata, mode, layer_name):
    module = FakeDeepseekV32Attention(layer_name)
    forward_context.cudagraph_runtime_mode = mode
    forward_context.attn_metadata = {"model.layers.0.self_attn.attn": metadata}
    with _load_function("_dsa_serving_capture_parity")(module, serving_captures_forward=False):
        module._sparse_indexer_and_attn()
    assert module.indexer_runs == [True]


def test_skip_indexer_measurement_keeps_its_switch(forward_context):
    module = FakeDeepseekV32Attention()
    module.skip_topk = True
    forward_context.attn_metadata = {"model.layers.0.self_attn.attn": _mla_metadata(dense=True)}
    with _load_function("_dsa_serving_capture_parity")(module, serving_captures_forward=False):
        module._sparse_indexer_and_attn()
    assert module.indexer_runs == [False]
    assert module.skip_topk is True


def test_hooks_are_removed_when_the_point_fails(forward_context):
    module = FakeDeepseekV32Attention()
    parity = _load_function("_dsa_serving_capture_parity")
    with pytest.raises(RuntimeError, match="point failed"), parity(module, serving_captures_forward=False):
        raise RuntimeError("point failed")
    assert "_should_prepare_mqa_query" not in vars(module)
    assert "_sparse_indexer_and_attn" not in vars(module)


def test_a_layer_without_the_pinned_hooks_is_refused(forward_context):
    class Legacy:
        skip_topk = False

    parity = _load_function("_dsa_serving_capture_parity")
    with pytest.raises(RuntimeError, match="re-derive"), parity(Legacy(), serving_captures_forward=True):
        pass


@pytest.mark.parametrize(("family100", "ceiling"), [(True, 1024), (False, 512)])
def test_capture_ceiling_is_serving_default_per_platform(monkeypatch, family100, ceiling):
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(is_device_capability_family=lambda family: family100)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    assert _load_function("_serving_cudagraph_ceiling")() == ceiling
