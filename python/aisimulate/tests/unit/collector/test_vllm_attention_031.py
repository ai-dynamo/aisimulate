# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for native 0.31 metadata and packed-cache delegation."""

import ast
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from collector.version_resolver import _check_compat, resolve_module
from collector.vllm.registry import REGISTRY

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/collect_attn_v2.py"


class Tensor(np.ndarray):
    def to(self, target):
        return self if isinstance(target, str) else self.astype(target)


def tensor(values, *, dtype=None, device=None):
    return np.asarray(values, dtype=dtype).view(Tensor)


@pytest.fixture
def adapter():
    calls = {}
    cache = object()
    layout = SimpleNamespace(name="native-selected-layout")

    def negotiate(config, supported, specs):
        calls["negotiation"] = (config, supported, specs)
        return layout

    def native_strides(spec, blocks, layers, selected_layout):
        calls["strides"] = (spec, blocks, layers, selected_layout)
        return (4096, 512, 256, 32, 1)

    def native_views(raw, spec, blocks, selected_layout, placement):
        calls["views"] = (raw, spec, blocks, selected_layout, placement)
        return [cache]

    namespace = {
        "accumulate": accumulate,
        "torch": SimpleNamespace(
            int8=np.int8,
            int32=np.int32,
            int64=np.int64,
            tensor=tensor,
            zeros=lambda shape, **kwargs: tensor(np.zeros(shape), **kwargs),
            arange=lambda count, **kwargs: tensor(np.arange(count), **kwargs),
            randperm=lambda count, **kwargs: tensor(np.arange(count)[::-1], **kwargs),
            cat=lambda values: np.concatenate(values).view(Tensor),
        ),
        "CommonAttentionMetadata": SimpleNamespace,
        "KVCacheTensor": SimpleNamespace,
        "get_supported_kv_cache_layouts": lambda backends: [layout],
        "resolve_kv_cache_layout": negotiate,
        "compute_layout_strides": native_strides,
        "create_kv_cache_views": native_views,
    }
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace), calls, cache


@pytest.mark.parametrize("seq_lens,query_lens", [([70, 5], [1, 1]), ([70, 5], [70, 5])])
def test_paged_positions_metadata_and_native_cache_placement(adapter, seq_lens, query_lens):
    module, calls, cache = adapter
    batch = SimpleNamespace(batch_size=2, seq_lens=seq_lens, query_lens=query_lens)
    spec = SimpleNamespace(block_size=64, page_size_bytes=512)
    config, backend = object(), object()
    metadata, actual_cache, history = module.prepare_cache(
        batch, config, backend, spec, 8, "cpu", query_lens == seq_lens
    )
    assert actual_cache is cache
    np.testing.assert_array_equal(metadata.query_start_loc, [0, query_lens[0], sum(query_lens)])
    np.testing.assert_array_equal(metadata.seq_lens, seq_lens)
    assert metadata.num_actual_tokens == sum(query_lens)
    assert metadata.max_seq_len == 70 and metadata.max_query_len == max(query_lens)
    assert metadata.causal is True
    assert not hasattr(metadata, "_seq_lens_cpu") and not hasattr(metadata, "_num_computed_tokens_cpu")

    expected_query, expected_history, expected_positions = [], [], []
    for request, (length, queries) in enumerate(zip(seq_lens, query_lens, strict=True)):
        for position in range(length):
            block = int(metadata.block_table_tensor[request, position // 64])
            assert block > 0
            slot = block * 64 + position % 64
            if position < length - queries:
                expected_history.append(slot)
            else:
                expected_query.append(slot)
                expected_positions.append(position)
    np.testing.assert_array_equal(metadata.slot_mapping, expected_query)
    np.testing.assert_array_equal(metadata.positions, expected_positions)
    np.testing.assert_array_equal(history, expected_history)
    assert len(set(expected_query + expected_history)) == sum(seq_lens)
    assert not set(expected_query) & set(expected_history)
    assert calls["negotiation"] == (config, [["native-selected-layout"]], [spec])
    assert calls["strides"][:3] == (spec, 8, 1)
    raw, actual_spec, blocks, _, placement = calls["views"]
    assert actual_spec is spec and blocks == 8
    assert raw.dtype == np.int8 and raw.nbytes == 4096 and not raw.any()
    assert (placement.layer_stride, placement.block_stride) == (4096, 512)


def test_allocation_error_and_other_backend_are_not_silent_fallbacks(adapter):
    module, calls, _ = adapter
    with pytest.raises(ValueError, match="null block"):
        module.prepare_cache(
            SimpleNamespace(batch_size=1, seq_lens=[65], query_lens=[1]),
            object(),
            object(),
            SimpleNamespace(block_size=64),
            2,
            "cpu",
            False,
        )
    assert calls == {}
    module.validate_backend("TRITON_ATTN")
    with pytest.raises(NotImplementedError, match="FLASH_ATTN"):
        module.validate_backend("FLASH_ATTN")
    with pytest.raises(NotImplementedError, match="non-windowed"):
        module.run_attention_torch(1, 64, 4, 1, 192, True, True, 128, perf_filename="unused")


def test_current_route_is_exact_and_legacy_route_remains():
    for op in ("attention_context", "attention_generation"):
        entry = next(entry for entry in REGISTRY if entry.op == op)
        assert resolve_module(entry, "0.25.0") == "collector.vllm.collect_attn_v1"
        assert resolve_module(entry, "0.31.0") == "collector.vllm.collect_attn_v2"
    tree = ast.parse(SOURCE.read_text())
    compat = next(node.value.value for node in tree.body if isinstance(node, ast.Assign))
    assert _check_compat(compat, "0.31.0+cu130")
    assert not _check_compat(compat, "0.32.0")
