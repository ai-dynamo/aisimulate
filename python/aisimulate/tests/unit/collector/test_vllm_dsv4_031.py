# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import math
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from collector.version_resolver import resolve_module
from collector.vllm.registry import REGISTRY

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/collect_dsv4_attn_v2.py"


class Tensor(np.ndarray):
    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)


def load(functions, namespace):
    nodes = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in functions]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


def make_common(batch, block_size, num_blocks, device, is_context, **kwargs):
    table = np.zeros((len(batch.seq_lens), max(math.ceil(n / block_size) for n in batch.seq_lens)), dtype=np.int32)
    slots, cursor = [], 1
    for i, (length, query) in enumerate(zip(batch.seq_lens, batch.query_lens, strict=True)):
        count = math.ceil(length / block_size)
        table[i, :count] = np.arange(cursor, cursor + count)
        cursor += count
        for pos in range(length - query, length):
            slots.append(table[i, pos // block_size] * block_size + pos % block_size)
    assert cursor <= num_blocks
    return SimpleNamespace(block_table_tensor=table.view(Tensor), slot_mapping=np.array(slots)), None


@pytest.mark.parametrize("kernel_size", [4, 2])
def test_group_pages_do_not_alias_and_virtual_split_preserves_slot_mapping(kernel_size):
    namespace = {
        "create_common_metadata": make_common,
        "torch": SimpleNamespace(
            where=lambda *args: np.where(*args).view(Tensor), arange=lambda n, **kw: np.arange(n, dtype=kw["dtype"])
        ),
    }
    module = load({"_group_common_metadata"}, namespace)
    batch = SimpleNamespace(seq_lens=[6, 3], query_lens=[2, 1], batch_size=2)
    common = SimpleNamespace(is_prefilling=object())
    first = module._group_common_metadata(common, batch, 4, kernel_size, 10, 0, "cpu")
    second = module._group_common_metadata(common, batch, 4, kernel_size, 10, 3, "cpu")
    np.testing.assert_array_equal(second.slot_mapping - first.slot_mapping, [12, 12, 12])
    for table, slots in [
        (first.block_table_tensor, first.slot_mapping),
        (second.block_table_tensor, second.slot_mapping),
    ]:
        expected = []
        for i, (length, query) in enumerate(zip(batch.seq_lens, batch.query_lens, strict=True)):
            for pos in range(length - query, length):
                expected.append(table[i, pos // kernel_size] * kernel_size + pos % kernel_size)
        np.testing.assert_array_equal(slots, expected)
    assert not set(first.slot_mapping) & set(second.slot_mapping)
    assert second.is_prefilling is common.is_prefilling
    with pytest.raises(ValueError, match="divide"):
        module._group_common_metadata(common, batch, 4, 3, 10, 3, "cpu")


@pytest.mark.parametrize("allocation_fails", [False, True])
def test_native_packed_groups_allocate_once_and_restore_capacity_override(allocation_fails):
    class Uniform:
        def __init__(self, specs):
            self.kv_cache_specs = specs
            self.block_size = 256

    main_spec, index_spec, state_spec = [SimpleNamespace(block_size=n) for n in (256, 256, 4)]
    groups = [
        SimpleNamespace(layer_names=["main", "index"], kv_cache_spec=Uniform({"main": main_spec, "index": index_spec})),
        SimpleNamespace(layer_names=["state"], kv_cache_spec=state_spec),
    ]
    layers = {
        name: SimpleNamespace(
            prefix=name, get_kv_cache_spec=lambda config, spec=spec: spec, get_attn_backend=lambda: "native"
        )
        for name, spec in {"main": main_spec, "index": index_spec, "state": state_spec}.items()
    }
    swa = SimpleNamespace(prefix="swa", get_kv_cache_spec=lambda config: None)
    layers["swa"] = swa
    module_layer = layers["main"]
    module_layer.swa_cache_layer = swa
    module_layer.indexer = SimpleNamespace(k_cache=layers["index"], compressor=None)
    module_layer.compressor = SimpleNamespace(state_cache=layers["state"])
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )
    common = SimpleNamespace(seq_lens_cpu_upper_bound=np.array([257]), query_start_loc_cpu=np.array([0, 1]))
    calls = []

    def cache_config(cfg, actual_groups, available_memory):
        assert actual_groups is groups and available_memory == 0
        assert cfg.cache_config.num_gpu_blocks_override == 68  # null + 2 latent/indexer + 65 state pages
        if allocation_fails:
            raise RuntimeError("native allocation rejected")
        return SimpleNamespace(num_blocks=68)

    class Attention:
        def __init__(self, backend, names, spec, group_id):
            assert backend == "native"
            self.layer_names, self.spec = names, spec

        def create_metadata_builders(self, cfg, device, kernel_size):
            calls.append(("builder", self.layer_names, kernel_size))

        def get_metadata_builder(self):
            return SimpleNamespace(build=lambda prefix, grouped: grouped)

    backing = object()
    caches = dict.fromkeys(("main", "index", "state"), backing)

    def allocate(cfg, device, layout, kernels):
        calls.append(("allocate", kernels))
        return caches

    def bind(actual_caches, ctx, *, kv_cache_groups):
        assert actual_caches is caches and ctx is layers and kv_cache_groups is groups
        calls.append(("bind",))

    module = load(
        {"_build_metadata_and_bind_caches"},
        {
            "torch": SimpleNamespace(device=lambda device: device),
            "pairwise": pairwise,
            "math": math,
            "UniformTypeKVCacheSpecs": Uniform,
            "AttentionGroup": Attention,
            "BatchSpec": lambda **kwargs: SimpleNamespace(**kwargs),
            "get_supported_kv_cache_layouts": lambda backends: [SimpleNamespace(name="BLHNC")],
            "resolve_kv_cache_layout": lambda *args: SimpleNamespace(name="BLHNC"),
            "get_kv_cache_groups": lambda *args: groups,
            "get_kv_cache_config_from_groups": cache_config,
            "prepare_kernel_block_sizes": lambda cfg, attn: [256, 4],
            "allocate_kv_cache": allocate,
            "bind_kv_cache_to_layers": bind,
            "_group_common_metadata": lambda common, batch, block, kernel, blocks, offset, device: (block, offset),
        },
    )
    if allocation_fails:
        with pytest.raises(RuntimeError, match="native allocation rejected"):
            module._build_metadata_and_bind_caches(module_layer, config, common, device="cpu")
    else:
        metadata = module._build_metadata_and_bind_caches(module_layer, config, common, device="cpu")
        assert metadata == {"main": (256, 0), "index": (256, 0), "state": (4, 2)}
        assert sum(c[0] == "allocate" for c in calls) == 1
        assert calls[1] == ("bind",)
    assert config.cache_config.num_gpu_blocks_override is None


def test_dsv4_routes_are_exact_and_legacy_is_retained():
    for entry in REGISTRY:
        if entry.op.startswith("dsv4_"):
            assert resolve_module(entry, "0.31.0") == "collector.vllm.collect_dsv4_attn_v2"
            assert resolve_module(entry, "0.25.0") == "collector.vllm.collect_dsv4_attn_v1"
