# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic observer checks; these fixtures do not qualify GPU cache layouts."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from collector.fpm_forward.runtime import fpm_memory_observer as legacy

pytestmark = pytest.mark.unit


class _Offsets:
    def __init__(self, values):
        self.values = values
        self.reads = 0

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        self.reads += 1
        return list(self.values)


def _fixture(monkeypatch):
    monkeypatch.setitem(sys.modules, "fpm_memory_observer", legacy)
    observer = importlib.import_module("collector.fpm_forward.runtime.instrumentation.kimi_k31")
    runner_type = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"})
    group_type = type("AttentionGroup", (), {"__module__": "vllm.v1.worker.utils"})
    indexer_backend = type(
        "KimiK3IndexerBackend",
        (),
        {
            "__module__": "vllm.models.kimi_k3.common.dsa_cache",
            "get_name": staticmethod(lambda: "DEEPSEEK_V32_INDEXER"),
            "writes_kv_cache_from_block_table": staticmethod(lambda: True),
        },
    )
    mla_backend = type(
        "FlashInferMLASparseTRTLLMBackend",
        (),
        {
            "__module__": "vllm.v1.attention.backends.mla.flashinfer_mla_sparse",
            "get_name": staticmethod(lambda: "FLASHINFER_MLA_SPARSE"),
            "writes_kv_cache_from_block_table": staticmethod(lambda: False),
        },
    )
    indexer = group_type()
    indexer.backend, indexer.layer_names = indexer_backend, ["indexer.0", "indexer.1"]
    indexer.kernel_block_size, indexer.kernel_block_stride = 64, 4128
    indexer.kernel_block_offsets = _Offsets(range(32))
    indexer.kernel_block_table = SimpleNamespace(shape=(1, 8, 128), dtype="torch.int32")
    indexer.maps_slot_mapping = False

    def forbidden_mapping(*_):
        raise AssertionError("An observer must not call native cache/block mapping")

    indexer.map_kv_cache_to_kernel_blocks = forbidden_mapping
    indexer.map_to_kernel_block_table = forbidden_mapping
    mla = group_type()
    mla.backend, mla.layer_names = mla_backend, ["mla.0"]
    mla.kernel_block_size, mla.kernel_block_stride = None, 0
    runner = runner_type()
    runner.kernel_block_sizes, runner.attn_groups = [2048], [[indexer, mla]]
    cache = {
        "groups": [{"block_size_tokens": 2048, "layer_names": ["indexer.0", "indexer.1", "mla.0"]}],
        "layer_components": {"indexer.0": [], "indexer.1": [], "mla.0": []},
    }
    sources = {
        "vllm/v1/worker/gpu/model_runner.py": "pinned-fixture",
        "vllm/v1/worker/utils.py": "pinned-fixture",
        "vllm/models/kimi_k3/common/dsa_cache.py": "pinned-fixture",
        "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py": "pinned-fixture",
    }
    return observer, SimpleNamespace(model_runner=runner), cache, sources, indexer, mla


def test_reads_initialized_kernel_mapping_without_constructing_views(monkeypatch):
    observer, owner, cache, sources, indexer, _ = _fixture(monkeypatch)
    observer.attention_groups(owner, cache, sources)
    mapped = cache["kernel_block_views"]
    assert set(mapped) == {"indexer.0", "indexer.1"}
    assert (
        mapped["indexer.0"]
        == mapped["indexer.1"]
        == {
            "attention_group_class": "vllm.v1.worker.utils.AttentionGroup",
            "backend_class": "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerBackend",
            "kernel_block_size": 64,
            "kernel_block_stride": 4128,
            "kernel_block_offsets": list(range(32)),
            "kernel_block_table_shape": [1, 8, 128],
            "kernel_block_table_dtype": "torch.int32",
            "maps_slot_mapping": False,
            "writes_kv_cache_from_block_table": True,
        }
    )
    assert indexer.kernel_block_offsets.reads == 1
    assert cache["model_runner"]["kernel_block_sizes"] == [2048]
    assert cache["layer_components"] == {"indexer.0": [], "indexer.1": [], "mla.0": []}


@pytest.mark.parametrize("field", ["kernel_block_offsets", "kernel_block_table", "kernel_block_size"])
def test_rejects_missing_initialized_mapping(monkeypatch, field):
    observer, owner, cache, sources, indexer, _ = _fixture(monkeypatch)
    setattr(indexer, field, None)
    with pytest.raises(ValueError, match="mapping is incomplete"):
        observer.attention_groups(owner, cache, sources)


@pytest.mark.parametrize("stride", [-1, True, None])
def test_rejects_unresolved_native_mapping_stride(monkeypatch, stride):
    observer, owner, cache, sources, indexer, _ = _fixture(monkeypatch)
    indexer.kernel_block_stride = stride
    with pytest.raises(ValueError, match="stride is unresolved"):
        observer.attention_groups(owner, cache, sources)


def test_rejects_unpinned_mapping_implementation(monkeypatch):
    observer, owner, cache, sources, _, _ = _fixture(monkeypatch)
    del sources["vllm/v1/worker/utils.py"]
    with pytest.raises(ValueError, match="mapping source is not pinned"):
        observer.attention_groups(owner, cache, sources)


def test_rejects_duplicate_and_uncovered_backend_layers(monkeypatch):
    observer, owner, cache, sources, _, mla = _fixture(monkeypatch)
    mla.layer_names = ["indexer.0"]
    with pytest.raises(ValueError, match="coverage is inconsistent"):
        observer.attention_groups(owner, cache, sources)
    mla.layer_names = []
    with pytest.raises(ValueError, match="do not cover"):
        observer.attention_groups(owner, cache, sources)
