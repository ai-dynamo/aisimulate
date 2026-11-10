# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""framework_kv_block_size: the cache page comes from the model's first attention layer.

The fakes reproduce vLLM 0.30.0's rules: AttentionBackend.get_preferred_block_size
(v1/attention/backend.py:150-158) and select_common_block_size (v1/worker/utils.py:330-385).
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
UTILS = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "utils.py"


def _backend(*sizes):
    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return list(sizes)

        @classmethod
        def get_preferred_block_size(cls, default_block_size):
            return default_block_size if default_block_size in sizes else min(sizes)

    return Backend


def _select_common_block_size(kv_manager_block_size, backends):
    if all(kv_manager_block_size in b.get_supported_kernel_block_sizes() for b in backends):
        return kv_manager_block_size
    common = set.intersection(*(set(b.get_supported_kernel_block_sizes()) for b in backends))
    return max(size for size in common if kv_manager_block_size % size == 0)


@pytest.fixture
def kv_block_size(monkeypatch):
    for name, attrs in (
        ("vllm", {}),
        ("vllm.config", {}),
        ("vllm.config.cache", {"CacheConfig": SimpleNamespace(DEFAULT_BLOCK_SIZE=16)}),
        ("vllm.v1", {}),
        ("vllm.v1.worker", {}),
        ("vllm.v1.worker.utils", {"select_common_block_size": _select_common_block_size}),
    ):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    tree = ast.parse(UTILS.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "framework_kv_block_size")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(UTILS), "exec"), namespace)
    return namespace["framework_kv_block_size"]


def test_dsa_sparse_mla_runs_at_the_indexer_cache_page(kv_block_size):
    flashinfer_mla_sparse = _backend(32, 64)  # flashinfer_mla_sparse.py:85-86 @0.30.0
    dsa_indexer = _backend(64)  # mla/indexer.py:203-204 @0.30.0 (CUDA)
    # alone, the sparse MLA backend would pick its own preferred 32 ...
    assert kv_block_size(flashinfer_mla_sparse) == 32
    # ... but serving sizes the cache from the indexer, registered first, and keeps 64
    assert kv_block_size(flashinfer_mla_sparse, cache_backend_cls=dsa_indexer) == 64
    assert kv_block_size(flashinfer_mla_sparse, cache_backend_cls=dsa_indexer, user_block_size=32) == 32


def test_first_layer_backend_is_the_default_cache_backend(kv_block_size):
    flash_attn = _backend(16, 32, 64)
    assert kv_block_size(flash_attn) == 16
    assert kv_block_size(_backend(64)) == 64
