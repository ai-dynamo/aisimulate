# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang chunked-prefill derivation works on a freshly loaded module."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
DSV4_SOURCE = REPO_ROOT / "collector" / "sglang" / "deepseekv4_sparse_modules.py"


def _load_fresh(*names: str) -> dict:
    """Execute the module's own top-level bindings for ``names``: functions and plain assignments."""
    tree = ast.parse(DSV4_SOURCE.read_text(encoding="utf-8"), filename=str(DSV4_SOURCE))

    def bound_names(node):
        if isinstance(node, ast.FunctionDef):
            return {node.name}
        if isinstance(node, ast.Assign):
            return {target.id for target in node.targets if isinstance(target, ast.Name)}
        return set()

    selected = [node for node in tree.body if bound_names(node) & set(names)]
    loaded: dict = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(DSV4_SOURCE), "exec"), loaded)
    return loaded


def _stub_sglang(monkeypatch, gpu_mem_mb):
    class ServerArgs:
        """sglang>=0.5.20: a msgspec Struct, which rejects the bare ``__new__``."""

        def __new__(cls):
            raise TypeError("Struct.__new__ needs its fields")

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.cuda_graph_config",
        SimpleNamespace(default_cuda_graph_config=lambda: SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=ServerArgs))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils.common",
        SimpleNamespace(get_device_memory_capacity=lambda _device: gpu_mem_mb),
    )


def test_fresh_module_derives_and_caches_the_chunk_size(monkeypatch):
    ns = _load_fresh("_CHUNKED_PREFILL", "_sglang_chunked_prefill_size", "_chunked_prefill_size_from_gpu_mem")
    # The cache starts empty; the first call must not read an unbound global.
    assert "_CHUNKED_PREFILL" in ns and ns["_CHUNKED_PREFILL"] is None

    _stub_sglang(monkeypatch, gpu_mem_mb=180 * 1024)
    assert ns["_sglang_chunked_prefill_size"]() == 16384
    assert ns["_CHUNKED_PREFILL"] == 16384

    # Cached per process: a different device answer no longer changes it.
    _stub_sglang(monkeypatch, gpu_mem_mb=80 * 1024)
    assert ns["_sglang_chunked_prefill_size"]() == 16384


@pytest.mark.parametrize(
    ("gpu_mem_mb", "expected"),
    [(24 * 1024, 2048), (48 * 1024, 4096), (80 * 1024, 8192), (180 * 1024, 16384)],
)
def test_chunk_size_follows_sglang_memory_tiers(monkeypatch, gpu_mem_mb, expected):
    ns = _load_fresh("_CHUNKED_PREFILL", "_sglang_chunked_prefill_size", "_chunked_prefill_size_from_gpu_mem")
    _stub_sglang(monkeypatch, gpu_mem_mb=gpu_mem_mb)
    assert ns["_sglang_chunked_prefill_size"]() == expected
