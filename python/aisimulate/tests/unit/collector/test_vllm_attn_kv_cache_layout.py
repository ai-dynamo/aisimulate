# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""collect_attn.kv_cache_in_resolved_layout: the collector's paged KV cache must be laid out
in the layout the framework RESOLVED for the backend under test (RFC #42082), not a fixed HND.

Origin: SM120 collection 2026-10-01 — FlashInfer resolves NHD outside SM100 and its XQA
decode asserted on the collector's contiguous [B, H, N, 2D] memory for every fp8-KV GQA
generation case (20,458 failures); SM100 (HND) and FLASH_ATTN never noticed.
The helper is pure torch, so the test imports it through ast, without vLLM."""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "collect_attn.py"


def _helper():
    tree = ast.parse(_SRC.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "kv_cache_in_resolved_layout")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_SRC), "exec"), ns)
    return ns["kv_cache_in_resolved_layout"]


# KVCacheLayout.layer_view_order values (vllm/v1/kv_cache_layout.py @v0.30.0): logical [B, H, N, C]
LBHNC = (0, 1, 2, 3)  # HND, identity
LBNHC = (0, 2, 1, 3)  # NHD


@pytest.mark.parametrize("order", [LBHNC, LBNHC, (1, 0, 2, 3)])
def test_logical_view_is_unchanged_and_physical_order_is_contiguous(order):
    f = _helper()
    kv = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)  # [B, H, N, C]
    out = f(kv, order)
    assert out.shape == kv.shape and torch.equal(out, kv)  # same logical [B, H, N, C] values
    assert out.permute(*order).is_contiguous()  # what the backend's permute(*layer_view_order) needs


def test_identity_layout_returns_contiguous_input():
    f = _helper()
    kv = torch.zeros(2, 3, 4, 6)
    assert f(kv, LBHNC).data_ptr() == kv.data_ptr()


def test_collector_uses_the_resolved_layout():
    src = _SRC.read_text()
    assert "kv_cache_in_resolved_layout(" in src
    assert "get_resolved_kv_cache_layout().layer_view_order" in src


def test_flashinfer_decode_label_names_the_kernel():
    """FlashInferTrtllmAPIDecode carries XQA and trtllm-gen in one class; the label must keep the kernel."""
    src = _SRC.read_text()
    assert 'kernel = getattr(phase_metadata, "kernel", None)' in src
    vocab = (Path(__file__).resolve().parents[3] / "collector" / "kernel_source_backends.yaml").read_text()
    for label in (
        "vllm_flashinfer_flashinfertrtllmapidecode_xqa",
        "vllm_flashinfer_flashinfertrtllmapidecode_trtllm_gen",
    ):
        assert label in vocab
