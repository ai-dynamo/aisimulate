# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A collector-raised FIXME(kernel-limit) failure is reported as ``kernel_limit``,
everything else as ``unexpected`` (pipeline summaries classified by hand until
2026-10-08). Runs with PYTHONPATH=collector like test_registry_table_producers."""
import sys

import pytest

pytestmark = pytest.mark.unit


def _collect():
    if "torch" not in sys.modules:
        from unittest.mock import MagicMock

        _torch = MagicMock()
        _torch.AcceleratorError = type("AcceleratorError", (Exception,), {})
        sys.modules["torch"] = _torch
    import collect

    return collect


def test_kernel_limit_prefix_is_classified():
    collect = _collect()
    assert collect._classify_exception(RuntimeError("FIXME(kernel-limit): no head_dim=192 kernel")) == "kernel_limit"
    assert collect._classify_exception(ValueError("  FIXME(kernel-limit): divisibility")) == "kernel_limit"
    assert collect._classify_exception(RuntimeError("CUDA error: an illegal memory access")) == "unexpected"
    assert collect._classify_exception(AssertionError()) == "unexpected"


@pytest.mark.parametrize(
    "path, needle",
    [
        ("collector/sglang/collect_moe.py", "FIXME(kernel-limit): SGLang Triton gated BF16 GELU"),
        ("collector/trtllm/collect_moe.py", "FIXME(kernel-limit): TRT-LLM fused MoE requires"),
        ("collector/trtllm/collect_attn.py", "FIXME(kernel-limit): TRT-LLM trtllm-gen FMHA has no head_dim=192"),
        ("collector/vllm/collect_gemm.py", "FIXME(kernel-limit): vLLM fp8_block GEMM"),
    ],
)
def test_deterministic_guards_carry_the_prefix(path, needle):
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    assert needle in (root / path).read_text(encoding="utf-8")
