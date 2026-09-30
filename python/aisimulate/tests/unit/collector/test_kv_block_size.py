# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV page size is a per-SM branch (owner decision 2026-09-30 after the B200
campaign caught the 64 literal running a different FMHA cubin on SM100)."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "utils.py"


def _kv_block_size():
    """The module imports vllm at import time; lift just the pure function."""
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kv_block_size")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["kv_block_size"]


kv_block_size = _kv_block_size()

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sm,op,expected", [
    (90, "attention", 64), (90, "mla", 64),
    (100, "attention", 16), (103, "attention", 16), (100, "mla", 32), (103, "mla", 32),
    (120, "attention", 64), (120, "mla", 64),
])
def test_page_size_per_sm(sm, op, expected, monkeypatch):
    monkeypatch.delenv("AIS_KV_BLOCK_SIZE", raising=False)
    assert kv_block_size(sm, op) == expected


def test_ab_override_hook(monkeypatch):
    monkeypatch.setenv("AIS_KV_BLOCK_SIZE", "16")
    assert kv_block_size(90, "attention") == 16
