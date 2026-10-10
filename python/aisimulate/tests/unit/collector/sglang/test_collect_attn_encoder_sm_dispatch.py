# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The encoder collector picks the vision-attention lane the way sglang does: by capability MAJOR
(vision.py:1298-1303 @0.5.21), so SM103 (B300) shares FA4 with SM100 instead of falling to Triton."""
import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_attn_encoder.py"


def _serving_vision_backend():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "serving_vision_backend")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["serving_vision_backend"]


@pytest.mark.parametrize(
    "sm, expected",
    [(89, "triton_attn"), (90, "fa3"), (100, "fa4"), (103, "fa4"), (120, "triton_attn"), (121, "triton_attn")],
)
def test_vision_backend_follows_capability_major(sm, expected):
    assert _serving_vision_backend()(sm) == expected


def test_run_dispatch_uses_the_family_helper_not_exact_sm():
    src = SOURCE.read_text(encoding="utf-8")
    assert "if sm == 100" not in src and "if sm == 90" not in src
    assert 'backend = serving_vision_backend(sm)' in src
