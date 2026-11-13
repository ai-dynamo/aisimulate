# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free regression tests for TRT MLA precision labels.

The MoE padding and autotuning checks that lived here moved to
test_collect_moe_worker_recycle.py with the 1.4.0rc2 case body.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[4] / "collector" / "trtllm"


@pytest.mark.parametrize("sm", [80, 89, 90, 100, 103, 107, 120, 121])
@pytest.mark.parametrize("kv_dtype", ["bfloat16", "fp8"])
@pytest.mark.parametrize("context", [True, False])
def test_mla_logged_compute_precision(sm, kv_dtype, context):
    tree = ast.parse((COLLECTOR / "collect_mla.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_attn_for_backend"
    )
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "log_perf"
    )
    fields = next(item.value.elts[0] for item in call.keywords if item.arg == "item_list")
    dtype_expr = next(
        value
        for key, value in zip(fields.keys, fields.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "mla_dtype"
    )
    actual = eval(
        compile(ast.Expression(dtype_expr), "collect_mla.py", "eval"),
        {
            "dtype_str": kv_dtype,
            "is_context_phase": context,
            "get_sm_version": lambda: sm,
        },
    )
    # thop/attentionOp.cpp:1466-1470 at the pin: FP8 context MLA on SM90/100/103/107/120.
    assert actual == (kv_dtype if context and sm in (90, 100, 103, 107, 120) else "bfloat16")
