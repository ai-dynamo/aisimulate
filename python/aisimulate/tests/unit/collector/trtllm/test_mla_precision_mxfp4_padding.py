# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free regression tests for TRT precision labels and native MXFP4 padding."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[4] / "collector" / "trtllm"


@pytest.mark.parametrize("sm", [90, 100, 103, 120])
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
    assert actual == (kv_dtype if context else "bfloat16")


class ReachedNativeBuilder(Exception):
    pass


@pytest.mark.parametrize("sm", [100, 103])
@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("quant", ["w4a16_mxfp4", "w4a8_mxfp4_mxfp8"])
def test_unaligned_mxfp4_reaches_native_weight_padding(sm, tp, quant):
    tree = ast.parse((COLLECTOR / "collect_moe.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_moe_torch")
    native_builder = MagicMock(side_effect=ReachedNativeBuilder)
    namespace = {
        "torch": MagicMock(),
        "gc_collect": lambda: None,
        "aic_debug": 0,
        "QuantAlgo": MagicMock(),
        "QuantConfig": MagicMock(),
        "Mapping": SimpleNamespace,
        "SimpleNamespace": SimpleNamespace,
        "ModelConfig": lambda **_: SimpleNamespace(),
        "NON_GATED_MOE_MODELS": [],
        "ActivationType": MagicMock(),
        "get_sm_version": lambda: sm,
        "_MXFP4_MOE_TYPES": {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"},
        "RenormalizeMoeRoutingMethod": MagicMock(),
        "create_moe": native_builder,
        "inspect": SimpleNamespace(signature=lambda _: SimpleNamespace(parameters={})),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "collect_moe.py", "exec"), namespace)
    with pytest.raises(ReachedNativeBuilder):
        namespace["run_moe_torch"](
            quant, [1], 2880, 2880, 4, 128, tp, 1, False, "openai/gpt-oss-120b", perf_filename="unused"
        )
    assert native_builder.call_args.kwargs["intermediate_size"] == 2880
    assert native_builder.call_args.kwargs["model_config"].mapping.moe_tp_size == tp
