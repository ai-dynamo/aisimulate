# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free regression tests for TRT precision labels and native MXFP4 padding."""

import ast
import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[4] / "collector" / "trtllm"


@pytest.mark.parametrize("sm", [80, 89, 90, 100, 103, 120, 121])
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
    assert actual == (kv_dtype if context and sm in (90, 100, 103, 120) else "bfloat16")


class ReachedNativeBuilder(Exception):
    pass


@pytest.mark.parametrize("sm", [90, 100, 103, 120])
@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("quant", ["w4a16_mxfp4", "w4a8_mxfp4_mxfp8"])
@pytest.mark.parametrize("model_name", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "other-mxfp4-model"])
def test_unaligned_mxfp4_honors_native_padding_window(sm, tp, quant, model_name):
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
    if sm in (90, 120):
        with pytest.raises(ValueError, match="weight-layout alignment"):
            namespace["run_moe_torch"](quant, [1], 2880, 2880, 4, 128, tp, 1, False, model_name, perf_filename="unused")
        native_builder.assert_not_called()
        return
    with pytest.raises(ReachedNativeBuilder):
        namespace["run_moe_torch"](quant, [1], 2880, 2880, 4, 128, tp, 1, False, model_name, perf_filename="unused")
    assert native_builder.call_args.kwargs["intermediate_size"] == 2880
    assert native_builder.call_args.kwargs["model_config"].mapping.moe_tp_size == tp
    assert native_builder.call_args.kwargs["bias"] == model_name.startswith("openai/gpt-oss-")
    if model_name.startswith("openai/gpt-oss-"):
        assert namespace["RenormalizeMoeRoutingMethod"].call_args.kwargs["output_dtype"] is namespace["torch"].bfloat16
    else:
        assert not namespace["RenormalizeMoeRoutingMethod"].call_args.kwargs


@pytest.mark.parametrize("outcomes", ["fail,fail", "fail,pass", "pass,fail", "pass,pass", "fatal", "cached"])
def test_moe_autotuning_requires_success_or_loaded_cache(outcomes):
    tree = ast.parse((COLLECTOR / "collect_moe.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_moe_torch")
    tuning = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "cache_loaded"
    )

    class AcceleratorError(RuntimeError):
        pass

    failure = RuntimeError("ordinary tuning failure")
    fatal = AcceleratorError("CUDA context lost")
    effects = {"fail": failure, "pass": None, "fatal": fatal, "cached": None}
    forward = MagicMock(side_effect=[effects[item] for item in outcomes.split(",")])
    namespace = {
        "torch": SimpleNamespace(
            AcceleratorError=AcceleratorError,
            cuda=SimpleNamespace(synchronize=lambda: None),
            inference_mode=nullcontext,
        ),
        "cache_loaded": outcomes == "cached",
        "num_tokens_lists": [2, 4, 8],
        "max_tokens": 4,
        "inspect": inspect,
        "autotune": nullcontext,
        "moe": SimpleNamespace(forward=forward),
        "hidden_states_max_tokens": MagicMock(),
        "logits_max_tokens": MagicMock(),
        "min_latency_mode": False,
    }
    block = compile(ast.Module(body=[tuning], type_ignores=[]), "collect_moe.py", "exec")
    if outcomes == "fail,fail":
        with pytest.raises(RuntimeError, match="any eligible token count") as caught:
            exec(block, namespace)
        assert caught.value.__cause__ is failure
    elif outcomes == "fatal":
        with pytest.raises(AcceleratorError) as caught:
            exec(block, namespace)
        assert caught.value is fatal
    else:
        exec(block, namespace)
    expected_calls = 0 if outcomes == "cached" else 1 if outcomes == "fatal" else 2
    assert forward.call_count == expected_calls
