# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang fp8_block GEMM weight preparation and timed call (AIC-1744).

``collector/sglang/collect_gemm.py`` imports ``sglang`` (and ``torch``) at
module scope and only runs on a GPU node with those installed. These tests load
``_prepare_fp8_block_weights`` (plus the ``cdiv``/``scale_shape`` helpers it
calls) and ``create_gemm`` (nested inside ``run_gemm``) straight from the
source AST and execute the real bodies against a minimal fake ``torch``. The
sandboxed globals resolve only the names a test wires in, so a call the
fp8_block path is not allowed to make raises ``NameError``.

The weight is repacked the way serving loads it, once at setup, through
serving's ``requant_block_scale_ue8m0_for_deepgemm``; the timed ``gemm_op``
only calls the block-FP8 linear serving dispatched.
"""

import ast
import functools
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_gemm.py"


def _load_functions(*names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded


FLOAT8, FLOAT32, INT32, BFLOAT16 = object(), object(), object(), object()


class _FakeTensor:
    """Shape/dtype-tracking stand-in; values are never read by this test."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def clamp(self, min=None, max=None):  # noqa: A002 - mirrors torch.Tensor.clamp's kwarg names
        return self

    def to(self, dtype):
        return _FakeTensor(self.shape, dtype)

    def __sub__(self, _other):
        return self

    def __mul__(self, _other):
        return self

    __rmul__ = __mul__


class _FakeParameter:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.shape = data.shape
        self.requires_grad = requires_grad


def _normalize_fake_shape(shape: tuple) -> tuple:
    """Mirror ``torch.randn``/``torch.empty``'s dual call shape (sizes or one tuple)."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def _fake_torch():
    return SimpleNamespace(
        float8_e4m3fn=FLOAT8,
        float32=FLOAT32,
        bfloat16=BFLOAT16,
        finfo=lambda _dtype: SimpleNamespace(min=-448.0, max=448.0),
        rand=lambda *shape, device=None: _FakeTensor(tuple(shape), FLOAT32),
        randn=lambda *shape, device=None, dtype=None: _FakeTensor(_normalize_fake_shape(shape), dtype),
        empty=lambda *shape, device=None, dtype=None: _FakeTensor(_normalize_fake_shape(shape), dtype),
        nn=SimpleNamespace(Parameter=_FakeParameter),
    )


def deepgemm_runner(**_kwargs):
    return "deepgemm-out"


def triton_runner(**_kwargs):
    return "triton-out"


def _requant_recorder(calls: list):
    def requant_block_scale_ue8m0_for_deepgemm(
        weight, weight_scale, weight_block_size, use_deepgemm_runner, output_dtype=None, weight_shape=None
    ):
        calls.append(
            {
                "weight": weight,
                "weight_scale": weight_scale,
                "block": list(weight_block_size),
                "use_deepgemm_runner": use_deepgemm_runner,
                "output_dtype": output_dtype,
                "weight_shape": weight_shape,
            }
        )
        if use_deepgemm_runner:
            weight_scale.data = weight_scale.data.to(INT32)  # stand-in for the packed UE8M0 scale
        return use_deepgemm_runner

    return requant_block_scale_ue8m0_for_deepgemm


def _prepare(calls: list):
    return _load_functions(
        "_prepare_fp8_block_weights",
        "cdiv",
        "scale_shape",
        namespace={
            "torch": _fake_torch(),
            "_BLOCK_SIZE": [128, 128],
            "requant_block_scale_ue8m0_for_deepgemm": _requant_recorder(calls),
            "deepgemm_w8a8_block_fp8_linear_with_fallback": deepgemm_runner,
        },
    )["_prepare_fp8_block_weights"]


@pytest.mark.parametrize(
    ("linear", "use_deepgemm_runner"),
    [
        (deepgemm_runner, True),
        (functools.partial(deepgemm_runner, act_scale_ue8m0=True), True),
        (triton_runner, False),
    ],
)
def test_fp8_block_weights_go_through_serving_requant_once(linear, use_deepgemm_runner):
    calls = []
    weight, weight_scale = _prepare(calls)(256, 384, "cpu", linear)

    (call,) = calls
    assert call["use_deepgemm_runner"] is use_deepgemm_runner
    assert call["block"] == [128, 128]
    assert call["output_dtype"] is BFLOAT16
    assert call["weight_shape"] == (256, 384)
    assert call["weight"] is weight and call["weight_scale"] is weight_scale
    assert weight.requires_grad is False and weight_scale.requires_grad is False
    assert weight.data.dtype is FLOAT8 and weight.data.shape == (256, 384)
    # A checkpoint without ue8m0 scales loads fp32 [N/128, K/128] block scales.
    assert weight_scale.data.shape == (2, 3)
    assert weight_scale.data.dtype is (INT32 if use_deepgemm_runner else FLOAT32)


def _load_create_gemm(namespace: dict):
    """AST-extract ``create_gemm`` (nested inside ``run_gemm``) plus the module-level
    helpers its ``fp8_block`` branch calls; its free variables become globals."""
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_prepare_fp8_block_weights", "cdiv", "scale_shape")
    ]
    run_gemm = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gemm")
    create_gemm = next(
        node for node in ast.walk(run_gemm) if isinstance(node, ast.FunctionDef) and node.name == "create_gemm"
    )
    loaded = dict(namespace)
    exec(compile(ast.Module(body=[*helpers, create_gemm], type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded["create_gemm"]


def test_fp8_block_gemm_op_calls_the_dispatched_linear_and_never_repacks():
    requant_calls, linear_calls = [], []

    def block_linear(**kwargs):
        linear_calls.append(kwargs)
        return "out"

    namespace = {
        "torch": _fake_torch(),
        "_BLOCK_SIZE": [128, 128],
        "requant_block_scale_ue8m0_for_deepgemm": _requant_recorder(requant_calls),
        "deepgemm_w8a8_block_fp8_linear_with_fallback": block_linear,
        "block_linear": block_linear,
        "gemm_type": "fp8_block",
        "M": 4,
        "N": 256,
        "K": 256,
        "device": "cpu",
    }
    gemm_op = _load_create_gemm(namespace)()
    assert len(requant_calls) == 1  # setup-time repacking, exactly once

    for _ in range(5):
        assert gemm_op() == "out"
    assert len(requant_calls) == 1  # zero repacking from the timed op

    call = linear_calls[0]
    # Fp8LinearMethod.apply for an unquantized activation (fp8.py:1427-1435 @v0.5.23).
    assert set(call) == {"input", "weight", "block_size", "weight_scale", "input_scale", "bias"}
    assert call["input"].shape == (4, 256) and call["input"].dtype is BFLOAT16
    assert call["weight"] is requant_calls[0]["weight"]
    assert call["weight_scale"] is requant_calls[0]["weight_scale"]
    assert call["block_size"] == [128, 128]
    assert call["input_scale"] is None and call["bias"] is None
    assert len(linear_calls) == 5


def cutlass_runner(**_kwargs):
    return "cutlass-out"


def _label(linear, n, k):
    return _load_functions(
        "_fp8_block_kernel_source",
        namespace={
            "torch": _fake_torch(),
            "_BLOCK_SIZE": [128, 128],
            "deepgemm_w8a8_block_fp8_linear_with_fallback": deepgemm_runner,
            "cutlass_w8a8_block_fp8_linear_with_fallback": cutlass_runner,
            "triton_w8a8_block_fp8_linear": triton_runner,
            # _deepgemm_block_fp8_supported at fp8_utils.py:1449-1459 @v0.5.23.
            "_deepgemm_block_fp8_supported": lambda shape, block, dtype: (
                block == [128, 128] and dtype is BFLOAT16 and shape[0] % 64 == 0 and shape[1] % 128 == 0
            ),
        },
    )["_fp8_block_kernel_source"](linear, n, k)


@pytest.mark.parametrize(
    ("linear", "n", "k", "label"),
    [
        (deepgemm_runner, 4096, 7168, "sglang_deepgemm_gemm_nt_f8f8bf16"),
        (functools.partial(deepgemm_runner, act_scale_ue8m0=True), 4096, 7168, "sglang_deepgemm_gemm_nt_f8f8bf16"),
        (deepgemm_runner, 3336, 4096, "sglang_triton_w8a8_block_fp8"),
        (deepgemm_runner, 4096, 1000, "sglang_triton_w8a8_block_fp8"),
        (cutlass_runner, 4096, 7168, "sglang_cutlass_w8a8_block_fp8"),
        (cutlass_runner, 4160, 7168, "sglang_triton_w8a8_block_fp8"),
        (triton_runner, 4096, 7168, "sglang_triton_w8a8_block_fp8"),
    ],
)
def test_fp8_block_label_names_the_kernel_the_dispatched_linear_runs(linear, n, k, label):
    assert _label(linear, n, k) == label


def test_fp8_block_label_refuses_an_unknown_linear():
    with pytest.raises(RuntimeError, match="unexpected block-FP8 linear"):
        _label(lambda **_kwargs: None, 4096, 7168)
