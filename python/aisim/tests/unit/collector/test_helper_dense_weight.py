# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helper.fill_dense_weight_: dense, finite expert weights in every storage dtype."""

import pytest
from collector import helper

pytestmark = pytest.mark.unit


@pytest.fixture
def torch():
    module = pytest.importorskip("torch")
    # Other collector tests leave a mock "torch" in sys.modules; these checks need the real one.
    if not isinstance(getattr(module, "__file__", None), str):
        pytest.skip("torch is a test stub here")
    return module


def test_fp8_weights_spread_over_the_e4m3_exponents(torch, monkeypatch):
    # A small chunk forces several float32 temporaries, as a large expert tensor would.
    monkeypatch.setattr(helper, "_DENSE_FILL_CHUNK", 1000)
    w = torch.zeros(4, 64, 96, dtype=torch.float8_e4m3fn)
    helper.fill_dense_weight_(w)
    v = w.float()
    assert torch.isfinite(v).all()
    assert (v == 0).float().mean() < 0.02
    assert v.abs().max() <= 448
    assert len(torch.unique(v.abs())) > 60  # many exponent/mantissa codes, not one value


def test_wide_floats_are_checkpoint_like(torch):
    w = torch.zeros(256, 256, dtype=torch.bfloat16)
    helper.fill_dense_weight_(w)
    assert 0.015 < w.float().std() < 0.025


@pytest.mark.parametrize("dtype_name", ["uint8", "int8", "int32"])
def test_packed_integer_weights_use_every_bit(torch, dtype_name):
    dtype = getattr(torch, dtype_name)
    w = torch.zeros(128, 128, dtype=dtype)
    helper.fill_dense_weight_(w)
    bits = w.view(torch.uint8)
    assert (bits & 0x0F).bool().float().mean() > 0.9  # low nibble
    assert (bits & 0xF0).bool().float().mean() > 0.9  # high nibble
    if dtype == torch.int32:
        assert (w < 0).any()


def test_an_unknown_storage_dtype_fails_loudly(torch):
    with pytest.raises(TypeError):
        helper.fill_dense_weight_(torch.zeros(4, dtype=torch.int64))


def test_weight_matrix_names():
    names = ["w13_weight", "w2_weight", "w13_weight_packed", "w2_qweight"]
    others = ["w13_weight_scale", "w13_weight_scale_2", "w13_weight_shape", "w13_weight_g_idx", "w13_qzeros"]
    assert all(n.endswith(helper.MOE_WEIGHT_MATRIX_SUFFIXES) for n in names)
    assert not any(n.endswith(helper.MOE_WEIGHT_MATRIX_SUFFIXES) for n in others)
