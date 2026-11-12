# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resident expert weights alone can exceed a GPU before workspace is considered."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/trtllm/collect_moe.py"


def _guard():
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_refuse_moe_weight_capacity")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


def test_nemotron_bf16_resident_weights_exceed_h100_before_allocation():
    # Two non-gated matrices: 512 experts * 8192 * 5120 * 2 matrices * 2 bytes = 80 GiB.
    with pytest.raises(RuntimeError, match="FIXME\\(kernel-limit\\): does not fit: resident expert weights"):
        _guard()(8192, 5120, 512, 1, 1, 16, False, 85_019_590_656)


@pytest.mark.parametrize("tp,ep,bits", [(2, 1, 16), (1, 2, 16), (1, 1, 8)])
def test_smaller_weight_bound_does_not_claim_workspace_fit(tp, ep, bits):
    _guard()(8192, 5120, 512, tp, ep, bits, False, 85_019_590_656)


def test_gated_three_matrix_bound():
    # Three 2-byte matrices of 2*4 elements: 48 bytes, independent of a model name.
    with pytest.raises(RuntimeError, match="at least 48 bytes"):
        _guard()(2, 4, 1, 1, 1, 16, True, 47)
