# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only inverse-CDF regression; exercise the actual sampler implementation."""

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _sampler(monkeypatch, uniforms):
    source = Path(__file__).resolve().parents[3] / "collector" / "helper.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample_power_law")
    calls = []

    def rand(size):
        calls.append(size)
        assert size == len(uniforms)
        return uniforms.copy()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(rand=rand, exp=np.exp))
    namespace = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["sample_power_law"], calls


def test_unit_exponent_is_log_uniform_and_consumes_one_rng_draw(monkeypatch):
    u = np.linspace(0, 1, 5)
    sample, calls = _sampler(monkeypatch, u)
    values = sample(len(u), 1.0, 0.01, 2.0)
    np.testing.assert_allclose(values, np.geomspace(0.01, 2.0, len(u)), rtol=1e-12)
    assert np.isfinite(values).all()
    assert calls == [len(u)]


@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, 3.0])
def test_other_exponents_keep_the_existing_inverse_cdf(monkeypatch, alpha):
    u = np.array([0.0, 0.13, 0.5, 0.93, 1.0])
    sample, _ = _sampler(monkeypatch, u)
    expected = ((2.0 ** (1 - alpha) - 0.01 ** (1 - alpha)) * u + 0.01 ** (1 - alpha)) ** (1 / (1 - alpha))
    np.testing.assert_array_equal(sample(len(u), alpha, 0.01, 2.0), expected)
