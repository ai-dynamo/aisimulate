# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A collector-raised FIXME(kernel-limit) failure is reported as ``kernel_limit``,
everything else as ``unexpected`` (pipeline summaries classified by hand until
2026-10-08). Runs with PYTHONPATH=collector like test_registry_table_producers."""

import collect
import pytest

pytestmark = pytest.mark.unit


def test_kernel_limit_prefix_is_classified():
    assert collect._classify_exception(RuntimeError("FIXME(kernel-limit): no head_dim=192 kernel")) == "kernel_limit"
    assert collect._classify_exception(ValueError("  FIXME(kernel-limit): divisibility")) == "kernel_limit"
    assert collect._classify_exception(RuntimeError("CUDA error: an illegal memory access")) == "unexpected"
    assert collect._classify_exception(AssertionError()) == "unexpected"
