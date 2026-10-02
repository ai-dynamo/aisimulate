# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for model-specific collector case declarations."""

from __future__ import annotations

import pytest
from collector.case_generator import get_common_moe_test_cases, get_sglang_moe_backend

pytestmark = pytest.mark.unit


QWEN38_MAX_FP8 = "Qwen/Qwen3.8-2.4T-A95B-FP8"


def test_qwen38_fp8_recipe_pins_only_blackwell_moe_runner(monkeypatch):
    """The FP8 artifact follows its serving recipe; bf16 keeps the base default."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", QWEN38_MAX_FP8)

    cases = get_common_moe_test_cases(backend="sglang")
    assert cases

    for sm_version in (100, 103):
        assert {get_sglang_moe_backend(case, "fp8_block", sm_version) for case in cases} == {"flashinfer_trtllm"}
        assert {get_sglang_moe_backend(case, "bfloat16", sm_version) for case in cases} == {"triton"}
