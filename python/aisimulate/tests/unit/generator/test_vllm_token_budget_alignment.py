# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vllm token budget renders as a multiple of 64.

vLLM stores per-cache-group slot mappings as rows of one 2D buffer whose row
width is max_num_batched_tokens; a non-multiple-of-8 budget misaligns every
row after the first to 64 bytes and crashes kernels that assert alignment
(DeepSeek-V4 cutedsl compress @ 0.29.0). The generator must therefore never
render a non-idiomatic budget. See backend_config_mapping.yaml, max_num_tokens.
"""
import pytest

from aisimulate.generator.rendering.engine import render_backend_parameters

pytestmark = pytest.mark.unit


def _vllm_budget(value):
    out = render_backend_parameters({"max_num_tokens": value}, "vllm")
    return out.get("max_num_tokens", {}).get("max-num-batched-tokens")


def test_odd_budget_rounds_up_to_64():
    assert _vllm_budget(6012) == 6016


def test_aligned_budget_unchanged():
    assert _vllm_budget(6016) == 6016
    assert _vllm_budget(2048) == 2048


def test_rounds_up_never_down():
    assert _vllm_budget(6017) == 6080


def test_absent_budget_stays_omitted():
    out = render_backend_parameters({}, "vllm")
    assert "max_num_tokens" not in out


def test_trtllm_budget_passes_through_unrounded():
    out = render_backend_parameters({"max_num_tokens": 6012}, "trtllm")
    assert out["max_num_tokens"]["max_num_tokens"] == 6012
