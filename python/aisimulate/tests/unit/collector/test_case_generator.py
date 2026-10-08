# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for model-specific collector case declarations."""

from __future__ import annotations

import pytest

from collector import case_generator
from collector.case_generator import get_attention_head_configs, get_common_moe_test_cases, get_sglang_moe_backend

pytestmark = pytest.mark.unit


# Two profiles at the same physical (heads, kv, head_dim, window) tuple: one no-sink,
# one with sink. population_key omits sink, so they collide on the same key.
_SINK_COLLISION_SWEEP = {
    "head_profiles": [
        {"id": "nosink", "head_dim": 64, "window_sizes": [0], "query_head_counts": [8], "kv_head_options": [1]},
        {
            "id": "sink",
            "head_dim": 64,
            "window_sizes": [0],
            "query_head_counts": [8],
            "kv_head_options": [1],
            "has_attention_sink": True,
        },
    ],
}


def _dedup_sink(monkeypatch, *, xpu):
    monkeypatch.setattr(case_generator, "_xpu_available", lambda: xpu)
    configs = get_attention_head_configs(_SINK_COLLISION_SWEEP, phase="context", include_model_profiles=False)
    assert len(configs) == 1  # both fold onto one persisted key either way
    return configs[0]


def test_sink_wins_shared_key_on_xpu(monkeypatch) -> None:
    # XPU: sink selects a distinct kernel, so it replaces the no-sink first-seen case.
    assert _dedup_sink(monkeypatch, xpu=True).has_attention_sink is True


def test_sink_dedup_is_noop_off_xpu(monkeypatch) -> None:
    # NV/default: sink is a same-kernel arg, so the upstream first-seen (no-sink) case stands.
    assert _dedup_sink(monkeypatch, xpu=False).has_attention_sink is False


QWEN38_MAX_FP8 = "Qwen/Qwen3.8-2.4T-A95B-FP8"


def test_qwen38_fp8_recipe_pins_only_blackwell_moe_runner(monkeypatch):
    """The FP8 artifact follows its serving recipe; bf16 keeps the base default."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", QWEN38_MAX_FP8)

    cases = get_common_moe_test_cases(backend="sglang")
    assert cases

    for sm_version in (100, 103):
        assert {get_sglang_moe_backend(case, "fp8_block", sm_version) for case in cases} == {"flashinfer_trtllm"}
        assert {get_sglang_moe_backend(case, "bfloat16", sm_version) for case in cases} == {"triton"}
