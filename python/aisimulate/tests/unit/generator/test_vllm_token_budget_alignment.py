# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every rendered token budget is a multiple of 64, on every backend.

Engines size per-cache-group slot-mapping rows and scheduler chunks by these
budgets; a non-idiomatic width misaligns every row after the first (vLLM
0.29 DeepSeek-V4: 6012 crashes the cutedsl compress kernel at first forward,
6016 passes). The alignment lives in ONE place — backend_config_mapping.yaml —
so rule formulas stay plain and no backend is a special case (owner decision
2026-09-24).
"""
import pytest

from aisimulate.generator.rendering.engine import render_backend_parameters

pytestmark = pytest.mark.unit


def _render(params, backend):
    return render_backend_parameters(params, backend)


@pytest.mark.parametrize(
    ("backend", "param", "key"),
    [
        ("vllm", "max_num_tokens", "max-num-batched-tokens"),
        ("trtllm", "max_num_tokens", "max_num_tokens"),
        ("sglang", "max_prefill_tokens", "max-prefill-tokens"),
    ],
)
def test_odd_budget_rounds_up_to_64_on_every_backend(backend, param, key):
    assert _render({param: 6012}, backend)[param][key] == 6016
    assert _render({param: 6017}, backend)[param][key] == 6080   # up, never down
    assert _render({param: 6016}, backend)[param][key] == 6016
    assert _render({param: 2048}, backend)[param][key] == 2048


def test_sglang_chunked_prefill_size_follows_the_aligned_budget():
    out = _render({"enable_chunked_prefill": True, "max_num_tokens": 6012}, "sglang")
    assert out["enable_chunked_prefill"]["chunked-prefill-size"] == 6016


def test_absent_budget_stays_omitted():
    for backend in ("vllm", "trtllm"):
        assert "max_num_tokens" not in _render({}, backend)
    assert "max_prefill_tokens" not in _render({}, "sglang")


def test_rule_engine_formulas_stay_plain_and_render_aligned():
    """The vllm rule plugin emits the plain formula (512+4000+1500 = 6012, the
    DSV4 crash trigger); the mapping is what aligns it."""
    from aisimulate.generator.rendering.rule_engine import apply_rule_plugins

    pv = {
        "SlaConfig": {"isl": 4000, "osl": 500},
        "DynConfig": {},
        "params": {"agg": {"max_batch_size": 512}},
    }
    apply_rule_plugins(pv, backend="vllm")
    agg = pv["params"]["agg"]
    unrounded = agg["max_batch_size"] + 4000 + 1500
    assert agg["max_num_tokens"] == unrounded
    rendered = _render({"max_num_tokens": agg["max_num_tokens"]}, "vllm")
    assert rendered["max_num_tokens"]["max-num-batched-tokens"] % 64 == 0
    assert rendered["max_num_tokens"]["max-num-batched-tokens"] >= unrounded
