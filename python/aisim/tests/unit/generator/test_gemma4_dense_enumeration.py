# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Producer-side (generator) coverage for dense versus routed Gemma 4.

``check_is_moe`` also drives ``enumerate_profiling_configs``: the wideEP gate,
the multi-node memory floor, and ``build_disagg_parallel_lists(is_moe=...)``.
Before the classification fix dense Gemma 4 took the MoE branch here and the
generator emitted expert-parallel profiling candidates for a dense checkpoint.
These tests pin the branch each checkpoint kind takes without needing a
database, a rendering template, or a real silicon sweep.
"""

import pytest

import aisim.generator.enumerate as enumerate_module
from aisim.generator.enumerate import enumerate_profiling_configs
from aisim.generator.request.sweeper import _resolve_model_facts

pytestmark = pytest.mark.unit

_DENSE = "nvidia/Gemma-4-31B-IT-NVFP4"
_ROUTED = "nvidia/Gemma-4-26B-A4B-NVFP4"

_GIB = 2**30


def _stub_enumeration(monkeypatch, *, weight_gib: int, calls: list[dict]):
    """Replace the DB/render lookups and record the branch decisions.

    ``_build_agg_dgd_for_candidate`` failures are swallowed by the enumeration
    loop, so it is stubbed to succeed: without that the candidate lists could be
    empty and every shape assertion below would pass vacuously.
    """
    monkeypatch.setattr(
        enumerate_module,
        "_get_system_config",
        lambda _system: {"gpus_per_node": 8, "vram_per_gpu": 180 * _GIB},
    )
    monkeypatch.setattr(enumerate_module, "_estimate_model_weight_bytes", lambda _path: weight_gib * _GIB)
    monkeypatch.setattr(enumerate_module, "check_model_hardware_support", lambda *args, **kwargs: True)
    monkeypatch.setattr(enumerate_module, "_build_agg_dgd_for_candidate", lambda **kwargs: {"stub": True})

    real_build = enumerate_module.build_disagg_parallel_lists

    def record(**kwargs):
        calls.append(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(enumerate_module, "build_disagg_parallel_lists", record)


def test_dense_gemma4_producer_configs_are_expert_parallel_free(monkeypatch):
    """A dense checkpoint must only yield pure TP/DP producer candidates."""
    calls: list[dict] = []
    _stub_enumeration(monkeypatch, weight_gib=60, calls=calls)

    prefill, decode, fits, _required_tp = enumerate_profiling_configs(_DENSE, "gb300", "vllm")

    assert fits is True
    assert prefill and decode
    for candidate in (*prefill, *decode):
        assert (candidate.moe_tp, candidate.moe_ep) == (1, 1)
        assert candidate.num_gpus == candidate.tp * candidate.pp * candidate.dp
        assert candidate.num_gpus <= 8
    # wideEP never turns on, and the multi-node memory floor stays off with it.
    assert calls
    assert all(call["is_moe"] is False for call in calls)
    assert all(call["prefill_enable_wideep"] is False and call["decode_enable_wideep"] is False for call in calls)


def test_routed_gemma4_producer_configs_keep_expert_parallel_shapes(monkeypatch):
    """The routed checkpoint must still reach the MoE branch and wideEP ladder."""
    calls: list[dict] = []
    # 8 x 180 GiB of node memory < 2 x model weight, so wideEP auto-enables.
    _stub_enumeration(monkeypatch, weight_gib=800, calls=calls)

    prefill, decode, fits, _required_tp = enumerate_profiling_configs(_ROUTED, "gb300", "vllm")

    assert fits is True
    assert prefill and decode
    assert calls
    assert all(call["is_moe"] is True for call in calls)
    assert all(call["prefill_enable_wideep"] is True and call["decode_enable_wideep"] is True for call in calls)
    assert any(candidate.moe_ep > 1 for candidate in (*prefill, *decode))


@pytest.mark.parametrize(("model_path", "expected"), [(_DENSE, False), (_ROUTED, True)])
def test_generator_model_facts_follow_the_classification(model_path, expected):
    """``ModelFacts.is_moe`` reaches the rule engine, so pin both kinds."""
    assert _resolve_model_facts(model_path, {}).is_moe is expected
