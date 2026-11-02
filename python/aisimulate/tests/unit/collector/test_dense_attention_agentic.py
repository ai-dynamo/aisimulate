# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the native collector getters without importing GPU-only frameworks."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.attention_memory import filter_attention_cases
from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def collector_functions(backend, memory_gib=None):
    path = ROOT / "collector" / backend / "collect_attn.py"
    names = {
        "_int_list",
        "_skip_trtllm_sm120_fp8_context_fmha",
        "_generation_target_sequence_lengths",
        "get_context_attention_test_cases",
        "get_generation_attention_test_cases",
        "_attention_memory_bytes",
        "default_attention_backend",
    }
    nodes = [
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    cuda = SimpleNamespace(
        is_available=lambda: memory_gib is not None,
        current_device=lambda: 0,
        get_device_properties=lambda _device: SimpleNamespace(total_memory=int(memory_gib * 2**30)),
    )
    ns = {
        "os": os,
        "tensorrt_llm": SimpleNamespace(__version__="1.3.0rc29"),
        "get_sm_version": lambda: 103,
        "get_attention_context_shape_sweeps": get_attention_context_shape_sweeps,
        "get_attention_generation_shape_sweeps": get_attention_generation_shape_sweeps,
        "get_attention_head_configs": get_attention_head_configs,
        "torch": SimpleNamespace(cuda=cuda, bfloat16="bf16"),
        "current_platform": SimpleNamespace(fp8_dtype=lambda: "fp8"),
        "filter_attention_cases": filter_attention_cases,
        "_deployment_tokens_per_block": lambda: 32,
        "_attention_memory_backend": lambda *_args: ("FLASHINFER", 16),
        "use_trtllm_attention": lambda *_args: True,
        "can_use_trtllm_attention": lambda *_args, **_kwargs: True,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_shared_context_anchors_and_budgets_reach_actual_getters(monkeypatch, backend):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    cases = collector_functions(backend)["get_context_attention_test_cases"]()
    gqa = {(c[0], c[1]) for c in cases if c[2:5] == [4, 1, 128]}
    mha = {(c[0], c[1]) for c in cases if c[2:5] == [4, 4, 128]}
    for pairs in (gqa, mha):
        assert {(16, s) for s in (24576, 32768, 40960)} <= pairs
        assert {(1, 262144), (2, 262144)} <= pairs
        assert (4, 262144) not in pairs
    assert all(c[0] * c[1] <= 655360 for c in cases)
    assert all(c[0] * c[1] * c[3] * c[4] * 2 < 2147483647 for c in cases)


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_decode_keeps_long_histories_and_largest_batch_anchor(monkeypatch, backend):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    cases = collector_functions(backend)["get_generation_attention_test_cases"]()
    pairs = {(c[0], c[1] + 1) for c in cases if c[2:5] == [32, 1, 128]}
    assert {(1, s) for s in (65536, 131072, 262144, 524288, 1048576)} <= pairs
    assert (256, 65536) in pairs
    assert (32, 1048576) not in pairs
    assert all(c[0] * (c[1] + 1) <= 16777216 for c in cases)


def test_xpu_declarations_keep_their_previous_limits():
    context = get_attention_context_shape_sweeps("vllm_xpu")[0]
    generation = get_attention_generation_shape_sweeps("vllm_xpu")[0]
    assert context["max_tokens_self_attention"] == 65536
    assert context["max_tokens_grouped_query_attention"] == 131072
    assert max(context["sequence_lengths"]) == 16384
    assert max(generation["sequence_lengths"]) == 131072
    assert generation["min_batch_options_per_sequence"] == 4
    assert generation["drop_largest_sequence_for_batch_at_least"] == 256


def test_memory_admission_uses_device_capacity_and_logs_drops(capsys):
    ns = collector_functions("trtllm", 80)
    # QKV, packed QKV and output exceed an 80GiB GPU; Blackwell retains it.
    case = [16, 40960, 96, 1, 256, 0, False, False, True, "TRTLLM"]
    estimate = ns["_attention_memory_bytes"]
    assert 80 * 2**30 < estimate(case) < 268 * 2**30 * 0.9
    assert filter_attention_cases([case], estimate, ns["torch"].cuda, "trtllm attention") == []
    assert "dropped 1/1 cases" in capsys.readouterr().out
    assert filter_attention_cases([case], estimate, collector_functions("trtllm", 268)["torch"].cuda, "attention") == [
        case
    ]
    assert filter_attention_cases([case], estimate, collector_functions("trtllm")["torch"].cuda, "attention") == [case]


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_memory_filter_is_applied_by_actual_getters(monkeypatch, backend):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    ns = collector_functions(backend, 80)
    for name in ("get_context_attention_test_cases", "get_generation_attention_test_cases"):
        cases = ns[name]()
        assert cases
        assert all(ns["_attention_memory_bytes"](c) <= 80 * 2**30 * 0.9 for c in cases)


def test_vllm_workspace_estimate_distinguishes_native_and_wrapper_paths():
    ns = collector_functions("vllm", 268)
    case = [16, 40960, 96, 1, 256, False, True, 0]
    native = ns["_attention_memory_bytes"](case)
    ns["use_trtllm_attention"] = lambda *_args: False
    wrapper = ns["_attention_memory_bytes"](case)
    assert wrapper - native == 16 * 40960 * 96 * 256 * 16
    assert native < 268 * 2**30 * 0.9 < wrapper


def test_vllm_large_batch_decode_uses_static_native_dispatch():
    ns = collector_functions("vllm", 268)
    # Runtime decode selection remains native above the old 256-token
    # preference threshold in use_trtllm_attention.
    ns["use_trtllm_attention"] = lambda *_args: pytest.fail("prefill selector used for decode")
    case = [512, 32767, 32, 1, 128, False, False, 0]
    native = ns["_attention_memory_bytes"](case)
    ns["can_use_trtllm_attention"] = lambda *_args, **_kwargs: False
    wrapper = ns["_attention_memory_bytes"](case)
    assert wrapper - native == 8192 * 32 * 128 * 16
