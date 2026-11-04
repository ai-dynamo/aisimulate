# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the TRT-LLM MoE worker-recycling policy.

``collector/trtllm/collect_moe.py`` imports torch and tensorrt_llm at module
scope, so the policy helpers and the ``run_moe_torch`` wrapper are AST-extracted
and executed against fakes (same technique as ``test_collect_moe_ep.py``).
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "trtllm" / "collect_moe.py"
NAMES = ("_cases_since_worker_start", "_worker_restart_policy", "_recycle_worker_if_due", "run_moe_torch")
RELEASE = "_release_trtllm_memory_buffers"
EXIT_CODE_RESTART = 10

CASE_ARGS = ("nvfp4", [1, 2, 4], 7168, 2048, 8, 256, 1, 8, False, "deepseek-ai/DeepSeek-V3")
CASE_KWARGS = {"distributed": "power_law", "power_law_alpha": 1.2, "perf_filename": "moe_perf.txt", "device": "cuda:3"}


def _load(*, allocated_mb: float = 0.0):
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))

    def wanted(node) -> bool:
        if isinstance(node, ast.FunctionDef):
            return node.name in NAMES
        if isinstance(node, ast.Assign):
            return any(isinstance(target, ast.Name) and target.id in NAMES for target in node.targets)
        return False

    selected = [node for node in tree.body if wanted(node)]
    assert len(selected) == len(NAMES)
    cases: list = []
    gc_calls: list = []
    namespace = {
        "os": os,
        "sys": sys,
        "EXIT_CODE_RESTART": EXIT_CODE_RESTART,
        "torch": SimpleNamespace(
            device=lambda value: f"device:{value}",
            cuda=SimpleNamespace(
                memory_allocated=lambda device: allocated_mb * 2**20,
                memory_reserved=lambda device: allocated_mb * 2**20,
            ),
        ),
        "_print_live_cuda_tensors": lambda device: None,
        "_release_trtllm_memory_buffers": lambda: 0,
        "gc_collect": lambda: gc_calls.append(True),
        "_run_moe_case": lambda *args, **kwargs: cases.append((args, kwargs)),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace, cases, gc_calls


def _run_cases(namespace, count: int) -> None:
    for _ in range(count):
        namespace["run_moe_torch"](*CASE_ARGS, **CASE_KWARGS)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("TRTLLM_MOE_RESTART_WORKER", "TRTLLM_MOE_RESTART_EVERY", "TRTLLM_MOE_RESTART_LEAK_MB"):
        monkeypatch.delenv(name, raising=False)


def test_default_policy_recycles_after_sixteen_cases(clean_env):
    namespace, cases, gc_calls = _load()
    _run_cases(namespace, 15)
    assert len(cases) == 15 and len(gc_calls) == 15
    with pytest.raises(SystemExit) as caught:
        _run_cases(namespace, 1)
    assert caught.value.code == EXIT_CODE_RESTART
    assert len(cases) == 16, "the 16th case must still run (and persist) before the worker exits"


def test_leak_guard_recycles_immediately_when_memory_stays_allocated(clean_env):
    namespace, cases, _ = _load(allocated_mb=2048)
    with pytest.raises(SystemExit) as caught:
        _run_cases(namespace, 1)
    assert caught.value.code == EXIT_CODE_RESTART
    assert len(cases) == 1


def test_zero_disables_the_count_and_the_guard_and_the_legacy_switch_disables_recycling(clean_env, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOE_RESTART_EVERY", "0")
    monkeypatch.setenv("TRTLLM_MOE_RESTART_LEAK_MB", "0")
    namespace, cases, gc_calls = _load(allocated_mb=8192)
    _run_cases(namespace, 50)
    assert len(cases) == 50 and len(gc_calls) == 50
    # the legacy master switch wins over any count and over the default leak guard
    monkeypatch.delenv("TRTLLM_MOE_RESTART_LEAK_MB")
    monkeypatch.setenv("TRTLLM_MOE_RESTART_WORKER", "0")
    monkeypatch.setenv("TRTLLM_MOE_RESTART_EVERY", "1")
    namespace, cases, _ = _load(allocated_mb=8192)
    _run_cases(namespace, 40)
    assert len(cases) == 40


def test_negative_values_are_rejected(clean_env, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOE_RESTART_EVERY", "-1")
    namespace, _, _ = _load()
    with pytest.raises(ValueError, match="must be >= 0"):
        _run_cases(namespace, 1)


def test_wrapper_forwards_the_case_unchanged_and_recycles_after_it(clean_env, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOE_RESTART_EVERY", "2")
    namespace, cases, gc_calls = _load()
    _run_cases(namespace, 1)
    assert cases == [(CASE_ARGS, CASE_KWARGS)]
    assert gc_calls == [True], "cleanup runs after the case body returned"


def test_release_drops_every_trtllm_workspace_block_including_reserved(monkeypatch):
    import types

    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == RELEASE)
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

    class _Tensor:
        def __init__(self, mib):
            self.mib = mib

        def numel(self):
            return self.mib * 2**20

        def element_size(self):
            return 1

    class _Block:
        def __init__(self, mib, reserved):
            self.buffer = _Tensor(mib)
            self.is_reserved = reserved

    blocks = {"moe_workspace": [_Block(7168, True), _Block(3584, False)], "other": [_Block(28, False)]}
    pool = SimpleNamespace(buffers=blocks)
    utils = types.ModuleType("tensorrt_llm._torch.memory_buffer_utils")
    utils.get_memory_buffers = lambda: pool
    pkg = types.ModuleType("tensorrt_llm._torch")
    pkg.memory_buffer_utils = utils
    monkeypatch.setitem(sys.modules, "tensorrt_llm", types.ModuleType("tensorrt_llm"))
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch", pkg)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.memory_buffer_utils", utils)
    assert namespace[RELEASE]() == 7168 + 3584 + 28
    assert pool.buffers == {}


class _TokenTensor:
    def __init__(self, tokens):
        self.tokens = tokens

    def bfloat16(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, index):
        assert isinstance(index, slice)
        return _TokenTensor(min(self.tokens, index.stop))


def _run_sweep(tokens, *, fail_stage=None, fail_tokens=None):
    """Run the actual case body with native allocation/timing calls replaced."""
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_moe_case")
    calls = []
    rows = []
    state = {"stage": "dry_run"}
    failure = RuntimeError("native shape failed")

    def forward(hidden, logits, **kwargs):
        assert hidden.tokens == logits.tokens
        calls.append((state["stage"], hidden.tokens))
        if state["stage"] == fail_stage and hidden.tokens == fail_tokens:
            raise failure

    @contextmanager
    def autotune():
        state["stage"] = "tuning"
        yield

    @contextmanager
    def benchmark(**kwargs):
        state["stage"] = "measurement"
        kwargs["kernel_func"]()
        yield {"latency_ms": calls[-1][1] / 1000, "power_stats": {}}

    device = SimpleNamespace(index=0)
    torch = MagicMock()
    torch.device.return_value = device
    torch.randn.side_effect = lambda shape: _TokenTensor(shape[0])
    torch.inference_mode = nullcontext
    moe = SimpleNamespace(forward=forward, to=lambda device: None)
    builder = MagicMock(return_value=moe)
    namespace = {
        "torch": torch,
        "gc_collect": lambda: None,
        "aic_debug": 0,
        "QuantConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "Mapping": SimpleNamespace,
        "SimpleNamespace": SimpleNamespace,
        "ModelConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "NON_GATED_MOE_MODELS": [],
        "ActivationType": None,
        "get_sm_version": lambda: 100,
        "_MXFP4_MOE_TYPES": {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"},
        "RenormalizeMoeRoutingMethod": lambda topk: None,
        "_declared_moe_routing": lambda model: None,
        "create_moe": builder,
        "inspect": inspect,
        "balanced_logits": lambda count, experts, topk: _TokenTensor(count),
        "cleanup_empty_json_files": lambda path: None,
        "moe_tune_path": "unused",
        "glob": SimpleNamespace(glob=lambda pattern: []),
        "autotune": autotune,
        "benchmark_with_power": benchmark,
        "log_perf": lambda **kwargs: rows.extend(kwargs["item_list"]),
        "tensorrt_llm": SimpleNamespace(__version__="test"),
    }
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

    def run():
        namespace["_run_moe_case"]("bfloat16", tokens, 4096, 2048, 2, 8, 2, 2, False, "model", perf_filename="unused")

    return run, calls, rows, builder, failure


def test_moe_sweep_measures_every_requested_shape_with_unchanged_identity():
    # Unsorted input also exercises native capacity: the last item is not max.
    tokens = [2, 65536, 32768]
    run, calls, rows, builder, _ = _run_sweep(tokens)
    run()

    assert builder.call_args.kwargs["model_config"].moe_max_num_tokens == 65536
    assert calls == [("dry_run", 65536)] + [("tuning", n) for n in reversed(tokens)] + [
        ("measurement", n) for n in tokens
    ]
    assert [row["num_tokens"] for row in rows] == tokens
    assert [row["latency"] for row in rows] == [n / 1000 for n in tokens]
    assert {(row["moe_tp_size"], row["moe_ep_size"], row["moe_dtype"], row["distribution"]) for row in rows} == {
        (2, 2, "bfloat16", "balanced")
    }


@pytest.mark.parametrize("stage", ["dry_run", "tuning", "measurement"])
def test_moe_sweep_propagates_large_shape_failure_without_smaller_substitution(stage):
    run, calls, rows, _, failure = _run_sweep([2, 32768, 65536], fail_stage=stage, fail_tokens=65536)
    with pytest.raises(RuntimeError, match="native shape failed") as caught:
        run()
    assert caught.value is failure
    assert calls[-1] == (stage, 65536)
    # Measurement failures preserve only genuinely measured earlier rows.
    assert [row["num_tokens"] for row in rows] == ([2, 32768] if stage == "measurement" else [])
