# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the TRT-LLM MoE worker-recycling policy.

``collector/trtllm/collect_moe.py`` imports torch and tensorrt_llm at module
scope, so the policy helpers and the ``run_moe_torch`` wrapper are AST-extracted
and executed against fakes (same technique as ``test_collect_moe_ep.py``).
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_default_policy_recycles_after_sixteen_cases(clean_env, capsys):
    namespace, cases, gc_calls = _load()
    _run_cases(namespace, 15)
    assert len(cases) == 15 and len(gc_calls) == 15
    with pytest.raises(SystemExit) as caught:
        _run_cases(namespace, 1)
    assert caught.value.code == EXIT_CODE_RESTART
    assert len(cases) == 16, "the 16th case must still run (and persist) before the worker exits"
    out = capsys.readouterr().out
    assert "trtllm.moe: recycling worker after 16 case(s)" in out
    assert "every 16" in out
    assert out.count("MiB allocated, ") == 16, "one memory line per case, recycle or not"


def test_leak_guard_recycles_immediately_when_memory_stays_allocated(clean_env, capsys):
    namespace, cases, _ = _load(allocated_mb=2048)
    with pytest.raises(SystemExit) as caught:
        _run_cases(namespace, 1)
    assert caught.value.code == EXIT_CODE_RESTART
    assert len(cases) == 1
    out = capsys.readouterr().out
    assert "2048 MiB still allocated, leak guard" in out


def test_every_one_restores_the_legacy_per_case_recycle(clean_env, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOE_RESTART_EVERY", "1")
    namespace, cases, _ = _load()
    with pytest.raises(SystemExit) as caught:
        _run_cases(namespace, 1)
    assert caught.value.code == EXIT_CODE_RESTART and len(cases) == 1


def test_legacy_switch_disables_recycling(clean_env, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOE_RESTART_WORKER", "0")
    monkeypatch.setenv("TRTLLM_MOE_RESTART_EVERY", "1")
    namespace, cases, _ = _load(allocated_mb=8192)
    _run_cases(namespace, 40)
    assert len(cases) == 40


def test_zero_disables_the_count_and_the_guard(clean_env, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOE_RESTART_EVERY", "0")
    monkeypatch.setenv("TRTLLM_MOE_RESTART_LEAK_MB", "0")
    namespace, cases, gc_calls = _load(allocated_mb=8192)
    _run_cases(namespace, 50)
    assert len(cases) == 50 and len(gc_calls) == 50


@pytest.mark.parametrize("name", ["TRTLLM_MOE_RESTART_EVERY", "TRTLLM_MOE_RESTART_LEAK_MB"])
def test_negative_values_are_rejected(clean_env, monkeypatch, name):
    monkeypatch.setenv(name, "-1")
    namespace, _, _ = _load()
    with pytest.raises(ValueError, match="must be >= 0"):
        _run_cases(namespace, 1)


def test_wrapper_forwards_the_case_unchanged_and_recycles_after_it(clean_env, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOE_RESTART_EVERY", "2")
    namespace, cases, gc_calls = _load()
    _run_cases(namespace, 1)
    assert cases == [(CASE_ARGS, CASE_KWARGS)]
    assert gc_calls == [True], "cleanup runs after the case body returned"


def test_case_body_no_longer_exits_on_its_own():
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_moe_case")
    exits = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
    ]
    assert exits == [], "worker recycling belongs to run_moe_torch's policy, not the case body"
    assert "TRTLLM_MOE_RESTART_WORKER" not in ast.get_source_segment(SOURCE_PATH.read_text(), body)


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


def test_release_is_a_no_op_without_the_trtllm_module(monkeypatch):
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == RELEASE)
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    monkeypatch.setitem(sys.modules, "tensorrt_llm", None)
    assert namespace[RELEASE]() == 0
