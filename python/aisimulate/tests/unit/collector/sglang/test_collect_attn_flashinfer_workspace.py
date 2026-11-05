# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FlashInfer workspace floor goes through the knob sglang 0.5.21 reads
(envs.SGLANG_FLASHINFER_WORKSPACE_SIZE), not the retired global_config attribute
(ed9109ec wrote the latter and changed nothing: l40s pipeline 72343559)."""

import ast
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_attn.py"


def _load():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_ensure_flashinfer_workspace")
        or (
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "_FLASHINFER_WORKSPACE_MIN_BYTES" for t in n.targets)
        )
    ]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["_ensure_flashinfer_workspace"], ns["_FLASHINFER_WORKSPACE_MIN_BYTES"]


class _EnvInt:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


@pytest.mark.parametrize(("initial", "expected"), [(384 * 1024 * 1024, 4 << 30), (8 << 30, 8 << 30)])
def test_0521_envs_knob_is_raised_to_the_floor_and_a_larger_setting_is_kept(monkeypatch, initial, expected):
    fn, floor = _load()
    assert floor == 4 << 30
    knob = _EnvInt(initial)
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = types.SimpleNamespace(SGLANG_FLASHINFER_WORKSPACE_SIZE=knob)
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", environ)
    assert fn() == expected and knob.get() == expected
