# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FlashInfer workspace floor holds for the whole worker process.

sglang v0.5.23 allocates the shared workspace once per process under
``get_buffer("flashinfer_workspace")`` (flashinfer_backend.py:456-463), sized from
``envs.SGLANG_FLASHINFER_WORKSPACE_SIZE``, which the backend constructor itself resets to
512 MiB for Qwen2/Qwen3/MiMo/Qwen3-VL architectures (:421-431). The collector therefore
raises the knob AND creates the named buffer at the floor before the first backend."""

import ast
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_attn.py"


class _EnvInt:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


class _Buffers:
    """runtime_context.get_buffer: the first factory call wins for the process."""

    def __init__(self):
        self.store = {}
        self.factory_calls = 0

    def get_buffer(self, name, factory):
        if name not in self.store:
            self.factory_calls += 1
            self.store[name] = factory()
        return self.store[name]


class _Tensor:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


def _load(monkeypatch, knob, buffers):
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
    fake_torch = types.SimpleNamespace(uint8="uint8", empty=lambda n, dtype, device: _Tensor(n))
    ns = {"torch": fake_torch, "get_buffer": buffers.get_buffer}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(__import__("sys").modules, name, types.ModuleType(name))
    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = types.SimpleNamespace(SGLANG_FLASHINFER_WORKSPACE_SIZE=knob)
    monkeypatch.setitem(__import__("sys").modules, "sglang.srt.environ", environ)
    return ns["_ensure_flashinfer_workspace"], ns["_FLASHINFER_WORKSPACE_MIN_BYTES"]


@pytest.mark.parametrize(("initial", "expected"), [(384 * 1024 * 1024, 4 << 30), (8 << 30, 8 << 30)])
def test_knob_and_shared_buffer_take_the_floor_and_a_larger_setting_is_kept(monkeypatch, initial, expected):
    knob, buffers = _EnvInt(initial), _Buffers()
    fn, floor = _load(monkeypatch, knob, buffers)
    assert floor == 4 << 30
    assert fn("cuda:0") == expected
    assert knob.get() == expected
    assert buffers.store["flashinfer_workspace"].numel() == expected


def test_a_backend_lowering_the_knob_later_does_not_shrink_the_process_workspace(monkeypatch):
    """A Qwen3/MiMo row resets the knob to 512 MiB inside FlashInferAttnBackend.__init__;
    the buffer the backend then asks for is the one the collector already created."""
    knob, buffers = _EnvInt(384 * 1024 * 1024), _Buffers()
    fn, floor = _load(monkeypatch, knob, buffers)
    fn("cuda:0")
    knob.set(512 * 1024 * 1024)  # flashinfer_backend.py:421-431
    workspace = buffers.get_buffer("flashinfer_workspace", lambda: _Tensor(knob.get()))
    assert workspace.numel() == floor
    assert buffers.factory_calls == 1
    # The next case raises the knob again for TRTLLMHAAttnBackend, which reads it directly.
    assert fn("cuda:0") == floor and knob.get() == floor
