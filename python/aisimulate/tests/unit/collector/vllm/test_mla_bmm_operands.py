# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM MLA BMM rows time torch.bmm on serving's operand layouts (no CUDA needed).

Serving @v0.32.0: BMM1 reads q_nope as the nope slice of q viewed (B, N, P + R) and
writes through a transposed view of a token-major (B, N, L) buffer
(model_executor/layers/attention/mla_attention.py:1022-1027,1087-1088); the W_UV
projection reads the (B, N, L) attention output transposed and writes through a
transposed (B, N, V) view (:1383-1384,1413).
"""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/vllm/collect_mla_bmm.py"


class _Tensor:
    def __init__(self, name, shape, view=()):
        self.name, self.shape, self.view = name, tuple(shape), tuple(view)

    def __getitem__(self, index):
        last = index[-1]
        return _Tensor(self.name, (*self.shape[:-1], last.stop), (*self.view, ("slice", last.stop)))

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return _Tensor(self.name, shape, (*self.view, ("transpose", a, b)))


def _run(name, num_tokens=3, num_heads=16):
    bmm_calls = []

    def allocate(kind):
        def make(shape, **_kwargs):
            return _Tensor(f"{kind}{len(shape)}:{shape}", shape)

        return make

    @contextmanager
    def benchmark(**kwargs):
        kwargs["kernel_func"]()
        yield {"latency_ms": 0.01, "power_stats": None}

    torch = SimpleNamespace(
        bfloat16="bf16",
        randn=allocate("randn"),
        empty=allocate("empty"),
        bmm=lambda a, b, *, out: bmm_calls.append((a, b, out)),
        cuda=SimpleNamespace(set_device=lambda device: None),
        set_default_device=lambda device: None,
    )
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "benchmark_with_power": benchmark, "_log_row": lambda **kwargs: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    namespace[name](num_tokens, num_heads, "bfloat16", 1, 1, perf_filename="mla_bmm_perf.txt")
    assert len(bmm_calls) == 1
    return bmm_calls[0]


def test_absorb_bmm_reads_the_q_slice_and_writes_a_token_major_buffer():
    q_nope, w_uk_t, out = _run("run_mla_gen_pre")
    assert q_nope.shape == (16, 3, 128)
    assert q_nope.view == (("slice", 128), ("transpose", 0, 1))
    assert q_nope.name.startswith("randn3:(3, 16, 192)")
    assert w_uk_t.shape == (16, 128, 512) and w_uk_t.view == ()
    assert out.shape == (16, 3, 512)
    assert out.view == (("transpose", 0, 1),) and out.name.startswith("empty3:(3, 16, 512)")


def test_w_uv_bmm_reads_and_writes_transposed_token_major_views():
    x, w_uv, out = _run("run_mla_gen_post")
    assert x.shape == (16, 3, 512) and x.view == (("transpose", 0, 1),)
    assert w_uv.shape == (16, 512, 128) and w_uv.view == ()
    assert out.shape == (16, 3, 128) and out.view == (("transpose", 0, 1),)
