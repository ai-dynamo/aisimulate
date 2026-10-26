# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute collector setup/timer contracts on CPU; no native arithmetic claim."""

import ast
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from collector.version_resolver import _check_compat

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/vllm/collect_kda.py"


class Tensor(np.ndarray):
    def view(self, *shape):
        return self.reshape(*shape)

    def transpose(self, first, second):
        return np.swapaxes(self, first, second)

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)

    def split(self, size, dim):
        return np.split(self, range(size, self.shape[dim], size), axis=dim)

    def element_size(self):
        return self.dtype.itemsize


def tensor(value, dtype):
    return np.asarray(value, dtype=dtype).view(Tensor)


def setup(monkeypatch, *, head=12, fail_recurrence=False, selected="triton"):
    records = SimpleNamespace(conv=[], core=[], rows=[], timers=[], bind=[], resolve=[], shapes=[], gates=[])

    def array(*shape, dtype, device=None):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return tensor(np.ones(shape), dtype)

    torch = SimpleNamespace(
        device=lambda x: x,
        cuda=SimpleNamespace(set_device=lambda _: None),
        bfloat16=np.float16,
        float32=np.float32,
        int32=np.int32,
        int8=np.int8,
        randn=array,
        empty=array,
        empty_like=lambda x: tensor(np.ones(x.shape), x.dtype),
        zeros=lambda *shape, **kw: array(*shape, **kw) * 0,
        arange=lambda *args, dtype, device: tensor(np.arange(*args), dtype),
        full=lambda shape, fill_value, dtype, device: tensor(np.full(shape, fill_value), dtype),
    )

    def shapes(**kwargs):
        records.shapes.append(kwargs)
        return ((kwargs["conv_kernel_size"] - 1 + kwargs["num_spec"], 3 * head * 128), (head, 128, 128))

    def bind(holder, pages):
        records.bind.append(pages.shape)
        records.pages = pages
        offset, states = 0, []
        for shape, dtype in zip(holder.get_state_shape(), holder.get_state_dtype(), strict=True):
            size = math.prod(shape) * np.dtype(dtype).itemsize
            raw = np.asarray(pages[:, 0, 0, offset : offset + size])
            states.append(raw.view(dtype).reshape(-1, *shape).view(Tensor))
            offset += size
        holder.kv_cache = tuple(states)

    def resolve(*args):
        records.resolve.append(args)
        return selected

    def conv(x, state, weight, bias, **kwargs):
        records.conv.append((x, state, kwargs))
        for i, part in enumerate(kwargs["out"].split(head * 128, -1)):
            part[...] = i + 1

    def core(**kwargs):
        records.core.append(kwargs)
        if fail_recurrence:
            raise RuntimeError("observed native failure")

    modules = {
        "vllm.model_executor.layers.mamba.abstract": {"MambaBase": SimpleNamespace(bind_kv_cache=bind)},
        "vllm.model_executor.layers.mamba.mamba_utils": {
            "MambaStateDtypeCalculator": SimpleNamespace(kda_state_dtype=lambda *a: (np.float16, np.float32)),
            "MambaStateShapeCalculator": SimpleNamespace(kda_state_shape=shapes),
        },
        "vllm.models.kimi_k3.nvidia.kda": {"resolve_kda_decode_backend": resolve},
        "vllm.model_executor.layers.mamba.ops.causal_conv1d": {"causal_conv1d_update": conv},
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": {"fused_recurrent_kda": core},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    @contextmanager
    def benchmark(**kwargs):
        records.timers.append({k: v for k, v in kwargs.items() if k != "kernel_func"})
        kwargs["kernel_func"]()
        yield {"latency_ms": 0.01, "power_stats": None}

    namespace = {
        "torch": torch,
        "math": math,
        "SimpleNamespace": SimpleNamespace,
        "_check_compat": _check_compat,
        "KdaRuntimeNotAuditedError": ValueError,
        "_require_audited_runtime": lambda *a, **kw: records.gates.append((a, kw)),
        "benchmark_with_power": benchmark,
        "NUM_WARMUPS": 3,
        "NUM_RUNS": 10,
        "KDA_LOWER_BOUND": -5,
        "_cleanup": lambda _: None,
        "_format_failures": lambda rows: ";".join(rows),
        "_log": lambda common, latency, kernel, *a: records.rows.append((dict(common), kernel)),
    }
    names = {"_kimi031_verify_state_pool", "run_kda_verify_benchmark"}
    body = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace, records


def invoke(ns, head=12, width=2, version="0.31.0"):
    ns["run_kda_verify_benchmark"](
        7168, 4, head, 128, head, 128, [2], [width], "moonshotai/Kimi-K3", "unused.txt", version
    )


@pytest.mark.parametrize("head", [12, 24, 48, 96])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_target_native_inputs_and_original_timing_boundaries(monkeypatch, head, width):
    ns, r = setup(monkeypatch, head=head)
    invoke(ns, head, width)
    x, conv_state, conv = r.conv[0]
    core = r.core[0]
    dim = head * 128
    projection_width = ((4 * dim + 128 + head + 15) // 16) * 16
    assert r.resolve == [("auto", head, 128, 4, width - 1, np.float16, np.float16, np.float32)]
    assert r.shapes[0]["num_spec"] == width - 1
    assert conv_state.shape == (2 * width + 1, 3 * dim, width + 2)
    assert core["initial_state"].shape == (2 * width + 1, head, 128, 128)
    assert conv_state.strides[0] == core["initial_state"].strides[0] == r.bind[0][-1]
    assert np.shares_memory(conv_state, r.pages)
    assert np.shares_memory(core["initial_state"], r.pages)
    np.testing.assert_array_equal(core["ssm_state_indices"], np.arange(1, 2 * width + 1).reshape(2, width))
    np.testing.assert_array_equal(conv["conv_state_indices"], [1, 1 + width])
    np.testing.assert_array_equal(core["num_accepted_tokens"], [width, width])
    np.testing.assert_array_equal(core["cu_seqlens"], [0, width, 2 * width])
    assert x.strides[0] // 2 == projection_width
    assert core["raw_beta"].strides[1] // 2 == projection_width
    for i, name in enumerate(("q", "k", "v")):
        assert core[name].strides[1] // 2 == 3 * dim
        assert np.shares_memory(core[name], conv["out"])
        np.testing.assert_array_equal(core[name], i + 1)
    assert core["out"].shape == (1, 2 * width, head, 128)
    assert [kernel for _, kernel in r.rows] == ["causal_conv1d_update", "fused_recurrent_kda"]
    assert (
        r.rows[0][0]
        == r.rows[1][0]
        == {
            "phase": "verify",
            "batch_size": 2,
            "seq_len": width,
            "num_tokens": 2 * width,
            "d_model": 7168,
            "d_conv": 4,
            "num_k_heads": head,
            "head_k_dim": 128,
            "num_v_heads": head,
            "head_v_dim": 128,
            "model_name": "moonshotai/Kimi-K3",
        }
    )
    assert r.timers == [{"device": "cuda:0", "num_warmups": 3, "num_runs": 10, "repeat_n": 1}] * 2


def test_preview_fixture_does_not_change(monkeypatch):
    ns, r = setup(monkeypatch)
    invoke(ns, version="0.1.dev19262")
    assert not r.resolve and not r.bind and not r.gates
    x, state, conv = r.conv[0]
    core = r.core[0]
    assert state.shape == (2, 4608, 5)
    np.testing.assert_array_equal(conv["conv_state_indices"], [0, 1])
    np.testing.assert_array_equal(core["ssm_state_indices"], [[0, 1], [2, 3]])
    assert not np.shares_memory(core["q"], conv["out"])
    assert x.strides[0] // 2 == 4608


def test_observed_native_failure_remains_incomplete(monkeypatch):
    ns, r = setup(monkeypatch, fail_recurrence=True)
    with pytest.raises(RuntimeError, match="batch_size=2 draft_tokens=2: RuntimeError: observed native failure"):
        invoke(ns)
    assert [kernel for _, kernel in r.rows] == ["causal_conv1d_update"]


def test_changed_native_selection_raises_without_fallback(monkeypatch):
    ns, r = setup(monkeypatch, selected="native")
    with pytest.raises(RuntimeError, match="expected native auto Triton dispatch"):
        invoke(ns)
    assert not r.rows and not r.conv and not r.core
