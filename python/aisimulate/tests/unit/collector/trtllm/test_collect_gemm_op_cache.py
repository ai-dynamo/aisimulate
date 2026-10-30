# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the TRT-LLM GEMM collector's per-(gemm_type, n, k) module cache.

``collector/trtllm/collect_gemm.py`` imports torch and tensorrt_llm at module
scope, so ``_gemm_ops`` / ``run_gemm`` are AST-extracted and executed against
fakes (same technique as ``test_collect_moe_ep.py``).
"""

from __future__ import annotations

import ast
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "trtllm" / "collect_gemm.py"
NAMES = ("_gemm_op_cache", "_gemm_ops", "run_gemm")
L2_BYTES = 126 * 2**20


class _Tensor:
    def __init__(self, shape):
        self.shape = shape

    def to(self, *args, **kwargs):
        return self


class _Linear:
    instances: ClassVar[list] = []

    def __init__(self, in_features, out_features, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.loaded = []
        self.post_loaded = 0
        self.forwards: list = []
        _Linear.instances.append(self)

    def load_weights(self, weights):
        self.loaded.append(weights)

    def post_load_weights(self):
        self.post_loaded += 1

    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        self.forwards.append(x)


def _load():
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))

    def wanted(node) -> bool:
        if isinstance(node, ast.FunctionDef):
            return node.name in NAMES
        if isinstance(node, ast.AnnAssign):
            return isinstance(node.target, ast.Name) and node.target.id in NAMES
        return False

    selected = [node for node in tree.body if wanted(node)]
    assert len(selected) == len(NAMES)
    _Linear.instances = []
    built: list = []
    empty_cache_calls: list = []
    benchmarks: list = []
    published: list = []

    @contextmanager
    def benchmark_with_power(*, device, kernel_func, repeat_n):
        benchmarks.append({"forwards_before": [len(op.forwards) for op in _Linear.instances], "repeat_n": repeat_n})
        kernel_func()
        yield {"latency_ms": 1.0, "power_stats": None}

    namespace = {
        "math": math,
        "torch": SimpleNamespace(
            device=lambda value: value,
            bfloat16="bf16",
            randn=lambda shape, dtype: _Tensor(shape),
            set_default_device=lambda device: None,
            cuda=SimpleNamespace(
                set_device=lambda device: None,
                empty_cache=lambda: empty_cache_calls.append(True),
                get_device_name=lambda device: "B200",
            ),
        ),
        "Linear": _Linear,
        "_build_weights": lambda gemm_type, n, k, device, dtype, group_size, x: (
            built.append((gemm_type, n, k, group_size)) or {"weight": (n, k)}
        ),
        "_get_l2_cache_bytes": lambda index: L2_BYTES,
        "get_sm_version": lambda: 100,
        "is_sm_100f": lambda sm: sm in (100, 103),
        "QuantConfig": lambda **kwargs: kwargs,
        "QuantAlgo": SimpleNamespace(FP8="fp8", FP8_BLOCK_SCALES="fp8_block", NVFP4="nvfp4"),
        "benchmark_with_power": benchmark_with_power,
        "log_perf": lambda **kwargs: published.append(kwargs),
        "tensorrt_llm": SimpleNamespace(__version__="1.3.0rc29"),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace, built, empty_cache_calls, benchmarks, published


def _ops(namespace, gemm_type, n, k, *, copies):
    return namespace["_gemm_ops"](
        gemm_type,
        n,
        k,
        quant_config=None,
        group_size=None,
        outside_loop_count=copies,
        device=SimpleNamespace(index=0),
        dtype="bf16",
        x=_Tensor((1, k)),
    )


def test_same_key_reuses_the_built_modules_and_a_new_key_replaces_them():
    namespace, built, empty_cache_calls, _, _ = _load()
    first = _ops(namespace, "nvfp4", 4096, 4096, copies=5)
    assert len(first) == 5 and len(built) == 5 and len(_Linear.instances) == 5
    assert empty_cache_calls == [True], "the previous key's modules are dropped before building"
    assert _ops(namespace, "nvfp4", 4096, 4096, copies=5) is first
    assert len(built) == 5, "a second m for the same (gemm_type, n, k) builds nothing"
    second = _ops(namespace, "nvfp4", 8192, 4096, copies=2)
    assert second is not first and len(second) == 2
    assert len(built) == 7 and len(empty_cache_calls) == 2
    assert list(namespace["_gemm_op_cache"]) == [("nvfp4", 8192, 4096, 2)], "one key is held at a time"


def test_run_gemm_builds_once_per_key_dry_runs_every_copy_and_divides_the_latency():
    namespace, built, _, benchmarks, published = _load()
    n = k = 1024  # 2 MiB bf16 weight -> 5 copies to span the 126 MiB L2
    for m in (8192, 4096, 1):
        namespace["run_gemm"]("bfloat16", m, n, k, perf_filename="gemm_perf.txt", device="cuda:0")
    assert len(built) == 5 and len(_Linear.instances) == 5
    expected_copies = min(5, math.ceil(L2_BYTES / (n * k * 2)))
    assert expected_copies == 5
    # Before each capture every cached copy saw its dry run at the current m (JIT / lazy init).
    assert [b["forwards_before"] for b in benchmarks] == [[1] * 5, [3] * 5, [5] * 5]
    assert all(b["repeat_n"] == 1 for b in benchmarks)
    assert [row["item_list"][0]["m"] for row in published] == [8192, 4096, 1]
    assert all(row["item_list"][0]["latency"] == pytest.approx(1.0 / 5) for row in published)
    assert all(row["kernel_source"] == "torch_flow" for row in published)


def test_fp8_block_post_load_runs_once_per_copy_and_is_tagged_deepgemm_on_sm100():
    namespace, built, _, _, published = _load()
    n, k = 16384, 7168  # 112 MiB fp8 weight -> 2 copies
    namespace["run_gemm"]("fp8_block", 64, n, k, perf_filename="gemm_perf.txt", device="cuda:0")
    namespace["run_gemm"]("fp8_block", 32, n, k, perf_filename="gemm_perf.txt", device="cuda:0")
    assert built == [("fp8_block", n, k, 128)] * 2
    assert [op.post_loaded for op in _Linear.instances] == [1, 1]
    expected_quant = {"quant_algo": "fp8_block", "group_size": 128}
    assert [op.kwargs["quant_config"] for op in _Linear.instances] == [expected_quant] * 2
    assert [row["kernel_source"] for row in published] == ["deepgemm", "deepgemm"]
    assert all(row["item_list"][0]["latency"] == pytest.approx(0.5) for row in published)


def test_fp8_alignment_limit_still_raises_before_any_module_is_built():
    namespace, built, _, _, _ = _load()
    with pytest.raises(RuntimeError, match="FIXME\\(kernel-limit\\)"):
        namespace["run_gemm"]("fp8", 16, 24, 4096, perf_filename="gemm_perf.txt", device="cuda:0")
    assert built == [] and _Linear.instances == []
