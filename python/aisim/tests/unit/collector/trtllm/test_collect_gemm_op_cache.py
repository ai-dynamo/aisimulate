# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the TRT-LLM GEMM collector's per-(gemm_type, n, k) module cache,
its serving-warmup autotune pass and the nvfp4 kernel_source lookup.

``collector/trtllm/collect_gemm.py`` imports torch and tensorrt_llm at module
scope, so ``_gemm_ops`` / ``run_gemm`` and their helpers are AST-extracted and
executed against fakes (same technique as ``test_collect_moe_ep.py``).
"""

from __future__ import annotations

import ast
import math
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "trtllm" / "collect_gemm.py"
NAMES = (
    "_gemm_op_cache",
    "_SERVING_WARMUP_NUM_TOKENS",
    "_gemm_tuned_num_tokens",
    "_NVFP4_GEMM_OP",
    "_autotune",
    "_nvfp4_kernel_source",
    "_gemm_ops",
    "run_gemm",
)
L2_BYTES = 126 * 2**20
NVFP4_GEMM_OP = "trtllm::nvfp4_gemm::gemm"


class _Tensor:
    def __init__(self, shape):
        self.shape = shape

    def to(self, *args, **kwargs):
        return self


class _FakeAutoTuner:
    """Stands in for TRT-LLM's AutoTuner singleton and its ``autotune()`` context.

    ``tactics`` maps an nvfp4 case's M to the cached (backend, sub_tactic) a
    lookup returns; an M missing from it is a cache miss. Forwards run under
    ``autotune()`` are recorded per pass as (copy index, M); under ``capture()``
    each nvfp4 forward records the nvfp4_gemm call the real dispatch would.
    """

    def __init__(self, tactics):
        self.tactics = tactics
        self.passes: list = []
        self._pass = None
        self._capture = None
        self.profiling_cache = SimpleNamespace(search_cache=self._search_cache)

    @contextmanager
    def autotune(self):
        self._pass = []
        try:
            yield
        finally:
            self.passes.append(self._pass)
            self._pass = None

    @contextmanager
    def capture(self):
        self._capture = SimpleNamespace(_captured_contexts=[])
        try:
            yield self._capture
        finally:
            self._capture = None

    def forward(self, linear, x) -> bool:
        """Record one Linear forward; True when it ran under ``autotune()``."""
        if self._pass is not None:
            self._pass.append((_Linear.instances.index(linear), x.shape[0]))
            return True
        quant = linear.kwargs["quant_config"]
        if self._capture is not None and quant and quant["quant_algo"] == "nvfp4":
            self._capture._captured_contexts.append(
                {"custom_op": "trtllm::fp4_quantize", "runners": [], "tuning_config": None, "inputs": [x]}
            )
            self._capture._captured_contexts.append(
                {
                    "custom_op": NVFP4_GEMM_OP,
                    "runners": ["nvfp4 runner"],
                    "tuning_config": "nvfp4 config",
                    "inputs": [x],
                }
            )
        return False

    def _get_input_sizes(self, inputs):
        return [t.shape for t in inputs]

    def _search_cache(self, custom_op, runners, sizes, tuning_config):
        assert (custom_op, runners, tuning_config) == (NVFP4_GEMM_OP, ["nvfp4 runner"], "nvfp4 config")
        tactic = self.tactics.get(sizes[0][0])
        return (False, 0, -1, float("inf")) if tactic is None else (True, 0, tactic, 0.1)


class _Linear:
    instances: ClassVar[list] = []
    tuner: ClassVar[_FakeAutoTuner | None] = None

    def __init__(self, in_features, out_features, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.loaded = []
        self.post_loaded = 0
        self.forwards: list = []  # forwards outside autotune()
        _Linear.instances.append(self)

    def load_weights(self, weights):
        self.loaded.append(weights)

    def post_load_weights(self):
        self.post_loaded += 1

    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        if not _Linear.tuner.forward(self, x):
            self.forwards.append(x)


def _load(tactics=None):
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
    tuner = _FakeAutoTuner(tactics or {})
    _Linear.tuner = tuner
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
        "time": time,
        "torch": SimpleNamespace(
            device=lambda value: value,
            bfloat16="bf16",
            randn=lambda shape, dtype, device=None: _Tensor(shape),
            set_default_device=lambda device: None,
            cuda=SimpleNamespace(
                set_device=lambda device: None,
                empty_cache=lambda: empty_cache_calls.append(True),
                get_device_name=lambda device: "B200",
                synchronize=lambda: None,
            ),
        ),
        "AutoTuner": SimpleNamespace(get=lambda: tuner),
        "autotune": tuner.autotune,
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
        "tensorrt_llm": SimpleNamespace(__version__="1.4.0rc2"),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace, built, empty_cache_calls, benchmarks, published, tuner


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


def _run(namespace, gemm_type, m, n, k):
    namespace["run_gemm"](gemm_type, m, n, k, perf_filename="gemm_perf.txt", device="cuda:0")


def test_same_key_reuses_the_built_modules_and_a_new_key_replaces_them():
    namespace, built, empty_cache_calls, _, _, tuner = _load()
    first = _ops(namespace, "nvfp4", 4096, 4096, copies=5)
    assert len(first) == 5 and len(built) == 5 and len(_Linear.instances) == 5
    assert empty_cache_calls == [True], "the previous key's modules are dropped before building"
    assert _ops(namespace, "nvfp4", 4096, 4096, copies=5) is first
    assert len(built) == 5, "a second m for the same (gemm_type, n, k) builds nothing"
    second = _ops(namespace, "nvfp4", 8192, 4096, copies=2)
    assert second is not first and len(second) == 2
    assert len(built) == 7 and len(empty_cache_calls) == 2
    assert list(namespace["_gemm_op_cache"]) == [("nvfp4", 8192, 4096, 2)], "one key is held at a time"
    # Each new key's first copy runs serving's warmup pass once: max_num_tokens, then one token.
    assert tuner.passes == [[(0, 8192), (0, 1)], [(5, 8192), (5, 1)]]


def test_run_gemm_builds_and_tunes_once_per_key_dry_runs_every_copy_and_divides_the_latency(capsys):
    namespace, built, _, benchmarks, published, tuner = _load()
    n = k = 1024  # 2 MiB bf16 weight -> 5 copies to span the 126 MiB L2
    for m in (8192, 4096, 1):
        _run(namespace, "bfloat16", m, n, k)
    assert len(built) == 5 and len(_Linear.instances) == 5
    # Serving post-loads every module, not only fp8_block ones.
    assert [op.post_loaded for op in _Linear.instances] == [1] * 5
    assert tuner.passes == [[(0, 8192), (0, 1)]], "one tuning pass per key, not per case"
    log = capsys.readouterr().out.splitlines()
    assert len(log) == 1 and log[0].startswith("TRT-LLM autotune gemm bfloat16 n=1024 k=1024 num_tokens=[8192, 1]: ")
    assert log[0].endswith(" s")
    expected_copies = min(5, math.ceil(L2_BYTES / (n * k * 2)))
    assert expected_copies == 5
    # Before each capture every cached copy saw its dry run at the current m (JIT / lazy init).
    assert [b["forwards_before"] for b in benchmarks] == [[1] * 5, [3] * 5, [5] * 5]
    assert all(b["repeat_n"] == 1 for b in benchmarks)
    assert [row["item_list"][0]["m"] for row in published] == [8192, 4096, 1]
    assert all(row["item_list"][0]["latency"] == pytest.approx(1.0 / 5) for row in published)
    assert all(row["kernel_source"] == "torch_flow" for row in published)


def test_a_case_above_the_warmup_m_first_tunes_once_at_its_own_m():
    namespace, _, _, _, published, tuner = _load()
    for m in (32768, 16384, 8192, 1):
        _run(namespace, "bfloat16", m, 4096, 4096)
    assert tuner.passes == [[(0, 8192), (0, 1)], [(0, 32768)]]
    assert [row["item_list"][0]["m"] for row in published] == [32768, 16384, 8192, 1]


def test_fp8_block_post_load_runs_once_per_copy_and_is_tagged_deepgemm_on_sm100():
    namespace, built, _, _, published, _ = _load()
    n, k = 16384, 7168  # 112 MiB fp8 weight -> 2 copies
    _run(namespace, "fp8_block", 64, n, k)
    _run(namespace, "fp8_block", 32, n, k)
    assert built == [("fp8_block", n, k, 128)] * 2
    assert [op.post_loaded for op in _Linear.instances] == [1, 1]
    expected_quant = {"quant_algo": "fp8_block", "group_size": 128}
    assert [op.kwargs["quant_config"] for op in _Linear.instances] == [expected_quant] * 2
    assert [row["kernel_source"] for row in published] == ["deepgemm", "deepgemm"]
    assert all(row["item_list"][0]["latency"] == pytest.approx(0.5) for row in published)


def test_fp8_block_with_k_off_the_128_blocks_is_refused_before_the_build_on_sm90_only():
    namespace, built, _, _, published, _ = _load()
    namespace["get_sm_version"] = lambda: 90
    with pytest.raises(ValueError, match=r"^FIXME\(kernel-limit\): fp8_block GEMM on SM90"):
        _run(namespace, "fp8_block", 64, 4096, 1856)
    assert built == [] and published == []
    _run(namespace, "fp8_block", 64, 4096, 1920)  # whole blocks run
    assert set(built) == {("fp8_block", 4096, 1920, 128)}
    namespace["get_sm_version"] = lambda: 100
    _run(namespace, "fp8_block", 64, 4096, 1856)  # DeepGEMM takes it on SM100
    assert built[-1] == ("fp8_block", 4096, 1856, 128)


def test_nvfp4_kernel_source_names_the_backend_the_tuner_picked_for_the_case_m():
    namespace, _, _, _, published, tuner = _load({64: ("cutedsl", 7), 8: ("cuda_core", 0), 1: ("cublaslt", 2)})
    for m in (64, 8, 1):
        _run(namespace, "nvfp4", m, 4096, 4096)
    assert [row["kernel_source"] for row in published] == ["nvfp4_cutedsl", "nvfp4_cuda_core", "nvfp4_cublaslt"]
    assert len(tuner.passes) == 1


def test_nvfp4_tuner_miss_raises_before_timing():
    namespace, _, _, benchmarks, published, _ = _load({8: ("cutlass", 3)})
    with pytest.raises(RuntimeError, match=r"trtllm::nvfp4_gemm::gemm has no tuned tactic"):
        _run(namespace, "nvfp4", 64, 4096, 4096)
    assert benchmarks == [] and published == []


def test_unaligned_fp8_runs_instead_of_being_refused():
    """The collector does not predict TRT-LLM's fp8 alignment limit. TRT-LLM serves
    m <= 8 with n in {1, 4, 8, 12, 24} on SM89/SM120 (rows exist), and where the
    kernel rejects the shape its own TORCH_CHECK fails the case, which is recorded."""
    namespace, built, _, _, published, _ = _load()
    _run(namespace, "fp8", 16, 24, 4096)
    assert built and set(built) == {("fp8", 24, 4096, None)}
    assert [row["item_list"][0]["n"] for row in published] == [24]


class _FakeTensor:
    """Records the device and view shape the nvfp4 weight build passes along."""

    def __init__(self, name, shape=None, device="cuda"):
        self.name = name
        self.shape = shape
        self.device = device

    def abs(self):
        return self

    def max(self):
        return self

    def float(self):
        return self

    def __rtruediv__(self, other):
        return _FakeTensor(f"{other}/{self.name}", device=self.device)

    def cpu(self):
        return _FakeTensor(self.name, self.shape, device="cpu")

    def view(self, *shape):
        return _FakeTensor(self.name, shape, self.device)


def test_nvfp4_weights_unswizzle_block_scales_on_the_device_without_a_version_branch():
    """``trtllm::nvfp4_block_scale_interleave_reverse`` exists in no pinned release, so
    the build must take ``block_scale_interleave_reverse`` on the padded
    [pad_up(n, 128), -1] scale matrix whatever ``tensorrt_llm.__version__`` says. The
    sandbox has no ``tensorrt_llm`` and only these two ops, so a version check or the
    other op name raises."""
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    wanted = ("pad_up", "_build_weights")
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(selected) == 2
    reversed_inputs: list = []

    def block_scale_interleave_reverse(scales):
        reversed_inputs.append(scales)
        return _FakeTensor("w_sf_unswizzled", scales.shape, scales.device)

    namespace = {
        "torch": SimpleNamespace(
            bfloat16="bf16",
            float8_e4m3fn="e4m3",
            randn=lambda shape, dtype, device: _FakeTensor("w", shape, device),
            ops=SimpleNamespace(
                trtllm=SimpleNamespace(
                    fp4_quantize=lambda w, scale, block, ue8m0: (_FakeTensor("w_fp4"), _FakeTensor("w_sf")),
                    block_scale_interleave_reverse=block_scale_interleave_reverse,
                )
            ),
        ),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

    weights = namespace["_build_weights"]("nvfp4", 8, 4096, "cuda", "bf16", 16, _FakeTensor("x", (4, 4096)))

    assert [(t.shape, t.device) for t in reversed_inputs] == [((128, -1), "cuda")]
    assert set(weights) == {"weight", "weight_scale", "weight_scale_2", "input_scale"}
    assert weights["weight_scale"].shape == ("e4m3",)
