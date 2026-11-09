# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the TRT-LLM MoE worker-recycling policy.

``collector/trtllm/collect_moe.py`` imports torch and tensorrt_llm at module
scope, so the policy helpers and the ``run_moe_torch`` wrapper are AST-extracted
and executed against fakes (same technique as ``test_collect_moe_ep.py``).
"""

from __future__ import annotations

import ast
import enum
import os
import sys
import time
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


def test_release_drops_every_trtllm_workspace_block_and_the_cutlass_runner_scratch():
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == RELEASE)

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
    runner = MagicMock()
    namespace: dict = {"get_memory_buffers": lambda: pool, "MoERunner": runner}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    assert namespace[RELEASE]() == 7168 + 3584 + 28
    assert pool.buffers == {}
    runner.clear_all_workspaces.assert_called_once_with()


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


class _Activation(enum.Enum):
    Swiglu = 0
    Relu2 = 1


class _ConfigurableMoE:
    pass


SWEEP_HELPERS = {"_refuse_moe_weight_capacity", "_run_moe_case", "_tuning_cache_path"}
SWEEP_CONSTANTS = {"_SERVING_MAX_NUM_TOKENS", "_MXFP4_MOE_TYPES"}


def _run_sweep(
    tokens,
    *,
    fail_stage=None,
    fail_tokens=None,
    min_latency=False,
    moe_type="bfloat16",
    shape=(4096, 2048, 2, 8, 2, 2),
    sm=100,
    builder_error=None,
    routing_scales_from_backend=False,
):
    """Run the actual case body with native allocation/timing calls replaced."""
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))

    def wanted(node) -> bool:
        if isinstance(node, ast.FunctionDef):
            return node.name in SWEEP_HELPERS
        if isinstance(node, ast.Assign):
            return any(isinstance(target, ast.Name) and target.id in SWEEP_CONSTANTS for target in node.targets)
        return False

    bodies = [node for node in tree.body if wanted(node)]
    assert len(bodies) == len(SWEEP_HELPERS) + len(SWEEP_CONSTANTS)
    calls = []
    rows = []
    state = {"stage": "built"}
    failure = RuntimeError("native shape failed")
    tune_paths = []
    resolved = []
    warmups = []

    def forward(hidden, logits, **kwargs):
        assert hidden.tokens == logits.tokens
        calls.append((state["stage"], hidden.tokens, kwargs["do_finalize"]))
        if state["stage"] == fail_stage and hidden.tokens == fail_tokens:
            raise failure

    @contextmanager
    def autotune(*, cache_path):
        tune_paths.append(cache_path)
        state["stage"] = "tuning"
        yield
        state["stage"] = "tuned"

    @contextmanager
    def benchmark(**kwargs):
        warmups.append(kwargs["min_warmup_seconds"])
        state["stage"] = "measurement"
        kwargs["kernel_func"]()
        yield {"latency_ms": calls[-1][1] / 1000, "power_stats": {}}

    class ModelConfig(SimpleNamespace):
        @staticmethod
        def resolve_moe_backend(requested, architecture, quant_config):
            resolved.append((requested, architecture, quant_config.quant_algo))
            return "CUTLASS"

    device = SimpleNamespace(index=0)
    torch = MagicMock()
    torch.device.return_value = device
    torch.cuda.get_device_properties.return_value = SimpleNamespace(total_memory=80 * 2**30)
    torch.randn.side_effect = lambda shape: _TokenTensor(shape[0])
    torch.inference_mode = nullcontext
    routing = SimpleNamespace(output_dtype="fp32")
    moe = _ConfigurableMoE()
    moe.forward = forward
    moe.to = lambda device: None
    moe.backend = SimpleNamespace(input_requirement=SimpleNamespace(routing_scales_dtype="bf16"))
    builder = MagicMock(return_value=moe, side_effect=builder_error)
    layer = SimpleNamespace(
        routing_method=routing,
        activation=SimpleNamespace(kind=_Activation.Swiglu),
        bias=False,
        apply_router_weight_on_input=False,
        router_logits_dtype="fp32",
        routing_scales_from_backend=routing_scales_from_backend,
    )
    namespace = {
        "torch": torch,
        "time": time,
        "gc_collect": lambda: None,
        "aic_debug": 0,
        "_moe_quant_config": lambda moe_type, model_name: SimpleNamespace(quant_algo=moe_type),
        "_declared_moe_routing": lambda model: SimpleNamespace(architecture="Qwen3MoeForCausalLM"),
        "get_sm_version": lambda: sm,
        "ModelConfig": ModelConfig,
        "Mapping": SimpleNamespace,
        "SimpleNamespace": SimpleNamespace,
        "_serving_moe_layer": lambda *args: layer,
        "ActivationType": _Activation,
        "create_moe": builder,
        "ConfigurableMoE": _ConfigurableMoE,
        "_moe_kernel_source": lambda moe, min_latency_mode: "label",
        "initialize_dummy_weights": lambda moe: calls.append(("dummy_weights", None, None)),
        "_post_load_weights": lambda moe: calls.append(("post_load", None, None)),
        "balanced_logits": lambda count, experts, topk: _TokenTensor(count),
        "power_law_logits_v3": lambda count, experts, topk, ep, alpha: _TokenTensor(count),
        "autotune": autotune,
        "moe_tune_path": "/tune",
        "_TRTLLM_VERSION": "9.9",
        "benchmark_with_power": benchmark,
        "FIRST_POINT_WARMUP_SECONDS": 0.1,
        "log_perf": lambda **kwargs: rows.extend((row, kwargs["kernel_source"]) for row in kwargs["item_list"]),
        "tensorrt_llm": SimpleNamespace(__version__="9.9"),
    }
    exec(compile(ast.Module(body=bodies, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

    def run():
        namespace["_run_moe_case"](
            moe_type, tokens, *shape, min_latency, "model", "balanced", 0.0, perf_filename="unused"
        )

    return SimpleNamespace(
        run=run,
        calls=calls,
        rows=rows,
        builder=builder,
        failure=failure,
        tune_paths=tune_paths,
        resolved=resolved,
        warmups=warmups,
        routing=routing,
    )


def test_moe_sweep_tunes_as_serving_warmup_then_measures_every_requested_shape():
    # Unsorted input also exercises native capacity: the last item is not max.
    tokens = [2, 65536, 32768]
    sweep = _run_sweep(tokens)
    sweep.run()

    model_config = sweep.builder.call_args.kwargs["model_config"]
    # Tuned as a default deployment (max_num_tokens 8192, which bounds both MoE
    # tunings); the sweep's largest shape still runs in one chunk.
    assert model_config.max_num_tokens == 8192
    assert model_config.moe_max_num_tokens == 65536
    assert model_config.moe_backend == "CUTLASS"
    assert sweep.resolved == [("AUTO", "Qwen3MoeForCausalLM", "bfloat16")]
    assert sweep.builder.call_args.kwargs["allow_backend_degradation"] is True
    # Serving's dummy weights and post_load_weights first, then serving's warmup
    # (max_num_tokens, then 1) under autotune(), then one graph measurement per
    # requested shape.
    assert sweep.calls == (
        [("dummy_weights", None, None), ("post_load", None, None), ("tuning", 8192, True), ("tuning", 1, True)]
        + [("measurement", n, True) for n in tokens]
    )
    assert sweep.tune_paths == ["/tune/sm100_trtllm9.9_bfloat16_h4096_i1024_e4_k2.json"]
    assert [row["num_tokens"] for row, _ in sweep.rows] == tokens
    assert [row["latency"] for row, _ in sweep.rows] == [n / 1000 for n in tokens]
    # Only the first point, right after the autotune pass, warms up for a fixed time.
    assert sweep.warmups == [0.1, 0.0, 0.0]
    assert {source for _, source in sweep.rows} == {"label"}
    assert {
        (row["moe_tp_size"], row["moe_ep_size"], row["moe_dtype"], row["distribution"]) for row, _ in sweep.rows
    } == {(2, 2, "bfloat16", "balanced")}


def test_short_sweep_still_tunes_at_the_serving_max_num_tokens():
    sweep = _run_sweep([1, 16])
    sweep.run()
    assert sweep.builder.call_args.kwargs["model_config"].max_num_tokens == 8192
    assert [call for call in sweep.calls if call[0] == "tuning"] == [("tuning", 8192, True), ("tuning", 1, True)]


def test_min_latency_row_pins_trtllm_without_degradation_and_skips_finalize():
    sweep = _run_sweep([1, 4], min_latency=True)
    sweep.run()
    assert sweep.resolved == []
    assert sweep.builder.call_args.kwargs["model_config"].moe_backend == "TRTLLM"
    assert sweep.builder.call_args.kwargs["allow_backend_degradation"] is False
    assert {call[2] for call in sweep.calls if call[0] not in ("dummy_weights", "post_load")} == {False}


def test_qwen3_routing_output_dtype_is_read_back_from_the_bound_backend():
    sweep = _run_sweep([1], routing_scales_from_backend=True)
    sweep.run()
    assert sweep.routing.output_dtype == "bf16"


@pytest.mark.parametrize("stage", ["tuning", "measurement"])
def test_moe_sweep_propagates_large_shape_failure_without_smaller_substitution(stage):
    # Tuning forwards serving's max_num_tokens (8192); measurement every requested shape.
    fail_tokens = 8192 if stage == "tuning" else 65536
    sweep = _run_sweep([2, 32768, 65536], fail_stage=stage, fail_tokens=fail_tokens)
    with pytest.raises(RuntimeError, match="native shape failed") as caught:
        sweep.run()
    assert caught.value is sweep.failure
    assert sweep.calls[-1] == (stage, fail_tokens, True)
    # Measurement failures preserve only genuinely measured earlier rows.
    assert [row["num_tokens"] for row, _ in sweep.rows] == ([2, 32768] if stage == "measurement" else [])


class _ReachedBuilder(Exception):
    pass


@pytest.mark.parametrize("sm", [90, 100, 120])
@pytest.mark.parametrize(
    ("moe_type", "shape"),
    [
        # Shapes the former 256-bit weight-layout guard refused on every SM:
        # rc2's CUTLASS op checks 128-bit TMA alignment only (moeOp.cpp:654-662),
        # CUTLASS FP8 pads in post_load_weights and the MXFP4 paths pad on load.
        ("bfloat16", (2688, 1856, 6, 128, 32, 1)),  # Nemotron-3-Nano TP32: 58 per rank
        ("fp8", (4096, 1536, 8, 128, 32, 1)),  # Qwen3-235B TP32: 48 per rank
        ("w4a16_mxfp4", (2880, 2880, 4, 128, 4, 1)),
        ("w4a8_mxfp4_mxfp8", (2880, 2880, 4, 128, 8, 1)),
    ],
)
def test_shapes_serving_pads_reach_the_framework_builder(sm, moe_type, shape):
    sweep = _run_sweep([1], moe_type=moe_type, shape=shape, sm=sm, builder_error=_ReachedBuilder)
    with pytest.raises(_ReachedBuilder):
        sweep.run()
    kwargs = sweep.builder.call_args.kwargs
    assert kwargs["intermediate_size"] == shape[1]
    assert kwargs["model_config"].mapping.moe_tp_size == shape[4]


def test_fp8_block_alignment_guard_still_raises_before_the_build():
    sweep = _run_sweep([1], moe_type="fp8_block", shape=(7168, 2048, 8, 256, 32, 1))
    with pytest.raises(ValueError, match="FIXME\\(kernel-limit\\): fp8_block MoE requires 128-aligned"):
        sweep.run()
    sweep.builder.assert_not_called()
