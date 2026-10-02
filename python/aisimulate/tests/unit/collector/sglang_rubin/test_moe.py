# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import dataclasses
import inspect
import os
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from collector import case_generator
from collector.sglang_rubin import collect_moe, runtime

pytestmark = pytest.mark.unit


@pytest.fixture
def pilot_cases(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", collect_moe.PILOT_MODEL)
    return collect_moe.get_moe_test_cases()


def test_case_contract_retains_declared_topologies_for_campaign_selection(pilot_cases):
    assert pilot_cases
    assert {case[0] for case in pilot_cases} == {"nvfp4"}
    assert {case[8] for case in pilot_cases} == {collect_moe.PILOT_MODEL}
    assert {case[12] for case in pilot_cases} == {"flashinfer_trtllm"}
    assert {tuple(case[2:6]) for case in pilot_cases} == {(6144, 2048, 8, 256)}
    assert {(4, 1), (1, 4)} <= {tuple(case[6:8]) for case in pilot_cases}
    assert all(len(case) == 27 for case in pilot_cases)
    signature = inspect.signature(collect_moe.run_moe_torch)
    selected = [case for case in pilot_cases if case[6:8] == [4, 1]]
    assert len(selected) == 81
    for case in selected:
        bound = signature.bind(*case, perf_filename="unused")
        assert bound.arguments["scoring_func"] == "sigmoid"
        assert bound.arguments["routing_method_type"] == "DeepSeekV3"
        assert bound.arguments["has_correction_bias"] is True


def test_getter_deduplicates_physical_cases_and_rejects_conflicting_routing(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", collect_moe.PILOT_MODEL)
    case = case_generator.get_common_moe_test_cases(backend="sglang")[0]
    monkeypatch.setattr(case_generator, "get_common_moe_test_cases", lambda **_: [case, case])
    assert len(collect_moe.get_moe_test_cases()) == len(case.num_tokens_list)
    changed = dataclasses.replace(case, sglang_moe_routed_scaling_factor=1.0)
    monkeypatch.setattr(case_generator, "get_common_moe_test_cases", lambda **_: [case, changed])
    with pytest.raises(ValueError, match="conflicting execution semantics"):
        collect_moe.get_moe_test_cases()


@pytest.mark.parametrize("field,value", [(0, "fp8_block"), (6, 8), (7, 4), (12, "flashinfer_cutlass"), (18, "softmax")])
def test_rejects_unqualified_execution_before_framework_import(pilot_cases, field, value):
    case = next(case.copy() for case in pilot_cases if case[6:8] == [4, 1])
    case[field] = value
    with pytest.raises(ValueError, match="Unsupported Rubin MoE case"):
        collect_moe.run_moe_torch(*case, perf_filename="unused")


@pytest.mark.parametrize("value", [None, "1"])
def test_direct_moe_execution_rejects_mismatched_environment_before_sglang_import(monkeypatch, pilot_cases, value):
    name = "SGLANG_ENABLE_MOE_DEFERRED_FINALIZE"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    inventory = {
        "observed": {
            "platform": {"system": "Linux", "machine": "aarch64"},
            "reported_build_environment": dict(runtime.EXPECTED_BUILD_ENV),
            "serving_environment": {name: os.environ.get(name)},
            "package_versions": {
                "sglang": {"version": collect_moe.SGLANG_DISTRIBUTION_VERSION},
                "torch": {"version": "test-fake"},
            },
            "cuda": {"available": True, "devices": [{"capability": [10, 7]}], "torch_cuda_version": "13.5"},
        }
    }
    monkeypatch.setattr(runtime, "collect_inventory", lambda: inventory)
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "sglang" or name.startswith("sglang."):
            pytest.fail("SGLang loaded before rejecting the serving environment")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    case = next(case for case in pilot_cases if case[6:8] == [4, 1])
    with pytest.raises(RuntimeError, match=f"Serving environment {name}"):
        collect_moe.run_moe_torch(*case, perf_filename="unused")
    assert os.environ.get(name) == value


def _constructed_layer():
    method = type("ModelOptNvFp4FusedMoEMethod", (), {})()
    backend = SimpleNamespace(value="flashinfer_trtllm")
    fused = SimpleNamespace(
        __name__="fused_experts_none_to_flashinfer_trtllm",
        __module__="sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
    )
    method._moe_runner_backend = backend
    method.runner = SimpleNamespace(runner_backend=backend, fused_func=fused)
    method.enable_flashinfer_trtllm_moe = True
    return SimpleNamespace(quant_method=method, g1_scale_c=object(), supports_deferred_finalize=False)


def test_provenance_requires_constructed_runner_and_fused_routing():
    layer = _constructed_layer()
    bypassed = type("BypassedTopKOutput", (), {})()
    assert collect_moe._kernel_source(layer, bypassed) == "sglang_flashinfer_trtllm_moe"
    with pytest.raises(RuntimeError, match="expected fused TRTLLM path"):
        collect_moe._kernel_source(layer, type("StandardTopKOutput", (), {})())
    layer.quant_method.runner.runner_backend = SimpleNamespace(value="flashinfer_cutlass")
    with pytest.raises(RuntimeError, match="runner_backend=flashinfer_cutlass"):
        collect_moe._kernel_source(layer, bypassed)


@pytest.mark.parametrize("deferred_finalize", [True, None])
def test_constructed_moe_layer_must_disable_deferred_finalize(deferred_finalize):
    layer = _constructed_layer()
    layer.supports_deferred_finalize = deferred_finalize
    with pytest.raises(RuntimeError, match="deferred finalization disabled"):
        collect_moe._kernel_source(layer, type("BypassedTopKOutput", (), {})())


@pytest.fixture
def moe_runtime(monkeypatch, pilot_cases):
    from collector import helper

    state = SimpleNamespace(events=[], calls=[], rows=[], phase=None, failure=None, extend_autotune=False)
    state.case = next(case.copy() for case in pilot_cases if case[6:8] == [4, 1])
    skip_ops = {"configured_skip"}

    class Tensor:
        def __init__(self, *shape, **_):
            self.shape = shape
            self.source = self

        def to(self, **_):
            return self

        def __getitem__(self, rows):
            assert isinstance(rows, slice) and rows.start is None and rows.step is None
            view = Tensor(min(rows.stop, self.shape[0]), *self.shape[1:])
            view.source = self
            return view

    class BypassedTopKOutput:
        def __init__(self, hidden, logits):
            self.logits = logits

    class Layer:
        def __init__(self, **_):
            constructed = _constructed_layer()
            self.quant_method = constructed.quant_method
            self.quant_method.process_weights_after_loading = lambda _: None
            self.g1_scale_c = constructed.g1_scale_c
            self.supports_deferred_finalize = constructed.supports_deferred_finalize
            self.should_fuse_routed_scaling_factor_in_topk = True

        def to(self, device):
            return self

        def named_parameters(self):
            return []

        def parameters(self):
            return []

        def __call__(self, hidden, topk_output):
            state.calls.append((state.phase, hidden, topk_output.logits))
            if state.failure == "warmup":
                raise RuntimeError("warmup failure")

    stream = SimpleNamespace(
        wait_stream=lambda current: state.events.append(("wait", current)),
        synchronize=lambda: state.events.append("synchronize"),
    )

    @contextmanager
    def stream_context(selected):
        assert selected is stream
        state.events.append("stream_enter")
        yield
        state.events.append("stream_exit")

    @contextmanager
    def autotune(enabled, **kwargs):
        assert enabled is True
        assert kwargs == {"skip_ops": skip_ops}
        state.events.append("autotune_enter")
        if state.failure == "autotune":
            raise RuntimeError("native autotune failure")
        state.phase = "tuning"
        yield
        state.phase = None
        state.events.append("autotune_exit")

    def get_skip_ops(runner):
        assert runner is None
        return skip_ops

    tuner = SimpleNamespace(clear_cache=lambda: state.events.append("clear_cache"), is_tuning_mode=False)
    state.tuner = tuner

    def extend_autotune():
        if isinstance(state.extend_autotune, Exception):
            raise state.extend_autotune
        return state.extend_autotune

    @contextmanager
    def benchmark(**kwargs):
        assert {key: value for key, value in kwargs.items() if key != "kernel_func"} == {
            "device": "cuda:0",
            "num_warmups": 5,
            "num_runs": 10,
            "repeat_n": 1,
        }
        state.events.append("capture")
        state.phase = "capture"
        if state.failure == "benchmark":
            raise RuntimeError("benchmark failure")
        kwargs["kernel_func"]()
        yield {"latency_ms": 5.0, "power_stats": None}

    modules = {
        "torch": SimpleNamespace(
            bfloat16="bfloat16",
            float32="float32",
            no_grad=nullcontext,
            device=lambda _: nullcontext(),
            zeros=Tensor,
            randn=Tensor,
            cuda=SimpleNamespace(
                set_device=lambda _: None,
                Stream=lambda **_: stream,
                current_stream=lambda _: "current_stream",
                stream=stream_context,
                empty_cache=lambda: None,
                memory_allocated=lambda _: 0,
                get_device_name=lambda _: "VR200",
                get_device_properties=lambda _: SimpleNamespace(total_memory=1024),
            ),
        ),
        "flashinfer.autotuner": SimpleNamespace(autotune=autotune, AutoTuner=SimpleNamespace(get=lambda: tuner)),
        "sglang.srt.environ": SimpleNamespace(
            envs=SimpleNamespace(SGLANG_FLASHINFER_AUTOTUNE_EXTEND=SimpleNamespace(get=extend_autotune))
        ),
        "sglang.srt.model_executor.runner.flashinfer_autotune": SimpleNamespace(
            get_flashinfer_autotune_skip_ops=get_skip_ops
        ),
        "sglang.srt.layers.moe.ep_moe.layer": SimpleNamespace(get_moe_impl_class=lambda _: Layer),
        "sglang.srt.layers.moe.topk": SimpleNamespace(TopK=lambda **_: BypassedTopKOutput),
        "sglang.srt.layers.moe.utils": SimpleNamespace(RoutingMethodType=SimpleNamespace(DeepSeekV3="DeepSeekV3")),
        "sglang.srt.layers.quantization.modelopt_quant": SimpleNamespace(ModelOptFp4Config=lambda **_: object()),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(collect_moe, "_require_runtime", lambda: None)
    monkeypatch.setattr(collect_moe, "_rank_local_context", lambda _: nullcontext())
    monkeypatch.setattr(collect_moe.importlib.metadata, "version", lambda _: collect_moe.SGLANG_DISTRIBUTION_VERSION)
    monkeypatch.setattr(helper, "balanced_logits", lambda tokens, experts, *_: Tensor(tokens, experts))
    monkeypatch.setattr(helper, "power_law_logits_v3", lambda tokens, experts, *_: Tensor(tokens, experts))
    monkeypatch.setattr(helper, "benchmark_with_power", benchmark)
    monkeypatch.setattr(helper, "log_perf", lambda **kwargs: state.rows.append(kwargs) or True)
    return state


@pytest.mark.parametrize("distribution", ["balanced", "power_law", "uniform"])
@pytest.mark.parametrize("num_tokens", [1, 3, 31, 32, 33, 128, 1024, 16384])
def test_moe_tunes_only_decode_views_and_measures_every_full_input(moe_runtime, distribution, num_tokens):
    state = moe_runtime
    state.case[1] = num_tokens
    state.case[9] = distribution

    collect_moe.run_moe_torch(*state.case, perf_filename="unused")

    assert state.events == [
        "clear_cache",
        ("wait", "current_stream"),
        "stream_enter",
        "autotune_enter",
        "autotune_exit",
        "stream_exit",
        "synchronize",
        "capture",
        "clear_cache",
    ]
    assert [phase for phase, _, _ in state.calls] == ["tuning"] * 5 + ["capture"] * 5
    assert [(hidden.source, logits.source) for _, hidden, logits in state.calls[:5]] == [
        (hidden, logits) for _, hidden, logits in state.calls[5:]
    ]
    assert len({id(logits.source) for _, _, logits in state.calls}) == 5
    assert all(
        hidden.shape == (min(num_tokens, 32), 6144) and logits.shape == (min(num_tokens, 32), 256)
        for _, hidden, logits in state.calls[:5]
    )
    assert all(
        hidden.shape == (num_tokens, 6144) and logits.shape == (num_tokens, 256)
        for _, hidden, logits in state.calls[5:]
    )
    assert state.rows[0]["item_list"][0]["latency"] == 1.0


@pytest.mark.parametrize("failure", ["autotune", "warmup", "benchmark"])
def test_moe_tuning_failures_propagate_before_capture_or_rows(moe_runtime, failure):
    state = moe_runtime
    state.failure = failure

    with pytest.raises(RuntimeError, match=f"{failure} failure"):
        collect_moe.run_moe_torch(*state.case, perf_filename="unused")

    if failure != "benchmark":
        assert "capture" not in state.events
    assert state.events[0] == state.events[-1] == "clear_cache"
    assert state.events.count("clear_cache") == 2
    assert not state.rows


@pytest.mark.parametrize("value", [True, ValueError("native environment failure")])
def test_moe_rejects_unsupported_native_extend_policy_before_timing(moe_runtime, value):
    state = moe_runtime
    state.extend_autotune = value
    expected = ValueError if isinstance(value, Exception) else RuntimeError
    with pytest.raises(expected, match="native environment failure|EXTEND"):
        collect_moe.run_moe_torch(*state.case, perf_filename="unused")
    assert not state.calls
    assert not state.rows
    assert not state.events


def test_moe_rejects_an_active_autotune_context_without_clearing_its_cache(moe_runtime):
    state = moe_runtime
    state.tuner.is_tuning_mode = True
    with pytest.raises(RuntimeError, match="dedicated collector worker"):
        collect_moe.run_moe_torch(*state.case, perf_filename="unused")
    assert not state.calls
    assert not state.rows
    assert not state.events


def test_moe_policy_preserves_all_81_selected_stock_cases(moe_runtime, pilot_cases):
    state = moe_runtime
    for case in (case for case in pilot_cases if case[6:8] == [4, 1]):
        start = len(state.calls)
        collect_moe.run_moe_torch(*case, perf_filename="unused")
        calls = state.calls[start:]
        assert len(calls) == 10
        assert {hidden.shape[0] for _, hidden, _ in calls[:5]} == {min(case[1], 32)}
        assert {hidden.shape[0] for _, hidden, _ in calls[5:]} == {case[1]}
    assert len(state.rows) == 81
    assert all(row["item_list"][0]["latency"] == 1.0 for row in state.rows)
