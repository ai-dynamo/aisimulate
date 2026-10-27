# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE callback cleanup must release cycles after return, never during timing."""

import gc
import inspect
import pickle
import weakref
from types import SimpleNamespace

import pytest

from .test_getter_deduplication import _install_vllm_stubs, _load_collector

pytestmark = pytest.mark.unit


@pytest.fixture
def collector(monkeypatch):
    _install_vllm_stubs(monkeypatch)
    return _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")


def test_success_releases_native_style_cycles_after_callback_frame_returns(collector, monkeypatch):
    class ExpertLayer:
        def __init__(self):
            self.parameter = SimpleNamespace(weight_loader=self.load_weight)

        def load_weight(self):
            pass

    references, stages = [], []
    collect = gc.collect
    collect()

    def callback(tokens, *, device):
        layer = ExpertLayer()
        references.append(weakref.ref(layer))
        for token in tokens:
            stages.append(("timing", token, device))
        return len(tokens)

    def cleanup():
        assert references[-1]() is not None
        assert all(frame.function != "callback" for frame in inspect.stack())
        stages.append(("cleanup",))
        collect()
        assert all(ref() is None for ref in references)

    monkeypatch.setattr(collector.gc, "collect", cleanup)
    wrapped = collector._collect_after_success(callback)
    for _ in range(3):
        assert wrapped([1, 2, 16384], device="cuda:0") == 3
    assert stages == [("timing", 1, "cuda:0"), ("timing", 2, "cuda:0"), ("timing", 16384, "cuda:0"), ("cleanup",)] * 3


def test_callback_exception_propagates_without_success_cleanup(collector, monkeypatch):
    error = RuntimeError("native callback failed")
    calls = []
    monkeypatch.setattr(collector.gc, "collect", lambda: calls.append("cleanup"))

    def callback():
        raise error

    with pytest.raises(RuntimeError) as caught:
        collector._collect_after_success(callback)()
    assert caught.value is error
    assert not calls


def test_public_callback_identity_and_signature_are_preserved(collector):
    callback = collector.run_moe_torch
    assert callback.__name__ == "run_moe_torch"
    assert callback.__module__ == "collector.vllm.collect_moe"
    assert callback.__qualname__ == "run_moe_torch"
    assert inspect.signature(callback) == inspect.signature(callback.__wrapped__)
    assert callback.__doc__ == callback.__wrapped__.__doc__
    assert pickle.loads(pickle.dumps(callback)) is callback


def test_wrapper_returns_original_result_without_rewriting_inputs(collector, monkeypatch):
    sentinel, tokens = object(), [1, 2, 16384]
    calls = []
    monkeypatch.setattr(collector.gc, "collect", lambda: calls.append("cleanup"))

    def callback(actual_tokens, *, options):
        assert actual_tokens is tokens and options is sentinel
        return sentinel

    assert collector._collect_after_success(callback)(tokens, options=sentinel) is sentinel
    assert calls == ["cleanup"]
