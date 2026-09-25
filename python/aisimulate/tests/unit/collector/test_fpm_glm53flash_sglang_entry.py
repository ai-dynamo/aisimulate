# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU launch routing regression; this does not qualify native GPU execution."""

import sys
from types import SimpleNamespace

import pytest
from collector.fpm_forward import sglang_driver

pytestmark = pytest.mark.unit


def test_observer_reaches_constructor_class_when_public_engine_is_a_proxy(monkeypatch):
    calls = []

    def observed(server):
        calls.append(server)

    class NativeEngine:
        run_scheduler_process_func = staticmethod(lambda server: pytest.fail("unobserved native launch"))

        def __init__(self, *, server_args):
            self.run_scheduler_process_func(server_args)

    class PublicProxy:
        def __getattr__(self, name):
            return getattr(NativeEngine, name)

        def __call__(self, **kwargs):
            return NativeEngine(**kwargs)

    public = PublicProxy()
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=public))
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", SimpleNamespace(Engine=NativeEngine))
    monkeypatch.setattr(sglang_driver, "observed_scheduler_process", observed)
    engine = sglang_driver.create_observed_engine("frozen-server-args")
    assert isinstance(engine, NativeEngine)
    assert calls == ["frozen-server-args"]
    assert NativeEngine.run_scheduler_process_func is observed
    assert "run_scheduler_process_func" not in vars(public)
