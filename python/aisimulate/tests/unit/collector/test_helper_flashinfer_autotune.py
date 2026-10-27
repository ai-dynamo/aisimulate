# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``helper.flashinfer_autotuned`` times FlashInfer-backed kernels on the autotuned
tactic (serving truth) instead of FlashInfer's untuned fallback, whose default kernel
moved between FlashInfer 0.6.12 and 0.6.18 and broke the SM100 nvfp4 MoE EP sweeps of
the vLLM 0.30.0 / SGLang 0.5.21 campaign tables (2026-10-27)."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest

from collector import helper
from collector.helper import (
    FLASHINFER_AUTOTUNE_BUDGET_ENV,
    FLASHINFER_AUTOTUNE_ENV,
    FlashInferTuneBudget,
    flashinfer_autotuned,
)

pytestmark = pytest.mark.unit


class _FakeTuner:
    instance = None

    def __init__(self):
        self.profiling_cache = {}
        # FlashInfer's own per-tactic profiling budget (AutoTuner.__init__ defaults).
        self.warmup = 3
        self.repeat = 10
        self.stream_delay_micro_secs = 5000

    def budget(self):
        return (self.warmup, self.repeat, self.stream_delay_micro_secs)

    @classmethod
    def get(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """A stand-in ``flashinfer.autotuner`` that records every ``autotune`` context."""
    _FakeTuner.instance = None
    events: list[tuple] = []
    active: list[tuple] = []

    @contextmanager
    def autotune(tune_mode=True, cache=None, tuning_buckets=None, round_up=None, skip_ops=None):
        entry = (tune_mode, tuning_buckets)
        events.append(("enter", *entry))
        active.append(entry)
        try:
            yield
        finally:
            active.pop()
            events.append(("exit", *entry))

    module = types.ModuleType("flashinfer.autotuner")
    module.autotune = autotune
    module.AutoTuner = _FakeTuner
    package = types.ModuleType("flashinfer")
    package.autotuner = module
    monkeypatch.setitem(sys.modules, "flashinfer", package)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", module)
    monkeypatch.delenv(FLASHINFER_AUTOTUNE_ENV, raising=False)
    monkeypatch.delenv(FLASHINFER_AUTOTUNE_BUDGET_ENV, raising=False)
    # Healthy CUDA context unless a test says otherwise.
    monkeypatch.setattr(helper, "cuda_context_error", lambda device=None: None)
    return types.SimpleNamespace(events=events, active=active, tuner=_FakeTuner)


def test_tunes_once_at_the_exact_token_count_then_times_under_lookup_only(fake_flashinfer, capsys):
    seen: list[tuple] = []

    def tune_fn():
        # The tuning forward runs inside tune_mode=True with the exact bucket and
        # stands in for the autotuner filling its cache.
        seen.append((tuple(fake_flashinfer.active), fake_flashinfer.tuner.get().budget()))
        fake_flashinfer.tuner.get().profiling_cache[("moe", 320)] = "tactic"

    with flashinfer_autotuned(320, tune_fn, label="vllm_trtllm_nvfp4") as tuned:
        assert tuned is True
        assert fake_flashinfer.active == [(False, (320,))]
        # The reduced budget applies to the tuning pass only; the timed body sees
        # FlashInfer's own values again.
        assert fake_flashinfer.tuner.get().budget() == (3, 10, 5000)

    # Tuned under the collector's default budget: 1 warm-up, 3 timed, 1 ms delay.
    assert seen == [(((True, (320,)),), (1, 3, 1000))]
    assert fake_flashinfer.tuner.get().budget() == (3, 10, 5000)
    assert fake_flashinfer.events == [
        ("enter", True, (320,)),
        ("exit", True, (320,)),
        ("enter", False, (320,)),
        ("exit", False, (320,)),
    ]
    out = capsys.readouterr().out
    assert "FlashInfer autotune: 1 new tactic entries at num_tokens=320 vllm_trtllm_nvfp4 [" in out
    assert "s, warmup=1 repeat=3 delay_us=1000]" in out


def test_budget_env_overrides_or_keeps_flashinfer_defaults(fake_flashinfer, monkeypatch, capsys):
    tuner = fake_flashinfer.tuner.get()
    seen = []

    monkeypatch.setenv(FLASHINFER_AUTOTUNE_BUDGET_ENV, "2, 5, 2000")
    with flashinfer_autotuned(64, lambda: seen.append(tuner.budget())):
        pass
    assert seen == [(2, 5, 2000)]
    assert tuner.budget() == (3, 10, 5000)
    assert "warmup=2 repeat=5 delay_us=2000]" in capsys.readouterr().out

    monkeypatch.setenv(FLASHINFER_AUTOTUNE_BUDGET_ENV, "flashinfer")
    with flashinfer_autotuned(64, lambda: seen.append(tuner.budget())):
        pass
    assert seen[-1] == (3, 10, 5000)
    assert "flashinfer defaults]" in capsys.readouterr().out


@pytest.mark.parametrize("raw", ["3,10", "a,b,c", "1,0,1000", "-1,3,1000", "1,3,-5"])
def test_budget_env_rejects_malformed_values(monkeypatch, raw):
    monkeypatch.setenv(FLASHINFER_AUTOTUNE_BUDGET_ENV, raw)
    with pytest.raises(ValueError, match=FLASHINFER_AUTOTUNE_BUDGET_ENV):
        helper.flashinfer_autotune_budget()


def test_budget_default_and_restore_when_tune_fn_raises(fake_flashinfer, monkeypatch):
    monkeypatch.delenv(FLASHINFER_AUTOTUNE_BUDGET_ENV, raising=False)
    assert helper.flashinfer_autotune_budget() == FlashInferTuneBudget(1, 3, 1000)
    tuner = fake_flashinfer.tuner.get()

    def boom():
        raise RuntimeError("tuning forward failed")

    with pytest.raises(RuntimeError, match="tuning forward failed"), flashinfer_autotuned(8, boom):
        pass
    assert tuner.budget() == (3, 10, 5000)
    assert fake_flashinfer.active == []


def test_env_switch_restores_the_untuned_measurement(fake_flashinfer, monkeypatch):
    monkeypatch.setenv(FLASHINFER_AUTOTUNE_ENV, "0")
    calls = []

    with flashinfer_autotuned(1024, lambda: calls.append("tuned")) as tuned:
        assert tuned is False
        assert fake_flashinfer.active == []

    assert calls == []
    assert fake_flashinfer.events == []


def test_without_flashinfer_the_body_runs_untouched(monkeypatch):
    monkeypatch.delenv(FLASHINFER_AUTOTUNE_ENV, raising=False)
    monkeypatch.setattr(helper, "_flashinfer_autotune_api", lambda: (None, None))
    calls = []

    with flashinfer_autotuned(64, lambda: calls.append("tuned")) as tuned:
        assert tuned is False

    assert calls == []


def test_default_is_on_and_accepts_the_shared_bool_grammar(monkeypatch):
    monkeypatch.delenv(FLASHINFER_AUTOTUNE_ENV, raising=False)
    assert helper.flashinfer_autotune_enabled() is True
    for value, expected in (("false", False), ("no", False), ("1", True), ("TRUE", True)):
        monkeypatch.setenv(FLASHINFER_AUTOTUNE_ENV, value)
        assert helper.flashinfer_autotune_enabled() is expected


def test_both_moe_collectors_wrap_their_timing_in_the_tuning_context():
    """The call sites are GPU-only; pin the wiring statically so a refactor cannot
    silently drop the tuning pass from either collector."""
    from pathlib import Path

    collector_root = Path(helper.__file__).resolve().parent
    for backend in ("vllm", "sglang"):
        source = (collector_root / backend / "collect_moe.py").read_text(encoding="utf-8")
        assert "flashinfer_autotuned," in source, backend
        assert source.count("flashinfer_autotuned(num_tokens, ") == 1, backend
        tuned_at = source.index("flashinfer_autotuned(num_tokens, ")
        timed_at = source.index("benchmark_with_power(", tuned_at)
        assert 0 < timed_at - tuned_at < 400, f"{backend}: tuning context must directly wrap the timing"


def test_poisoned_context_after_the_tuning_pass_is_reported_as_the_autotune_failure(fake_flashinfer, monkeypatch):
    """gb200 job 512476645 (2026-10-27): a Llama-4 (topk=1) nvfp4 case faulted inside a
    candidate tactic; FlashInfer swallowed it, the chosen tactic then failed with an opaque
    'Got CUDA error', and the next case on the worker inherited the dead context."""
    tuner = fake_flashinfer.tuner.get()
    tuner.stats = types.SimpleNamespace(failed_tactics={"flashinfer::trtllm_fp4_block_scale_moe": {(16, 3), (32, 7)}})
    dead = "AcceleratorError: CUDA error: an illegal memory access was encountered"
    monkeypatch.setattr(helper, "cuda_context_error", lambda device=None: dead)
    timed = []

    with pytest.raises(RuntimeError) as info, flashinfer_autotuned(1, lambda: None, label="llama4_nvfp4"):
        timed.append("timed")

    msg = str(info.value)
    assert "FlashInfer autotune pass at num_tokens=1 llama4_nvfp4 left the CUDA context in error state" in msg
    assert "illegal memory access" in msg
    assert "2 failed tactics" in msg and "(16, 3)" in msg
    assert timed == []  # the measurement never ran on the dead context
    assert fake_flashinfer.events == [("enter", True, (1,)), ("exit", True, (1,))]


def test_tune_fn_error_on_a_healthy_context_propagates_unchanged(fake_flashinfer):
    def tune_fn():
        raise ValueError("shape rejected")

    with pytest.raises(ValueError, match="shape rejected"), flashinfer_autotuned(8, tune_fn):
        pass


def test_cuda_context_error_probe_reports_a_dead_context(monkeypatch):
    class _Cuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def synchronize(device):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    fake_torch = types.SimpleNamespace(cuda=_Cuda, ones=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert helper.cuda_context_error() == "RuntimeError: CUDA error: an illegal memory access was encountered"

    class _HealthyCuda(_Cuda):
        @staticmethod
        def synchronize(device):
            return None

    class _Tensor:
        def sum(self):
            return self

        def item(self):
            return 1.0

    fake_torch = types.SimpleNamespace(cuda=_HealthyCuda, ones=lambda *a, **k: _Tensor())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert helper.cuda_context_error(0) is None
