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
from collector.helper import FLASHINFER_AUTOTUNE_ENV, flashinfer_autotuned

pytestmark = pytest.mark.unit


class _FakeTuner:
    instance = None

    def __init__(self):
        self.profiling_cache = {}

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
    return types.SimpleNamespace(events=events, active=active, tuner=_FakeTuner)


def test_tunes_once_at_the_exact_token_count_then_times_under_lookup_only(fake_flashinfer, capsys):
    seen: list[tuple] = []

    def tune_fn():
        # The tuning forward runs inside tune_mode=True with the exact bucket and
        # stands in for the autotuner filling its cache.
        seen.append(tuple(fake_flashinfer.active))
        fake_flashinfer.tuner.get().profiling_cache[("moe", 320)] = "tactic"

    with flashinfer_autotuned(320, tune_fn, label="vllm_trtllm_nvfp4") as tuned:
        assert tuned is True
        assert fake_flashinfer.active == [(False, (320,))]

    assert seen == [((True, (320,)),)]
    assert fake_flashinfer.events == [
        ("enter", True, (320,)),
        ("exit", True, (320,)),
        ("enter", False, (320,)),
        ("exit", False, (320,)),
    ]
    out = capsys.readouterr().out
    assert "FlashInfer autotune: 1 new tactic entries at num_tokens=320 vllm_trtllm_nvfp4" in out


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
