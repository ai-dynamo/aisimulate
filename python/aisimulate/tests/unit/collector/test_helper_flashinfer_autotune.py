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
from typing import ClassVar

import pytest
from collector import helper
from collector.helper import (
    FLASHINFER_AUTOTUNE_BUDGET_ENV,
    FLASHINFER_AUTOTUNE_ENV,
    FlashInferTuneBudget,
    flashinfer_autotuned,
)

pytestmark = pytest.mark.unit

_REAL_DEVICE_TAG = helper._flashinfer_device_tag


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
    # Healthy CUDA context unless a test says otherwise; no device tag (other test modules
    # leave a stub ``torch`` in sys.modules, so the tag must not depend on the import).
    monkeypatch.setattr(helper, "cuda_context_error", lambda device=None: None)
    monkeypatch.setattr(helper, "_flashinfer_device_tag", lambda: "")
    return types.SimpleNamespace(events=events, active=active, tuner=_FakeTuner)


def test_tunes_once_at_the_exact_token_count_then_times_under_lookup_only(fake_flashinfer, capsys):
    seen: list[tuple] = []

    # A key tuned by an earlier case on this worker is not reported again.
    fake_flashinfer.tuner.get().profiling_cache[("moe", 64)] = ((64, 1), None)

    def tune_fn():
        # The tuning forward runs inside tune_mode=True with the exact bucket and
        # stands in for the autotuner filling its cache.
        seen.append((tuple(fake_flashinfer.active), fake_flashinfer.tuner.get().budget()))
        fake_flashinfer.tuner.get().profiling_cache[("moe", 320)] = ("tactic", "profile")

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
    assert "FlashInfer autotune chosen at num_tokens=320 vllm_trtllm_nvfp4: ('moe', 320) -> tactic 'tactic'" in out
    assert "WARNING" not in out


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


def test_autotuner_fallback_messages_are_surfaced_and_logging_is_restored(fake_flashinfer, capsys):
    """b300 sglang 0.5.21 (2026-10-28): the Nemotron-3-Nano nvfp4 relu2 shape came back
    identical to the untuned table while every other shape moved -- FlashInfer's
    ``MoERunner.get_valid_tactics`` swallows the C++ exception, logs it at DEBUG and
    returns ``[]``, so the op is timed on the fallback with no visible trace. The tuning
    pass captures that log and prints the explanation next to the case."""
    import logging

    fi_logger = logging.getLogger("flashinfer")
    # Other test modules leave this logger at an explicit level (seen at ERROR in
    # the full collector run); the scenario needs the production default, NOTSET.
    inherited_level = fi_logger.level
    fi_logger.setLevel(logging.NOTSET)
    # The threshold the capture pins pre-existing handlers at is the logger's EFFECTIVE level,
    # inherited from the root logger — WARNING by default, INFO when another test module
    # configured logging in this process; read it instead of assuming.
    effective_before = fi_logger.getEffectiveLevel()
    stream = logging.StreamHandler()
    fi_logger.addHandler(stream)
    try:
        seen_levels = []

        def tune_fn():
            seen_levels.append((fi_logger.level, stream.level))
            fi_logger.debug(
                "[Autotuner]: Failed to get valid tactics for (E2m1, 6, 2688, 1920, 128, Relu2). "
                "Error occurred: No kernel found for the given options: mActType 6 tile 192"
            )
            fi_logger.debug("[Autotuner]: profiling runner MoERunner tactic (128, 3)")
            fi_logger.debug("[AutoTuner]: Using fallback tactic for trtllm_fp4_block_scale_moe")
            fi_logger.debug("[AutoTuner]: Using fallback tactic for trtllm_fp4_block_scale_moe")  # duplicate

        with flashinfer_autotuned(768, tune_fn, label="sglang_flashinfer_trtllm_moe"):
            pass

        # Captured at DEBUG, while the pre-existing stream handler was pinned at the old
        # effective threshold so FlashInfer's stderr output did not change.
        assert seen_levels == [(logging.DEBUG, effective_before)]
        assert fi_logger.level == logging.NOTSET
        assert stream.level == logging.NOTSET
        assert all(h is not fi_logger for h in fi_logger.handlers) and stream in fi_logger.handlers
    finally:
        fi_logger.removeHandler(stream)
        fi_logger.setLevel(inherited_level)

    out = capsys.readouterr().out
    assert "FlashInfer autotune: 0 new tactic entries at num_tokens=768 sglang_flashinfer_trtllm_moe [" in out
    warning = [line for line in out.splitlines() if line.startswith("FlashInfer autotune WARNING")]
    assert len(warning) == 1
    assert "num_tokens=768 sglang_flashinfer_trtllm_moe: 2 autotuner messages" in warning[0]
    assert "No kernel found for the given options: mActType 6 tile 192" in warning[0]
    assert warning[0].count("Using fallback tactic") == 1
    assert "profiling runner" not in warning[0]
    assert "tactics failed while profiling" not in warning[0]


def test_per_tactic_profiling_failures_are_counted_from_the_tuner_statistics(fake_flashinfer, capsys):
    """Every candidate tactic runs the whole op on synthetic inputs; when e.g. the routing
    kernel faults on them FlashInfer marks the tactic inf at DEBUG, records it in
    ``stats.failed_tactics`` and -- if none survives -- serves the fallback. The pass
    reports the new failures from the statistics even when no log line matched."""
    import logging

    tuner = fake_flashinfer.tuner.get()
    op = "flashinfer::trtllm_fp4_block_scale_moe::MoERunner"
    tuner.stats = types.SimpleNamespace(failed_tactics={op: {(8, 1)}})
    fi_logger = logging.getLogger("flashinfer")

    def tune_fn():
        tuner.stats.failed_tactics[op].update({(16, 2), (32, 5)})
        fi_logger.debug(
            "[Autotuner]: Failed when profiling MoERunner (16, 2), shapes=[...]. Error occurred: "
            "CUDA error in trtllm_fused_moe_routing_custom.cu:1045: an illegal memory access"
        )
        fi_logger.info("[Autotuner]: Skipped 2 unsupported tactic(s) for trtllm_fp4_block_scale_moe")

    with flashinfer_autotuned(3072, tune_fn, label="sglang_flashinfer_trtllm_moe"):
        pass

    out = capsys.readouterr().out
    warning = [line for line in out.splitlines() if line.startswith("FlashInfer autotune WARNING")]
    assert len(warning) == 1
    assert f"tactics failed while profiling: {op}=2" in warning[0]
    assert "routing_custom.cu:1045" in warning[0]
    assert "Skipped 2 unsupported tactic(s)" in warning[0]


def test_newly_enumerated_tactic_tables_are_printed_per_instance_key(fake_flashinfer, monkeypatch, capsys):
    """Probe 77139257 (b200 sglang 0.5.21, 2026-10-28): the Nemotron relu2 nvfp4 cases wrote one
    cache entry each after a 0.02-0.9 s pass and still measured the untuned latency -- the
    autotuner may simply have had one candidate. Print the launcher-enumerated tactic list
    per instance key (intermediate size, activation, ... included) for every key the pass
    added, reading FlashInfer's already-built module only (no JIT build from here)."""

    class _Runner:
        valid_tactics_dict: ClassVar[dict] = {("E2m1", 8, 7168, 2048, 256, "Swiglu", 768): [(64, 0), (64, 1), (128, 0)]}

    calls = []

    class _Impl:
        currsize = 0

        @staticmethod
        def cache_info():
            return types.SimpleNamespace(currsize=_Impl.currsize)

    def get_module():
        calls.append("get")
        return types.SimpleNamespace(MoERunner=_Runner)

    core = types.ModuleType("flashinfer.fused_moe.core")
    core._get_trtllm_moe_sm100_module_impl = _Impl
    core.get_trtllm_moe_sm100_module = get_module
    fused = types.ModuleType("flashinfer.fused_moe")
    fused.core = core
    sys.modules["flashinfer"].fused_moe = fused
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", core)

    # Module not built yet: nothing is read and the factory is not called.
    with flashinfer_autotuned(8, lambda: None, label="cold"):
        pass
    assert calls == []
    assert "FlashInfer autotune tactics" not in capsys.readouterr().out

    def tune_fn():
        _Impl.currsize = 1  # the forward built the module ...
        _Runner.valid_tactics_dict[("E2m1", 6, 2688, 1920, 128, "Relu2", 768)] = [(64, 3)]  # ... and enumerated

    with flashinfer_autotuned(768, tune_fn, label="sglang_flashinfer_trtllm_moe"):
        pass
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("FlashInfer autotune tactics")]
    # The module came up during this pass, so every key it holds is new to the snapshot:
    # the control shape's three tactics and the single Relu2 candidate.
    assert len(lines) == 2
    assert (
        "FlashInfer autotune tactics at num_tokens=768 sglang_flashinfer_trtllm_moe: trtllm_sm100 "
        "('E2m1', 8, 7168, 2048, 256, 'Swiglu', 768) -> 3 tactics [(64, 0), (64, 1), (128, 0)]"
    ) in lines
    assert (
        "FlashInfer autotune tactics at num_tokens=768 sglang_flashinfer_trtllm_moe: trtllm_sm100 "
        "('E2m1', 6, 2688, 1920, 128, 'Relu2', 768) -> 1 tactics [(64, 3)]"
    ) in lines

    # Next pass: only keys added since the snapshot are reported.
    def tune_fn_ep4():
        _Runner.valid_tactics_dict[("E2m1", 6, 2688, 1920, 32, "Relu2", 768)] = [(64, 3)]

    with flashinfer_autotuned(768, tune_fn_ep4, label="sglang_flashinfer_trtllm_moe"):
        pass
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("FlashInfer autotune tactics")]
    assert lines == [
        "FlashInfer autotune tactics at num_tokens=768 sglang_flashinfer_trtllm_moe: trtllm_sm100 "
        "('E2m1', 6, 2688, 1920, 32, 'Relu2', 768) -> 1 tactics [(64, 3)]"
    ]


def test_autotune_lines_carry_the_cuda_device_tag(fake_flashinfer, monkeypatch, capsys):
    """Parallel workers share one log and tune the same key independently; the device tag
    is what lets a per-worker tactic choice be reconstructed afterwards (gb200 sglang part07
    could not be: no worker tag on the autotune lines)."""

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 3

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=_Cuda))
    monkeypatch.setattr(helper, "_flashinfer_device_tag", _REAL_DEVICE_TAG)  # fixture blanks it
    assert helper._flashinfer_device_tag() == " dev3"

    def tune_fn():
        fake_flashinfer.tuner.get().profiling_cache[("moe", 96)] = ((128, 2), None)

    with flashinfer_autotuned(96, tune_fn, label="sglang_flashinfer_trtllm_moe"):
        pass
    out = capsys.readouterr().out
    assert "FlashInfer autotune: 1 new tactic entries at num_tokens=96 sglang_flashinfer_trtllm_moe [" in out
    assert "delay_us=1000] dev3\n" in out
    assert "-> tactic (128, 2) dev3\n" in out
