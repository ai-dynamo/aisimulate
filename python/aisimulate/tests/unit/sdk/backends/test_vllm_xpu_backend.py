# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU backend selection and XPU-calibrated overrides (VLLMXPUBackend)."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.backends.vllm_backend import VLLMBackend
from aisimulate_core.sdk.backends.vllm_backend_xpu import (
    BURST_A,
    BURST_BS_EXP,
    BURST_CTX_EXP,
    GPT_OSS_ARCH,
    VLLMXPUBackend,
)
from aisimulate_core.sdk.perf_database import is_xpu_system
from aisimulate_core.sdk.step_estimate import MixedStepInput, StepEstimate

pytestmark = pytest.mark.unit

# Arbitrary base generation_moe latency for fixtures; tests assert self-consistency
# (delta == law - base), never a specific measured value.
_FAKE_MOE_MS = 10.0


# ---------------------------------------------------------------------------
# System classification + backend routing
# ---------------------------------------------------------------------------


def test_is_xpu_system_b60_true() -> None:
    # b60's system spec has no sm_version → XPU.
    assert is_xpu_system("b60") is True


def test_is_xpu_system_nvidia_false() -> None:
    # NVIDIA specs carry sm_version → not XPU.
    assert is_xpu_system("h200_sxm") is False


def test_is_xpu_system_none_false() -> None:
    assert is_xpu_system(None) is False


def test_is_xpu_system_unknown_raises() -> None:
    # An unresolvable spec must fail loud, not silently classify as XPU.
    with pytest.raises(ValueError):
        is_xpu_system("nonexistent_typo_system")


def test_get_backend_vllm_on_xpu_routes_to_xpu_backend() -> None:
    assert isinstance(get_backend("vllm", "b60"), VLLMXPUBackend)


def test_get_backend_vllm_on_nvidia_uses_base_backend() -> None:
    backend = get_backend("vllm", "h200_sxm")
    assert isinstance(backend, VLLMBackend)
    assert not isinstance(backend, VLLMXPUBackend)


def test_get_backend_vllm_without_system_uses_base_backend() -> None:
    # No system name (disagg/AFD/encoder paths) → generic backend, never XPU.
    assert type(get_backend("vllm")) is VLLMBackend


@pytest.mark.parametrize("system_name", [None, "h200_sxm"])
def test_get_backend_vllm_no_xpu_system_is_exact_base(system_name) -> None:
    # No system or an NVIDIA system, type is base VLLMBackend
    assert type(get_backend("vllm", system_name)) is VLLMBackend


def test_vllm_xpu_backend_is_a_vllm_backend() -> None:
    # XPU backend inherits every non-overridden behavior from the base vLLM backend.
    assert issubclass(VLLMXPUBackend, VLLMBackend)


# ---------------------------------------------------------------------------
# gpt-oss decode-MoE correction: gating + wiring (not the empirical values)
# ---------------------------------------------------------------------------


def _fake_gpt_oss_model(*, moe_tp=1, moe_ep=1, topk=4, num_experts=32, arch=GPT_OSS_ARCH):
    moe_op = MagicMock()
    moe_op._name = "generation_moe"
    moe_op._moe_tp_size = moe_tp
    moe_op._moe_ep_size = moe_ep
    moe_op._topk = topk
    moe_op._num_experts = num_experts
    model = MagicMock()
    model.architecture = arch
    model.generation_ops = [moe_op]
    return model


def test_gpt_oss_moe_correction_applies_when_unsharded() -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model(moe_tp=1, moe_ep=1)
    corr = backend._gpt_oss_moe_correction(model, {"generation_moe": _FAKE_MOE_MS}, gen_tokens=4)
    assert corr is not None
    delta, target = corr
    assert target == pytest.approx(backend._gpt_oss_moe_ms(4, 4, 32))
    assert delta == pytest.approx(target - _FAKE_MOE_MS)


def test_gpt_oss_moe_correction_skips_when_sharded() -> None:
    # moe_tp * moe_ep > 1: the collected per-GPU slice is already correct.
    backend = VLLMXPUBackend()
    for moe_tp, moe_ep in [(2, 1), (1, 4), (2, 2)]:
        model = _fake_gpt_oss_model(moe_tp=moe_tp, moe_ep=moe_ep)
        assert backend._gpt_oss_moe_correction(model, {"generation_moe": _FAKE_MOE_MS}, gen_tokens=4) is None


def test_gpt_oss_moe_correction_skips_non_gpt_oss() -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model(arch="LlamaForCausalLM")
    assert backend._gpt_oss_moe_correction(model, {"generation_moe": _FAKE_MOE_MS}, gen_tokens=4) is None


def test_gpt_oss_moe_correction_skips_single_token() -> None:
    # bs<=1 decode: the law is a no-op (matches the collected router).
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model()
    assert backend._gpt_oss_moe_correction(model, {"generation_moe": _FAKE_MOE_MS}, gen_tokens=1) is None


def test_gpt_oss_moe_correction_skips_without_moe_op_in_per_ops() -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model()
    assert backend._gpt_oss_moe_correction(model, {"generation_attention": 1.0}, gen_tokens=4) is None


# ---------------------------------------------------------------------------
# Decode-step seam applies the correction (run_agg's TPOT path)
# ---------------------------------------------------------------------------


def test_get_genonly_step_estimate_applies_gpt_oss_correction(monkeypatch) -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model(moe_tp=1, moe_ep=1)

    def fake_super(self, model_, database_, runtime_config_, gen_tokens, isl, osl):
        return StepEstimate(latency_ms=100.0, energy_wms=0.0, per_op_latency_ms={"generation_moe": _FAKE_MOE_MS})

    monkeypatch.setattr(VLLMBackend, "_get_genonly_step_estimate", fake_super)
    monkeypatch.setattr(VLLMXPUBackend, "_max_kv_slots", lambda self, *a, **k: 10_000)

    est = backend._get_genonly_step_estimate(model, MagicMock(), MagicMock(), 4, 1024, 1024)
    target = backend._gpt_oss_moe_ms(4, 4, 32)
    assert est.per_op_latency_ms["generation_moe"] == pytest.approx(target)
    assert est.latency_ms == pytest.approx(100.0 + (target - _FAKE_MOE_MS))
    # Fitted value, not the collected measurement: provenance must say so.
    assert est.per_op_source["generation_moe"] == "estimated"


def test_get_genonly_step_estimate_untouched_for_sharded(monkeypatch) -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model(moe_tp=2, moe_ep=1)

    def fake_super(self, model_, database_, runtime_config_, gen_tokens, isl, osl):
        return StepEstimate(latency_ms=100.0, energy_wms=0.0, per_op_latency_ms={"generation_moe": _FAKE_MOE_MS})

    monkeypatch.setattr(VLLMBackend, "_get_genonly_step_estimate", fake_super)
    monkeypatch.setattr(VLLMXPUBackend, "_max_kv_slots", lambda self, *a, **k: 10_000)

    est = backend._get_genonly_step_estimate(model, MagicMock(), MagicMock(), 4, 1024, 1024)
    # Sharded: no correction — collected value and step latency unchanged.
    assert est.per_op_latency_ms["generation_moe"] == pytest.approx(_FAKE_MOE_MS)
    assert est.latency_ms == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TTFT (XPU-calibrated): the three regimes of _compute_ttft
# ---------------------------------------------------------------------------


def _ttft_kwargs(**overrides):
    kwargs = dict(
        prefill_step_ms=0.0,
        genonly_step_latency_ms=1.0,
        encoder_latency_ms=0.0,
        steps_to_finish_ctx=3.0,
        decode_iterations=5.0,
    )
    kwargs.update(overrides)
    return kwargs


def _patch_ttft_deps(monkeypatch, *, own_prefill=10.0, eff_bs=None, gen=2.0, tpot=1.5):
    monkeypatch.setattr(VLLMXPUBackend, "_mix_step_gen_tokens", lambda self, *a, **k: 5, raising=False)
    monkeypatch.setattr(
        VLLMXPUBackend, "run_mixed", lambda self, *a, **k: StepEstimate(latency_ms=40.0, energy_wms=0.0)
    )
    monkeypatch.setattr(VLLMXPUBackend, "_own_prefill_ms", lambda self, *a, **k: own_prefill)
    monkeypatch.setattr(VLLMXPUBackend, "_get_genonly_step_latency", lambda self, *a, **k: (gen, 0, {}, {}, ()))
    monkeypatch.setattr(VLLMXPUBackend, "_decode_tpot_ms", staticmethod(lambda *a, **k: tpot))
    if eff_bs is not None:
        monkeypatch.setattr(VLLMXPUBackend, "_effective_decode_bs", lambda self, *a, **k: eff_bs)


def test_compute_ttft_single_request_is_raw_prefill(monkeypatch) -> None:
    # b<=1: no queuing, just encoder + own prefill (no burst/admission factor).
    backend = VLLMXPUBackend()
    _patch_ttft_deps(monkeypatch, own_prefill=10.0)
    ttft = backend._compute_ttft(
        MagicMock(), MagicMock(), MagicMock(), 1, 100, 10, 100, 0, **_ttft_kwargs(encoder_latency_ms=2.0)
    )
    assert ttft == pytest.approx(12.0)


def test_compute_ttft_burst_applies_capped_factor(monkeypatch) -> None:
    # eff_bs == b: prefill x burst factor, capped at b.
    backend = VLLMXPUBackend()
    _patch_ttft_deps(monkeypatch, own_prefill=10.0, eff_bs=4, gen=2.0)
    b, isl, ctx = 4, 100, 400
    prefill_ctx = min(ctx, b * isl)  # 400
    factor = 1.0 + BURST_A * math.log2(b) ** BURST_BS_EXP * ((prefill_ctx / isl) * 1.0) ** BURST_CTX_EXP
    ttft = backend._compute_ttft(MagicMock(), MagicMock(), MagicMock(), b, isl, 10, ctx, 0, **_ttft_kwargs())
    assert ttft == pytest.approx(10.0 * min(factor, float(b)))


def test_compute_ttft_admission_adds_queue_wait(monkeypatch) -> None:
    # eff_bs < b: own prefill + Little's-law wait for the (b-eff_bs) queued requests.
    backend = VLLMXPUBackend()
    _patch_ttft_deps(monkeypatch, own_prefill=10.0, eff_bs=2, tpot=1.5)
    b, osl = 4, 10
    ttft = backend._compute_ttft(MagicMock(), MagicMock(), MagicMock(), b, 100, osl, 400, 0, **_ttft_kwargs())
    expected = 10.0 + (b - 2) / 2 * (10.0 + osl * 1.5)
    assert ttft == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TPOT (XPU-calibrated): delegation vs the mixed-step blend
# ---------------------------------------------------------------------------


def _tpot_kwargs(**overrides):
    kwargs = dict(
        num_mix_steps=5,
        num_genonly_steps=0,
        num_mix_steps_for_tpot_calc=0,
        mix_step_latency_ms=4.0,
        genonly_step_latency_ms=1.0,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("b,osl", [(1, 10), (8, 1)])
def test_compute_tpot_delegates_to_base_for_trivial_batch(monkeypatch, b, osl) -> None:
    # b<=1 or osl<=1: no mixed-step accounting, defer to the base backend.
    backend = VLLMXPUBackend()
    monkeypatch.setattr(VLLMBackend, "_compute_tpot", lambda self, **k: 99.0)
    got = backend._compute_tpot(b=b, isl=100, osl=osl, ctx_tokens=200, model=MagicMock(), **_tpot_kwargs())
    assert got == 99.0


def test_compute_tpot_blends_mix_and_genonly_steps(monkeypatch) -> None:
    # A request sees (eff_bs - prefillers)/eff_bs of the mix steps; rest are gen-only.
    backend = VLLMXPUBackend()
    monkeypatch.setattr(VLLMXPUBackend, "_effective_decode_bs", lambda self, *a, **k: 8)
    b, isl, osl, ctx = 8, 100, 10, 200
    got = backend._compute_tpot(
        b=b,
        isl=isl,
        osl=osl,
        ctx_tokens=ctx,
        model=MagicMock(),
        database=MagicMock(),
        runtime_config=MagicMock(),
        **_tpot_kwargs(num_mix_steps=5, mix_step_latency_ms=4.0, genonly_step_latency_ms=1.0),
    )
    prefillers = max(1.0, ctx / isl)  # 2.0
    nmix_eff = min(5 * (8 - prefillers) / 8, float(osl))  # 3.75
    expected = (4.0 * nmix_eff + 1.0 * (osl - nmix_eff)) / osl
    assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# KV-slot cap and run_mixed budget split
# ---------------------------------------------------------------------------


def test_max_kv_slots_divides_kv_budget_by_per_sequence_bytes(monkeypatch) -> None:
    backend = VLLMXPUBackend()
    backend._agg_free_gpu_frac = 0.5
    backend._agg_ctx_tokens = 1
    gib = 1 << 30
    database = SimpleNamespace(system_spec={"gpu": {"mem_capacity": 20 * gib}}, version="0.28.0")
    monkeypatch.setattr(
        backend, "_get_memory_usage", lambda *a, **k: {"weights": 1.0, "activations": 0.5, "nccl": 0.5, "others": 0.0}
    )
    model = MagicMock()
    model.get_kvcache_bytes_per_sequence.return_value = gib
    # kv_budget = 20*0.5 - 2 = 8 GiB; 8 GiB / 1 GiB-per-seq = 8 slots.
    assert backend._max_kv_slots(model, database, isl=512, osl=512) == 8


def test_max_kv_slots_returns_one_when_budget_exhausted(monkeypatch) -> None:
    backend = VLLMXPUBackend()
    backend._agg_free_gpu_frac = 0.1
    backend._agg_ctx_tokens = 1
    gib = 1 << 30
    database = SimpleNamespace(system_spec={"gpu": {"mem_capacity": 4 * gib}}, version="0.28.0")
    # non-KV footprint exceeds mem*util -> no room for KV -> at least 1.
    monkeypatch.setattr(
        backend, "_get_memory_usage", lambda *a, **k: {"weights": 10.0, "activations": 0.0, "nccl": 0.0, "others": 0.0}
    )
    model = MagicMock()
    model.get_kvcache_bytes_per_sequence.return_value = gib
    assert backend._max_kv_slots(model, database, isl=512, osl=512) == 1


def test_run_mixed_reserves_decode_slots_from_the_budget(monkeypatch) -> None:
    # ctx_tokens is the TOTAL per-step budget: cap decoders at min(Little, KV), then
    # hand the base only the leftover prefill tokens.
    backend = VLLMXPUBackend()
    monkeypatch.setattr(VLLMXPUBackend, "_visual_context_tokens", lambda self, *a, **k: 0)
    monkeypatch.setattr(VLLMXPUBackend, "_max_kv_slots", lambda self, *a, **k: 3)
    captured = {}
    monkeypatch.setattr(VLLMBackend, "run_mixed", lambda self, m, d, rc, step: captured.setdefault("step", step))
    rc = SimpleNamespace(isl=100, osl=100)
    backend.run_mixed(MagicMock(), MagicMock(), rc, MixedStepInput(context_tokens=1000, num_decode_requests=50))
    # steady_running = round(1000*100/200)=500; cap = min(500, 3) = 3.
    assert captured["step"].num_decode_requests == 3
    assert captured["step"].context_tokens == 1000 - 3
