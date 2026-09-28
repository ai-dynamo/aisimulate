# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU backend selection and XPU-calibrated overrides (VLLMXPUBackend)."""

from unittest.mock import MagicMock

import pytest

from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.backends.vllm_backend import VLLMBackend
from aisimulate_core.sdk.backends.vllm_backend_xpu import GPT_OSS_ARCH, VLLMXPUBackend
from aisimulate_core.sdk.perf_database import is_xpu_system
from aisimulate_core.sdk.step_estimate import StepEstimate

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
    assert not isinstance(get_backend("vllm"), VLLMXPUBackend)


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
