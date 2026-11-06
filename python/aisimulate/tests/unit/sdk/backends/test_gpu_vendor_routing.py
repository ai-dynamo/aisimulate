# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendor identification for backend selection (`gpu.vendor`).

`VLLMXPUBackend` carries coefficients fitted on Intel B60 hardware. They used
to be selected by "the system spec has no sm_version", which is true of *every*
non-NVIDIA device — so any new non-CUDA system (Ascend NPU) would silently
inherit Intel-fitted TTFT/TPOT laws. Selection is now a positive match on the
spec's declared vendor, with the old inference kept only for specs written
before the key existed.
"""

import pytest

from aisimulate_core.sdk import common
from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.backends.vllm_backend import VLLMBackend
from aisimulate_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
from aisimulate_core.sdk.config import ModelConfig, RuntimeConfig
from aisimulate_core.sdk.perf_database import (
    is_blackwell_system,
    is_hopper_system,
    is_xpu_system,
    is_xpu_system_spec,
)
from aisimulate_core.sdk.step_estimate import StepEstimate

pytestmark = pytest.mark.unit


def test_b60_declares_intel_and_stays_xpu() -> None:
    assert is_xpu_system("b60") is True
    assert isinstance(get_backend("vllm", "b60"), VLLMXPUBackend)


def test_legacy_spec_without_vendor_keeps_its_old_classification() -> None:
    # Pre-`vendor` specs: sm_version ⇒ NVIDIA, otherwise the only non-NVIDIA
    # system that existed then (Intel XPU).
    assert is_xpu_system_spec({"gpu": {"sm_version": 90}}) is False
    assert is_xpu_system_spec({"gpu": {"mem_capacity": 1}}) is True


def test_declared_vendor_wins_over_the_absent_sm_version(monkeypatch) -> None:
    """The must-fix: a non-NVIDIA, non-Intel device must not be routed to the
    Intel-calibrated backend just because it carries no sm_version."""
    ascend_spec = {"gpu": {"vendor": "ascend", "mem_capacity": 80 * (1 << 30)}}
    assert is_xpu_system_spec(ascend_spec) is False

    backend = get_backend("vllm", "ignored", system_spec=ascend_spec)
    assert type(backend) is VLLMBackend
    assert not isinstance(backend, VLLMXPUBackend)

    # Prediction-level check: drive the routed Ascend VLLMBackend through an
    # aggregate estimate and verify numeric output, step provenance, and
    # VLLMBackend's calibrated TTFT terms (_prefill_dispatch_overhead_ms and
    # _ttft_queuing_factor) rather than XPU burst-factor calibration.
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    model = MagicMock()
    model.model_path = "Qwen/Qwen3-0.6B"
    model._nextn = 0
    model._num_layers = 28
    model.encoder_ops = []
    model.config = ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )
    database = SimpleNamespace(
        backend="vllm",
        version="0.30.0rc2",
        system="ascend_950dt",
        system_spec=ascend_spec,
    )
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *a, **k: StepEstimate(
            latency_ms=10.0,
            energy_wms=100.0,
            per_op_latency_ms={"context_gemm": 10.0},
            per_op_source={"context_gemm": "silicon"},
        ),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *a, **k: StepEstimate(
            latency_ms=2.0,
            energy_wms=20.0,
            per_op_latency_ms={"generation_gemm": 2.0},
            per_op_source={"generation_gemm": "silicon"},
        ),
    )
    monkeypatch.setattr(
        backend,
        "_get_memory_usage",
        lambda *a, **k: {
            "total": 1.0,
            "weights": 0.5,
            "activations": 0.25,
            "kvcache": 0.25,
            "nccl": 0.0,
            "others": 0.0,
        },
    )

    b, isl, osl, ctx_tokens = 4, 128, 16, 256
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=b, beam_width=1, isl=isl, osl=osl, prefix=0, engine_step_backend="rust"),
        ctx_tokens=ctx_tokens,
    )
    result = summary.get_result_dict()
    expected_qf = backend._ttft_queuing_factor(b=b, steps_to_finish_ctx=2.0)
    expected_ttft = (10.0 * 1.0 + 28 * 0.8) * expected_qf
    assert expected_qf > 1.0
    assert result["ttft"] == pytest.approx(expected_ttft)
    assert result["tpot"] > 0.0
    assert result["system"] == "ascend_950dt"
    assert summary.get_per_ops_source() == {
        "mix_step": {"context_gemm": "silicon"},
        "genonly_step": {"generation_gemm": "silicon"},
    }


def test_declared_intel_vendor_routes_to_the_xpu_backend(monkeypatch) -> None:
    intel_spec = {"gpu": {"vendor": " Intel ", "sm_version": 90}}
    assert is_xpu_system_spec(intel_spec) is True
    assert isinstance(get_backend("vllm", "ignored", system_spec=intel_spec), VLLMXPUBackend)

    # NVIDIA Hopper/Blackwell classifiers must honor explicit gpu.vendor: a
    # non-NVIDIA spec carrying sm_version must not enter Hopper/Blackwell paths,
    # while explicit nvidia and legacy vendor-less specs keep their SM classification.
    specs = {
        "intel_sm90": {"gpu": {"vendor": "intel", "sm_version": 90}},
        "intel_sm100": {"gpu": {"vendor": "intel", "sm_version": 100}},
        "nvidia_sm90": {"gpu": {"vendor": "nvidia", "sm_version": 90}},
        "nvidia_sm100": {"gpu": {"vendor": "nvidia", "sm_version": 100}},
        "legacy_sm90": {"gpu": {"sm_version": 90}},
        "legacy_sm100": {"gpu": {"sm_version": 100}},
    }
    monkeypatch.setattr("aisimulate_core.sdk.perf_database.load_system_spec", lambda name, *a, **k: specs[name])
    assert is_hopper_system("intel_sm90") is False
    assert is_blackwell_system("intel_sm100") is False
    assert is_hopper_system("nvidia_sm90") is True
    assert is_blackwell_system("nvidia_sm100") is True
    assert is_hopper_system("legacy_sm90") is True
    assert is_blackwell_system("legacy_sm100") is True


def test_unknown_vendor_fails_loud() -> None:
    # An unrecognized or explicitly null vendor must not fall back to a
    # calibration by accident.
    with pytest.raises(ValueError, match="gpu.vendor"):
        is_xpu_system_spec({"gpu": {"vendor": "acme"}})
    with pytest.raises(ValueError, match="gpu.vendor"):
        is_xpu_system_spec({"gpu": {"vendor": 7}})
    with pytest.raises(ValueError, match="gpu.vendor"):
        is_xpu_system_spec({"gpu": {"vendor": None}})
    with pytest.raises(ValueError, match="gpu.vendor"):
        get_backend("vllm", "ignored", system_spec={"gpu": {"vendor": None}})
