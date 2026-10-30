# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendor discrimination for backend selection (`gpu.vendor`).

`VLLMXPUBackend` carries coefficients fitted on Intel B60 hardware. They used
to be selected by "the system spec has no sm_version", which is true of *every*
non-NVIDIA device — so any new non-CUDA system (Ascend NPU) would silently
inherit Intel-fitted TTFT/TPOT laws. Selection is now a positive match on the
spec's declared vendor, with the old inference kept only for specs written
before the key existed.
"""

import pytest

from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.backends.vllm_backend import VLLMBackend
from aisimulate_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
from aisimulate_core.sdk.perf_database import is_xpu_system, is_xpu_system_spec

pytestmark = pytest.mark.unit


def test_b60_declares_intel_and_stays_xpu() -> None:
    assert is_xpu_system("b60") is True
    assert isinstance(get_backend("vllm", "b60"), VLLMXPUBackend)


def test_legacy_spec_without_vendor_keeps_its_old_classification() -> None:
    # Pre-`vendor` specs: sm_version ⇒ NVIDIA, otherwise the only non-NVIDIA
    # system that existed then (Intel XPU).
    assert is_xpu_system_spec({"gpu": {"sm_version": 90}}) is False
    assert is_xpu_system_spec({"gpu": {"mem_capacity": 1}}) is True


def test_declared_vendor_wins_over_the_absent_sm_version() -> None:
    """The must-fix: a non-NVIDIA, non-Intel device must not be routed to the
    Intel-calibrated backend just because it carries no sm_version."""
    ascend_spec = {"gpu": {"vendor": "ascend", "mem_capacity": 1}}
    assert is_xpu_system_spec(ascend_spec) is False

    backend = get_backend("vllm", "ignored", system_spec=ascend_spec)
    assert type(backend) is VLLMBackend
    assert not isinstance(backend, VLLMXPUBackend)


def test_declared_intel_vendor_routes_to_the_xpu_backend() -> None:
    intel_spec = {"gpu": {"vendor": " Intel ", "sm_version": 90}}
    assert is_xpu_system_spec(intel_spec) is True
    assert isinstance(get_backend("vllm", "ignored", system_spec=intel_spec), VLLMXPUBackend)


def test_unknown_vendor_fails_loud() -> None:
    # An unrecognized vendor must not fall back to a calibration by accident.
    with pytest.raises(ValueError, match="gpu.vendor"):
        is_xpu_system_spec({"gpu": {"vendor": "acme"}})
    with pytest.raises(ValueError, match="gpu.vendor"):
        is_xpu_system_spec({"gpu": {"vendor": 7}})
