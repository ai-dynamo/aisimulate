# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""best_available wiring is inert without a pin and fails loudly with one."""

import pytest

from aisim_core.data import ensure as ensure_mod
from aisim_core.sdk import rust_engine_step as res

pytestmark = pytest.mark.unit


def test_no_pin_does_not_touch_config(monkeypatch):
    monkeypatch.delenv("AISIM_DATASET", raising=False)
    monkeypatch.delenv("AISIM_DATA_LOCAL", raising=False)
    assert res._ensure_dataset({"system": "h200"}, None) is None


def test_pin_from_environment_reaches_ensure(monkeypatch, tmp_path):
    monkeypatch.setenv("AISIM_DATA_LOCAL", str(tmp_path))
    with pytest.raises(ensure_mod.MissingDataError):  # empty local root: precise error, no fallback
        res._ensure_dataset({"system": "h200", "backend": "vllm", "backend_version": "0.1", "model": "m"}, None)


def test_reentrancy_guard_skips(monkeypatch, tmp_path):
    monkeypatch.setenv("AISIM_DATA_LOCAL", str(tmp_path))
    with ensure_mod._guard():
        assert res._ensure_dataset({}, None) is None
