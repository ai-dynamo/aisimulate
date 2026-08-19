# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator_core.sdk import engine, perf_database


def test_explicit_database_controls_use_exact_request_scoped_view(monkeypatch):
    expected = SimpleNamespace(name="configured-view")
    calls = []

    def fake_get_database_view(system, backend, version, **kwargs):
        calls.append((system, backend, version, kwargs))
        return expected

    monkeypatch.setattr(perf_database, "get_database_view", fake_get_database_view)

    actual = engine._maybe_load_database(
        "gb200_nv18",
        "sglang",
        "0.5.6",
        "/tmp/pinned-performance-data",
        database_mode="HYBRID",
        transfer_policy=["xshape", "xquant"],
    )

    assert actual is expected
    assert calls == [
        (
            "gb200_nv18",
            "sglang",
            "0.5.6",
            {
                "systems_paths": "/tmp/pinned-performance-data",
                "allow_missing_data": False,
                "database_mode": "HYBRID",
                "transfer_policy": ["xshape", "xquant"],
            },
        )
    ]


def test_explicit_database_controls_require_exact_backend_version():
    with pytest.raises(ValueError, match="require an exact backend_version"):
        engine._maybe_load_database(
            "gb200_nv18",
            "sglang",
            None,
            "/tmp/pinned-performance-data",
            database_mode="SILICON",
        )
