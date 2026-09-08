# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled aic-core engines carry MTP compute depth, not acceptance."""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import engine

pytestmark = pytest.mark.unit


def test_compile_engine_applies_nextn_compute_cost_only(monkeypatch):
    captured = {}

    def _capture_spec(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", _capture_spec)
    monkeypatch.setattr(engine, "_maybe_load_database", lambda *a, **k: None)
    monkeypatch.setattr(engine.aiconfigurator_core, "engine_spec_bincode_from_json", lambda s: b"")

    engine.compile_engine(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        nextn=1,
    )

    model = captured["model"]
    assert model._nextn == 1
    assert not hasattr(model, "_nextn_accepted")
    assert model._mtp_scale_factor == pytest.approx((model._num_layers + 1) / model._num_layers)
    assert captured["kwargs"]["nextn"] == 1
    assert "nextn_accepted" not in captured["kwargs"]


def test_compile_engine_rejects_removed_nextn_accepted_parameter():
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'nextn_accepted'"):
        engine.compile_engine(
            "Qwen/Qwen3-32B",
            "h200_sxm",
            "trtllm",
            nextn=1,
            nextn_accepted=0.7,
        )


def test_compile_engine_propagates_attention_backend_to_model_config(monkeypatch):
    captured = {}

    def _capture_spec(model, **_kwargs):
        captured["model"] = model
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", _capture_spec)
    monkeypatch.setattr(engine, "_maybe_load_database", lambda *a, **k: None)
    monkeypatch.setattr(engine.aiconfigurator_core, "engine_spec_bincode_from_json", lambda s: b"")

    engine.compile_engine(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        attention_backend="fa3",
    )

    assert captured["model"].config.attention_backend == "fa3"


def test_compile_engine_propagates_database_mode_to_database_view(monkeypatch):
    captured = {}

    def _capture_database(*args):
        captured["database_args"] = args
        return None

    monkeypatch.setattr(engine, "build_engine_spec_json", lambda *a, **k: "{}")
    monkeypatch.setattr(engine, "_maybe_load_database", _capture_database)
    monkeypatch.setattr(engine.aiconfigurator_core, "engine_spec_bincode_from_json", lambda s: b"")

    engine.compile_engine(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        database_mode="EMPIRICAL",
        shared_layer=True,
        transfer_policy=["xshape", "xquant"],
        strict_provenance=True,
    )

    assert captured["database_args"][4] == "EMPIRICAL"
    assert captured["database_args"][5] is True
    assert captured["database_args"][6] == ["xshape", "xquant"]
    assert captured["database_args"][7] is True


def test_maybe_load_database_builds_formula_only_empirical_view(monkeypatch):
    from aiconfigurator_core.sdk import perf_database

    captured = {}
    sentinel = object()

    def _capture_view(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(perf_database, "get_database_view", _capture_view)

    result = engine._maybe_load_database(
        "h200_sxm",
        "vllm",
        "0.25.1",
        None,
        "EMPIRICAL",
        True,
        ["xshape", "xquant"],
        False,
    )

    assert result is sentinel
    assert captured["args"] == ("h200_sxm", "vllm", "0.25.1")
    assert captured["kwargs"]["allow_missing_data"] is True
    assert captured["kwargs"]["database_mode"] == "EMPIRICAL"
    assert captured["kwargs"]["shared_layer"] is True
    assert captured["kwargs"]["transfer_policy"] == ["xshape", "xquant"]
    assert captured["kwargs"]["strict_provenance"] is False


def test_maybe_load_database_does_not_silently_downgrade_explicit_mode(monkeypatch):
    from aiconfigurator_core.sdk import perf_database

    def _fail_view(*_args, **_kwargs):
        raise ValueError("unsupported database mode")

    monkeypatch.setattr(perf_database, "get_database_view", _fail_view)

    with pytest.raises(ValueError, match="unsupported database mode"):
        engine._maybe_load_database("h200_sxm", "vllm", "0.25.1", None, "EMPIRICAL", None, None, None)
