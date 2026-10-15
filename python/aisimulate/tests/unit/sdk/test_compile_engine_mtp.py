# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled aic-core engines carry MTP compute depth, not acceptance."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aisimulate.sdk import common, engine

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("system", "explicit", "expected"),
    [
        ("h200_sxm", None, common.FMHAQuantMode.bfloat16),
        ("h200_sxm", "fp8", common.FMHAQuantMode.fp8),
        ("b200_sxm", None, common.FMHAQuantMode.fp8),
        ("b300_sxm", None, common.FMHAQuantMode.fp8),
    ],
)
def test_compile_sglang_mla_resolves_compute_before_building_ops(monkeypatch, system, explicit, expected):
    captured = {}

    def capture(model, **kwargs):
        captured["model"] = model
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", capture)
    monkeypatch.setattr(engine, "_maybe_load_database", lambda *a, **k: None)
    monkeypatch.setattr(engine.aisimulate_core, "engine_spec_bincode_from_json", lambda s: b"")
    engine.compile_engine(
        "deepseek-ai/DeepSeek-V3",
        system,
        "sglang",
        "current",
        tp_size=8,
        moe_tp_size=8,
        fmha_quant_mode=explicit,
        kvcache_quant_mode="bfloat16" if system == "h200_sxm" else "fp8",
    )
    model = captured["model"]
    assert model.config.fmha_quant_mode == expected
    block = next(op for op in model.context_ops if op._name == "context_mla_block")
    serialized = json.loads(block._spec_json())["Fallback"]
    assert serialized["primary"]["MlaModuleContext"]["fmha_quant_mode"] == expected.name
    attention = next(op["ContextMla"] for op in serialized["fallback"] if "ContextMla" in op)
    assert attention["fmha_quant_mode"] == expected.name


def test_compile_engine_applies_nextn_compute_cost_only(monkeypatch):
    captured = {}

    def _capture_spec(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", _capture_spec)
    monkeypatch.setattr(engine, "_maybe_load_database", lambda *a, **k: None)
    monkeypatch.setattr(engine.aisimulate_core, "engine_spec_bincode_from_json", lambda s: b"")

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
    monkeypatch.setattr(engine.aisimulate_core, "engine_spec_bincode_from_json", lambda s: b"")

    engine.compile_engine(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        attention_backend="fa3",
    )

    assert captured["model"].config.attention_backend == "fa3"


def test_compile_engine_propagates_moe_kernel_source_to_model_config(monkeypatch):
    captured = {}

    def _capture_spec(model, **_kwargs):
        captured["model"] = model
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", _capture_spec)
    monkeypatch.setattr(engine, "_maybe_load_database", lambda *a, **k: None)
    monkeypatch.setattr(engine.aisimulate_core, "engine_spec_bincode_from_json", lambda s: b"")

    engine.compile_engine(
        "Qwen/Qwen3-30B-A3B",
        "h200_sxm",
        "trtllm",
        moe_kernel_source="sglang_flashinfer_trtllm_moe",
    )

    assert captured["model"].config.moe_kernel_source == "sglang_flashinfer_trtllm_moe"


def test_compile_engine_propagates_database_mode_to_database_view(monkeypatch):
    captured = {}

    def _capture_database(*args):
        captured["database_args"] = args
        return None

    def _capture_spec(*_args, **kwargs):
        captured["spec_kwargs"] = kwargs
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", _capture_spec)
    monkeypatch.setattr(engine, "_maybe_load_database", _capture_database)
    monkeypatch.setattr(engine.aisimulate_core, "engine_spec_bincode_from_json", lambda s: b"")

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
    assert captured["spec_kwargs"]["database_mode"] == "EMPIRICAL"
    assert captured["spec_kwargs"]["shared_layer"] is True
    assert captured["spec_kwargs"]["transfer_policy"] == ["xshape", "xquant"]
    assert captured["spec_kwargs"]["strict_provenance"] is True


def test_maybe_load_database_builds_formula_only_empirical_view(monkeypatch):
    from aisimulate_core.sdk import perf_database

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


@pytest.mark.parametrize(
    ("database_mode", "shared_layer", "transfer_policy", "strict_provenance"),
    [
        ("EMPIRICAL", None, None, None),
        (None, False, None, None),
        (None, None, [], None),
        (None, None, None, True),
    ],
)
def test_maybe_load_database_does_not_silently_downgrade_explicit_policy(
    monkeypatch,
    database_mode,
    shared_layer,
    transfer_policy,
    strict_provenance,
):
    from aisimulate_core.sdk import perf_database

    def _fail_view(*_args, **_kwargs):
        raise ValueError("unsupported database mode")

    monkeypatch.setattr(perf_database, "get_database_view", _fail_view)

    with pytest.raises(ValueError, match="unsupported database mode"):
        engine._maybe_load_database(
            "h200_sxm",
            "vllm",
            "0.25.1",
            None,
            database_mode,
            shared_layer,
            transfer_policy,
            strict_provenance,
        )


def test_maybe_load_database_keeps_default_load_tolerant(monkeypatch):
    from aisimulate_core.sdk import perf_database

    def _fail_view(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(perf_database, "get_database_view", _fail_view)

    assert engine._maybe_load_database("h200_sxm", "vllm", "0.25.1", None, None, None, None, None) is None


def test_maybe_load_database_leaves_empty_view_for_native_reload(monkeypatch):
    from aisimulate_core.sdk import perf_database

    monkeypatch.setattr(perf_database, "get_database_view", lambda *_args, **_kwargs: None)

    assert engine._maybe_load_database("h200_sxm", "vllm", "0.25.1", None, None, None, None, True) is None


def test_engine_config_preserves_explicit_database_policy_without_view(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(tp_size=1, pp_size=1), _nextn=0)
    monkeypatch.setattr(engine, "_literal_backend_version", lambda *_args: "0.25.1")

    config = engine._engine_config_dict(
        model=model,
        model_path="Qwen/Qwen3-32B",
        system="h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        kv_block_size=None,
        systems_path=None,
        nextn=0,
        database=None,
        database_mode="EMPIRICAL",
        shared_layer=False,
        transfer_policy="balanced",
        strict_provenance=True,
    )

    assert config["database_mode"] == "EMPIRICAL"
    assert config["enable_shared_layer"] is False
    assert config["transfer_policy"] == ["xquant", "xshape"]
    assert config["strict_provenance"] is True
