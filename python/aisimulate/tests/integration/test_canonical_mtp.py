# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit MTP retains its target identity and the established iteration cost."""

import copy

import pytest

from aisimulate_core.sdk import RustForwardPassPerfModel, common, models
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.speculation import SpeculationConfig

pytestmark = pytest.mark.integration


def _request(model="nvidia/GLM-5.2-NVFP4", backend="sglang"):
    return {
        "model": model,
        "system": "b200_sxm",
        "backend": backend,
        "backend_version": "0.5.14" if backend == "sglang" else "0.24.0",
        "worker_type": "aggregated",
        "tp": 8,
        "moe_tp_size": 8,
        "moe_ep_size": 1,
        "estimation_mode": "op_level",
        "speculation": {"kind": "mtp", "params": {"num_speculative_tokens": 3}},
    }


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("model", ["nvidia/GLM-5.2-NVFP4", "deepseek-ai/DeepSeek-V4-Pro"])
def test_canonical_mtp_round_trip_retains_explicit_method(model, backend):
    request = _request(model, backend)
    normalized = RustForwardPassPerfModel.normalize_config(request)
    assert normalized["model"] == model
    assert normalized["backend"] == backend
    assert normalized["nextn"] == 0
    assert normalized["speculation"] == request["speculation"]
    assert RustForwardPassPerfModel.normalize_config(normalized) == normalized


def test_canonical_mtp_compiler_bridge_translates_only_cost_depth(monkeypatch):
    from aisimulate_core.sdk import engine

    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        raise InvalidEngineConfigurationError("fixture stopped before performance data loading")

    monkeypatch.setattr(engine, "compile_engine", capture)
    with pytest.raises(ValueError, match="fixture stopped"):
        RustForwardPassPerfModel.best_available(_request())
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "nvidia/GLM-5.2-NVFP4"
    assert kwargs["nextn"] == 0
    assert kwargs["speculation"] == {"kind": "mtp", "params": {"depth": 3}}
    assert "expected_accepted_tokens" not in kwargs
    assert "seed" not in kwargs


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_canonical_mtp_glm_iteration_cost_matches_legacy_nextn(backend):
    request = _request(backend=backend)
    legacy_request = {key: value for key, value in request.items() if key != "speculation"}
    legacy_request["nextn"] = 3
    explicit = RustForwardPassPerfModel.best_available(request)
    legacy = RustForwardPassPerfModel.best_available(legacy_request)
    try:
        # The existing nextn engine is the reference: method spelling must not
        # change the draft+verify iteration graph or add an acceptance discount.
        explicit_ms = explicit.predict_decode_latency_total(1, 1024)
        assert explicit_ms > 0
        assert explicit_ms == legacy.predict_decode_latency_total(1, 1024)
        saved = explicit.diagnostics()["provenance"]["config"]
        assert saved["speculation"] == request["speculation"]
        assert saved["model"] == request["model"]
        assert saved["nextn"] == 0
    finally:
        explicit.close()
        legacy.close()


def _model_config(**changes):
    return ModelConfig(
        tp_size=8,
        pp_size=1,
        moe_tp_size=8,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.fp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        **changes,
    )


@pytest.mark.parametrize(
    ("model_path", "family", "expected_scale"),
    [
        # Three extra target-shaped layers: 78+3 for GLM, 61+3 for DSV4 Pro.
        ("nvidia/GLM-5.2-NVFP4", "DEEPSEEKV32", 81 / 78),
        ("deepseek-ai/DeepSeek-V4-Pro", "DEEPSEEKV4", 64 / 61),
    ],
)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_explicit_mtp_preserves_target_graph_and_legacy_cost(model_path, family, expected_scale, backend):
    legacy = models.get_model(model_path, _model_config(nextn=3), backend)
    authored = _model_config(speculation=SpeculationConfig(kind="mtp", params={"depth": 3}))
    original = copy.deepcopy(authored.speculation)
    explicit = models.get_model(model_path, authored, backend)
    assert explicit.model_path == model_path
    assert explicit.model_family == family
    assert type(explicit) is type(legacy)
    assert explicit.architecture == legacy.architecture
    assert explicit.verify_width == 4
    assert explicit._mtp_scale_factor == pytest.approx(expected_scale)
    assert [op._spec_json() for op in explicit.context_ops] == [op._spec_json() for op in legacy.context_ops]
    assert [op._spec_json() for op in explicit.generation_ops] == [op._spec_json() for op in legacy.generation_ops]
    assert authored.speculation == original
    assert authored.nextn == 0


def test_explicit_mtp_does_not_relabel_kimi_dspark_graph():
    config = _model_config(speculation=SpeculationConfig(kind="mtp", params={"depth": 3}))
    with pytest.raises(InvalidEngineConfigurationError, match="nextn graph models DSPARK"):
        models.get_model("moonshotai/Kimi-K3", config, "sglang")


def test_explicit_mtp_preserves_dsv41_separate_execution_contract():
    config = _model_config(speculation=SpeculationConfig(kind="mtp", params={"depth": 3}))
    with pytest.raises(NotImplementedError, match="DSpark has a separate execution contract"):
        models.get_model("deepseek-ai/DeepSeek-V4.1-Flash", config, "vllm")


def test_canonical_cache_separates_method_and_depth():
    requests = []
    for kind, depth in (("mtp", 1), ("mtp", 3), ("ngram", 3)):
        request = _request(backend="vllm")
        request["speculation"] = {"kind": kind, "params": {"num_speculative_tokens": depth}}
        requests.append(request)
    predictions = []
    # Query alternating identities twice to exercise any cached construction.
    for request in requests + requests:
        model = RustForwardPassPerfModel.best_available(request)
        try:
            predictions.append(model.predict_decode_latency_total(1, 1024))
            assert model.diagnostics()["provenance"]["config"]["speculation"] == request["speculation"]
        finally:
            model.close()
    assert predictions[:3] == predictions[3:]
    assert len(set(predictions[:3])) == 3
    assert predictions[1] > predictions[2]  # Same verify width, additional MTP draft cost.
