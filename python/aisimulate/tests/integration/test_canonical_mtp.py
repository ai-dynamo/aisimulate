# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The explicit cost override retains the target and existing NextN graph."""

import pytest

from aisimulate_core.sdk import RustForwardPassPerfModel, models
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


def test_compiler_bridge_translates_depth_without_acceptance(monkeypatch):
    from aisimulate_core.sdk import engine

    def capture(model, *args, **kwargs):
        assert model == "nvidia/GLM-5.2-NVFP4"
        assert kwargs["nextn"] == 0
        assert kwargs["speculation"] == {"kind": "mtp", "params": {"depth": 3}}
        assert "expected_accepted_tokens" not in kwargs and "seed" not in kwargs
        raise InvalidEngineConfigurationError("checked compiler arguments")

    monkeypatch.setattr(engine, "compile_engine", capture)
    with pytest.raises(ValueError, match="checked compiler arguments"):
        RustForwardPassPerfModel.best_available(_request())


@pytest.mark.parametrize("model", ["nvidia/GLM-5.2-NVFP4", "deepseek-ai/DeepSeek-V4-Pro"])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_override_preserves_target_graph_and_legacy_cost(model, backend):
    shape = {"tp_size": 8, "moe_tp_size": 8, "moe_ep_size": 1}
    legacy_graph = models.get_model(model, ModelConfig(**shape, nextn=3), backend)
    explicit_graph = models.get_model(
        model, ModelConfig(**shape, speculation=SpeculationConfig(kind="mtp", params={"depth": 3})), backend
    )
    assert explicit_graph.model_path == model
    assert type(explicit_graph) is type(legacy_graph)
    assert explicit_graph.architecture == legacy_graph.architecture
    assert [op._spec_json() for op in explicit_graph.generation_ops] == [
        op._spec_json() for op in legacy_graph.generation_ops
    ]
    request = _request(model, backend)
    legacy_request = {key: value for key, value in request.items() if key != "speculation"}
    explicit = RustForwardPassPerfModel.best_available(request)
    legacy = RustForwardPassPerfModel.best_available({**legacy_request, "nextn": 3})
    try:
        cost = explicit.predict_decode_latency_total(1, 1024)
        assert cost > 0 and cost == legacy.predict_decode_latency_total(1, 1024)
        saved = explicit.diagnostics()["provenance"]["config"]
        assert saved["model"] == model and saved["nextn"] == 0
        assert saved["speculation"] == request["speculation"]
    finally:
        explicit.close()
        legacy.close()


def test_flat_mtp_does_not_relabel_dspark_and_rejects_depth_conflicts():
    from aisimulate.runner import _materialize_engine_role

    authored = {
        "aic_model_path": "moonshotai/Kimi-K3",
        "aic_system": "b200_sxm",
        "aic_tp_size": 8,
        "aic_moe_tp_size": 8,
        "aic_moe_ep_size": 1,
        "num_gpu_blocks": 8192,
        "speculation": {"kind": "mtp", "num_speculative_tokens": 3, "expected_accepted_tokens": 1.5},
    }
    resolved = _materialize_engine_role("vllm", "0.24.0", {"tp": 8}, authored, "aggregated")
    config = resolved["rank"]["timing_model"]["config"]
    assert config["speculation"] == {"kind": "mtp", "params": {"num_speculative_tokens": 3}}
    with pytest.raises(ValueError, match="DSPARK"):
        RustForwardPassPerfModel.best_available({**config, "worker_type": "aggregated", "estimation_mode": "op_level"})
    resolved["rank"]["aic_nextn"] = 2
    with pytest.raises(ValueError, match="depth conflicts"):
        _materialize_engine_role("vllm", "0.24.0", {"tp": 8}, {"rank": resolved["rank"]}, "aggregated")


def test_cost_cache_separates_method_and_depth():
    identities = [("mtp", 1), ("mtp", 3), ("ngram", 3)]
    predictions = []
    for kind, depth in identities * 2:
        request = _request(backend="vllm")
        request["speculation"] = {"kind": kind, "params": {"num_speculative_tokens": depth}}
        model = RustForwardPassPerfModel.best_available(request)
        try:
            predictions.append(model.predict_decode_latency_total(1, 1024))
            assert model.diagnostics()["provenance"]["config"]["speculation"] == request["speculation"]
        finally:
            model.close()
    assert predictions[:3] == predictions[3:]
    assert len(set(predictions[:3])) == 3
    assert predictions[1] > predictions[2]  # Same verify width, additional MTP draft cost.


def test_legacy_kimi_metadata_names_dspark_without_changing_its_graph():
    from aisimulate_core.sdk import engine

    target = "moonshotai/Kimi-K3"
    graph = models.get_model(target, ModelConfig(nextn=3, tp_size=8, moe_tp_size=8, moe_ep_size=1), "sglang")
    operations = [op._spec_json() for op in graph.generation_ops]
    assert any(op._name == "draft_attention" for op in graph.generation_ops)
    identity = engine._engine_config_dict(
        model=graph,
        model_path=target,
        system="b200_sxm",
        backend="sglang",
        backend_version="0.5.14",
        kv_block_size=None,
        systems_path=None,
        nextn=3,
    )
    assert identity["nextn"] == 3
    assert identity["speculation_metadata"]["kind"] == "dspark"
    assert identity["speculation_metadata"]["verify_width"] == 4
    assert identity["speculation_metadata"]["max_accepted_draft_tokens"] == 3
    assert [op._spec_json() for op in graph.generation_ops] == operations

    request = _request(target)
    request.pop("speculation")
    model = RustForwardPassPerfModel.best_available({**request, "nextn": 3})
    try:
        assert model.speculation_metadata() == identity["speculation_metadata"]
        assert model.diagnostics()["provenance"]["config"]["nextn"] == 3
    finally:
        model.close()
