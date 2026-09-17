# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical construction through the real Python/Rust boundary."""

import json

import pytest

from aiconfigurator_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def request(**changes):
    return ForwardPassPerfModelConfig(
        model="Qwen/Qwen3-32B",
        system="h200_sxm",
        backend="vllm",
        worker_type="aggregated",
        **changes,
    )


def unavailable_compiler(monkeypatch):
    from aiconfigurator_core.sdk import engine

    calls = []

    def unavailable(*args, **kwargs):
        calls.append(kwargs["forward_model"])
        raise ValueError("fixture performance data is unavailable")

    monkeypatch.setattr(engine, "compile_engine", unavailable)
    return calls


def test_auto_searches_priority_with_default_deny(monkeypatch):
    calls = unavailable_compiler(monkeypatch)
    model = RustForwardPassPerfModel.best_available(request())
    diagnostics = model.diagnostics()
    assert calls == ["op_level", "fpm"]
    assert diagnostics["source"] == "fallback_regression"
    assert diagnostics["readiness"] != "ready"
    provenance = diagnostics["provenance"]
    assert provenance["requested_estimation_mode"] == "auto"
    assert provenance["selected_estimation_mode"] == "fpm_regression"
    assert provenance["config"]["fallback_policy"] == "deny"
    assert len(provenance["selection_failures"]) == 2


def test_explicit_deny_is_strict_and_allow_uses_remaining_priority(monkeypatch):
    calls = unavailable_compiler(monkeypatch)
    with pytest.raises(ValueError, match="fixture performance data"):
        RustForwardPassPerfModel.best_available(request(estimation_mode="op_level"))
    assert calls == ["op_level"]
    calls.clear()
    model = RustForwardPassPerfModel.best_available(
        request(estimation_mode="fpm_interpolation", fallback_policy="allow")
    )
    assert calls == ["fpm", "op_level"]
    assert model.diagnostics()["provenance"]["selected_estimation_mode"] == "fpm_regression"


def test_explicit_regression_has_independent_sampling_and_does_not_compile(monkeypatch):
    calls = unavailable_compiler(monkeypatch)
    model = RustForwardPassPerfModel.best_available(
        request(
            estimation_mode="fpm_regression",
            estimator_config={
                "fpm_regression": {
                    "sampling": {"bins_per_axis": [4, 16], "max_observations": 2},
                    "min_observations": 2,
                },
                "correction": {
                    "sampling": {"max_observations": 32},
                    "min_observations": 8,
                },
            },
        )
    )
    for count in (1, 2, 4, 8):
        metrics = {
            "version": 1,
            "worker_id": "test",
            "dp_rank": 0,
            "counter_id": count,
            "wall_time": 0.001 * count,
            "scheduled_requests": {
                "num_decode_requests": count,
                "sum_decode_kv_tokens": count * 64,
            },
        }
        model.tune_with_fpms(metrics)
    diagnostics = model.diagnostics()
    assert calls == []
    assert diagnostics["retained_observations"] == 2
    resolved = diagnostics["provenance"]["config"]["estimator_config"]
    assert resolved["fpm_regression"]["sampling"]["bins_per_axis"] == [4, 16]
    assert resolved["correction"]["sampling"]["max_observations"] == 32
    assert resolved["correction"]["feature_space"] == "legacy_workload"


def test_invalid_config_never_falls_back_and_nested_errors_have_paths(monkeypatch):
    calls = unavailable_compiler(monkeypatch)
    with pytest.raises(ValueError, match=r"estimator_config\.fpm_regression\.fit"):
        RustForwardPassPerfModel.best_available(request(estimator_config={"fpm_regression": {"fit": {"bogus": 1}}}))
    with pytest.raises(ValueError, match="bins_per_axis"):
        RustForwardPassPerfModel.best_available(
            request(estimator_config={"fpm_regression": {"sampling": {"bins_per_axis": [0, 4]}}})
        )
    assert calls == []


def test_legacy_selection_does_not_become_auto():
    import aiconfigurator_core

    legacy = {
        "schema_version": 1,
        "model_name": "Qwen/Qwen3-32B",
        "system_name": "h200_sxm",
        "backend": "vllm",
        "tp_size": 2,
        "pp_size": 1,
        "forward_model": "fpm",
    }
    config = ForwardPassPerfModelConfig.from_legacy_engine_config(legacy, "decode")
    assert config.estimation_mode == "fpm_interpolation"
    assert config.fallback_policy == "deny"
    assert config.worker_type == "decode"
    normalized = json.loads(aiconfigurator_core.RustForwardPassPerfModel.normalize_config(json.dumps(config.to_dict())))
    assert normalized["tp"] == 2
    assert normalized["estimation_mode"] == "fpm_interpolation"


def test_invalid_quantization_does_not_degrade_to_another_estimator(monkeypatch):
    from aiconfigurator_core.sdk import engine

    original = engine.compile_engine
    calls = []

    def recording(*args, **kwargs):
        calls.append(kwargs["forward_model"])
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "compile_engine", recording)
    with pytest.raises(ValueError, match="definitely_invalid_quant"):
        RustForwardPassPerfModel.best_available(request(gemm_quant_mode="definitely_invalid_quant"))
    assert calls == ["op_level"]


def test_resolver_uses_exact_topology_version_pins_and_isolates_cached_configs(monkeypatch):
    import aiconfigurator_core
    from aisimulate.sweeper.config import SearchSpace
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

    seen = []

    class Model:
        def __init__(self, config):
            seen.append(config)
            self.config = json.loads(
                aiconfigurator_core.RustForwardPassPerfModel.normalize_config(json.dumps(config.to_dict()))
            )
            self.config["backend_version"] = "0.24.0" if config.backend == "vllm" else "0.5.10"
            self.config["estimation_mode"] = "op_level"

        def diagnostics(self):
            return {
                "readiness": "ready",
                "provenance": {"config": self.config, "selected_systems_root": self.config["systems_paths"][0]},
            }

        def close(self):
            pass

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", Model)
    space = SearchSpace(
        model_name="Qwen/Qwen3-32B",
        hardware_sku="h200_sxm",
        backend=["vllm", "sglang"],
        backend_version={"vllm": "current"},
    )
    resolver = ForwardPassEstimatorResolver(space)
    sample = {
        "deployment_mode": "agg",
        "model_name": space.model_name,
        "hardware_sku": space.hardware_sku,
        "backend": "vllm",
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "agg_block_size": 64,
    }
    first = resolver.resolve_candidate(sample)["agg"]
    first.config["estimator_config"]["correction"]["enabled"] = False
    cached = resolver.resolve_candidate(sample)["agg"]
    assert cached.config["estimator_config"]["correction"]["enabled"] is True
    assert len(seen) == 1
    assert seen[0].worker_type == "aggregated"
    assert (seen[0].tp, seen[0].kv_block_size, seen[0].backend_version) == (2, 64, "current")
    resolver.resolve_candidate({**sample, "tp": 4, "backend": "sglang", "agg_block_size": 1})
    assert (seen[-1].tp, seen[-1].kv_block_size, seen[-1].backend_version) == (4, 1, None)


def test_search_rejects_unknown_controls_and_policies_on_custom_timing():
    from aisimulate.sweeper.config import SearchSpace

    common = {"model_name": "m", "hardware_sku": "h200_sxm", "deployment_mode": ["agg"]}
    with pytest.raises(ValueError, match="estimation_mode"):
        SearchSpace(**common, estimation_mode="typo")
    with pytest.raises(ValueError, match="default timing"):
        SearchSpace(
            **common, database_mode="HYBRID", agg_timing_model={"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
        )
    with pytest.raises(ValueError, match="unknown estimator override"):
        SearchSpace(**common, role_estimator_controls={"agg": {"estimation_mod": "op_level"}})


def test_raw_config_rejects_trailing_json():
    import aiconfigurator_core

    payload = json.dumps(request(estimation_mode="fpm_regression").to_dict()) + " {}"
    with pytest.raises(ValueError, match="trailing characters"):
        aiconfigurator_core.RustForwardPassPerfModel.normalize_config(payload)
