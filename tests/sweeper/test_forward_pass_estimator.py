# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical construction through the real Python/Rust boundary."""

import json
from pathlib import Path

import pytest

from aiconfigurator_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture
def custom_systems(tmp_path):
    from importlib.resources import files

    packaged = Path(str(files("aiconfigurator_core") / "systems"))
    for entry in packaged.iterdir():
        (tmp_path / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (tmp_path / "review_h200.yaml").write_text((packaged / "h200_sxm.yaml").read_text())
    return tmp_path


@pytest.mark.parametrize("discovery", ["sdk", "ordered_sdk", "environment"])
def test_omitted_roots_preserve_configured_discovery(custom_systems, monkeypatch, discovery):
    import aiconfigurator_core
    from aiconfigurator_core.sdk import perf_database
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.config import SmartSearchConfig
    from aisimulate.sweeper.search import Sweeper

    monkeypatch.setattr(perf_database, "_SYSTEMS_PATHS", perf_database.get_systems_paths())
    monkeypatch.delenv("AICONFIGURATOR_SYSTEMS_PATH", raising=False)
    roots = [str(custom_systems)]
    if discovery == "ordered_sdk":
        empty = custom_systems / "empty"
        empty.mkdir()
        roots.insert(0, str(empty))
    if discovery in {"sdk", "ordered_sdk"}:
        perf_database.set_systems_paths(roots)
    else:
        perf_database.set_systems_paths("default")
        monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(custom_systems))
    cfg = SmartSearchConfig(
        search_space={
            "model_name": "Qwen/Qwen3-32B",
            "hardware_sku": "review_h200",
            "backend": ["vllm"],
            "backend_version": "0.24.0",
            "deployment_mode": ["agg"],
            "gpu_budget": 2,
            "context_length": 4096,
            "parallel_configs": [{"tp": 2, "replicas": 1}],
            "agg_max_num_batched_tokens": [8192],
            "agg_max_num_seqs": [256],
            "agg_num_gpu_blocks": 256,
            "agg_block_size": 64,
        },
        workload={"isl": 128, "osl": 2, "request_count": 1, "concurrency": 1},
        sweep={"max_rounds": 1, "candidates_per_round": 1, "parallel_evals": 1},
    )
    assert cfg.search_space.systems_paths is None
    cfg = SmartSearchConfig.model_validate_json(cfg.model_dump_json())
    result = Sweeper(runner_factory=EngineReplayRunnerFactory(), show_progress=False).run(cfg, top_n=None)
    assert result.model_dump(mode="json")["counts"]["feasible"] == 1
    raw = aiconfigurator_core.RustForwardPassPerfModel.best_available(
        json.dumps(
            {
                "model": "Qwen/Qwen3-32B",
                "system": "review_h200",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "worker_type": "aggregated",
                "tp": 2,
            }
        )
    )
    assert json.loads(raw.diagnostics())["provenance"]["config"]["systems_paths"] == [str(custom_systems)]
    from aisimulate.sweeper.forward_pass_estimator import resolve_systems_paths

    assert resolve_systems_paths(None) == tuple(roots)
    assert resolve_systems_paths(["default"]) != (str(custom_systems),)


def test_prediction_pins_populated_version_slots(custom_systems):
    from dataclasses import replace

    from aiconfigurator_core.sdk.perf_database import resolve_query_version
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory

    next_version = resolve_query_version("review_h200", "vllm", "next", systems_paths=[str(custom_systems)])
    for requested, literal in [("current", "0.24.0"), ("next", next_version), ("0.24.0", "0.24.0")]:
        cfg = CorePredictionConfig.model_validate(
            {
                "engine": {
                    "model": "Qwen/Qwen3-32B",
                    "hardware": "review_h200",
                    "backend": "vllm",
                    "backend_version": requested,
                    "context_length": 4096,
                    "systems_paths": [str(custom_systems)],
                    "workers": {
                        "aggregated": {
                            "parallelism": {"tensor": 2},
                            "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                        }
                    },
                },
                "traffic": {
                    "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 2},
                    "load": {"type": "concurrency", "concurrency": 1},
                    "stop": {"requests": 1},
                },
            }
        )
        replay = prediction_to_replay_spec(cfg)
        assert replay.backend_deployment.backend_version == literal
        runner = EngineReplayRunnerFactory().create(0)
        try:
            assert runner.run(replay).metrics["completed_requests"] == 1
            if requested == "current":
                bad = replace(replay, backend_deployment=replace(replay.backend_deployment, backend_version="0.19.0"))
                with pytest.raises(ValueError, match="conflicts with"):
                    runner.run(bad)
        finally:
            runner.close()


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("ratio", [0.5, 1.0])
def test_kv_relative_load_uses_resolved_roots(custom_systems, mode, ratio):
    from dataclasses import asdict

    from aisimulate.sweeper.config import SearchSpace, Workload
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver
    from aisimulate.sweeper.kv_load import resolve_kv_load
    from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
    from aisimulate.sweeper.sample import unroll_sample

    role = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    parallel = role if mode == "agg" else DisaggParallelConfig(prefill=role, decode=role)
    space = SearchSpace(
        model_name="Qwen/Qwen3-32B",
        hardware_sku="review_h200",
        backend=["vllm"],
        deployment_mode=[mode],
        systems_paths=[str(custom_systems)],
        gpu_budget=4,
    )
    roles = ("agg",) if mode == "agg" else ("prefill", "decode")
    selection = {"deployment_mode": mode, "backend": "vllm"}
    for name in roles:
        selection.update({f"{name}_max_num_batched_tokens": 8192, f"{name}_max_num_seqs": 256})
    sample = unroll_sample(search_space=space, selection=selection, parallel_config=parallel)
    estimators = ForwardPassEstimatorResolver(space).resolve_candidate(sample)
    sample["forward_pass_estimators"] = {name: asdict(estimator) for name, estimator in estimators.items()}
    result = resolve_kv_load(
        sample,
        workload=Workload(isl=128, osl=2, kv_load_ratio=ratio, request_count=1),
        parallel_config=parallel,
        ratio=ratio,
        backend_version="0.24.0",
    )
    assert result.concurrency == max(1, int(ratio * result.concurrency_capacity))
    assert set(result.role_capacity_tokens) == set(roles)
    assert all(tokens > 0 for tokens in result.role_capacity_tokens.values())


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
    with pytest.raises(ValueError, match="estimator settings require default timing"):
        SearchSpace(
            **common,
            agg_timing_model={"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
            role_estimator_controls={"agg": {"estimation_mode": "fpm_regression"}},
        )


def test_raw_config_rejects_trailing_json():
    import aiconfigurator_core

    payload = json.dumps(request(estimation_mode="fpm_regression").to_dict()) + " {}"
    with pytest.raises(ValueError, match="trailing characters"):
        aiconfigurator_core.RustForwardPassPerfModel.normalize_config(payload)


@pytest.mark.parametrize(
    "selection",
    [{}, {"agg_forward_model": "fpm"}, {"estimation_mode": "fpm_regression"}],
)
def test_saved_search_preserves_estimator_selection(selection):
    from aisimulate.sweeper.config import SearchSpace
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

    space = SearchSpace(model_name="Qwen/Qwen3-32B", hardware_sku="h200_sxm", **selection)
    sample = {
        "backend": "vllm",
        "hardware_sku": "h200_sxm",
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "agg_block_size": 64,
    }
    expected = "fpm_interpolation" if "agg_forward_model" in selection else selection.get("estimation_mode", "auto")
    for _ in range(2):
        resolved = ForwardPassEstimatorResolver(space)._request(sample, "agg")
        assert resolved.estimation_mode == expected
        assert resolved.fallback_policy == "deny"
        serialized = space.model_dump_json()
        assert "agg_forward_model" not in json.loads(serialized)
        space = SearchSpace.model_validate_json(serialized)
    assert space.agg_forward_model == ("fpm" if expected == "fpm_interpolation" else "op_level")


def test_mixed_timing_still_enforces_cold_regression_on_default_role():
    from aisimulate.config.cli import CoreRecommendationConfig
    from aisimulate.recommend import recommendation_to_sweeper
    from aisimulate.sweeper.forward_pass_estimator import (
        ForwardPassEstimatorResolutionError,
        ForwardPassEstimatorResolver,
    )

    public = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "model": "Qwen/Qwen3-32B",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "mode": "disaggregated",
                "context_length": 4096,
                "workers": {
                    "prefill": {
                        "timing": {
                            "estimation_mode": "fpm_regression",
                            "fallback_policy": "deny",
                        }
                    },
                    "decode": {"timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}},
                },
            },
            "optimization": {
                "target": "throughput",
                "constraints": {"max_candidate_gpus": 4},
            },
        }
    )
    space = recommendation_to_sweeper(public).search_space
    sample = {
        "deployment_mode": "disagg",
        "backend": "vllm",
        "hardware_sku": "h200_sxm",
        "prefill_tp": 2,
        "prefill_pp": 1,
        "prefill_attention_dp": 1,
        "prefill_moe_tp": 1,
        "prefill_moe_ep": 1,
        "prefill_block_size": 64,
        "prefill_timing_model": space.prefill_timing_model,
        "decode_timing_model": space.decode_timing_model,
    }
    with pytest.raises(ForwardPassEstimatorResolutionError, match="prefill is not ready"):
        ForwardPassEstimatorResolver(space).resolve_candidate(sample)
