# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller transport of canonical spline settings, without constructing a fit."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import yaml

from aisimulate.cli_args import _apply_overrides
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.cli import prediction_mapping
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.forward_pass_estimator import (
    ForwardPassEstimatorResolutionError,
    ForwardPassEstimatorResolver,
)
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import (
    ForwardPassEstimatorSpec,
    ReplaySpec,
    canonical_json,
)
from aisimulate.sweeper.sample import unroll_sample
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FITS = [
    {"kind": "linear"},
    {"kind": "spline"},
    {
        "kind": "spline",
        "spline": {"knots_per_axis": 3, "search": {"kind": "periodic", "step": 17}},
    },
    {
        "kind": "spline",
        "rebuild_interval": None,
        "spline": {
            "knots_per_axis": 2,
            "search": {
                "kind": "adaptive",
                "window": 12,
                "trigger": 4,
                "tolerance": 0.125,
                "absolute_tolerance_ms": 0.5,
                "cooldown": 9,
            },
        },
    },
]


def _engine():
    return {
        "model": "example/model",
        "hardware": "h200_sxm",
        "backend": "vllm",
        "backend_version": "test",
        "mode": "aggregated",
        "context_length": 4096,
        "estimation_mode": "fpm_regression",
        "workers": {
            "aggregated": {"kv_cache": {"capacity": {"type": "fixed", "blocks": 256}}},
        },
    }


@pytest.mark.parametrize("fit", FITS)
def test_spline_cli_yaml_and_replay_preserve_authored_settings(tmp_path, fit):
    raw = {"engine": _engine()}
    encoded_fit = yaml.safe_dump(fit, default_flow_style=True).strip()
    _apply_overrides(
        raw,
        [f"engine.estimator_config.fpm_regression.fit={encoded_fit}"],
        command="predict",
    )
    config = CorePredictionConfig.model_validate(raw)
    expected = {"fpm_regression": {"fit": fit}}
    assert config.engine.estimator_config == expected
    saved = tmp_path / "prediction.yaml"
    saved.write_text(yaml.safe_dump(prediction_mapping(config)))
    restored = CorePredictionConfig.from_yaml(saved)
    deployment = prediction_to_replay_spec(restored).backend_deployment
    request = deployment.agg_engine_args["timing_model"]["config"]
    assert request["estimation_mode"] == "fpm_regression"
    assert request["fallback_policy"] == "deny"
    assert request["worker_type"] == "aggregated"
    assert request["estimator_config"] == expected
    # The caller must not expand Rust defaults or reinterpret the linear alias.
    assert raw["engine"]["estimator_config"] == expected
    serialized = json.loads(
        canonical_json(ReplaySpec(backend_deployment=deployment, workload={}, goal={})),
    )
    saved_engine = serialized["backend_deployment"]["agg_engine_args"]
    assert saved_engine["timing_model"]["config"] == request


def test_spline_role_settings_replace_global_dictionary_without_cross_role_mutation():
    raw = _engine()
    worker = raw["workers"].pop("aggregated")
    raw["mode"] = "disaggregated"
    global_controls = {
        "fpm_regression": {"fit": FITS[0]},
        "correction": {"enabled": False},
    }
    decode_controls = {"fpm_regression": {"fit": FITS[3]}}
    raw["estimator_config"] = deepcopy(global_controls)
    raw["workers"] = {
        "prefill": deepcopy(worker),
        "decode": {
            **deepcopy(worker),
            "timing": {"estimator_config": deepcopy(decode_controls)},
        },
    }
    config = CorePredictionConfig.model_validate({"engine": raw})
    deployment = prediction_to_replay_spec(config).backend_deployment
    prefill = deployment.prefill_engine_args["timing_model"]["config"]
    decode = deployment.decode_engine_args["timing_model"]["config"]
    assert prefill["estimator_config"] == global_controls
    assert decode["estimator_config"] == decode_controls
    assert (prefill["worker_type"], decode["worker_type"]) == ("prefill", "decode")
    search = decode["estimator_config"]["fpm_regression"]["fit"]["spline"]["search"]
    search["trigger"] = 9
    assert config.engine.workers.decode.timing.estimator_config == decode_controls
    assert config.engine.estimator_config == global_controls


@pytest.mark.parametrize("fit", FITS[1:])
def test_spline_recommendation_candidate_preserves_resolved_settings(tmp_path, fit):
    source = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "estimator_config": {"fpm_regression": {"fit": deepcopy(fit)}},
            },
            "optimization": {"constraints": {"max_candidate_gpus": 1}},
        }
    )
    space = recommendation_to_sweeper(source).search_space
    sample = unroll_sample(
        search_space=space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        },
        parallel_config=ReplicaParallelConfig(
            ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1),
            replicas=1,
        ),
    )
    request = ForwardPassEstimatorResolver(space)._request(sample, "agg")
    expected = {"fpm_regression": {"fit": fit}}
    assert request.estimator_config == expected
    # Isolate serialization from the intentional rejection of untrained offline fits.
    resolved = ForwardPassEstimatorSpec(
        config=replace(request, backend_version="test").to_dict(),
    )
    deployment = build_backend_deployment(
        sample,
        backend_version="test",
        forward_pass_estimators={"agg": resolved},
    )
    prediction = _candidate_prediction(
        source,
        sample,
        ReplaySpec(backend_deployment=deployment, workload={}, goal={}),
        adapter_sections={},
    )
    saved = tmp_path / "candidate.yaml"
    saved.write_text(yaml.safe_dump(prediction))
    restored = CorePredictionConfig.from_yaml(saved)
    reloaded = prediction_to_replay_spec(restored).backend_deployment
    timing = reloaded.agg_engine_args["timing_model"]["config"]
    assert timing["estimator_config"] == expected
    metadata = deployment.performance_model_metadata["aggregated"]["config"]
    assert metadata["estimator_config"] == expected


def test_spline_settings_are_part_of_resolver_cache_identity(monkeypatch):
    calls = []

    class ReadyModel:
        def __init__(self, request):
            calls.append(request.to_dict())
            self.config = {**request.to_dict(), "backend_version": "test"}

        def diagnostics(self):
            return {"readiness": "ready", "provenance": {"config": self.config}}

        def close(self):
            pass

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", ReadyModel)
    resolver = ForwardPassEstimatorResolver(
        SearchSpace(model_name="m", hardware_sku="h200_sxm"),
    )
    request = ForwardPassPerfModelConfig(
        model="m",
        system="h200_sxm",
        backend="vllm",
        worker_type="decode",
        estimation_mode="fpm_regression",
    )
    for fit in FITS:
        configured = replace(
            request,
            estimator_config={"fpm_regression": {"fit": deepcopy(fit)}},
        )
        first = resolver._resolve(configured, "decode")
        first.config["estimator_config"].clear()
        cached = resolver._resolve(configured, "decode")
        assert cached.config["estimator_config"] == {"fpm_regression": {"fit": fit}}
    assert len(calls) == len(FITS)


def test_spline_selection_does_not_bypass_offline_readiness(monkeypatch):
    class ColdModel:
        def diagnostics(self):
            return {"readiness": "insufficient_data", "provenance": {"config": {}}}

        def close(self):
            pass

    monkeypatch.setattr(
        RustForwardPassPerfModel,
        "best_available",
        lambda request: ColdModel(),
    )
    resolver = ForwardPassEstimatorResolver(
        SearchSpace(model_name="m", hardware_sku="h200_sxm"),
    )
    request = ForwardPassPerfModelConfig(
        model="m",
        system="h200_sxm",
        backend="vllm",
        worker_type="decode",
        estimation_mode="fpm_regression",
        estimator_config={"fpm_regression": {"fit": {"kind": "spline"}}},
    )
    with pytest.raises(
        ForwardPassEstimatorResolutionError,
        match="requires training observations",
    ):
        resolver._resolve(request, "decode")
