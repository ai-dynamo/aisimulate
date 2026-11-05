# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym transport integration with the unchanged caller-managed first fitting and native updates."""

from functools import wraps

import pytest
from test_learned_scenario import observation

from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor


@pytest.fixture
def native_backend():
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, fpm_learned

    request = ForwardPassPerfModelConfig(
        model="Qwen/Qwen3-8B", system="h200_sxm", worker_type="aggregated", backend="vllm", backend_version="0.25.1"
    )
    return RustForwardPassPerfModel, request, fpm_learned


def test_gym_caller_first_fit_threshold_and_worker_isolation(native_backend, monkeypatch):
    native, request, trainer = native_backend

    calls = []
    original = trainer.train

    @wraps(original)
    def train(*args, **kwargs):
        calls.append(kwargs["online_checkpoint_dir"])
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "train", train)
    model = LearnedScenarioPredictor("gbdt", native, request, trainer, online=True)
    try:
        for index in range(20):
            row = observation(index, actual=12)
            assert model.predict(row.prediction_input()).value_ms is None
            model.tune([row])
        assert model.predict(observation(20).prediction_input()).value_ms == pytest.approx(12)
        assert model.predict(observation(20, worker="other").prediction_input()).value_ms is None
        assert model.predict(observation(20, kind="prefill").prediction_input()).value_ms is None
        scope = model.diagnostics()["scopes"][0]
        assert scope["ready"] and scope["bootstrap_observations"] == 20
        assert scope["feedback_observations"] == 0
        assert len(calls) == len(model.seed_checkpoint_paths()) == 1
        assert len(model.seed_artifact_paths()) == 1
        model.tune([observation(20, actual=12)])
        assert model.diagnostics()["scopes"][0]["feedback_observations"] == 1
    finally:
        model.close()


@pytest.mark.parametrize("offline_count", [18, 20])
def test_native_gym_offline_b_lineage_and_legacy_seed(native_backend, offline_count):
    native, request, trainer = native_backend
    model = LearnedScenarioPredictor(
        "gbdt", native, request, trainer, [observation(index, actual=12) for index in range(offline_count)], online=True
    )
    try:
        if offline_count < 20:
            assert model.predict(observation(18).prediction_input()).value_ms is None
            model.tune([observation(18, actual=12)])
            assert model.predict(observation(19).prediction_input()).value_ms is None
            model.tune([observation(19, actual=12)])
        else:
            config = next(iter(model.seed_configurations().values()))
            assert "checkpoint_path" in config["estimator_config"]["fpm_learned"]
        assert model.predict(observation(20).prediction_input()).value_ms == pytest.approx(12)
        scope = model.diagnostics()["scopes"][0]
        assert scope["offline_observations"] == offline_count
        assert scope["bootstrap_observations"] == 20 - offline_count
        assert len(model.seed_checkpoint_paths()) == 1
    finally:
        model.close()
