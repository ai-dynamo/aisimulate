# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real C-store admission and interpolation/online competition through the SDK."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy

import pytest

from aisimulate_core.sdk import fpm_learned
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from . import test_fpm_mixed_composition_integration as shared_fixtures
from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies
from .test_fpm_mixed_composition_integration import (
    _config,
    _iteration,
)

pytestmark = pytest.mark.unit
MIXED = "contains_locally_mixed"
mixed_seed = shared_fixtures.mixed_seed
phase_roots = shared_fixtures.phase_roots


@pytest.fixture
def c_checkpoint(mixed_seed, tmp_path):
    config, _, _ = mixed_seed
    rows = [_iteration(wall_ms=12.0, index=index) for index in range(20)]
    checkpoint = tmp_path / "mixed-c-seed"
    artifact = fpm_learned.train(
        rows,
        "aggregated",
        features=fpm_learned.FEATURE_PRESETS["fpm5"],
        max_iter=3,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        min_store_rows=20,
        online_checkpoint_dir=checkpoint,
        identity=config,
    )
    assert set(artifact["stores"]) == {MIXED}
    return checkpoint, rows


def _selector_config(mixed_seed, *, enabled=True):
    config = _config(mixed_seed, enabled=enabled, adaptation="causal_selector")
    config["estimator_config"]["fpm_learned"].update(
        update_every=1, history_weight=0.25, history_weight_ramp_observations=8
    )
    config["estimator_config"]["selector"] = {"alpha": 1.0, "warmup": 3, "margin": 0.05, "persistence": 2}
    return config


def _state(model, kind=MIXED, query_type="mixed"):
    return next(
        state
        for state in model.learned_selector_diagnostics()["states"]
        if state["store"] == kind and state["query_type"] == query_type
    )


def _all_state(model):
    return (
        model.learned_expert_artifacts(),
        model.learned_selector_diagnostics(),
        model.learned_online_diagnostics(),
        model.mixed_composition_state(),
    )


def test_mixed_selector_bootstrap_preserves_pure_state_and_uses_only_future_paired_losses(mixed_seed, c_checkpoint):
    checkpoint, bootstrap = c_checkpoint
    selector_config = _selector_config(mixed_seed)
    model = RustForwardPassPerfModel.best_available(selector_config)
    pure_control = RustForwardPassPerfModel.best_available(_selector_config(mixed_seed, enabled=False))
    prior_config = _config(mixed_seed, adaptation="frozen")
    # Interpolation preparation strips inactive selector controls.
    prior = RustForwardPassPerfModel.best_available(prior_config)
    online_config = _config(mixed_seed, checkpoint, enabled=False)
    # Compare the same fitting recipe, independently of adaptation defaults.
    online_config["estimator_config"]["fpm_learned"].update(
        update_every=1, history_weight=0.25, history_weight_ramp_observations=8
    )
    online = RustForwardPassPerfModel.best_available(online_config)
    try:
        # Existing pure selector evidence and pending/learned state predate the
        # mixed addition and must survive it without a reset or reconstruction.
        for index in range(6):
            pure = _iteration("decode", wall_ms=7.0, index=index)
            for current in (model, pure_control):
                current.tune_with_fpms([pure])
        pure_before = _state(model, "pure_decode", "decode")
        initial_experts = model.learned_expert_artifacts()
        for index, observation in enumerate(bootstrap):
            assert model.estimate_forward_pass_time_ms(observation) == prior.estimate_forward_pass_time_ms(observation)
            assert _state(model)["observations"] == 0
            assert MIXED not in model.learned_expert_artifacts()["online"]["stores"]
            for current in (model, prior):
                current.tune_with_fpms([observation])
            assert model.mixed_composition_route(observation)["calibration_admissions"] == index + 1

        # The hand-derived 4ms prior is capped at factor 2 after these 12ms labels.
        query = _iteration()
        assert model.estimate_forward_pass_time_ms(query) == 8.0
        calibration = model.mixed_composition_state()
        assert model.append_mixed_learned_checkpoint(checkpoint) == 1
        assert model.mixed_composition_state() == calibration
        assert _state(model, "pure_decode", "decode") == pure_before
        experts = model.learned_expert_artifacts()
        assert experts["stable"] == initial_experts["stable"]
        assert MIXED not in experts["stable"]["stores"]
        assert experts["online"]["stores"][MIXED] == online.export_learned_artifact()["stores"][MIXED]
        for kind in ("pure_decode", "pure_prefill"):
            assert experts["online"]["stores"][kind] == initial_experts["online"]["stores"][kind]
        assert model.learned_online_diagnostics()["stores"][MIXED]["accepted_observations"] == 0
        assert _state(model)["observations"] == 0
        assert _state(model)["selected"] == "stable"
        assert _state(model)["stable_source"] == "interpolation"

        for admitted in range(1, 9):
            observation = _iteration(wall_ms=12.0, index=20 + admitted)
            expected_expert = "stable" if admitted <= 4 else "online"
            expected = prior if expected_expert == "stable" else online
            before = _all_state(model)
            result = model.estimate_forward_pass_with_route(observation)
            assert result["latency_ms"] == expected.estimate_forward_pass_time_ms(observation)
            route = result["temporary_mixed_composition"]
            assert route["selected_expert"] == route["preferred_expert"] == expected_expert
            assert route["online_ready"]
            assert route["paired_comparisons"] == admitted - 1
            assert route["warmup_complete"] == (admitted > 3)
            if expected_expert == "stable":
                assert route["stable_available"] is True
                assert route["online_available"] is None
            else:
                assert route["stable_available"] is None
                assert route["online_available"] is True
            hidden_target = deepcopy(observation)
            hidden_target[0]["wall_time"] = 123.0
            assert model.estimate_forward_pass_with_route(hidden_target) == result
            assert _all_state(model) == before
            for current in (model, prior, online):
                current.tune_with_fpms([observation])
            state = _state(model)
            assert state["observations"] == admitted
            assert state["stable_ewma"] == pytest.approx(1 / 3)
            assert state["online_ewma"] == pytest.approx(0.0, abs=1e-14)
            assert state["selected"] == ("stable" if admitted < 4 else "online")
            assert (
                model.learned_expert_artifacts()["online"]["stores"][MIXED]
                == online.export_learned_artifact()["stores"][MIXED]
            )
            assert model.mixed_composition_state() == prior.mixed_composition_state()
        assert _state(model)["switches"] == 1
        assert _state(model, "pure_decode", "decode") == pure_before
        pure = _iteration("decode", wall_ms=8.0)
        assert model.estimate_forward_pass_time_ms(pure) == pure_control.estimate_forward_pass_time_ms(pure)
        for current in (model, pure_control):
            current.tune_with_fpms([pure])
        assert _state(model, "pure_decode", "decode") == _state(pure_control, "pure_decode", "decode")
    finally:
        for current in (model, pure_control, prior, online):
            current.close()


def test_ready_mixed_online_expert_covers_prior_miss_without_inventing_paired_evidence(mixed_seed, c_checkpoint):
    checkpoint, _ = c_checkpoint
    model = RustForwardPassPerfModel.best_available(_selector_config(mixed_seed))
    prior = RustForwardPassPerfModel.best_available(_config(mixed_seed, adaptation="frozen"))
    try:
        query = _iteration(wall_ms=12.0)
        query[0]["scheduled_requests"].update(num_decode_requests=8, sum_decode_kv_tokens=256)
        assert prior.estimate_forward_pass_time_ms(query) is None
        assert model.estimate_forward_pass_time_ms(query) is None
        assert model.append_mixed_learned_checkpoint(checkpoint) == 1
        before = model.mixed_composition_state()
        result = model.estimate_forward_pass_with_route(query)
        assert result["latency_ms"] == pytest.approx(12.0)
        route = result["temporary_mixed_composition"]
        assert route["selected_expert"] == "online" and route["preferred_expert"] == "stable"
        assert route["stable_available"] is False and route["online_available"] is True
        assert route["online_ready"]
        model.tune_with_fpms([query])
        assert model.learned_online_diagnostics()["stores"][MIXED]["accepted_observations"] == 1
        assert _state(model)["observations"] == 0
        assert _state(model)["selected"] == "stable"
        assert model.mixed_composition_state() == before
    finally:
        model.close()
        prior.close()


@pytest.mark.parametrize("adaptation", ["online", "causal_selector"])
@pytest.mark.parametrize("group_history", [False, True])
@pytest.mark.parametrize("refresh", ["eager", "lazy"])
def test_mixed_append_preserves_recent_feedback_ages(mixed_seed, tmp_path, adaptation, group_history, refresh):
    identity, _, _ = mixed_seed
    checkpoint = tmp_path / "sparse-c-seed"
    # Training IDs are positions in the source, not accepted online labels.
    # Skipped inputs give these 20 anchors IDs 1000..1019.
    rows = [[] for _ in range(1000)] + [_iteration(wall_ms=12.0, index=i) for i in range(20)]
    fpm_learned.train(
        rows,
        "aggregated",
        features=fpm_learned.FEATURE_PRESETS["fpm5"],
        max_iter=3,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        min_store_rows=20,
        online_checkpoint_dir=checkpoint,
        identity=identity,
    )
    config = _selector_config(mixed_seed)
    estimator = config["estimator_config"]
    estimator["fpm_learned"].update(
        adaptation=adaptation,
        update_every=2,
        recent_max_age_observations=3,
        history_bin_grouping=group_history,
        derivative_refresh=refresh,
    )
    if adaptation == "online":
        estimator.pop("selector")
        estimator["correction"]["enabled"] = False
    control_config = deepcopy(config)
    control_config["estimator_config"].pop("temporary_mixed_composition")
    model = RustForwardPassPerfModel.best_available(config)
    control = RustForwardPassPerfModel.best_available(control_config)
    resumed = None
    try:
        for index, wall_ms in enumerate([7.0, 9.0, 11.0]):
            for current in (model, control):
                current.tune_with_fpms([_iteration("decode", wall_ms=wall_ms, index=index)])
        pure = "pure_decode"
        before = model.learned_online_diagnostics()["stores"][pure]
        assert before["recent_rows"] == 2 and before["pending_rows"] == 1
        assert model.append_mixed_learned_checkpoint(checkpoint) == 1
        if adaptation == "online":
            # The shifted clock and IDs must also survive serialized continuation.
            saved = tmp_path / "appended-runtime"
            model.save_online_checkpoint(saved)
            resume_config = deepcopy(config)
            resume_config["estimator_config"]["fpm_learned"]["checkpoint_path"] = str(saved)
            resumed = RustForwardPassPerfModel.best_available(resume_config)
        candidates = [model] + ([resumed] if resumed else [])
        for current in [control, *candidates]:
            current.flush_updates()
        assert control.learned_online_diagnostics()["stores"][pure]["recent_rows"] == 3

        def assert_pure_matches_control():
            expected = control.learned_online_diagnostics()["stores"][pure]
            for candidate in candidates:
                actual = candidate.learned_online_diagnostics()["stores"][pure]
                for field in ("recent_rows", "pending_rows", "accepted_observations", "evicted_observations"):
                    assert actual[field] == expected[field]
                query = _iteration("decode")
                assert candidate.estimate_forward_pass_time_ms(query) == control.estimate_forward_pass_time_ms(query)
                if adaptation == "online":
                    assert (
                        candidate.export_learned_artifact()["stores"][pure]
                        == control.export_learned_artifact()["stores"][pure]
                    )
                else:
                    assert (
                        candidate.learned_expert_artifacts()["online"]["stores"][pure]
                        == control.learned_expert_artifacts()["online"]["stores"][pure]
                    )

        assert_pure_matches_control()
        # Continue past the age boundary: translation must not postpone expiry.
        for index in range(6):
            for current in [control, *candidates]:
                current.tune_with_fpms([_iteration("decode", wall_ms=13.0 + index, index=3 + index)])
                current.flush_updates()
            assert_pure_matches_control()
        assert control.learned_online_diagnostics()["stores"][pure]["evicted_observations"] == 6
    finally:
        for current in (model, control, resumed):
            if current is not None:
                current.close()


def test_mixed_append_and_legacy_snapshot_fail_atomically_and_selector_checkpoint_remains_unsupported(
    mixed_seed, c_checkpoint, tmp_path
):
    checkpoint, bootstrap = c_checkpoint
    model = RustForwardPassPerfModel.best_available(_selector_config(mixed_seed))
    try:
        model.tune_with_fpms(bootstrap)
        wrong_identity = tmp_path / "wrong-c-identity"
        shutil.copytree(checkpoint, wrong_identity)
        manifest_path = wrong_identity / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["model"] = "wrong/deployment"
        manifest_path.write_text(json.dumps(manifest))
        before = _all_state(model)
        with pytest.raises(ValueError, match="identity"):
            model.append_mixed_learned_checkpoint(wrong_identity)
        assert _all_state(model) == before
        assert model.append_mixed_learned_checkpoint(checkpoint) == 1
        after = _all_state(model)
        with pytest.raises(ValueError):
            model.append_mixed_learned_checkpoint(checkpoint)
        assert _all_state(model) == after
        legacy = deepcopy(model.mixed_composition_state())
        legacy["version"] = "original_a_availability_priority_attention_moe_v2"
        with pytest.raises(ValueError, match="saved mixed state does not match"):
            model.restore_mixed_composition_state(legacy)
        assert _all_state(model) == after
        destination = tmp_path / "unsupported-selector-resume"
        with pytest.raises(ValueError, match="selector"):
            model.save_online_checkpoint(destination)
        assert not destination.exists()
        assert _all_state(model) == after
    finally:
        model.close()
