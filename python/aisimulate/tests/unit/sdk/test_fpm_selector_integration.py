# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the combined native model with two independent canonical experts."""

from __future__ import annotations

import copy

import pytest

from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies
from .test_fpm_online_integration import IDENTITY, iteration, train

pytestmark = pytest.mark.unit


def test_selector_defaults_match_explicit_gym_recipe(tmp_path):
    _, seed, _ = train(tmp_path)
    config = {
        **IDENTITY,
        "estimation_mode": "fpm_learned",
        "estimator_config": {"fpm_learned": {"checkpoint_path": str(seed), "adaptation": "causal_selector"}},
    }
    explicit = copy.deepcopy(config)
    explicit["estimator_config"]["fpm_learned"].update(
        recent_window=8,
        recent_sampling="fifo",
        history_bin_grouping=True,
        history_weight=0.25,
        history_weight_ramp_observations=8,
        update_every=4,
        update_mode="structural",
        update_step=1.0,
        derivative_refresh="lazy",
        eager_refresh_interval=32,
        split_sampling_rate=0.5,
        split_rank_tolerance=0.01,
    )
    explicit["estimator_config"]["correction"] = {
        "enabled": True,
        "mode": "hybrid",
        "min_observations": 5,
        "sampling": {"max_observations": 64, "bins_per_axis": [4, 4]},
        "factor_bounds": {"min": 0.5, "max": 2.0},
    }
    explicit["estimator_config"]["selector"] = {"alpha": 0.4, "warmup": 5, "margin": 0.05, "persistence": 1}
    assert RustForwardPassPerfModel.normalize_config(config) == RustForwardPassPerfModel.normalize_config(explicit)
    models = [RustForwardPassPerfModel.best_available(settings) for settings in (config, explicit)]
    for index in range(39):
        row = iteration(index, shifted=index < 20)
        assert models[0].estimate_forward_pass_time_ms(row) == models[1].estimate_forward_pass_time_ms(row)
        for model in models:
            model.tune_with_fpms([row])
        assert models[0].learned_selector_diagnostics() == models[1].learned_selector_diagnostics()
    state = models[0].learned_online_diagnostics()["stores"]["pure_decode"]
    assert state["accepted_observations"] == 39
    assert state["recent_rows"] == 8
    assert state["pending_rows"] == 3
    assert state["publications"] == 9


def test_selector_explicit_standalone_values_and_partial_calibration_round_trip(tmp_path):
    _, seed, _ = train(tmp_path)
    overrides = {
        "checkpoint_path": str(seed),
        "adaptation": "causal_selector",
        # Each differs from the selector recipe but equals a standalone default.
        "recent_window": 64,
        "history_weight": 0.1,
        "history_weight_ramp_observations": 32,
        "update_every": 1,
        "recent_sampling": "fifo",
    }
    config = {
        **IDENTITY,
        "estimation_mode": "fpm_learned",
        "estimator_config": {
            "fpm_learned": overrides,
            "correction": {"sampling": {"max_observations": 32}},
            "selector": {"warmup": 3, "margin": 0},
        },
    }
    saved = RustForwardPassPerfModel.normalize_config(config)
    controls = saved["estimator_config"]
    assert all(controls["fpm_learned"][key] == value for key, value in overrides.items())
    assert controls["selector"] == {"alpha": 0.4, "warmup": 3, "margin": 0, "persistence": 1}
    assert controls["correction"]["enabled"] is True
    assert controls["correction"]["sampling"]["max_observations"] == 32
    assert controls["correction"]["sampling"]["bins_per_axis"] == [4, 4]
    assert saved == RustForwardPassPerfModel.normalize_config(saved)
    assert RustForwardPassPerfModel.best_available(saved).diagnostics()["source"] == "learned_causal_selector"


@pytest.mark.parametrize("persistence", [1, 3])
def test_selector_matches_independent_experts_and_causal_recurrence(tmp_path, persistence):
    _, seed, artifact = train(tmp_path)
    correction = {
        "enabled": True,
        "mode": "hybrid",
        "sampling": {"max_observations": 32, "bins_per_axis": [2, 2]},
    }
    selector = {"alpha": 0.5, "warmup": 3, "margin": 0.025, "persistence": persistence}

    def load(adaptation):
        return RustForwardPassPerfModel.best_available(
            {
                **IDENTITY,
                "estimation_mode": "fpm_learned",
                "estimator_config": {
                    "correction": {"enabled": False} if adaptation == "online" else correction,
                    "selector": selector if adaptation == "causal_selector" else {},
                    "fpm_learned": {
                        "checkpoint_path": str(seed),
                        "adaptation": adaptation,
                        "update_mode": "leaf_only",
                        "recent_window": 8,
                        "update_every": 1,
                        "history_weight": 0.01,
                        # Compare identical experts, independent of mode defaults.
                        "history_weight_ramp_observations": 0,
                    },
                },
            }
        )

    stable, online, combined = [load(mode) for mode in ("frozen", "online", "causal_selector")]
    assert stable.diagnostics()["source"] == "learned"
    assert online.diagnostics()["source"] == "learned_online"
    assert combined.diagnostics()["source"] == "learned_causal_selector"
    assert stable.learned_selector_diagnostics() is None
    assert online.learned_selector_diagnostics() is None
    active, streak, switches = 0, 0, 0
    errors = None
    for n in range(1, 97):
        row = iteration(n % 16)
        # Fixed-grid positions, changing label regimes. The current label is
        # never used for selection or prediction until tune_with_fpms admits it.
        for rank in row:
            rank["wall_time"] *= 1.8 if n <= 48 else 0.7
        candidates = [model.estimate_forward_pass_time_ms(row) for model in (stable, online)]
        before = combined.learned_selector_diagnostics()
        assert combined.estimate_forward_pass_time_ms(row) == pytest.approx(candidates[active], rel=1e-13)
        hidden_label = copy.deepcopy(row)
        for rank in hidden_label:
            rank["wall_time"] = 123.0
        assert combined.estimate_forward_pass_time_ms(hidden_label) == combined.estimate_forward_pass_time_ms(row)
        assert combined.learned_selector_diagnostics() == before

        y = max(rank["wall_time"] for rank in row) * 1000
        losses = [abs(value / y - 1) for value in candidates]
        errors = (
            losses
            if errors is None
            else [
                (1 - selector["alpha"]) * old + selector["alpha"] * loss
                for old, loss in zip(errors, losses, strict=True)
            ]
        )
        preferred = n >= selector["warmup"] and errors[1 - active] < (1 - selector["margin"]) * errors[active]
        streak = streak + 1 if preferred else 0
        if streak >= persistence:
            active, streak, switches = 1 - active, 0, switches + 1
        for model in (stable, online, combined):
            model.tune_with_fpms([row])
        # The model source identifies its mode even when the selected expert changes.
        assert online.diagnostics()["source"] == "learned_online"
        assert combined.diagnostics()["source"] == "learned_causal_selector"
        diagnostics = combined.learned_selector_diagnostics()
        state = next(s for s in diagnostics["states"] if s["store"] == "pure_decode" and s["query_type"] == "decode")
        assert state["observations"] == n
        assert state["selected"] == ("stable", "online")[active]
        assert state["stable_ewma"] == pytest.approx(errors[0], rel=1e-13)
        assert state["online_ewma"] == pytest.approx(errors[1], rel=1e-13)
        assert state["streak"] == streak
        assert state["switches"] == switches
        assert all(s["observations"] == 0 for s in diagnostics["states"] if s is not state)
    assert switches > 0, "The reference trajectory must exercise a real branch switch."
    combined.flush_updates()
    assert combined.diagnostics()["source"] == "learned_causal_selector"
    assert combined.learned_selector_diagnostics() == diagnostics
    assert stable.export_learned_artifact() == artifact
    assert combined.learned_online_diagnostics()["stores"]["pure_decode"]["accepted_observations"] == 96
    with pytest.raises(ValueError, match="selector"):
        combined.export_learned_artifact()
    destination = tmp_path / "unsupported-resume"
    with pytest.raises(ValueError, match="selector"):
        combined.save_online_checkpoint(destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "mode,adaptation",
    [
        ("auto", "frozen"),
        ("op_level", "frozen"),
        ("fpm_interpolation", "frozen"),
        ("fpm_regression", "frozen"),
        ("fpm_learned", "frozen"),
        ("fpm_learned", "online"),
    ],
)
def test_inactive_selector_controls_reject_before_source_loading(tmp_path, mode, adaptation):
    settings = {"selector": {"alpha": 0.2}, "correction": {"enabled": False}}
    if mode == "fpm_learned":
        settings["fpm_learned"] = {
            "adaptation": adaptation,
            "checkpoint_path": str(tmp_path / "missing-checkpoint"),
        }
    config = {**IDENTITY, "estimation_mode": mode, "estimator_config": settings}
    with pytest.raises(ValueError, match="selector requires.*causal_selector"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("selector", [{}, {"alpha": 0.2, "warmup": 3, "margin": 0.05, "persistence": 2}])
def test_selector_saved_config_preserves_controls_and_parent_identity(tmp_path, selector):
    _, seed, _ = train(tmp_path)
    config = {
        **IDENTITY,
        "estimation_mode": "fpm_learned",
        "speculation": {"kind": "none"},
        "estimator_config": {
            "fpm_learned": {"adaptation": "causal_selector", "checkpoint_path": str(seed)},
            "correction": {"enabled": True, "mode": "hybrid"},
            "selector": selector,
        },
    }
    saved = RustForwardPassPerfModel.normalize_config(config)
    assert ("selector" in saved["estimator_config"]) == bool(selector)
    if selector:
        assert saved["estimator_config"]["selector"] == selector
    assert RustForwardPassPerfModel.normalize_config(ForwardPassPerfModelConfig(**saved)) == saved
    assert RustForwardPassPerfModel.learned_identity(saved) == RustForwardPassPerfModel.learned_identity(IDENTITY)
    original = RustForwardPassPerfModel.best_available(config)
    restored = RustForwardPassPerfModel.best_available(saved)
    assert original.diagnostics()["source"] == restored.diagnostics()["source"] == "learned_causal_selector"
    assert restored.learned_selector_diagnostics() == original.learned_selector_diagnostics()
    assert restored.estimate_forward_pass_time_ms(iteration(2)) == original.estimate_forward_pass_time_ms(iteration(2))
