# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public config, bootstrap, causal quality routing, and independent learning."""

import json
from copy import deepcopy

import pytest

from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from . import test_fpm_mixed_composition_integration as shared
from . import test_fpm_mixed_selector_integration as selector
from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies

pytestmark = pytest.mark.unit
mixed_seed = shared.mixed_seed
phase_roots = shared.phase_roots
c_checkpoint = selector.c_checkpoint


def config(seed):
    value = shared._config(seed, adaptation="online")
    value["estimator_config"]["fpm_learned"]["update_every"] = 1000
    value["estimator_config"]["temporary_mixed_composition"]["quality_gate"] = {
        "alpha": 1.0,
        "min_comparisons": 1,
    }
    return value


@pytest.mark.parametrize("adaptation", ["frozen", "online", "quality_gate", "causal_selector"])
def test_zero_time_mixed_feedback_does_not_interrupt_public_batch(mixed_seed, adaptation):
    authored = config(mixed_seed) if adaptation == "quality_gate" else shared._config(mixed_seed, adaptation=adaptation)
    if adaptation == "frozen":
        authored["estimator_config"]["correction"] = {"enabled": True, "mode": "hybrid"}
    model = RustForwardPassPerfModel.best_available(authored)
    control = RustForwardPassPerfModel.best_available(authored)
    try:
        first = shared._iteration(wall_ms=6.0, index=1)
        zero = shared._iteration(wall_ms=0.0, index=2)
        last = shared._iteration(wall_ms=8.0, index=3)
        control.tune_with_fpms([first, last])
        model.tune_with_fpms([first, zero, last])
        assert model.mixed_composition_state() == control.mixed_composition_state()
        route = model.estimate_forward_pass_with_route(last)
        assert route["temporary_mixed_composition"]["calibration_admissions"] == 2
        assert route == control.estimate_forward_pass_with_route(last)
        if adaptation == "causal_selector":
            assert model.learned_expert_artifacts() == control.learned_expert_artifacts()
            assert model.learned_selector_diagnostics() == control.learned_selector_diagnostics()
        else:
            assert model.export_learned_artifact() == control.export_learned_artifact()
    finally:
        model.close()
        control.close()


def test_quality_gate_preserves_bootstrap_and_checks_future_labels(mixed_seed, c_checkpoint, tmp_path):
    authored = config(mixed_seed)
    normalized = RustForwardPassPerfModel.normalize_config(authored)
    assert RustForwardPassPerfModel.normalize_config(json.loads(json.dumps(normalized))) == normalized
    assert (
        normalized["estimator_config"]["temporary_mixed_composition"]["quality_gate"]
        == authored["estimator_config"]["temporary_mixed_composition"]["quality_gate"]
    )
    checkpoint, rows = c_checkpoint
    model = RustForwardPassPerfModel.best_available(normalized)
    control_config = shared._config(mixed_seed, checkpoint, enabled=False)
    control_config["estimator_config"]["fpm_learned"]["update_every"] = 1000
    control = RustForwardPassPerfModel.best_available(control_config)
    try:
        for row in rows:
            assert model.estimate_forward_pass_with_route(row)["temporary_mixed_composition"]["paired_comparisons"] == 0
            model.tune_with_fpms([row])
        before = model.mixed_composition_state()
        assert model.append_mixed_learned_checkpoint(checkpoint) == 1
        assert model.mixed_composition_state() == before
        # Native synthetic interpolation 4ms, calibration capped at 2 => 8ms;
        # GBDT training seed 12ms. Availability must not select it yet.
        assert model.estimate_forward_pass_time_ms(rows[0]) == 8.0
        for i, (wall, expert) in enumerate([(8.0, "stable"), (12.0, "online"), (8.0, "stable")], 1):
            row = shared._iteration(wall_ms=wall, index=20 + i)
            before_query = model.estimate_forward_pass_with_route(row)
            hidden = deepcopy(row)
            hidden[0]["wall_time"] = 999.0
            assert model.estimate_forward_pass_with_route(hidden) == before_query
            for current in (model, control):
                current.tune_with_fpms([row])
            result = model.estimate_forward_pass_with_route(row)
            route = result["temporary_mixed_composition"]
            assert route["selected_expert"] == expert
            assert route["paired_comparisons"] == i
            assert result["latency_ms"] == pytest.approx(8.0 if expert == "stable" else 12.0)
            assert model.mixed_composition_route(row) == route
            artifact = model.export_learned_artifact()
            assert (
                artifact["stores"]["contains_locally_mixed"]
                == control.export_learned_artifact()["stores"]["contains_locally_mixed"]
            )
        state = model.mixed_composition_state()
        model.restore_mixed_composition_state(json.loads(json.dumps(state)))
        assert model.mixed_composition_state() == state
        corrupt = deepcopy(state)
        del corrupt["quality_gate"]
        with pytest.raises(ValueError, match="quality gate presence"):
            model.restore_mixed_composition_state(corrupt)
        assert model.mixed_composition_state() == state
        saved = tmp_path / "online-with-quality"
        model.save_online_checkpoint(saved)
        resume_config = deepcopy(authored)
        resume_config["estimator_config"]["fpm_learned"]["checkpoint_path"] = str(saved)
        resumed = RustForwardPassPerfModel.best_available(resume_config)
        try:
            assert resumed.mixed_composition_state() == state
            assert resumed.estimate_forward_pass_with_route(row) == model.estimate_forward_pass_with_route(row)
            for current in (model, resumed):
                current.tune_with_fpms([shared._iteration(wall_ms=12.0, index=24)])
                assert (
                    current.estimate_forward_pass_with_route(row)["temporary_mixed_composition"]["selected_expert"]
                    == "online"
                )
            assert resumed.mixed_composition_state() == model.mixed_composition_state()
            assert resumed.export_learned_artifact() == model.export_learned_artifact()
        finally:
            resumed.close()
    finally:
        model.close()
        control.close()


def test_quality_gate_preserves_coverage_escape(mixed_seed, c_checkpoint):
    model = RustForwardPassPerfModel.best_available(config(mixed_seed))
    try:
        model.append_mixed_learned_checkpoint(c_checkpoint[0])
        row = shared._iteration(wall_ms=12.0)
        row[0]["scheduled_requests"].update(num_decode_requests=8, sum_decode_kv_tokens=256)
        result = model.estimate_forward_pass_with_route(row)
        assert result["latency_ms"] == pytest.approx(12.0)
        assert result["temporary_mixed_composition"]["coverage_escape"]
        model.tune_with_fpms([row])
        assert model.estimate_forward_pass_with_route(row)["temporary_mixed_composition"]["paired_comparisons"] == 0
    finally:
        model.close()


@pytest.mark.parametrize("gate", [{"alpha": 0}, {"alpha": 1.1}, {"min_comparisons": 0}, {"unknown": True}])
def test_quality_gate_rejects_invalid_controls(mixed_seed, gate):
    authored = config(mixed_seed)
    authored["estimator_config"]["temporary_mixed_composition"]["quality_gate"] = gate
    with pytest.raises(ValueError):
        RustForwardPassPerfModel.normalize_config(authored)


@pytest.mark.parametrize("adaptation", ["frozen", "causal_selector"])
def test_quality_gate_is_limited_to_standalone_online_gbdt(mixed_seed, adaptation):
    authored = config(mixed_seed)
    authored["estimator_config"]["fpm_learned"]["adaptation"] = adaptation
    with pytest.raises(ValueError, match="requires standalone online"):
        RustForwardPassPerfModel.normalize_config(authored)
