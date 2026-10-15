# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python contract for Rust-owned spline configuration and online state."""

import json
import math
from copy import deepcopy

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit

DEFAULT_SEARCH = {
    "kind": "adaptive",
    "window": 16,
    "trigger": 8,
    "tolerance": 0.05,
    "absolute_tolerance_ms": 1.0,
    "cooldown": 64,
}


def _config(fit, **regression):
    return ForwardPassPerfModelConfig(
        model="test/model",
        system="test",
        backend="vllm",
        worker_type="decode",
        estimation_mode="fpm_regression",
        estimator_config={"fpm_regression": {"fit": fit, **regression}},
    )


def _sample(index):
    batch = index % 7 + 1
    kv = index * index * 13 + 11
    # Hand-specified positive affine surface; this exercises transport/state,
    # while the Rust numerical-oracle tests establish fitting accuracy.
    return {
        "version": 1,
        "wall_time": (2.0 + 0.001 * kv + 0.1 * batch) / 1000,
        "scheduled_requests": {"num_decode_requests": batch, "sum_decode_kv_tokens": kv},
    }


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize(
    ("spline", "expected"),
    [
        (None, {"knots_per_axis": 2, "search": DEFAULT_SEARCH}),
        (
            {"knots_per_axis": 3, "search": {"kind": "periodic"}},
            {"knots_per_axis": 3, "search": {"kind": "periodic", "step": 64}},
        ),
        (
            {"search": {**DEFAULT_SEARCH, "window": 12, "trigger": 3, "absolute_tolerance_ms": 0.0}},
            {
                "knots_per_axis": 2,
                "search": {**DEFAULT_SEARCH, "window": 12, "trigger": 3, "absolute_tolerance_ms": 0.0},
            },
        ),
    ],
)
def test_spline_defaults_and_saved_config_restart_without_learned_state(as_mapping, spline, expected):
    fit = {"kind": "spline", "rebuild_interval": 17}
    if spline is not None:
        fit["spline"] = deepcopy(spline)
    config = _config(fit)
    authored = deepcopy(config.to_dict())
    model = RustForwardPassPerfModel.best_available(authored if as_mapping else config)
    resolved = model.diagnostics()["provenance"]["config"]
    assert resolved["estimator_config"]["fpm_regression"]["fit"] == {
        "kind": "spline",
        "singular_ridge_scale": 1e-9,
        "rebuild_interval": 17,
        "spline": expected,
    }
    assert config.to_dict() == authored
    for index in range(1, 34):
        model.tune_with_fpms(_sample(index))
    assert model.regression_store_diagnostics()[0]["spline"]["initialized"]
    restored = RustForwardPassPerfModel.best_available(json.loads(json.dumps(resolved)))
    assert restored.diagnostics()["provenance"]["config"] == resolved
    assert restored.regression_store_diagnostics() == [
        {
            "workload_kind": "pure_decode",
            "ready": False,
            "retained_observations": 0,
            "spline": {
                "initialized": False,
                "ready": False,
                "accepted_observations": 0,
                "knot_searches": 0,
                "last_search_observation": None,
                "numerical_rebuilds": 0,
                "batch_fallbacks": 0,
            },
        }
    ]


@pytest.mark.parametrize("knots", [2, 3])
def test_periodic_search_counts_accepted_observations_and_queries_do_not_train(knots):
    model = RustForwardPassPerfModel.best_available(
        _config(
            {
                "kind": "spline",
                "spline": {"knots_per_axis": knots, "search": {"kind": "periodic", "step": 64}},
            }
        )
    )
    assert model.estimate_forward_pass_time_ms(_sample(1)) is None
    for index in range(1, 65):
        if index in (32, 64):
            before = model.regression_store_diagnostics()
            query = _sample(index)
            prediction = model.estimate_forward_pass_time_ms(query)
            query["wall_time"] = 999.0
            assert model.estimate_forward_pass_time_ms(query) == prediction
            assert model.regression_store_diagnostics() == before
            rejected = {**query, "wall_time": 0.0}
            model.tune_with_fpms(rejected)
            assert model.regression_store_diagnostics() == before
            for queued in ({}, {"num_decode_requests": 7, "sum_decode_kv_tokens": 4096}):
                idle = {
                    "version": 1,
                    "wall_time": 1.0,
                    "scheduled_requests": {},
                    "queued_requests": queued,
                }
                assert model.estimate_forward_pass_time_ms(idle) == 0.0
                model.tune_with_fpms(idle)
                assert model.regression_store_diagnostics() == before
        model.tune_with_fpms(_sample(index))
        if index in (5, 31, 32, 63, 64):
            store = model.regression_store_diagnostics()[0]
            state = store["spline"]
            assert store["ready"]
            assert state["accepted_observations"] == index
            assert state["initialized"] == (index >= 32)
            assert state["knot_searches"] == (0 if index < 32 else 1 if index < 64 else 2)
            assert state["last_search_observation"] == (None if index < 32 else 32 if index < 64 else 64)
            prediction = model.estimate_forward_pass_time_ms(_sample(index))
            assert prediction is not None and math.isfinite(prediction) and prediction > 0


def test_spline_initial_search_respects_larger_minimum_observation_count():
    model = RustForwardPassPerfModel.best_available(_config({"kind": "spline"}, min_observations=40))
    for index in range(1, 40):
        model.tune_with_fpms(_sample(index))
    store = model.regression_store_diagnostics()[0]
    assert not store["ready"]
    assert not store["spline"]["initialized"]
    model.tune_with_fpms(_sample(40))
    store = model.regression_store_diagnostics()[0]
    assert store["ready"]
    assert store["spline"]["last_search_observation"] == 40


def test_spline_component_cannot_make_store_ready_without_linear_fit():
    model = RustForwardPassPerfModel.best_available(
        _config(
            {"kind": "spline", "spline": {"search": {"kind": "periodic", "step": 64}}},
            sampling={"bins_per_axis": [1, 1], "max_observations": 64},
        )
    )
    # Negative overall covariance makes linear NNLS abstain, while the rising
    # tail still permits a positive spline segment slope.
    for kv in range(32):
        latency_ms = 100.0 if kv == 0 else 1.0 if kv < 24 else 10.0
        model.tune_with_fpms(
            {
                "version": 1,
                "wall_time": latency_ms / 1000,
                "scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": kv},
            }
        )
    store = model.regression_store_diagnostics()[0]
    assert store["spline"]["ready"]
    assert not store["ready"]
    assert model.diagnostics()["readiness"] == "insufficient_data"
    for kv in (0, 15, 31, 32):
        assert (
            model.estimate_forward_pass_time_ms(
                {
                    "version": 1,
                    "scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": kv},
                }
            )
            is None
        )
    assert model.regression_store_diagnostics()[0] == store


def test_linear_alias_and_omitted_kind_preserve_predictions_and_diagnostics():
    models = [
        RustForwardPassPerfModel.best_available(
            _config(fit, sampling={"bins_per_axis": [1, 1], "max_observations": 64})
        )
        for fit in ({}, {"kind": "linear"}, {"kind": "standardized_nnls"})
    ]
    for index in range(1, 90):
        forecasts = [model.estimate_forward_pass_time_ms(_sample(index)) for model in models]
        assert forecasts[0] == forecasts[1] == forecasts[2]
        for model in models:
            model.tune_with_fpms(_sample(index))
    for model in models:
        assert model.regression_store_diagnostics() == [
            {
                "workload_kind": "pure_decode",
                "ready": True,
                "retained_observations": 64,
            }
        ]
        fit = model.diagnostics()["provenance"]["config"]["estimator_config"]["fpm_regression"]["fit"]
        assert fit["kind"] == "standardized_nnls"
        assert "spline" not in fit


@pytest.mark.parametrize("field", ["tolerance", "absolute_tolerance_ms"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_spline_facade_rejects_nonfinite_search_settings(field, invalid):
    config = _config({"kind": "spline", "spline": {"search": {"kind": "adaptive", field: invalid}}})
    with pytest.raises(ValueError, match=rf"estimator_config\.fpm_regression\.fit\.spline\.search\.{field}"):
        RustForwardPassPerfModel.best_available(config)
