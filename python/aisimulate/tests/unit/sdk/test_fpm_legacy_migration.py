# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit legacy estimator controls survive migration and saved-config reloads."""

import copy
import json
import pickle

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, ForwardPassPerfOptions, RustForwardPassPerfModel

pytestmark = pytest.mark.unit


def _legacy_config(estimator_config=None, **extra):
    if estimator_config is not None:
        extra["estimator_config"] = json.dumps(estimator_config)
    return {
        "schema_version": 1,
        "model_name": "test/model",
        "system_name": "test-system",
        "backend": "vllm",
        "tp_size": 1,
        "pp_size": 1,
        "extra": extra,
    }


def _reload(canonical):
    return ForwardPassPerfModelConfig(**json.loads(json.dumps(canonical.to_dict())))


def test_migration_merges_saved_and_caller_controls_before_round_trip():
    legacy = _legacy_config(
        {
            "fpm_interpolation": {"method": "sol"},
            "features": {"ffn_token_weight": 2},
            "correction": {"enabled": False, "factor_bounds": {"max": 3}},
            "fpm_regression": {"fit": {"kind": "standardized_nnls"}},
        }
    )
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(
        legacy,
        "prefill",
        {
            "max_observations": 128,
            "min_observations": 8,
            "min_faster_correction_factor": 0.9,
            "regression_attention_kv_weight": 7,
            "bucket_shape": [4, 16],
            "regression_ridge_scale": 0,
        },
    )
    controls = RustForwardPassPerfModel.normalize_config(_reload(canonical))["estimator_config"]
    assert controls["fpm_interpolation"]["method"] == "sol"
    assert controls["features"] == {
        "attention_kv_weight": 7,
        "prefill_attention_pair_weight": 1,
        "ffn_token_weight": 2,
    }
    assert controls["correction"]["enabled"] is False
    assert controls["correction"]["factor_bounds"] == {"min": 0.9, "max": 3}
    for store in ("correction", "fpm_regression"):
        assert controls[store]["sampling"] == {"bins_per_axis": [4, 16], "max_observations": 128}
        assert controls[store]["min_observations"] == 8
    assert controls["fpm_regression"]["fit"] == {
        "kind": "standardized_nnls",
        "singular_ridge_scale": 0,
        "rebuild_interval": None,
    }


@pytest.mark.parametrize("options", [None, {"max_observations": 96}])
@pytest.mark.parametrize(
    ("spline", "expected"),
    [
        (
            None,
            {
                "knots_per_axis": 2,
                "search": {
                    "kind": "adaptive",
                    "window": 16,
                    "trigger": 8,
                    "tolerance": 0.05,
                    "absolute_tolerance_ms": 1.0,
                    "cooldown": 64,
                },
            },
        ),
        (
            {"knots_per_axis": 3, "search": {"kind": "periodic", "step": 17}},
            {"knots_per_axis": 3, "search": {"kind": "periodic", "step": 17}},
        ),
        (
            {"search": {"kind": "adaptive", "window": 9, "trigger": 3, "tolerance": 0.125}},
            {
                "knots_per_axis": 2,
                "search": {
                    "kind": "adaptive",
                    "window": 9,
                    "trigger": 3,
                    "tolerance": 0.125,
                    "absolute_tolerance_ms": 1.0,
                    "cooldown": 64,
                },
            },
        ),
    ],
)
def test_spline_controls_resolve_during_migration_and_survive_normalized_reload(options, spline, expected):
    fit = {"kind": "spline", "rebuild_interval": 17}
    if spline is not None:
        fit["spline"] = copy.deepcopy(spline)
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(
        _legacy_config({"fpm_regression": {"fit": fit}}), "decode", options
    )
    expected_fit = {
        "kind": "spline",
        "singular_ridge_scale": 1e-9,
        "rebuild_interval": 17,
        "spline": expected,
    }
    # Migration itself emits spline defaults, before full request normalization.
    assert canonical.estimator_config["fpm_regression"]["fit"] == expected_fit
    normalized = RustForwardPassPerfModel.normalize_config(_reload(canonical))
    assert normalized["estimator_config"]["fpm_regression"]["fit"] == expected_fit
    assert normalized["estimator_config"]["fpm_regression"]["sampling"]["max_observations"] == (
        64 if options is None else 96
    )
    restored = ForwardPassPerfModelConfig(**json.loads(json.dumps(normalized)))
    assert RustForwardPassPerfModel.normalize_config(restored) == normalized


@pytest.mark.parametrize("options", [None, {}, ForwardPassPerfOptions()])
def test_omitted_caller_fields_do_not_override_saved_controls(options):
    saved = {
        "correction": {"enabled": False, "sampling": {"max_observations": 128}},
        "fpm_regression": {"fit": {"singular_ridge_scale": 0}},
        "features": {"attention_kv_weight": 7},
    }
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(_legacy_config(saved), "prefill", options)
    controls = _reload(canonical).estimator_config
    assert controls["correction"]["enabled"] is False
    assert controls["correction"]["sampling"]["max_observations"] == 128
    assert controls["fpm_regression"]["sampling"]["max_observations"] == 64
    assert controls["fpm_regression"]["fit"]["singular_ridge_scale"] == 0
    assert controls["features"]["attention_kv_weight"] == 7


def test_legacy_options_serialize_presence_including_explicit_defaults():
    options = ForwardPassPerfOptions(64, min_faster_correction_factor=None, regression_ridge_scale=0)
    assert options.min_observations == 5  # Existing attribute defaults remain available.
    assert options.to_dict() == {
        "max_observations": 64,
        "min_faster_correction_factor": None,
        "regression_ridge_scale": 0,
    }
    loaded = ForwardPassPerfOptions(**json.loads(json.dumps(options.to_dict())))
    assert loaded.to_dict() == options.to_dict()
    saved = {"features": {"attention_kv_weight": 7}}
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(_legacy_config(saved), "decode", loaded)
    controls = _reload(canonical).estimator_config
    assert controls["features"]["attention_kv_weight"] == 7
    assert controls["correction"]["factor_bounds"]["min"] is None
    assert controls["fpm_regression"]["fit"]["singular_ridge_scale"] == 0


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
@pytest.mark.parametrize("controls", [{}, {"max_observations": 64, "regression_ridge_scale": 0}])
def test_legacy_options_copy_preserves_explicit_fields(clone, controls):
    assert clone(ForwardPassPerfOptions(**controls)).to_dict() == controls


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("controls", [{}, {"max_observations": 64, "regression_ridge_scale": 0}])
def test_legacy_options_pickle_preserves_explicit_fields(protocol, controls):
    options = ForwardPassPerfOptions(**controls)
    assert pickle.loads(pickle.dumps(options, protocol=protocol)).to_dict() == controls


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_legacy_options_pickle_without_presence_preserves_saved_controls(protocol):
    options = ForwardPassPerfOptions(max_observations=128, min_observations=8, regression_ridge_scale=0)
    # Older pickles saved dataclass attributes without constructor-presence metadata.
    object.__delattr__(options, "_explicit_fields")
    restored = pickle.loads(pickle.dumps(options, protocol=protocol))
    assert restored.to_dict() == vars(options)
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(_legacy_config(), "prefill", restored)
    controls = _reload(canonical).estimator_config
    for store in ("correction", "fpm_regression"):
        assert controls[store]["sampling"]["max_observations"] == 128
        assert controls[store]["min_observations"] == 8
    assert controls["fpm_regression"]["fit"]["singular_ridge_scale"] == 0


@pytest.mark.parametrize(
    ("saved", "options"),
    [
        ({"features": {"attention_kv_weight": 1}}, {"regression_attention_kv_weight": 1.0}),
        ({"correction": {"factor_bounds": {"min": None}}}, {"min_faster_correction_factor": None}),
        ({"correction": {"factor_bounds": {"min": 0.5}}}, {"min_faster_correction_factor": 0.5}),
        ({"fpm_regression": {"sampling": {"bins_per_axis": [4, 16]}}}, {"bucket_shape": [4, 16]}),
        ({"correction": {"sampling": {"bins_per_axis": [8, 8]}}}, {"bucket_count": 64}),
    ],
)
def test_agreeing_overlap_is_accepted(saved, options):
    saved = saved | {"fpm_interpolation": {"method": "sol"}}
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(_legacy_config(saved), "decode", options)
    assert (
        RustForwardPassPerfModel.normalize_config(_reload(canonical))["estimator_config"] == canonical.estimator_config
    )


@pytest.mark.parametrize(
    ("saved", "options", "field"),
    [
        (
            {"fpm_regression": {"sampling": {"max_observations": 128}}},
            ForwardPassPerfOptions(max_observations=64),
            "fpm_regression.sampling.max_observations",
        ),
        (
            {"features": {"attention_kv_weight": 7}},
            {"regression_attention_kv_weight": 1},
            "features.attention_kv_weight",
        ),
        (
            {"correction": {"factor_bounds": {"min": 0.5}}},
            {"min_faster_correction_factor": None},
            "correction.factor_bounds.min",
        ),
        (
            {"correction": {"sampling": {"bins_per_axis": [4, 16]}}},
            {"bucket_shape": [4, 4]},
            "correction.sampling.bins_per_axis",
        ),
        (
            {"fpm_regression": {"fit": {"singular_ridge_scale": 1e-9}}},
            {"regression_ridge_scale": 0},
            "fpm_regression.fit.singular_ridge_scale",
        ),
    ],
)
def test_conflicting_overlap_identifies_canonical_field(saved, options, field):
    with pytest.raises(ValueError, match=rf"conflicting explicit values for estimator_config\.{field}"):
        ForwardPassPerfModelConfig.from_legacy_engine_config(_legacy_config(saved), "prefill", options)


def test_sampling_validation_uses_merged_values():
    saved = {
        "correction": {"sampling": {"max_observations": 128}},
        "fpm_regression": {"sampling": {"max_observations": 128}},
    }
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(
        _legacy_config(saved), "prefill", {"min_observations": 80}
    )
    controls = _reload(canonical).estimator_config
    assert controls["correction"]["min_observations"] == 80
    assert controls["fpm_regression"]["min_observations"] == 80
    with pytest.raises(ValueError, match="min_observations must be <= max_observations"):
        ForwardPassPerfModelConfig.from_legacy_engine_config(
            _legacy_config(saved), "prefill", {"min_observations": 129}
        )


@pytest.mark.parametrize("canonical_method", [None, "direct"])
def test_omitted_and_agreeing_interpolation_methods_accept_legacy_selection(canonical_method):
    controls = {"fpm_interpolation": {} if canonical_method is None else {"method": canonical_method}}
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(
        _legacy_config(controls, fpm_interpolation="direct"), "prefill"
    )
    assert _reload(canonical).estimator_config["fpm_interpolation"]["method"] == "direct"


@pytest.mark.parametrize("canonical_method,legacy_method", [("auto", "direct"), ("direct", "auto"), ("sol", "direct")])
def test_explicit_interpolation_method_conflicts_are_symmetric(canonical_method, legacy_method):
    with pytest.raises(ValueError, match=r"estimator_config\.fpm_interpolation\.method"):
        ForwardPassPerfModelConfig.from_legacy_engine_config(
            _legacy_config({"fpm_interpolation": {"method": canonical_method}}, fpm_interpolation=legacy_method),
            "prefill",
        )


@pytest.mark.parametrize("legacy_path", ["/tmp/fpm.parquet", "/tmp/./fpm.parquet", "/tmp//fpm.parquet"])
def test_equivalent_parquet_paths_merge(legacy_path):
    legacy = _legacy_config({"fpm_interpolation": {"fpm_parquet_path": "/tmp/fpm.parquet"}})
    legacy.update(forward_model="fpm", fpm_parquet_path=legacy_path)
    canonical = ForwardPassPerfModelConfig.from_legacy_engine_config(legacy, "prefill")
    assert _reload(canonical).estimator_config["fpm_interpolation"]["fpm_parquet_path"] == "/tmp/fpm.parquet"


@pytest.mark.parametrize("legacy_path", ["/tmp/different.parquet", "/tmp/subdir/../fpm.parquet"])
def test_different_parquet_paths_conflict_without_filesystem_resolution(legacy_path):
    legacy = _legacy_config({"fpm_interpolation": {"fpm_parquet_path": "/tmp/fpm.parquet"}})
    legacy.update(forward_model="fpm", fpm_parquet_path=legacy_path)
    with pytest.raises(ValueError, match=r"estimator_config\.fpm_interpolation\.fpm_parquet_path"):
        ForwardPassPerfModelConfig.from_legacy_engine_config(legacy, "prefill")


@pytest.mark.parametrize(
    ("saved", "options", "reason"),
    [
        ({"correction": {"enabld": False}}, {}, "unknown field"),
        ({"fpm_regression": {"sampling": {"bins_per_axis": [4]}}}, {}, "length"),
        ({"correction": None}, {}, "invalid type"),
        ({}, {"max_observatons": 128}, "unknown field"),
        ({}, {"min_observations": 0}, "min_observations must be >= 1"),
        ({}, {"max_observations": False}, "invalid type"),
        ({}, {"bucket_count": 3}, "bucket_count must be a perfect square"),
        ({"fpm_regression": {"sampling": {"bins_per_axis": [0, 4]}}}, {}, "bins_per_axis must be positive"),
    ],
)
def test_migration_rejects_unknown_and_invalid_controls(saved, options, reason):
    with pytest.raises(ValueError, match=reason):
        ForwardPassPerfModelConfig.from_legacy_engine_config(_legacy_config(saved), "prefill", options)


@pytest.mark.parametrize("options", [{"text_only": False}, {"unrecorded_quant_modes": []}, {"collect_coverage": False}])
def test_explicit_skipped_defaults_migrate_without_becoming_null(options):
    config = ForwardPassPerfModelConfig.from_legacy_engine_config(
        _legacy_config({"fpm_interpolation": options}), "prefill"
    )
    normalized = RustForwardPassPerfModel.normalize_config(_reload(config))
    assert normalized["estimator_config"]["fpm_interpolation"]["method"] == "sol"
