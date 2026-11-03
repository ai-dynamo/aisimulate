# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public contract for strict five-second FPM scenario comparisons."""

from __future__ import annotations

import json
import math
from copy import deepcopy

from scripts.fpm_accuracy.contract import COUNTS, WORKLOADS, keys, path, require, sha, text, validate_summary

SCHEMA_VERSION = 13
PROTOCOL_ID = "fpm-scenarios-v13"
TRAIN_PERCENTAGES = tuple(range(10, 100, 10))
DEFAULT_TRAIN_PERCENT = 50
ONLINE_UPDATE_INTERVAL_MS = 5000
METHODS = ("warmup", "nowarmup", "linear", "spline", "gbdt", "hybrid", "selector")
SCENARIOS = ("a", "b", "c", "ac", "bc")
ONLINE_SCENARIOS = frozenset(("c", "ac", "bc"))
OFFLINE_ARTIFACT_SCENARIOS = frozenset(("a", "ac"))

GBDT_FROZEN_RECIPE = {
    "id": "gbdt_pr460_frozen_api_v13",
    "label": "frozen GBDT",
    "feature_preset": "fpm5",
    "features": [
        "num_active_ranks",
        "num_prefill_requests",
        "sum_prefill_tokens",
        "sum_prefill_kv_tokens",
        "num_decode_requests",
        "sum_decode_kv_tokens",
        "max_rank_prefill_tokens",
        "max_rank_decode_kv_tokens",
        "max_rank_num_decode_requests",
        "mean_prefill_chunk",
        "mean_prefill_kv",
        "prefill_attention_pairs",
        "mean_decode_kv",
        "log1p_sum_prefill_tokens",
        "log1p_sum_prefill_kv_tokens",
        "log1p_sum_decode_kv_tokens",
        "log1p_prefill_attention_pairs",
    ],
    "training": {
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "learning_rate": 0.2,
        "min_samples_leaf": 5,
        "max_bins": 255,
        "max_depth": None,
        "l2_regularization": 0.0,
        "loss": "squared_error",
        "early_stopping": False,
        "random_state": 0,
        "min_store_rows": 20,
        "target": "log_ms",
    },
    "model_scope": "worker_id_within_case",
    "implementation": "AISim canonical performance model API",
}

GBDT_ONLINE_RECIPE = {
    **deepcopy(GBDT_FROZEN_RECIPE),
    **{
        "id": "gbdt_pr460_grouped_history_fifo_api_v13",
        "label": "GBDT + grouped history + recent FIFO",
        "correction_enabled": False,
        "history_weight_population": "original_history_rows_and_current_recent_rows",
        "model_scope": "worker_id_and_native_workload_store",
        "online": {
            "update_mode": "structural",
            "history_bin_grouping": True,
            "recent_sampling": "fifo",
            "recent_window": 64,
            "update_every": 1,
            "history_weight": 0.1,
            "history_weight_ramp_observations": 32,
            "derivative_refresh": "lazy",
            "split_sampling_rate": 0.5,
            "split_rank_tolerance": 0.01,
            "eager_refresh_interval": 32,
            "seed": 0,
            "mutable_tree_tail": None,
            "update_step": 1.0,
            "recent_max_age_observations": None,
        },
    },
}

HYBRID_ONLINE_RECIPE = {
    **deepcopy(GBDT_FROZEN_RECIPE),
    **{
        "id": "gbdt_pr493_hybrid_calibration_api_v13",
        "label": "GBDT + hybrid calibration",
        "online": {"update_mode": "frozen"},
        "correction": {
            "enabled": True,
            "mode": "hybrid",
            "feature_space": "legacy_workload",
            "sampling": {"bins_per_axis": [4, 4], "max_observations": 64},
            "min_observations": 5,
            "factor_bounds": {"min": 0.5, "max": 2.0},
            "max_num_tokens": 8192,
            "max_batch_size": 512,
            "max_kv_tokens": 2000000,
        },
    },
}

SELECTOR_ONLINE_RECIPE = {
    **deepcopy(HYBRID_ONLINE_RECIPE),
    **{
        "id": "gbdt_selector_w8_hybrid64_alpha04_api_v13",
        "label": "GBDT selector · W8 / hybrid 64 / alpha 0.4",
        "selection_scope": "worker_id_forest_store_and_query_type",
        "calibration_scope": "worker_id_forest_store_and_query_type",
        "online": {
            "update_mode": "structural",
            "history_bin_grouping": True,
            "recent_sampling": "fifo",
            "recent_window": 8,
            "update_every": 4,
            "history_weight": 0.25,
            "history_weight_ramp_observations": 8,
            "derivative_refresh": "lazy",
            "split_sampling_rate": 0.5,
            "split_rank_tolerance": 0.01,
            "eager_refresh_interval": 32,
            "seed": 0,
            "mutable_tree_tail": None,
            "update_step": 1.0,
            "recent_max_age_observations": None,
            "adaptation": "causal_selector",
        },
        "selector": {"alpha": 0.4, "warmup": 5, "margin": 0.05, "persistence": 1},
    },
}

PROTOCOL_METADATA = {
    "methods": {
        "warmup": {"label": "Interpolation · KV warmup on"},
        "nowarmup": {"label": "Interpolation · KV warmup off"},
        "linear": {"label": "Linear regression"},
        "spline": {"label": "Spline regression"},
        "gbdt": {
            "label": "GBDT",
            "offline_recipe": GBDT_FROZEN_RECIPE,
            "online_recipe": GBDT_ONLINE_RECIPE,
            "offline_model": "Fit the 150-tree, 15-leaf, learning-rate-0.2 fpm5 seed using AISim, then freeze it.",
            "online_update": "AISim structural online GBDT with grouped history and a "
            "64-observation FIFO per native store.",
        },
        "hybrid": {
            "label": "GBDT + hybrid calibration",
            "online_recipe": HYBRID_ONLINE_RECIPE,
            "online_update": "Freeze the initial forest; AISim hybrid calibration consumes admitted C labels.",
        },
        "selector": {
            "label": "GBDT + causal selector",
            "online_recipe": SELECTOR_ONLINE_RECIPE,
            "online_update": "AISim selects between the stable hybrid-calibrated expert and "
            "the online GBDT using past paired errors.",
        },
    },
    "scenarios": {
        "a": {
            "label": "A · Self-benchmark → frozen",
            "description": "Fit the self-benchmark, then score all target observations without updates.",
        },
        "b": {
            "label": "B · Distributed holdout",
            "description": "Train a deterministic fraction within each workload kind; freeze and "
            "score the disjoint remainder.",
        },
        "c": {
            "label": "C · Strict online cold start",
            "description": "Start without offline data. Score every observation; admit one "
            "observation per five-second tick, including warmup.",
        },
        "ac": {
            "label": "A + C · Self-benchmark → online",
            "description": "Seed from the self-benchmark, then score all targets with five-second online feedback.",
        },
        "bc": {
            "label": "B + C · Temporal holdout → online",
            "description": "Train on the chronological prefix; score the remainder with five-second online feedback.",
        },
    },
    "online_schedule": {
        "interval_ms": 5000,
        "time_basis": "event_time_ms",
        "selection_policy": "first_observation_at_or_after_tick",
        "warmup_policy": "scheduled_only",
        "description": "Ticks begin five seconds after the first scored observation. Select the "
        "first observation at or after each elapsed tick; predict and score "
        "before admitting its label. Warmup uses exactly this schedule. Gaps "
        "skip ticks, ties consume a tick once, and there is no final flush or "
        "catch-up.",
    },
    "artifact_selection": "Select the first artifact by (artifact_id, path, sha256), independently of target "
    "error. Regression prefers KV warmup on.",
    "coverage": "MAPE averages absolute percentage errors over successful predictions. Coverage retains all "
    "eligible targets, including cold-start misses.",
    "split": {
        "default_train_percent": 50,
        "train_percentages": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "b": "Deterministic distributed holdout within each workload kind.",
        "bc": "First floor(N * train_percent / 100) chronological observations.",
    },
    "null_results": "Null denotes inapplicable methods; missing inputs or unsupported APIs retain an explicit status.",
    "mixed_fallback": {
        "implementation": "AISim PR #639 canonical API",
        "scenarios": ["a", "ac"],
        "original_a": "Source-bound self-benchmark manifest; never synthesized from B or C.",
        "gbdt": "A uses raw composition. A+C uses calibrated composition and the native "
        "continuous quality checker with defaults alpha=0.05, min_comparisons=5.",
        "hybrid": "A+C calibrates interpolation for missing mixed stores and keeps forests frozen.",
        "selector": "A+C competes calibrated interpolation against a causally bootstrapped mixed forest.",
    },
    "online_model": "Models and native diagnostics use the evaluated AISim wheel. Bootstrap consumes only "
    "previously scored, scheduled observations; no local model or selector reimplementation.",
    "offline_model": "A/B GBDT uses the same 150x15 seed recipe; evaluation labels are never admitted.",
    "online_update_count": "Successful native adapter tune calls; failed calls and submitted observations are "
    "counted separately.",
    "training_count": "B/B+C count offline trace observations; A/A+C count self-benchmark inputs per target "
    "worker; strict C is zero.",
    "interpolation": "Use the selected measured self-benchmark table and retain resolved native method diagnostics.",
    "timing": "Fit time covers preparation and initial model construction. Prediction/update times include "
    "complete adapter calls, including cold bootstrap and lazy fits. Event time is replayed without "
    "sleeping.",
}

ROW_FIELDS = (
    "configuration_id",
    "configuration_path",
    "snapshot_id",
    "model",
    "gpu",
    "framework",
    "framework_version",
    "parallelism",
    "worker_role",
    "status",
    "ordering",
    "membership_sha256",
    "measurement_count",
)
RESULT_STATUSES = [
    "chronology_unavailable",
    "evaluated",
    "no_fpm_input",
    "no_measurements",
    "no_test_data",
    "no_training_data",
    "predictor_error",
    "supporting_evidence_only",
    "unsupported_predictor",
    "unsupported_protocol",
]


def methods_for_version(version):
    require(type(version) is int and version == SCHEMA_VERSION, "unsupported scenario schema")
    return METHODS


def selector_recipe_for_version(version):
    methods_for_version(version)
    return SELECTOR_ONLINE_RECIPE


def applicable(method, scenario, version=SCHEMA_VERSION):
    methods_for_version(version)
    if method not in METHODS or scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario/method: {scenario}/{method}")
    return not (
        (method in ("warmup", "nowarmup") and scenario != "a")
        or (method in ("hybrid", "selector") and scenario not in ONLINE_SCENARIOS)
    )


def _model_evidence(result, method, scenario):
    name = "offline_model" if scenario in ("a", "b") else "online_model"
    evidence = result[name]
    keys(evidence, ("recipe", "diagnostics"))
    expected = (
        GBDT_FROZEN_RECIPE
        if name == "offline_model"
        else {"gbdt": GBDT_ONLINE_RECIPE, "hybrid": HYBRID_ONLINE_RECIPE, "selector": SELECTOR_ONLINE_RECIPE}[method]
    )
    require(evidence["recipe"] == expected, "scenario recipe mismatch")
    diagnostics = evidence["diagnostics"]
    require(
        isinstance(diagnostics, dict) if result["status"] == "evaluated" else diagnostics is None,
        "invalid native model evidence",
    )
    if diagnostics is not None:
        adaptation = (
            "causal_selector"
            if method == "selector"
            else "online"
            if method == "gbdt" and scenario in ONLINE_SCENARIOS
            else "frozen"
        )
        require(diagnostics.get("method") == method, "native scenario method mismatch")
        require(diagnostics.get("adaptation") == adaptation, "native scenario adaptation mismatch")
        require(
            diagnostics.get("model_scope") == "case_worker_and_native_workload_store",
            "native scenario model scope mismatch",
        )
        require(diagnostics.get("recipe") == expected, "native scenario recipe mismatch")
    # Native diagnostic fields evolve with the evaluated wheel. Require finite
    # JSON while binding the adapter-owned identity above to the public recipe.
    json.dumps(diagnostics, allow_nan=False)


def validate_train_percent(value):
    """Accept only the published integer training percentages."""
    require(type(value) is int and value in TRAIN_PERCENTAGES, "train_percent must be an integer from 10 to 90 by 10")
    return value


def _count(value):
    require(type(value) is int and value >= 0, "invalid scenario count")


def _number(value, *, nullable=False):
    require(
        (nullable and value is None) or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
        "invalid scenario number",
    )


def _metrics(metrics, expected_count):
    keys(metrics, WORKLOADS)
    for metric in metrics.values():
        keys(metric, (*COUNTS, "mape_pct", "coverage_pct"))
        for name in COUNTS:
            _count(metric[name])
        measured, predicted = metric["measured_count"], metric["predicted_count"]
        require(
            measured == predicted + metric["unavailable_count"] + metric["error_count"], "invalid scenario coverage"
        )
        require(metric["tuning_error_count"] <= measured, "invalid scenario tuning errors")
        if predicted:
            _number(metric["mape_pct"])
        else:
            require(metric["mape_pct"] is None, "missing predictions cannot have scenario MAPE")
        if measured:
            _number(metric["coverage_pct"])
            require(math.isclose(metric["coverage_pct"], predicted / measured * 100), "incorrect scenario coverage")
        else:
            require(metric["coverage_pct"] is None, "empty population cannot have scenario coverage")
    require(metrics["all"]["measured_count"] == expected_count, "scenario test count mismatch")
    for name in COUNTS:
        require(
            metrics["all"][name] == sum(metrics[workload][name] for workload in WORKLOADS[1:]),
            "inconsistent scenario workload totals",
        )
    predicted = metrics["all"]["predicted_count"]
    if predicted:
        weighted = sum((metrics[k]["mape_pct"] or 0) * metrics[k]["predicted_count"] for k in WORKLOADS[1:]) / predicted
        require(
            math.isclose(weighted, metrics["all"]["mape_pct"], rel_tol=1e-9, abs_tol=1e-9), "scenario MAPE mismatch"
        )


def _result(result, row, scenario, method, version, *, train_percent=DEFAULT_TRAIN_PERCENT):
    validate_train_percent(train_percent)
    require(version >= 4 or train_percent == DEFAULT_TRAIN_PERCENT, "legacy scenarios require a 50% training split")
    keys(
        result,
        (
            "status",
            "artifact",
            "training_count",
            "test_count",
            "online_update_count",
            "training_population_sha256",
            "scoring_population_sha256",
            "metrics",
            "cost",
            "cold_start",
            *(("reason",) if "reason" in result else ()),
            *(("interpolation",) if "interpolation" in result else ()),
            *(("online_schedule",) if version >= 2 and scenario in ONLINE_SCENARIOS else ()),
            *(("offline_model",) if version >= 12 and method == "gbdt" and scenario in ("a", "b") else ()),
            *(
                ("online_model",)
                if (version >= 5 and method == "gbdt" and scenario in ONLINE_SCENARIOS)
                or (version >= 7 and method == "hybrid" and applicable(method, scenario, version))
                or (version >= 9 and method == "selector")
                else ()
            ),
        ),
    )
    status = result["status"]
    require(status in RESULT_STATUSES, "invalid scenario status")
    if method in ("gbdt", "hybrid", "selector"):
        _model_evidence(result, method, scenario)
    if status != "evaluated":
        require("reason" in result, "missing scenario status explanation")
    if "reason" in result:
        text(result["reason"])
    if "interpolation" in result:
        require(method in ("warmup", "nowarmup") and scenario == "a", "unexpected interpolation evidence")
        require(status == "evaluated", "unavailable interpolation cannot claim a selected method")
        details = result["interpolation"]
        keys(details, ("method", "timing_only", "text_only", "metadata_recovered"))
        require(details["method"] in ("sol", "direct"), "invalid interpolation method")
        for flag in ("timing_only", "text_only", "metadata_recovered"):
            require(type(details[flag]) is bool, "invalid interpolation evidence flag")
        require(not details["timing_only"] or details["method"] == "direct", "timing-only interpolation must be direct")
    require(row["status"] == "ready" or status == row["status"], "scenario source status mismatch")
    for name in ("training_count", "test_count", "online_update_count"):
        _count(result[name])
    sha(result["scoring_population_sha256"], 64)
    if result["training_population_sha256"] is not None:
        sha(result["training_population_sha256"], 64)
    require(
        (scenario in ("b", "bc") and result["training_count"] > 0)
        == (result["training_population_sha256"] is not None),
        "scenario training population mismatch",
    )
    if scenario in ("b", "bc"):
        require(
            result["training_count"] + result["test_count"] == row["measurement_count"],
            "scenario train/test partition mismatch",
        )
        if result["training_population_sha256"] is not None:
            require(
                result["training_population_sha256"] != result["scoring_population_sha256"],
                "identical train/test populations",
            )
    elif scenario == "c":
        require(result["training_count"] == 0, "online-only scenario contains offline training")
    if scenario == "bc":
        require(
            result["training_count"] == row["measurement_count"] * train_percent // 100,
            "incorrect temporal split boundary",
        )
    if (
        (scenario == "bc" or (version >= 2 and scenario in ONLINE_SCENARIOS))
        and row["status"] == "ready"
        and row["ordering"] != "chronological"
    ):
        require(status == "chronology_unavailable", "temporal evaluation lacks chronological source")
    if status == "chronology_unavailable":
        require(
            (version >= 2 and scenario in ONLINE_SCENARIOS)
            or (version == 1 and scenario == "bc" and row["ordering"] != "chronological"),
            "incorrect chronology status",
        )
    if status == "no_fpm_input":
        require(
            scenario in OFFLINE_ARTIFACT_SCENARIOS and result["artifact"] is None, "incorrect missing artifact status"
        )
    if status == "no_test_data":
        require(result["test_count"] == 0, "incorrect missing test status")
    if status == "no_training_data":
        require(result["training_count"] == 0 and scenario != "c", "incorrect missing training status")
    artifact = result["artifact"]
    if artifact is not None:
        require(scenario in OFFLINE_ARTIFACT_SCENARIOS, "unexpected scenario artifact")
        keys(artifact, ("id", "path", "sha256", "metadata_path", "metadata_sha256"))
        text(artifact["id"])
        for field in ("path", "metadata_path"):
            path(artifact[field])
        for field in ("sha256", "metadata_sha256"):
            sha(artifact[field], 64)
        if method in ("warmup", "nowarmup"):
            require((".kv-off." in artifact["path"].lower()) == (method == "nowarmup"), "wrong scenario FPM mode")
    if status == "evaluated" and scenario in OFFLINE_ARTIFACT_SCENARIOS:
        require(artifact is not None, "offline scenario lacks artifact")
    _metrics(result["metrics"], result["test_count"])
    metric = result["metrics"]["all"]
    costs = result["cost"]
    keys(costs, ("fit_ms", "predict_us", "update_us", "predict_count", "update_count"))
    for field in ("fit_ms", "predict_us", "update_us"):
        _number(costs[field], nullable=True)
    for field in ("predict_count", "update_count"):
        _count(costs[field])
        require(costs[field] <= result["test_count"], "excess scenario timing invocations")
    for field in ("predict", "update"):
        require((costs[field + "_count"] == 0) == (costs[field + "_us"] is None), "missing scenario invocation time")
    if version >= 2 and scenario in ONLINE_SCENARIOS:
        _online_schedule(result, scenario, version)
    else:
        require(
            result["online_update_count"] + metric["tuning_error_count"] == costs["update_count"],
            "inconsistent scenario update outcomes",
        )
    if scenario not in ONLINE_SCENARIOS:
        require(costs["update_count"] == 0, "frozen scenario performed online updates")
    cold = result["cold_start"]
    keys(cold, ("unavailable_before_first_prediction", "first_prediction_index"))
    unavailable, first = cold["unavailable_before_first_prediction"], cold["first_prediction_index"]
    if status == "evaluated":
        require(result["test_count"] > 0, "evaluated scenario has no test data")
        require(costs["fit_ms"] is not None, "evaluated scenario lacks fit timing")
        require(costs["predict_count"] == result["test_count"], "scenario skipped predictions")
        if version == 1 and scenario in ONLINE_SCENARIOS:
            require(costs["update_count"] == result["test_count"], "online scenario skipped updates")
        _count(unavailable)
        require(unavailable <= metric["unavailable_count"], "invalid cold start count")
        if metric["predicted_count"]:
            _count(first)
            require(unavailable <= first < result["test_count"], "invalid first prediction index")
        else:
            require(first is None and unavailable == metric["unavailable_count"], "inconsistent unready scenario")
    else:
        require(costs["predict_count"] == costs["update_count"] == 0, "unavailable scenario ran predictions")
        require(unavailable is None and first is None, "unavailable predictor has cold start evidence")
        outcome = "error_count" if status == "predictor_error" else "unavailable_count"
        require(metric[outcome] == result["test_count"], "incorrect unavailable scenario outcomes")


def _online_schedule(result, scenario, version):
    schedule = result["online_schedule"]
    scheduled = "scheduled_update_count" if version >= 3 else "batch_update_count"
    unused = "skipped_observation_count" if version >= 3 else "pending_observation_count"
    counts = (
        scheduled,
        "warmup_update_count",
        "update_error_count",
        "submitted_observation_count",
        unused,
    )
    keys(
        schedule,
        (
            "interval_ms",
            "time_basis",
            "warmup_policy",
            "elapsed_ms",
            *counts,
            *(("selection_policy",) if version >= 3 else ()),
        ),
    )
    if version >= 3:
        require(
            schedule["selection_policy"] == "first_observation_at_or_after_tick", "incorrect online point selection"
        )
    require(
        type(schedule["interval_ms"]) is int and schedule["interval_ms"] == ONLINE_UPDATE_INTERVAL_MS,
        "incorrect online update interval",
    )
    require(schedule["time_basis"] == "event_time_ms", "incorrect online time basis")
    require(
        schedule["warmup_policy"] == ("scheduled_only" if scenario == "c" else "none"),
        "incorrect online warmup policy",
    )
    _number(schedule["elapsed_ms"], nullable=True)
    for field in counts:
        _count(schedule[field])
    costs = result["cost"]
    submitted = schedule["submitted_observation_count"]
    failed_rows = result["metrics"]["all"]["tuning_error_count"]
    require(
        costs["update_count"] == schedule[scheduled] + schedule["warmup_update_count"],
        "inconsistent scheduled update calls",
    )
    require(
        costs["update_count"] == result["online_update_count"] + schedule["update_error_count"],
        "inconsistent scheduled update outcomes",
    )
    require(costs["update_count"] <= submitted <= result["test_count"], "invalid submitted observations")
    require(schedule["update_error_count"] <= failed_rows <= submitted, "invalid failed update observations")
    require((schedule["update_error_count"] == 0) == (failed_rows == 0), "inconsistent failed update observations")
    require(result["online_update_count"] <= submitted - failed_rows, "invalid successful update observations")
    if version >= 3:
        require(submitted == costs["update_count"], "each online update must submit exactly one observation")
        require(
            failed_rows == schedule["update_error_count"], "each failed update must contain exactly one observation"
        )
    require(schedule["warmup_update_count"] == 0, "strict schedule contains per-point warmup updates")
    if result["status"] == "evaluated":
        require(schedule["elapsed_ms"] is not None, "online scenario lacks elapsed event time")
        require(
            submitted + schedule[unused] == result["test_count"],
            "online observation accounting mismatch",
        )
        require(
            schedule[scheduled] <= schedule["elapsed_ms"] // ONLINE_UPDATE_INTERVAL_MS,
            "online scenario updated faster than its cadence",
        )
    else:
        require(
            schedule["elapsed_ms"] is None and all(schedule[field] == 0 for field in counts),
            "unavailable scenario claims online execution",
        )


def _validate_method_results(methods, row, source, scenario, version, train_percent):
    keys(methods, methods_for_version(version))
    population = None
    training_population = None
    regression = source["results"].get("regression")
    require(regression is not None or source["status"] != "ready", "scenario source lacks regression population")
    original = regression["metrics"] if regression is not None else None
    for method, result in methods.items():
        if not applicable(method, scenario, version):
            require(result is None, "unsupported scenario must be blank")
            continue
        require(isinstance(result, dict), "supported scenario is missing its outcome")
        _result(result, row, scenario, method, version, train_percent=train_percent)
        current = (
            result["scoring_population_sha256"],
            result["test_count"],
            tuple(result["metrics"][kind]["measured_count"] for kind in WORKLOADS[1:]),
        )
        require(population is None or population == current, "methods scored different scenario populations")
        population = current
        if scenario in ("b", "bc"):
            current_training = (result["training_population_sha256"], result["training_count"])
            require(
                training_population is None or training_population == current_training,
                "methods trained on different scenario populations",
            )
            training_population = current_training
        for kind in WORKLOADS[1:]:
            count = original[kind]["measured_count"] if original is not None else 0
            measured = result["metrics"][kind]["measured_count"]
            if scenario == "b":
                expected = count - (count * train_percent + 99) // 100
                require(measured == expected, "incorrect distributed holdout")
            elif scenario == "bc":
                require(measured <= count, "temporal holdout exceeds source workload population")
            else:
                require(measured == count, "scenario skipped target workload observations")
        if scenario not in ("b", "bc"):
            require(result["test_count"] == row["measurement_count"], "scenario skipped target observations")
    return population, training_population


def validate_scenario_row(row, source, version):
    """Validate one row against an already validated complete source-summary row."""
    require(type(version) is int and version == SCHEMA_VERSION, "unsupported scenario schema")
    keys(row, (*ROW_FIELDS, "results", *(("split_results",) if version >= 4 else ())))
    require(all(row[field] == source[field] for field in ROW_FIELDS), "scenario source membership mismatch")
    keys(row["results"], SCENARIOS)
    population_counts = {}

    def validate_methods(methods, scenario, train_percent):
        population, training = _validate_method_results(methods, row, source, scenario, version, train_percent)
        for selected in (population, training):
            if selected is None or selected[0] is None:
                continue
            identity, count = selected[:2]
            require(
                population_counts.setdefault(identity, count) == count,
                "population hash has inconsistent observation counts",
            )
        return population, training

    for scenario, methods in row["results"].items():
        validate_methods(methods, scenario, DEFAULT_TRAIN_PERCENT)
    if version >= 4:
        keys(row["split_results"], tuple(str(percent) for percent in TRAIN_PERCENTAGES))
        boundaries = {}
        for percent in TRAIN_PERCENTAGES:
            selected = row["split_results"][str(percent)]
            keys(selected, ("b", "bc"))
            if percent == DEFAULT_TRAIN_PERCENT:
                require(
                    all(selected[scenario] == row["results"][scenario] for scenario in ("b", "bc")),
                    "default split differs from main scenario results",
                )
            for scenario, methods in selected.items():
                populations = validate_methods(methods, scenario, percent)
                if scenario == "bc":
                    boundary = row["measurement_count"] * percent // 100
                    require(
                        boundaries.setdefault(boundary, populations) == populations,
                        "identical temporal boundaries have different populations",
                    )
    return row


def validate_scenarios(document, summary):
    """Bind a complete scenario matrix to its already published measurement snapshot."""
    validate_summary(summary)
    keys(document, ("schema_version", "protocol_id", "snapshot", "methods", "scenarios", "protocol_metadata", "rows"))
    require(
        type(document["schema_version"]) is int and document["schema_version"] == SCHEMA_VERSION,
        "unsupported scenario schema",
    )
    version = document["schema_version"]
    require(document["protocol_id"] == f"fpm-scenarios-v{version}", "unsupported scenario protocol")
    require(document["snapshot"] == summary["snapshot"], "scenario snapshot mismatch")
    require(
        document["methods"] == list(methods_for_version(version)) and document["scenarios"] == list(SCENARIOS),
        "wrong scenario matrix",
    )
    require(
        document["protocol_metadata"] == PROTOCOL_METADATA,
        "scenario protocol description mismatch",
    )
    require(
        isinstance(document["rows"], list) and len(document["rows"]) == len(summary["rows"]), "missing scenario rows"
    )
    identities = set()
    source_rows = {(row["configuration_id"], row["snapshot_id"]): row for row in summary["rows"]}
    for row in document["rows"]:
        identity = (row["configuration_id"], row["snapshot_id"])
        require(identity not in identities and identity in source_rows, "invalid scenario configuration identity")
        identities.add(identity)
        validate_scenario_row(row, source_rows[identity], version)
    return document
