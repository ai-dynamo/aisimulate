# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executed public FPM evidence over real metadata and synthetic Parquets."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Reuse the config-only, verified unregistered architecture and its guards.
import test_fpm_profile_workflow as workflow

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.detail import build_prediction_details
from aisimulate.recommend import run_recommendation
from aisimulate.runner import EngineReplayRunnerFactory, _normalize_engine_replay_report
from aisimulate.sweeper.replay import ReplayOutputRequirements
from aisimulate.sweeper.result import SweepResult
from aisimulate_core.sdk import RustForwardPassPerfModel

pytestmark = pytest.mark.unit

profile = workflow.profile
timing_systems = workflow.timing_systems
forbid_registered_model = workflow.forbid_registered_model


@pytest.fixture
def evidence_systems(timing_systems):
    path = Path(timing_systems) / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    templates = [row for row in pq.read_table(path).to_pylist() if row["workload_kind"] == "prefill"]
    rows = []
    # Prefill token interpolation at 6 lies 1/4 along [4,12]; KV=2 is
    # 1/4 along [0,8]. The four weights are 9/16,3/16,3/16,1/16.
    points = [
        ("prefill", 1, 1, 0, 2.0),
        ("prefill", 1, 4, 0, 10.0),
        ("prefill", 1, 12, 0, 18.0),
        ("prefill", 1, 4, 8, 30.0),
        ("prefill", 1, 12, 8, 46.0),
        ("decode", 1, 0, 0, 3.0),
        ("decode", 1, 0, 1, 4.0),
        ("decode", 1, 0, 64, 67.0),
        ("decode", 2, 0, 4, 10.0),
        ("decode", 2, 0, 20, 26.0),
        ("decode", 4, 0, 8, 20.0),
        ("decode", 4, 0, 24, 52.0),
        # A decreasing measured curve exercises marginal-decode clamping.
        ("decode", 6, 0, 12, 30.0),
        ("decode", 6, 0, 24, 10.0),
    ]
    for template in templates:
        for phase, batch, tokens, kv, latency in points:
            rows.append(
                {
                    **template,
                    "workload_kind": phase,
                    "batch_size": batch,
                    "total_prefill_tokens": tokens,
                    "total_kv_read_tokens": kv,
                    "latency_ms": latency,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    metadata.update(row_count=len(rows), parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    return timing_systems


def canonical(profile, systems):
    return {
        "model": profile["model"],
        "fpm_profile": profile,
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "worker_type": "aggregated",
        "tp": 2,
        "moe_tp_size": 2,
        "moe_ep_size": 1,
        "systems_paths": [systems],
        "estimation_mode": "fpm_interpolation",
        "estimator_config": {"fpm_interpolation": {"method": "direct"}, "correction": {"min_observations": 1}},
    }


def prefill(tokens=4, kv=0):
    return {
        "scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": tokens, "sum_prefill_kv_tokens": kv}
    }


def decode(batch=3, kv=12):
    return {"scheduled_requests": {"num_decode_requests": batch, "sum_decode_kv_tokens": kv}}


@pytest.mark.parametrize("collect_coverage", [False, True])
@pytest.mark.parametrize(
    "metrics,latency,resolution,weights,measurements",
    [
        (prefill(), 10.0, "exact_lookup", [1.0], [(1, 4, 0, 10)]),
        (prefill(6), 12.0, "within_curve_interpolation", [0.75, 0.25], [(1, 4, 0, 10), (1, 12, 0, 18)]),
        (
            prefill(6, 2),
            17.5,
            "cross_kv_interpolation",
            [0.5625, 0.1875, 0.1875, 0.0625],
            [(1, 4, 0, 10), (1, 12, 0, 18), (1, 4, 8, 30), (1, 12, 8, 46)],
        ),
        (decode(2, 4), 10.0, "exact_lookup", [1.0], [(2, None, 4, 10)]),
        (decode(2, 12), 18.0, "within_curve_interpolation", [0.5, 0.5], [(2, None, 4, 10), (2, None, 20, 26)]),
        (
            decode(),
            23.0,
            "cross_batch_interpolation",
            [0.25, 0.25, 0.375, 0.125],
            [(2, None, 4, 10), (2, None, 20, 26), (4, None, 8, 20), (4, None, 24, 52)],
        ),
    ],
)
def test_canonical_detailed_query_support(
    profile, evidence_systems, metrics, latency, resolution, weights, measurements, collect_coverage
):
    config = canonical(profile, evidence_systems)
    config["estimator_config"]["fpm_interpolation"]["collect_coverage"] = collect_coverage
    model = RustForwardPassPerfModel.best_available(config)
    detailed = model.estimate_forward_pass_detailed(metrics)
    assert detailed["latency_ms"] == model.estimate_forward_pass_time_ms(metrics) == latency
    assert detailed["native_latency_ms"] == latency
    assert detailed["correction_factor"] == 1.0
    evidence = detailed["ranks"][0]["queries"][0]
    assert evidence["model_path"] == profile["model"]
    assert evidence["resolution"] == resolution
    assert evidence["latency_ms"] == latency
    assert evidence["decode_baseline"] is False
    assert [point["weight"] for point in evidence["support"]] == weights
    assert [
        (
            point["coordinates"]["batch_size"],
            point["coordinates"]["total_prefill_tokens"],
            point["coordinates"]["total_kv_read_tokens"],
            point["latency_ms"],
        )
        for point in evidence["support"]
    ] == measurements
    assert json.loads(json.dumps(detailed)) == detailed
    coverage = model.fpm_query_coverage()
    if collect_coverage:
        assert coverage["queries"] == {
            "measured": 2 if resolution == "exact_lookup" else 0,
            "interpolated": 0 if resolution == "exact_lookup" else 2,
            "unsupported": 0,
        }
    else:
        assert coverage is None


def test_detailed_coverage_retains_mixed_baseline_and_unsupported_queries(profile, evidence_systems):
    config = canonical(profile, evidence_systems)
    config["estimator_config"]["fpm_interpolation"]["collect_coverage"] = True
    model = RustForwardPassPerfModel.best_available(config)
    mixed = {"scheduled_requests": {**prefill()["scheduled_requests"], **decode()["scheduled_requests"]}}
    # The measured 10 ms prefill plus interpolated decode 23 ms minus its
    # 15 ms interpolated floor is 18 ms. All three native lookups count once.
    detailed = model.estimate_forward_pass_detailed(mixed)
    assert detailed["latency_ms"] == 18.0
    assert len(detailed["ranks"][0]["queries"]) == 3
    coverage = model.fpm_query_coverage()
    assert coverage["queries"] == {"measured": 1, "interpolated": 2, "unsupported": 0}
    assert coverage["mixed_decode_baseline"] == {"measured": 0, "interpolated": 1, "unsupported": 0}
    with pytest.raises(Exception, match="unsupported|outside"):
        model.estimate_forward_pass_detailed(prefill(20))
    assert model.estimate_forward_pass_detailed({})["latency_ms"] == 0.0
    coverage = model.fpm_query_coverage()
    assert coverage["queries"] == {"measured": 1, "interpolated": 2, "unsupported": 1}
    assert coverage["gaps"][0]["coordinates"] == {
        "batch_size": 1.0,
        "total_prefill_tokens": 20.0,
        "total_kv_read_tokens": 0.0,
    }


def test_mixed_baseline_rank_max_correction_and_clamp(profile, evidence_systems):
    model = RustForwardPassPerfModel.best_available(canonical(profile, evidence_systems))
    mixed = {"scheduled_requests": {**prefill()["scheduled_requests"], **decode()["scheduled_requests"]}}
    metrics = [mixed, decode(4, 24)]
    detail = model.estimate_forward_pass_detailed(metrics)
    assert detail["latency_ms"] == model.estimate_forward_pass_time_ms(metrics) == 52.0
    assert detail["max_rank"] == 1
    rank = detail["ranks"][0]
    assert [
        rank[key] for key in ("prefill_ms", "decode_ms", "decode_baseline_ms", "marginal_decode_ms", "latency_ms")
    ] == [10, 23, 15, 8, 18]
    baseline = rank["queries"][2]
    assert baseline["decode_baseline"] is True
    assert baseline["query"]["total_kv_read_tokens"] == 12
    assert [(p["coordinates"]["total_kv_read_tokens"], p["weight"]) for p in baseline["support"]] == [
        (4, 0.5),
        (8, 0.5),
    ]
    observed = deepcopy(metrics)
    observed[1]["wall_time"] = 0.104  # seconds: twice the 52 ms native maximum
    model.tune_with_fpms([observed])
    corrected = model.estimate_forward_pass_detailed(metrics)
    assert corrected["correction_factor"] == 2.0
    assert corrected["latency_ms"] == model.estimate_forward_pass_time_ms(metrics) == 104.0
    assert corrected["ranks"] == detail["ranks"]
    clamped = {"scheduled_requests": {**prefill()["scheduled_requests"], **decode(6, 24)["scheduled_requests"]}}
    detail = model.estimate_forward_pass_detailed(clamped)
    assert detail["ranks"][0]["decode_ms"] == 10
    assert detail["ranks"][0]["decode_baseline_ms"] == 30
    assert detail["ranks"][0]["marginal_decode_ms"] == 0
    assert detail["ranks"][0]["latency_ms"] == 10


def test_uncovered_empty_and_non_fpm_results(profile, evidence_systems):
    config = canonical(profile, evidence_systems)
    model = RustForwardPassPerfModel.best_available(config)
    for query in (prefill(20), prefill(4, 20), decode(3, 30)):
        for method in (model.estimate_forward_pass_time_ms, model.estimate_forward_pass_detailed):
            with pytest.raises(Exception, match="unsupported|outside"):
                method(query)
    empty = model.estimate_forward_pass_detailed({})
    assert empty == {
        "latency_ms": 0.0,
        "native_latency_ms": 0.0,
        "correction_factor": 1.0,
        "max_rank": None,
        "ranks": [],
    }
    config["estimation_mode"] = "fpm_regression"
    regression = RustForwardPassPerfModel.best_available(config)
    assert regression.estimate_forward_pass_detailed(decode()) == {
        "latency_ms": None,
        "native_latency_ms": None,
        "correction_factor": None,
        "max_rank": None,
        "ranks": [],
    }


def prediction(profile, systems):
    return {
        "engine": {**workflow._engine(profile), "systems_paths": [systems]},
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 1, "output_tokens": 3},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 3},
        },
    }


def records(evidence):
    return [
        record
        for provider in evidence["providers"]
        for phase in provider["phases"]
        for operation in phase["operations"]
        for record in operation["fpm_estimates"]
    ]


@pytest.mark.parametrize("collect_coverage", [False, True])
def test_native_prediction_replay_and_source_detail_roundtrip(profile, evidence_systems, collect_coverage):
    import jsonschema

    raw = prediction(profile, evidence_systems)
    raw["engine"]["workers"]["aggregated"]["timing"]["estimator_config"]["fpm_interpolation"]["collect_coverage"] = (
        collect_coverage
    )
    config = CorePredictionConfig.model_validate(raw)
    # Round-trip the saved public config before lowering it to the real runner.
    config = CorePredictionConfig.model_validate_json(config.model_dump_json())
    spec = prediction_to_replay_spec(config)
    runner = EngineReplayRunnerFactory().create(0)
    try:
        plain = runner.run(spec, output_requirements=ReplayOutputRequirements(include_raw_report=True))
        report = runner.run(spec, output_requirements=ReplayOutputRequirements(capture_performance_diagnostics=True))
    finally:
        runner.close()
    assert "fpm_query_evidence" not in plain.metadata
    for replay in (plain, report):
        native = replay.metadata["native_report"]
        if collect_coverage:
            assert native["fpm_query_coverage"]["status"] == "covered"
            assert native["fpm_query_coverage"]["queries"] == {"measured": 6, "interpolated": 3, "unsupported": 0}
        else:
            assert "fpm_query_coverage" not in native
    wall_clock_metrics = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
    assert {k: v for k, v in report.metrics.items() if k not in wall_clock_metrics} == {
        k: v for k, v in plain.metrics.items() if k not in wall_clock_metrics
    }
    assert report.metrics["power_w"] is None
    assert report.metrics["power_coverage"] is None
    evidence = report.metadata["fpm_query_evidence"]
    assert evidence["aggregation"] == "identical_estimates_with_invocation_counts"
    estimates = records(evidence)
    assert len(estimates) == 3  # prefill emits the first token; two subsequent decode steps
    assert all(record["count"] == 3 for record in estimates)
    assert sum(record["count"] for record in estimates) == 9
    assert all(record["latency_scale"] == 1.0 for record in estimates)
    assert {q["resolution"] for r in estimates for rank in r["estimate"]["ranks"] for q in rank["queries"]} == {
        "exact_lookup",
        "within_curve_interpolation",
    }
    native = report.metadata["native_report"]
    detail = build_prediction_details(native, ["source"])
    schema = json.loads((Path(__file__).resolve().parents[1] / "docs/cli/prediction-details.schema.json").read_text())
    jsonschema.validate(detail, schema)
    jsonschema.validate(build_prediction_details(native, ["time"]), schema)
    assert detail["sections"]["source"]["status"] == "available"
    operations = [op for phase in detail["sections"]["source"]["phases"] for op in phase["operations"]]
    assert all(op["source"] == "silicon" and op["fpm_estimates"] for op in operations)
    assert all(op["sol"] is None for phase in native["performance_diagnostics"]["phases"] for op in phase["operations"])
    normalized = _normalize_engine_replay_report(json.loads(json.dumps(native)), include_native_report=False)
    assert "native_report" not in normalized.metadata
    assert normalized.metadata["fpm_query_evidence"] == evidence


@pytest.mark.parametrize("supervised", [False, True])
def test_public_recommendation_retains_requested_evidence(profile, evidence_systems, monkeypatch, supervised):
    # Keep orchestration in this process so the analytical-graph guards remain
    # active; estimation, candidate scoring and replay use the real native API.
    if not supervised:
        monkeypatch.setattr("aisimulate.supervision.in_supervised_process", lambda: True)
        monkeypatch.setattr("aisimulate.resources.GuardedRunnerFactory.admit_wave", None)
    raw = prediction(profile, evidence_systems)
    raw["engine"]["mode"] = "aggregated"
    raw["engine"]["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1}
    config = CoreRecommendationConfig.model_validate(raw)
    result = run_recommendation(
        config,
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(),
        show_progress=False,
        output_requirements=ReplayOutputRequirements(capture_performance_diagnostics=True),
    )
    assert result.counts.feasible == 1
    saved = SweepResult.model_validate_json(result.model_dump_json())
    metadata = saved.candidates[0].provenance.runner_metadata
    evidence = metadata["fpm_query_evidence"]
    assert records(evidence)
    assert all(record["count"] == 3 for record in records(evidence))
    assert saved.candidates[0].metrics["power_w"] is None


def test_prediction_cli_saves_direct_query_evidence(profile, evidence_systems, tmp_path, capsys):
    import aisimulate.main as cli

    config = tmp_path / "prediction.json"
    config.write_text(json.dumps(prediction(profile, evidence_systems)))
    output = tmp_path / "output"
    assert (
        cli.main(
            ["predict", "--config", str(config), "--output-dir", str(output), "--detail", "source", "--format", "json"]
        )
        == 0
    )
    displayed = json.loads(capsys.readouterr().out)
    assert displayed["details"]["sections"]["source"]["status"] == "available"
    saved = json.loads((output / "prediction.json").read_text())
    assert records(saved["fpm_query_evidence"])
    assert saved["details"] == displayed["details"]


@pytest.fixture
def native_prediction_report(profile, evidence_systems):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction(profile, evidence_systems)))
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(spec, output_requirements=ReplayOutputRequirements(capture_performance_diagnostics=True))
    finally:
        runner.close()
    return report.metadata["native_report"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("estimate",), {}),
        (("estimate", "latency_ms"), float("inf")),
        (("estimate", "native_latency_ms"), float("nan")),
        (("estimate", "correction_factor"), -1),
        (("estimate", "max_rank"), True),
        (("estimate", "max_rank"), 0.5),
        (("estimate", "ranks"), {}),
        (("estimate", "ranks", 0), None),
        (("estimate", "ranks", 0, "rank"), -1),
        (("estimate", "ranks", 0, "rank"), True),
        (("estimate", "ranks", 0, "latency_ms"), True),
        (("estimate", "ranks", 0, "prefill_ms"), "1"),
        (("estimate", "ranks", 0, "decode_ms"), float("inf")),
        (("estimate", "ranks", 0, "decode_baseline_ms"), False),
        (("estimate", "ranks", 0, "marginal_decode_ms"), -1),
        (("estimate", "ranks", 0, "queries"), {}),
        (("estimate", "ranks", 0, "queries", 0), []),
        (("estimate", "ranks", 0, "queries", 0, "phase"), "generation"),
        (("estimate", "ranks", 0, "queries", 0, "model_path"), 1),
        (("estimate", "ranks", 0, "queries", 0, "model_path"), ""),
        (("estimate", "ranks", 0, "queries", 0, "decode_baseline"), 0),
        (("estimate", "ranks", 0, "queries", 0, "query"), {}),
        (("estimate", "ranks", 0, "queries", 0, "query", "batch_size"), True),
        (("estimate", "ranks", 0, "queries", 0, "query", "total_prefill_tokens"), -1),
        (("estimate", "ranks", 0, "queries", 0, "query", "total_kv_read_tokens"), float("inf")),
        (("estimate", "ranks", 0, "queries", 0, "resolution"), "nearest"),
        (("estimate", "ranks", 0, "queries", 0, "latency_ms"), float("nan")),
        (("estimate", "ranks", 0, "queries", 0, "support"), []),
        (("estimate", "ranks", 0, "queries", 0, "support"), {}),
        (("estimate", "ranks", 0, "queries", 0, "support", 0), None),
        (("estimate", "ranks", 0, "queries", 0, "support", 0, "coordinates"), []),
        (("estimate", "ranks", 0, "queries", 0, "support", 0, "coordinates", "batch_size"), -1),
        (("estimate", "ranks", 0, "queries", 0, "support", 0, "coordinates", "total_prefill_tokens"), "1"),
        (("estimate", "ranks", 0, "queries", 0, "support", 0, "coordinates", "total_kv_read_tokens"), float("nan")),
        (("estimate", "ranks", 0, "queries", 0, "support", 0, "latency_ms"), "1"),
        (("estimate", "ranks", 0, "queries", 0, "support", 0, "weight"), float("nan")),
        (("estimate", "ranks", 0, "queries", 0, "support", 0, "weight"), -0.1),
        (("estimate", "ranks", 0, "queries", 0, "support", 0, "weight"), 1.1),
    ],
)
def test_prediction_details_reject_malformed_fpm_evidence(native_prediction_report, path, value):
    record = native_prediction_report["performance_diagnostics"]["phases"][0]["operations"][0]["fpm_estimates"][0]
    for key in path[:-1]:
        record = record[key]
    record[path[-1]] = value
    expected_path = "performance_diagnostics.phases[0].operations[0].fpm_estimates[0]" + "".join(
        f"[{key}]" if isinstance(key, int) else f".{key}" for key in path
    )
    for section in ("source", "time"):
        with pytest.raises(ValueError) as error:
            build_prediction_details(native_prediction_report, (section,))
        assert str(error.value).startswith(expected_path)


def test_prediction_details_reject_missing_and_unknown_fpm_fields(native_prediction_report):
    prefix = ("performance_diagnostics", "phases", 0, "operations", 0, "fpm_estimates", 0)
    for path in [
        (),
        ("estimate",),
        ("estimate", "ranks", 0),
        ("estimate", "ranks", 0, "queries", 0),
        ("estimate", "ranks", 0, "queries", 0, "query"),
        ("estimate", "ranks", 0, "queries", 0, "support", 0),
        ("estimate", "ranks", 0, "queries", 0, "support", 0, "coordinates"),
    ]:
        record = native_prediction_report
        for key in (*prefix, *path):
            record = record[key]
        for field in (*record, "unknown"):
            invalid = deepcopy(native_prediction_report)
            target = invalid
            for key in (*prefix, *path):
                target = target[key]
            if field == "unknown":
                target[field] = None
            else:
                del target[field]
            expected_path = "".join(
                f"[{key}]" if isinstance(key, int) else f".{key}" for key in (*prefix, *path, field)
            ).lstrip(".")
            for section in ("source", "time"):
                with pytest.raises(ValueError) as error:
                    build_prediction_details(invalid, (section,))
                assert str(error.value).startswith(f"{expected_path} must be")


def test_prediction_details_preserve_valid_nested_fpm_evidence(profile, evidence_systems, native_prediction_report):
    import jsonschema

    config = canonical(profile, evidence_systems)
    model = RustForwardPassPerfModel.best_available(config)
    estimates = [
        model.estimate_forward_pass_detailed(metrics)
        for metrics in (prefill(), prefill(6), prefill(6, 2), decode(), {})
    ]
    config["estimation_mode"] = "fpm_regression"
    estimates.append(RustForwardPassPerfModel.best_available(config).estimate_forward_pass_detailed(decode()))
    operation = native_prediction_report["performance_diagnostics"]["phases"][0]["operations"][0]
    operation["fpm_estimates"] = [{"estimate": estimate, "count": 1, "latency_scale": 1.0} for estimate in estimates]
    original = deepcopy(native_prediction_report)
    details = build_prediction_details(native_prediction_report, ("source", "time"))
    schema = json.loads((Path(__file__).resolve().parents[1] / "docs/cli/prediction-details.schema.json").read_text())
    jsonschema.validate(details, schema)
    for section in (details["sections"]["source"], details["sections"]["time"]["diagnostics"]):
        published = section["phases"][0]["operations"][0]["fpm_estimates"]
        assert published == operation["fpm_estimates"]
        published[0]["estimate"]["ranks"].clear()
    assert native_prediction_report == original

    # FPM evidence remains optional for pre-existing diagnostics producers.
    del operation["fpm_estimates"]
    details = build_prediction_details(native_prediction_report, ("source", "time"))
    jsonschema.validate(details, schema)
    for section in (details["sections"]["source"], details["sections"]["time"]["diagnostics"]):
        assert "fpm_estimates" not in section["phases"][0]["operations"][0]
