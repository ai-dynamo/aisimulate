# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Withheld timings use the native interpolator without an analytical graph."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate.support.interpolation_validation import evaluate_interpolation_holdout
from aisimulate.support.schema import SupportRequest
from aisimulate_core.sdk import engine
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY

pytestmark = pytest.mark.unit


def _write_pair(path, rows, *, schema_version=7):
    pq.write_table(pa.Table.from_pylist(rows), path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": schema_version,
                "coordinate_system": "iteration_totals_balanced_v1",
                "measurement_policy": "dynamo_native_single_sample_v1",
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.25.1",
                "row_count": len(rows),
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "collector_attempt_ids": ["synthetic-validation-fixture"],
            }
        )
    )


def _execution_evidence(rows, directory, *, max_capture=512):
    """Synthetic source-bound metadata; the CLI tests exercise its native reader."""
    directory.mkdir()
    evidence = []
    for cell_id in sorted({row["cell_id"] for row in rows}):
        cell_rows = [row for row in rows if row["cell_id"] == cell_id]
        identity = {
            "source_plan_sha256": "a" * 64,
            "collector_attempt_id": "synthetic-source",
            "runtime_run_id": f"run-{cell_id}",
            "runtime_grid_digest": f"grid-{cell_id}",
        }
        points = []
        for row in cell_rows:
            row.update(identity)
            axis = row["total_prefill_tokens"] or row["batch_size"]
            captured = axis <= max_capture
            points.append(
                {
                    **{
                        key: row[key]
                        for key in ("workload_kind", "batch_size", "total_prefill_tokens", "total_kv_read_tokens")
                    },
                    "expected_cudagraph_mode": ("PIECEWISE" if row["workload_kind"] == "prefill" else "FULL")
                    if captured
                    else "NONE",
                    "expected_capture_size": axis if captured else None,
                }
            )
        config = {
            "mode": "FULL_AND_PIECEWISE",
            "capture_sizes": sorted({p["expected_capture_size"] for p in points if p["expected_capture_size"]}),
        }
        graphs = []
        for rank in range(cell_rows[0]["dp"]):
            path = directory / f"{cell_id}-{rank}.json"
            path.write_text(json.dumps({"cudagraph": config, "points": points}))
            graphs.append(
                {
                    "dp_rank": rank,
                    "config": copy.deepcopy(config),
                    "source": {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    },
                }
            )
        evidence.append({"cell_id": cell_id, "identity": identity, "points": points, "native_graph_config": graphs})
    return evidence


@pytest.fixture
def holdout_case(tmp_path, monkeypatch, request):
    def reject_graph(*_args, **_kwargs):
        raise AssertionError("holdout must not construct an analytical model or substitute op-level timings")

    monkeypatch.setattr(engine, "get_model", reject_graph)
    monkeypatch.setattr(engine, "build_model_config", reject_graph)
    topology = getattr(request, "param", "tp")
    is_moe = topology != "tp"
    profile = {
        "schema_version": 1,
        "model": "test/unregistered-holdout-model",
        "model_revision": "synthetic-immutable-v1",
        "architecture": "UnregisteredDecoderForCausalLM",
        "context_length": 8192,
        "num_experts": 8 if is_moe else 0,
        "provenance": "Synthetic metadata for native path tests; no silicon qualification.",
        "deployments": [
            {
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.25.1",
                "tp": 1 if topology == "dep" else 2,
                "dp": 2 if topology == "dep" else 1,
                "moe_tp": 1,
                "moe_ep": 2 if is_moe else 1,
                "gemm_quant_mode": "fp8",
                "moe_quant_mode": "fp8",
                "fmha_quant_mode": "bfloat16",
                "comm_quant_mode": "half",
                "kv_cache_dtype": "fp8",
                "resources": {
                    "kv_bytes_per_token": 10,
                    "cache_layout": "linear",
                    "max_num_tokens": 256,
                    "max_batch_size": 4,
                    "provenance": "Pending runtime memory; sufficient for standalone forward timing.",
                },
            }
        ],
    }
    deployment = profile["deployments"][0]
    onboarding = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe" if is_moe else "dense",
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": {
                "tensor_parallel": deployment["tp"],
                "attention_data_parallel": deployment["dp"],
                "moe_tensor_parallel": deployment["moe_tp"],
                "moe_expert_parallel": deployment["moe_ep"],
                "context_length": 8192,
            },
            "fpm_profile": profile,
        }
    )
    identity = onboarding.profile_deployment().model_dump(mode="json", exclude={"resources"})
    rows = []
    for phase in ("prefill", "decode"):
        for batch in (1, 2, 4):
            for prefill in (16, 32, 64, 128, 256) if phase == "prefill" else (0,):
                for kv in (0, 64, 256, 1024, 2048) if phase == "prefill" else (4, 16, 64, 256, 2048):
                    rows.append(
                        {
                            **identity,
                            **dict(zip(EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY, strict=True)),
                            "model_path": profile["model"],
                            "cell_id": f"synthetic-{phase}",
                            "weight_quantization": "synthetic",
                            "workload_kind": phase,
                            "partition_policy": "balanced_v1",
                            "batch_size": batch,
                            "total_prefill_tokens": prefill,
                            "total_kv_read_tokens": kv,
                            "latency_ms": 10 + batch / 4 + prefill / 100 + kv / 1000,
                            "kv_seed_regime": "real_kv",
                        }
                    )
    source = tmp_path / "source"
    source.mkdir()
    packaged = Path(engine.__file__).parents[1] / "systems/h200_sxm.yaml"
    (source / "h200_sxm.yaml").write_bytes(packaged.read_bytes())
    parquet = source / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    parquet.parent.mkdir(parents=True)
    _write_pair(parquet, rows)
    return onboarding, source, parquet, rows


@pytest.mark.parametrize("holdout_case", ["tp", "dep", "tep"], indirect=True)
def test_native_holdout_is_graph_independent_and_removes_all_coordinates(holdout_case, tmp_path):
    request, source, parquet, original_rows = holdout_case
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "assessment"
    report = evaluate_interpolation_holdout(request, systems_root=source, output_dir=output)
    assert not request.profile_deployment().resources.memory_ready
    assert report["status"] == "passed"
    assert report["serving_accuracy"] == "not_assessed"
    assert report["source_artifacts_unchanged"]
    assert before == {path: path.read_bytes() for path in before}
    assert report["native_query_coverage"]["queries"] == {
        "measured": 0,
        "interpolated": 18,
        "unsupported": 0,
    }
    for phase, expected in (("prefill", 15), ("decode", 3)):
        phase_report = report["phases"][phase]
        assert phase_report["selected_count"] == expected
        assert phase_report["absolute_relative_error"]["p95"] == pytest.approx(0, abs=1e-14)
        assert phase_report["absolute_relative_error"]["bias"] == pytest.approx(0, abs=1e-14)
    config = report["native_diagnostics"]["provenance"]["config"]
    assert config["estimation_mode"] == "fpm_interpolation"
    assert config["fallback_policy"] == "deny"
    assert config["systems_paths"] == [str(output / "systems")]
    assert config["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    remaining = pq.read_table(report["artifacts"]["training_parquet"]["path"]).to_pylist()
    assert len(remaining) == len(original_rows) - 18
    for point in report["predictions"]:
        assert not any(
            row["workload_kind"] == point["phase"]
            and all(row[name] == value for name, value in point["coordinates"].items())
            for row in remaining
        )
    plan = json.loads((output / "holdout-plan.json").read_text())
    assert plan["retained_envelope_anchors"]
    assert plan["capture_boundaries"]["status"] == "not_assessed"
    assert plan["inputs"]["parquet"]["sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert json.loads((output / "interpolation-validation.json").read_text()) == report


def test_selection_is_deterministic_and_input_order_independent(holdout_case, tmp_path):
    request, source, parquet, rows = holdout_case
    reports = []
    for index in range(3):
        if index == 1:
            _write_pair(parquet, list(reversed(rows)))
        reports.append(
            evaluate_interpolation_holdout(
                request, systems_root=source, output_dir=tmp_path / f"assessment-{index}", seed=42 if index < 2 else 43
            )
        )
    assert reports[0]["predictions"] == reports[1]["predictions"]
    assert reports[1]["predictions"] != reports[2]["predictions"]


def test_holdout_retains_both_sides_of_graph_eager_transition(holdout_case, tmp_path):
    request, source, parquet, rows = holdout_case
    template = next(row for row in rows if row["workload_kind"] == "prefill")
    # Measured campaign shape: the 513-token eager point is much slower than
    # the adjacent 512-token PIECEWISE point. Withholding it must not erase
    # the runtime transition that the full table already measured.
    timings = {256: 38.0, 512: 75.982, 513: 678.985, 1024: 718.120, 2048: 796.54}
    prefill = [
        {**template, "batch_size": 1, "total_prefill_tokens": tokens, "total_kv_read_tokens": 0, "latency_ms": timing}
        for tokens, timing in timings.items()
    ]
    rows = prefill + [row for row in rows if row["workload_kind"] == "decode"]
    evidence = _execution_evidence(rows, tmp_path / "native")
    _write_pair(parquet, rows)
    old = evaluate_interpolation_holdout(request, systems_root=source, output_dir=tmp_path / "old", seed=0)
    assert old["status"] == "failed"
    missed = next(point for point in old["predictions"] if point["phase"] == "prefill")
    assert missed["coordinates"]["total_prefill_tokens"] == 513
    assert missed["predicted_ms"] == pytest.approx(77.2361758)
    previous = {path: path.read_bytes() for path in (tmp_path / "old").rglob("*") if path.is_file()}
    before = {
        path: path.read_bytes() for root in (source, tmp_path / "native") for path in root.rglob("*") if path.is_file()
    }
    report = evaluate_interpolation_holdout(
        request, systems_root=source, output_dir=tmp_path / "assessment", seed=0, execution_evidence=evidence
    )
    remaining = pq.read_table(report["artifacts"]["training_parquet"]["path"]).to_pylist()
    retained_tokens = {row["total_prefill_tokens"] for row in remaining if row["workload_kind"] == "prefill"}
    assert {512, 513} <= retained_tokens
    assert report["native_query_coverage"]["queries"]["measured"] == 0
    assert report["capture_boundaries"]["status"] == "assessed"
    assert report["capture_boundaries"]["per_point_observed_dispatch"] == "unreported"
    assert {point["total_prefill_tokens"] for point in report["capture_boundaries"]["retained_anchors"]} == {512, 513}
    assert report["phases"]["prefill"]["selected_count"] == 1
    assert report["phases"]["prefill"]["absolute_relative_error"]["p95"] < 0.001
    assert before == {path: path.read_bytes() for path in before}
    assert previous == {path: path.read_bytes() for path in previous}


def test_boundary_selection_is_deterministic_with_ragged_curves_and_fake_rows(holdout_case, tmp_path):
    request, source, parquet, rows = holdout_case
    rows = [
        row
        for row in rows
        if not (
            row["workload_kind"] == "prefill"
            and row["total_kv_read_tokens"] == 64
            and row["total_prefill_tokens"] == 64
        )
    ]
    for row in rows:
        if (
            row["workload_kind"] == "prefill"
            and row["total_kv_read_tokens"] == 256
            and row["total_prefill_tokens"] == 64
        ):
            row["kv_seed_regime"] = "fake_fallback"
    evidence = _execution_evidence(rows, tmp_path / "native", max_capture=32)
    _write_pair(parquet, rows)
    first = evaluate_interpolation_holdout(
        request, systems_root=source, output_dir=tmp_path / "first", execution_evidence=evidence
    )
    _write_pair(parquet, list(reversed(rows)))
    reordered = copy.deepcopy(list(reversed(evidence)))
    for cell in reordered:
        cell["points"].reverse()
        cell["native_graph_config"].reverse()
    second = evaluate_interpolation_holdout(
        request, systems_root=source, output_dir=tmp_path / "second", execution_evidence=reordered
    )
    assert first["predictions"] == second["predictions"]
    assert first["capture_boundaries"] == second["capture_boundaries"]
    anchors = first["capture_boundaries"]["retained_anchors"]
    assert any(point["total_prefill_tokens"] == 128 and point["total_kv_read_tokens"] == 64 for point in anchors)
    assert not any(point["total_prefill_tokens"] == 64 and point["total_kv_read_tokens"] == 256 for point in anchors)
    assert first["native_query_coverage"]["queries"]["measured"] == 0


@pytest.mark.parametrize("missing", ["point_mode", "unknown_mode", "capture_size", "graph_config"])
def test_incomplete_boundary_evidence_does_not_claim_qualification(holdout_case, tmp_path, missing):
    request, source, parquet, rows = holdout_case
    evidence = _execution_evidence(rows, tmp_path / "native")
    cell = next(cell for cell in evidence if cell["cell_id"] == "synthetic-prefill")
    if missing == "graph_config":
        cell["native_graph_config"][0]["config"] = None
    elif missing == "unknown_mode":
        cell["points"][0]["expected_cudagraph_mode"] = "FUTURE_MODE"
    elif missing == "capture_size":
        cell["points"][0].pop("expected_capture_size")
    else:
        cell["points"][0].pop("expected_cudagraph_mode")
    _write_pair(parquet, rows)
    report = evaluate_interpolation_holdout(
        request, systems_root=source, output_dir=tmp_path / "assessment", execution_evidence=evidence
    )
    assert report["status"] == report["capture_boundaries"]["status"] == "incomplete"
    assert all(phase["status"] == "passed" for phase in report["phases"].values())
    assert report["capture_boundaries"]["issues"]
    assert report["capture_boundaries"]["retained_anchors"] == []


@pytest.mark.parametrize(
    "corruption", ["identity", "coordinates", "duplicate", "rank", "graph_rank", "capture", "source"]
)
@pytest.mark.parametrize("holdout_case", ["dep"], indirect=True)
def test_boundary_evidence_rejects_contradictions(holdout_case, tmp_path, corruption):
    request, source, parquet, rows = holdout_case
    evidence = _execution_evidence(rows, tmp_path / "native")
    _write_pair(parquet, rows)
    cell = next(cell for cell in evidence if cell["cell_id"] == "synthetic-prefill")
    if corruption == "identity":
        cell["identity"]["collector_attempt_id"] = "another-attempt"
    elif corruption == "coordinates":
        cell["points"].pop()
    elif corruption == "duplicate":
        cell["points"].append(cell["points"][0])
    elif corruption == "rank":
        cell["native_graph_config"][1]["dp_rank"] = 0
    elif corruption == "graph_rank":
        cell["native_graph_config"][1]["config"]["mode"] = "NONE"
    elif corruption == "capture":
        cell["points"][0]["expected_capture_size"] = 9999
    else:
        Path(cell["native_graph_config"][0]["source"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="holdout"):
        evaluate_interpolation_holdout(
            request, systems_root=source, output_dir=tmp_path / "assessment", execution_evidence=evidence
        )
    assert not (tmp_path / "assessment/holdout-plan.json").exists()


@pytest.mark.parametrize("with_evidence", [False, True])
def test_errors_are_against_withheld_measurements_with_editable_gates(holdout_case, tmp_path, with_evidence):
    request, source, parquet, rows = holdout_case
    evidence = _execution_evidence(rows, tmp_path / "native", max_capture=32) if with_evidence else None
    _write_pair(parquet, rows)
    baseline = evaluate_interpolation_holdout(
        request, systems_root=source, output_dir=tmp_path / "baseline", execution_evidence=evidence
    )
    for row in rows:
        if any(
            row["workload_kind"] == point["phase"]
            and all(row[name] == value for name, value in point["coordinates"].items())
            for point in baseline["predictions"]
        ):
            row["latency_ms"] *= 2
    _write_pair(parquet, rows)
    strict = evaluate_interpolation_holdout(
        request, systems_root=source, output_dir=tmp_path / "strict", execution_evidence=evidence
    )
    assert strict["status"] == "failed"
    for phase in ("prefill", "decode"):
        distribution = strict["phases"][phase]["absolute_relative_error"]
        assert distribution == pytest.approx({"p50": 0.5, "p95": 0.5, "max": 0.5, "bias": -0.5})
        assert strict["phases"][phase]["unsupported_count"] == 0
    relaxed = evaluate_interpolation_holdout(
        request,
        systems_root=source,
        output_dir=tmp_path / "relaxed",
        max_p95_relative_error=0.6,
        execution_evidence=evidence,
    )
    assert relaxed["status"] == "passed"
    assert relaxed["predictions"] == strict["predictions"]
    assert relaxed["policy"]["max_p95_relative_error"] == 0.6


@pytest.mark.parametrize("with_evidence", [False, True])
def test_ragged_unsupported_holdout_is_not_an_error_sample_or_a_pass(holdout_case, tmp_path, with_evidence):
    request, source, parquet, rows = holdout_case
    template = next(row for row in rows if row["workload_kind"] == "prefill")
    sparse_prefill = [
        {**template, "batch_size": 1, "total_prefill_tokens": tokens, "total_kv_read_tokens": kv}
        for tokens, kv in ((16, 0), (32, 0), (64, 128), (128, 256), (256, 256))
    ]
    rows = sparse_prefill + [row for row in rows if row["workload_kind"] == "decode"]
    evidence = _execution_evidence(rows, tmp_path / "native") if with_evidence else None
    _write_pair(parquet, rows)
    report = evaluate_interpolation_holdout(
        request, systems_root=source, output_dir=tmp_path / "assessment", execution_evidence=evidence
    )
    assert report["status"] == "failed"
    prefill = report["phases"]["prefill"]
    assert prefill["selected_count"] == prefill["unsupported_count"] == 1
    assert prefill["unsupported_fraction"] == 1
    assert prefill["absolute_relative_error"] is None
    assert report["native_query_coverage"]["queries"]["unsupported"] == 1
    assert report["predictions"][0]["status"] == "unsupported"
    assert "predicted_ms" not in report["predictions"][0]
    permissive = evaluate_interpolation_holdout(
        request,
        systems_root=source,
        output_dir=tmp_path / "permissive",
        max_unsupported_fraction=1.0,
        execution_evidence=evidence,
    )
    assert permissive["status"] == "failed"  # An entirely unsupported phase has no error assessment.


def test_sparse_tables_are_incomplete_and_fake_rows_are_not_holdouts(holdout_case, tmp_path):
    request, source, parquet, rows = holdout_case
    retained = []
    for phase in ("prefill", "decode"):
        phase_rows = [row for row in rows if row["workload_kind"] == phase and row["batch_size"] == 1]
        retained.extend([phase_rows[0], phase_rows[-1]])
        retained.append({**phase_rows[-1], "total_kv_read_tokens": 9000, "kv_seed_regime": "fake_fallback"})
    _write_pair(parquet, retained)
    report = evaluate_interpolation_holdout(request, systems_root=source, output_dir=tmp_path / "assessment")
    assert report["status"] == "incomplete"
    assert report["predictions"] == []
    assert all(phase["status"] == "not_assessed" for phase in report["phases"].values())
    plan = json.loads(Path(report["artifacts"]["holdout_plan"]["path"]).read_text())
    assert plan["excluded_fake_fallback_count"] == 2
    assert plan["removed_row_count"] == 0


@pytest.mark.parametrize("corruption", ["digest", "row_count", "identity", "duplicate", "coordinate_collision"])
def test_source_validation_remains_native_and_rejects_leaking_duplicates(holdout_case, tmp_path, corruption):
    request, source, parquet, rows = holdout_case
    if corruption in {"duplicate", "coordinate_collision"}:
        duplicate = copy.deepcopy(rows[0])
        if corruption == "coordinate_collision":
            duplicate["cell_id"] += "-other-DP-evidence"
            duplicate["weight_quantization"] = "other-source"
        rows.append(duplicate)
        _write_pair(parquet, rows)
    else:
        metadata_path = parquet.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            {"digest": {"parquet_sha256": "0" * 64}, "row_count": {"row_count": 1}, "identity": {"system": "b200_sxm"}}[
                corruption
            ]
        )
        metadata_path.write_text(json.dumps(metadata))
    before = parquet.read_bytes()
    with pytest.raises((ValueError, PerfDataNotAvailableError)):
        evaluate_interpolation_holdout(request, systems_root=source, output_dir=tmp_path / "assessment")
    assert parquet.read_bytes() == before
    assert not (tmp_path / "assessment/holdout-plan.json").exists()


def test_other_identity_and_legacy_metadata_are_preserved_without_becoming_holdouts(holdout_case, tmp_path):
    request, source, parquet, rows = holdout_case
    for row in rows:
        for name in EXECUTION_COLUMNS:
            row.pop(name)
    foreign = {**rows[0], "model_path": "test/another-model", "latency_ms": 9999}
    _write_pair(parquet, [*rows, foreign], schema_version=6)
    report = evaluate_interpolation_holdout(request, systems_root=source, output_dir=tmp_path / "assessment")
    assert report["status"] == "passed"
    remaining = pq.read_table(report["artifacts"]["training_parquet"]["path"]).to_pylist()
    assert foreign in remaining
    metadata = json.loads(Path(report["artifacts"]["training_metadata"]["path"]).read_text())
    assert metadata["schema_version"] == 6
    assert metadata["collector_attempt_ids"] == ["synthetic-validation-fixture"]


@pytest.mark.parametrize(
    "options",
    [
        {"max_points_per_phase": 0},
        {"max_points_per_phase": True},
        {"seed": -1},
        {"seed": 0.1},
        {"max_p95_relative_error": float("nan")},
        {"max_p95_relative_error": -0.1},
        {"max_unsupported_fraction": 1.1},
        {"max_unsupported_fraction": True},
    ],
)
def test_invalid_policy_fails_before_writing(holdout_case, tmp_path, options):
    request, source, _parquet, _rows = holdout_case
    with pytest.raises(ValueError):
        evaluate_interpolation_holdout(request, systems_root=source, output_dir=tmp_path / "assessment", **options)
    assert not (tmp_path / "assessment").exists()


@pytest.mark.parametrize("destination", ["source", "nested", "parent", "existing", "symlink"])
def test_assessment_never_overwrites_source_or_prior_outputs(holdout_case, tmp_path, destination):
    request, source, _parquet, _rows = holdout_case
    output = {
        "source": source,
        "nested": source / "assessment",
        "parent": tmp_path,
        "existing": tmp_path / "previous",
        "symlink": tmp_path / "link",
    }[destination]
    if destination == "existing":
        output.mkdir()
        (output / "evidence.json").write_text("preserve me")
    elif destination == "symlink":
        output.symlink_to(source, target_is_directory=True)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        evaluate_interpolation_holdout(request, systems_root=source, output_dir=output)
    assert before == {path: path.read_bytes() for path in before}
