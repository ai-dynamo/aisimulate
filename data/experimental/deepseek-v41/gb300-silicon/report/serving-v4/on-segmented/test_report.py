# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic reporting checks; no fixture is measured evidence or a published report."""

import hashlib
import io
import json
import tarfile

import compare_paired_outputs
import package_report
import pytest
import render_report


def test_paired_output_ids_are_bound_to_original_sse_and_missing_is_unavailable():
    planned = {"request_id": "r", "input_token_ids": [1, 2], "output_tokens": 1}
    planned["input_token_ids_sha256"] = compare_paired_outputs.digest([1, 2])
    data = {"nvext": {"completion_token_ids": [3]}}
    raw = {
        **planned,
        "done": True,
        "errors": [],
        "requested_output_tokens": 1,
        "summary": {"valid": True, "completion_tokens": 1},
        "events": [{"is_output": True, "token_ids": [3], "data": data, "raw_data": json.dumps(data)}],
    }
    assert compare_paired_outputs.returned_ids(raw, planned) == ([3], None)
    raw["events"][0]["token_ids"] = [4]
    with pytest.raises(ValueError, match="original HTTP frame"):
        compare_paired_outputs.returned_ids(raw, planned)
    raw["events"][0].pop("token_ids")
    assert compare_paired_outputs.returned_ids(raw, planned) == (None, {"unavailable_output_frames": [0]})


@pytest.mark.parametrize("mutation", ["duplicate_request", "wrong_trial", "wrong_run", "wrong_counts"])
def test_paired_public_evidence_rejects_relabeling(mutation):
    rows = [
        {
            "on_request_id": f"on-{i}",
            "off_request_id": f"off-{i}",
            "trial_index": i // 24,
            "trial_seed": 93051000 + i // 24,
            "status": "equal",
        }
        for i in range(960)
    ]
    value = {
        "schema": "dsv41.paired.returned-token-ids.v1",
        "paired_request_count": 960,
        "paired_cohort_count": 680,
        "paired_trial_count": 40,
        "sampling_exclusion": False,
        "timing_observations_unchanged": True,
        "physical_runs": {"on": "on-run"},
        "requests": rows,
        "requests_by_status": {"equal": 960},
    }
    report = {"segments": [{"physical_run_id": "on-run"}]}
    render_report.validate_paired_outputs(value, report)
    if mutation == "duplicate_request":
        rows[1]["on_request_id"] = rows[0]["on_request_id"]
    elif mutation == "wrong_trial":
        rows[0]["trial_seed"] += 1
    elif mutation == "wrong_run":
        value["physical_runs"]["on"] = "different-run"
    else:
        value["requests_by_status"] = {"different": 960}
    with pytest.raises(ValueError):
        render_report.validate_paired_outputs(value, report)


def test_descriptive_errors_do_not_cancel_and_keep_missing_denominator():
    value = render_report.descriptive([(10, 8), (10, 12)], 3)
    assert value["mean_signed_error_percent"] == pytest.approx(0)
    assert value["wape_percent"] == pytest.approx(20)
    assert value["predicted"] == 2 and value["observed"] == 3 and value["missing"] == 1
    assert not any("ci" in key for key in value)


def test_runtime_continuity_discloses_distinct_raw_identities_without_ci_pooling():
    reference = "1" * 64
    segments = [
        {
            "receipt": {
                "runtime_identity_sha256": reference,
                "runtime_identity_equivalence": {
                    "kind": "exact",
                    "actual_runtime_identity_sha256": reference,
                },
            }
        },
        {
            "receipt": {
                "runtime_identity_sha256": reference,
                "runtime_identity_equivalence": {
                    "kind": "reviewed_aggregated_bootstrap_port_only",
                    "actual_runtime_identity_sha256": "2" * 64,
                    "comparison_reference_runtime_identity_sha256": reference,
                },
            }
        },
    ]
    text = "\n".join(render_report.runtime_continuity_lines(segments))
    assert "1" * 16 in text and "2" * 16 in text
    assert "other 494" in text and "neither changes" in text and "pooling confidence intervals" in text
    segments[1]["receipt"]["runtime_identity_equivalence"]["actual_runtime_identity_sha256"] = reference
    with pytest.raises(ValueError, match="unknown runtime equivalence"):
        render_report.runtime_continuity_lines(segments)


@pytest.mark.parametrize("mutation", ["tokens", "cohort", "run", "cache", "grams", "summary"])
def test_content_diagnostics_reject_unbound_inputs_and_descriptive_relabeling(mutation):
    plan = {"run_id": "origin", "cohorts": []}
    report = {
        "input_files_sha256": {"logical_plan": "a"},
        "segments": [
            {
                "physical_run_id": "origin",
                "e2e_cases": [],
                "receipt": {
                    "source_inputs_sha256": {
                        "closed_audit": "b",
                        "execution": "c",
                        "combined_summary": "d",
                    }
                },
            }
        ],
    }
    value = {
        "schema": "dsv41.segment-content-controls.v1",
        "content_coverage_complete": True,
        "sampling_exclusion": False,
        "missing_cohort_ids": [],
        "logical_run_id": "origin",
        "logical_plan_sha256": "a",
        "bindings": [
            {
                "physical_run_id": "origin",
                "closed_audit_sha256": "b",
                "execution_sha256": "c",
                "original_summary_sha256": "d",
            }
        ],
        "records": [],
        "summary": {},
    }
    for purpose in ("engram-distinct-text", "engram-repeated-text"):
        repeated = purpose == "engram-repeated-text"
        tokens = [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3] if repeated else [2, 3, 1, 2, 3, 1]]
        requests = [
            {
                "request_id": f"{purpose}/{i}",
                "input_token_ids": t,
                "input_token_ids_sha256": compare_paired_outputs.digest(t),
            }
            for i, t in enumerate(tokens)
        ]
        case = {
            "cohort_id": purpose,
            "purpose": purpose,
            "trial_index": 0,
            "trial_seed": 93051000,
            "requests": requests,
        }
        plan["cohorts"].append(case)
        report["segments"][0]["e2e_cases"].append({"cohort_id": purpose})
        value["records"].append(
            {k: case[k] for k in ("cohort_id", "purpose", "trial_index", "trial_seed")}
            | {
                "physical_run_id": "origin",
                "request_ids": [r["request_id"] for r in requests],
                "input_hashes": [r["input_token_ids_sha256"] for r in requests],
                "initial_cached_tokens": [0, 0],
                "first_prefill_dispatch_ids": [1, 1],
                "same_first_prefill_dispatch": True,
                "cold_same_first_prefill_dispatch": True,
                "equal_sequences": repeated,
                "raw_ngram_overlap": {"3": dict(left_unique=3, right_unique=3, intersection=3, union=3)},
            }
        )
        value["summary"][purpose] = dict(
            observed_trials=1,
            planned_trials=1,
            equal_sequences=int(repeated),
            same_first_prefill_dispatch=1,
            cold_same_first_prefill_dispatch=1,
            same_raw_3gram_sets=1,
            initial_cache_patterns={"[0, 0]": 1},
        )
    render_report.validate_content_controls(value, report, plan)
    row = value["records"][0]
    if mutation == "tokens":
        row["input_hashes"][0] = "changed"
    elif mutation == "cohort":
        value["records"].pop()
    elif mutation == "run":
        row["physical_run_id"] = "other"
    elif mutation == "cache":
        row["initial_cached_tokens"] = [0, 256]
    elif mutation == "grams":
        row["raw_ngram_overlap"]["3"]["intersection"] = 0
    else:
        value["summary"]["engram-distinct-text"]["same_first_prefill_dispatch"] = 0
    with pytest.raises(ValueError):
        render_report.validate_content_controls(value, report, plan)


def test_native_common_support_keeps_lifecycle_identity_and_full_missing_denominator():
    reports = {}
    for mode in render_report.MODES:
        reports[mode] = {"segments": []}
        for physical, observed, predicted in (("run1", 10, 8), ("run2", 100, 50)):
            row = dict(dispatch_id=1, phase="decode", observed_ms=observed, predicted_ms=predicted, status="predicted")
            if mode == "silicon" and physical == "run2":
                row.update(status="unavailable", failure_type="ValueError", failure="uncovered")
                row.pop("predicted_ms")
            reports[mode]["segments"].append(
                {
                    "e2e_cases": [],
                    "trace_cases": [{"physical_run_id": physical, "cohort_id": "c", "intervals": [row]}],
                }
            )
    result = render_report.summarize(reports)
    assert result["common_native_intervals"] == 1
    assert result["trace"]["silicon"]["all"]["observed"] == 2
    assert result["trace"]["silicon"]["all"]["missing"] == 1
    assert result["trace_common_support"]["hybrid"]["all"]["wape_percent"] == pytest.approx(20)
    assert result["trace"]["hybrid"]["all"]["wape_percent"] == pytest.approx(100 * 52 / 110)


def test_no_prediction_is_not_zero_error():
    assert render_report.descriptive([], 3) == {"observed": 3, "predicted": 0, "missing": 3}


def test_prediction_count_cannot_exceed_observed_denominator():
    with pytest.raises(ValueError, match="denominator"):
        render_report.descriptive([(1, 2)], 0)


def test_packager_compares_actual_source_bytes_before_publication(monkeypatch):
    monkeypatch.setattr(package_report, "commit_file", lambda *args: b"actual source bytes")
    with pytest.raises(ValueError, match="actual comparison source"):
        package_report.bind_sources({"prediction_sources": {"engine_compiler": "0" * 64}}, ".", "a" * 40, ".", "b" * 40)


def test_archive_is_bound_to_actual_physical_run_and_admitted_closure(tmp_path):
    values = {
        "execution.json": {"run_id": "physical-A"},
        "closed-segment-audit.json": {"valid": True},
        "combined-summary.json": {"run_id": "physical-A", "cohorts": []},
        "status.json": {"worker_exit": 0},
    }
    path = tmp_path / "raw.tar"
    hashes = {}
    with tarfile.open(path, "w") as archive:
        for name, value in values.items():
            raw = json.dumps(value).encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
            hashes[name] = hashlib.sha256(raw).hexdigest()
    segment = {
        "physical_run_id": "physical-A",
        "receipt": {
            "source_inputs_sha256": {
                "execution": hashes["execution.json"],
                "closed_audit": hashes["closed-segment-audit.json"],
                "combined_summary": hashes["combined-summary.json"],
                "status": hashes["status.json"],
            }
        },
    }
    assert package_report.bind_archive(path, segment)["physical_run_id"] == "physical-A"
    segment["physical_run_id"] = "physical-B"
    with pytest.raises(ValueError, match="physical lifecycle"):
        package_report.bind_archive(path, segment)
    segment["physical_run_id"] = "physical-A"
    segment["receipt"]["source_inputs_sha256"]["closed_audit"] = "0" * 64
    with pytest.raises(ValueError, match="closure bytes"):
        package_report.bind_archive(path, segment)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_nonfinite_or_nonpositive_values_cannot_enter_statistics(value):
    with pytest.raises(ValueError, match="invalid metric"):
        render_report.descriptive([(1, value)], 1)


def test_privacy_walk_checks_nested_keys_and_values():
    assert list(package_report.private_strings({"source": [{"/Users/private/path": "nested"}]}))
    assert list(package_report.private_strings({"source": "job on node017"}))
    assert list(package_report.private_strings({"source": "https://service.internal/status"}))
    assert not list(package_report.private_strings({"failure": "no measured data at <private path>"}))


def test_native_observation_comparison_preserves_unreturned_overlap_and_order():
    rows = [
        {
            "cohort_id": "c",
            "intervals": [
                {"dispatch_id": 1, "phase": "decode", "observed_ms": 2, "native_work_role": "unreturned_overlap_output"}
            ],
        }
    ]
    assert render_report.native_view(rows) == [("c", [(1, "decode", 2, "unreturned_overlap_output")])]


def semantic_reports():
    plan = {"requested_trials": 100, "sampling_role": "main", "run_id": "origin", "cohorts": []}
    for trial in range(100):
        for purpose in ("one", "two"):
            plan["cohorts"].append(
                dict(
                    cohort_id=f"{trial}-{purpose}",
                    trial_index=trial,
                    trial_seed=93051000 + trial,
                    purpose=purpose,
                    comparison_role="primary",
                )
            )
    reports = {}
    for mode in render_report.MODES:
        e2e = [
            {**c, "status": "predicted", "observed": {"ttft_ms": 1}, "prediction": {"ttft_ms": 1}}
            for c in plan["cohorts"][:41]
        ]
        traces = [{**c, "intervals": []} for c in plan["cohorts"][:41]]
        reports[mode] = dict(
            schema="dsv41.closed.segments.comparison.v1",
            logical_run_id="origin",
            prediction_config=dict(
                model_name="deepseek-ai/DeepSeek-V4.1-Flash",
                system_name="gb300",
                backend="sglang",
                backend_version="0.0.0.dev0",
                tp_size=4,
                pp_size=1,
                attention_dp_size=1,
                moe_tp_size=4,
                moe_ep_size=1,
                decoder_replay=True,
                forward_model="op_level",
                database_mode=mode.upper(),
                strict_provenance=True,
                enable_shared_layer=False,
            ),
            summary=dict(
                pooled_confidence_intervals=None,
                statistical_precision_completed=False,
                fixed_plan_completed=False,
                coverage_complete=False,
                observed_primary_cohorts=41,
                predicted_e2e_cohorts=41,
                planned_primary_cohorts=200,
            ),
            prediction_sources={},
            systems_identity={},
            segments=[
                dict(
                    physical_run_id="segment",
                    receipt={"valid": True},
                    observed_original_cohorts=41,
                    e2e_cases=e2e,
                    trace_cases=traces,
                    statistics=dict(
                        complete_same_lifecycle_trial_indices=list(range(20)),
                        boundary_trial_indices=[20],
                        ci_scope="conditional on fixed wall-deadline",
                        complete_trial_e2e={},
                    ),
                )
            ],
        )
    return reports, plan


def test_partial_scope_requires_explicit_render_flag_and_keeps_boundary():
    reports, plan = semantic_reports()
    render_report.validate_reports(reports, plan, allow_partial=True)
    with pytest.raises(ValueError, match="incomplete main"):
        render_report.validate_reports(reports, plan, allow_partial=False)


@pytest.mark.parametrize(
    "mutation", ["pooled_ci", "lost_boundary", "wrong_seed", "changed_observation", "wrong_source", "false_complete"]
)
def test_invalid_reporting_scope_is_rejected(mutation):
    reports, plan = semantic_reports()
    report = reports["silicon"]
    if mutation == "pooled_ci":
        report["summary"]["pooled_confidence_intervals"] = [0, 1]
    elif mutation == "lost_boundary":
        report["segments"][0]["e2e_cases"].pop()
    elif mutation == "wrong_seed":
        report["segments"][0]["e2e_cases"][0]["trial_seed"] += 1
    elif mutation == "changed_observation":
        report["segments"][0]["e2e_cases"][0]["observed"]["ttft_ms"] = 2
    elif mutation == "wrong_source":
        report["prediction_sources"]["native_extension"] = "different"
    elif mutation == "false_complete":
        report["segments"][0]["statistics"]["complete_same_lifecycle_trial_indices"].append(20)
    with pytest.raises(ValueError):
        render_report.validate_reports(reports, plan, allow_partial=True)
