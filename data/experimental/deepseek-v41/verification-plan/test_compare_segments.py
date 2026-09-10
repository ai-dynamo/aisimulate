# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic semantic fixtures; these are not measured timing evidence."""

import uuid
from copy import deepcopy

import compare_segments as subject
import pytest
from test_compare_trace import fixture as native_fixture


def logical_plan(count=20):
    plan = {
        "sampling_role": "main",
        "dataset_role": "verification",
        "corpus_role": "primary",
        "run_id": str(uuid.uuid4()),
        "requested_trials": count,
        "cohorts": [],
    }
    for trial in range(count):
        for purpose in ("short", "prefix-warm-A", "prefix-reuse-B", "last"):
            cid = f"trial-{trial}/{purpose}"
            case = {
                "cohort_id": cid,
                "trial_index": trial,
                "trial_seed": 93051000 + trial,
                "purpose": purpose,
                "comparison_role": "setup" if purpose == "prefix-warm-A" else "primary",
                "sampling_role": "main",
                "corpus_role": "primary",
                "requests": [
                    {
                        "request_id": cid + "/r",
                        "input_token_ids": [1, 2, 3],
                        "input_token_ids_sha256": subject.trace.digest([1, 2, 3]),
                        "output_tokens": 1,
                    }
                ],
            }
            if purpose == "prefix-reuse-B":
                case["seed_cohort_id"] = f"trial-{trial}/prefix-warm-A"
            plan["cohorts"].append(case)
    return plan


def reseal(segment, logical):
    receipt, measurement = segment["receipt"], segment["measurement"]
    receipt.update(logical_plan_canonical_sha256=subject.trace.digest(logical))
    measurement["source_bindings"].update(
        audit_canonical_sha256=subject.trace.digest(segment["audit"]),
        plan_canonical_sha256=subject.trace.digest(segment["plan"]),
    )
    for name, key in (
        ("audit", "audit"),
        ("physical_plan", "plan"),
        ("client_summary", "client"),
        ("measurement", "measurement"),
        ("scheduler_receipt", "scheduler"),
    ):
        receipt[name + "_canonical_sha256"] = subject.trace.digest(segment[key])


def segment_fixture(logical, *, offset=0, length=None):
    run = str(uuid.uuid4()) if offset else logical["run_id"]
    cases, mappings = subject.expected_physical_cases(logical, offset, run)
    physical = deepcopy(logical)
    if offset:
        physical.update(
            run_id=run,
            logical_study_run_id=logical["run_id"],
            continuation=True,
            completed_main_cohorts_before_segment=offset,
            cohorts=cases,
        )
    cases = cases if length is None else cases[:length]
    template, _, measurement = native_fixture(diagnostic=True)
    audit = deepcopy(template)
    audit.update(active_rows=[], cohorts=[])
    client = {"run_id": run, "cohorts": []}
    for i, case in enumerate(cases):
        row = deepcopy(template["active_rows"][0])
        row.update(line=i + 1, dispatch_id=i + 1)
        rid = f"native-{i}"
        row["dispatch_requests"][0]["rid"] = rid
        proof = deepcopy(template["cohorts"][0])
        proof.update(cohort_id=case["cohort_id"], rows=[{"file": row["file"], "line": i + 1}])
        request = proof["native_request_proof"]["requests"][0]
        request.update(native_rid=rid, request_id=case["requests"][0]["request_id"])
        proof["native_request_proof"]["iteration_roles"] = {str(i + 1): "useful_request_work"}
        audit["active_rows"].append(row)
        audit["cohorts"].append(proof)
        observed = {k: case[k] for k in ("cohort_id", "trial_index", "trial_seed", "purpose")}
        observed.update(
            valid=True,
            started={"monotonic_ns": (i + 1) * 100000000},
            finished={"monotonic_ns": (i + 1) * 100000000 + 10000},
            output_tokens_per_second=100,
            cache_control={"policy": "native-clear-before-cohort", "acknowledgements": [{"status": "success"}]},
            requests=[
                {
                    "request_id": case["requests"][0]["request_id"],
                    "valid": True,
                    "input_token_ids_sha256": subject.trace.digest([1, 2, 3]),
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "started_monotonic_ns": (i + 1) * 100000000 + 1,
                }
            ],
        )
        if case.get("continuation_setup"):
            observed["continuation_setup"] = True
        if case["purpose"] == "prefix-reuse-B":
            observed["cache_control"] = {"policy": "preserve-A-prefix"}
        client["cohorts"].append(observed)
    producer = audit["producer_audits"][0]
    producer["run_id"] = run
    for key in ("attempted", "enqueued", "dequeued", "active_sent", "sent", "sequence_allocated", "send_attempted"):
        producer[key] = len(cases)
    producer["dispatch_audit"].update(
        started=len(cases),
        completed=len(cases),
        records=[
            {"dispatch_id": r["dispatch_id"], "requests": deepcopy(r["dispatch_requests"])}
            for r in audit["active_rows"]
        ],
    )
    measurement["run_id"] = run
    receipt = {
        "schema": "dsv41.closed.segment.admission.v1",
        "valid": True,
        "physical_run_id": run,
        "logical_run_id": logical["run_id"],
        "restore_seed_mappings": mappings,
    }
    for key in (
        "normalizer_source_sha256",
        "runtime_identity_sha256",
        "frozen_budget_file_sha256",
        "warmup_dispatch_proof_file_sha256",
        "stop_contract_file_sha256",
        "raw_evidence_inventory_sha256",
    ):
        receipt[key] = "e" * 64
    segment = {
        "audit": audit,
        "plan": physical,
        "client": client,
        "measurement": measurement,
        "scheduler": {"run_id": run},
        "receipt": receipt,
    }
    reseal(segment, logical)
    return segment


def test_actual_stable_on_plan_metadata_is_not_timing_evidence():
    # No private source is required by the public test; fixed seed rules are
    # independently exercised here, not attributed to an actual closed run.
    plan = logical_plan(100)
    assert len(subject.logical_contract(plan)) == 100


def test_partial_first_lifecycle_preserves_boundary_and_original_identity():
    plan = logical_plan()
    segment = segment_fixture(plan, length=6)
    before = deepcopy(segment)
    proof = subject.qualify_segment(plan, segment, 0)
    assert proof["next_offset"] == 6
    assert subject.complete_trial_indices(plan, proof["originals"]) == [0]
    assert segment == before
    assert proof["originals"][-1]["trial_index"] == 1


def runtime_equivalence(segment, *, reviewed=False):
    reference = segment["receipt"]["runtime_identity_sha256"]
    if not reviewed:
        return {"kind": "exact", "actual_runtime_identity_sha256": reference}
    return {
        "kind": "reviewed_aggregated_bootstrap_port_only",
        "actual_runtime_identity_sha256": "a" * 64,
        "comparison_reference_runtime_identity_sha256": reference,
        "control_source_transition": {"transition": "dsv41.continuation.aggregated-port-equivalence.v1"},
        "server_arguments_comparison": {"contract": "dsv41.continuation.aggregated-port-equivalence.v1"},
        "actual_installed_source_sha256": {"example/actual_source.py": "b" * 64},
        "continuation_file_sha256": "c" * 64,
        "review_addendum_sha256": "d" * 64,
        "review_addendum": {"schema": "dsv41.continuation.aggregated-port-addendum.v1"},
        "source_evidence_receipt_sha256": "1" * 64,
        "frozen_original_plan_sha256": "2" * 64,
        "original_closed_audit_sha256": "3" * 64,
        "original_progress_sha256": "4" * 64,
        "physical_plan_sha256": "5" * 64,
        "prior_completed_cohorts": 6,
        "remaining_cohorts": 74,
    }


def bind_runtime_equivalence(segment, plan, proof):
    segment["receipt"]["runtime_identity_equivalence"] = proof
    segment["measurement"]["source_bindings"]["runtime_identity_equivalence_sha256"] = subject.trace.digest(proof)
    reseal(segment, plan)


@pytest.mark.parametrize("reviewed", [False, True])
def test_bound_runtime_equivalence_preserves_raw_identity_and_all_observations(reviewed):
    plan = logical_plan()
    segment = segment_fixture(plan, length=6)
    bind_runtime_equivalence(segment, plan, runtime_equivalence(segment, reviewed=reviewed))
    before = deepcopy(segment)
    assert subject.qualify_segment(plan, segment, 0)["next_offset"] == 6
    assert segment == before


@pytest.mark.parametrize("missing", ["proof", "hash"])
def test_runtime_equivalence_requires_both_fields_even_after_outer_receipt_reseal(missing):
    plan = logical_plan()
    segment = segment_fixture(plan, length=6)
    bind_runtime_equivalence(segment, plan, runtime_equivalence(segment))
    if missing == "proof":
        del segment["receipt"]["runtime_identity_equivalence"]
    else:
        del segment["measurement"]["source_bindings"]["runtime_identity_equivalence_sha256"]
    reseal(segment, plan)
    with pytest.raises(ValueError, match="both be present"):
        subject.qualify_segment(plan, segment, 0)


def test_runtime_equivalence_content_tamper_is_rejected_without_other_receipt_changes():
    plan = logical_plan()
    segment = segment_fixture(plan, length=6)
    bind_runtime_equivalence(segment, plan, runtime_equivalence(segment, reviewed=True))
    segment["receipt"]["runtime_identity_equivalence"]["review_addendum"]["schema"] = "forged"
    # This nested receipt object is outside the existing measurement hash.
    with pytest.raises(ValueError, match="proof hash differs"):
        subject.qualify_segment(plan, segment, 0)


@pytest.mark.parametrize(
    "mutation",
    [
        "null",
        "unknown_kind",
        "extra_exact_field",
        "exact_reference",
        "invalid_actual_hash",
        "reviewed_reference",
        "concealed_actual",
        "missing_reviewed_field",
        "extra_reviewed_field",
        "invalid_continuation_hash",
        "invalid_addendum_hash",
        "unknown_control_schema",
        "unknown_argument_schema",
        "unknown_addendum_schema",
        "nonobject_addendum",
        "empty_sources",
        "nonobject_sources",
        "invalid_source_hash",
        "invalid_prior_count",
        "empty_remaining",
        "invalid_origin_proof_hash",
    ],
)
def test_resealed_equivalence_cannot_relabel_runtime_identity_or_schema(mutation):
    plan = logical_plan()
    segment = segment_fixture(plan, length=6)
    exact = mutation in {"null", "unknown_kind", "extra_exact_field", "exact_reference", "invalid_actual_hash"}
    proof = runtime_equivalence(segment, reviewed=not exact)
    if mutation == "null":
        proof = None
    elif mutation == "unknown_kind":
        proof["kind"] = "arbitrary_runtime_equivalence"
    elif mutation == "extra_exact_field":
        proof["ignored_difference"] = "kernel"
    elif mutation == "exact_reference":
        proof["actual_runtime_identity_sha256"] = "f" * 64
    elif mutation == "invalid_actual_hash":
        proof["actual_runtime_identity_sha256"] = True
    elif mutation == "reviewed_reference":
        proof["comparison_reference_runtime_identity_sha256"] = "f" * 64
    elif mutation == "concealed_actual":
        proof["actual_runtime_identity_sha256"] = proof["comparison_reference_runtime_identity_sha256"]
    elif mutation == "missing_reviewed_field":
        del proof["control_source_transition"]
    elif mutation == "extra_reviewed_field":
        proof["unreviewed_exception"] = "dtype"
    elif mutation == "invalid_continuation_hash":
        proof["continuation_file_sha256"] = "invalid"
    elif mutation == "invalid_addendum_hash":
        proof["review_addendum_sha256"] = None
    elif mutation == "unknown_control_schema":
        proof["control_source_transition"]["transition"] = "other"
    elif mutation == "unknown_argument_schema":
        proof["server_arguments_comparison"]["contract"] = "other"
    elif mutation == "unknown_addendum_schema":
        proof["review_addendum"]["schema"] = "other"
    elif mutation == "nonobject_addendum":
        proof["review_addendum"] = []
    elif mutation == "empty_sources":
        proof["actual_installed_source_sha256"] = {}
    elif mutation == "nonobject_sources":
        proof["actual_installed_source_sha256"] = []
    elif mutation == "invalid_source_hash":
        proof["actual_installed_source_sha256"]["example/actual_source.py"] = "invalid"
    elif mutation == "invalid_prior_count":
        proof["prior_completed_cohorts"] = True
    elif mutation == "empty_remaining":
        proof["remaining_cohorts"] = 0
    elif mutation == "invalid_origin_proof_hash":
        proof["original_progress_sha256"] = "invalid"
    bind_runtime_equivalence(segment, plan, proof)
    with pytest.raises(ValueError):
        subject.qualify_segment(plan, segment, 0)


def test_continuation_restore_seed_is_bound_to_actual_audit_and_original_token_ids():
    plan = logical_plan()
    segment = segment_fixture(plan, offset=2, length=3)
    proof = subject.qualify_segment(plan, segment, 2)
    assert proof["next_offset"] == 4
    assert proof["restores"][0]["actual_setup_cohort_id"] == segment["plan"]["cohorts"][0]["cohort_id"]
    assert proof["originals"][0]["seed_cohort_id"] == plan["cohorts"][1]["cohort_id"]
    assert subject.complete_trial_indices(plan, proof["originals"]) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "trial",
        "seed",
        "duplicate",
        "missing_boundary",
        "physical_run",
        "audit_hash",
        "native_token",
        "native_loss",
        "client_token",
        "cold_ack",
    ],
)
def test_segment_rejects_identity_and_native_proof_changes(mutation):
    plan = logical_plan()
    segment = segment_fixture(plan, length=6)
    if mutation == "trial":
        segment["client"]["cohorts"][0]["trial_index"] = 1
    elif mutation == "seed":
        segment["plan"]["cohorts"][0]["trial_seed"] += 1
    elif mutation == "duplicate":
        segment["client"]["cohorts"][1] = deepcopy(segment["client"]["cohorts"][0])
    elif mutation == "missing_boundary":
        segment["client"]["cohorts"].pop()
    elif mutation == "physical_run":
        segment["audit"]["producer_audits"][0]["run_id"] = str(uuid.uuid4())
    elif mutation == "audit_hash":
        segment["receipt"]["audit_canonical_sha256"] = "f" * 64
    elif mutation == "native_token":
        segment["audit"]["active_rows"][0]["dispatch_requests"][0]["input_token_ids_sha256"] = "f" * 64
    elif mutation == "native_loss":
        segment["audit"]["producer_audits"][0]["queue_full"] = 1
    elif mutation == "client_token":
        segment["client"]["cohorts"][0]["requests"][0]["input_token_ids_sha256"] = "f" * 64
    else:
        segment["client"]["cohorts"][0]["cache_control"]["acknowledgements"] = []
    if mutation != "audit_hash":
        reseal(segment, plan)
    with pytest.raises(ValueError):
        subject.qualify_segment(plan, segment, 0)


@pytest.mark.parametrize(
    "mutation", ["mapping", "setup_native_request", "setup_tokens", "wrong_offset", "a_not_immediate"]
)
def test_restore_cannot_be_replaced_with_previous_lifecycle_or_unwitnessed_seed(mutation):
    plan = logical_plan()
    segment = segment_fixture(plan, offset=2, length=3)
    if mutation == "mapping":
        segment["receipt"]["restore_seed_mappings"][0]["actual_setup_cohort_id"] = plan["cohorts"][1]["cohort_id"]
    elif mutation == "setup_native_request":
        segment["audit"]["cohorts"][0]["native_request_proof"]["requests"][0]["request_id"] = "old-request"
    elif mutation == "setup_tokens":
        segment["plan"]["cohorts"][0]["requests"][0]["input_token_ids"] = [9, 9, 9]
    elif mutation == "wrong_offset":
        segment["plan"]["completed_main_cohorts_before_segment"] = 1
    else:
        segment["client"]["cohorts"][0]["finished"]["monotonic_ns"] = 999999999999
    reseal(segment, plan)
    with pytest.raises(ValueError):
        subject.qualify_segment(plan, segment, 2)


def stats_rows(plan):
    e2e, trace = [], []
    for case in plan["cohorts"]:
        if case["comparison_role"] != "primary":
            continue
        identity = {k: case[k] for k in ("cohort_id", "purpose", "trial_index", "trial_seed")}
        e2e.append(identity | {"status": "predicted", "observed": {"ttft_ms": 10}, "prediction": {"ttft_ms": 11}})
        trace.append(identity | {"intervals": [{"status": "predicted", "observed_ms": 10, "predicted_ms": 11}]})
    return e2e, trace


def test_full_segment_ci_and_logical_completion_do_not_create_pooled_ci():
    plan = logical_plan()
    e2e, trace = stats_rows(plan)
    stats = subject.segment_statistics(plan, {"originals": plan["cohorts"]}, e2e, trace)
    assert len(stats["complete_same_lifecycle_trial_indices"]) == 20
    assert "observed_mean_bootstrap_ci95" in stats["complete_trial_e2e"]["short"]["metrics"]["ttft_ms"]
    report = {"e2e_cases": e2e, "trace_cases": trace}
    summary = subject.logical_summary(plan, [report], len(plan["cohorts"]))
    assert summary["fixed_plan_completed"] and summary["coverage_complete"]
    assert not summary["statistical_precision_completed"]
    assert summary["pooled_confidence_intervals"] is None
    assert "bootstrap" not in str(summary["e2e_descriptive"])
    assert "bootstrap" not in str(summary["trace_descriptive"])
    with pytest.raises(ValueError, match="duplicate cohort across"):
        subject.logical_summary(plan, [report, report], len(plan["cohorts"]))


def test_incomplete_boundary_trial_stays_descriptive_and_missing_prediction_in_denominator():
    plan = logical_plan()
    originals = plan["cohorts"][:-1]
    subset = deepcopy(plan)
    subset["cohorts"] = originals
    e2e, trace = stats_rows(subset)
    e2e[0]["status"] = "prediction_unavailable"
    trace[0]["intervals"][0]["status"] = "prediction_unavailable"
    stats = subject.segment_statistics(plan, {"originals": originals}, e2e, trace)
    assert stats["boundary_trial_indices"] == [19]
    assert "bootstrap" not in str(stats["complete_trial_e2e"])
    assert stats["all_observed_e2e_descriptive"]["short"]["coverage"] == {"planned_trials": 20, "predicted_trials": 19}
    assert len(e2e) == 59


def exposure_fixture():
    execution = {
        "concurrent_serving_run_id": "physical",
        "start_utc": "2026-09-10T14:37:35Z",
        "end_utc": "2026-09-10T14:37:47Z",
    }
    start, end = 1789051055000000000, 1789051068000000000
    combined = {
        "run_id": "physical",
        "cohorts": [
            {"cohort_id": "before", "started": {"unix_ns": start - 20}, "finished": {"unix_ns": start}},
            {"cohort_id": "exposed", "started": {"unix_ns": start - 1}, "finished": {"unix_ns": start + 1}},
            {"cohort_id": "after", "started": {"unix_ns": end}, "finished": {"unix_ns": end + 20}},
        ],
    }
    audit = {
        "active_rows": [
            {"dispatch_id": 2, "dispatch_started_unix_ns": end - 1, "interval_reported_unix_ns": end + 20},
            {"dispatch_id": 3, "dispatch_started_unix_ns": end, "interval_reported_unix_ns": end + 20},
        ]
    }
    annotation = {
        "schema": "dsv41.execution.exposure.v1",
        "physical_run_id": "physical",
        "original_execution_receipt_sha256": "a" * 64,
        "reported_start_utc": execution["start_utc"],
        "reported_end_utc": execution["end_utc"],
        "start_unix_ns": start,
        "end_unix_ns": end,
        "timestamp_resolution_ns": 10**9,
        "affected_cohort_ids": ["exposed"],
        "affected_dispatch_ids": [2],
        "sampling_exclusion": False,
        "interval_semantics": "conservative host envelope, not exact GPU event overlap",
    }
    return annotation, execution, combined, audit


def test_exposure_keeps_half_open_host_envelope_and_never_excludes_samples():
    annotation, execution, combined, audit = exposure_fixture()
    before = deepcopy((combined, audit))
    assert subject.qualify_exposure(annotation, execution, "a" * 64, combined, audit) == annotation
    assert before == (combined, audit)


@pytest.mark.parametrize(
    "mutation", ["excluded", "end", "missing_cohort", "extra_dispatch", "source", "run", "resolution"]
)
def test_exposure_rejects_retroactive_filtering_or_wrong_overlap_inventory(mutation):
    annotation, execution, combined, audit = exposure_fixture()
    if mutation == "excluded":
        annotation["sampling_exclusion"] = True
    elif mutation == "end":
        annotation["end_unix_ns"] -= 10**9
    elif mutation == "missing_cohort":
        annotation["affected_cohort_ids"] = []
    elif mutation == "extra_dispatch":
        annotation["affected_dispatch_ids"].append(3)
    elif mutation == "source":
        annotation["original_execution_receipt_sha256"] = "b" * 64
    elif mutation == "run":
        annotation["physical_run_id"] = "other"
    else:
        annotation["timestamp_resolution_ns"] = 1
    with pytest.raises(ValueError):
        subject.qualify_exposure(annotation, execution, "a" * 64, combined, audit)


def test_failure_keeps_explanation_and_category_without_private_paths():
    message = (
        "PerfDataNotAvailableError: no measured SILICON data at /private/run/table.parquet "
        "under '/Users/name/my private project/calibration'; batch=1, prefix=256, x=128; "
        "overlay=/calibration/root; uri=file:///tmp/private.csv"
    )
    failure = subject.disclosed_failure(message, systems_path="/calibration/root")
    assert failure["failure_category"] == "measured_coverage_missing"
    assert "batch=1, prefix=256, x=128" in failure["failure"]
    assert failure["failure_sha256"] == subject.hashlib.sha256(message.encode()).hexdigest()
    for path in ("/private", "/Users", "/calibration", "/tmp"):
        assert path not in failure["failure"]
    assert "<explicit systems overlay>" in failure["failure"]


@pytest.mark.parametrize(
    "message,category",
    [
        ("InvalidPerfData: shape field missing", "calibration_data_rejected"),
        ("capacity pressure cannot be modeled", "capacity_qualification_failed"),
        ("heterogeneous replay is unsupported", "unsupported_workload_or_policy"),
        ("backend replay raised a conversion error", "replay_or_prediction_failed"),
    ],
)
def test_failure_categories_preserve_source_explanation(message, category):
    result = subject.disclosed_failure(message, systems_path="/calibration")
    assert result["failure_category"] == category
    assert result["failure"] == message


def test_segment_ci_discloses_deadline_conditioning_and_no_power_claim():
    plan = logical_plan()
    report = subject.segment_statistics(
        plan, {"originals": plan["cohorts"][:4]}, *stats_rows({"cohorts": plan["cohorts"][:4]})
    )
    assert "fixed wall-deadline stopping boundary" in report["ci_scope"]
    assert "not a separately prespecified sample powered to 5% precision" in report["ci_scope"]
