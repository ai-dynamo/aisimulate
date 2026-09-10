# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare independently closed physical lifecycles of one frozen main study.

Each segment retains its original run, requests, trial indices, seeds and audit.
A temporary statistical-label view is used only to reuse the existing complete
native/transport/token proof validator; it is never a single-run final report.
Complete same-lifecycle whole trials can have conditional per-segment intervals.
Cross-lifecycle totals are descriptive, even when the frozen plan is completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import compare_e2e as e2e
import compare_trace as trace
from compare_forward import file_hash, resolved_model_identity


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def qualify_exposure(annotation, execution, execution_sha, combined, audit):
    """Audit a conservative host-envelope annotation; never filter observations."""
    require(annotation.get("schema") == "dsv41.execution.exposure.v1", "unknown exposure schema")
    require(
        annotation.get("physical_run_id") == execution["concurrent_serving_run_id"] == combined["run_id"],
        "exposure physical lifecycle differs",
    )
    require(annotation.get("original_execution_receipt_sha256") == execution_sha, "exposure receipt binding differs")
    require(
        annotation.get("timestamp_resolution_ns") == 10**9 and annotation.get("sampling_exclusion") is False,
        "exposure resolution/exclusion policy differs",
    )
    for key in ("start", "end"):
        require(annotation.get("reported_" + key + "_utc") == execution[key + "_utc"], "reported exposure time changed")
    times = [datetime.fromisoformat(execution[k + "_utc"].replace("Z", "+00:00")) for k in ("start", "end")]
    require(
        all(t.utcoffset().total_seconds() == 0 and t.microsecond == 0 for t in times), "exposure must use UTC seconds"
    )
    start, end = int(times[0].timestamp()) * 10**9, (int(times[1].timestamp()) + 1) * 10**9
    require(
        start < end and annotation.get("start_unix_ns") == start and annotation.get("end_unix_ns") == end,
        "exposure did not preserve the conservative second-resolution interval",
    )
    cohorts = sorted(
        c["cohort_id"]
        for c in combined["cohorts"]
        if c["started"]["unix_ns"] < end and c["finished"]["unix_ns"] > start
    )
    dispatches = sorted(
        r["dispatch_id"]
        for r in audit["active_rows"]
        if r["dispatch_started_unix_ns"] < end and r["interval_reported_unix_ns"] > start
    )
    require(
        annotation.get("affected_cohort_ids") == cohorts and annotation.get("affected_dispatch_ids") == dispatches,
        "exposure overlap inventory differs",
    )
    require(
        isinstance(annotation.get("interval_semantics"), str) and bool(annotation["interval_semantics"]),
        "host-envelope timing semantics must be disclosed",
    )
    return deepcopy(annotation)


def logical_contract(plan):
    require(
        plan.get("sampling_role") == "main"
        and plan.get("dataset_role") == "verification"
        and plan.get("corpus_role") == "primary",
        "one frozen primary main plan is required",
    )
    count = plan.get("requested_trials")
    require(type(count) is int and 20 <= count <= 100, "invalid frozen logical budget")
    uuid.UUID(plan["run_id"])
    cases = plan["cohorts"]
    require(cases and len({c["cohort_id"] for c in cases}) == len(cases), "duplicate logical cohort")
    request_ids = [r["request_id"] for c in cases for r in c["requests"]]
    require(len(set(request_ids)) == len(request_ids), "duplicate logical request")
    groups = defaultdict(list)
    for case in cases:
        index = case["trial_index"]
        require(
            type(index) is int
            and 0 <= index < count
            and type(case["trial_seed"]) is int
            and case["trial_seed"] == 93051000 + index
            and case.get("sampling_role") == "main"
            and case.get("corpus_role") == "primary",
            "original trial/seed/stratum identity differs",
        )
        groups[index].append(case)
    require(set(groups) == set(range(count)), "logical trial range is incomplete")
    shape = [(c["purpose"], c["comparison_role"]) for c in groups[0]]
    require(len({p for p, _ in shape}) == len(shape), "duplicate logical scenario")
    require(any(role == "primary" for _, role in shape), "no primary scenarios")
    require(
        all([(c["purpose"], c["comparison_role"]) for c in groups[i]] == shape for i in range(count))
        and cases == [c for i in range(count) for c in groups[i]],
        "logical trials must retain the same complete ordered scenario set",
    )
    for case in cases:
        if case["purpose"] == "prefix-reuse-B":
            seeds = [c for c in groups[case["trial_index"]] if c["cohort_id"] == case.get("seed_cohort_id")]
            require(len(seeds) == 1 and seeds[0]["purpose"] == "prefix-warm-A", "logical prefix seed changed")
    return groups


def expected_physical_cases(logical, offset, run_id):
    cases = deepcopy(logical["cohorts"][offset:])
    require(cases, "cannot resample a completed logical plan")
    restored = []
    if offset and cases[0]["purpose"] == "prefix-reuse-B":
        b = cases[0]
        a = next(c for c in logical["cohorts"] if c["cohort_id"] == b["seed_cohort_id"])
        setup = deepcopy(a)
        setup.update(continuation_setup=True, comparison_role="setup")
        setup["cohort_id"] = str(uuid.uuid5(uuid.UUID(run_id), "restore-prefix/" + a["cohort_id"]))
        for request in setup["requests"]:
            request["request_id"] = str(uuid.uuid5(uuid.UUID(run_id), "restore-prefix/" + request["request_id"]))
        cases.insert(0, setup)
        restored.append(
            {
                "original_seed_cohort_id": a["cohort_id"],
                "actual_setup_cohort_id": setup["cohort_id"],
                "reuse_cohort_id": b["cohort_id"],
            }
        )
    return cases, restored


def qualify_runtime_equivalence(receipt, measurement):
    """Bind a normalizer's reviewed equivalence proof without granting new exceptions."""
    field = "runtime_identity_equivalence"
    bindings = measurement["source_bindings"]
    bound = field + "_sha256"
    present = field in receipt, bound in bindings
    if not any(present):
        return  # Legacy receipts precede the optional equivalence proof.
    require(all(present), "runtime equivalence proof and hash must both be present")
    proof = receipt[field]
    require(type(proof) is dict, "runtime equivalence proof must be an object")
    trace.require_hash(bindings[bound], bound)
    require(bindings[bound] == trace.digest(proof), "runtime equivalence proof hash differs")
    actual = proof.get("actual_runtime_identity_sha256")
    trace.require_hash(actual, "actual runtime identity")
    kind = proof.get("kind")
    if kind == "exact":
        require(set(proof) == {"kind", "actual_runtime_identity_sha256"}, "unknown exact equivalence fields")
        require(actual == receipt["runtime_identity_sha256"], "exact runtime identity differs from comparison")
        return
    require(kind == "reviewed_aggregated_bootstrap_port_only", "unknown runtime equivalence kind")
    require(
        set(proof)
        == {
            "kind",
            "actual_runtime_identity_sha256",
            "comparison_reference_runtime_identity_sha256",
            "control_source_transition",
            "server_arguments_comparison",
            "actual_installed_source_sha256",
            "continuation_file_sha256",
            "review_addendum_sha256",
            "review_addendum",
            "source_evidence_receipt_sha256",
            "frozen_original_plan_sha256",
            "original_closed_audit_sha256",
            "original_progress_sha256",
            "physical_plan_sha256",
            "prior_completed_cohorts",
            "remaining_cohorts",
        },
        "unknown reviewed equivalence fields",
    )
    reference = proof["comparison_reference_runtime_identity_sha256"]
    trace.require_hash(reference, "comparison reference runtime identity")
    require(
        reference == receipt["runtime_identity_sha256"] and actual != reference,
        "reviewed runtime comparison reference differs or conceals the raw identity",
    )
    for key in (
        "continuation_file_sha256",
        "review_addendum_sha256",
        "source_evidence_receipt_sha256",
        "frozen_original_plan_sha256",
        "original_closed_audit_sha256",
        "original_progress_sha256",
        "physical_plan_sha256",
    ):
        trace.require_hash(proof[key], key)
    trace.integer(proof["prior_completed_cohorts"], "prior completed cohorts")
    trace.integer(proof["remaining_cohorts"], "remaining cohorts", minimum=1)
    for key, schema_key, schema in (
        ("control_source_transition", "transition", "dsv41.continuation.aggregated-port-equivalence.v1"),
        ("server_arguments_comparison", "contract", "dsv41.continuation.aggregated-port-equivalence.v1"),
        ("review_addendum", "schema", "dsv41.continuation.aggregated-port-addendum.v1"),
    ):
        require(type(proof[key]) is dict and proof[key].get(schema_key) == schema, "unknown " + key + " schema")
    sources = proof["actual_installed_source_sha256"]
    require(type(sources) is dict and sources, "missing reviewed installed source hashes")
    for name, sha in sources.items():
        require(type(name) is str and name, "invalid reviewed installed source name")
        trace.require_hash(sha, "reviewed installed source")


def qualify_segment(logical, segment, offset):
    """Validate originals before using an internal non-statistical proof view."""
    logical_contract(logical)
    audit, physical, client, measurement, scheduler, receipt = (
        segment[key] for key in ("audit", "plan", "client", "measurement", "scheduler", "receipt")
    )
    run_id = physical["run_id"]
    uuid.UUID(run_id)
    require(
        receipt.get("schema") == "dsv41.closed.segment.admission.v1" and receipt.get("valid") is True,
        "missing closed-segment normalization receipt",
    )
    require(
        receipt.get("physical_run_id")
        == measurement.get("run_id")
        == client.get("run_id")
        == scheduler.get("run_id")
        == run_id,
        "physical lifecycle identities differ",
    )
    require(receipt.get("logical_run_id") == logical["run_id"], "logical study identity differs")
    require(receipt.get("logical_plan_canonical_sha256") == trace.digest(logical), "logical plan binding differs")
    for key, obj in (
        ("audit", audit),
        ("physical_plan", physical),
        ("client_summary", client),
        ("measurement", measurement),
        ("scheduler_receipt", scheduler),
    ):
        require(receipt.get(key + "_canonical_sha256") == trace.digest(obj), "segment receipt binding differs: " + key)
    for key in (
        "normalizer_source_sha256",
        "runtime_identity_sha256",
        "frozen_budget_file_sha256",
        "warmup_dispatch_proof_file_sha256",
        "stop_contract_file_sha256",
        "raw_evidence_inventory_sha256",
    ):
        trace.require_hash(receipt.get(key), key)
    require(
        receipt.get("normalizer_source_sha256") == measurement["source_bindings"]["measurement_builder_sha256"],
        "normalizer source binding differs",
    )
    qualify_runtime_equivalence(receipt, measurement)
    require(
        measurement["source_bindings"].get("audit_canonical_sha256") == trace.digest(audit)
        and measurement["source_bindings"].get("plan_canonical_sha256") == trace.digest(physical),
        "original audit/physical plan hash differs",
    )
    require(
        audit.get("complete_requested_study") is False and audit.get("qualification_scope"),
        "independent physical segment requires explicit closure projection",
    )
    require(
        physical.get("sampling_role") == logical["sampling_role"]
        and physical.get("dataset_role") == logical["dataset_role"]
        and physical.get("corpus_role") == logical["corpus_role"]
        and physical.get("requested_trials") == logical["requested_trials"],
        "physical plan relabeled logical budget",
    )
    if offset:
        require(
            physical.get("logical_study_run_id") == logical["run_id"]
            and physical.get("continuation") is True
            and physical.get("completed_main_cohorts_before_segment") == offset,
            "continuation offset or original study differs",
        )
    else:
        require(physical == logical, "first physical plan is not the original frozen plan")
    expected, restores = expected_physical_cases(logical, offset, run_id)
    require(physical["cohorts"] == expected, "physical request/trial/seed sequence differs from frozen suffix")
    require(receipt.get("restore_seed_mappings") == restores, "restore-A mapping differs from actual physical plan")
    observed = client["cohorts"]
    require(observed and len({c["cohort_id"] for c in observed}) == len(observed), "empty or duplicate observed cohort")
    selected = physical["cohorts"][: len(observed)]
    require(len(selected) == len(observed), "more observations than physical plan")
    for planned, actual in zip(selected, observed, strict=True):
        require(
            all(actual.get(k) == planned.get(k) for k in ("cohort_id", "trial_index", "trial_seed", "purpose")),
            "observed boundary cohort was omitted or original identity changed",
        )
        require(bool(actual.get("continuation_setup")) == bool(planned.get("continuation_setup")), "setup role differs")
        e2e.checked_requests(planned, actual)
    # Retain and validate every original identity first. The existing proof
    # checker assumes a full statistical grid. Unique one-case labels here are
    # solely an adapter for its native interval/token checks, not a new study.
    view = deepcopy(physical)
    view.update(requested_trials=1, cohorts=deepcopy(selected))
    for case in view["cohorts"]:
        case.update(trial_index=0, purpose=case["cohort_id"], comparison_role="primary")
    proof_measurement = deepcopy(measurement)
    proof_measurement["source_bindings"]["plan_canonical_sha256"] = trace.digest(view)
    covered = trace.qualify_inputs(audit, view, proof_measurement, diagnostic=True)
    require(set(covered) == {c["cohort_id"] for c in selected}, "native proof omits a physical main/setup cohort")
    # Audit may also contain canary, warmup and pilot. Every actual main cohort
    # must appear in the independently bound client summary, including the tail.
    planned_ids = {c["cohort_id"] for c in physical["cohorts"]}
    audited_ids = {c["cohort_id"] for c in audit["cohorts"]}
    require(audited_ids & planned_ids == set(covered), "observed native boundary cohort missing from client projection")
    by_id = {c["cohort_id"]: c for c in selected}
    clients = {c["cohort_id"]: c for c in observed}
    for case in selected:
        seed = case.get("seed_cohort_id")
        mapping = next((m for m in restores if m["reuse_cohort_id"] == case["cohort_id"]), None)
        if mapping:
            seed = mapping["actual_setup_cohort_id"]
        if case["purpose"] == "prefix-reuse-B":
            require(seed in by_id, "same-lifecycle prefix seed was not observed")
            index = selected.index(case)
            require(index > 0 and selected[index - 1]["cohort_id"] == seed, "prefix A/B continuity changed")
            replay_case = dict(case, seed_cohort_id=seed)
            # Empty engine is enough to verify real token/hash/ack/order inputs;
            # replay construction does not execute or inspect timing here.
            e2e.replay_spec(
                {}, replay_case, clients[case["cohort_id"]], seed_cohort=by_id[seed], seed_observed=clients[seed]
            )
        else:
            require(
                e2e.cold_cache_ack(clients[case["cohort_id"]]["cache_control"]), "cold cohort lacks native clear ack"
            )
    originals = [c for c in selected if not c.get("continuation_setup")]
    require(originals, "segment contains only setup work")
    require(originals == logical["cohorts"][offset : offset + len(originals)], "logical coverage has a gap")
    return {
        "selected": selected,
        "originals": originals,
        "covered": covered,
        "restores": restores,
        "next_offset": offset + len(originals),
    }


def complete_trial_indices(logical, originals):
    groups = logical_contract(logical)
    present = {c["cohort_id"] for c in originals}
    return [i for i, cases in groups.items() if {c["cohort_id"] for c in cases} <= present]


def segment_statistics(logical, proof, e2e_rows, trace_rows):
    complete = set(complete_trial_indices(logical, proof["originals"]))
    return {
        "complete_same_lifecycle_trial_indices": sorted(complete),
        "boundary_trial_indices": sorted({c["trial_index"] for c in proof["originals"]} - complete),
        "ci_scope": (
            "approximate whole-trial bootstrap conditional on this observed physical lifecycle and its fixed "
            "wall-deadline stopping boundary; only the resulting complete-trial subset is eligible, not a "
            "separately prespecified sample powered to 5% precision; missing predictions suppress CI"
        ),
        "all_observed_e2e_descriptive": e2e.paired_summary(e2e_rows, final=False),
        "all_observed_trace_descriptive": trace.independent_trial_summary(trace_rows, final=False),
        "complete_trial_e2e": e2e.paired_summary([r for r in e2e_rows if r["trial_index"] in complete], final=True),
        "complete_trial_trace": trace.independent_trial_summary(
            [r for r in trace_rows if r["trial_index"] in complete], final=True
        ),
    }


def disclosed_failure(message, *, systems_path):
    """Keep an explainable error and original digest without private paths."""
    lower = message.lower()
    if "perfdatanotavailable" in lower or "no measured silicon data" in lower:
        category = "measured_coverage_missing"
    elif "invalidperfdata" in lower:
        category = "calibration_data_rejected"
    elif "capacity" in lower:
        category = "capacity_qualification_failed"
    elif "unsupported" in lower or "not supported" in lower:
        category = "unsupported_workload_or_policy"
    else:
        category = "replay_or_prediction_failed"
    sanitized = message.replace(str(systems_path), "<explicit systems overlay>") if systems_path else message
    sanitized = re.sub(r"""(["'])/[^\n]*?\1""", "'<private path>'", sanitized)
    sanitized = re.sub(r"file://\S+", "<private path>", sanitized)
    sanitized = re.sub(r"(?<![\w/])/(?:[^\s\[\](){}<>,;:'\"]+)", "<private path>", sanitized)
    return {
        "failure": sanitized,
        "failure_category": category,
        "failure_sha256": hashlib.sha256(message.encode()).hexdigest(),
    }


def compare_segment(logical, segment, proof, config, predictor, replay):
    audit, physical, client, measurement, scheduler = (
        segment[k] for k in ("audit", "plan", "client", "measurement", "scheduler")
    )
    trace.qualify_prediction_config(config, measurement)
    require(measurement["backend"] == "sglang", "this segment adapter currently qualifies GB300 SGLang only")
    engine = e2e.sglang_engine(scheduler, config)
    compared_plan = dict(physical, cohorts=proof["selected"])
    traces = trace.compare_cohorts(
        audit,
        compared_plan,
        proof["covered"],
        predictor,
        backend="sglang",
        forward_model=config["forward_model"],
        decoder_replay=config["decoder_replay"],
    )
    clients = {c["cohort_id"]: c for c in client["cohorts"]}
    planned = {c["cohort_id"]: c for c in proof["selected"]}
    output = []
    for cohort in proof["originals"]:
        if cohort["comparison_role"] != "primary":
            continue
        seed = cohort.get("seed_cohort_id")
        mapping = next((m for m in proof["restores"] if m["reuse_cohort_id"] == cohort["cohort_id"]), None)
        case = cohort
        if mapping:
            seed = mapping["actual_setup_cohort_id"]
            case = dict(cohort, seed_cohort_id=seed)
        result = e2e.compare_cohort(
            engine,
            case,
            clients[cohort["cohort_id"]],
            replay,
            seed_cohort=planned.get(seed),
            seed_observed=clients.get(seed),
        )
        result["physical_run_id"] = physical["run_id"]
        if mapping:
            result["actual_restore_seed_mapping"] = mapping
        if "failure" in result:
            result.update(disclosed_failure(result["failure"], systems_path=config["systems_path"]))
        e2e.attach_cache_comparison(result, proof["covered"][cohort["cohort_id"]])
        output.append(result)
    for row in traces:
        row["physical_run_id"] = physical["run_id"]
    return {
        "physical_run_id": physical["run_id"],
        "closed_native_intervals": len(audit["active_rows"]),
        "observed_original_cohorts": len(proof["originals"]),
        "observed_setup_cohorts": len(proof["selected"]) - len(proof["originals"]),
        "receipt": segment["receipt"],
        "e2e_cases": output,
        "trace_cases": traces,
        "statistics": segment_statistics(logical, proof, output, traces),
    }


def logical_summary(logical, reports, offset):
    rows = [r for report in reports for r in report["e2e_cases"]]
    traces = [r for report in reports for r in report["trace_cases"]]
    planned = sum(c["comparison_role"] == "primary" for c in logical["cohorts"])
    require(len({r["cohort_id"] for r in rows}) == len(rows), "duplicate cohort across physical lifecycles")
    require(len({r["cohort_id"] for r in traces}) == len(traces), "duplicate trace cohort across physical lifecycles")
    return {
        "fixed_plan_completed": offset == len(logical["cohorts"]),
        "coverage_complete": offset == len(logical["cohorts"]),
        "statistical_precision_completed": False,
        "statistical_precision_note": (
            "no cross-lifecycle population CI; original pilot precision target remains separately recorded"
        ),
        "planned_logical_trials": logical["requested_trials"],
        "planned_primary_cohorts": planned,
        "observed_primary_cohorts": len(rows),
        "unobserved_primary_cohorts": planned - len(rows),
        "predicted_e2e_cohorts": sum(r["status"] == "predicted" for r in rows),
        "missing_e2e_predictions": sum(r["status"] != "predicted" for r in rows),
        "pooled_confidence_intervals": None,
        "e2e_descriptive": e2e.paired_summary(rows, final=False),
        "trace_descriptive": trace.independent_trial_summary(traces, final=False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("logical-plan", "segments-file", "prediction-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "logical_plan": args.logical_plan,
        "segments_index": args.segments_file,
        "prediction_config": args.prediction_config,
    }
    logical, index, config = [json.loads(paths[k].read_bytes()) for k in paths]
    logical_contract(logical)
    require(index.get("schema") == "dsv41.closed.segments.index.v1" and index.get("segments"), "invalid segments index")
    segments = []
    for number, item in enumerate(index["segments"]):
        segment = {}
        for key, filename in {
            "audit": "audit.json",
            "plan": "plan.json",
            "client": "client-summary.json",
            "measurement": "measurement.json",
            "scheduler": "scheduler-receipt.json",
            "receipt": "segment-receipt.json",
        }.items():
            path = (args.segments_file.parent / item / filename).resolve(strict=True)
            paths[f"segment_{number}_{key}"] = path
            segment[key] = json.loads(path.read_bytes())
        segment_paths = {key: paths[f"segment_{number}_{key}"] for key in segment}
        e2e.qualify_e2e_sources(segment["measurement"], segment_paths["client"], segment_paths["scheduler"])
        require(segment["client"]["plan_sha256"] == file_hash(segment_paths["plan"]), "actual client plan bytes differ")
        require(
            segment["receipt"]["logical_plan_file_sha256"] == file_hash(args.logical_plan),
            "original logical file differs",
        )
        segments.append(segment)
    hashes = {key: file_hash(path) for key, path in paths.items()}
    proofs, offset, run_ids, runtime, budget = [], 0, set(), None, None
    for segment in segments:
        proof = qualify_segment(logical, segment, offset)
        run_id = segment["plan"]["run_id"]
        require(run_id not in run_ids, "duplicate physical lifecycle")
        run_ids.add(run_id)
        current = segment["receipt"]["runtime_identity_sha256"]
        current_budget = segment["receipt"]["frozen_budget_file_sha256"]
        require(runtime is None or runtime == current, "runtime source/config changed across lifecycles")
        require(budget is None or budget == current_budget, "frozen budget changed across lifecycles")
        runtime, budget, offset = current, current_budget, proof["next_offset"]
        proofs.append(proof)
    import aisimulate._runtime as native
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    sources = e2e.comparison_sources() | {"segment_comparison": file_hash(__file__)}
    system = trace.systems_identity(config)
    identity = resolved_model_identity(
        config, {"input_provenance": {"config_sha256": segments[0]["measurement"]["model_config_canonical_sha256"]}}
    )
    predictor = RustForwardPassPerfModel.from_native(config)
    reports = [
        compare_segment(
            logical, segment, proof, config, predictor.estimate_forward_pass_time_ms, native.run_replay_json
        )
        for segment, proof in zip(segments, proofs, strict=True)
    ]
    require(
        hashes == {key: file_hash(path) for key, path in paths.items()}, "segment evidence changed during comparison"
    )
    require(system == trace.systems_identity(config), "calibration changed during comparison")
    require(
        sources == e2e.comparison_sources() | {"segment_comparison": file_hash(__file__)}, "prediction source changed"
    )
    result = {
        "schema": "dsv41.closed.segments.comparison.v1",
        "logical_run_id": logical["run_id"],
        "qualification": "independently_closed_physical_segments_not_single_run_final",
        "input_files_sha256": hashes,
        "prediction_sources": sources,
        "systems_identity": system,
        "resolved_model_identity": identity,
        "prediction_config": {k: v for k, v in config.items() if k != "systems_path"},
        "summary": logical_summary(logical, reports, offset),
        "segments": reports,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
