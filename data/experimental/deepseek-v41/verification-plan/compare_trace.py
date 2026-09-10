# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare closed native FPM intervals; incomplete segments are diagnostic only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import analyze_e2e
import compare_forward
import normalize_fpm
from compare_forward import canonical, error_summary, file_hash, resolved_model_identity
from normalize_fpm import prediction_input

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
CHECKPOINT_SHA = "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d"
RUNTIMES = {
    "sglang": ("gb300", "0.0.0.dev0", "sglang_inclusive_query"),
    "vllm": ("gb200", "0.1.dev20904+g179dd0fa9", "vllm_past_kv"),
}
PRODUCERS = {
    "sglang": (
        "sglang.srt.observability.forward_pass_metrics",
        "197783e4f501a3651a0bfe376eda29f35cd4a7bf18b6e7b9eebf094ff285f5da",
    ),
    "vllm": (
        "dynamo.vllm.instrumented_scheduler",
        "765586f5891908f074c7dccae3a35edf4fb2629688f2bfdae11ab0de732ae298",
    ),
}


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"missing or invalid source hash: {label}")


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid integer: {label}")
    return value


def qualify_measurement(measurement):
    backend = measurement.get("backend")
    if backend not in RUNTIMES:
        raise ValueError("unqualified measurement backend")
    system, version, semantics = RUNTIMES[backend]
    required = {
        "model_name": MODEL,
        "system_name": system,
        "backend_version": version,
        "execution_mode": "eager",
        "tp_size": 4,
        "moe_ep_size": 1,
        "producer_semantics": semantics,
        "model_config_canonical_sha256": CHECKPOINT_SHA,
    }
    defaults = {
        "pp_size": 1,
        "attention_dp_size": 1,
        "cp_size": 1,
        "moe_tp_size": 4,
        "nextn": 0,
        "engram_residency": "hbm_tp_sharded",
        "input_modality": "text",
    }
    for key, expected in (required | defaults).items():
        actual = measurement.get(key, defaults.get(key))
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"unqualified measurement field: {key}")
    replay = measurement.get("decoder_replay")
    if type(replay) is not bool or (backend == "vllm" and replay):
        raise ValueError("unqualified measured decoder replay policy")
    if (
        not isinstance(measurement.get("runtime_digest"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", measurement["runtime_digest"]) is None
    ):
        raise ValueError("immutable runtime image digest required")
    if measurement.get("producer_source_sha256") != PRODUCERS[backend][1]:
        raise ValueError("native FPM source has not been qualified for these KV semantics")
    require_hash(measurement.get("observer_source_sha256"), "observer")


def qualify_inputs(audit, plan, measurement, *, diagnostic=False):
    """Return covered frozen main cohorts, never promote a partial lifecycle.

    source_bindings hashes the canonical audit/plan payloads and retains hashes
    of the actual execution, worker config, measurement builder and audit tool.
    Callers retain those original private receipts; this API emits no raw paths.
    """
    qualify_measurement(measurement)
    if plan.get("sampling_role") != "main" or plan.get("dataset_role") != "verification":
        raise ValueError("comparison requires an independent main verification plan")
    if not plan.get("run_id") or measurement.get("run_id") != plan["run_id"]:
        raise ValueError("measurement and plan run identities differ")
    bindings = measurement.get("source_bindings", {})
    for key in (
        "audit_canonical_sha256",
        "plan_canonical_sha256",
        "execution_file_sha256",
        "worker_config_file_sha256",
        "measurement_builder_sha256",
        "audit_tool_sha256",
    ):
        require_hash(bindings.get(key), key)
    for label, value in bindings.items():
        if not isinstance(label, str) or re.fullmatch(r"[a-z][a-z0-9_]*", label) is None:
            raise ValueError("source binding labels must not contain private paths")
        require_hash(value, label)
    if bindings["audit_canonical_sha256"] != digest(audit) or bindings["plan_canonical_sha256"] != digest(plan):
        raise ValueError("measurement source bindings differ from the actual audit or frozen plan")
    if audit.get("valid") is not True or audit.get("errors") != []:
        raise ValueError("closed native and transport qualification must pass first")
    projected = audit.get("complete_requested_study") is False
    if diagnostic:
        if not projected or not audit.get("qualification_scope"):
            raise ValueError("diagnostic mode requires an explicit closed-segment projection")
        require_hash(audit.get("original_combined_summary_sha256"), "closed segment original summary")
    elif projected:
        raise ValueError("incomplete closed segment cannot qualify a final main report")

    producers = audit.get("producer_audits", [])
    if len(producers) != 1:
        raise ValueError("qualified DP1 requires one observed native producer")
    producer = producers[0]
    module, source = PRODUCERS[measurement["backend"]]
    if (
        producer.get("run_id") != plan["run_id"]
        or producer.get("module") != module
        or producer.get("source_sha256") != source
        or producer.get("observer_sha256") != measurement["observer_source_sha256"]
        or producer.get("dp_rank") != 0
        or producer.get("errors") != []
        or producer.get("thread_alive_at_shutdown") is not False
        or any(
            producer.get(k) is not True
            for k in ("complete", "run_exited", "shutdown_called", "dispatch_audit_required")
        )
    ):
        raise ValueError("native producer closure or source identity differs")
    dispatch = producer.get("dispatch_audit", {})
    names = ("attempted", "enqueued", "dequeued", "active_sent", "sent", "sequence_allocated", "send_attempted")
    for name in names:
        integer(producer.get(name), name)
    if (
        any(producer.get(k) != 0 for k in ("queue_full", "send_again", "send_error", "publish_suppressed"))
        or not producer["attempted"] == producer["enqueued"] == producer["dequeued"] == producer["active_sent"]
        or not producer["sent"] == producer["sequence_allocated"] == producer["send_attempted"]
        or dispatch.get("started") != producer["active_sent"]
        or dispatch.get("completed") != producer["active_sent"]
        or dispatch.get("pending") != []
        or len(dispatch.get("records", [])) != producer["active_sent"]
    ):
        raise ValueError("native producer intervals or queue did not reconcile")
    active = audit.get("active_rows", [])
    if len(active) != producer["active_sent"]:
        raise ValueError("native audit omits active intervals")
    by_location = {(r["file"], r["line"]): r for r in active}
    locations = set(by_location)
    if len(locations) != len(active) or len({r["dispatch_id"] for r in active}) != len(active):
        raise ValueError("duplicate native interval identity")
    records = {r["dispatch_id"]: r for r in dispatch["records"]}
    if len(records) != len(dispatch["records"]) or set(records) != {r["dispatch_id"] for r in active}:
        raise ValueError("native producer dispatch records are missing or duplicated")
    for row in active:
        original = records[row["dispatch_id"]]["requests"]
        enriched = {r["rid"]: r for r in row["dispatch_requests"]}
        if (
            len(enriched) != len(original)
            or len({r["rid"] for r in original}) != len(original)
            or any(
                r["rid"] not in enriched or any(enriched[r["rid"]].get(k) != v for k, v in r.items()) for r in original
            )
        ):
            raise ValueError("native dispatch geometry differs from the closed producer snapshot")
    if any(r["fpm"].get("worker_id") != producer["worker_id"] or r["fpm"].get("dp_rank") != 0 for r in active):
        raise ValueError("native interval belongs to another worker")
    trace_files = audit.get("files", [])
    if not trace_files or len({f["file"] for f in trace_files}) != len(trace_files):
        raise ValueError("missing or duplicate trace source file receipt")
    for item in trace_files:
        require_hash(item.get("sha256"), "trace file")
    if not {r["file"] for r in active} <= {f["file"] for f in trace_files}:
        raise ValueError("native interval is not bound to an audited trace file")

    planned = plan.get("cohorts", [])
    count = integer(plan.get("requested_trials"), "requested_trials", 1)
    if not planned or len({c["cohort_id"] for c in planned}) != len(planned):
        raise ValueError("empty or duplicate frozen main cohorts")
    request_ids = [r["request_id"] for c in planned for r in c["requests"]]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("duplicate frozen main request identity")
    for case in planned:
        if integer(case["trial_index"], "trial_index") >= count:
            raise ValueError("main trial lies outside its frozen budget")
        integer(case["trial_seed"], "trial_seed")
    if {c["trial_index"] for c in planned} != set(range(count)):
        raise ValueError("frozen main plan does not contain every requested trial")
    scenarios = defaultdict(list)
    for case in planned:
        if case.get("corpus_role", "unspecified") != plan.get("corpus_role", "unspecified"):
            raise ValueError("one frozen corpus stratum is required per comparison")
        if case.get("comparison_role") == "primary":
            scenarios[case["purpose"]].append(case)
    if not scenarios or any(
        len(cases) != count
        or {c["trial_index"] for c in cases} != set(range(count))
        or len({c["trial_seed"] for c in cases}) != count
        for cases in scenarios.values()
    ):
        raise ValueError("each main scenario requires its full independent trial budget")
    cohorts = {c["cohort_id"]: c for c in audit.get("cohorts", [])}
    if len(cohorts) != len(audit.get("cohorts", [])):
        raise ValueError("duplicate audited cohort identity")
    covered = [c for c in planned if c["cohort_id"] in cohorts]
    if not covered or (not diagnostic and len(covered) != len(planned)):
        raise ValueError("closed audit does not cover every planned main cohort")
    if diagnostic and covered != planned[: len(covered)]:
        raise ValueError("diagnostic main coverage is not a continuous frozen prefix")
    assigned = set()
    for case in covered:
        cohort = cohorts[case["cohort_id"]]
        proof = cohort.get("native_request_proof", {})
        requests = proof.get("requests", [])
        if (
            len(requests) != len(case["requests"])
            or len({r["request_id"] for r in requests}) != len(requests)
            or {r["request_id"] for r in requests} != {r["request_id"] for r in case["requests"]}
        ):
            raise ValueError("audited native request identities differ from main plan")
        by_id = {r["request_id"]: r for r in requests}
        native_ids = [r.get("native_rid") for r in requests]
        if any(not isinstance(rid, str) or not rid for rid in native_ids) or len(set(native_ids)) != len(native_ids):
            raise ValueError("missing or duplicate native request mapping")
        cohort_dispatches = [
            request
            for location in cohort.get("rows", [])
            for request in by_location.get((location["file"], location["line"]), {}).get("dispatch_requests", [])
        ]
        for request in case["requests"]:
            actual = by_id[request["request_id"]]
            tokens = request["input_token_ids"]
            if not tokens or any(type(token) is not int or not 0 <= token < 2**32 for token in tokens):
                raise ValueError("planned token IDs must be explicit uint32 values")
            token_hash = digest(tokens)
            if request.get("input_token_ids_sha256") != token_hash:
                raise ValueError("planned token hash differs from actual frozen token IDs")
            dispatches = [r for r in cohort_dispatches if r["rid"] == actual["native_rid"]]
            if not dispatches or any(
                r.get("input_token_ids_sha256") != token_hash
                or r.get("prompt_tokens") != len(tokens)
                or r.get("max_new_tokens") != request["output_tokens"]
                for r in dispatches
            ):
                raise ValueError("planned token content or output limit differs from witnessed native request")
            if (
                actual.get("prefill_new_tokens", -1) + actual.get("initial_cross_request_cached_tokens", -1)
                != len(request["input_token_ids"])
                or actual.get("required_decode_iterations") != request["output_tokens"] - 1
            ):
                raise ValueError("audited native request geometry differs from the planned input/output")
        if not cohort.get("rows"):
            raise ValueError("main cohort has no native intervals")
        row_keys = {(r["file"], r["line"]) for r in cohort["rows"]}
        if len(row_keys) != len(cohort["rows"]) or not row_keys <= locations or row_keys & assigned:
            raise ValueError("main native intervals are missing, repeated or shared between cohorts")
        assigned.update(row_keys)
    return {c["cohort_id"]: cohorts[c["cohort_id"]] for c in covered}


def qualify_prediction_config(config, measurement):
    qualify_measurement(measurement)
    for field in ("model_name", "system_name", "backend", "backend_version", "tp_size", "moe_ep_size"):
        if config.get(field) != measurement[field]:
            raise ValueError(f"prediction/measurement mismatch: {field}")
    # Apply the same pure-TP, inferred-precision and strict-overlay contract to
    # both explicitly qualified backends; the backend identity was checked above.
    common = dict(config, system_name="gb300", backend="sglang", backend_version="0.0.0.dev0")
    if measurement["backend"] == "vllm" and config.get("forward_model") == "fpm":
        # The qualified vLLM runtime uses FP8 KV, whose Collector FMHA identity
        # differs from the SDK's default BF16 label. Require independent native
        # evidence, not just a matching table label, before selecting that cell.
        if config.get("activation_dtype") != "fp8" or measurement.get("fmha_quant_mode") != "fp8":
            raise ValueError("GB200 FPM requires the independently qualified FP8 FMHA identity")
        require_hash(measurement.get("source_bindings", {}).get("fmha_identity_receipt_sha256"), "native FMHA receipt")
        common["activation_dtype"] = None
    compare_forward.validate_prediction_contract(
        common, {"execution_profile": "decoder_bounded" if measurement["decoder_replay"] else "full"}
    )


def compare_interval(row, predictor, *, backend, forward_model, decoder_replay):
    if backend not in RUNTIMES or forward_model not in {"op_level", "fpm"}:
        raise ValueError("unknown native producer or forward model")
    measured = row["fpm"]["wall_time"] * 1000
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError("heartbeats/nonfinite values cannot become latency observations")
    requests = row.get("dispatch_requests", [])
    if not requests or len({r["rid"] for r in requests}) != len(requests):
        raise ValueError("missing or duplicate per-request native dispatch evidence")
    prefills = [r for r in requests if r["phase"] == "prefill"]
    decodes = [r for r in requests if r["phase"] == "decode"]
    if len(prefills) + len(decodes) != len(requests):
        raise ValueError("unknown native request phase")
    for request in prefills:
        integer(request["query_tokens"], "native query", 1)
        integer(request["prefix_tokens"], "native prefix")
    scheduled = row["fpm"]["scheduled_requests"]
    expected = {
        "num_prefill_requests": len(prefills),
        "num_decode_requests": len(decodes),
        "sum_prefill_tokens": sum(r["query_tokens"] for r in prefills),
        "sum_prefill_kv_tokens": sum(r["prefix_tokens"] for r in prefills),
    }
    for key, value in expected.items():
        if integer(scheduled.get(key), key) != value:
            raise ValueError("native per-request dispatch and FPM aggregate geometry differ")
    context_key = "inclusive_context_tokens" if backend == "sglang" else "past_kv_tokens"
    contexts = [integer(r.get(context_key), context_key, 1 if backend == "sglang" else 0) for r in decodes]
    if integer(scheduled.get("sum_decode_kv_tokens"), "decode KV sum") != sum(contexts):
        raise ValueError("native per-request decode contexts differ from the FPM KV axis")
    native_prefill_lengths = [
        integer(r.get("prompt_tokens"), "original native prompt", 1) if backend == "vllm" else r["query_tokens"]
        for r in prefills
    ]
    for field, lengths in (
        ("var_prefill_length", native_prefill_lengths),
        ("var_decode_kv_tokens", contexts),
    ):
        variance = scheduled.get(field)
        if (
            type(variance) not in (int, float)
            or not math.isfinite(variance)
            or not math.isclose(variance, statistics.pvariance(lengths) if lengths else 0.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError("native per-request lengths and FPM variance differ")
    metrics, bridge = prediction_input(
        row["fpm"],
        producer_semantics=RUNTIMES[backend][2],
        target_axis="whole_forward_past_kv" if forward_model == "fpm" else "op_level_inclusive_query",
    )
    metrics.pop("wall_time", None)
    query_variance = statistics.pvariance([r["query_tokens"] for r in prefills]) if prefills else 0.0
    if forward_model == "op_level":
        metrics["scheduled_requests"]["var_prefill_length"] = query_variance
    variance_bridge = {
        "native_prefill_axis": "original_prompt_tokens" if backend == "vllm" else "current_query_tokens",
        "native_prefill_variance": scheduled["var_prefill_length"],
        "current_query_variance": query_variance,
        "prediction_prefill_axis": "current_query_tokens"
        if forward_model == "op_level"
        else ("original_prompt_tokens" if backend == "vllm" else "current_query_tokens"),
        "prediction_prefill_variance": metrics["scheduled_requests"]["var_prefill_length"],
    }
    result = {
        "counter_id": row["fpm"]["counter_id"],
        "dispatch_id": row["dispatch_id"],
        "phase": "mixed" if prefills and decodes else "prefill" if prefills else "decode",
        "observed_ms": measured,
        "axis_bridge": bridge,
        "variance_bridge": variance_bridge,
        "native_scheduled_requests": dict(scheduled),
        "scheduled_requests": metrics["scheduled_requests"],
    }
    try:
        if decoder_replay and len({(r["query_tokens"], r["prefix_tokens"]) for r in prefills}) > 1:
            raise ValueError("bounded heterogeneous query/prefix cannot be represented by aggregate-only FPM input")
        predicted = predictor(metrics)
        if type(predicted) not in (int, float) or not math.isfinite(predicted) or predicted <= 0:
            raise ValueError("native estimator returned no finite positive prediction")
    except Exception as error:
        # Retain the missing result and reason, without exporting private roots
        # that native database exceptions may include.
        message = re.sub(r"(?<![\w:])/(?:[^\s'\"()]+)", "<path>", str(error))
        result.update(status="prediction_unavailable", failure_type=type(error).__name__, failure=message)
    else:
        result.update(status="predicted", predicted_ms=predicted, signed_error_percent=100 * (predicted / measured - 1))
    return result


def trace_summary(rows):
    result = error_summary(rows)
    if result.get("predicted_points"):
        result["weighting"] = "signed/APE: equal correlated native intervals; WAPE: observed-latency weighted"
        result["p90_absolute_error_percent_across_intervals"] = result.pop(
            "p90_absolute_error_percent_across_configurations"
        )
    return result


def independent_trial_summary(cohorts, *, final, seed=94051000, resamples=5000):
    """Resample complete trials, preserving every correlated native interval."""
    groups = defaultdict(list)
    seen = set()
    for cohort in cohorts:
        identity = (cohort["purpose"], cohort["trial_index"])
        if identity in seen:
            raise ValueError("duplicate independent trace trial within scenario")
        seen.add(identity)
        groups[cohort["purpose"]].append(cohort)
    output = {}
    for purpose, group in sorted(groups.items()):
        if len({c["trial_seed"] for c in group}) != len(group):
            raise ValueError("duplicate independent trace seed within scenario")
        trials = []
        for cohort in group:
            rows = cohort["intervals"]
            if not rows or any(r["status"] != "predicted" for r in rows):
                continue
            observed = sum(r["observed_ms"] for r in rows)
            predicted = sum(r["predicted_ms"] for r in rows)
            absolute = sum(abs(r["predicted_ms"] - r["observed_ms"]) for r in rows)
            trials.append((observed, absolute, 100 * (predicted / observed - 1)))
        result = {
            "observed_trials": len(group),
            "fully_predicted_trials": len(trials),
            "missing_prediction_trial_indices": [
                c["trial_index"]
                for c in group
                if not c["intervals"] or any(r["status"] != "predicted" for r in c["intervals"])
            ],
            "weighting": "equal trials for total-forward bias; observed-latency weighted interval WAPE",
        }
        if trials:

            def metrics(sample):
                return {
                    "mean_trial_total_forward_signed_error_percent": statistics.mean(t[2] for t in sample),
                    "interval_wape_percent": 100 * sum(t[1] for t in sample) / sum(t[0] for t in sample),
                }

            result.update(metrics(trials))
            if final and len(trials) >= 20 and len(trials) == len(group):
                rng = random.Random(seed)
                samples = [metrics(rng.choices(trials, k=len(trials))) for _ in range(resamples)]
                result["whole_trial_bootstrap_ci95"] = {
                    key: [analyze_e2e.quantile([s[key] for s in samples], q) for q in (0.025, 0.975)]
                    for key in samples[0]
                }
                result["bootstrap_resamples"] = resamples
                result["bootstrap_seed"] = seed
        output[purpose] = result
    return output


def compare_cohorts(audit, plan, cohorts, predictor, *, backend, forward_model, decoder_replay):
    locations = {(r["file"], r["line"]): r for r in audit["active_rows"]}
    results = []
    for case in plan["cohorts"]:
        if case["cohort_id"] not in cohorts or case.get("comparison_role") != "primary":
            continue
        cohort = cohorts[case["cohort_id"]]
        rows = []
        for location in cohort["rows"]:
            row = locations[location["file"], location["line"]]
            compared = compare_interval(
                row, predictor, backend=backend, forward_model=forward_model, decoder_replay=decoder_replay
            )
            role = cohort["native_request_proof"]["iteration_roles"][str(row["dispatch_id"])]
            if role not in {"useful_request_work", "unreturned_overlap_output", "mixed_useful_and_unreturned"}:
                raise ValueError("unknown native interval attribution role")
            compared["native_work_role"] = role
            rows.append(compared)
        results.append(
            {
                "cohort_id": case["cohort_id"],
                "purpose": case["purpose"],
                "trial_index": case["trial_index"],
                "trial_seed": case["trial_seed"],
                "declared_coverage_role": case.get("coverage_role", "unspecified"),
                "corpus_role": case.get("corpus_role", "unspecified"),
                "summary": trace_summary(rows),
                "intervals": rows,
            }
        )
    if not results:
        raise ValueError("no covered primary main cohort to compare")
    return results


def systems_identity(config):
    """Complete explicitly selected system overlay inventory; no inherited roots."""
    import yaml

    root = Path(config["systems_path"]).resolve(strict=True)
    name = config["system_name"] + ".yaml"
    spec = yaml.safe_load((root / name).read_text())
    relative = Path(spec["data_dir"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("system data escapes the explicitly selected overlay")
    data = (root / relative).resolve(strict=True)
    if not data.is_relative_to(root) or not data.is_dir():
        raise ValueError("system data escapes the explicitly selected overlay")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinked system inputs are not admitted")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_hash(path)
    return {
        "inventory_scope": "complete selected overlay including unused files",
        "files_sha256": files,
        "system_spec_sha256": files[name],
        "inventory_sha256": digest(files),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("audit", "plan", "measurement", "prediction-config", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument(
        "--diagnostic", action="store_true", help="explicit closed-segment diagnostic; never a final main report"
    )
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("audit", "plan", "measurement", "prediction_config")}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    audit, plan, measurement, config = [json.loads(p.read_bytes()) for p in paths.values()]
    cohorts = qualify_inputs(audit, plan, measurement, diagnostic=args.diagnostic)
    qualify_prediction_config(config, measurement)
    import aisimulate._runtime as native
    from aiconfigurator_core.sdk import engine, rust_engine_step
    from aiconfigurator_core.sdk.models import deepseek_v41
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    observed_config = {"input_provenance": {"config_sha256": measurement["model_config_canonical_sha256"]}}
    resolved = resolved_model_identity(config, observed_config)
    system = systems_identity(config)
    model = RustForwardPassPerfModel.from_native(config)
    results = compare_cohorts(
        audit,
        plan,
        cohorts,
        model.estimate_forward_pass_time_ms,
        backend=measurement["backend"],
        forward_model=config.get("forward_model", "op_level"),
        decoder_replay=measurement["decoder_replay"],
    )
    if hashes != {name: file_hash(path) for name, path in paths.items()}:
        raise ValueError("audited source inputs changed during comparison")
    if system != systems_identity(config) or resolved != resolved_model_identity(config, observed_config):
        raise ValueError("prediction sources changed during comparison")
    public_config = dict(config)
    if Path(public_config["systems_path"]).is_absolute():
        public_config["systems_path"] = "<explicit systems overlay; see hashed inventory>"
    rows = [row for result in results for row in result["intervals"]]
    report = {
        "schema": "dsv41.trace.comparison.v1",
        "run_id": plan["run_id"],
        "qualification": "closed_segment_diagnostic" if args.diagnostic else "qualified_closed_main",
        "complete_requested_study": not args.diagnostic,
        "requested_trials": plan["requested_trials"],
        "planned_main_cohorts": len(plan["cohorts"]),
        "covered_main_cohorts": len(cohorts),
        "missing_main_cohort_ids": [c["cohort_id"] for c in plan["cohorts"] if c["cohort_id"] not in cohorts],
        "prediction_config": public_config,
        "correction_fitting": False,
        "observed_target": "sglang_existing_gpu_event_interval"
        if measurement["backend"] == "sglang"
        else "vllm_cpu_schedule_output_or_adjacent_output_interval",
        "scope": "covered main primary cohorts; all attributed work including unreturned overlap output",
        "uncertainty": "whole-trial bootstrap within each scenario; intervals are never independent replicates",
        "measurement_identity": {
            key: measurement[key]
            for key in (
                "backend",
                "backend_version",
                "system_name",
                "model_name",
                "decoder_replay",
                "execution_mode",
                "runtime_digest",
                "producer_semantics",
                "producer_source_sha256",
                "observer_source_sha256",
                "source_bindings",
            )
        },
        "effective_fmha_quant_mode": measurement.get("fmha_quant_mode"),
        "input_sha256": hashes,
        "resolved_model_identity": resolved,
        "systems_identity": system,
        "trace_source_sha256": {f"trace-{i:04d}": item["sha256"] for i, item in enumerate(audit["files"])},
        "prediction_sources_sha256": {
            name: file_hash(module.__file__)
            for name, module in {
                "analysis": __import__(__name__),
                "statistics": analyze_e2e,
                "normalization": normalize_fpm,
                "forward_comparison": compare_forward,
                "adapter": rust_engine_step,
                "model": deepseek_v41,
                "engine": engine,
                "native_extension": native,
            }.items()
        },
        "summary": trace_summary(rows),
        "independent_trial_summary": independent_trial_summary(results, final=not args.diagnostic),
        "cohorts": results,
    }
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
