# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay real requests with independent timing predictions and explicit cache state.

HTTP latency includes frontend/transport costs absent from the native engine.
No observed forward or response duration is supplied to the timing provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from analyze_e2e import analyze, finite_positive, quantile, trial_metrics
from compare_forward import (
    canonical,
    file_hash,
    resolved_model_identity,
)
from compare_trace import qualify_inputs, qualify_prediction_config, systems_identity


def positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(f"invalid resolved scheduler field: {field}")
    return value


def qualify_e2e_sources(measurement, client_path, scheduler_path):
    bindings = measurement["source_bindings"]
    for key, path in (
        ("client_summary_file_sha256", client_path),
        ("scheduler_receipt_file_sha256", scheduler_path),
    ):
        if bindings.get(key) != file_hash(path):
            raise ValueError("HTTP/scheduler evidence differs from the frozen measurement source: " + key)


def cold_cache_ack(control):
    return (
        control.get("policy") == "native-clear-before-cohort"
        and bool(control.get("acknowledgements"))
        and all(a.get("status") == "success" for a in control["acknowledgements"])
    )


def comparison_sources():
    import analyze_e2e
    import compare_forward
    import compare_trace
    import normalize_fpm

    import aisimulate._runtime as native
    from aiconfigurator_core.sdk import engine, rust_engine_step
    from aiconfigurator_core.sdk.models import deepseek_v41

    modules = {
        "trial_statistics": analyze_e2e,
        "forward_identity": compare_forward,
        "trace_qualification": compare_trace,
        "axis_bridge": normalize_fpm,
        "engine_compiler": engine,
        "native_python_bridge": rust_engine_step,
        "model_graph": deepseek_v41,
        "native_extension": native,
    }
    return {
        "e2e_comparison": file_hash(__file__),
        **{key: file_hash(module.__file__) for key, module in modules.items()},
    }


def timing_provider(config):
    return {
        "model": config["model_name"],
        "backend": config["backend"],
        "system": config["system_name"],
        "backend_version": config["backend_version"],
        "tp": 4,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "nextn": 0,
        "decoder_replay": config["decoder_replay"],
        "database_mode": config["database_mode"],
        "enable_shared_layer": False,
        "strict_provenance": True,
        "systems_path": config["systems_path"],
        "forward_model": config.get("forward_model", "op_level"),
        **({"fmha_dtype": config["activation_dtype"]} if config.get("activation_dtype") is not None else {}),
        **({"fpm_fmha_dtype": config["fpm_fmha_dtype"]} if config.get("fpm_fmha_dtype") is not None else {}),
    }


def sglang_engine(receipt, config):
    """Map observed settings, keeping unsupported scheduling details visible."""
    if receipt.get("schema") != "dsv41.serving.scheduler.receipt.v1":
        raise ValueError("requires the resolved SGLang scheduler receipt")
    actual, flags = receipt["actual_native_startup"], receipt["configured_scheduler"]
    expected = {
        "tp_size": 4,
        "dp_size": 1,
        "moe_dp_size": 1,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "disable_radix_cache": False,
        "enable_mixed_chunk": False,
        "schedule_policy": "fcfs",
        "speculative_algorithm": None,
        "enable_decoder_swa_bounded_replay": config["decoder_replay"],
    }
    if any(type(flags.get(k)) is not type(v) or flags.get(k) != v for k, v in expected.items()):
        raise ValueError("scheduler receipt differs from qualified SGLang serving policy")
    keys = (
        "page_size",
        "total_kv_blocks",
        "max_total_num_tokens",
        "max_running_requests",
        "max_prefill_tokens",
        "chunked_prefill_size",
        "model_context_len",
    )
    for field in keys:
        positive_int(actual[field], field)
    if actual["total_kv_blocks"] * actual["page_size"] != actual["max_total_num_tokens"]:
        raise ValueError("resolved logical KV capacity is inconsistent")
    return {
        "tensor_parallel_size": 4,
        "dp_size": 1,
        "num_gpu_blocks_is_explicit": True,
        "rank": {
            "backend": "sglang",
            "block_size": actual["page_size"],
            "num_gpu_blocks": actual["total_kv_blocks"],
            "max_num_seqs": actual["max_running_requests"],
            "max_num_batched_tokens": actual["max_prefill_tokens"],
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "sglang": {
                "schedule_policy": "fifo",
                "max_prefill_tokens": actual["max_prefill_tokens"],
                "chunked_prefill_size": actual["chunked_prefill_size"],
                "schedule_conservativeness": flags["schedule_conservativeness"],
            },
            "timing_model": {
                "type": "external",
                "provider": "aic",
                "config": timing_provider(config),
            },
        },
    }


def vllm_engine(receipt, config, plan):
    """Condition on observed absence of capacity pressure, without flattening pools."""
    if receipt.get("schema") != "dsv41.scheduler.resolved.v1":
        raise ValueError("requires the resolved vLLM scheduler receipt")
    parallel = receipt["parallel"]
    required = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
    }
    if any(type(parallel.get(k)) is not type(v) or parallel.get(k) != v for k, v in required.items()):
        raise ValueError("vLLM scheduler topology differs from pure TP4")
    if (
        receipt["model"]["enforce_eager"] is not True
        or receipt["speculative_config"] is not None
        or receipt["scheduler"]["policy"] != "fcfs"
        or receipt["cache"]["enable_prefix_caching"] is not True
        or receipt["scheduler"]["enable_chunked_prefill"] is not True
    ):
        raise ValueError("vLLM serving policy differs from the qualified eager contract")
    block = positive_int(receipt["scheduler_block_size_tokens"], "scheduler block size")
    max_len = positive_int(receipt["model"]["max_model_len"], "context limit")
    groups = {c["cohort_id"]: c for c in plan["cohorts"]}
    footprints = []
    for cohort in groups.values():
        requests = list(cohort["requests"])
        if cohort.get("seed_cohort_id"):
            requests += groups[cohort["seed_cohort_id"]]["requests"]
        if any(len(r["input_token_ids"]) + r["output_tokens"] > max_len for r in requests):
            raise ValueError("frozen workload exceeds the measured native context limit")
        # Enough logical blocks for every request in a cohort and its seed,
        # including one lookahead and two spare pages/request. This is computed
        # solely from the frozen workload, not observed latency or physical pools.
        footprints.append(sum((len(r["input_token_ids"]) + r["output_tokens"] + block) // block + 2 for r in requests))
    return {
        "tensor_parallel_size": 4,
        "dp_size": 1,
        "num_gpu_blocks_is_explicit": True,
        "rank": {
            "backend": "vllm",
            "block_size": block,
            "num_gpu_blocks": max(footprints),
            "max_model_len": max_len,
            "max_num_seqs": positive_int(receipt["max_running_requests"], "running requests"),
            "max_num_batched_tokens": positive_int(receipt["max_scheduled_tokens"], "scheduled tokens"),
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "timing_model": {
                "type": "external",
                "provider": "aic",
                "config": timing_provider(config),
            },
        },
    }


def capacity_eligibility(audit, cohort):
    """Require actual native outcomes for every interval and retain rejection reasons."""
    producers = audit["producer_audits"]
    if len(producers) != 1:
        return "missing single-producer cache pressure receipt"
    producer = producers[0]
    dispatch = producer["dispatch_audit"]
    final = dispatch.get("cache_pressure", {})
    if (
        final.get("schema") != "dsv41.cache.pressure.v1"
        or final.get("run_id") != producer["run_id"]
        or not final.get("epoch_id")
    ):
        return "missing native cache pressure identity"
    records = {r["dispatch_id"]: r for r in dispatch["records"]}
    by_location = {(r["file"], r["line"]): r for r in audit["active_rows"]}
    states = [final]
    for location in cohort["rows"]:
        row = by_location[location["file"], location["line"]]
        state = records.get(row["dispatch_id"], {}).get("cache_pressure")
        if state is None:
            return "missing per-dispatch cache pressure witness"
        states.append(state)
    for state in states:
        if (
            state.get("run_id") != final["run_id"]
            or state.get("epoch_id") != final["epoch_id"]
            or any(
                type(state.get(k)) is not int or state[k] != 0
                for k in ("allocation_refusals", "allocation_exceptions", "preemptions")
            )
            or type(state.get("allocation_attempts")) is not int
            or state["allocation_attempts"] < 1
            or type(state.get("watermark_blocks")) is not int
            or type(state.get("minimum_free_blocks")) is not int
            or state["minimum_free_blocks"] <= state["watermark_blocks"]
        ):
            return "native capacity pressure or incomplete pressure witness; unconstrained replay is ineligible"
    return None


def checked_requests(cohort, observed):
    """Join by request identity; preserve real input IDs and submit offsets."""
    if observed.get("valid") is not True:
        raise ValueError("cannot replay an invalid HTTP cohort as a completed comparison")
    by_id = {r["request_id"]: r for r in observed["requests"]}
    if len(by_id) != len(observed["requests"]) or set(by_id) != {r["request_id"] for r in cohort["requests"]}:
        raise ValueError("observed request identity differs from frozen plan")
    result = []
    for request in cohort["requests"]:
        row = by_id[request["request_id"]]
        tokens = request["input_token_ids"]
        if not tokens or any(type(t) is not int or not 0 <= t < 2**32 for t in tokens):
            raise ValueError("real input token IDs are required")
        digest = hashlib.sha256(canonical(tokens).encode()).hexdigest()
        if request["input_token_ids_sha256"] != digest or row["input_token_ids_sha256"] != digest:
            raise ValueError("real input token hash differs from plan/client")
        if (
            row.get("valid") is not True
            or row["prompt_tokens"] != len(tokens)
            or row["completion_tokens"] != request["output_tokens"]
        ):
            raise ValueError("HTTP usage does not match fixed request lengths")
        start = positive_int(row["started_monotonic_ns"], "request start")
        if start < observed["started"]["monotonic_ns"]:
            raise ValueError("request precedes cohort start")
        result.append(
            {
                "id": request["request_id"],
                "start_ns": start,
                "input_tokens": len(tokens),
                "input_token_ids": tokens,
                "output_tokens": positive_int(request["output_tokens"], "output length"),
            }
        )
    return result


def replay_spec(engine, cohort, observed, *, seed_cohort=None, seed_observed=None):
    """Fresh engine per cold cohort; seed A and reuse B share one cache."""
    requests = checked_requests(cohort, observed)
    control = observed["cache_control"]
    origin = observed["started"]["monotonic_ns"]
    if cohort["purpose"] == "prefix-reuse-B":
        if (
            seed_cohort is None
            or seed_observed is None
            or seed_cohort["cohort_id"] != cohort["seed_cohort_id"]
            or seed_cohort["trial_index"] != cohort["trial_index"]
            or seed_cohort["purpose"] != "prefix-warm-A"
            or control.get("policy") != "preserve-A-prefix"
        ):
            raise ValueError("prefix reuse requires its same-trial seed cohort")
        if seed_observed["finished"]["monotonic_ns"] > origin:
            raise ValueError("prefix B was submitted before seed A completed")
        if not cold_cache_ack(seed_observed["cache_control"]):
            raise ValueError("seed A must start from an acknowledged cold cache")
        requests = checked_requests(seed_cohort, seed_observed) + requests
        origin = seed_observed["started"]["monotonic_ns"]
    elif not cold_cache_ack(control):
        raise ValueError("cold comparison requires successful native cache-clear acknowledgements")
    materialized = []
    for request in requests:
        value = dict(request)
        value["arrival_time_ms"] = (value.pop("start_ns") - origin) / 1e6
        materialized.append(value)
    return {
        "version": 1,
        "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
        "engine": deepcopy(engine),
        "record_per_request": True,
        "requests": materialized,
    }, (observed["started"]["monotonic_ns"] - origin) / 1e6


def observed_metrics(cohort):
    """Add response completion measures without changing the frozen N analysis."""
    metrics = trial_metrics(cohort)
    if not metrics:
        return metrics
    for output, field in (
        ("request_latency_ms", "latency_ms"),
        ("last_token_latency_ms", "last_token_latency_ms"),
    ):
        values = [r.get(field) for r in cohort["requests"]]
        if all(finite_positive(v) for v in values):
            metrics[output] = statistics.mean(values)
    return metrics


def predicted_metrics(report, request_ids, cohort_start_ms):
    records = report["per_request"]
    by_id = {r["request_id"]: r for r in records}
    if len(by_id) != len(records) or not set(request_ids).issubset(by_id):
        raise ValueError("native replay omitted or duplicated requests")
    selected = [by_id[rid] for rid in request_ids]
    if any(r["terminal_status"] != "completed" for r in selected):
        raise ValueError("native replay did not complete all comparison requests")
    duration = max(r["terminal_time_ms"] for r in selected) - cohort_start_ms
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("native replay has an invalid cohort completion time")
    metrics = {
        "ttft_ms": statistics.mean(r["ttft_ms"] for r in selected),
        "request_latency_ms": statistics.mean(r["terminal_time_ms"] - r["arrival_time_ms"] for r in selected),
        "last_token_latency_ms": statistics.mean(r["last_token_ms"] - r["arrival_time_ms"] for r in selected),
        "output_tokens_per_second": 1000 * sum(r["output_length"] for r in selected) / duration,
    }
    if all(r["output_length"] > 1 for r in selected):
        metrics["average_tpot_ms"] = statistics.mean(
            (r["last_token_ms"] - r["first_token_ms"]) / (r["output_length"] - 1) for r in selected
        )
        # The native report exposes each request's mean ITL, not its token-gap distribution.
        metrics["exact_itl_ms"] = statistics.mean(r["itl_ms"] for r in selected)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in metrics.values()):
        raise ValueError("native replay returned non-positive/non-finite metrics")
    return metrics, [
        {
            k: r[k]
            for k in (
                "request_id",
                "arrival_time_ms",
                "ttft_ms",
                "first_token_ms",
                "last_token_ms",
                "terminal_time_ms",
                "itl_ms",
                "input_length",
                "output_length",
                "reused_input_tokens",
                "terminal_status",
            )
        }
        for r in selected
    ]


def attach_cache_comparison(result, audited_cohort):
    """Expose scheduling/cache disagreement without removing its timing error."""
    observed = {
        r["request_id"]: r["initial_cross_request_cached_tokens"]
        for r in audited_cohort["native_request_proof"]["requests"]
    }
    result["native_initial_cached_tokens"] = observed
    result["cache_semantics_match"] = None
    if result["status"] == "predicted":
        matches = []
        for request in result["requests"]:
            request["native_initial_cached_tokens"] = observed[request["request_id"]]
            request["cache_reuse_matches_native"] = (
                request["reused_input_tokens"] == request["native_initial_cached_tokens"]
            )
            matches.append(request["cache_reuse_matches_native"])
        result["cache_semantics_match"] = all(matches)


def compare_cohort(engine, cohort, observed, native_replay, *, seed_cohort=None, seed_observed=None):
    spec, offset = replay_spec(engine, cohort, observed, seed_cohort=seed_cohort, seed_observed=seed_observed)
    result = {
        "cohort_id": cohort["cohort_id"],
        "purpose": cohort["purpose"],
        "trial_index": cohort["trial_index"],
        "trial_seed": cohort["trial_seed"],
        "declared_coverage_role": cohort.get("coverage_role", "unspecified"),
        "corpus_role": cohort.get("corpus_role", "unspecified"),
        "observed": observed_metrics(observed),
        "replay_spec_sha256": hashlib.sha256(canonical(spec).encode()).hexdigest(),
    }
    try:
        raw = json.loads(native_replay(canonical(spec)))
        prediction, records = predicted_metrics(raw, [r["request_id"] for r in cohort["requests"]], offset)
    except Exception as error:
        result.update(
            status="prediction_unavailable",
            failure_type=type(error).__name__,
            failure=str(error),
        )
    else:
        result.update(
            status="predicted",
            prediction=prediction,
            requests=records,
            native_result_sha256=hashlib.sha256(canonical(raw).encode()).hexdigest(),
        )
        result["signed_error_percent"] = {
            k: 100 * (prediction[k] / v - 1) for k, v in result["observed"].items() if k in prediction
        }
    return result


def paired_summary(rows, *, final, seed=94051000):
    """Bootstrap whole independent trials within each scenario, pairing model/data."""
    points = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: {"planned_trials": 0, "predicted_trials": 0})
    seen = set()
    for row in rows:
        key = (row["purpose"], row["trial_index"])
        if key in seen:
            raise ValueError("duplicate independent trial within scenario")
        seen.add(key)
        counts[row["purpose"]]["planned_trials"] += 1
        if row["status"] != "predicted":
            continue
        counts[row["purpose"]]["predicted_trials"] += 1
        for metric, observed in row["observed"].items():
            if metric in row["prediction"]:
                points[row["purpose"]][metric].append((observed, row["prediction"][metric]))
    out = {}
    for purpose, metrics in points.items():
        out[purpose] = {"coverage": counts[purpose], "metrics": {}}
        for metric, pairs in metrics.items():
            errors = [100 * (p / o - 1) for o, p in pairs]
            value = {
                "independent_trials": len(pairs),
                "observed_mean": statistics.mean(o for o, _ in pairs),
                "predicted_mean": statistics.mean(p for _, p in pairs),
                "mean_signed_error_percent": statistics.mean(errors),
                "median_absolute_error_percent": statistics.median(abs(e) for e in errors),
                "p90_absolute_error_percent_across_trials": quantile([abs(e) for e in errors], 0.9),
                "wape_percent": 100 * sum(abs(p - o) for o, p in pairs) / sum(o for o, _ in pairs),
            }
            if final and len(pairs) >= 20 and len(pairs) == counts[purpose]["planned_trials"]:
                rng = random.Random(seed)
                value.update(bootstrap_resamples=5000, bootstrap_seed=seed)
                draws = [rng.choices(pairs, k=len(pairs)) for _ in range(5000)]
                for name, fn in (
                    (
                        "observed_mean",
                        lambda sample: statistics.mean(o for o, _ in sample),
                    ),
                    (
                        "paired_ratio_error_percent",
                        lambda sample: 100 * (sum(p for _, p in sample) / sum(o for o, _ in sample) - 1),
                    ),
                ):
                    values = [fn(sample) for sample in draws]
                    value[name + "_bootstrap_ci95"] = [
                        quantile(values, 0.025),
                        quantile(values, 0.975),
                    ]
            out[purpose]["metrics"][metric] = value
    for purpose in counts:
        out.setdefault(purpose, {"coverage": counts[purpose], "metrics": {}})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "audit",
        "plan",
        "client-summary",
        "measurement",
        "scheduler-receipt",
        "prediction-config",
        "output",
    ):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    fields = (
        "audit",
        "plan",
        "client_summary",
        "measurement",
        "scheduler_receipt",
        "prediction_config",
    )
    hashes = {k: file_hash(getattr(args, k)) for k in fields}
    inputs = {k: json.loads(getattr(args, k).read_bytes()) for k in fields}
    audit, plan, client, measurement, receipt, config = (inputs[k] for k in fields)
    audited = qualify_inputs(audit, plan, measurement, diagnostic=args.diagnostic)
    qualify_prediction_config(config, measurement)
    qualify_e2e_sources(measurement, args.client_summary, args.scheduler_receipt)
    if (
        receipt["run_id"] != plan["run_id"]
        or client["run_id"] != plan["run_id"]
        or client["plan_sha256"] != file_hash(args.plan)
    ):
        raise ValueError("client/scheduler source does not bind to the original plan/run")
    engine = (
        sglang_engine(receipt, config) if measurement["backend"] == "sglang" else vllm_engine(receipt, config, plan)
    )
    client_analysis = analyze(plan, client["cohorts"])
    if not args.diagnostic and not client_analysis["client_coverage_complete"]:
        raise ValueError("final E2E comparison requires complete independent main trials")
    import aisimulate._runtime as native

    sources = comparison_sources()
    system_identity = systems_identity(config)
    model_identity = resolved_model_identity(
        config,
        {"input_provenance": {"config_sha256": measurement["model_config_canonical_sha256"]}},
    )
    observed = {c["cohort_id"]: c for c in client["cohorts"]}
    planned = {c["cohort_id"]: c for c in plan["cohorts"]}
    results = []
    for cohort in plan["cohorts"]:
        key = cohort["cohort_id"]
        if cohort["comparison_role"] != "primary" or key not in audited:
            continue
        seed = cohort.get("seed_cohort_id")
        capacity_failure = capacity_eligibility(audit, audited[key]) if measurement["backend"] == "vllm" else None
        if capacity_failure:
            results.append(
                {
                    "cohort_id": key,
                    "purpose": cohort["purpose"],
                    "trial_index": cohort["trial_index"],
                    "trial_seed": cohort["trial_seed"],
                    "declared_coverage_role": cohort.get("coverage_role", "unspecified"),
                    "corpus_role": cohort.get("corpus_role", "unspecified"),
                    "observed": observed_metrics(observed[key]),
                    "status": "prediction_unavailable",
                    "failure_type": "CapacityQualificationError",
                    "failure": capacity_failure,
                }
            )
            continue
        result = compare_cohort(
            engine,
            cohort,
            observed[key],
            native.run_replay_json,
            seed_cohort=planned.get(seed),
            seed_observed=observed.get(seed),
        )
        attach_cache_comparison(result, audited[key])
        results.append(result)
    if system_identity != systems_identity(config):
        raise ValueError("prediction tables changed during replay")
    if hashes != {k: file_hash(getattr(args, k)) for k in fields}:
        raise ValueError("audited inputs changed during replay")
    if sources != comparison_sources():
        raise ValueError("prediction or analysis sources changed during replay")
    public_config, public_engine = deepcopy(config), deepcopy(engine)
    if Path(config["systems_path"]).is_absolute():
        public_config["systems_path"] = "<explicit systems overlay; see hashed inventory>"
        public_engine["rank"]["timing_model"]["config"]["systems_path"] = public_config["systems_path"]
    for row in results:
        if "failure" in row:
            row["failure"] = row["failure"].replace(config["systems_path"], "<explicit systems overlay>")
    report = {
        "schema": "dsv41.e2e.comparison.v1",
        "run_id": plan["run_id"],
        "complete_requested_study": not args.diagnostic,
        "scope": "closed-segment diagnostic" if args.diagnostic else "complete independent main trials",
        "measurement_identity": {
            k: measurement[k]
            for k in (
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
                "model_config_canonical_sha256",
                "source_bindings",
            )
        },
        "prediction_config": public_config,
        "effective_fmha_quant_mode": measurement.get("fmha_quant_mode"),
        "replay_engine": public_engine,
        "capacity_policy": "observed logical capacity"
        if measurement["backend"] == "sglang"
        else "capacity-unconstrained logical replay, admitted only with native no-pressure witnesses",
        "input_sha256": hashes,
        "analysis_source_sha256": file_hash(__file__),
        "native_extension_sha256": file_hash(native.__file__),
        "prediction_sources": sources,
        "systems_identity": system_identity,
        "resolved_model_identity": model_identity,
        "correction_fitting": False,
        "client_analysis": client_analysis,
        "supplementary_metrics": {
            "request_latency_ms": "request arrival to response completion",
            "last_token_latency_ms": "request arrival to last output token",
            "sampling_role": "post-plan descriptive metrics; excluded from the frozen pilot N rule",
        },
        "limitations": [
            "HTTP includes frontend, transport and response completion; native replay excludes them.",
            "The inspected SGLang and vLLM replay paths charge a separate first-output decode after prefill.",
            "Runtime overlap and dynamic admission policies are approximated by the existing scheduler.",
            "Runtime context limit is recorded but not a SGLang replay control; this study stays below it.",
            "SGLang uses observed logical capacity. vLLM's heterogeneous pools are not flattened: "
            "logical capacity is an unconstrained frozen-workload envelope, allowed only with native zero-pressure "
            "evidence. Allocator memory accuracy is not validated.",
            "ITL comparison is the mean per request, equally weighted per trial; no modeled token-tail distribution.",
        ],
        "points": paired_summary(results, final=not args.diagnostic),
        "cohorts": results,
    }
    with args.output.open("x") as out:
        out.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
