# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render admitted physical-segment comparisons; never fit predictions or pool CIs."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

MODES = ("sol", "hybrid", "silicon")
COLORS = {"sol": "#9271B1", "hybrid": "#D88320", "silicon": "#157A8C"}
METRICS = {
    "ttft_ms": "TTFT",
    "average_tpot_ms": "Mean time per output token",
    "output_tokens_per_second": "Finite-cohort throughput",
    "request_latency_ms": "Request completion latency",
    "last_token_latency_ms": "Time to last output token",
    "exact_itl_ms": "Mean inter-token latency",
}


REPORT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "descriptive_metrics.py").is_file())
_metrics_spec = importlib.util.spec_from_file_location(
    "dsv41_descriptive_metrics", REPORT_ROOT / "descriptive_metrics.py"
)
descriptive_update = importlib.util.module_from_spec(_metrics_spec)
_metrics_spec.loader.exec_module(descriptive_update)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def descriptive(pairs, observed_count):
    require(
        type(observed_count) is int and observed_count >= len(pairs),
        "invalid coverage denominator",
    )
    result = {
        "observed": observed_count,
        "predicted": len(pairs),
        "missing": observed_count - len(pairs),
    }
    if not pairs:
        return result
    require(
        all(math.isfinite(v) and v > 0 for pair in pairs for v in pair),
        "invalid metric value",
    )
    errors = [100 * (p / o - 1) for o, p in pairs]
    result.update(
        mape_percent=statistics.mean(abs(e) for e in errors),
        mean_signed_error_percent=statistics.mean(errors),
        median_ape_percent=statistics.median(abs(e) for e in errors),
        p90_ape_percent=quantile([abs(e) for e in errors], 0.9),
        wape_percent=100 * sum(abs(p - o) for o, p in pairs) / sum(o for o, _ in pairs),
    )
    return result


def flatten(report, field):
    return [row for segment in report["segments"] for row in segment[field]]


def native_view(rows):
    return [
        (
            r["cohort_id"],
            [(i["dispatch_id"], i["phase"], i["observed_ms"], i["native_work_role"]) for i in r["intervals"]],
        )
        for r in rows
    ]


def validate_paired_outputs(paired, report):
    require(
        paired["schema"] == "dsv41.paired.returned-token-ids.v1"
        and paired["paired_request_count"] == 960
        and paired["paired_cohort_count"] == 680
        and paired["paired_trial_count"] == 40
        and paired["sampling_exclusion"] is False
        and paired["timing_observations_unchanged"] is True,
        "invalid paired output evidence",
    )
    require(
        paired["physical_runs"]["on"] == report["segments"][0]["physical_run_id"],
        "paired output physical lifecycle differs",
    )
    rows = paired["requests"]
    require(
        len(rows) == 960
        and len({r["on_request_id"] for r in rows}) == 960
        and len({r["off_request_id"] for r in rows}) == 960
        and collections.Counter(r["trial_index"] for r in rows) == dict.fromkeys(range(40), 24)
        and all(r["trial_seed"] == 93051000 + r["trial_index"] for r in rows),
        "paired output request or frozen trial identity differs",
    )
    counts = collections.Counter(r["status"] for r in rows)
    require(
        set(counts) <= {"equal", "different", "unavailable"} and dict(counts) == paired["requests_by_status"],
        "paired output counts differ from actual rows",
    )


def validate_reports(reports, plan, *, allow_partial):
    require(set(reports) == set(MODES), "all three independently computed modes are required")
    require(plan["requested_trials"] == 100 and plan["sampling_role"] == "main", "wrong frozen ON main plan")
    require(len({c["cohort_id"] for c in plan["cohorts"]}) == len(plan["cohorts"]), "duplicate logical cohort")
    identities = []
    originals = plan["cohorts"]
    require({c["trial_index"] for c in originals} == set(range(100)), "frozen plan omits a trial")
    require(all(c["trial_seed"] == 93051000 + c["trial_index"] for c in originals), "frozen trial seed changed")
    for mode, report in reports.items():
        require(report["schema"] == "dsv41.closed.segments.comparison.v1", "unknown segment comparison schema")
        require(report["logical_run_id"] == plan["run_id"], "logical lifecycle differs")
        expected = dict(
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
        )
        cfg = report["prediction_config"]
        require(
            all(type(cfg.get(k)) is type(v) and cfg[k] == v for k, v in expected.items()), "prediction policy differs"
        )
        require(
            all(cfg.get(k) is None for k in ("activation_dtype", "fpm_fmha_dtype", "weight_dtype", "moe_dtype")),
            "unexpected precision override",
        )
        summary = report["summary"]
        require(summary["pooled_confidence_intervals"] is None, "pooled cross-lifecycle CI is prohibited")
        require(summary["statistical_precision_completed"] is False, "cross-lifecycle precision cannot be qualified")
        require(summary["fixed_plan_completed"] == summary["coverage_complete"], "completion fields disagree")
        require(allow_partial or summary["fixed_plan_completed"], "incomplete main requires explicit partial rendering")
        offset, seen_runs = 0, set()
        segment_identity = []
        for segment in report["segments"]:
            run = segment["physical_run_id"]
            require(
                run not in seen_runs and segment["receipt"]["valid"] is True, "duplicate or unqualified physical run"
            )
            seen_runs.add(run)
            count = segment["observed_original_cohorts"]
            require(type(count) is int and count > 0, "empty physical segment")
            covered = originals[offset : offset + count]
            require(len(covered) == count, "segment exceeds frozen plan")
            primary = [c for c in covered if c["comparison_role"] == "primary"]
            for field in ("e2e_cases", "trace_cases"):
                require(
                    [r["cohort_id"] for r in segment[field]] == [c["cohort_id"] for c in primary],
                    "boundary cohort missing, duplicated, or reordered",
                )
                for row, original in zip(segment[field], primary, strict=True):
                    require(
                        all(row[k] == original[k] for k in ("trial_index", "trial_seed", "purpose")),
                        "original trial identity changed",
                    )
            ids = {c["cohort_id"] for c in covered}
            complete = sorted(
                i
                for i in range(plan["requested_trials"])
                if {c["cohort_id"] for c in originals if c["trial_index"] == i} <= ids
            )
            stat = segment["statistics"]
            require(stat["complete_same_lifecycle_trial_indices"] == complete, "ineligible complete-trial CI subset")
            require(
                stat["boundary_trial_indices"] == sorted({c["trial_index"] for c in covered} - set(complete)),
                "boundary trial was lost",
            )
            require("fixed wall-deadline" in stat["ci_scope"], "CI stopping-boundary qualification missing")
            for point in stat["complete_trial_e2e"].values():
                for values in point["metrics"].values():
                    if "paired_ratio_error_percent_bootstrap_ci95" in values:
                        require(
                            len(complete) >= 20 and point["coverage"]["predicted_trials"] == len(complete),
                            "CI includes missing predictions or too few trials",
                        )
            segment_identity.append((run, count, segment["receipt"]))
            offset += count
        require(summary["fixed_plan_completed"] == (offset == len(originals)), "incorrect logical completion")
        observed = flatten(report, "e2e_cases")
        require(summary["observed_primary_cohorts"] == len(observed), "incorrect observed coverage")
        require(
            summary["predicted_e2e_cohorts"] == sum(r["status"] == "predicted" for r in observed),
            "incorrect prediction coverage",
        )
        require(
            summary["planned_primary_cohorts"] == sum(c["comparison_role"] == "primary" for c in originals),
            "wrong planned denominator",
        )
        identities.append(
            (
                segment_identity,
                [(r["cohort_id"], r["observed"]) for r in observed],
                native_view(flatten(report, "trace_cases")),
                report["prediction_sources"],
                report["systems_identity"],
            )
        )
    require(
        all(identity == identities[0] for identity in identities[1:]),
        "modes differ in actual data, source, or calibration",
    )


def summarize(reports):
    http = {m: flatten(r, "e2e_cases") for m, r in reports.items()}
    common = set.intersection(*[{r["cohort_id"] for r in rows if r["status"] == "predicted"} for rows in http.values()])
    native = {
        mode: {
            (case["physical_run_id"], case["cohort_id"], row["dispatch_id"]): row
            for case in flatten(report, "trace_cases")
            for row in case["intervals"]
        }
        for mode, report in reports.items()
    }
    require(
        all(
            len(native[mode]) == sum(len(case["intervals"]) for case in flatten(report, "trace_cases"))
            for mode, report in reports.items()
        ),
        "duplicate physical native interval in report",
    )
    common_native = set.intersection(
        *[{key for key, row in values.items() if row["status"] == "predicted"} for values in native.values()]
    )
    result = {
        "common_http_cohorts": len(common),
        "common_native_intervals": len(common_native),
        "http": {},
        "http_common_support": {},
        "trace": {},
        "trace_common_support": {},
        "missing": {},
    }
    for mode, report in reports.items():
        for label, rows in (
            ("http", http[mode]),
            ("http_common_support", [r for r in http[mode] if r["cohort_id"] in common]),
        ):
            result[label][mode] = {}
            for metric in METRICS:
                selected = [r for r in rows if metric in r["observed"]]
                pairs = [
                    (r["observed"][metric], r["prediction"][metric])
                    for r in selected
                    if r["status"] == "predicted" and metric in r["prediction"]
                ]
                result[label][mode][metric] = descriptive(pairs, len(selected))
        intervals = list(native[mode].values())
        for label, subset in (
            ("trace", intervals),
            ("trace_common_support", [native[mode][key] for key in sorted(common_native)]),
        ):
            result[label][mode] = {}
            for phase in ("all", "prefill", "decode", "mixed"):
                rows = [r for r in subset if phase == "all" or r["phase"] == phase]
                result[label][mode][phase] = descriptive(
                    [(r["observed_ms"], r["predicted_ms"]) for r in rows if r["status"] == "predicted"], len(rows)
                )
        result["missing"][mode] = {
            "http": [
                {
                    k: r[k]
                    for k in (
                        "physical_run_id",
                        "cohort_id",
                        "purpose",
                        "trial_index",
                        "failure_category",
                        "failure",
                        "failure_sha256",
                    )
                }
                for r in http[mode]
                if r["status"] != "predicted"
            ],
            "trace_failure_groups": dict(
                collections.Counter(
                    (r["failure_type"] + ": " + r["failure"]) for r in intervals if r["status"] != "predicted"
                )
            ),
            "cache_mismatches": sum(r.get("cache_semantics_match") is False for r in http[mode]),
        }
    return result


def table_line(mode, metric, value):
    cols = [mode.upper(), metric, f"{value['predicted']}/{value['observed']}"]
    cols += (
        [
            f"{value[key]:+.2f}%" if key == "mean_signed_error_percent" else f"{value[key]:.2f}%"
            for key in (
                "mean_signed_error_percent",
                "median_ape_percent",
                "p90_ape_percent",
                "mape_percent",
                "wape_percent",
            )
        ]
        if value["predicted"]
        else ["unavailable"] * 5
    )
    return "| " + " | ".join(cols) + " |"


def validate_content_controls(value, report, plan):
    require(
        value["schema"] == "dsv41.segment-content-controls.v1"
        and value["content_coverage_complete"] is True
        and value["sampling_exclusion"] is False
        and value["missing_cohort_ids"] == []
        and value["logical_run_id"] == plan["run_id"]
        and value["logical_plan_sha256"] == report["input_files_sha256"]["logical_plan"],
        "content diagnostic plan/coverage differs",
    )
    expected_bindings = [
        {
            "physical_run_id": s["physical_run_id"],
            "closed_audit_sha256": s["receipt"]["source_inputs_sha256"]["closed_audit"],
            "execution_sha256": s["receipt"]["source_inputs_sha256"]["execution"],
            "original_summary_sha256": s["receipt"]["source_inputs_sha256"]["combined_summary"],
        }
        for s in report["segments"]
    ]
    require(value["bindings"] == expected_bindings, "content diagnostic physical evidence differs")
    cases = {
        c["cohort_id"]: c for c in plan["cohorts"] if c["purpose"] in ("engram-distinct-text", "engram-repeated-text")
    }
    rows = value["records"]
    require(
        len(rows) == len(cases) and {r["cohort_id"] for r in rows} == set(cases), "content cases missing/duplicated"
    )
    run_by_cohort = {c["cohort_id"]: s["physical_run_id"] for s in report["segments"] for c in s["e2e_cases"]}
    for row in rows:
        case = cases[row["cohort_id"]]
        require(
            all(row[k] == case[k] for k in ("trial_index", "trial_seed", "purpose"))
            and row["physical_run_id"] == run_by_cohort[row["cohort_id"]]
            and row["request_ids"] == [r["request_id"] for r in case["requests"]]
            and row["input_hashes"] == [r["input_token_ids_sha256"] for r in case["requests"]],
            "content diagnostic changed frozen inputs",
        )
        require(
            row["initial_cached_tokens"] == [0, 0]
            and row["same_first_prefill_dispatch"]
            == (row["first_prefill_dispatch_ids"][0] == row["first_prefill_dispatch_ids"][1])
            and row["cold_same_first_prefill_dispatch"] == row["same_first_prefill_dispatch"],
            "native content cache/batch diagnostic differs",
        )
        sequences = [r["input_token_ids"] for r in case["requests"]]
        require(
            len(sequences) == 2 and row["equal_sequences"] == (sequences[0] == sequences[1]), "sequence control differs"
        )
        grams = [{tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)} for tokens in sequences]
        actual = row["raw_ngram_overlap"]["3"]
        require(
            actual["left_unique"] == len(grams[0])
            and actual["right_unique"] == len(grams[1])
            and actual["intersection"] == len(grams[0] & grams[1])
            and actual["union"] == len(grams[0] | grams[1]),
            "raw trigram diagnostic differs from original tokens",
        )
    require(
        set(value["summary"]) == {"engram-distinct-text", "engram-repeated-text"},
        "content scenario summary missing",
    )
    for purpose, summary in value["summary"].items():
        selected = [r for r in rows if r["purpose"] == purpose]
        require(summary["observed_trials"] == summary["planned_trials"] == len(selected), "content trial count differs")
        for key in ("equal_sequences", "same_first_prefill_dispatch", "cold_same_first_prefill_dispatch"):
            require(summary[key] == sum(r[key] for r in selected), "content summary differs from observations")
        require(
            summary["same_raw_3gram_sets"]
            == sum(
                r["raw_ngram_overlap"]["3"]["intersection"] == r["raw_ngram_overlap"]["3"]["union"] for r in selected
            ),
            "content trigram summary differs",
        )
        require(
            summary["same_raw_3gram_sets"] == len(selected)
            and summary["equal_sequences"] == (len(selected) if purpose == "engram-repeated-text" else 0)
            and summary["initial_cache_patterns"] == {"[0, 0]": len(selected)},
            "content-control narrative does not describe the actual observations",
        )


def runtime_continuity_lines(segments):
    proofs = [s["receipt"].get("runtime_identity_equivalence") for s in segments]
    if not any(proofs):
        return []  # Preserve reproduction of the earlier legacy report schema.
    require(all(isinstance(p, dict) for p in proofs), "incomplete runtime continuity evidence")
    lines = [
        "",
        "## Verified runtime continuity",
        "",
        "| Segment | Actual runtime identity SHA256 (prefix) | Comparison reference SHA256 (prefix) | Admission |",
        "|---|---|---|---|",
    ]
    reviewed = []
    for index, (segment, proof) in enumerate(zip(segments, proofs, strict=True), 1):
        actual = proof["actual_runtime_identity_sha256"]
        reference = segment["receipt"]["runtime_identity_sha256"]
        require(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in (actual, reference)), "invalid runtime identity")
        kind = proof["kind"]
        if kind == "exact":
            require(actual == reference, "exact runtime identity changed")
            label = "exact origin identity"
        else:
            require(
                kind == "reviewed_aggregated_bootstrap_port_only"
                and proof["comparison_reference_runtime_identity_sha256"] == reference
                and actual != reference,
                "unknown runtime equivalence",
            )
            reviewed.append(proof)
            label = "reviewed inactive PD bootstrap port and continuation-control change"
        lines.append(f"| {index} | `{actual[:16]}` | `{reference[:16]}` | {label} |")
    if reviewed:
        lines += [
            "",
            "Both actual server configurations use the literal `null` disaggregation mode. "
            "The successor's automatically allocated integer PD bootstrap port differs; the other 494 "
            "server arguments are exactly equal. Actual installed parser and mode-consumer sources "
            "establish that this port is unused by the aggregated inference path. A separately reviewed "
            "change to `continuation.py` admits only this difference; the regenerated bundle manifest "
            "records that change. Every other measurement, observer and client source remains identical. "
            "The original failed attempt produced no HTTP observations and remains in the private evidence.",
            "",
            "The continuation's actual configuration, control-source hashes, original audit/progress and "
            "exact remaining frozen plan are checked before assigning the origin comparison reference. "
            "Both raw runtime identities and the original configuration hashes remain in the per-segment "
            "receipts, together with the immutable review addendum. This exception neither changes the "
            "frozen sample budget nor permits pooling confidence intervals across physical lifecycles.",
        ]
    return lines


def markdown(reports, plan, budget, summary, paired_outputs=None, content_controls=None):
    first = reports["hybrid"]
    complete = first["summary"]["fixed_plan_completed"]
    lines = [
        "# Real silicon versus prediction: GB300 TP4 / Decoder ON / physical segments",
        "",
        f"Frozen logical-plan coverage is **{'complete' if complete else 'partial'}**: "
        f"{first['summary']['observed_primary_cohorts']}/{first['summary']['planned_primary_cohorts']} "
        "metric-bearing cohorts "
        f"across {len(first['segments'])} independently closed physical runtimes. "
        "This report preserves each lifecycle; it does not treat their combined observations as one run.",
        "",
        "The pinned SGLang eager text runtime, real requests and native FPM intervals are compared with unchanged "
        "SOL, HYBRID and strict SILICON models. The prefix-refined calibration overlay is fixed before these "
        "comparisons. No correction factor is fitted. [The OFF report](../off-prefix-refined/README.md) and "
        "[independent forward holdouts](../../prefix-refinement-v1/README.md) retain their distinct timing boundaries.",
        "Here FPM means the observed per-iteration telemetry. All three prediction modes use the op-level "
        "engine; this report does not qualify a GB300 whole-forward FPM lookup table.",
        "",
        f"The independent 10-trial pilot requested N={budget['required_trials']}; the prespecified cap froze "
        f"N={budget['stage_trials']} trials per scenario. The cap does not establish the requested 5% precision. "
        "The pilot rule takes the largest scenario/metric requirement from (1.96 * CV / 0.05)^2, "
        "with at least 20 trials and rounding up to a multiple of 10 before the cap. "
        "The deadline selects the next complete cohort boundary; a trial may span runtimes. All boundary-trial "
        "cohorts remain in descriptive errors and coverage. Only trials with every scenario in one physical "
        "runtime enter that runtime's approximate whole-trial bootstrap, and only with at least 20 complete "
        "trials and full prediction coverage for the metric. These pointwise 95% intervals are conditional "
        "on the observed lifecycle and fixed-deadline stopping boundary, not a separately powered sample "
        "or simultaneous coverage of all metrics. **No cross-lifecycle confidence interval is published.**",
        "",
        "## Physical lifecycle coverage",
        "",
        "| Segment | Original cohorts | Complete trial indices for conditional CI | Boundary trial indices | "
        "Native intervals in complete physical audit |",
        "|---|---:|---|---|---:|",
    ]
    for i, segment in enumerate(first["segments"], 1):
        stat = segment["statistics"]
        eligible = stat["complete_same_lifecycle_trial_indices"]
        text = f"{eligible[0]}-{eligible[-1]} ({len(eligible)})" if eligible else "none"
        lines.append(
            f"| {i} | {segment['observed_original_cohorts']} | {text} | {stat['boundary_trial_indices']} | "
            f"{segment['closed_native_intervals']} |"
        )
    lines += [
        "",
        "Physical audit totals also include setup, warmup, pilot and unreturned overlap work. Only "
        "the frozen primary main cohorts contribute to comparison errors below. TP ranks and consecutive "
        "native intervals do not increase the independent trial count.",
    ]
    lines += runtime_continuity_lines(first["segments"])
    lines += [
        "",
        "![Descriptive MAPE and WAPE](descriptive-error-comparison.png)",
        "",
        "[Descriptive metric receipt](descriptive-metrics.json) binds the unchanged comparison files. "
        "These additions do not change conditional lifecycle confidence intervals or create a pooled interval.",
        "",
        "## HTTP serving errors",
        "",
        "Descriptive MAPE is `100 * mean(abs(prediction / observation - 1))`, weighting supported "
        "scenario/trial cohorts equally. WAPE uses the same supported pairs and is total absolute "
        "error divided by total observed value. p90 APE is a percentile of prediction errors, not p90 "
        "request latency. Error values use available predictions; missing predictions remain in the "
        "coverage denominator. The common-support table compares identical predicted cohorts across modes. "
        "HTTP mean inter-token latency does not establish exact per-token or tail gaps.",
        "",
        "| Mode | Metric | Predicted / observed | Mean signed error | Median APE | p90 APE | MAPE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for metric, label in METRICS.items():
            lines.append(table_line(mode, label, summary["http"][mode][metric]))
    lines += [
        "",
        f"### Same supported subset: {summary['common_http_cohorts']} cohorts",
        "",
        "| Mode | Metric | Predicted / observed | Mean signed error | Median APE | p90 APE | MAPE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for metric, label in METRICS.items():
            lines.append(table_line(mode, label, summary["http_common_support"][mode][metric]))
    lines += [
        "",
        "## Native forward intervals",
        "",
        "The target is the native SGLang GPU-event interval. It differs from HTTP E2E and synchronized "
        "prepare/forward/sample benchmark holdouts. Every attributed interval, including unreturned "
        "overlap output, is retained. The table is descriptive over correlated intervals; per-scenario "
        "conditional whole-trial intervals are retained separately in each segment result.",
        "",
        "| Mode | Phase | Predicted / observed | Mean signed error | Median APE | p90 APE | MAPE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for phase, value in summary["trace"][mode].items():
            lines.append(table_line(mode, phase, value))
    lines += [
        "",
        f"### Same supported native subset: {summary['common_native_intervals']} intervals",
        "",
        "Different coverage can make an error summary look better. This table compares the same "
        "physical-run / cohort / dispatch intervals in all modes, without dropping missing rows from "
        "the full-coverage table above.",
        "",
        "| Mode | Phase | Predicted / observed | Mean signed error | Median APE | p90 APE | MAPE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for phase, value in summary["trace_common_support"][mode].items():
            lines.append(table_line(mode, phase, value))
    lines += ["", "## Missing predictions and observed execution limits", ""]
    for mode in MODES:
        missing = summary["missing"][mode]
        lines.append(
            f"- {mode.upper()}: {len(missing['http'])} HTTP cohorts lack predictions; "
            f"{sum(missing['trace_failure_groups'].values())} native intervals lack predictions; "
            f"{missing['cache_mismatches']} predicted cohorts disagree with observed initial prefix reuse."
        )
    lines += [
        "",
        "The retained failures identify heterogeneous Decoder-ON prefill query/prefix batches that the "
        "aggregate input cannot represent, plus strict SILICON decode at batch 3 outside the batch-1/2 "
        "calibration grid. The prefix-refined buckets resolve the earlier shifted-prefix gaps for the "
        "supported cases; they do not establish arbitrary batch support. "
        "Observed-versus-predicted cache reuse remains a separate limit. "
        "Missing predictions retain their reasons and denominator in `missing-predictions.json`; they are "
        "not assigned zero error. The existing replay emits its first output after a decode iteration, "
        "which can increase modeled TTFT relative to serving. Native serving overlap is also distinct "
        "from the analytical eager graph. SOL is an idealized lower bound, not calibrated hardware latency.",
        "",
        "## Disclosed host activity",
        "",
    ]
    exposures = [e for s in first["segments"] for e in s["receipt"].get("execution_exposures", [])]
    for exposure in exposures:
        lines.append(
            f"A separate CPU-only analysis ran from {exposure['reported_start_utc']} through "
            f"{exposure['reported_end_utc']} on the serving host. At one-second timestamp resolution, "
            f"the conservative end-inclusive-second window intersects {len(exposure['affected_cohort_ids'])} "
            f"HTTP cohorts and {len(exposure['affected_dispatch_ids'])} dispatch-to-report host envelopes. "
            "These are not exact GPU-event overlap intervals or an estimate of causal slowdown. "
            "All affected observations and eligible trials remain included; none were filtered or replaced."
        )
    lines += [
        "",
        "Known exposure annotations and source hashes are retained in each segment receipt. "
        "Their presence does not establish that other interference was absent.",
    ]
    if paired_outputs:
        counts = paired_outputs["requests_by_status"]
        lines += [
            "",
            "## Prespecified paired output check",
            "",
            "The first 40 OFF/ON trials have exactly paired inputs and requested output lengths "
            "(680 cohorts / 960 requests). Actual returned token IDs are checked independently "
            "against the original HTTP frames: "
            f"{counts.get('equal', 0)} request pairs have equal output sequences, "
            f"{counts.get('different', 0)} differ, and "
            f"{counts.get('unavailable', 0)} lack comparable output IDs. "
            "This describes returned sequences; it does not establish hidden-state numerical equivalence "
            "or task quality. No timing observations are removed or corrected based on these outcomes. "
            "`paired-output-ids.json` retains per-request sequence hashes, status and source bindings.",
        ]
    if content_controls:
        repeated = content_controls["summary"]["engram-repeated-text"]
        distinct = content_controls["summary"]["engram-distinct-text"]
        lines += [
            "",
            "## Content-control limits",
            "",
            f"Both two-request content controls retain {repeated['observed_trials']} trials. Native observations "
            f"place both requests in the same first-prefill dispatch in {repeated['same_first_prefill_dispatch']} "
            f"repeated-sequence trials and {distinct['same_first_prefill_dispatch']} different-offset trials. "
            "Both requests begin with zero cached tokens in every trial. Different-offset sequences are unequal, "
            "but their raw trigram sets are identical in every trial; this is not a disjoint-ngram locality test. "
            "Raw n-grams do not establish compressed Engram addresses or hardware-cache hits, and a native KV "
            "reset does not flush the Engram table or GPU caches. All timings remain included. "
            "`content-controls.json` binds the descriptive records to original inputs and both closed physical audits.",
        ]
    lines += [
        "",
        "## Reproduction",
        "",
        "`python render_report.py --directory .` validates all three result sets, recomputes descriptive tables "
        "and redraws the figures. A partial package requires `--allow-partial`. `provenance.json` records "
        "the separate SILICON model/binary checkout and FPM comparison-tool checkout; original raw evidence "
        "stays private. `artifact-hashes.json` binds every published artifact. Rendering never changes "
        "predictions, observations, calibration or frozen sample budgets.",
        "",
    ]
    return "\n".join(lines)


def plots(root, reports):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = reports["hybrid"]
    for index, segment in enumerate(first["segments"]):
        purposes = list(segment["statistics"]["complete_trial_e2e"])
        fig, axes = plt.subplots(1, 3, figsize=(15, max(6, len(purposes) * 0.45)), sharey=True)
        for ax, metric in zip(axes, list(METRICS)[:3], strict=True):
            for mode_index, mode in enumerate(MODES):
                points = reports[mode]["segments"][index]["statistics"]["complete_trial_e2e"]
                for row, purpose in enumerate(purposes):
                    value = points.get(purpose, {}).get("metrics", {}).get(metric)
                    if value is None:
                        continue
                    center = 100 * (value["predicted_mean"] / value["observed_mean"] - 1)
                    y = row + (mode_index - 1) * 0.19
                    ax.plot(center, y, "o", color=COLORS[mode], ms=3.5, label=mode.upper() if row == 0 else None)
                    interval = value.get("paired_ratio_error_percent_bootstrap_ci95")
                    if interval:
                        ax.plot(interval, [y, y], color=COLORS[mode], lw=1.1)
            ax.axvline(0, color="#777777", lw=0.6)
            ax.set_xscale("symlog", linthresh=20)
            ax.margins(x=0.1)
            for row, purpose in enumerate(purposes):
                if all(
                    metric
                    not in reports[mode]["segments"][index]["statistics"]["complete_trial_e2e"]
                    .get(purpose, {})
                    .get("metrics", {})
                    for mode in MODES
                ):
                    ax.text(
                        0.5,
                        row,
                        "unavailable",
                        transform=ax.get_yaxis_transform(),
                        color="#777777",
                        fontsize=8,
                        va="center",
                        ha="center",
                    )
            ax.set_title(METRICS[metric])
            ax.set_xlabel("Ratio-of-means error (%) / symmetric log scale")
            ax.grid(axis="x", alpha=0.2)
        axes[0].set_yticks(range(len(purposes)), purposes)
        axes[0].set_ylim(-0.7, len(purposes) - 0.3)
        axes[0].invert_yaxis()
        axes[-1].legend(loc="best")
        n = len(segment["statistics"]["complete_same_lifecycle_trial_indices"])
        fig.suptitle(f"GB300 Decoder ON · physical segment {index + 1} · {n} complete trials", fontsize=14)
        fig.text(
            0.5,
            0.015,
            "Pointwise conditional 95% whole-trial bootstrap; fixed-deadline subset. "
            "Missing coverage has no CI. No pooled CI.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.96))
        for suffix in ("png", "pdf"):
            metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
            fig.savefig(
                root / f"segment-{index + 1}-conditional-errors.{suffix}",
                dpi=180,
                bbox_inches="tight",
                metadata=metadata,
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve(strict=True)
    reports = {mode: read(root / f"comparison-{mode}.json.gz") for mode in MODES}
    plan, budget = read(root / "logical-plan.json.gz"), read(root / "main-budget.json")
    validate_reports(reports, plan, allow_partial=args.allow_partial)
    require(
        budget["stage_trials"] == plan["requested_trials"] and budget["required_trials"] > budget["stage_trials"],
        "frozen capped budget differs",
    )
    descriptive_update.validate_frozen(root)
    summary = summarize(reports)
    descriptive_update.write(root, {k: v for k, v in summary.items() if k != "missing"})
    descriptive_update.plot_serving(root, summary)
    (root / "descriptive-statistics.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "missing"}, indent=2) + "\n"
    )
    (root / "missing-predictions.json").write_text(json.dumps(summary["missing"], indent=2) + "\n")
    paired_path = root / "paired-output-ids.json"
    paired = read(paired_path) if paired_path.exists() else None
    if paired:
        validate_paired_outputs(paired, reports["hybrid"])
    content_path = root / "content-controls.json"
    content = read(content_path) if content_path.exists() else None
    if content:
        validate_content_controls(content, reports["hybrid"], plan)
    (root / "README.md").write_text(markdown(reports, plan, budget, summary, paired, content))
    with (root / "lifecycle-scenario-statistics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "mode",
                "physical_run_id",
                "scenario",
                "metric",
                "complete_trials",
                "predicted_trials",
                "observed_mean",
                "predicted_mean",
                "mape_percent",
                "wape_percent",
                "ratio_error_ci95_low",
                "ratio_error_ci95_high",
            ]
        )
        for mode, report in reports.items():
            for segment in report["segments"]:
                for purpose, values in segment["statistics"]["complete_trial_e2e"].items():
                    for metric, point in values["metrics"].items():
                        indices = set(segment["statistics"]["complete_same_lifecycle_trial_indices"])
                        selected = [
                            r
                            for r in segment["e2e_cases"]
                            if r["purpose"] == purpose
                            and r["trial_index"] in indices
                            and r["status"] == "predicted"
                            and metric in r["observed"]
                        ]
                        pair_values = descriptive(
                            [(r["observed"][metric], r["prediction"][metric]) for r in selected],
                            len(selected),
                        )
                        interval = point.get("paired_ratio_error_percent_bootstrap_ci95", [None, None])
                        writer.writerow(
                            [
                                mode,
                                segment["physical_run_id"],
                                purpose,
                                metric,
                                len(segment["statistics"]["complete_same_lifecycle_trial_indices"]),
                                values["coverage"]["predicted_trials"],
                                point["observed_mean"],
                                point["predicted_mean"],
                                pair_values.get("mape_percent"),
                                pair_values.get("wape_percent"),
                                *interval,
                            ]
                        )
    plots(root, reports)
    hashes = {p.name: checksum(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    (root / "artifact-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
