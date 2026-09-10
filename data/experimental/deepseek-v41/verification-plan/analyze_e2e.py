# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze independent HTTP trials; never treat token gaps as independent trials.

Input is the verification harness's immutable plan plus client progress JSON.
This checks client coverage, not FPM transport or producer qualification. Only
closed, separately audited runs may be used in a final accuracy report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def trial_metrics(cohort):
    requests = cohort["requests"]
    if not requests or cohort.get("valid") is not True or any(r.get("valid") is not True for r in requests):
        return {}
    metrics = {"output_tokens_per_second": cohort.get("output_tokens_per_second")}
    # Equal weight per request within a trial, then equal weight per trial.
    # A longer response must not turn its correlated gaps into extra trials.
    for field in ("ttft_ms", "average_tpot_ms"):
        values = [r.get(field) for r in requests]
        if all(finite_positive(v) for v in values):
            metrics[field] = statistics.mean(values)
    if all(r.get("exact_itl_available") is True and r.get("itl_ms") for r in requests):
        gaps = [r["itl_ms"] for r in requests]
        if all(finite_positive(v) for row in gaps for v in row):
            metrics["exact_itl_ms"] = statistics.mean(statistics.mean(row) for row in gaps)
    return {key: value for key, value in metrics.items() if finite_positive(value)}


def summarize(values, *, main, seed=92031519, resamples=5000):
    mean = statistics.mean(values)
    cv = statistics.stdev(values) / mean if len(values) > 1 else None
    result = {
        "independent_trials": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
        "cv": cv,
    }
    if not main:
        # NIST normal approximation is a budget estimate; the fixed main-stage
        # bootstrap interval reports the precision actually achieved.
        required = max(20, math.ceil(((1.96 * cv / 0.05) ** 2) / 10) * 10) if cv is not None else None
        result.update(required_main_trials=required, campaign_stage_trials=min(required, 100) if required else None)
    elif len(values) >= 20:
        rng = random.Random(seed)
        means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(resamples)]
        interval = [quantile(means, 0.025), quantile(means, 0.975)]
        result.update(
            mean_bootstrap_ci95=interval,
            relative_half_width=(interval[1] - interval[0]) / (2 * mean),
            bootstrap_resamples=resamples,
        )
    return result


def analyze(plan, progress):
    role = plan["sampling_role"]
    if role not in {"pilot", "main"}:
        raise ValueError("sampling_role must identify pilot or main")
    expected = {c["cohort_id"]: c for c in plan["cohorts"]}
    if len(expected) != len(plan["cohorts"]):
        raise ValueError("duplicate cohort IDs in plan")
    seen = set()
    values = defaultdict(lambda: defaultdict(list))
    failures = []
    invalid_cohorts = []
    primary = {c["purpose"] for c in expected.values() if c.get("comparison_role") == "primary"}
    if not primary:
        raise ValueError("plan has no primary scenarios")
    planned_trials = defaultdict(set)
    planned_seeds = defaultdict(set)
    planned_requests = set()
    for cohort in expected.values():
        for request in cohort["requests"]:
            request_id = request["request_id"]
            if request_id in planned_requests:
                raise ValueError("duplicate planned request ID")
            planned_requests.add(request_id)
        if cohort.get("comparison_role") == "primary":
            purpose = cohort["purpose"]
            trial, seed = cohort["trial_index"], cohort["trial_seed"]
            if trial in planned_trials[purpose] or seed in planned_seeds[purpose]:
                raise ValueError("duplicate independent trial or seed within a scenario")
            planned_trials[purpose].add(trial)
            planned_seeds[purpose].add(seed)
    if any(len(trials) != plan["requested_trials"] for trials in planned_trials.values()):
        raise ValueError("planned scenario count differs from requested trials")
    completed = defaultdict(int)
    for row in progress:
        key = row["cohort_id"]
        if key not in expected or key in seen:
            raise ValueError(f"unplanned or duplicate cohort: {key}")
        seen.add(key)
        target = expected[key]
        if any(row.get(k) != target.get(k) for k in ("purpose", "trial_index", "trial_seed")):
            raise ValueError(f"cohort trial identity mismatch: {key}")
        request_ids = [r.get("request_id") for r in row["requests"]]
        expected_ids = {r["request_id"] for r in target["requests"]}
        if len(request_ids) != len(set(request_ids)) or set(request_ids) != expected_ids:
            raise ValueError(f"cohort request identity mismatch: {key}")
        if row.get("valid") is not True or any(r.get("valid") is not True for r in row["requests"]):
            invalid_cohorts.append(key)
        if target.get("comparison_role") != "primary":
            continue
        metrics = trial_metrics(row)
        if not metrics:
            failures.append({"cohort_id": key, "reason": "client_invalid_or_missing_metrics"})
            continue
        completed[row["purpose"]] += 1
        for metric, value in metrics.items():
            values[row["purpose"]][metric].append(value)
    result = {
        "schema": "dsv41.e2e.trial-summary.v1",
        "run_id": plan["run_id"],
        "sampling_role": role,
        "requested_trials": plan["requested_trials"],
        "primary_scenarios": len(primary),
        "planned_cohorts": len(expected),
        "observed_cohorts": len(seen),
        "missing_cohorts": sorted(set(expected) - seen),
        "invalid_client_cohorts": invalid_cohorts,
        "failed_primary_cohorts": failures,
        "client_coverage_complete": len(seen) == len(expected)
        and not invalid_cohorts
        and not failures
        and all(completed[p] == plan["requested_trials"] for p in primary),
        "qualification": "client statistics only; closed-run and FPM audits are separate",
        "points": {
            purpose: {metric: summarize(v, main=role == "main") for metric, v in sorted(metrics.items())}
            for purpose, metrics in sorted(values.items())
        },
    }
    complete_metrics = all(
        len(values[p][m]) >= plan["requested_trials"]
        for p in primary
        for m in ("ttft_ms", "average_tpot_ms", "output_tokens_per_second")
    )
    if role == "pilot" and result["client_coverage_complete"] and complete_metrics and plan["requested_trials"] >= 10:
        required = max(
            result["points"][p][m]["required_main_trials"]
            for p in primary
            for m in ("ttft_ms", "average_tpot_ms", "output_tokens_per_second")
        )
        result["main_stage_budget"] = {
            "required_trials": required,
            "stage_trials": min(required, 100),
            "precision_target_exceeds_stage_cap": required > 100,
        }
    else:
        result["main_stage_budget"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.plan.read_bytes()), json.loads(args.progress.read_bytes()))
    result["inputs"] = {
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "progress_sha256": hashlib.sha256(args.progress.read_bytes()).hexdigest(),
    }
    # Do not overwrite an earlier report when input artifacts change.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
