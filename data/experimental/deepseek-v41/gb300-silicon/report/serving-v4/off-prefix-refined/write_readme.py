# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write descriptive summaries from frozen comparisons; never fit predictions."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

from render_report import METRICS, load_reports


def quantile(values, fraction):
    values = sorted(values)
    x = (len(values) - 1) * fraction
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


def summary(pairs):
    if not pairs:
        return "— | — | — | —"
    errors = [100 * (p / o - 1) for o, p in pairs]
    absolute = [abs(e) for e in errors]
    wape = 100 * sum(abs(p - o) for o, p in pairs) / sum(o for o, _ in pairs)
    return (
        f"{statistics.mean(errors):+.2f}% | {statistics.median(absolute):.2f}% | "
        f"{quantile(absolute, 0.9):.2f}% | {wape:.2f}%"
    )


def interval(value):
    return "—" if value is None else f"[{value[0]:+.2f}%, {value[1]:+.2f}%]"


def failure_category(row):
    if "DeepSeek-V4.1 attention has no measured SILICON data" in row.get("failure", ""):
        return "missing measured attention geometry"
    return row["failure_type"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve()
    reports = load_reports(root)
    if not args.diagnostic and any(not r["complete_requested_study"] for pair in reports.values() for r in pair):
        raise ValueError("partial studies require explicit diagnostic labeling")
    first, trace_first = next(iter(reports.values()))
    identity = first["measurement_identity"]
    profile = "ON" if identity["decoder_replay"] else "OFF"
    corpus = {c["corpus_role"] for c in first["cohorts"]}
    if len(corpus) != 1:
        raise ValueError("one corpus stratum per report")
    title = f"{identity['system_name'].upper()} TP4 / Decoder {profile} / {next(iter(corpus))}"
    lines = [f"# Real silicon versus prediction: {title}", ""]
    if args.diagnostic:
        lines += [
            "**Incomplete diagnostic evidence only. No final main-stage confidence interval or "
            "completed accuracy claim.**",
            "",
        ]
    lines += [
        f"The observed runtime is `{identity['backend']} {identity['backend_version']}` in "
        f"eager text AR, with DSpark disabled. "
        "These comparisons use one physical runtime lifecycle. Each mode uses the same real "
        "HTTP requests and native intervals. "
        "Model, calibration, binary, source and raw-evidence identities are recorded in the result files. "
        "No correction factor is fitted to these observations.",
        "",
        f"The frozen main plan requests **{trace_first['requested_trials']} independent trials per scenario**. "
        f"The closed audit covers "
        f"**{trace_first['covered_main_cohorts']}/{trace_first['planned_main_cohorts']}** planned main cohorts "
        f"including setup, and **{len(first['cohorts'])}** metric-bearing cohorts. "
        "Warmup, canary, pilot and prefix preparation are separate roles. Repeated decode "
        "intervals and TP ranks do not increase the independent trial count.",
        "",
        "The adjacent `main-budget.json` and original pilot/main statistics retain the sample-size decision. "
        "The rule takes the largest required N across TTFT, mean time per token, and throughput: "
        "at least 20, rounded up to a multiple of 10 from `(1.96 * pilot_CV / 0.05)^2`, capped at 100. "
        "Each corpus/profile uses independent pilot and main seeds. Achieved uncertainty is reported below; "
        "the budget is an estimate, not a precision guarantee.",
        "",
        "This is a separately versioned prediction against the same measured OFF lifecycle. "
        "[The original baseline](../off-baseline/README.md) is preserved. The refined OFF overlay adds two "
        "previously measured pilot attention cells at prefix 256 to 126 original formal configurations. "
        "Original formal rows win duplicate keys. No HTTP or verification-interval timing is fitted into tables. "
        "`refinement-comparison.json` records which predictions changed; the real observations and native "
        "model binary are identical. New ON calibration and 92 fresh profile/configuration holdouts belong "
        "to the separate [refinement forward report](../../prefix-refinement-v1/README.md).",
        "",
        "## HTTP serving errors",
        "",
        "Rows below are descriptive across scenario/trial observations, equally weighted per "
        "cohort. They do not represent a production traffic mixture. "
        "Mean signed error is `(prediction / observation - 1) * 100`; WAPE is total absolute "
        "error divided by total observed value. "
        "p90 APE is a percentile of prediction errors, not p90 request latency. "
        "Time per output token is the HTTP request-level mean; coalesced frames do not "
        "establish exact individual token gaps or tail ITL.",
        "",
        "| Mode | Metric | Predicted / observed cohorts | Mean signed error | Median APE | p90 APE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    supported = [
        set(c["cohort_id"] for c in pair[0]["cohorts"] if c["status"] == "predicted") for pair in reports.values()
    ]
    common = set.intersection(*supported)
    for mode, (e2e, _) in reports.items():
        for metric, (label, _) in METRICS.items():
            observed = [c for c in e2e["cohorts"] if metric in c["observed"]]
            pairs = [(c["observed"][metric], c["prediction"][metric]) for c in observed if c["status"] == "predicted"]
            lines.append(f"| {mode.upper()} | {label} | {len(pairs)}/{len(observed)} | {summary(pairs)} |")
    lines += [
        "",
        f"### Same supported subset: {len(common)} cohorts",
        "",
        "| Mode | Metric | Mean signed error | Median APE | p90 APE | WAPE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode, (e2e, _) in reports.items():
        for metric, (label, _) in METRICS.items():
            pairs = [
                (c["observed"][metric], c["prediction"][metric])
                for c in e2e["cohorts"]
                if c["cohort_id"] in common and metric in c["observed"]
            ]
            lines.append(f"| {mode.upper()} | {label} | {summary(pairs)} |")
    lines += [
        "",
        "![HTTP means and paired errors](e2e-comparison.png)",
        "",
        "![Request completion and mean ITL](e2e-completion-comparison.png)",
        "",
        "Response completion and last-token latency are supplementary metrics added after the sampling plan; "
        "they do not change its pilot-derived N. Exact ITL is a request mean, not a token-tail statistic.",
        "",
        "[Scenario means and coverage (CSV)](scenario-means.csv) includes real values, predictions, "
        "paired error intervals and unmatched observations. The original real-token request plans are included "
        "as `main-plan.json.gz` and `pilot-plan.json.gz`.",
        "",
        "## Scenario confidence and coverage",
        "",
        "Intervals resample whole independent trials within each scenario. They are "
        "conditional on this runtime lifecycle and frozen calibration. "
        "Missing predictions suppress that scenario's final interval. The interval targets "
        "the ratio of predicted and observed means. "
        "Legacy scenario names containing `heldout` are identifiers; E2E semantic cases are "
        "not all geometrically disjoint from calibration.",
        "",
        "| Mode | Scenario | Predicted / observed trials | "
        + " | ".join(label + " error CI95" for label, _ in METRICS.values())
        + " |",
        "|---|---|---:|" + "---:|" * len(METRICS),
    ]
    for mode, (e2e, _) in reports.items():
        for purpose, point in e2e["points"].items():
            cover = point["coverage"]
            intervals = [
                interval(point["metrics"].get(k, {}).get("paired_ratio_error_percent_bootstrap_ci95")) for k in METRICS
            ]
            lines.append(
                f"| {mode.upper()} | {purpose} | {cover['predicted_trials']}/{cover['planned_trials']} "
                f"| {' | '.join(intervals)} |"
            )
    precision = [
        (purpose, metric, point["relative_half_width"])
        for purpose, metrics in first["client_analysis"]["points"].items()
        for metric, point in metrics.items()
        if metric in ("ttft_ms", "average_tpot_ms", "output_tokens_per_second") and "relative_half_width" in point
    ]
    if precision:
        misses = [(p, m, w) for p, m, w in precision if w > 0.05]
        lines += [
            "",
            f"Observed-mean precision target: {len(precision) - len(misses)}/{len(precision)} "
            f"required scenario/metric pairs "
            "achieve a 95% interval relative half-width of at most 5%. Main sample size stays frozen after the pilot.",
        ]
        lines += [f"- Wider interval: `{p}` / `{m}`: {w:.2%} relative half-width." for p, m, w in misses]
    lines += [
        "",
        "## Native forward intervals",
        "",
        f"Observed timing target: `{trace_first['observed_target']}`. "
        "This differs from HTTP E2E and from the synchronized prepare/forward/sample component-study holdouts. "
        "All attributed native work, including unreturned overlap output, is retained. "
        "Interval statistics below are descriptive because consecutive intervals are correlated.",
        "",
        "| Mode | Phase | Predicted / observed intervals | Mean signed error | Median APE | p90 APE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode, (_, trace) in reports.items():
        rows = [r for c in trace["cohorts"] for r in c["intervals"]]
        for phase in ["all", "prefill", "decode", "mixed"]:
            selected = [r for r in rows if phase == "all" or r["phase"] == phase]
            if selected:
                pairs = [(r["observed_ms"], r["predicted_ms"]) for r in selected if r["status"] == "predicted"]
                lines.append(f"| {mode.upper()} | {phase} | {len(pairs)}/{len(selected)} | {summary(pairs)} |")
    lines += [
        "",
        "![Native interval and whole-trial errors](forward-trace-comparison.png)",
        "",
        "## Cache behavior, missing predictions and limitations",
        "",
        "| Mode | Predicted requests with cache witnesses | Initial cache-reuse mismatches | "
        "Missing E2E cohorts | Missing native intervals |",
        "|---|---:|---:|---:|---:|",
    ]
    failure_rows = []
    for mode, (e2e, trace) in reports.items():
        requests = [r for c in e2e["cohorts"] for r in c.get("requests", []) if "cache_reuse_matches_native" in r]
        missing_e2e = [c for c in e2e["cohorts"] if c["status"] != "predicted"]
        missing_trace = [r for c in trace["cohorts"] for r in c["intervals"] if r["status"] != "predicted"]
        lines.append(
            f"| {mode.upper()} | {len(requests)} | "
            f"{sum(not r['cache_reuse_matches_native'] for r in requests)} | "
            f"{len(missing_e2e)} | {len(missing_trace)} |"
        )
        if missing_e2e:
            counts = Counter((c["purpose"], failure_category(c)) for c in missing_e2e)
            failure_rows += [f"| {mode.upper()} | {p} | {kind} | {n} |" for (p, kind), n in sorted(counts.items())]
    if failure_rows:
        lines += [
            "",
            "| Mode | Missing E2E scenario | Failure category | Cohorts |",
            "|---|---|---|---:|",
            *failure_rows,
        ]
    common_intervals = set.intersection(
        *[
            {
                (c["cohort_id"], r["dispatch_id"])
                for c in trace["cohorts"]
                for r in c["intervals"]
                if r["status"] == "predicted"
            }
            for _, trace in reports.values()
        ]
    )
    lines += [
        "",
        f"### Same native supported subset: {len(common_intervals)} intervals",
        "",
        "| Mode | Mean signed error | Median APE | p90 APE | WAPE |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, (_, trace) in reports.items():
        pairs = [
            (r["observed_ms"], r["predicted_ms"])
            for c in trace["cohorts"]
            for r in c["intervals"]
            if (c["cohort_id"], r["dispatch_id"]) in common_intervals
        ]
        lines.append(f"| {mode.upper()} | {summary(pairs)} |")
    roles = Counter(c["declared_coverage_role"] for c in first["cohorts"])
    lines += [
        "",
        f"Frozen planned coverage roles: `{dict(roles)}`. A coverage candidate is not proof of "
        f"interpolation for every actual geometry. "
        "Lookup coverage and cache-semantics agreement are separate findings. Cache "
        "disagreements remain in timing error statistics; missing predictions remain in the coverage denominator. "
        "Error statistics use supported pairs.",
        "",
    ]
    lines += [f"- {limitation}" for limitation in first["limitations"]]
    lines += [
        "- SOL is an analytical lower bound; its underprediction is reported without an empirical correction.",
        "- The original short corpus uses repeated offsets: distinct token sequences do not imply distinct "
        "Engram ngram working sets. Dedicated content strata are reported separately; no "
        "physical cache-locality claim is made.",
        "- Complete data collection does not imply accurate predictions or qualify every operation as measured.",
        "- Internal allocation details, original runtime logs and request-mapping evidence "
        "are retained separately from the public report.",
        "",
        "## Reproduction",
        "",
        "The adjacent compressed comparison results retain per-cohort observations/predictions, per-interval geometry, "
        "missing results and source/identity hashes. Decompression preserves the original JSON bytes. "
        "The comparison adapters replay actual request/token inputs and call the independent native timing model; "
        "the render scripts consume frozen outputs and perform no model fit. "
        "`plot-input-hashes.json` binds plotted inputs. "
        "PNG and standalone PDF figures are included.",
        "",
        "`closure-receipt.json` records the checkpoint, immutable image, installed packages, actual scheduler "
        "configuration, runtime strategy, counts and source revisions. `artifact-provenance.json` records "
        "original/decompressed bytes and the portable configs, whose only change is making `systems_path` "
        "relative to the repository root. `measurement.json` binds the privately retained full native audit.",
        "",
        "The source adapters used for this report are pinned in `closure-receipt.json` under "
        "`source.fpm_tools_commit`: `data/experimental/deepseek-v41/verification-plan/compare_e2e.py` and "
        "`compare_trace.py` in that revision. Full prediction replay also requires the bound native audit "
        "and client/scheduler receipts; those private runtime artifacts are not replaced by this public summary. "
        "Plot/table reproduction needs only Python 3.13, Matplotlib, and the adjacent compressed results:",
        "",
        "```sh",
        "python write_readme.py --directory .",
        "python render_report.py --directory .",
        "```",
        "",
        "Run those commands from this report directory. The machine-readable scenario means retain all "
        "observed trials even when a mode has no supported prediction. The original model inputs and timing "
        "observations remain unchanged by the rendering steps.",
        "",
    ]
    fields = [
        "mode",
        "scenario",
        "metric",
        "unit",
        "observed_trials",
        "predicted_trials",
        "all_observed_mean",
        "matched_observed_mean",
        "predicted_mean",
        "ratio_error_percent",
        "ratio_error_ci95_low",
        "ratio_error_ci95_high",
    ]
    with (root / "scenario-means.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for mode, (e2e, _) in reports.items():
            for purpose, point in e2e["points"].items():
                for metric, (_, unit) in METRICS.items():
                    observed = [
                        c["observed"][metric]
                        for c in e2e["cohorts"]
                        if c["purpose"] == purpose and metric in c["observed"]
                    ]
                    stats = point["metrics"].get(metric, {})
                    ci = stats.get("paired_ratio_error_percent_bootstrap_ci95", [None, None])
                    writer.writerow(
                        {
                            "mode": mode,
                            "scenario": purpose,
                            "metric": metric,
                            "unit": unit,
                            "observed_trials": len(observed),
                            "predicted_trials": stats.get("independent_trials", 0),
                            "all_observed_mean": statistics.mean(observed) if observed else None,
                            "matched_observed_mean": stats.get("observed_mean"),
                            "predicted_mean": stats.get("predicted_mean"),
                            "ratio_error_percent": 100 * (stats["predicted_mean"] / stats["observed_mean"] - 1)
                            if stats
                            else None,
                            "ratio_error_ci95_low": ci[0],
                            "ratio_error_ci95_high": ci[1],
                        }
                    )
    (root / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
