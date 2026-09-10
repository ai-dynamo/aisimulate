# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate unchanged observations and render the corrected-model reports."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from replay import DATA_PREFIX, MODEL_REF, MODES, NATIVE_SHA, TOOLS_PREFIX, TOOLS_REF, read, sha


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def frozen_report(repo, path):
    data = subprocess.check_output(["git", "show", f"{MODEL_REF}:{DATA_PREFIX}/report/{path}"], cwd=repo)
    return json.loads(gzip.decompress(data) if path.endswith("gz") else data)


def observed_view(rows, kind):
    if kind == "forward":
        fields = ["case_id", "phase", "batch_size", "query", "prefix", "median_ms", "rank_max_ms", "observed_ms"]
        return [{k: row[k] for k in fields} for row in rows]
    result = []
    for row in rows:
        item = {k: row[k] for k in ("cohort_id", "purpose", "trial_index", "trial_seed")}
        if "physical_run_id" in row:
            item["physical_run_id"] = row["physical_run_id"]
        if kind == "http":
            item["observed"] = row["observed"]
            item["native_initial_cached_tokens"] = row.get("native_initial_cached_tokens")
        else:
            fields = [
                "counter_id",
                "dispatch_id",
                "phase",
                "observed_ms",
                "native_scheduled_requests",
                "scheduled_requests",
                "native_work_role",
            ]
            item["intervals"] = [{k: interval.get(k) for k in fields} for interval in row["intervals"]]
        result.append(item)
    return result


def check_observations(new, old, kind):
    actual, original = observed_view(new, kind), observed_view(old, kind)
    if actual != original:
        raise ValueError("original observed values, identities, or geometry changed: " + kind)
    return {
        "kind": kind,
        "records": len(actual),
        "unchanged": True,
        "canonical_observations_sha256": hashlib.sha256(canonical(actual).encode()).hexdigest(),
    }


def main():
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=source)
    root = parser.parse_args().directory.resolve()
    # Locate the repository independently of where this report is rendered.
    repo = next(p for p in source.parents if (p / "Cargo.toml").is_file())
    with tempfile.TemporaryDirectory(prefix="dsv41-metrics-") as tmp:
        tools = Path(tmp)
        for name in ("analyze_e2e", "compare_forward", "normalize_fpm"):
            data = subprocess.check_output(["git", "show", f"{TOOLS_REF}:{TOOLS_PREFIX}/{name}.py"], cwd=repo)
            (tools / f"{name}.py").write_bytes(data)
        sys.path.insert(0, str(tools))
        error_summary = importlib.import_module("compare_forward").error_summary
        summaries, preservation, plots, details = [], [], {}, {}

        def summarize(scope, mode, metric, rows, *, weighting):
            summary = error_summary(rows)
            summary["weighting"] = weighting
            paired = [r for r in rows if r["status"] == "predicted"]
            summary.update(
                scope=scope,
                mode=mode.upper(),
                metric=metric,
                observed_mean_supported=statistics.mean(r["observed_ms"] for r in paired) if paired else None,
                predicted_mean_supported=statistics.mean(r["predicted_ms"] for r in paired) if paired else None,
            )
            summaries.append(summary)
            return summary

        def serving(scope, mode, http, trace):
            for metric in (
                "ttft_ms",
                "average_tpot_ms",
                "exact_itl_ms",
                "output_tokens_per_second",
                "request_latency_ms",
                "last_token_latency_ms",
            ):
                rows = []
                for cohort in http:
                    if metric not in cohort["observed"]:
                        continue
                    row = {"observed_ms": cohort["observed"][metric], "status": "prediction_unavailable"}
                    if cohort["status"] == "predicted" and metric in cohort["prediction"]:
                        row.update(
                            status="predicted",
                            predicted_ms=cohort["prediction"][metric],
                            signed_error_percent=cohort["signed_error_percent"][metric],
                        )
                    rows.append(row)
                summarize(
                    scope,
                    mode,
                    metric,
                    rows,
                    weighting=(
                        "MAPE: equal supported HTTP cohort/trial pairs; "
                        "WAPE: observed-value weighted within this metric"
                    ),
                )
            intervals = [r for c in trace for r in c["intervals"]]
            summarize(
                scope,
                mode,
                "native_forward_interval_ms",
                intervals,
                weighting="MAPE: equal supported correlated native intervals; WAPE: observed interval latency weighted",
            )
            details[f"{scope}-{mode}"] = {
                "observed_http_cohorts": len(http),
                "observed_native_intervals": len(intervals),
                "missing_http_predictions": sum(c["status"] != "predicted" for c in http),
                "missing_native_predictions": sum(r["status"] != "predicted" for r in intervals),
                "cache_semantics_disagreements": sum(c.get("cache_semantics_match") is False for c in http),
            }

        for scope in ("forward", "off", "on"):
            provenance = read(root / scope / "replay-provenance.json")
            if provenance["native_sha256"] != NATIVE_SHA or provenance["analysis_commit"] != TOOLS_REF:
                raise ValueError("prediction source pin mismatch")
            if provenance["replay_program_sha256"] != sha(source / "replay.py"):
                raise ValueError("replay source changed after generation")
            for job in provenance["jobs"]:
                path = root / scope / job["output"]
                if sha(path) != job["output_sha256"]:
                    raise ValueError("comparison output changed after replay")
        for profile in ("full", "decoder_bounded"):
            for mode in MODES:
                name = f"{profile}-{mode}-results.json"
                new = read(root / "forward" / name)
                old = frozen_report(repo, "prefix-refinement-v1/" + name)
                if (
                    new["observation_sha256"] != old["observation_sha256"]
                    or new["heldout_plan_sha256"] != old["heldout_plan_sha256"]
                ):
                    raise ValueError("forward observation/plan binding changed")
                preservation.append(
                    check_observations(new["cases"], old["cases"], "forward") | {"scope": profile, "mode": mode}
                )
                summary = summarize(
                    f"forward-{profile}", mode, "forward_ms", new["cases"], weighting=new["summary"]["weighting"]
                )
                if any(summary[k] != new["summary"][k] for k in ("mape_percent", "wape_percent")):
                    raise ValueError("forward metric recomputation differs")
                plots[(profile, mode)] = new["cases"]
        for mode in MODES:
            http = read(root / "off" / f"e2e-{mode}-results.json.gz")
            trace = read(root / "off" / f"trace-{mode}-results.json.gz")
            old_http = frozen_report(repo, f"serving-v4/off-prefix-refined/e2e-{mode}.json.gz")
            old_trace = frozen_report(repo, f"serving-v4/off-prefix-refined/trace-{mode}.json.gz")
            preservation.append(
                check_observations(http["cohorts"], old_http["cohorts"], "http") | {"scope": "off", "mode": mode}
            )
            preservation.append(
                check_observations(trace["cohorts"], old_trace["cohorts"], "trace") | {"scope": "off", "mode": mode}
            )
            serving("serving-off", mode, http["cohorts"], trace["cohorts"])
            on = read(root / "on" / f"{mode}-results.json.gz")
            old_on = frozen_report(repo, f"serving-v4/on-segmented/comparison-{mode}.json.gz")
            if on["summary"]["pooled_confidence_intervals"] is not None or not on["summary"]["coverage_complete"]:
                raise ValueError("segmented coverage/uncertainty contract changed")
            on_http, on_trace = [], []
            for segment, original in zip(on["segments"], old_on["segments"], strict=True):
                for key, kind in (("e2e_cases", "http"), ("trace_cases", "trace")):
                    preservation.append(
                        check_observations(segment[key], original[key], kind)
                        | {"scope": "on", "mode": mode, "physical_run_id": segment["physical_run_id"]}
                    )
                on_http.extend(segment["e2e_cases"])
                on_trace.extend(segment["trace_cases"])
            serving("serving-on", mode, on_http, on_trace)
        summary_payload = {
            "schema": "dsv41.corrected-sol-verification.v2",
            "correction_fitting": False,
            "new_measurements": False,
            "model_commit": MODEL_REF,
            "analysis_commit": TOOLS_REF,
            "native_sha256": NATIVE_SHA,
            "historical_observation_reference_commit": MODEL_REF,
            "observation_preservation": preservation,
            "coverage_and_cache": details,
            "metrics": summaries,
            "metric_definition": {
                "mape_percent": "100 * mean(abs(predicted-observed)/observed)",
                "wape_percent": "100 * sum(abs(predicted-observed))/sum(observed)",
            },
            "uncertainty": (
                "Descriptive aggregate MAPE/WAPE; paired scenario confidence intervals remain in exact"
                " comparison outputs. No cross-lifecycle pooled confidence interval."
            ),
        }
        (root / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
        fields = [
            "scope",
            "mode",
            "metric",
            "planned_points",
            "predicted_points",
            "observed_mean_supported",
            "predicted_mean_supported",
            "mape_percent",
            "wape_percent",
            "mean_signed_error_percent",
            "weighting",
        ]
        with (root / "comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(summaries)
        draw(root, plots, summaries)
        write_readme(root, summaries, details)
        inventory = {
            str(p.relative_to(root)): sha(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and p.name not in ("artifact-hashes.json", "replay_field.py")
            and "__pycache__" not in p.parts
            and "field-off" not in p.relative_to(root).parts
        }
        (root / "artifact-hashes.json").write_text(json.dumps({"files_sha256": inventory}, indent=2) + "\n")


def draw(root, plots, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, profile in enumerate(("full", "decoder_bounded")):
        for j, mode in enumerate(MODES):
            ax, rows = axes[i, j], plots[(profile, mode)]
            for phase, color in (("context", "#2374ab"), ("generation", "#d96c06")):
                group = [r for r in rows if r["phase"] == phase and r["status"] == "predicted"]
                ax.scatter(
                    [r["observed_ms"] for r in group],
                    [r["predicted_ms"] for r in group],
                    s=16,
                    alpha=0.8,
                    color=color,
                    label="Prefill" if phase == "context" else "Decode",
                )
            vals = [r[k] for r in rows if r["status"] == "predicted" for k in ("observed_ms", "predicted_ms")]
            ax.plot([min(vals) * 0.8, max(vals) * 1.2], [min(vals) * 0.8, max(vals) * 1.2], color="gray", ls="--", lw=1)
            ax.set(
                xscale="log" if mode == "sol" else "linear",
                yscale="log" if mode == "sol" else "linear",
                xlabel="Observed forward (ms)",
                ylabel="Predicted forward (ms)",
                title=f"{'OFF' if profile == 'full' else 'ON'} / {mode.upper()} / 46 points",
            )
            if mode != "sol":
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("GB300 TP4 · corrected model · unchanged independent forward observations")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"forward-comparison.{suffix}", dpi=180)
    plt.close(fig)
    metrics = ["native_forward_interval_ms", "ttft_ms", "average_tpot_ms", "output_tokens_per_second"]
    labels = ["Native interval", "HTTP TTFT", "HTTP mean TPOT", "HTTP output throughput"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for i, scope in enumerate(("serving-off", "serving-on")):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for offset, field, label, color in [
                (-0.18, "mape_percent", "MAPE", "#2374ab"),
                (0.18, "wape_percent", "WAPE", "#d96c06"),
            ]:
                values = [
                    next(
                        s for s in summaries if s["scope"] == scope and s["mode"] == m.upper() and s["metric"] == metric
                    )[field]
                    for m in MODES
                ]
                bars = ax.bar([x + offset for x in range(3)], values, width=0.34, color=color, label=label)
                ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=7, padding=2, rotation=90)
            ax.set_xticks(range(3), [m.upper() for m in MODES], fontsize=8)
            ax.set(
                yscale="log",
                ylabel="Absolute error (%) · log scale",
                title=f"{'OFF' if i == 0 else 'ON'} · {labels[j]}",
            )
            ax.margins(y=0.4)
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Serving validation · supported pairs only · missing predictions remain in coverage counts")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"serving-comparison.{suffix}", dpi=180)
    plt.close(fig)


def write_readme(root, summaries, details):
    lines = [
        "# GB300 TP4 verification after SOL review fixes",
        "",
        (
            "This report recomputes SOL, Hybrid, and SILICON predictions using the corrected V4.1 "
            "model and the versioned replicated-indexer tables. It preserves every original observ"
            "ed value and workload identity. No new measurements or fitted correction factors ente"
            "r these results."
        ),
        "",
        (
            f"Model: `{MODEL_REF}`. Analysis: `{TOOLS_REF}`. Native binary SHA-256: `{NATIVE_SHA}`. "
            "The native code was built from `09954c44`, whose prediction implementation "
            "is unchanged at the model commit. "
            "[Table derivation](../../indexer-identity-v2/README.md) corrects labels without changing latency bits."
        ),
        "",
        "## Independent forward holdouts",
        "",
        (
            "The original frozen refinement holds out 46 configurations per profile (36 prefill, 1"
            "0 decode), each with ten measured repetitions and four TP ranks. A point is the media"
            "n of rank-max repetitions, not ten or forty independent workloads. The same 92 config"
            "uration/profile pairs are evaluated in each prediction mode."
        ),
        "",
        "| Replay | Mode | Covered | Real mean (ms) | Predicted mean (ms) | MAPE | WAPE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        if not s["scope"].startswith("forward-"):
            continue
        profile = "OFF" if s["scope"] == "forward-full" else "ON"
        lines.append(
            f"| {profile} | {s['mode']} | {s['predicted_points']}/{s['planned_points']} | "
            f"{s['observed_mean_supported']:.3f} | {s['predicted_mean_supported']:.3f} | "
            f"{s['mape_percent']:.2f}% | {s['wape_percent']:.2f}% |"
        )
    lines += [
        "",
        "![Forward predictions versus real silicon](forward-comparison.png)",
        "",
        "## Native serving and HTTP validation",
        "",
        (
            "OFF retains 40 independent trials in one closed lifecycle: 600 primary HTTP cohorts a"
            "nd 15,756 native intervals. ON retains 100 logical trials across two independently cl"
            "osed physical lifecycles: 1,500 primary HTTP cohorts. Per-lifecycle complete-trial su"
            "bsets retain their conditional confidence intervals in the compressed outputs; bounda"
            "ry trials remain in descriptive statistics. There is no pooled cross-lifecycle confid"
            "ence interval or claim that the original precision target was met."
        ),
        "",
        (
            "Means below use the same supported pairs as each error statistic. MAPE weights those "
            "pairs equally; WAPE weights by their observed value in the metric's own units. Aggreg"
            "ate HTTP pairs are cohort/trial summaries; native intervals are correlated and do not"
            " count as independent trials. Missing results are excluded from error arithmetic and "
            "retained in the coverage denominator. These descriptive aggregates are not a producti"
            "on traffic mixture."
        ),
        "",
        "| Replay | Metric | Mode | Covered | Real mean | Predicted mean | MAPE | WAPE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    metric_labels = {
        "native_forward_interval_ms": "Native interval (ms)",
        "ttft_ms": "HTTP TTFT (ms)",
        "average_tpot_ms": "HTTP mean TPOT (ms)",
        "output_tokens_per_second": "HTTP output tokens/s",
    }
    for s in summaries:
        if s["scope"].startswith("serving-") and s["metric"] in metric_labels:
            lines.append(
                f"| {s['scope'].removeprefix('serving-').upper()} | {metric_labels[s['metric']]} | {s['mode']} | "
                f"{s['predicted_points']}/{s['planned_points']} | {s['observed_mean_supported']:.3f} | "
                f"{s['predicted_mean_supported']:.3f} | {s['mape_percent']:.2f}% | {s['wape_percent']:.2f}% |"
            )
    lines += [
        "",
        "![Serving errors with MAPE and WAPE](serving-comparison.png)",
        "",
        "## Coverage and interpretation",
        "",
    ]
    for label, item in details.items():
        lines.append(
            f"- {label}: {item['missing_http_predictions']} missing HTTP predictions; "
            f"{item['missing_native_predictions']} missing native intervals "
            f"out of {item['observed_native_intervals']}; "
            f"{item['cache_semantics_disagreements']} predicted cohorts disagree with observed initial cache reuse."
        )
    lines += [
        "",
        (
            "The corrected SOL roofline remains an idealized lower bound and strongly underpredict"
            "s runtime latency. The measured module composition performs much better on the contro"
            "lled forward holdouts, but the serving comparison includes scheduling, overlap, prepa"
            "ration, sampling, and HTTP costs outside that target. Coverage and cache agreement re"
            "main separate from accuracy. Exact token-gap ITL, request-completion latency, last-to"
            "ken latency, per-scenario paired intervals, and failure reasons are retained in JSON/"
            "CSV; HTTP mean TPOT is not a tail-ITL measurement."
        ),
        "",
        "## Reproduction and preservation",
        "",
        (
            "[summary.json](summary.json) verifies original observations against frozen historical"
            " reports at the model commit. [comparison.csv](comparison.csv) includes both errors, "
            "measured/predicted means, and all supported-pair counts. Each scope's `replay-provena"
            "nce.json` binds unchanged qualified inputs, portable prediction configs, the full tab"
            "le inventory, helper source hashes, and output hashes. Original reports and observati"
            "ons remain at their original paths. Private cluster paths and raw journals are not pu"
            "blished."
        ),
        "",
        ("Run from the repository root with the recorded corrected native extension and its source import paths:"),
        "",
        "```bash",
        "export PYTHONPATH=python/aisimulate/src:python/aisimulate",
        (
            "python data/experimental/deepseek-v41/gb300-silicon/report/sol-review-v2/replay.py fo"
            "rward --output-dir /path/to/fresh-report"
        ),
        (
            "python data/experimental/deepseek-v41/gb300-silicon/report/sol-review-v2/replay.py of"
            "f --output-dir /path/to/fresh-report --off-qualified /path/to/qualified-off --off-raw"
            " /path/to/original-off"
        ),
        (
            "python data/experimental/deepseek-v41/gb300-silicon/report/sol-review-v2/replay.py on"
            " --output-dir /path/to/fresh-report --on-qualified /path/to/qualified-segments"
        ),
        (
            "python data/experimental/deepseek-v41/gb300-silicon/report/sol-review-v2/render.py "
            "--directory /path/to/fresh-report"
        ),
        "```",
        "",
        (
            "Replay refuses to overwrite an existing report scope. Rendering validates prediction "
            "bytes and unchanged observation values before producing tables and standalone PNG/PDF"
            " figures. Rendering performs no prediction fit and does not alter compressed comparis"
            "ons. The final render command writes the explicitly selected fresh report directory."
        ),
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
