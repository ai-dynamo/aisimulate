# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the exact public paired records and stored intervals; no predictor calls."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from export_public import encoded, metric_rows, privacy_check, require, sha


def read(path):
    blob = path.read_bytes()
    return json.loads(gzip.decompress(blob) if path.suffix == ".gz" else blob)


def interval_rows(e2e, trace, summary):
    rows = []

    def add(source, purpose, metric, estimand, bounds, n, seed, resamples):
        rows.append(
            dict(
                source_analysis=source,
                purpose=purpose,
                metric=metric,
                estimand=estimand,
                independent_trials=n,
                low=None if bounds is None else bounds[0],
                high=None if bounds is None else bounds[1],
                bootstrap_seed=seed,
                bootstrap_resamples=resamples,
                interval_available=bounds is not None,
            )
        )

    for purpose, point in sorted(e2e["points"].items()):
        for metric, value in sorted(point["metrics"].items()):
            for estimand in ("observed_mean", "paired_ratio_error_percent"):
                add(
                    "e2e_paired",
                    purpose,
                    metric,
                    estimand,
                    value.get(estimand + "_bootstrap_ci95"),
                    value["independent_trials"],
                    value.get("bootstrap_seed"),
                    value.get("bootstrap_resamples"),
                )
    for purpose, value in sorted(trace["independent_trial_summary"].items()):
        for estimand in (
            "mean_trial_total_forward_signed_error_percent",
            "interval_mape_percent",
            "interval_wape_percent",
        ):
            add(
                "native_whole_trial",
                purpose,
                "native_forward_interval_ms",
                estimand,
                value.get("whole_trial_bootstrap_ci95", {}).get(estimand),
                value["fully_predicted_trials"],
                value.get("bootstrap_seed"),
                value.get("bootstrap_resamples"),
            )
    for purpose, point in sorted(summary["observation_statistics"]["points"].items()):
        for metric, value in sorted(point.items()):
            add(
                "frozen_observed_main",
                purpose,
                metric,
                "observed_mean",
                value.get("mean_bootstrap_ci95"),
                value["independent_trials"],
                92031519,
                value.get("bootstrap_resamples"),
            )
    return rows


def write_csv(path, rows):
    with path.open("x") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = {r["metric"]: r for r in rows if r["purpose"] is None}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        (
            "ttft_ms",
            "average_tpot_ms",
            "output_tokens_per_second",
            "native_forward_interval_ms",
        ),
        (
            "HTTP TTFT",
            "HTTP mean TPOT",
            "HTTP output throughput",
            "Native forward interval",
        ),
        strict=True,
    ):
        row = selected[metric]
        for i, key, color in [
            (0, "mape_percent", "#2675a6"),
            (1, "wape_percent", "#ca772d"),
        ]:
            value = row[key]
            if value is None:
                ax.text(i, 0, "unavailable", ha="center")
            else:
                ax.bar(i, value, color=color, width=0.65)
                ax.text(i, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1], ["MAPE", "WAPE"])
        ax.set_title(
            f"{title}\n{row['predicted_points']:,}/{row['observed_points']:,} supported",
            fontsize=10,
        )
        ax.set_ylabel("Absolute prediction error (%)")
        ax.set_ylim(
            bottom=0,
            top=max(row["mape_percent"] or 0, row["wape_percent"] or 0, 1) * 1.2,
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("GB200 TP4 · ordinary serving · prefix retention128 · Decoder OFF", fontsize=13)
    fig.supxlabel(
        "30 independent trials per scenario. Two observed TTFT precision misses: short 8.90%, prefix reuse 5.65%.\n"
        "All cache disagreements retained. Native intervals are correlated; these bars have no pooled CI.",
        fontsize=9,
    )
    fig.savefig(
        output / "comparison.png",
        dpi=180,
        metadata={"Software": "AISimulate frozen paired-report renderer"},
    )
    fig.savefig(
        output / "comparison.pdf",
        metadata={
            "CreationDate": None,
            "ModDate": None,
            "Creator": "AISimulate frozen paired-report renderer",
        },
    )
    plt.close(fig)


def markdown(rows, summary, provenance):
    lines = [
        "# GB200 TP4 ordinary-serving FPM verification",
        "",
        "The frozen model predicts every measured primary cohort and native interval, but errors are large. "
        "This report retains every observed pair, including cache disagreements and unmet precision targets. "
        "It provides validation evidence; it does not establish accurate serving predictions.",
        "",
        "Native prefix retention is **128**, with observed prefix reuse **512 tokens**. Decoder replay is OFF; "
        "execution is eager, TP4/EP1/DP1/PP1, Engram in HBM, with the pinned checkpoint's mixed precision "
        "and explicit FP8 FMHA table identity. "
        "The results are conditional on this policy, one physical lifecycle and the unchanged 126-point calibration.",
        "",
        "| Metric | Supported / observed | Observed mean (supported) | Predicted mean | MAPE | WAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["purpose"] is not None:
            continue
        values = [
            "unavailable" if row[k] is None else f"{row[k]:.4f}"
            for k in (
                "observed_mean_supported",
                "predicted_mean_supported",
                "mape_percent",
                "wape_percent",
            )
        ]
        lines.append(
            f"| {row['metric']} | {row['predicted_points']}/{row['observed_points']} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "![Paired MAPE and WAPE](comparison.png)",
        "",
        "## Coverage and uncertainty",
        "",
        "The main stage has 30 independent trials per eligible scenario: 15 primary scenarios, 450 primary cohorts "
        "and 660 primary requests. Accuracy uses those 450 cohorts and their 12,531 native intervals. "
        "Full main coverage is 510 cohorts/720 requests, including 60 setup cohorts/requests. "
        "Ten pilot trials determine the fixed sample budget and are excluded. All phases together contain "
        "705 cohorts/994 requests and 17,230 active dispatches plus 41 heartbeat records; those are "
        "lifecycle counts, not accuracy denominators.",
        "",
        "MAPE, WAPE and supported means use identical successful observed/predicted pairs. "
        "Per-scenario values and original intervals remain in the compressed reports and CSV. "
        "Output tokens per second is finite-cohort throughput, not saturation throughput or serving capacity. "
        "All 30 trials are retained. Native intervals are correlated and are never independent replicates. "
        "The frozen sample budget remained 30; the existing 100-trial cap was not active.",
        "",
        "Two observed TTFT cells miss the 5% relative half-width target:",
        "",
        "| Scenario | Observed mean, ms | Pointwise 95% CI, ms | Relative half-width |",
        "|---|---:|---|---:|",
    ]
    for row in summary["observation_precision_misses"]:
        lines.append(
            f"| {row['purpose']} | {row['mean']:.7f} | {row['mean_bootstrap_ci95'][0]:.9f}"
            f" to {row['mean_bootstrap_ci95'][1]:.9f} | {100 * row['relative_half_width']:.4f}% |"
        )
    lines += [
        "",
        "The other 43 preregistered observation cells and 15 supplementary exact-ITL cells meet 5%. "
        "Observation intervals use 5,000 whole-trial resamples with seed 92031519. "
        "Comparison intervals use 5,000 paired whole-trial resamples with seed 94051000. "
        "E2E stores an observed-mean CI and a **ratio-of-sums error CI**: `100*(sum(predicted)/sum(observed)-1)`. "
        "The latter is not a CI for mean per-trial percentage error, MAPE or WAPE. E2E MAPE/WAPE CIs are absent. "
        "Native intervals retain whole-trial CIs for mean trial total-forward bias, interval MAPE and interval WAPE. "
        "All CIs are pointwise; no pooled cross-scenario CI is claimed. Error p90 is across whole trials "
        "(or correlated native intervals where labeled), not token-tail latency. Mean ITL/TPOT is not tail ITL. "
        "Response-completion and last-token metrics are supplementary and did not enter the pilot N rule.",
        "",
        "## Cache, calibration and runtime limits",
        "",
        "All 13 cache-semantic disagreements among 450 primary cohorts are retained: "
        "13/30 engram-repeated-text cohorts. Native and replay cache-token counts remain alongside every request. "
        "All 30 prefix-reuse-B main cohorts observed 512 cached tokens. "
        "The native pressure witness reports 24,958 allocation attempts, zero refusals/exceptions/preemptions, "
        "and minimum free physical blocks 198,246 above watermark 0 (198,374 total). "
        "Shared physical blocks are not converted to logical token capacity; allocator memory accuracy "
        "is not validated.",
        "",
        "The calibration contains 126/126 measured points (100 prefill, 26 decode). "
        "Calibration consumer self-queries are excluded from accuracy. Native query-coordinate coverage is:",
        "",
        "| Native coordinate region | Supported / observed intervals |",
        "|---|---:|",
    ]
    for name, row in summary["calibration_coordinate_coverage"]["region_metrics"].items():
        lines.append(f"| {name} | {row['predicted_points']}/{row['observed_points']} |")
    lines += [
        "",
        "These are aggregate coordinates actually submitted to the whole-forward predictor after the "
        "declared axis bridge. "
        "Exact coordinate equality does not mean identical request content, homogeneous lengths or schedule. "
        "Inside a raw bounding box is not proof of interpolation coverage; outside it is not "
        "automatically a failed prediction. "
        "Existing SOL anchors, clamps and floors are unchanged. Prediction success and "
        "measured-coordinate coverage are separate. "
        "Native region labels do not classify the HTTP replay's potentially different queries. The "
        "heterogeneous HTTP cohort "
        "contains three queued requests, while max-num-seqs is two; HTTP concurrency is not native batch size. "
        "The normalized corpus role is primary; frozen coverage/stress roles are preserved.",
        "",
        "The ordinary free-autoregressive canary has 8/10 equal output sequences and 2/10 different sequences, "
        "with 2/16 aligned token positions different. All outputs and timings are retained, and "
        "reference tokens do not "
        "control sampling. This comparison is not a main accuracy denominator and does not establish "
        "model-quality equivalence.",
        "",
        "Native observations are vLLM CPU schedule/output or adjacent-output intervals, not SGLang "
        "GPU-event intervals. "
        "HTTP adds frontend, transport and completion costs. The current vLLM Aggregated replay still "
        "charges a separate "
        "first-output decode after prefill. Current SGLang emits its first token at final prefill. "
        "[CURRENT_REPLAY_SEMANTICS.md](../CURRENT_REPLAY_SEMANTICS.md) corrects stale shared prose in "
        "the untouched E2E report. "
        "No decode time is subtracted from predictions. The large native interval error exists "
        "independently of that HTTP "
        "first-token limitation; no causal kernel explanation is established. These are retention128 "
        "results, not default-retention0 results.",
        "",
        "## Provenance and reproduction",
        "",
        f"Prediction model/native identity: commit `{provenance['prediction_commit']}`, native extension "
        f"`{provenance['prediction_native_extension_sha256']}`. "
        "The forward analysis helper has a separately recorded source hash. "
        "Its fixes preserve observed/native values and predictor formulas/configuration; the original failed attempt "
        "remains private and unchanged. They are not a model refit.",
        "",
        f"Base ARM64 image digest: `{provenance['base_arm64_image_digest']}`. "
        f"Actual prepared squashfs bytes SHA-256: `{provenance['actual_prepared_squashfs_sha256']}` "
        f"({provenance['actual_prepared_squashfs_bytes']:,} bytes). These identify different objects.",
        "",
        "`provenance.json` binds the exact original private reports, source files and calibration inventory. "
        "The public data is an allowlisted presentation projection with ordinal trial/scenario/request/interval IDs. "
        "Retained measurement/prediction numbers and CI arrays are preserved; private counters, paths, "
        "allocation/node details and UUIDs are omitted. "
        "Original raw evidence stays private. No runtime/data-loader contract is changed and no "
        "recovered third-party runtime source is included.",
        "",
        "From the repository root, regenerate tables and figures into a new directory:",
        "",
        "```sh",
        "python data/experimental/deepseek-v41/gb200-fpm/ordinary-serving-retention128-v1/render_report.py \\",
        "  --output /path/to/new-report",
        "```",
        "",
        "The renderer validates `render-inputs.json`, rechecks paired metric arithmetic, and uses only "
        "the public compressed data. "
        "It never invokes FPM or inference. Python and Matplotlib are required. "
        "`export_public.py --help` documents private-evidence projection inputs; the exact private "
        "invocation and file hashes "
        "are retained in the private export receipt. Export requires the unchanged explicit FP8 config "
        "and existing calibration-v1 overlay. "
        "Gzip headers and PDF metadata are deterministic; regenerated image bytes require the recorded "
        "Matplotlib version. "
        "`report/plot-input-hashes.json` records renderer/input hashes and plotting versions.",
        "",
    ]
    return "\n".join(lines)


def render(root, output):
    root, output = root.resolve(), output.resolve()
    require(not output.exists(), "renderer requires a fresh output directory")
    manifest = read(root / "render-inputs.json")["files_sha256"]
    require(
        all(
            (root / name).resolve().is_relative_to(root) and sha(root / name) == digest
            for name, digest in manifest.items()
        ),
        "public render input differs",
    )
    e2e, trace, summary = [read(root / "reports/main" / (name + ".json.gz")) for name in ("e2e", "trace", "summary")]
    provenance = read(root / "provenance.json")
    rows = metric_rows(e2e, trace)
    require(
        rows == summary["metrics"],
        "public paired arithmetic differs from original export",
    )
    intervals = interval_rows(e2e, trace, summary)
    privacy_check(intervals)
    output.mkdir(parents=True)
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "scenario-confidence-intervals.csv", intervals)
    draw(rows, output)
    document = markdown(rows, summary, provenance)
    privacy_check(document)
    (output / "README.md").write_text(document)
    import matplotlib

    (output / "plot-input-hashes.json").write_bytes(
        encoded(
            {
                "source_files_sha256": manifest,
                "renderer_sha256": sha(__file__),
                "matplotlib_version": matplotlib.__version__,
                "metric_rows": len(rows),
                "interval_rows": len(intervals),
                "predictions_or_fitting_run": False,
            }
        )
    )
    require(
        all(sha(root / name) == digest for name, digest in manifest.items()),
        "public input changed during render",
    )
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.input_root, args.output)
