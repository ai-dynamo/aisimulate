# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render immutable forward comparisons; never fit or recompute predictions."""

import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
PROFILES = {"full": "Decoder OFF", "decoder_bounded": "Decoder ON"}
MODES = {"sol": "SOL", "hybrid": "HYBRID", "silicon": "SILICON"}
COLORS = {"sol": "#9271B1", "hybrid": "#D88320", "silicon": "#157A8C"}


REPORT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "descriptive_metrics.py").is_file())
_metrics_spec = importlib.util.spec_from_file_location(
    "dsv41_descriptive_metrics", REPORT_ROOT / "descriptive_metrics.py"
)
descriptive = importlib.util.module_from_spec(_metrics_spec)
_metrics_spec.loader.exec_module(descriptive)


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def summary(rows):
    signed = [r["signed_error_percent"] for r in rows]
    absolute = [abs(v) for v in signed]
    return [
        len(rows),
        statistics.mean(signed),
        statistics.mean(absolute),
        statistics.median(absolute),
        quantile(absolute, 0.9),
        100 * sum(abs(r["predicted_ms"] - r["observed_ms"]) for r in rows) / sum(r["observed_ms"] for r in rows),
    ]


def main():
    reports = {(p, m): json.loads((ROOT / f"{p}-{m}-results.json").read_bytes()) for p in PROFILES for m in MODES}
    descriptive.forward(ROOT, reports)
    lines = [
        "# GB300 independent native forward comparison",
        "",
        "The complete fixed holdout contains 38 configurations per Decoder profile. "
        "SILICON predicts all 38 with Decoder OFF and 28 with Decoder ON. The ten missing "
        "ON cases remain missing; HYBRID uses analytical fallback for those cases. "
        "These measurements do not establish HTTP E2E or whole-forward FPM accuracy.",
        "",
        "## Observed versus predicted",
        "",
        "Each observation is the median of ten independently invoked forwards, after "
        "taking the maximum of all four TP ranks for each invocation. One warmup per "
        "configuration is retained and excluded. Error is `(prediction / observation - 1) * 100`. "
        "Configurations have equal weight for signed bias, MAPE and APE percentiles. MAPE is "
        "`mean(abs(prediction / observation - 1)) * 100`; WAPE is "
        "`sum(abs(prediction - observation)) / sum(observation)`. The p90 column describes "
        "errors across configurations, not request tail latency.",
        "",
        "| Profile | Mode | Predicted / measured | Mean signed error | Median APE | p90 APE | MAPE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (profile, mode), report in reports.items():
        s = report["summary"]
        lines.append(
            f"| {PROFILES[profile]} | {MODES[mode]} | {s['predicted_points']}/38 | "
            f"{s['mean_signed_error_percent']:+.2f}% | {s['median_absolute_error_percent']:.2f}% | "
            f"{s['p90_absolute_error_percent_across_configurations']:.2f}% | "
            f"{s['mape_percent']:.2f}% | {s['wape_percent']:.2f}% |"
        )
    lines += [
        "",
        "SILICON enforces measured V4.1 table coverage with shared-layer reuse disabled. "
        "Generic embedding, normalization, activation and memory operations still use the "
        "existing empirical models; a stage total is not entirely measured. SOL is the "
        "analytical lower-bound model and its large negative bias is reported explicitly. "
        "No correction factor was fitted to these holdouts.",
        "",
        "![Prediction versus native forward](forward-comparison.png)",
        "",
        "Horizontal whiskers show the minimum and maximum of the ten measured repetitions, "
        "not confidence intervals. The dashed line denotes exact agreement. Missing "
        "predictions have no point on the scatter and remain in the coverage denominator.",
        "",
        "## Phase breakdown",
        "",
        "| Profile | Mode | Phase | Coverage | Mean signed error | Median APE | MAPE | WAPE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for (profile, mode), report in reports.items():
        for phase, s in report["by_phase"].items():
            lines.append(
                f"| {PROFILES[profile]} | {MODES[mode]} | {'Prefill' if phase == 'context' else 'Decode'} | "
                f"{s['predicted_points']}/{s['planned_points']} | {s['mean_signed_error_percent']:+.2f}% | "
                f"{s['median_absolute_error_percent']:.2f}% | {s['mape_percent']:.2f}% | {s['wape_percent']:.2f}% |"
            )
    lines += [
        "",
        "Compare modes on the same supported subset as well: a smaller coverage set can otherwise conceal hard cases.",
        "",
        "| Profile | Mode | Common strict subset | Mean signed error | Median APE | p90 APE | MAPE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        common = {r["case_id"] for r in reports[profile, "silicon"]["cases"] if r["status"] == "predicted"}
        for mode in MODES:
            rows = [r for r in reports[profile, mode]["cases"] if r["case_id"] in common]
            n, bias, mape, median, p90, wape = summary(rows)
            lines.append(
                f"| {PROFILES[profile]} | {MODES[mode]} | {n}/38 | "
                f"{bias:+.2f}% | {median:.2f}% | {p90:.2f}% | {mape:.2f}% | {wape:.2f}% |"
            )
    lines += [
        "",
        "## Repeat stability",
        "",
        "| Profile | Median within-point CV | Largest within-point CV | Case with largest CV | Repetitions (ms) |",
        "|---|---:|---:|---|---|",
    ]
    for profile in PROFILES:
        rows = reports[profile, "silicon"]["cases"]
        cvs = [statistics.stdev(r["rank_max_ms"]) / statistics.mean(r["rank_max_ms"]) for r in rows]
        worst = rows[max(range(len(rows)), key=lambda i: cvs[i])]
        times = ", ".join(f"{v:.3f}" for v in worst["rank_max_ms"])
        lines.append(
            f"| {PROFILES[profile]} | {statistics.median(cvs) * 100:.2f}% | "
            f"{max(cvs) * 100:.2f}% | {worst['case_id']} | {times} |"
        )
    lines += [
        "",
        "This is a separate precision attempt for all 38 holdouts in both profiles. "
        "The original three-repeat attempt is preserved [alongside this report](../README.md). "
        "The ON B2 / 96-new-token / zero-prefix observation changed from 379.867 ms "
        "to 187.947 ms (within-point CV 0.353%). Both slow original forwards remain in "
        "the original report. No sample was deleted, pooled across attempts, or used to refit "
        "the model. The largest new ON CV is 7.857% and is also retained. Stable repeats "
        "within this attempt do not establish stability across runtime lifecycles.",
        "",
        "## Coverage and limitations",
        "",
        "The frozen calibration contains 126 configurations per profile (100 prefill, "
        "26 decode); the separate holdout contains 38 (28 prefill, 10 decode). Calibration has one "
        "warmup and three measured repetitions per configuration. "
        "This separate precision holdout has one warmup and ten measured repetitions. Per profile "
        "that is 378 calibration and 380 precision holdout measured invocations, each requiring four "
        "rank records. TP ranks and component keys are not independent workloads. "
        "Both final calibration/holdout attempts passed their raw evidence admission checks. "
        "See the [study evidence and attempt notes](../../study/README.md) for collection details.",
        "",
        "Decoder OFF has 836 calibrated V4.1 physical module keys; Decoder ON has 830. "
        "Each has 80 GEMM/MoE/NCCL baseline keys. These calibration rows are never counted "
        "as independent accuracy results. The forward-only holdout ran without the "
        "component recorder and contributed no component rows to calibration.",
        "",
        "The ON holdout exposes exact-prefix coverage gaps after the bounded layers shift "
        "a long extension to its last 128 tokens. The current lookup requires the shifted "
        "prefix bucket to exist. Missing cases are listed below, with their full native "
        "failure retained in the result JSON. Filling these from holdout module timings "
        "would contaminate the test; this report performs no such refill.",
        "",
        "| Missing strict ON case | Batch | New tokens / request | Initial prefix |",
        "|---|---:|---:|---:|",
    ]
    for r in reports["decoder_bounded", "silicon"]["cases"]:
        if r["status"] != "predicted":
            lines.append(f"| {r['case_id']} | {r['batch_size']} | {r['query']} | {r['prefix']} |")
    lines += [
        "",
        "The observation boundary is the native SGLang synchronized wall time around "
        "preparation, forward and sampling. It includes shared Engram hash/history work "
        "omitted from the module graph. Native mHC retains its statistics stream; component "
        "measurement serializes that stream. Baseline expert routing is seeded uniform "
        "synthetic routing, while held-out forward uses real text. These are explicit "
        "possible sources of error, not measured causal attributions. Profiles were "
        "collected in separate runs, so their timing ratio is not a controlled paired "
        "estimate of Decoder optimization speedup.",
        "",
        "This is a fixed geometry grid with ten repetitions per point. It does not "
        "provide a production-workload confidence interval. Independent E2E trials will "
        "supply separate trial-level uncertainty estimates. Other corpora, larger batches, "
        "long contexts, candidate saturation, CUDA graphs, other parallel layouts, vision "
        "and DSpark remain outside this result.",
        "",
        "## Reproduction and identity",
        "",
        "Use the [pinned runtime, kernel and input evidence](../../study/README.md). The "
        "measurement uses four GB300 GPUs, pure TP4/EP1/DP1/PP1, SGLang `0.0.0.dev0`, "
        "ARM64 image `sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d`, "
        "checkpoint `fb2764a5cf321eaa5070ca8f9e892818f477c16d`, eager execution, DSpark off, "
        "Engram in HBM, and explicit NCCL with custom and FlashInfer AR fusion disabled. "
        "Resident unused vision weights are a runtime loading limitation; no vision work "
        "is executed. Do not reinterpret this as a graph-enabled or fused-AR result.",
        "",
        "1. In the SILICON checkout, rebuild the native extension and run "
        "`verify_study.py` with `PYTHONPATH=python/aisimulate/src:python/aisimulate`.",
        "2. Use the analysis utility from the sibling FPM PR at "
        "[`afcad7e2`](https://github.com/ai-dynamo/aisimulate/tree/"
        "afcad7e2/data/experimental/deepseek-v41/verification-plan). "
        "That utility calls the shared native forward API; it does not require FPM "
        "calibration to produce these op-level predictions. Keep the SILICON checkout "
        "as the working directory and Python source path.",
        "3. For each profile/mode, run `compare_forward.py --observations "
        "data/experimental/deepseek-v41/gb300-silicon/study/<profile>/precision-v2/forward-results.json "
        "--heldout-plan <FPM-checkout>/data/experimental/deepseek-v41/verification-plan/heldout.json "
        "--prediction-config "
        "data/experimental/deepseek-v41/gb300-silicon/report/precision-v2/<profile>-<mode>-config.json "
        "--output <new-results-file.json>`. The utility refuses to overwrite results.",
        "4. Run `render_report.py` to regenerate this document and PNG/PDF figures from "
        "the six stored comparison files. [SHA-256 receipt](artifact-hashes.json) pins "
        "the input/configuration/results and rendering source. Each result also pins "
        "the actually imported model, engine and native binary, independent of installed "
        "package metadata. Internal cluster logs and identifiers are kept separately.",
    ]
    lines += [
        "",
        "## Separate attempt comparison",
        "",
        "The model and calibration tables are unchanged. All 228 profile/mode predictions "
        "and availability decisions exactly match the original report; these are 76 unique "
        "profile/configuration pairs, not 228 independent measurements. Observations changed "
        "between separately launched native forward attempts. The following comparison "
        "keeps the original results visible rather than selecting the better error.",
        "",
        "| Profile | Mode | Original 3-repeat median APE | Precision 10-repeat median APE | "
        "Original MAPE | Precision MAPE | Original WAPE | Precision WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (profile, mode), report in reports.items():
        previous = json.loads((ROOT.parent / f"{profile}-{mode}-results.json").read_bytes())
        assert [(r["case_id"], r.get("predicted_ms"), r["status"]) for r in previous["cases"]] == [
            (r["case_id"], r.get("predicted_ms"), r["status"]) for r in report["cases"]
        ]
        a, b = previous["summary"], report["summary"]
        a["mape_percent"] = statistics.mean(
            abs(r["signed_error_percent"]) for r in previous["cases"] if r["status"] == "predicted"
        )
        lines.append(
            f"| {PROFILES[profile]} | {MODES[mode]} | "
            f"{a['median_absolute_error_percent']:.2f}% | {b['median_absolute_error_percent']:.2f}% | "
            f"{a['mape_percent']:.2f}% | {b['mape_percent']:.2f}% | "
            f"{a['wape_percent']:.2f}% | {b['wape_percent']:.2f}% |"
        )
    lines += [
        "",
        "The prediction receipt pins the resolved checkpoint configuration and inferred "
        "precision, the selected system specification and complete table overlay inventory, "
        "the native extension, and all comparison/axis/statistics sources. The native "
        "extension was rebuilt from source 8a4caf9c; its SHA is "
        "3e4de42a013ff5f37cdc7b5df7f7fdd70b8d70840f8c3c8cc2dcb968109451b4. "
        "Its predictions were checked against the original binary on all configurations.",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for i, profile in enumerate(PROFILES):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            rows = [r for r in reports[profile, mode]["cases"] if r["status"] == "predicted"]
            for phase, marker in [("context", "o"), ("generation", "^")]:
                selected = [r for r in rows if r["phase"] == phase]
                measured = [r["observed_ms"] for r in selected]
                predicted = [r["predicted_ms"] for r in selected]
                errors = [
                    [r["observed_ms"] - min(r["rank_max_ms"]) for r in selected],
                    [max(r["rank_max_ms"]) - r["observed_ms"] for r in selected],
                ]
                ax.errorbar(
                    measured,
                    predicted,
                    xerr=errors,
                    fmt=marker,
                    markersize=4,
                    alpha=0.75,
                    color=COLORS[mode],
                    label="Prefill" if phase == "context" else "Decode",
                )
            low = min(min(min(r["rank_max_ms"]), r["predicted_ms"]) for r in rows) * 0.8
            high = max(max(max(r["rank_max_ms"]), r["predicted_ms"]) for r in rows) * 1.1
            ax.plot([low, high], [low, high], "--", color="#78818C", linewidth=1)
            ax.set(
                xscale="log" if mode == "sol" else "linear",
                yscale="log" if mode == "sol" else "linear",
                xlim=(low, high),
                ylim=(low, high),
                title=f"{PROFILES[profile]} · {MODES[mode]} · {len(rows)}/38\n"
                f"MAPE {reports[profile, mode]['summary']['mape_percent']:.2f}% · "
                f"WAPE {reports[profile, mode]['summary']['wape_percent']:.2f}%",
                xlabel="Observed native forward (ms)",
                ylabel="Prediction (ms)",
            )
            if mode != "sol":
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(alpha=0.18, which="both")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "DeepSeek V4.1 Flash · GB300 TP4 · independent forward holdouts, precision v2",
        fontsize=15,
    )
    fig.savefig(ROOT / "forward-comparison.png", dpi=180)
    fig.savefig(
        ROOT / "forward-comparison.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    sources = sorted(p for p in ROOT.iterdir() if p.is_file() and p.name != "artifact-hashes.json")
    receipt = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (ROOT / "artifact-hashes.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
