# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render frozen, independently admitted GB200 holdout comparisons without fitting."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reports(directory):
    names = {"FPM": "fpm-results.json.gz", "SOL": "sol-results.json.gz"}
    reports = {mode: read(directory / name) for mode, name in names.items()}
    reference = reports["FPM"]
    for mode, report in reports.items():
        if report["schema"] != "dsv41.fpm.holdout.comparison.v1" or report["role"] != "independent_holdout":
            raise ValueError("renderer requires independently admitted native holdouts")
        if report["observation_repeats_per_point"] != 1 or report["confidence_intervals"] is not None:
            raise ValueError("this report has one observation per geometry and no repeat CI")
        if report["calibration_points"] != 126 or report["summary"]["planned_points"] != 38:
            raise ValueError("wrong frozen study size")
        expected = ("SILICON", "fpm") if mode == "FPM" else ("SOL", "op_level")
        config = report["prediction_contract"]
        if (config["database_mode"], config["forward_model"]) != expected:
            raise ValueError("wrong actual prediction path")
        if config.get("activation_dtype") is not None:
            raise ValueError("holdout comparison must preserve checkpoint-native analytical precision")
        selector = config.get("fpm_fmha_dtype")
        if selector != ("fp8" if mode == "FPM" else None):
            raise ValueError("wrong isolated FPM table selector")
        if mode == "FPM" and report["resolved_model_identity"].get("analytical_precision") != (
            "checkpoint_native_without_activation_override"
        ):
            raise ValueError("FPM analytical anchor precision is not qualified")
        if report["calibration_files_sha256"] != reference["calibration_files_sha256"]:
            raise ValueError("comparison overlays differ")
        sources = ["normalized_holdout", "holdout_admission", "normalizer_source"]
        if any(report["input_files_sha256"][key] != reference["input_files_sha256"][key] for key in sources):
            raise ValueError("comparison observations/admission differ")
        keys = ("case_id", "phase", "point", "native_fpm", "observed_ms", "native_run_id", "artifact_sha256")
        if len(report["cases"]) != 38 or any(
            any(a[key] != b[key] for key in keys) for a, b in zip(report["cases"], reference["cases"], strict=True)
        ):
            raise ValueError("real heldout observations differ across modes")
        for phase, summary in [(None, report["summary"]), *report["by_phase"].items()]:
            paired = [
                row
                for row in report["cases"]
                if row["status"] == "predicted" and (phase is None or row["phase"] == phase)
            ]
            summary["mape_percent"] = statistics.mean(
                abs(row["predicted_ms"] / row["observed_ms"] - 1) * 100 for row in paired
            )
    return reports


def metric(value, key):
    return "—" if key not in value else f"{value[key]:.2f}%"


def write_readme(directory, reports):
    lines = [
        "# Real silicon versus prediction: GB200 TP4 whole-forward holdouts",
        "",
        "This is **38 fresh, geometrically disjoint holdout configurations** (28 prefill, 10 decode) "
        "against the unchanged [126-point calibration](../calibration-v1/README.md). Each geometry has "
        "**one native timing sample**. Original real-token and KV histories, native warmup records, "
        "phase/attempt IDs, admission checks and runtime/source hashes accompany the results. "
        "Calibration self-queries are separate consumer checks and do not contribute to these errors.",
        "",
        "The measured configuration is one node, four GB200 GPUs, pure TP4/EP1/DP1/PP1/CP1, text AR, "
        "HBM Engram, eager execution, DSpark disabled and Decoder replay OFF. Installed vLLM is "
        "`0.1.dev20904+g179dd0fa9`; the composite Dynamo Python/native packages are 1.4.2 with four "
        "unchanged instrumentation modules from `54960177085413259859c88bd34ed0734d4c2ea9`. "
        "The frozen ARM64 image, actual prepared image, source manifest and selected "
        "`FLASHINFER_TRTLLM_MXFP4_MXFP8` kernel are recorded. A recipe commit is not the installed vLLM revision.",
        "",
        "## Forward accuracy",
        "",
        "The timing target is the native Dynamo FPM wall-time boundary: CPU schedule/output or adjacent "
        "output timing, depending on phase. Prefill uses a single step; decode uses a real-KV seeded "
        "steady-state second step. It is not a pure GPU device interval and is not interchangeable with "
        "the SGLang GPU-event study or HTTP TTFT/ITL. The consumer uses the native **past-KV** axis. "
        "The analytical op baseline adds the query once to obtain its inclusive attention length.",
        "",
        "| Prediction | Phase | Predicted / planned | Mean signed error | MAPE | Median APE | p90 APE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, report in reports.items():
        for phase, value in [("all", report["summary"]), *report["by_phase"].items()]:
            fields = [
                metric(value, key)
                for key in (
                    "mean_signed_error_percent",
                    "mape_percent",
                    "median_absolute_error_percent",
                    "p90_absolute_error_percent",
                    "wape_percent",
                )
            ]
            lines.append(
                f"| {mode} | {phase} | {value['predicted_points']}/{value['planned_points']} | "
                + " | ".join(fields)
                + " |"
            )
    lines += [
        "",
        "Missing predictions stay in the coverage denominator; error statistics use supported pairs. "
        "Signed/APE statistics weight each configuration equally. WAPE divides total absolute latency "
        "error by total observed latency. MAPE is the mean per-configuration absolute percentage error; "
        "both metrics use the same supported pairs. p90 APE is a percentile of prediction errors, "
        "not p90 serving latency. "
        "No correction factor, outlier removal, replacement sample, or fitting to these holdouts is applied.",
        "",
        "MAPE is an additive report statistic computed from the original observed/predicted pairs. "
        "[Derived metric provenance](derived-error-metrics.json) binds those unchanged compressed results "
        "and the reporting source; it does not replace the original prediction provenance.",
        "",
        "![Real native forward timing and prediction](forward-comparison.png)",
        "",
        "[Per-configuration observations and predictions (CSV)](forward-comparison.csv) · "
        "[Standalone figure (PDF)](forward-comparison.pdf)",
        "",
        "## Interpretation and precision",
        "",
        "- FPM selects measured FP8 FMHA table cells with the isolated `fpm_fmha_dtype=fp8` selector under "
        "the actual FP8 KV/Collector identity contract. This is not a claim that all attention arithmetic "
        "uses FP8. Both its SOL interpolation anchors and the standalone SOL baseline keep checkpoint-native "
        "analytical precision and the full mixed-precision model, with no activation-dtype override. "
        "The original calibration files remain unchanged; the new selector does not relabel measured rows.",
        "- SOL is an idealized logical-work baseline. The inspected installed FlashMLA wrapper pads the "
        "16 local query heads at TP4 to 64 for its supported kernel shape, while this SOL graph uses the "
        "model's logical head count. Kernel padding, launch/runtime overhead and native efficiency are not "
        "fully represented by that baseline; the selector fix does not add them. Installed-source hashes "
        "for this inspection are retained in `artifact-provenance.json`.",
        "- One sample per fixed geometry supports descriptive errors. It does not estimate repeated-run "
        "measurement variance or justify confidence intervals. TP ranks, tokens, warmups and calibration "
        "rows do not create independent holdout repetitions.",
        "- The two native measurement phases have different producer/run IDs. Their attempts and point "
        "manifests differ from calibration. Matching an installed runtime does not require reusing a "
        "phase-specific compute hash or autotuning directory.",
        "- These original-text holdouts test the frozen geometry domain. Dedicated corpus controls, "
        "ordinary serving prefix reuse, request-length mixtures, TTFT/ITL and throughput are separate "
        "verification reports. No production traffic distribution or Engram cache-locality conclusion is implied.",
        "- Decoder replay ON remains an unverified vLLM runtime dependency. OFF data cannot satisfy an ON query.",
        "- Internal allocation details, execution paths and original worker logs stay in the private evidence archive.",
        "",
        "## Evidence and reproduction",
        "",
        "`holdout.json.gz` preserves the exact normalized native artifact and complete token-stream "
        "sidecar bytes, including excluded warmup roles. `admission-receipt.json` binds the original "
        "Collector revalidation, frozen manifests, unchanged calibration, actual runtime verification and "
        "normalizer. The comparison adapter independently reruns the existing native Collector validator, "
        "then invokes the actual Rust whole-forward consumer. It records loaded model/source/binary hashes.",
        "",
        "The observed runtime checks include a comparison of read-only verification probe windows with "
        "the native benchmark windows, recorded in `admission-receipt.json`. Those timing-boundary checks "
        "do not establish absence of every possible environmental effect.",
        "",
        "`artifact-provenance.json` records original/decompressed hashes, exact comparison source revision, "
        "native binary identity, and portable prediction configs. The only configuration adaptation is "
        "the repository-relative calibration path. `artifact-hashes.json` hashes published files. "
        "`plot-input-hashes.json` binds renderer inputs. No model estimation happens during rendering.",
        "",
        "From this directory, with Python 3.13 and Matplotlib:",
        "",
        "```sh",
        "python render_report.py --directory .",
        "```",
        "",
        "This reproduces the CSV, tables and PNG/PDF figure from the frozen comparison outputs. "
        "Full comparison re-execution uses the pinned `verification-plan/compare_fpm_holdout.py`, "
        "decompressed native holdout file, admission, normalizer source, exact calibration overlay and "
        "native build recorded in the provenance receipt. Private runtime evidence is required to repeat "
        "the independent raw-admission/export step.",
        "",
    ]
    (directory / "README.md").write_text("\n".join(lines))


def render(directory, reports):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    colors = {"prefill": "#116a8b", "decode": "#c36321"}
    ref = reports["FPM"]["cases"]
    for phase in ("prefill", "decode"):
        rows = [r for r in ref if r["phase"] == phase and r["status"] == "predicted"]
        axes[0].scatter(
            [r["observed_ms"] for r in rows],
            [r["predicted_ms"] for r in rows],
            label=phase.capitalize(),
            color=colors[phase],
            s=34,
            alpha=0.75,
        )
    usable = [r for r in ref if r["status"] == "predicted"]
    if usable:
        values = [r[k] for r in usable for k in ("observed_ms", "predicted_ms")]
        lo, hi = min(values) * 0.92, max(values) * 1.04
        axes[0].plot([lo, hi], [lo, hi], "--", color="#666666", linewidth=1)
        axes[0].set(xlim=(lo, hi), ylim=(lo, hi))
    axes[0].set(title="FPM interpolation", xlabel="Observed native forward (ms)", ylabel="Predicted forward (ms)")
    axes[0].legend(frameon=False, fontsize=9)
    for mode, color in (("FPM", "#116a8b"), ("SOL", "#9272a3")):
        rows = reports[mode]["cases"]
        indices = [i for i, r in enumerate(rows) if r["status"] == "predicted"]
        axes[1].scatter(
            indices, [rows[i]["signed_error_percent"] for i in indices], label=mode, color=color, s=28, alpha=0.8
        )
    axes[1].axhline(0, color="#666666", linestyle="--", linewidth=1)
    axes[1].axvline(27.5, color="#aaaaaa", linestyle=":", linewidth=1)
    axes[1].set(
        title="Every retained configuration",
        xlabel="Frozen holdout index (28 prefill + 10 decode)",
        ylabel="Signed prediction error (%)",
    )
    axes[1].legend(frameon=False, fontsize=9)
    fpm_rows = {r["case_id"]: r for r in reports["FPM"]["cases"]}
    for phase in ("prefill", "decode"):
        rows = [(i, r) for i, r in enumerate(ref) if r["phase"] == phase]
        axes[2].scatter(
            [r["observed_ms"] for _, r in rows],
            [i for i, _ in rows],
            color=colors[phase],
            s=19,
            label="Observed " + phase,
        )
        predicted = [(i, fpm_rows[r["case_id"]]) for i, r in rows if fpm_rows[r["case_id"]]["status"] == "predicted"]
        axes[2].scatter(
            [r["predicted_ms"] for _, r in predicted],
            [i for i, _ in predicted],
            marker="x",
            color="#222222",
            s=22,
            label="FPM" if phase == "prefill" else None,
        )
    axes[2].set(title="Observed and FPM values", xlabel="Native forward (ms)", ylabel="Frozen holdout index")
    axes[2].invert_yaxis()
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("GB200 · pure TP4 · Decoder OFF · 38 independent geometries", fontsize=15, weight="bold", y=1.01)
    fig.text(
        0.5,
        -0.02,
        "One sample per geometry; descriptive errors, no confidence intervals. "
        "Native wall time differs from HTTP E2E and pure GPU time.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(directory / "forward-comparison.png", bbox_inches="tight")
    fig.savefig(
        directory / "forward-comparison.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    reports = load_reports(directory)
    fields = [
        "mode",
        "case_id",
        "phase",
        "batch_size",
        "total_prefill_tokens",
        "total_kv_read_tokens",
        "observed_ms",
        "predicted_ms",
        "signed_error_percent",
        "status",
        "failure_type",
    ]
    with (directory / "forward-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for mode, report in reports.items():
            for case in report["cases"]:
                writer.writerow(
                    dict(
                        mode=mode,
                        case_id=case["case_id"],
                        phase=case["phase"],
                        **{key: case["point"].get(key, 0) for key in fields[3:6]},
                        **{key: case.get(key, "") for key in fields[6:]},
                    )
                )
    write_readme(directory, reports)
    render(directory, reports)
    inputs = [directory / name for name in ("fpm-results.json.gz", "sol-results.json.gz", "render_report.py")]
    (directory / "plot-input-hashes.json").write_text(json.dumps({p.name: sha(p) for p in inputs}, indent=2) + "\n")
    derived = {
        "schema": "dsv41.derived-error-metrics.v1",
        "derivation": "Add MAPE to the same supported pairs; original prediction result bytes unchanged",
        "renderer_sha256": sha(Path(__file__)),
        "inputs_sha256": {p.name: sha(p) for p in inputs[:2]},
        "metrics": {mode: {"all": report["summary"], **report["by_phase"]} for mode, report in reports.items()},
    }
    (directory / "derived-error-metrics.json").write_text(json.dumps(derived, sort_keys=True, indent=2) + "\n")
    hashes = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    (directory / "artifact-hashes.json").write_text(json.dumps(hashes, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
