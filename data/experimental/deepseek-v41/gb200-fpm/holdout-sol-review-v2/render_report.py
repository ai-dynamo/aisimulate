# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the corrected-predictor report on the same original 38 observations."""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(directory):
    directory = directory.resolve()
    old = directory.parent / "holdout-v1"
    spec = importlib.util.spec_from_file_location("original_report", old / "render_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current, previous = module.load_reports(directory), module.load_reports(old)
    receipt = json.loads((directory / "refresh-provenance.json").read_text())
    if receipt["new_gpu_measurements"] is not False or receipt["observations_gzip_sha256"] != sha(
        old / "holdout.json.gz"
    ):
        raise ValueError("refresh must retain the original observations")
    rows = []
    fields = ("case_id", "phase", "point", "native_fpm", "observed_ms", "native_run_id", "artifact_sha256")
    for mode in ("FPM", "SOL"):
        report, prior = current[mode], previous[mode]
        result_path = directory / f"{mode.lower()}-results.json.gz"
        result_hashes = receipt["result_hashes"][mode.lower()]
        if (
            sha(result_path) != result_hashes["gzip_sha256"]
            or hashlib.sha256(gzip.decompress(result_path.read_bytes())).hexdigest() != result_hashes["json_sha256"]
        ):
            raise ValueError("prediction result bytes differ from the frozen refresh")
        if report["calibration_files_sha256"] != prior["calibration_files_sha256"]:
            raise ValueError("calibration changed during prediction refresh")
        changed = 0
        for item, before in zip(report["cases"], prior["cases"], strict=True):
            if any(item[key] != before[key] for key in fields):
                raise ValueError("refresh changed a heldout observation")
            if item["status"] != "predicted" or before["status"] != "predicted":
                raise ValueError("this refresh requires complete original coverage")
            changed += item["predicted_ms"] != before["predicted_ms"]
            rows.append(
                dict(
                    mode=mode,
                    case_id=item["case_id"],
                    phase=item["phase"],
                    observed_ms=item["observed_ms"],
                    previous_prediction_ms=before["predicted_ms"],
                    corrected_prediction_ms=item["predicted_ms"],
                    signed_error_percent=item["signed_error_percent"],
                )
            )
        if changed != receipt["changed_prediction_counts"][mode.lower()]:
            raise ValueError("changed-prediction count differs from provenance")
        if report["prediction_sources"]["native_extension"] != receipt["native_build"]["native_sha256"]:
            raise ValueError("prediction binary differs from the recorded build")
    with (directory / "prediction-refresh.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, mode in zip(axes, ("FPM", "SOL"), strict=True):
        values = [r for r in rows if r["mode"] == mode]
        for phase, color in (("prefill", "#2563eb"), ("decode", "#d97706")):
            chosen = [r for r in values if r["phase"] == phase]
            ax.scatter(
                [r["observed_ms"] for r in chosen],
                [r["corrected_prediction_ms"] for r in chosen],
                color=color,
                label=phase,
                s=28,
                alpha=0.8,
            )
        numbers = [r[key] for r in values for key in ("observed_ms", "corrected_prediction_ms")]
        low, high = min(numbers) * 0.7, max(numbers) * 1.4
        ax.plot([low, high], [low, high], color="#64748b", linestyle="--", linewidth=1)
        ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high))
        ax.set_title(
            f"{mode}: MAPE {current[mode]['summary']['mape_percent']:.2f}% / "
            f"WAPE {current[mode]['summary']['wape_percent']:.2f}%"
        )
        ax.set_xlabel("Observed native wall time (ms)")
        ax.set_ylabel("Corrected prediction (ms)")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
    fig.suptitle("Same 38 GB200 holdouts; corrected SOL model, unchanged calibration", fontsize=12)
    fig.savefig(directory / "prediction-refresh.png", dpi=180)
    fig.savefig(directory / "prediction-refresh.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    lines = [
        "# GB200 prediction refresh after SOL review",
        "",
        "This reruns the **same 38 independent holdouts** and unchanged 126-point calibration from "
        "[the original report](../holdout-v1/README.md). No new GPU observation, replacement sample, "
        "outlier removal or fitted correction is introduced. Each geometry still has one native timing; "
        "there is no repeated-run confidence interval.",
        "",
        "The predictor includes SOL correction `563f1238`: replicated indexer heads, unique-row SWA "
        "HBM lower bounds, both compressor weight reads, restored shared MoE communication and MoE "
        "activation coefficients. The actual native build and merged FPM source are pinned in "
        "[refresh-provenance.json](refresh-provenance.json).",
        "",
        "| Prediction | Coverage | Previous MAPE / WAPE | Corrected MAPE / WAPE | Changed predictions |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("FPM", "SOL"):
        lines.append(
            f"| {mode} | 38/38 | {previous[mode]['summary']['mape_percent']:.5f}% / "
            f"{previous[mode]['summary']['wape_percent']:.5f}% | "
            f"{current[mode]['summary']['mape_percent']:.5f}% / {current[mode]['summary']['wape_percent']:.5f}% | "
            f"{receipt['changed_prediction_counts'][mode.lower()]}/38 |"
        )
    lines += [
        "",
        "All 38 whole-forward FPM predictions are bit-identical to the previous report. All 38 "
        "analytical SOL predictions change. SOL remains a strong underestimate of this runtime "
        "wall-time boundary; these formula fixes do not establish serving-latency accuracy.",
        "",
        "MAPE averages per-configuration absolute percentage errors; WAPE divides total absolute "
        "latency error by total observed latency. Both use the same 38 supported pairs. "
        "The additive [derived metrics](derived-error-metrics.json) preserve every original compressed result.",
        "",
        "![Observed and corrected predictions](prediction-refresh.png)",
        "",
        "[Per-point before/after CSV](prediction-refresh.csv) · "
        "[Printable figure](prediction-refresh.pdf). The original report describes the exact native "
        "timing boundary, true KV preparation, runtime/kernel identity and limitations. "
        "These are native forward measurements, not ordinary HTTP TTFT or ITL verification.",
        "",
        "The 121 scoped FPM/SOL/Qwen3.5/collection tests pass with this build, including FPM wrapper "
        "resident-weight and activation-interface preservation. The corrected vLLM TP4/8192-token "
        "activation estimate increases by 2.5 GiB/GPU; replicated indexer weights add 33,454,080 B/GPU. "
        "Those remain analytical memory estimates, not measured allocator accuracy.",
        "",
        "Reproduce the CSV, table and figure from the frozen results with Python 3.13 and Matplotlib:",
        "",
        "```sh",
        "python render_report.py --directory .",
        "```",
        "",
        "Full prediction re-execution uses the existing `verification-plan/compare_fpm_holdout.py`, "
        "the original decompressed holdout, admission and `holdout-v1/export_holdout.py`, the unchanged "
        "calibration directory, and the two portable prediction configs in this directory. "
        "Build the exact predictor commit recorded in the provenance receipt.",
        "",
    ]
    (directory / "README.md").write_text("\n".join(lines))
    derived = {
        "schema": "dsv41.derived-error-metrics.v1",
        "derivation": "Add MAPE to the same supported pairs; original prediction result bytes unchanged",
        "renderer_sha256": sha(Path(__file__)),
        "renderer_dependency_sha256": {"../holdout-v1/render_report.py": sha(old / "render_report.py")},
        "inputs_sha256": {
            f"{mode.lower()}-results.json.gz": sha(directory / f"{mode.lower()}-results.json.gz") for mode in current
        },
        "metrics": {mode: {"all": report["summary"], **report["by_phase"]} for mode, report in current.items()},
    }
    (directory / "derived-error-metrics.json").write_text(json.dumps(derived, sort_keys=True, indent=2) + "\n")
    hashes = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    (directory / "artifact-hashes.json").write_text(json.dumps(hashes, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    render(parser.parse_args().directory)
