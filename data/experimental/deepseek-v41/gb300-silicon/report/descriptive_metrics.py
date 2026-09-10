# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Describe frozen supported pairs without rerunning a predictor or changing CIs."""

import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen(root):
    receipt = json.loads((root / "descriptive-inputs.json").read_bytes())
    if receipt["original_artifact_hashes_sha256"] != sha(root / "original-artifact-hashes.json"):
        raise ValueError("original report receipt changed")
    original = json.loads((root / "original-artifact-hashes.json").read_bytes())
    original = original.get("files_sha256", original)
    for name, digest in receipt["comparison_files_sha256"].items():
        if original.get(name) != digest:
            raise ValueError("comparison is not bound to the original report receipt")
        if sha(root / name) != digest:
            raise ValueError(f"frozen comparison changed: {name}")
    return receipt


def paired(pairs, observed_count):
    if type(observed_count) is not int or observed_count < len(pairs):
        raise ValueError("invalid observed coverage denominator")
    if any(not math.isfinite(v) or v <= 0 for pair in pairs for v in pair):
        raise ValueError("metrics require finite positive observed/predicted values")
    result = {
        "observed": observed_count,
        "predicted": len(pairs),
        "missing": observed_count - len(pairs),
    }
    if pairs:
        result.update(
            mape_percent=statistics.mean(100 * abs(p / o - 1) for o, p in pairs),
            wape_percent=100 * sum(abs(p - o) for o, p in pairs) / sum(o for o, _ in pairs),
        )
    return result


def write(root, statistics_value):
    frozen = validate_frozen(root)
    output = {
        "schema": "dsv41.descriptive.metrics.v1",
        "analysis_source_sha256": sha(__file__),
        "original_inputs": frozen,
        "definitions": {
            "mape_percent": "100 * mean(abs(predicted / observed - 1)) on supported pairs",
            "wape_percent": "100 * sum(abs(predicted - observed)) / sum(observed) on the same supported pairs",
            "coverage": "Missing predictions remain in the observed denominator; "
            "no missing prediction is assigned a value.",
            "uncertainty": "Descriptive addition only; original confidence intervals "
            "and their trial/lifecycle boundaries are unchanged.",
            "prediction_identity": "Comparison files and their original predictor/source identities "
            "remain byte-identical.",
        },
        "statistics": statistics_value,
    }
    (root / "descriptive-metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def plot_serving(root, values):
    """Plot descriptive pair errors separately from the original conditional CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    targets = [
        ("http", "ttft_ms", "HTTP TTFT / cohorts"),
        ("http", "average_tpot_ms", "HTTP mean time per token / cohorts"),
        ("http", "output_tokens_per_second", "HTTP finite-cohort throughput / cohorts"),
        ("trace", "all", "Native GPU forward / intervals"),
    ]
    for axis, (section, metric, label) in zip(axes.flat, targets, strict=True):
        modes = list(values[section])
        for offset, key, name, color in (
            (-0.18, "mape_percent", "MAPE", "#D88320"),
            (0.18, "wape_percent", "WAPE", "#157A8C"),
        ):
            bars = axis.bar(
                [i + offset for i in range(len(modes))],
                [values[section][m][metric].get(key, 0) for m in modes],
                width=0.34,
                label=name,
                color=color,
            )
            axis.bar_label(
                bars,
                labels=[
                    f"{values[section][m][metric][key]:.2f}" if key in values[section][m][metric] else "missing"
                    for m in modes
                ],
                fontsize=8,
                padding=3,
            )
        axis.set(
            title=label,
            ylabel="Absolute error (%)",
            xticks=range(len(modes)),
            xticklabels=[
                f"{m.upper()}\n{values[section][m][metric]['predicted']}/{values[section][m][metric]['observed']}"
                for m in modes
            ],
        )
        axis.margins(y=0.22)
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Descriptive MAPE and WAPE · same supported pairs within each mode\n"
        "Missing predictions remain in coverage; no new confidence interval",
        fontsize=13,
    )
    fig.savefig(root / "descriptive-error-comparison.png", dpi=180)
    fig.savefig(root / "descriptive-error-comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def forward(root, reports):
    validate_frozen(root)
    result = {}
    for (profile, mode), report in reports.items():
        groups = {"all": report["cases"]}
        groups.update({phase: [r for r in report["cases"] if r["phase"] == phase] for phase in report["by_phase"]})
        result[f"{profile}-{mode}"] = {}
        for phase, rows in groups.items():
            values = paired(
                [(r["observed_ms"], r["predicted_ms"]) for r in rows if r["status"] == "predicted"],
                len(rows),
            )
            old = report["summary"] if phase == "all" else report["by_phase"][phase]
            if not math.isclose(
                values["wape_percent"],
                old["wape_percent"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("frozen forward WAPE disagrees with supported pairs")
            old["mape_percent"] = values["mape_percent"]
            result[f"{profile}-{mode}"][phase] = values
    for profile in {p for p, _ in reports}:
        common = set.intersection(
            *[
                {r["case_id"] for r in report["cases"] if r["status"] == "predicted"}
                for (p, _), report in reports.items()
                if p == profile
            ]
        )
        for (p, mode), report in reports.items():
            if p == profile:
                rows = [r for r in report["cases"] if r["case_id"] in common]
                result[f"{profile}-{mode}"]["common_support"] = paired(
                    [(r["observed_ms"], r["predicted_ms"]) for r in rows], len(rows)
                )
    write(root, result)


def serving(root, reports, metrics):
    validate_frozen(root)
    result = {
        "http": {},
        "http_common_support": {},
        "trace": {},
        "trace_common_support": {},
        "native_fully_predicted_trials": {},
    }
    common_http = set.intersection(
        *[{c["cohort_id"] for c in e["cohorts"] if c["status"] == "predicted"} for e, _ in reports.values()]
    )
    common_native = set.intersection(
        *[
            {
                (c["cohort_id"], r["dispatch_id"])
                for c in t["cohorts"]
                for r in c["intervals"]
                if r["status"] == "predicted"
            }
            for _, t in reports.values()
        ]
    )
    for mode, (e2e, trace) in reports.items():
        for label, cohorts in (
            ("http", e2e["cohorts"]),
            (
                "http_common_support",
                [c for c in e2e["cohorts"] if c["cohort_id"] in common_http],
            ),
        ):
            result[label][mode] = {}
            for metric in metrics:
                selected = [c for c in cohorts if metric in c["observed"]]
                result[label][mode][metric] = paired(
                    [(c["observed"][metric], c["prediction"][metric]) for c in selected if c["status"] == "predicted"],
                    len(selected),
                )
        for label, rows in (
            ("trace", [r for c in trace["cohorts"] for r in c["intervals"]]),
            (
                "trace_common_support",
                [
                    r
                    for c in trace["cohorts"]
                    for r in c["intervals"]
                    if (c["cohort_id"], r["dispatch_id"]) in common_native
                ],
            ),
        ):
            result[label][mode] = {}
            for phase in ("all", "prefill", "decode", "mixed"):
                selected = [r for r in rows if phase == "all" or r["phase"] == phase]
                result[label][mode][phase] = paired(
                    [(r["observed_ms"], r["predicted_ms"]) for r in selected if r["status"] == "predicted"],
                    len(selected),
                )
        result["native_fully_predicted_trials"][mode] = {}
        for purpose, point in trace.get("independent_trial_summary", {}).items():
            cohorts = [
                c
                for c in trace["cohorts"]
                if c["purpose"] == purpose
                and c["intervals"]
                and all(r["status"] == "predicted" for r in c["intervals"])
            ]
            rows = [r for c in cohorts for r in c["intervals"]]
            if len(cohorts) != point["fully_predicted_trials"]:
                raise ValueError("native trial support changed")
            values = paired([(r["observed_ms"], r["predicted_ms"]) for r in rows], len(rows))
            if rows:
                if not math.isclose(
                    values["wape_percent"],
                    point["interval_wape_percent"],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError("native interval subset differs from frozen WAPE")
                point["interval_mape_percent"] = values["mape_percent"]
            result["native_fully_predicted_trials"][mode][purpose] = values
    write(root, result)
    plot_serving(root, result)
    return result
