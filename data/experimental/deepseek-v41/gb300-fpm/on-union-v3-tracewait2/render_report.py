# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render frozen independent FPM verification; never calibrate or run a predictor."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

METRICS = (
    "ttft_ms",
    "average_tpot_ms",
    "exact_itl_ms",
    "output_tokens_per_second",
    "request_latency_ms",
    "last_token_latency_ms",
    "native_forward_interval_ms",
)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    if path.name == "summary.json" and not path.exists() and not path.is_symlink():
        path = path.with_name("summary.json.gz")
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith(".gz") else data)


def cases(directory):
    if (directory / "segments.json.gz").exists():
        document = read(directory / "segments.json.gz")
        require(document["summary"]["pooled_confidence_intervals"] is None, "pooled core-ON CI is forbidden")
        return (
            [c for s in document["segments"] for c in s["e2e_cases"]],
            [c for s in document["segments"] for c in s["trace_cases"]],
        )
    return read(directory / "e2e.json.gz")["cohorts"], read(directory / "trace.json.gz")["cohorts"]


def metrics(http, trace, purpose=None):
    http = [c for c in http if purpose is None or c["purpose"] == purpose]
    trace = [c for c in trace if purpose is None or c["purpose"] == purpose]
    result = []
    for metric in METRICS:
        if metric == "native_forward_interval_ms":
            observations = [r for c in trace for r in c["intervals"]]
            pairs = [(r["observed_ms"], r["predicted_ms"]) for r in observations if r["status"] == "predicted"]
        else:
            observations = [c for c in http if metric in c["observed"]]
            pairs = [
                (c["observed"][metric], c["prediction"][metric])
                for c in observations
                if c["status"] == "predicted" and metric in c["prediction"]
            ]
        require(
            all(
                type(y) in (int, float) and math.isfinite(y) and y > 0 and type(p) in (int, float) and math.isfinite(p)
                for y, p in pairs
            ),
            "invalid supported pair",
        )
        result.append(
            dict(
                metric=metric,
                purpose=purpose,
                planned_points=len(observations),
                predicted_points=len(pairs),
                mape_percent=statistics.mean(100 * abs(p - y) / y for y, p in pairs) if pairs else None,
                wape_percent=100 * sum(abs(p - y) for y, p in pairs) / sum(y for y, p in pairs) if pairs else None,
                observed_mean_supported=statistics.mean(y for y, _ in pairs) if pairs else None,
                predicted_mean_supported=statistics.mean(p for _, p in pairs) if pairs else None,
            )
        )
    return result


def verify_metrics(stored, computed):
    require(len(stored) == len(computed), "metric row count differs")
    for left, right in zip(stored, computed, strict=True):
        require(left["metric"] == right["metric"], "metric order differs")
        for key in ("planned_points", "predicted_points"):
            require(type(left[key]) is int and left[key] == right[key], "metric coverage changed")
        for key in ("mape_percent", "wape_percent", "observed_mean_supported", "predicted_mean_supported"):
            if right[key] is None:
                require(left[key] is None, "missing predictions became accuracy values")
            else:
                require(math.isclose(left[key], right[key], rel_tol=1e-12, abs_tol=1e-10), "metric arithmetic changed")


def render(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "renderer requires a fresh output directory")
    manifest = read(root / "render-inputs.json")
    for name, digest in manifest["files_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and sha(path) == digest, "render input changed")
    before = dict(manifest["files_sha256"])
    report = read(root / "report.json")
    require(
        report["dataset_role"] == "independent_verification"
        and report["calibration_self_queries_in_accuracy"] is False,
        "calibration self-queries are not independent accuracy",
    )
    rows = []
    for scope in report["scopes"]:
        directory = root / "reports" / scope["name"]
        stored = read(directory / "summary.json")
        require(
            stored["correction_fitting"] is False and stored["calibration_consumer_self_queries_are_accuracy"] is False,
            "unqualified accuracy report",
        )
        http, trace = cases(directory)
        aggregate = metrics(http, trace)
        verify_metrics(stored["metrics"], aggregate)
        # Preserve the exact stored aggregate floats. Independent recomputation is
        # a validation check, not a replacement for original reported values.
        rows.extend(r | {"scope": scope["name"], "purpose": None} for r in stored["metrics"])
        for purpose in sorted({c["purpose"] for c in http}):
            values = metrics(http, trace, purpose)
            if "per_scenario_metrics" in stored:
                original = [r for r in stored["per_scenario_metrics"] if r["purpose"] == purpose]
                verify_metrics(original, values)
                values = original
            rows.extend(r | {"scope": scope["name"], "purpose": purpose} for r in values)
    output.mkdir(parents=True, exist_ok=False)
    fields = (
        "scope",
        "purpose",
        "metric",
        "planned_points",
        "predicted_points",
        "observed_mean_supported",
        "predicted_mean_supported",
        "mape_percent",
        "wape_percent",
    )
    with (output / "comparison.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    draw(rows, report["profile"], output)
    lines = [
        "# GB300 TP4 independent whole-forward FPM verification",
        "",
        f"Decoder replay: **{report['profile'].upper()}**. Core, field and service remain separate.",
        "",
        "| Scope | Metric | Supported / observed | Real mean | Prediction mean | MAPE | WAPE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    def display(value):
        return "unavailable" if value is None else f"{value:.4f}"

    for row in rows:
        if row["purpose"] is not None:
            continue
        cells = [row["scope"], row["metric"], f"{row['predicted_points']}/{row['planned_points']}"]
        cells += [display(row[k]) for k in fields[5:]]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "![MAPE and WAPE](comparison.png)",
        "",
        "Both errors use identical supported pairs. Missing predictions remain in coverage "
        "and keep their original failure reasons.",
        "Observed coverage, prediction coverage and measured calibration-coordinate coverage "
        "are separate in each source summary.",
        "Native intervals are correlated. Original scenario CIs remain in the compressed comparison outputs; "
        "core ON stays segmented, with no pooled lifecycle CI.",
        "Core scenario rows spanning lifecycle segments are descriptive error aggregates, without a pooled CI.",
        "HTTP timing includes service costs outside the native DeviceTimer interval; mean TPOT is not tail ITL.",
        "The calibration-only consumer self-query checks appear separately under ../calibration "
        "and are excluded from every accuracy table and plot.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    require(before == {name: sha(root / name) for name in before}, "input changed during rendering")
    (output / "plot-input-hashes.json").write_text(
        json.dumps(
            dict(
                source_files_sha256=before,
                renderer_sha256=sha(__file__),
                calibration_self_queries_in_accuracy=False,
                aggregate_rows=sum(r["purpose"] is None for r in rows),
                scenario_rows=sum(r["purpose"] is not None for r in rows),
            ),
            indent=2,
        )
        + "\n"
    )
    return rows


def draw(rows, profile, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["ttft_ms", "average_tpot_ms", "output_tokens_per_second", "native_forward_interval_ms"]
    scopes = ["core", "field", "service"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, metric in zip(axes, names, strict=True):
        data = {r["scope"]: r for r in rows if r["metric"] == metric and r["purpose"] is None}
        for offset, key, color in [(-0.18, "mape_percent", "#2475ad"), (0.18, "wape_percent", "#d97b15")]:
            for i, scope in enumerate(scopes):
                value = data[scope][key]
                if value is None:
                    ax.text(i + offset, 0, "missing", rotation=90, va="bottom", ha="center", fontsize=7)
                else:
                    ax.bar(
                        i + offset, value, width=0.34, color=color, label=key.split("_")[0].upper() if i == 0 else None
                    )
        ax.set_xticks(range(3), scopes)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel("Absolute error (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle(f"GB300 TP4 · Decoder {profile.upper()} · independent whole-forward FPM verification")
    fig.savefig(output / "comparison.png", dpi=170)
    fig.savefig(output / "comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.input_root, args.output)
