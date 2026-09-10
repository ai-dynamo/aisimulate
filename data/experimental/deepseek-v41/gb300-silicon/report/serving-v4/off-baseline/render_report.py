# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Render frozen serving comparisons without changing observations or predictions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"sol": "#9271B1", "hybrid": "#D88320", "silicon": "#157A8C", "fpm": "#305AA8"}
METRICS = {
    "ttft_ms": ("TTFT", "ms"),
    "average_tpot_ms": ("Mean time per output token", "ms"),
    "output_tokens_per_second": ("Finite-cohort throughput", "tokens/s"),
    "request_latency_ms": ("Request completion latency", "ms"),
    "last_token_latency_ms": ("Time to last output token", "ms"),
    "exact_itl_ms": ("Mean inter-token latency", "ms"),
}


def load_reports(root):
    reports = {}
    for path in sorted(root.glob("e2e-*.json*")):
        suffix = ".json.gz" if path.name.endswith(".json.gz") else ".json"
        mode = path.name.removeprefix("e2e-").removesuffix(suffix)
        if mode in reports or mode not in COLORS:
            raise ValueError("duplicate or unknown prediction mode")

        def read_report(p):
            return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes())

        e2e = read_report(path)
        trace = read_report(root / f"trace-{mode}{suffix}")
        if e2e["run_id"] != trace["run_id"] or e2e["prediction_config"] != trace["prediction_config"]:
            raise ValueError("HTTP and trace comparisons have different run/model identities")
        for key in (
            "backend",
            "system_name",
            "backend_version",
            "decoder_replay",
            "runtime_digest",
            "observer_source_sha256",
        ):
            if e2e["measurement_identity"][key] != trace["measurement_identity"][key]:
                raise ValueError("HTTP and trace measurement identities differ")
        reports[mode] = (e2e, trace)
    if not reports or len({r[0]["run_id"] for r in reports.values()}) != 1:
        raise ValueError("one measured lifecycle per report is required")
    observed = {mode: {r["cohort_id"]: r["observed"] for r in pair[0]["cohorts"]} for mode, pair in reports.items()}
    if any(rows != next(iter(observed.values())) for rows in observed.values()):
        raise ValueError("prediction modes do not compare the same observed cohorts")
    return reports


def error_point(axis, center, interval, y, *, color, label=None):
    axis.plot(center, y, "o", color=color, ms=3.5, label=label)
    if interval:
        axis.plot(interval, [y, y], color=color, lw=1.2)
        axis.plot(interval, [y, y], "|", color=color, ms=4)


def finish(fig, root, name, title, footer):
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.text(0.5, 0.012, footer, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.055, 1, 0.96), w_pad=2.5, h_pad=3)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve()
    reports = load_reports(root)
    if not args.diagnostic and any(not r["complete_requested_study"] for pair in reports.values() for r in pair):
        raise ValueError("incomplete observations may only render with an explicit diagnostic label")
    first = next(iter(reports.values()))[0]
    identity = first["measurement_identity"]
    profile = "ON" if identity["decoder_replay"] else "OFF"
    title = f"{identity['system_name'].upper()} TP4 / {identity['backend']} / Decoder {profile}"
    if args.diagnostic:
        title += " — INCOMPLETE DIAGNOSTIC"
    purposes = list(first["points"])
    offset = {m: (i - (len(reports) - 1) / 2) * 0.18 for i, m in enumerate(reports)}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    groups = list(METRICS.items())
    for figure_name, metrics in (
        ("e2e-comparison", groups[:3]),
        ("e2e-completion-comparison", groups[3:]),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(17, 12), gridspec_kw={"height_ratios": [1, 1.6]})
        for column, (metric, (label, unit)) in enumerate(metrics):
            scatter, forest = axes[:, column]
            values = []
            for mode, (e2e, _) in reports.items():
                color = COLORS.get(mode, "#333333")
                for i, purpose in enumerate(purposes):
                    point = e2e["points"][purpose]["metrics"].get(metric)
                    if not point:
                        continue
                    x, y = point["observed_mean"], point["predicted_mean"]
                    scatter.scatter(x, y, s=28, c=color, alpha=0.8, label=mode.upper() if i == 0 else None)
                    values.extend((x, y))
                    center = 100 * (y / x - 1)
                    error_point(
                        forest,
                        center,
                        point.get("paired_ratio_error_percent_bootstrap_ci95"),
                        i + offset[mode],
                        color=color,
                    )
            if values:
                extent = [min(values) / 1.3, max(values) * 1.3]
                scatter.plot(extent, extent, "--", color="#555555", lw=1)
                scatter.set(xscale="log", yscale="log", xlim=extent, ylim=extent)
            scatter.set(title=label, xlabel=f"Real silicon mean ({unit})", ylabel=f"Prediction mean ({unit})")
            scatter.grid(True, alpha=0.15)
            if column == 0:
                scatter.legend(fontsize=8)
            forest.axvline(0, color="#555555", lw=1)
            forest.set(
                yticks=range(len(purposes)), yticklabels=purposes, xlabel="Ratio of means error (%)", xscale="symlog"
            )
            forest.invert_yaxis()
            forest.grid(True, axis="x", alpha=0.2)
        finish(
            fig,
            root,
            figure_name,
            title,
            "Points: scenario means. Bars: 95% paired whole-trial bootstrap intervals where fully qualified. "
            "Missing predictions remain in coverage tables; SOL is an analytical lower bound.",
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    scatter, bias, wape = axes
    values = []
    for mode, (_, trace) in reports.items():
        color = COLORS.get(mode, "#333333")
        rows = [r for c in trace["cohorts"] for r in c["intervals"] if r["status"] == "predicted"]
        if rows:
            x, y = [r["observed_ms"] for r in rows], [r["predicted_ms"] for r in rows]
            scatter.scatter(x, y, color=color, s=4, alpha=0.18, rasterized=True, label=f"{mode.upper()} ({len(rows)})")
            values.extend((min(x), max(x), min(y), max(y)))
        for i, purpose in enumerate(purposes):
            point = trace.get("independent_trial_summary", {}).get(purpose, {})
            ci = point.get("whole_trial_bootstrap_ci95", {})
            for axis, key in ((bias, "mean_trial_total_forward_signed_error_percent"), (wape, "interval_wape_percent")):
                if key in point:
                    error_point(axis, point[key], ci.get(key), i + offset[mode], color=color)
    if values:
        extent = [min(values) / 1.3, max(values) * 1.3]
        scatter.plot(extent, extent, "--", color="#555555", lw=1)
        scatter.set(xscale="log", yscale="log", xlim=extent, ylim=extent)
    scatter.set(
        title="Every predicted native interval", xlabel="Real silicon forward (ms)", ylabel="Prediction forward (ms)"
    )
    scatter.grid(True, alpha=0.15)
    scatter.legend(fontsize=8, markerscale=2)
    for axis, label in ((bias, "Mean trial total-forward bias (%)"), (wape, "Interval WAPE (%)")):
        axis.set(yticks=range(len(purposes)), yticklabels=purposes, xlabel=label)
        axis.invert_yaxis()
        axis.axvline(0, color="#555555", lw=1)
        axis.grid(True, axis="x", alpha=0.2)
    finish(
        fig,
        root,
        "forward-trace-comparison",
        title,
        "Native intervals within a trial are correlated. Bars resample entire trials, "
        "including unreturned native overlap work. "
        "Partial prediction coverage receives no final interval.",
    )
    inventory = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob("*.json*"))
        if p.name != "plot-input-hashes.json"
    }
    inventory["render_report.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root / "plot-input-hashes.json").write_text(json.dumps(inventory, indent=2) + "\n")


if __name__ == "__main__":
    main()
