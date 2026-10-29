# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize recorded-output AgentX replays without mixing request and point errors."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ape(reference: float | None, simulated: float | None) -> float | None:
    if not finite(reference) or reference <= 0 or not finite(simulated) or simulated < 0:
        return None
    return 100 * abs(simulated - reference) / reference


def percentage(numerator: int | None, denominator: int | None) -> float | None:
    return 100 * numerator / denominator if numerator is not None and denominator else None


def load_point(root: Path, point: dict, baseline: Path | None) -> dict:
    point_id = int(point["id"])
    reused = point_id == 446063 and baseline is not None
    directory = baseline if reused else root / f"point-{point_id}"
    report = directory / "observed-report"
    accuracy = read(report / "accuracy-summary.json")
    measurement = read(report / "measurement.json")
    execution = read(directory / "observed-replay-execution.json")
    prediction = read(directory / "observed-replay/prediction.json")
    manifest = read(directory / "source/source-manifest.json")
    if accuracy and int(accuracy["point_id"]) != point_id:
        raise ValueError(f"Accuracy point identity differs from selected point {point_id}")
    if measurement and not measurement["benchmark_url"].rstrip("/").endswith(f"/{point_id}"):
        raise ValueError(f"Measurement point identity differs from selected point {point_id}")
    finished = execution.get("status") == "completed" and execution.get("returncode") == 0
    if finished and accuracy and measurement and prediction:
        status = "completed"
    elif execution and not finished:
        status = execution.get("status", "incomplete")
    else:
        status = "missing_results"
    coverage = accuracy.get("coverage", {})
    strict = accuracy.get("request_e2e_output_length_matched", {})
    reference = accuracy.get("reference_successful_profile", {})
    original = manifest.get("reference", {})
    simulated = accuracy.get("native_successful_profile", {})
    statuses = coverage.get("native_profile_terminal_statuses")
    native_wall = prediction.get("wall_time_ms")
    row = {
        "point_id": point_id,
        "concurrency": int(point["conc"]),
        "reused_baseline": reused,
        "status": status,
        "benchmark_url": f"https://inferencex.semianalysis.com/inference/agentic/{point_id}",
        "report_path": os.path.relpath(report / "report.html", root) if (report / "report.html").exists() else None,
        "reference_eligible_e2e_requests": coverage.get("reference_positive_e2e_requests"),
        "strict_e2e_pairs": strict.get("count"),
        "strict_e2e_pair_coverage_percent": coverage.get("strict_output_e2e_coverage_percent"),
        "strict_request_e2e_mape_percent": strict.get("mape_percent"),
        "reference_mean_e2e_ms": reference.get("mean_e2e_ms", original.get("mean_e2e_ms")),
        "simulated_mean_e2e_ms": simulated.get("mean_e2e_ms"),
        "reference_output_throughput_tok_s": reference.get("output_throughput_tok_s", original.get("output_tput_tps")),
        "simulated_output_throughput_tok_s": simulated.get("output_throughput_tok_s"),
        "reference_successes": reference.get("successful_requests", original.get("profile_successful_requests")),
        "simulated_successes": simulated.get("successful_requests"),
        "reference_errors": coverage.get("reference_errors_excluded", original.get("profile_error_requests")),
        "simulated_failures": statuses.get("failed", 0) if statuses is not None else None,
        "simulated_cancellations": (
            statuses.get("canceled", 0) + statuses.get("cancelled", 0) if statuses is not None else None
        ),
        "native_profile_terminal_statuses": statuses,
        "telemetry_completion_status": accuracy.get("telemetry_completion_status"),
        "cli_wall_time_seconds": execution.get("wall_time_seconds") if finished else None,
        "attempt_wall_time_seconds": execution.get("wall_time_seconds"),
        "cli_wall_time_scope": execution.get("wall_time_scope"),
        "native_wall_time_seconds": native_wall / 1000 if finite(native_wall) else None,
        "aisim_commit": execution.get("aisim_commit"),
        "host": execution.get("host", {}).get("hostname"),
    }
    row["mean_e2e_point_ape_percent"] = ape(row["reference_mean_e2e_ms"], row["simulated_mean_e2e_ms"])
    row["output_throughput_point_ape_percent"] = ape(
        row["reference_output_throughput_tok_s"], row["simulated_output_throughput_tok_s"]
    )
    return row


def aggregate(rows: list[dict]) -> dict:
    scored = [
        row
        for row in rows
        if row["strict_e2e_pairs"] is not None
        and row["strict_e2e_pairs"] > 0
        and finite(row["strict_request_e2e_mape_percent"])
    ]
    pairs = sum(row["strict_e2e_pairs"] for row in scored)
    eligible = [
        row["reference_eligible_e2e_requests"] for row in rows if row["reference_eligible_e2e_requests"] is not None
    ]
    result = {
        "selected_points": len(rows),
        "completed_points": sum(row["status"] == "completed" for row in rows),
        "strict_e2e_scored_points": len(scored),
        "strict_e2e_pair_count": pairs,
        "reference_eligible_count": sum(eligible) if eligible else None,
        "points_with_reference_eligible_count": len(eligible),
        "pooled_request_e2e_mape_percent": (
            sum(row["strict_e2e_pairs"] * row["strict_request_e2e_mape_percent"] for row in scored) / pairs
            if pairs
            else None
        ),
    }
    result["strict_e2e_pair_coverage_percent"] = percentage(pairs, result["reference_eligible_count"])
    for target, source in [
        ("macro_point_request_e2e_mape", "strict_request_e2e_mape_percent"),
        ("macro_point_mean_e2e_ape", "mean_e2e_point_ape_percent"),
        ("macro_point_output_throughput_ape", "output_throughput_point_ape_percent"),
    ]:
        values = [row[source] for row in rows if finite(row[source])]
        result[f"{target}_percent"] = statistics.mean(values) if values else None
        result[f"{target}_point_count"] = len(values)
    walls = [row["cli_wall_time_seconds"] for row in rows if finite(row["cli_wall_time_seconds"])]
    result["sum_completed_cli_wall_seconds"] = sum(walls) if walls else None
    result["completed_cli_wall_point_count"] = len(walls)
    return result


def plot(root: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    x = [row["concurrency"] for row in rows]
    panels = [
        ("Mean E2E latency", "seconds", "mean_e2e_ms", 0.001),
        ("Output throughput", "tokens / second", "output_throughput_tok_s", 1),
    ]
    for axis, (title, unit, metric, scale) in zip(axes.flat, panels, strict=False):
        for source, color, label in [("reference", "#64748b", "Measured"), ("simulated", "#047857", "AISim")]:
            y = [row[f"{source}_{metric}"] * scale if finite(row[f"{source}_{metric}"]) else math.nan for row in rows]
            axis.plot(x, y, "o-", color=color, label=label)
        axis.set(title=title, ylabel=unit)
        axis.legend(frameon=False)
    axes[1, 0].plot(
        x,
        [
            row["strict_request_e2e_mape_percent"] if finite(row["strict_request_e2e_mape_percent"]) else math.nan
            for row in rows
        ],
        "o-",
        color="#2563eb",
    )
    axes[1, 0].set(title="Request E2E MAPE · source + actual output matched", ylabel="percent")
    for field, label, color in [
        ("cli_wall_time_seconds", "Complete CLI process", "#7c3aed"),
        ("native_wall_time_seconds", "Native prediction timer", "#d97706"),
    ]:
        axes[1, 1].plot(
            x, [row[field] if finite(row[field]) else math.nan for row in rows], "o-", label=label, color=color
        )
    axes[1, 1].set(title="CPU replay wall time", ylabel="seconds")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.set(xlabel="AgentX concurrency", xticks=x)
        if x:
            axis.set_xlim(min(x) - 0.5, max(x) + 0.5)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    reused = "; c1 is the previously completed run" if any(row["reused_baseline"] for row in rows) else ""
    fig.suptitle(f"GLM-5.3 NVFP4 · 8x B300 · SGLang · recorded output lengths{reused}", fontsize=12)
    for suffix in ["png", "svg"]:
        fig.savefig(root / f"concurrency-curve.{suffix}", dpi=150, facecolor="white")
    plt.close(fig)


def fmt(value: object, digits: int = 2) -> str:
    return f"{value:,.{digits}f}" if finite(value) else "—"


def render(root: Path, summary: dict) -> None:
    rows = summary["points"]
    body = []
    for row in rows:
        report = (
            f'<a href="{html.escape(row["report_path"], quote=True)}">Interactive report</a>'
            if row["report_path"]
            else html.escape(row["status"])
        )
        label = f"{row['concurrency']}" + (" · reused" if row["reused_baseline"] else "")
        body.append(
            f'<tr><td>{label}<br><a href="{row["benchmark_url"]}">{row["point_id"]}</a></td><td>{report}</td>'
            f"<td>{fmt(row['strict_request_e2e_mape_percent'])}%</td>"
            f"<td>{fmt(row['strict_e2e_pairs'], 0)} / {fmt(row['reference_eligible_e2e_requests'], 0)}"
            f"<br>{fmt(row['strict_e2e_pair_coverage_percent'])}%</td>"
            f"<td>{fmt(row['reference_mean_e2e_ms'])} / {fmt(row['simulated_mean_e2e_ms'])}</td>"
            f"<td>{fmt(row['mean_e2e_point_ape_percent'])}%</td>"
            f"<td>{fmt(row['reference_output_throughput_tok_s'])} / "
            f"{fmt(row['simulated_output_throughput_tok_s'])}</td>"
            f"<td>{fmt(row['output_throughput_point_ape_percent'])}%</td>"
            f"<td>{fmt(row['cli_wall_time_seconds'])} / {fmt(row['native_wall_time_seconds'])}</td>"
            f"<td>{fmt(row['reference_errors'], 0)} / {fmt(row['simulated_failures'], 0)} / "
            f"{fmt(row['simulated_cancellations'], 0)}</td></tr>"
        )
    aggregate_rows = []
    for name, group in summary["aggregates"].items():
        aggregate_rows.append(
            f"<tr><td>{html.escape(name.replace('_', ' '))}</td>"
            f"<td>{group['completed_points']} / {group['selected_points']}</td>"
            f"<td>{fmt(group['pooled_request_e2e_mape_percent'])}% · "
            f"{group['strict_e2e_pair_count']:,} pairs</td>"
            f"<td>{fmt(group['macro_point_request_e2e_mape_percent'])}% · "
            f"{group['macro_point_request_e2e_mape_point_count']} points</td>"
            f"<td>{fmt(group['macro_point_mean_e2e_ape_percent'])}% · "
            f"{group['macro_point_mean_e2e_ape_point_count']} points</td>"
            f"<td>{fmt(group['macro_point_output_throughput_ape_percent'])}% · "
            f"{group['macro_point_output_throughput_ape_point_count']} points</td></tr>"
        )
    notes = "".join(f"<li>{html.escape(text)}</li>" for text in summary["semantics"])
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM concurrency replay</title>
<style>body{{font:15px system-ui;margin:32px;color:#172033;background:#f8fafc}}main{{max-width:1450px;margin:auto}}
h1{{font-size:28px}}a{{color:#0369a1}}img{{width:100%;background:white;border-radius:10px}}
.scroll{{overflow-x:auto;margin:24px 0}}table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:12px;text-align:left;border-bottom:1px solid #e2e8f0;white-space:nowrap}}
th{{font-size:12px;color:#475569}}
li{{margin:8px 0;line-height:1.5}}.muted{{color:#475569}}</style><main>
<h1>GLM concurrency replay</h1><p>GLM-5.3 NVFP4 · 8x B300 · SGLang · recorded profile output lengths</p>
<p class="muted">Complete benchmark replays: preparation, 3,600-second profiling window, grace, and telemetry output.
Concurrency 1 is reused from the earlier run when supplied; every other row is a new replay.</p>
<p><a href="summary.csv">Summary CSV</a> · <a href="summary.json">Summary JSON</a> ·
<a href="concurrency-curve.svg">Vector plot</a></p>
<img src="concurrency-curve.png"
alt="Measured and simulated latency, throughput, request errors and CPU replay wall times by concurrency">
<div class="scroll"><table><thead><tr><th>Concurrency / point</th><th>Replay</th><th>Strict request<br>E2E MAPE</th>
<th>Pairs / eligible<br>coverage</th><th>Mean E2E ms<br>measured / simulated</th><th>Mean E2E<br>point APE</th>
<th>Output tokens/s<br>measured / simulated</th><th>Throughput<br>point APE</th><th>Wall seconds<br>CLI / native</th>
<th>Reference errors /<br>sim failures / cancels</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>
<div class="scroll"><table><thead><tr><th>Scope</th><th>Completed / selected</th><th>Pooled request<br>E2E MAPE</th>
<th>Equal-point mean<br>request E2E MAPE</th><th>Mean E2E<br>point APE macro mean</th>
<th>Throughput<br>point APE macro mean</th></tr></thead><tbody>{"".join(aggregate_rows)}</tbody></table></div>
<ul>{notes}</ul></main></html>"""
    (root / "index.html").write_text(page)


def summarize(root: Path, baseline: Path | None) -> dict:
    selected = json.loads((root / "selected-points.json").read_text())
    if len({int(point["id"]) for point in selected}) != len(selected):
        raise ValueError("Selected point IDs must be unique")
    rows = sorted((load_point(root, point, baseline) for point in selected), key=lambda row: row["concurrency"])
    result = {
        "schema": "aisimulate.agentx.concurrency.v1",
        "points": rows,
        "aggregates": {
            "new_replays": aggregate([row for row in rows if not row["reused_baseline"]]),
            "full_curve_including_reused_baseline": aggregate(rows),
        },
        "semantics": [
            "Strict request E2E MAPE uses unique source identity, successful reference and simulation outcomes, "
            "positive measured latency, and equal actual output length. Input lengths may differ.",
            "Pooled request MAPE weights each point's strict request MAPE by its valid pair count. The equal-point "
            "mean request MAPE gives every scored point equal weight; these are different statistics.",
            "Mean E2E and output-throughput point errors are APE on each full successful profile cohort. Their macro "
            "means give each available concurrency point equal weight. Realized request cohorts may differ.",
            "Throughput uses successful output tokens divided by the span from the first successful arrival to the "
            "last successful completion, including successful grace-period completions.",
            "Full CLI wall time comes from execution metadata and includes startup, supervision, corpus import, "
            "preparation, profiling, grace and artifact writing. Download, transfer, build and chart generation are "
            "excluded. Native wall time is prediction.json wall_time_ms / 1,000 and has a narrower timer scope.",
            "Concurrency 1, when supplied, reuses the earlier replay and contributes to the full-curve aggregate only. "
            "Summed full-curve wall time includes that historical run; it is not elapsed time for this batch.",
            "Missing runs and unavailable metrics remain null. Aggregates include available observations only and "
            "show their denominators. Profile failures and cancellations are retained; unfinished work without a "
            "terminal row is not counted as a completed outcome.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if rows:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {key: json.dumps(value) if isinstance(value, dict) else value for key, value in row.items()}
                for row in rows
            )
    plot(root, rows)
    render(root, result)
    return result


def self_check() -> None:
    base = {
        "status": "completed",
        "reference_eligible_e2e_requests": 10,
        "cli_wall_time_seconds": 5,
        "mean_e2e_point_ape_percent": 20,
        "output_throughput_point_ape_percent": 30,
    }
    first = {**base, "strict_e2e_pairs": 1, "strict_request_e2e_mape_percent": 10}
    second = {**base, "strict_e2e_pairs": 3, "strict_request_e2e_mape_percent": 50}
    result = aggregate([first, second])
    assert result["pooled_request_e2e_mape_percent"] == 40
    assert result["macro_point_request_e2e_mape_percent"] == 30
    assert result["strict_e2e_pair_coverage_percent"] == 20
    assert result["macro_point_mean_e2e_ape_percent"] == 20
    assert result["macro_point_output_throughput_ape_percent"] == 30
    with tempfile.TemporaryDirectory() as temporary:
        missing = load_point(Path(temporary), {"id": 999, "conc": 4}, None)
        assert missing["strict_request_e2e_mape_percent"] is None
        assert missing["cli_wall_time_seconds"] is None
        assert missing["simulated_cancellations"] is None
        result = aggregate([missing])
        assert result["pooled_request_e2e_mape_percent"] is None
        assert result["macro_point_mean_e2e_ape_percent"] is None
        assert result["strict_e2e_pair_coverage_percent"] is None
    assert ape(10, 15) == 50
    assert ape(0, 15) is None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("Self-check passed: pooled vs macro errors, missing observations, and coverage")
    if args.root:
        result = summarize(args.root.resolve(), args.baseline_root.resolve() if args.baseline_root else None)
        print(json.dumps(result["aggregates"], indent=2))
    elif not args.self_check:
        parser.error("--root is required unless --self-check is used")


if __name__ == "__main__":
    main()
