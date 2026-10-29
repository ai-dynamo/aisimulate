# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501 -- Captions and the self-contained HTML template retain literal prose.
"""Render native AISim telemetry and supplied benchmark comparisons offline.

This is an original plotting layout. It does not import InferenceX UI code.
Missing observations remain absent; aggregate comparisons never imply a
request identity match. The measurement JSON supplies the comparison scope.
"""

import argparse
import base64
import html
import json
import math
import textwrap
from collections import Counter
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

COLORS = {"primer": "#a55a05", "warmup": "#df8d22", "profile": "#147d92"}
INK = "#17324d"
BLUE = "#147d92"
ORANGE = "#df8d22"
PURPLE = "#8061b5"
PHASES = ("primer", "warmup", "profile")


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def phase(row):
    value = row.get("phase", "profile")
    return "profile" if value == "profiling" else value


def number(value, digits=3):
    if not finite(value):
        return "Unavailable"
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.{digits}g}"


def empty(ax, message="No observations available"):
    ax.text(
        0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#66788a", fontsize=10, wrap=True
    )


def decorate(ax, title, ylabel=None, xlabel="Absolute virtual time (s)"):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.16, color=INK)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def phase_background(ax, manifest):
    start = manifest.get("profile_start_ms")
    if finite(start) and start > 0:
        ax.axvspan(0, start / 1000, color=ORANGE, alpha=0.09, zorder=-2)
        ax.axvline(start / 1000, color=ORANGE, alpha=0.6, linewidth=0.8, linestyle="--")


def phase_legend(fig):
    handles = [
        Line2D([], [], color=color, marker="o", linestyle="", label=name.title()) for name, color in COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3, fontsize=9)


def save(fig, output, name, title, caption, figures):
    fig.suptitle(title, x=0.055, y=0.99, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.subplots_adjust(
        top=1 - 0.85 / fig.get_figheight(), bottom=0.09, left=0.09, right=0.97, hspace=0.48, wspace=0.45
    )
    fig.savefig(output / f"{name}.png", dpi=150, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white")
    plt.close(fig)
    figures.append({"name": name, "title": title, "caption": caption})


def timeline(requests, lifecycle, manifest, output, figures):
    def conversation(row):
        identity = row.get("agentic") or {}
        return (
            identity.get("lane_id") or "lane:unknown",
            identity.get("conversation_id") or row.get("session_id") or "unassigned",
        )

    keys = {}
    for row in requests:
        keys.setdefault(conversation(row), row["arrival_time_ms"])
    ordered = sorted(keys, key=lambda key: (keys[key], key))
    global_numbers = {key: index + 1 for index, key in enumerate(ordered)}
    panels = []
    folded = 0
    for selected in [{"primer", "warmup"}, {"profile"}]:
        rows = [row for row in requests if phase(row) in selected]
        panel_keys = list(dict.fromkeys(conversation(row) for row in rows))
        detailed = sorted(panel_keys, key=lambda key: (keys[key], key))[:24]
        rest_lanes = sorted({key[0] for key in panel_keys if key not in detailed})
        indices = {key: index for index, key in enumerate(detailed)}
        lane_indices = {lane: len(indices) + index for index, lane in enumerate(rest_lanes)}
        labels = [f"{key[0].replace('lane:', 'L')} · conversation {global_numbers[key]}" for key in detailed]
        labels += [f"{lane.replace('lane:', 'L')} · other conversations" for lane in rest_lanes]
        folded += len(panel_keys) - len(detailed)
        panels.append((rows, indices, lane_indices, labels))
    height = max(5.0, min(10.0, max(len(panel[3]) for panel in panels) * 0.19 + 3))
    fig, axes = plt.subplots(1, 2, figsize=(15, height), gridspec_kw={"width_ratios": [1, 2]})
    split = manifest.get("profile_start_ms") or 0.0
    end = max((row["terminal_time_ms"] for row in requests), default=split)
    for ax, (rows, indices, lane_indices, labels), title, limits in zip(
        axes,
        panels,
        ["Preparation · primer and warmup", "Profile · request arrival → terminal"],
        [(0, max(split, 1) / 1000), (split / 1000, max(end, split + 1) / 1000)],
        strict=True,
    ):
        for current in PHASES:
            segments = []
            for row in rows:
                if phase(row) != current:
                    continue
                key = conversation(row)
                y = indices[key] if key in indices else lane_indices[key[0]]
                segments.append([(row["arrival_time_ms"] / 1000, y), (row["terminal_time_ms"] / 1000, y)])
            ax.add_collection(LineCollection(segments, colors=COLORS[current], linewidths=4, alpha=0.72))
        decorate(ax, title)
        ax.set_xlim(*limits)
        ax.set_ylim(-1, max(len(labels), 1))
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        if not rows:
            empty(ax, "No requests in this phase")
    axes[0].set_ylabel("Lane / conversation (ordered by first arrival)")
    phase_legend(fig)
    events = Counter(row.get("event") for row in lifecycle)
    shown = (
        f" Per panel, the first 24 conversations have individual rows; {folded} remaining conversations are grouped by lane."
        if folded
        else " All conversations have individual rows."
    )
    save(
        fig,
        output,
        "01-request-timeline",
        "Request and conversation timeline",
        "Each segment covers one terminal request. Preparation and profile use different time ranges and conversation rows so preparation remains visible."
        + shown
        + f" Lifecycle: {events.get('dispatch', 0):,} dispatches, {events.get('causal_terminal', 0):,} causal terminals, {events.get('quiescent', 0):,} quiescent requests.",
        figures,
    )


def request_metrics(
    requests,
    manifest,
    output,
    figures,
    *,
    filename="02-request-metrics",
    figure_title="Request shapes and latency over simulated time",
):
    panels = [
        ("input_length", "Input sequence length", "tokens"),
        ("output_length", "Delivered output length", "tokens"),
        ("ttft_ms", "Time to first token", "ms"),
        ("e2e_latency_ms", "End-to-end latency", "ms"),
        ("itl_ms", "Mean inter-token latency / TPOT", "ms / token"),
        ("interactive_tps", "Per-request output token rate", "tokens / s"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for ax, (key, title, unit) in zip(axes.flat, panels, strict=False):
        observed = []
        for current in PHASES:
            points = []
            for row in requests:
                if phase(row) != current:
                    continue
                value = row.get(key)
                if key == "interactive_tps":
                    itl = row.get("itl_ms")
                    value = 1000 / itl if finite(itl) and itl > 0 else None
                if finite(value):
                    points.append((row["arrival_time_ms"] / 1000, value))
            if points:
                x, y = zip(*points, strict=True)
                ax.scatter(x, y, s=11, alpha=0.55, c=COLORS[current], linewidths=0, rasterized=True)
                observed.extend(y)
        decorate(ax, title, unit)
        phase_background(ax, manifest)
        positive = [value for value in observed if value > 0]
        if positive and max(positive) / min(positive) > 100:
            if len(positive) == len(observed):
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=max(min(positive), 1e-3))
                ax.set_ylim(bottom=0)
        if not observed:
            empty(ax)
    phase_legend(fig)
    save(
        fig,
        output,
        filename,
        figure_title,
        "Dots are native per-request observations at arrival time. TPOT uses mean ITL; requests with fewer than two output tokens have no TPOT. Output rate is 1000 / positive TPOT. Missing values are omitted, including unavailable canceled-request latency.",
        figures,
    )


def throughput(samples, manifest, output, figures):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.8))
    panels = [
        ("delivered_output_tokens", "Delivered output throughput", "tokens / s"),
        ("committed_prefill_tokens", "Executed prefill throughput", "tokens / s"),
        ("arriving_requests", "Arriving requests", "requests / s"),
        ("completed_requests", "Completed requests", "requests / s"),
    ]
    for ax, (key, title, unit) in zip(axes.flat, panels, strict=False):
        any_points = False
        for current, color in [("preparation", ORANGE), ("profile", BLUE)]:
            points = []
            for row in samples:
                duration = row["traffic"].get("duration_s")
                value = row["traffic"].get(key)
                if row.get("phase") == current and finite(duration) and duration > 0 and finite(value):
                    points.append((row["sampled_at_ms"] / 1000, value / duration))
            if points:
                x, y = zip(*points, strict=True)
                ax.plot(x, y, color=color, linewidth=1.2, marker=".", markersize=3, label=current.title())
                any_points = True
        decorate(ax, title, unit)
        phase_background(ax, manifest)
        if not any_points:
            empty(ax, "No positive-duration intervals")
        else:
            ax.legend(frameon=False, fontsize=9)
    save(
        fig,
        output,
        "03-throughput",
        "Throughput from native interval counters",
        "Rates divide interval counters by the exact interval duration. Prefill includes completed recomputation; delivered output respects output limits and cancellation. Baselines and zero-duration tails are omitted from rates, but remain in exported counters. Points are interval ends.",
        figures,
    )


def plot_optional(ax, x, values, label, **kwargs):
    if any(finite(value) for value in values):
        kwargs.setdefault("marker", ".")
        kwargs.setdefault("markersize", 2)
        ax.plot(x, [value if finite(value) else np.nan for value in values], label=label, **kwargs)
        return True
    return False


def scheduler_cache(
    samples,
    manifest,
    output,
    figures,
    *,
    filename="04-scheduler-cache",
    figure_title="Scheduler, cache, and speculative decoding",
):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    x = [row["sampled_at_ms"] / 1000 for row in samples]
    gauges = [row.get("prefill_scheduler_metrics", []) + row.get("decode_scheduler_metrics", []) for row in samples]
    ax = axes[0, 0]
    for field, label, color in [
        ("running_requests", "Running", BLUE),
        ("waiting_requests", "Scheduler waiting", ORANGE),
    ]:
        plot_optional(ax, x, [sum(gauge[field] for gauge in rows) for rows in gauges], label, color=color)
    plot_optional(
        ax,
        x,
        [
            row.get("router_pending_prefill_requests", 0) + row.get("router_pending_decode_requests", 0)
            for row in samples
        ],
        "Router pending",
        color=PURPLE,
    )
    decorate(ax, "Request queues", "requests")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    keys = sorted(
        {
            (role, gauge["worker_id"], gauge["dp_rank"])
            for row in samples
            for role in ["prefill", "decode"]
            for gauge in row.get(f"{role}_scheduler_metrics", [])
        }
    )
    for role, worker, rank in keys:
        values = [
            next(
                (
                    100 * gauge["physical_cache_usage"]
                    for gauge in row.get(f"{role}_scheduler_metrics", [])
                    if gauge["worker_id"] == worker and gauge["dp_rank"] == rank
                ),
                None,
            )
            for row in samples
        ]
        plot_optional(ax, x, values, f"{role} W{worker} R{rank}", linewidth=1.1)
    decorate(ax, "GPU KV resident usage", "% of rank capacity")
    if keys:
        ax.legend(frameon=False, fontsize=7, ncol=min(3, len(keys)))
    else:
        empty(ax)
    ax = axes[1, 0]
    host_rows = [row.get("host_caches", []) for row in samples]
    host_keys = sorted(
        {(row.get("role") or "shared", row.get("worker_id"), row.get("dp_rank")) for rows in host_rows for row in rows},
        key=str,
    )
    for key in host_keys:
        values = []
        for rows in host_rows:
            cache = next(
                (row for row in rows if (row.get("role") or "shared", row.get("worker_id"), row.get("dp_rank")) == key),
                None,
            )
            values.append(
                100 * cache["used_bytes"] / cache["capacity_bytes"]
                if cache and cache.get("capacity_bytes", 0) > 0
                else None
            )
        plot_optional(ax, x, values, str(key), linewidth=1.1)
    decorate(ax, "Host KV used capacity", "% of pool capacity")
    if host_keys:
        ax.legend(frameon=False, fontsize=7)
    else:
        empty(ax, "No modeled host cache in this run")
    ax = axes[1, 1]
    found = False
    for role in ["prefill", "decode"]:
        native, scheduler = [], []
        for row in samples:
            counters = row.get(f"{role}_interval_metrics", {})
            lookup = counters.get("native_prefix_cache_lookup") or {}
            query = lookup.get("queried_tokens")
            native.append(100 * lookup["matched_tokens"] / query if finite(query) and query > 0 else None)
            total = counters.get("cache_total_tokens")
            scheduler.append(
                100 * counters["cache_hit_tokens"] / total
                if finite(total) and total > 0 and finite(counters.get("cache_hit_tokens"))
                else None
            )
        found |= plot_optional(ax, x, native, f"{role}: native G1 lookup", linewidth=1.2)
        found |= plot_optional(ax, x, scheduler, f"{role}: scheduler reuse", linewidth=1.2, linestyle="--")
    decorate(ax, "Prefix reuse observations", "% of queried / considered tokens")
    if found:
        ax.legend(frameon=False, fontsize=8)
    else:
        empty(ax, "No native lookup or scheduler denominator")
    ax = axes[2, 0]
    for field, label, color in [
        ("hbm_reused_tokens", "HBM reuse", BLUE),
        ("host_reused_tokens", "Host reuse", PURPLE),
        ("uncached_tokens", "Uncached", ORANGE),
    ]:
        values = []
        for row in samples:
            counts = row["traffic"].get("first_admission", {})
            total = counts.get("input_tokens")
            values.append(100 * counts[field] / total if finite(total) and total > 0 else None)
        plot_optional(ax, x, values, label, color=color, linewidth=1.2)
    decorate(ax, "Prompt source at first admission", "% of admitted prompt tokens")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2, 1]
    found = plot_optional(
        ax,
        x,
        [row["traffic"].get("avg_accept_length") for row in samples],
        "Accepted tokens / decode forward",
        color=PURPLE,
        linewidth=1.4,
    )
    decorate(ax, "Speculative acceptance length", "tokens, including base token")
    if not found:
        empty(ax, "No decode acceptance observations")
    for ax in axes.flat:
        phase_background(ax, manifest)
        if x:
            ax.set_xlim(0, max(x[-1], 0.001))
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]]:
        ax.set_ylim(0, 100)
    save(
        fig,
        output,
        filename,
        figure_title,
        "Gauges show settled sampler state. Aggregated serving uses the native decode-role fields. Native G1 lookup counts include retries; scheduler reuse and first-admission source accounting have different denominators and are not interchangeable. Shared host pools are counted once. Missing denominators and unavailable metrics are omitted.",
        figures,
    )


def distributions(requests, reference, output, figures):
    panels = [
        ("input_length", "input_tokens", "Input sequence length", "tokens"),
        ("output_length", "output_tokens", "Output sequence length", "tokens"),
        ("ttft_ms", "ttft_ms", "Time to first token", "ms"),
        ("e2e_latency_ms", "e2e_ms", "End-to-end latency", "ms"),
        ("itl_ms", "tpot_ms", "Mean inter-token latency / TPOT", "ms / token"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for ax, (sim_key, ref_key, title, unit) in zip(axes.flat, panels, strict=False):
        values = []
        for rows, key, label, color in [
            (requests, sim_key, "Simulation", BLUE),
            (reference, ref_key, "Reference", ORANGE),
        ]:
            observed = sorted(
                row[key]
                for row in rows
                if phase(row) == "profile"
                and finite(row.get(key))
                and not row.get("error")
                and row.get("terminal_status", "completed") == "completed"
            )
            if observed:
                ax.plot(
                    observed,
                    np.arange(1, len(observed) + 1) / len(observed) * 100,
                    label=f"{label} (n={len(observed):,})",
                    color=color,
                    linewidth=1.5,
                )
                values.extend(observed)
        decorate(ax, title, "Cumulative share (%)", unit)
        positive = [value for value in values if value > 0]
        if positive and max(positive) / min(positive) > 100:
            ax.set_xscale("symlog", linthresh=min(positive))
        if values:
            ax.legend(frameon=False, fontsize=8)
        else:
            empty(ax)
    axes[2, 1].axis("off")
    axes[2, 1].text(0.02, 0.9, "Independent profile cohorts", fontsize=14, fontweight="bold", color=INK)
    axes[2, 1].text(
        0.02,
        0.76,
        "These CDFs compare successful profile observations.\nThey do not establish per-request identity matching.\n\nWarmup, errors, canceled requests, and missing metrics\nare excluded. Cohort sizes can differ. Use the supplied\ncomparison table for the declared accuracy scope.",
        fontsize=11,
        color=INK,
        va="top",
        linespacing=1.6,
    )
    save(
        fig,
        output,
        "05-reference-distributions",
        "Simulation and reference profile distributions",
        "Empirical cumulative distributions, independently filtered by metric. Every curve preserves its own count. This descriptive comparison does not calculate matched-request MAPE or assert identical realized workloads.",
        figures,
    )


def comparison_rows(measurement):
    rows = measurement.get("comparisons", [])
    if isinstance(rows, dict):
        return [{"metric": key, **value} for key, value in rows.items() if isinstance(value, dict)]
    return [row for row in rows if isinstance(row, dict)]


def comparison_plot(rows, output, figures):
    valid = [
        row for row in rows if finite(row.get("reference")) and row["reference"] > 0 and finite(row.get("simulated"))
    ]
    errors = [row for row in rows if finite(row.get("mape_percent"))]
    if not valid and not errors:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6.0, 0.42 * max(len(valid), len(errors)) + 2)))
    if valid:
        axes[0].barh(
            range(len(valid)), [100 * row["simulated"] / row["reference"] for row in valid], color=BLUE, alpha=0.85
        )
        axes[0].set_yticks(
            range(len(valid)), [textwrap.fill(row.get("metric", "Metric"), 24) for row in valid], fontsize=9
        )
        axes[0].axvline(100, color=ORANGE, linestyle="--", linewidth=1.5, label="Reference = 100%")
        axes[0].invert_yaxis()
        axes[0].legend(frameon=False, fontsize=9)
    else:
        empty(axes[0])
    decorate(axes[0], "Simulated value relative to reference", xlabel="% of reference")
    if errors:
        axes[1].barh(range(len(errors)), [row["mape_percent"] for row in errors], color=PURPLE, alpha=0.85)
        axes[1].set_yticks(
            range(len(errors)), [textwrap.fill(row.get("metric", "Metric"), 24) for row in errors], fontsize=9
        )
        axes[1].invert_yaxis()
    else:
        empty(axes[1], "No supplied MAPE results")
    decorate(axes[1], "Supplied comparison errors", xlabel="MAPE / aggregate APE (%) — see scope table")
    save(
        fig,
        output,
        "06-comparisons",
        "Latency and throughput comparison",
        "Values and error estimates come from measurement.json. Metric units, sample counts, and comparison scope are stated in the table. A single aggregate comparison is an absolute percentage error, not a request-level MAPE.",
        figures,
    )


def render_html(
    output, figures, prediction, measurement, requests, reference, lifecycle, samples, run_dir, reference_path
):
    manifest = prediction["telemetry"]
    title = measurement.get("title", "AgentX-inspired replay analysis · AISimulate")
    wall = measurement.get("wall_time_seconds", measurement.get("elapsed_wall_seconds"))
    if not finite(wall):
        wall = prediction.get("wall_time_ms", 0) / 1000 if finite(prediction.get("wall_time_ms")) else None
    duration = prediction.get("duration_ms")
    cards = [
        ("Terminal requests", f"{len(requests):,}"),
        ("Profile completed", number(prediction.get("completed_requests"))),
        ("Profile simulated time", f"{number(duration / 1000 if finite(duration) else None)} s"),
        ("Measured run wall time", f"{number(wall)} s"),
        ("Output throughput", f"{number(prediction.get('output_throughput_tok_s'))} tok/s"),
        ("Capture status", manifest.get("completion_status", "Unavailable")),
    ]
    esc = lambda value: html.escape(str(value))
    cards_html = "".join(
        f"<div class='card'><span>{esc(label)}</span><strong>{esc(value)}</strong></div>" for label, value in cards
    )
    figures_html = "".join(
        f"<figure id='{item['name']}'><h2>{esc(item['title'])}</h2><img alt='{esc(item['title'] + '. ' + item['caption'])}' src='data:image/png;base64,{base64.b64encode((output / (item['name'] + '.png')).read_bytes()).decode()}'><figcaption>{esc(item['caption'])}</figcaption></figure>"
        for item in figures
    )
    rows = comparison_rows(measurement)
    body = "".join(
        f"<tr><th>{esc(row.get('metric', ''))}</th><td>{esc(row.get('unit', ''))}</td><td>{number(row.get('reference'))}</td><td>{number(row.get('simulated'))}</td><td>{number(row.get('mape_percent'))}</td><td>{esc(row.get('count', row.get('n', '—')))}</td><td>{esc(row.get('scope', 'See measurement provenance'))}</td></tr>"
        for row in rows
    )
    comparison_html = (
        (
            "<section><h2>Comparison values</h2><div class='table-wrap'><table><thead><tr><th>Metric</th><th>Unit</th><th>Reference</th><th>Simulated</th><th>Error (%)</th><th>Count</th><th>Scope</th></tr></thead><tbody>"
            + body
            + "</tbody></table></div></section>"
        )
        if rows
        else "<section><h2>Comparison values</h2><p>No comparison measurements were supplied. The descriptive distributions do not substitute for an accuracy calculation.</p></section>"
    )
    notes = measurement.get("methodology", [])
    if isinstance(notes, str):
        notes = [notes]
    limitations = measurement.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    notes = list(notes)
    notes += [
        "Plots use native AISim virtual-clock telemetry; wall time is the measured host process duration, shown separately.",
        "Preparation and profile use the same absolute clock. Orange shading ends at the profile barrier; preparation rates are separate from profile rates.",
        "Null values are unavailable observations and are omitted. They are never converted to measured zeros. Positive-duration intervals alone define sampled rates.",
        "The request file contains terminal outcomes. A time-limited run can have unfinished work visible only in gauges and its completion status.",
        "The renderer consumes supplied comparison results; it does not infer per-request matching from the shape of two distributions.",
        "No server logs or power data are displayed.",
    ]
    methods = "".join(f"<li>{esc(note)}</li>" for note in notes)
    limits = "".join(f"<li>{esc(note)}</li>" for note in limitations)
    benchmark = measurement.get("benchmark_url")
    link = (
        f"<a href='{esc(benchmark)}'>Reference benchmark</a> · "
        if isinstance(benchmark, str) and benchmark.startswith("https://")
        else ""
    )
    provenance = {
        "run_directory": str(run_dir.resolve()),
        "reference_requests": str(reference_path.resolve()),
        "telemetry": manifest,
        "measurement": measurement,
        "record_counts_read": {
            "requests": len(requests),
            "samples": len(samples),
            "lifecycle": len(lifecycle),
            "reference_requests": len(reference),
        },
    }
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title>
<style>:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fa;color:{INK};font:16px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1450px;margin:auto;padding:34px 24px 70px}}h1{{font-size:32px;line-height:1.2;margin:8px 0 12px}}h2{{font-size:22px;margin:0 0 12px}}.eyebrow{{font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{BLUE}}}.lede{{max-width:950px;color:#486176}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px;margin:24px 0}}.card,section,figure{{background:white;border:1px solid #dce5ed;border-radius:12px}}.card{{padding:18px}}.card span{{display:block;font-size:12px;color:#587083}}.card strong{{display:block;font-size:22px;margin-top:5px}}section,figure{{padding:24px;margin:22px 0}}figure img{{width:100%;height:auto;display:block}}figcaption{{font-size:14px;color:#52697d;padding-top:12px}}a{{color:#126f87}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e2e9ef;vertical-align:top}}thead{{background:#f1f6f9}}.table-wrap{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}li{{margin:7px 0}}@media print{{body{{background:white}}figure,section{{break-inside:avoid;border:0}}}}</style>
<main><header><div class="eyebrow">Native AISim telemetry · standalone evidence report</div><h1>{esc(title)}</h1><p class="lede">Phase-aware workload, latency, throughput, scheduling, and cache observations from the simulated run. Reference comparisons retain their declared scope.</p><p>{link}PNG and SVG plots are saved beside this self-contained report.</p></header><div class="cards">{cards_html}</div>{comparison_html}{figures_html}<section><h2>Methods and reading guide</h2><ul>{methods}</ul>{"<h2>Known limitations</h2><ul>" + limits + "</ul>" if limits else ""}</section><section><h2>Provenance</h2><details><summary>Inputs, units, phase boundaries, and supplied measurements</summary><pre>{esc(json.dumps(provenance, indent=2))}</pre></details></section></main></html>"""
    (output / "report.html").write_text(document)
    (output / "analysis-summary.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prediction = read_json(args.run_dir / "prediction.json")
    manifest = prediction.get("telemetry")
    if not isinstance(manifest, dict) or manifest.get("clock") != "absolute_virtual_ms":
        raise ValueError("Expected native telemetry with an absolute_virtual_ms clock")
    requests = read_rows(args.run_dir / "telemetry-requests.jsonl")
    samples = read_rows(args.run_dir / "telemetry-samples.jsonl")
    lifecycle = read_rows(args.run_dir / "telemetry-lifecycle.jsonl")
    reference = read_json(args.reference)
    measurement = read_json(args.measurement)
    if not isinstance(reference, list) or not isinstance(measurement, dict):
        raise ValueError("Expected reference request array and measurement object")
    if any(
        not finite(row.get("arrival_time_ms"))
        or not finite(row.get("terminal_time_ms"))
        or row["arrival_time_ms"] > row["terminal_time_ms"]
        for row in requests
    ):
        raise ValueError("Request timestamp contract violated")
    if any(right["sampled_at_ms"] < left["sampled_at_ms"] for left, right in pairwise(samples)):
        raise ValueError("Telemetry samples must be ordered by virtual time")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )
    figures = []
    timeline(requests, lifecycle, manifest, args.output_dir, figures)
    request_metrics(requests, manifest, args.output_dir, figures)
    throughput(samples, manifest, args.output_dir, figures)
    scheduler_cache(samples, manifest, args.output_dir, figures)
    distributions(requests, reference, args.output_dir, figures)
    comparison_plot(comparison_rows(measurement), args.output_dir, figures)
    preparation_requests = [row for row in requests if phase(row) in {"primer", "warmup"}]
    preparation_samples = [row for row in samples if row.get("phase") == "preparation"]
    if preparation_requests:
        request_metrics(
            preparation_requests,
            manifest,
            args.output_dir,
            figures,
            filename="07-preparation-request-metrics",
            figure_title="Preparation only · request shapes and latency",
        )
    if preparation_samples:
        scheduler_cache(
            preparation_samples,
            manifest,
            args.output_dir,
            figures,
            filename="08-preparation-scheduler-cache",
            figure_title="Preparation only · scheduler, cache, and speculation",
        )
    render_html(
        args.output_dir,
        figures,
        prediction,
        measurement,
        requests,
        reference,
        lifecycle,
        samples,
        args.run_dir,
        args.reference,
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.html"),
                "figures": len(figures),
                "requests": len(requests),
                "samples": len(samples),
            }
        )
    )


if __name__ == "__main__":
    main()
