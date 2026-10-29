# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare recorded native predictions and plot them without estimating timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"Run is incomplete: {root}")
    rows = [
        json.loads(line)
        for line in (root / "native-queries.jsonl").read_text().splitlines()
    ]
    keyed = {
        (r["variant"], r["nextn"], r["batch_size"], r["context_length"]): r
        for r in rows
    }
    if len(keyed) != len(rows) or len(rows) != manifest["record_count"]:
        raise ValueError(f"Duplicate coordinates or record-count mismatch: {root}")
    return manifest, keyed


def compare(before: dict, after: dict) -> dict:
    if before.keys() != after.keys():
        raise ValueError("Native comparison grids differ")
    checks = {
        "non_attention_unchanged": 0,
        "all_full_unchanged": 0,
        "batch1_nextn0_unchanged": 0,
    }
    changes = []
    for key, old in before.items():
        new = after[key]
        old_ops = {op["name"]: op["latency_ms"] for op in old["operations"]}
        new_ops = {op["name"]: op["latency_ms"] for op in new["operations"]}
        if old_ops.keys() != new_ops.keys():
            raise ValueError(f"Operation keys differ: {key}")
        non_attention = [name for name in old_ops if name != "generation_attention"]
        if any(old_ops[name] != new_ops[name] for name in non_attention):
            raise ValueError(f"Non-attention operation changed: {key}")
        checks["non_attention_unchanged"] += 1
        if key[0] == "all_full":
            if old_ops != new_ops:
                raise ValueError(f"All-full attention changed: {key}")
            checks["all_full_unchanged"] += 1
        if key[1:3] == (0, 1):
            if old_ops != new_ops:
                raise ValueError(f"Measured-domain batch1/nextn0 changed: {key}")
            checks["batch1_nextn0_unchanged"] += 1
        if old["cycle_latency_ms"] != new["cycle_latency_ms"]:
            changes.append(
                {
                    "variant": key[0],
                    "nextn": key[1],
                    "batch_size": key[2],
                    "context_length": key[3],
                    "baseline_cycle_ms": old["cycle_latency_ms"],
                    "corrected_cycle_ms": new["cycle_latency_ms"],
                    "baseline_attention_ms": old["attention_latency_ms"],
                    "corrected_attention_ms": new["attention_latency_ms"],
                }
            )
    return {
        "scope": "Native prediction comparison; this is not a silicon-accuracy result",
        "compared_records": len(before),
        "checks": checks,
        "changed_records": len(changes),
        "changes": changes,
    }


def save(fig: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(
            output / f"{name}.{suffix}", dpi=180, facecolor="white", bbox_inches="tight"
        )
    plt.close(fig)


def predicted_panel(
    ax, runs: list[tuple[str, dict]], variant: str, nextn: int, batch: int, field: str
) -> None:
    for (label, rows), color, style, marker in zip(
        runs, ("#b55b20", "#12678a"), ("--", "-"), ("o", "x"), strict=True
    ):
        points = sorted(
            (key[3], row[field])
            for key, row in rows.items()
            if key[:3] == (variant, nextn, batch)
        )
        if not points:
            raise ValueError(f"Missing plot coordinate: {(variant, nextn, batch)}")
        ax.plot(
            [x / 1000 for x, _ in points],
            [y for _, y in points],
            label=label,
            color=color,
            linestyle=style,
            marker=marker,
            linewidth=2,
            markersize=5,
        )
    ax.set_xlabel("Past context (thousand tokens)")
    ax.set_xticks(
        [x / 1000 for x, _ in points], [f"{x / 1000:g}" for x, _ in points], rotation=30
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)


def plot_replay_comparison(root: Path) -> None:
    source = root / "comparison/comparison.json"
    if not source.exists():
        return
    comparison = json.loads(source.read_text())
    if comparison["schema"] != "aisimulate.glm_decode_slope_comparison.v1":
        raise ValueError("Unknown replay comparison schema")
    points = sorted(comparison["points"], key=lambda point: point["concurrency"])
    c1 = [point for point in points if point["concurrency"] == 1]
    if len(c1) != 1 or len({point["concurrency"] for point in points}) != len(points):
        raise ValueError(
            "Replay comparison needs one c1 point and unique concurrency values"
        )
    bins = c1[0]["cohort_comparison"]["common_cohort_input_bins"]
    labels, measured, baseline, corrected = [], [], [], []
    for group in bins:
        old, new = group["baseline"]["tpot"], group["candidate"]["tpot"]
        if (
            old["count"] != new["count"]
            or old["reference_mean_ms"] != new["reference_mean_ms"]
        ):
            raise ValueError("C1 TPOT bins do not share the same reference cohort")
        lower = group["reference_input_tokens_min_inclusive"] / 1000
        upper = group["reference_input_tokens_max_exclusive"]
        interval = f"{lower:g}k+" if upper is None else f"{lower:g}–{upper / 1000:g}k"
        labels.append(f"{interval}\nn={old['count']}")
        measured.append(old["reference_mean_ms"])
        baseline.append(old["simulated_mean_ms"])
        corrected.append(new["simulated_mean_ms"])
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.suptitle("Concurrency 1: request TPOT by measured input length", fontsize=14)
    for values, label, color, style in (
        (measured, "Measured", "#414a50", "-"),
        (baseline, "AISim baseline", "#b55b20", "--"),
        (corrected, "AISim corrected", "#12678a", "-"),
    ):
        ax.plot(
            range(len(labels)),
            values,
            marker="o",
            label=label,
            color=color,
            linestyle=style,
            linewidth=2,
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set(
        xlabel="Reference input-length bin (decimal: 1k = 1,000 tokens)",
        ylabel="Mean request TPOT (ms)",
        ylim=(0, None),
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    total = sum(group["baseline"]["tpot"]["count"] for group in bins)
    fig.supxlabel(
        f"Same {total} matched requests; equal recorded output lengths. Input lengths are not required to match.",
        fontsize=10,
    )
    save(fig, root, "c1-tpot-input-bins")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    fig.suptitle("GLM replay error against measured serving results", fontsize=14)
    concurrency = [point["concurrency"] for point in points]
    for ax, metric, title in zip(
        axes[:2],
        ("tpot", "e2e"),
        ("Request TPOT MAPE", "Request E2E MAPE"),
        strict=True,
    ):
        for variant, label, color, style in (
            ("baseline", "Baseline", "#b55b20", "--"),
            ("candidate", "Corrected", "#12678a", "-"),
        ):
            values = [
                point["cohort_comparison"]["common_cohort"][variant][metric][
                    "mape_percent"
                ]
                for point in points
            ]
            for point in points:
                cohort = point["cohort_comparison"]["common_cohort"]
                if (
                    cohort["baseline"][metric]["count"]
                    != cohort["candidate"][metric]["count"]
                ):
                    raise ValueError(f"Request cohorts differ for {metric}")
            ax.plot(
                concurrency,
                values,
                marker="o",
                label=label,
                color=color,
                linestyle=style,
                linewidth=2,
            )
        ax.set(title=title, ylabel="Mean absolute percentage error (%)")
    for variant, label, color, style in (
        ("baseline", "Baseline", "#b55b20", "--"),
        ("candidate", "Corrected", "#12678a", "-"),
    ):
        values = []
        for point in points:
            matches = [
                row
                for row in point[f"{variant}_full_profile_comparisons"]
                if row["metric"].startswith("Output throughput")
            ]
            if len(matches) != 1 or matches[0]["count"] != 1:
                raise ValueError(
                    "Output throughput must be one full-profile aggregate per point"
                )
            values.append(matches[0]["mape_percent"])
        axes[2].plot(
            concurrency,
            values,
            marker="o",
            label=label,
            color=color,
            linestyle=style,
            linewidth=2,
        )
    axes[2].set(
        title="Output throughput APE", ylabel="Aggregate absolute percentage error (%)"
    )
    for ax in axes:
        ax.set_xticks(concurrency)
        ax.set_xlabel("Agent concurrency")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.supxlabel(
        "Request panels: common matched cohorts. Throughput: one aggregate per run, using each run's realized full profile.",
        fontsize=10,
    )
    save(fig, root, "replay-errors-by-concurrency")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    baseline_manifest, before = read_run(args.root / "baseline-static")
    corrected_manifest, after = read_run(args.root / "corrected-static")
    result = compare(before, after)
    result["native_sha256"] = {
        "baseline": baseline_manifest["native"]["sha256"],
        "corrected": corrected_manifest["native"]["sha256"],
    }
    if len(set(result["native_sha256"].values())) != 2:
        raise ValueError("Baseline and corrected native binaries are identical")
    runs = [("Baseline", before), ("Corrected", after)]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), sharey="row", constrained_layout=True
    )
    fig.suptitle(
        "Native predicted attention — synthetic layer-reuse ablations, one request",
        fontsize=14,
    )
    for row, variant in enumerate(("all_full", "all_skip")):
        for col, nextn in enumerate((0, 3)):
            ax = axes[row, col]
            predicted_panel(ax, runs, variant, nextn, 1, "attention_latency_ms")
            ax.set_title(
                f"{'All full-indexer' if row == 0 else 'All reused-index'} layers; nextn={nextn}"
            )
            ax.set_ylabel("Attention per timing round (ms)")
        maximum = max(max(line.get_ydata()) for ax in axes[row] for line in ax.lines)
        axes[row, 0].set_ylim(0, maximum * 1.08)
    fig.supxlabel(
        "Includes native layer/MTP scaling. Native predictions, not GPU truth; round latency is not TPOT.",
        fontsize=10,
    )
    save(fig, args.root, "attention-ablations")

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), sharey="row", constrained_layout=True
    )
    fig.suptitle("Native predicted GLM-5.3 whole-round latency", fontsize=14)
    for row, batch in enumerate((1, 14)):
        for col, nextn in enumerate((0, 3)):
            ax = axes[row, col]
            predicted_panel(ax, runs, "checkpoint", nextn, batch, "cycle_latency_ms")
            ax.set_title(
                f"{batch} scheduled request{'s' if batch > 1 else ''}; nextn={nextn}"
            )
            ax.set_ylabel("Generation timing round (ms)")
        maximum = max(max(line.get_ydata()) for ax in axes[row] for line in ax.lines)
        axes[row, 0].set_ylim(0, maximum * 1.08)
    fig.supxlabel(
        "Original checkpoint layer mix. No scheduling or acceptance conversion; native predictions, not GPU truth.",
        fontsize=10,
    )
    save(fig, args.root, "checkpoint-rounds")

    measured = [
        json.loads(line)
        for line in (args.root / "baseline-static/measured-dsa-rows.jsonl")
        .read_text()
        .splitlines()
    ]
    if (args.root / "baseline-static/measured-dsa-rows.jsonl").read_bytes() != (
        args.root / "corrected-static/measured-dsa-rows.jsonl"
    ).read_bytes():
        raise ValueError("Original measured DSA rows changed")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.suptitle(
        "Original measured DSA modules — SGLang 0.5.14, B300, eight local heads",
        fontsize=14,
    )
    for ax, op, label in zip(
        axes,
        ("dsa_generation_module", "dsa_generation_module_skip_indexer"),
        ("Full indexer", "Reused index"),
        strict=True,
    ):
        for batch in (1, 4, 8):
            rows = sorted(
                (r["isl"] + r["step"], r["latency"])
                for r in measured
                if r["op_name"] == op and r["batch_size"] == batch
            )
            ax.plot(
                [x / 1000 for x, _ in rows],
                [y for _, y in rows],
                marker=".",
                label=f"Measured batch {batch}",
            )
        ax.set(
            title=label,
            xlabel="Measured sequence length (thousand tokens)",
            ylabel="One module (ms)",
        )
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.supxlabel(
        "FP8 KV / BF16 projections. Lines connect measured rows only; these are not full-model MTP measurements.",
        fontsize=10,
    )
    save(fig, args.root, "measured-dsa-modules")
    (args.root / "static-comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    plot_replay_comparison(args.root)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "changes"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
