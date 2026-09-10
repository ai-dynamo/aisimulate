# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit frozen forward comparisons and render them without fitting predictions."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / "Cargo.toml").is_file())
DATA = ROOT.parent.parent / "prefix-refinement-v1"
PROFILES = {"full": "Decoder OFF", "decoder_bounded": "Decoder ON"}
MODES = ("sol", "hybrid", "silicon")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def close(left, right):
    if not math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-10):
        raise ValueError(f"numeric mismatch: {left} != {right}")


def summarize(rows):
    supported = [r for r in rows if r["status"] == "predicted"]
    errors = [100 * (r["predicted_ms"] / r["observed_ms"] - 1) for r in supported]
    absolute = sorted(abs(e) for e in errors)
    position = 0.9 * (len(absolute) - 1)
    index = int(position)
    p90 = absolute[index] + (position - index) * (absolute[min(index + 1, len(absolute) - 1)] - absolute[index])
    return {
        "planned_points": len(rows),
        "predicted_points": len(supported),
        "mean_signed_error_percent": statistics.mean(errors),
        "median_absolute_error_percent": statistics.median(absolute),
        "p90_absolute_error_percent_across_configurations": p90,
        "wape_percent": 100
        * sum(abs(r["predicted_ms"] - r["observed_ms"]) for r in supported)
        / sum(r["observed_ms"] for r in supported),
    }


def audit(analysis_dir):
    import aisimulate._runtime as native
    from aiconfigurator_core.sdk import utils

    model_name = "deepseek-ai/DeepSeek-V4.1-Flash"
    checkpoint_path = utils._get_model_config_path() / f"{model_name.replace('/', '--')}_config.json"
    checkpoint = read(checkpoint_path)
    resolved = utils.get_model_config_from_model_path(model_name)["raw_config"]
    canonical_sha = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    actual_model = {
        "checkpoint_config_file_sha256": sha(checkpoint_path),
        "checkpoint_config_canonical_sha256": canonical_sha(checkpoint),
        "resolved_config_canonical_sha256": canonical_sha(resolved),
    }

    source_files = {
        "model_python_sha256": REPO / "python/aisimulate/src/aiconfigurator_core/sdk/models/deepseek_v41.py",
        "engine_python_sha256": REPO / "python/aisimulate/src/aiconfigurator_core/sdk/engine.py",
        "prediction_adapter_sha256": REPO / "python/aisimulate/src/aiconfigurator_core/sdk/rust_engine_step.py",
        "model_resolution_sha256": REPO / "python/aisimulate/src/aiconfigurator_core/sdk/utils.py",
        "native_extension_sha256": Path(native.__file__),
        "comparison_analysis_sha256": analysis_dir / "compare_forward.py",
        "statistics_analysis_sha256": analysis_dir / "analyze_e2e.py",
        "axis_normalization_sha256": analysis_dir / "normalize_fpm.py",
    }
    source_hashes = {key: sha(path) for key, path in source_files.items()}
    plan = read(DATA / "heldout-plan.json")
    frozen = read(DATA / "plan-receipt.json")
    for relative, digest in frozen["preserved_inputs_sha256"].items():
        if sha(DATA.parent / relative) != digest:
            raise ValueError("an original artifact changed")
    reports = {}
    checked = 0
    for profile in PROFILES:
        observation_path = DATA / profile / "heldout/forward-results.json"
        observed = read(observation_path)
        if (observed["status"], observed["case_count"], observed["iterations"], observed["warmup"]) != (
            "accepted",
            46,
            10,
            1,
        ):
            raise ValueError("independent holdout was not completely admitted")
        if observed["component_recorder"] or observed["used_cuda_graph"]:
            raise ValueError("wrong independent timing scope")
        if observed["execution_profile"] != profile or observed["source_sha256"] != frozen["source_sha256"]:
            raise ValueError("observed profile or native producer source mismatch")
        for mode in MODES:
            report = read(ROOT / f"{profile}-{mode}-results.json")
            config_path = ROOT / f"{profile}-{mode}-config.json"
            config = read(config_path)
            expected = {
                "model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
                "system_name": "gb300",
                "backend": "sglang",
                "backend_version": "0.0.0.dev0",
                "tp_size": 4,
                "pp_size": 1,
                "moe_tp_size": 4,
                "moe_ep_size": 1,
                "attention_dp_size": 1,
                "database_mode": mode.upper(),
                "decoder_replay": profile == "decoder_bounded",
                "enable_shared_layer": False,
                "strict_provenance": True,
            }
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError("prediction configuration differs from the qualified contract")
            if set(config) != set(expected) | {"schema_version", "systems_path"} or config["schema_version"] != 1:
                raise ValueError("unexpected configuration fields or precision overrides")
            systems = (REPO / config["systems_path"]).resolve()
            if systems != (DATA / profile / "systems").resolve():
                raise ValueError("prediction used another overlay")
            if report["prediction_sources"] != source_hashes:
                raise ValueError("prediction source files differ from the actual imported/code artifacts")
            if report["observation_sha256"] != sha(observation_path):
                raise ValueError("wrong observed attempt")
            if report["heldout_plan_sha256"] != sha(DATA / "heldout-points.json"):
                raise ValueError("wrong frozen workload manifest")
            if report["prediction_config_sha256"] != sha(config_path):
                raise ValueError("prediction configuration changed")
            if report["execution_profile"] != profile or report["database_mode"] != mode.upper():
                raise ValueError("profile or mode mismatch")
            if report["correction_fitting"] or report["forward_model"] != "op_level":
                raise ValueError("report is not the unfitted op-level estimator")
            if report["observed_target"] != observed["timing_boundary"]:
                raise ValueError("timing boundary mismatch")
            model = report["resolved_model_identity"]
            if any(model.get(key) != value for key, value in actual_model.items()):
                raise ValueError("model identity differs from the actual checkpoint and SDK resolution")
            if model["checkpoint_config_canonical_sha256"] != frozen["config_sha256"]:
                raise ValueError("checkpoint configuration mismatch")
            if (
                model["resolved_config_canonical_sha256"]
                != "22c3912140adeb60ecbd3c7a9b54e62997adf65ad4c69440f8082cf29f5ef0d4"
            ):
                raise ValueError("effective model precision changed")
            identity = report["systems_identity"]
            files = {str(p.relative_to(systems)): sha(p) for p in systems.rglob("*") if p.is_file()}
            if files != identity["files_sha256"]:
                raise ValueError("actual systems data differ from prediction inputs")
            inventory = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if inventory != identity["inventory_sha256"]:
                raise ValueError("systems inventory hash differs")
            if sha(systems / "gb300.yaml") != identity["system_spec_sha256"]:
                raise ValueError("system specification changed")
            for row, source, geometry in zip(report["cases"], observed["cases"], plan["cases"], strict=True):
                if any(row.get(key) != value for key, value in source.items()):
                    raise ValueError("a reported observation differs from its admitted raw aggregate")
                if any(row.get(key) != value for key, value in geometry.items()):
                    raise ValueError("a geometry differs from the frozen holdout")
                if len(row["rank_max_ms"]) != 10 or any(v <= 0 or not math.isfinite(v) for v in row["rank_max_ms"]):
                    raise ValueError("invalid repeat inventory")
                close(row["observed_ms"], statistics.median(row["rank_max_ms"]))
                b, q, prefix = row["batch_size"], row["query"], row["prefix"]
                context = row["phase"] == "context"
                scheduling = row["prediction_input"]["scheduled_requests"]
                expected_scheduling = {
                    "num_prefill_requests": b if context else 0,
                    "sum_prefill_tokens": b * q if context else 0,
                    "sum_prefill_kv_tokens": b * prefix if context else 0,
                    "num_decode_requests": 0 if context else b,
                    "sum_decode_kv_tokens": 0 if context else b * (prefix + 1),
                    "var_prefill_length": 0.0,
                    "var_decode_kv_tokens": 0.0,
                }
                if scheduling != expected_scheduling:
                    raise ValueError("logical workload was not correctly bridged to the native query")
                bridge = row["axis_bridge"]
                if bridge["producer_semantics"] != "sglang_inclusive_query" or bridge["delta"] != 0:
                    raise ValueError("unexpected decode-axis conversion")
                if not context and (row["canonical_past_kv"] != prefix or row["native_inclusive_kv"] != prefix + 1):
                    raise ValueError("decode off-by-one")
                if row["status"] != "predicted" or not math.isfinite(row["predicted_ms"]) or row["predicted_ms"] <= 0:
                    raise ValueError("a prediction is missing or invalid")
                close(row["signed_error_percent"], 100 * (row["predicted_ms"] / row["observed_ms"] - 1))
                checked += 1
            for key, value in summarize(report["cases"]).items():
                close(report["summary"][key], value)
            for phase in ("context", "generation"):
                for key, value in summarize([r for r in report["cases"] if r["phase"] == phase]).items():
                    close(report["by_phase"][phase][key], value)
            reports[profile, mode] = report
        for left, right in zip(reports[profile, "hybrid"]["cases"], reports[profile, "silicon"]["cases"], strict=True):
            close(left["predicted_ms"], right["predicted_ms"])
    return reports, {"status": "passed", "prediction_rows_checked": checked, "source_hashes": source_hashes}


def figure(reports):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6), layout="constrained")
    for axis, (profile, label) in zip(axes[0], PROFILES.items(), strict=True):
        measured = reports[profile, "silicon"]["cases"]
        x = [r["observed_ms"] for r in measured]
        low = [r["observed_ms"] - min(r["rank_max_ms"]) for r in measured]
        high = [max(r["rank_max_ms"]) - r["observed_ms"] for r in measured]
        axis.errorbar(
            x,
            [r["predicted_ms"] for r in measured],
            xerr=[low, high],
            fmt="o",
            markersize=4,
            color="#147D92",
            alpha=0.8,
            elinewidth=0.7,
            label="HYBRID = SILICON",
        )
        axis.scatter(
            x,
            [r["predicted_ms"] for r in reports[profile, "sol"]["cases"]],
            marker="^",
            s=22,
            color="#8961AE",
            label="SOL",
        )
        end = 1.06 * max(max(x), max(r["predicted_ms"] for r in measured), max(max(r["rank_max_ms"]) for r in measured))
        axis.plot([0, end], [0, end], "--", color="#9BA3AC", linewidth=1)
        axis.set(
            xlim=(0, end),
            ylim=(0, end),
            xlabel="Observed native forward (ms)",
            ylabel="Predicted forward (ms)",
            title=f"{label}: 46 / 46 supported",
        )
        axis.grid(alpha=0.15)
        axis.legend(loc="upper left", frameon=False)
    axis = axes[1, 0]
    for i, (profile, label) in enumerate(PROFILES.items()):
        values = [reports[profile, "silicon"]["by_phase"][phase]["wape_percent"] for phase in ("context", "generation")]
        bars = axis.bar([i * 3, i * 3 + 1], values, color=["#147D92", "#D08634"])
        axis.bar_label(bars, fmt="%.2f%%", padding=4)
    axis.set(
        xticks=[0, 1, 3, 4],
        xticklabels=["OFF prefill", "OFF decode", "ON prefill", "ON decode"],
        ylabel="WAPE (%)",
        ylim=(0, 15),
        title="Phase errors: HYBRID = SILICON",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis = axes[1, 1]
    rows = reports["full", "silicon"]["cases"]
    worst = max(rows, key=lambda r: statistics.stdev(r["rank_max_ms"]) / statistics.mean(r["rank_max_ms"]))
    axis.plot(range(1, 11), worst["rank_max_ms"], "o-", color="#BB573F", markersize=4)
    axis.axhline(worst["observed_ms"], linestyle="--", color="#596677", label=f"Median {worst['observed_ms']:.2f} ms")
    axis.set(
        xlabel="Measured repetition",
        ylabel="Native forward (ms)",
        xticks=range(1, 11),
        title="All repeats retained: OFF B1 / Q80 / P128",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.15)
    fig.suptitle("DeepSeek V4.1 Flash · GB300 TP4 · Prefix refinement v1", fontsize=15, fontweight="bold")
    fig.savefig(ROOT / "forward-comparison.png", dpi=180)
    fig.savefig(ROOT / "forward-comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def write_readme(reports):
    lines = [
        "# GB300 prefix refinement: independent forward validation",
        "",
        "Strict SILICON and HYBRID predict **46/46 fresh holdouts in both Decoder profiles**. "
        "Their predictions are identical on this set. WAPE is **7.23% with Decoder OFF** and "
        "**4.85% with Decoder ON**. This is native benchmark-forward validation; it does not establish HTTP "
        "or serving FPM accuracy. One OFF repetition was much slower and is retained below.",
        "",
        "![Independent forward results](forward-comparison.png)",
        "",
        "Points compare the median native forward with prediction. Horizontal whiskers span all ten measured "
        "repetitions, not confidence intervals. The line denotes exact agreement. The lower panels separate "
        "prefill/decode error and expose the largest repeat variation.",
        "",
        "## Accuracy and common support",
        "",
        "| Profile | Mode | Coverage | Mean signed error | Median APE | p90 APE | WAPE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (profile, mode), report in reports.items():
        s = report["summary"]
        lines.append(
            f"| {PROFILES[profile]} | {mode.upper()} | {s['predicted_points']}/{s['planned_points']} | "
            f"{s['mean_signed_error_percent']:+.2f}% | {s['median_absolute_error_percent']:.2f}% | "
            f"{s['p90_absolute_error_percent_across_configurations']:.2f}% | {s['wape_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "All three modes share the same 46 supported configurations per profile; the common-support "
            "statistics therefore equal the table above. No missing point was dropped from a denominator. "
            "Strict SILICON disables shared-source fallback. Existing empirical embedding, normalization, "
            "activation and memory operations remain empirical, so a successful strict query is not a claim "
            "that every forward cost was measured. SOL's roughly 96% underprediction against this native "
            "wall interval is shown explicitly; it is not a validated wall-latency forecast.",
            "",
            "| Profile | Phase | Mode | Coverage | Mean signed error | Median APE | WAPE |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for profile in PROFILES:
        for phase in ("context", "generation"):
            for mode in ("sol", "silicon"):
                s = reports[profile, mode]["by_phase"][phase]
                label = "SOL" if mode == "sol" else "HYBRID / SILICON"
                lines.append(
                    f"| {PROFILES[profile]} | {'Prefill' if phase == 'context' else 'Decode'} | {label} | "
                    f"{s['predicted_points']}/{s['planned_points']} | {s['mean_signed_error_percent']:+.2f}% | "
                    f"{s['median_absolute_error_percent']:.2f}% | {s['wape_percent']:.2f}% |"
                )
    lines.extend(
        [
            "",
            "The 36 prefill and 10 decode points are also reported separately: decode is underpredicted "
            "by about 10-12% despite the smaller combined WAPE. Signed error is `100*(prediction/observation-1)`. "
            "APE percentiles weight each logical configuration equally; WAPE is "
            "`100*sum(abs(prediction-observation))/sum(observation)`. The p90 column is a percentile across "
            "configurations, not request tail latency. Each observation is the median of ten invocation "
            "times, each invocation first taking the maximum across four TP ranks. There is no fitted correction.",
            "",
            "## Repeat stability",
            "",
            "| Profile | Median within-point CV | Maximum CV | Worst configuration |",
            "|---|---:|---:|---|",
        ]
    )
    for profile in PROFILES:
        rows = reports[profile, "silicon"]["cases"]
        cvs = [100 * statistics.stdev(r["rank_max_ms"]) / statistics.mean(r["rank_max_ms"]) for r in rows]
        worst = rows[cvs.index(max(cvs))]
        lines.append(
            f"| {PROFILES[profile]} | {statistics.median(cvs):.2f}% | {max(cvs):.2f}% | "
            f"{worst['case_id']}: B{worst['batch_size']}, Q{worst['query']}, P{worst['prefix']} |"
        )
    off = next(r for r in reports["full", "silicon"]["cases"] if r["case_id"] == "prefill-0004")
    lines.extend(
        [
            "",
            "OFF B1/Q80/P128 measured (ms): " + ", ".join(f"{v:.3f}" for v in off["rank_max_ms"]) + ". "
            "The 498.046 ms invocation remains in the raw evidence and the plot; the median is 201.893 ms. "
            "No repetition was removed or replaced. CV uses the sample standard deviation divided by the mean "
            "of the ten rank maxima. Low median CV does not establish stability across runtime lifecycles, "
            "and the cause of this slow invocation has not been established.",
            "",
            "## Why these samples",
            "",
            "This separate study was frozen before measurement: **18 new bounded calibration configurations** "
            "(one warmup plus three measured invocations each), and **46 new holdouts per profile** "
            "(one warmup plus ten measured invocations each). Thus there are 92 fresh profile/configuration "
            "pairs, 974 measured invocations per rank (`18*3+92*10`), and 110 warmups. TP ranks, repetitions, "
            "and physical module keys are not independent workload configurations.",
            "",
            "The count 18 is a conditional minimum under the existing exact-prefix consumer and fixed target: "
            "ten absent `(batch, effective late-layer prefix)` curves exposed by the old bounded holdouts, "
            "plus eight different curves required by the new Q130 target. One homogeneous calibration "
            "configuration can supply only one of these curves for its batch. This is not a universal minimum "
            "for V4.1 or a statistical sample-size guarantee. Ten holdout repetitions improve within-case "
            "precision; this bounded grid does not support a population-level accuracy confidence interval.",
            "",
            "The 46 whole-forward geometries per profile are disjoint from old and new calibration and old "
            "holdouts. Their forward-only runs contain no component recorder and contribute no fitted rows. "
            "Explicitly audited pilot reuse adds only attention at B1/P256/Q3 or Q128; formal same-key rows "
            "always win. The final module tables contain 848 OFF and 948 ON keys, plus the unchanged 80 "
            "baseline keys per profile. See the "
            "[frozen design and admission receipts](../../prefix-refinement-v1/README.md).",
            "",
            "These are different holdouts from the original 38-point study. Comparing their percentages does "
            "not isolate the benefit of calibration refinement. The [original three-repeat report](../README.md) "
            "and [38-point precision repeat](../precision-v2/README.md) remain unchanged, including their missing "
            "ON predictions and variable observations.",
            "",
            "## Identity, scope, and reproduction",
            "",
            "The target is the original SGLang synchronized one-batch wall interval, including preparation, "
            "all layers, shared Engram hashing, forward and sampling. It differs from GPU-timed serving FPM "
            "and HTTP latency. Runtime is SGLang 0.0.0.dev0, pinned ARM image `800cc9ad…`, text-only TP4/EP1, "
            "DP1/PP1, eager prefill/decode, sharded shared experts and explicit unfused NCCL. Decoder ON uses "
            "the verified current-extend tail semantics. The checkpoint and effective quantization are unchanged. "
            "For decode, canonical past KV K is seeded and native inclusive K+1 is predicted, "
            "without a second increment.",
            "",
            "The domain remains homogeneous B1/B2, at most 512 total new prefill tokens and native context "
            "at most 2048. This does not qualify heterogeneous/B3 requests, arbitrary exact-prefix buckets, "
            "longer contexts, new corpora, different kernels, scheduling or repeated-text cache semantics. "
            "No model formula or correction was changed for these results.",
            "",
            "Every result JSON binds the observed attempt, frozen point manifest, configuration, checkpoint, "
            "resolved model, actual source files and complete system-overlay hashes. The renderer independently "
            "checks all 276 predictions against admitted observations and plan geometry, recomputes statistics, "
            "checks source and data hashes, and verifies original artifact preservation. `artifact-hashes.json` "
            "binds this review and its outputs. To regenerate with the source-verified FPM comparison tools "
            "from the companion PR available at `$ANALYSIS_DIR`:",
            "",
            "```bash",
            "export PYTHONPATH=python/aisimulate/src:python/aisimulate",
            "python data/experimental/deepseek-v41/gb300-silicon/report/prefix-refinement-v1/render_report.py "
            '--analysis-dir "$ANALYSIS_DIR"',
            "```",
            "",
        ]
    )
    (ROOT / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    reports, result = audit(args.analysis_dir)
    figure(reports)
    write_readme(reports)
    result["files_sha256"] = {
        p.name: sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"
    }
    (ROOT / "artifact-hashes.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "prediction_rows_checked": result["prediction_rows_checked"]}))


if __name__ == "__main__":
    main()
