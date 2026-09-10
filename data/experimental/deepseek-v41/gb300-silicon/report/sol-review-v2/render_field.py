# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify supplementary observations against admitted inputs and report both errors."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from render import check_observations
from replay import MODES, NATIVE_SHA, TOOLS_REF, read, sha

METRICS = (
    "ttft_ms",
    "average_tpot_ms",
    "exact_itl_ms",
    "output_tokens_per_second",
    "request_latency_ms",
    "last_token_latency_ms",
    "native_forward_interval_ms",
)
LABELS = {
    "ttft_ms": "HTTP TTFT (ms)",
    "average_tpot_ms": "HTTP mean TPOT (ms)",
    "output_tokens_per_second": "HTTP output tokens/s",
    "native_forward_interval_ms": "Native interval (ms)",
}


def metrics(error_summary, profile, mode, http, trace, *, purpose=None):
    rows = []
    for metric in METRICS:
        pairs = []
        if metric == "native_forward_interval_ms":
            pairs = [r for cohort in trace for r in cohort["intervals"]]
            weighting = "MAPE: equal correlated native intervals; WAPE: observed interval latency weighted"
        else:
            for cohort in http:
                if metric not in cohort["observed"]:
                    continue
                pair = {"observed_ms": cohort["observed"][metric], "status": "prediction_unavailable"}
                if cohort["status"] == "predicted" and metric in cohort["prediction"]:
                    pair.update(
                        status="predicted",
                        predicted_ms=cohort["prediction"][metric],
                        signed_error_percent=cohort["signed_error_percent"][metric],
                    )
                pairs.append(pair)
            weighting = (
                "MAPE: equal supported HTTP cohort/trial pairs; WAPE: observed-value weighted within this metric"
            )
        summary = error_summary(pairs)
        supported = [r for r in pairs if r["status"] == "predicted"]
        rows.append(
            summary
            | {
                "scope": profile,
                "mode": mode.upper(),
                "metric": metric,
                "purpose": purpose,
                "weighting": weighting,
                "observed_mean_supported": statistics.mean(r["observed_ms"] for r in supported) if supported else None,
                "predicted_mean_supported": statistics.mean(r["predicted_ms"] for r in supported)
                if supported
                else None,
            }
        )
    return rows


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified", type=Path, required=True)
    for profile in ("off", "on"):
        parser.add_argument(f"--{profile}-raw", type=Path, required=True)
        parser.add_argument(f"--{profile}-content", type=Path, required=True)
    args = parser.parse_args()
    repo = next(p for p in root.parents if (p / "Cargo.toml").is_file())
    core = read(root / "summary.json")
    core_inventory = read(root / "artifact-hashes.json")["files_sha256"]
    for name, digest in core_inventory.items():
        if sha(root / name) != digest:
            raise ValueError("original core artifact changed")
    summaries, scenarios, preservation, coverage, content_summary, bindings = [], [], [], {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="dsv41-field-render-") as temporary:
        temporary = Path(temporary)
        sources = read(root / "off/replay-provenance.json")["analysis_source_sha256"]
        for path, digest in sources.items():
            data = subprocess.check_output(["git", "show", f"{TOOLS_REF}:{path}"], cwd=repo)
            destination = temporary / Path(path).name
            destination.write_bytes(data)
            if sha(destination) != digest:
                raise ValueError("frozen analysis helper differs")
        sys.path.insert(0, str(temporary))
        forward = importlib.import_module("compare_forward")
        trace_helper = importlib.import_module("compare_trace")
        http_helper = importlib.import_module("compare_e2e")
        for profile in ("off", "on"):
            target = root / f"field-{profile}"
            qualified = args.qualified / f"field-{profile}"
            raw = getattr(args, f"{profile}_raw")
            paths = {
                "audit": qualified / "audit.json",
                "measurement": qualified / "measurement.json",
                "plan": raw / "strata/field-notes/main-plan.json",
                "client-summary": raw / "strata/field-notes/main-client/summary.json",
                "scheduler-receipt": raw / "scheduler-receipt.json",
            }
            provenance = read(target / "replay-provenance.json")
            program = root / ("replay_field.py" if profile == "off" else "replay_field_on.py")
            if provenance["replay_program_sha256"] != sha(program):
                raise ValueError("supplement replay source changed")
            if provenance["native_sha256"] != NATIVE_SHA or provenance["analysis_commit"] != TOOLS_REF:
                raise ValueError("prediction identity differs")
            receipt_path = qualified / "supplemental-scope-receipt.json"
            if sha(receipt_path) != provenance["scope_admission_sha256"]:
                raise ValueError("closed lifetime admission differs")
            receipt = read(receipt_path)
            for name, digest in receipt["qualified_output_sha256"].items():
                if sha(qualified / name) != digest:
                    raise ValueError("qualified source no longer matches admission")
            for job in provenance["jobs"]:
                if sha(target / job["output"]) != job["output_sha256"]:
                    raise ValueError("comparison output changed")
                for name, digest in job["input_sha256"].items():
                    if sha(paths[name]) != digest:
                        raise ValueError("admitted input bytes changed")
                if sha(target / job["prediction_config"]) != job["prediction_config_sha256"]:
                    raise ValueError("prediction config changed")
            audit, plan, measurement = (read(paths[k]) for k in ("audit", "plan", "measurement"))
            audited = trace_helper.qualify_inputs(audit, plan, measurement)
            if plan["requested_trials"] != 30 or measurement["decoder_replay"] != (profile == "on"):
                raise ValueError("supplement profile or frozen trial count differs")
            http_helper.qualify_e2e_sources(measurement, paths["client-summary"], paths["scheduler-receipt"])
            client = {c["cohort_id"]: c for c in read(paths["client-summary"])["cohorts"]}
            # Re-extract observation fields only. The constant callback is discarded;
            # no model prediction or measured value is generated during validation.
            expected_trace = trace_helper.compare_cohorts(
                audit,
                plan,
                audited,
                lambda _: 1.0,
                backend="sglang",
                forward_model="op_level",
                decoder_replay=(profile == "on"),
            )
            expected_http = []
            for case in plan["cohorts"]:
                key = case["cohort_id"]
                if case["comparison_role"] != "primary" or key not in audited:
                    continue
                expected_http.append(
                    {
                        **{k: case[k] for k in ("cohort_id", "purpose", "trial_index", "trial_seed")},
                        "observed": http_helper.observed_metrics(client[key]),
                        "native_initial_cached_tokens": {
                            r["request_id"]: r["initial_cross_request_cached_tokens"]
                            for r in audited[key]["native_request_proof"]["requests"]
                        },
                    }
                )
            if len(expected_http) != 120:
                raise ValueError("expected 30 independent trials with four primary scenarios")
            for mode in MODES:
                http = read(target / f"e2e-{mode}-results.json.gz")
                trace = read(target / f"trace-{mode}-results.json.gz")
                for kind, actual, expected in (
                    ("http", http["cohorts"], expected_http),
                    ("trace", trace["cohorts"], expected_trace),
                ):
                    preservation.append(
                        check_observations(actual, expected, kind)
                        | {
                            "scope": f"field-{profile}",
                            "mode": mode,
                            "reference": "unchanged admitted client and native audit inputs",
                        }
                    )
                new_metrics = metrics(
                    forward.error_summary, f"field-{profile}", mode, http["cohorts"], trace["cohorts"]
                )
                if any(new_metrics[-1][key] != trace["summary"][key] for key in ("mape_percent", "wape_percent")):
                    raise ValueError("native aggregate metric differs from frozen helper output")
                summaries.extend(new_metrics)
                for purpose in sorted({c["purpose"] for c in expected_http}):
                    scenarios.extend(
                        metrics(
                            forward.error_summary,
                            f"field-{profile}",
                            mode,
                            [c for c in http["cohorts"] if c["purpose"] == purpose],
                            [c for c in trace["cohorts"] if c["purpose"] == purpose],
                            purpose=purpose,
                        )
                    )
                intervals = [r for c in trace["cohorts"] for r in c["intervals"]]
                coverage[f"field-{profile}-{mode}"] = {
                    "independent_trials": 30,
                    "primary_scenarios": 4,
                    "observed_http_cohorts": len(http["cohorts"]),
                    "observed_native_intervals": len(intervals),
                    "missing_http_predictions": sum(c["status"] != "predicted" for c in http["cohorts"]),
                    "missing_native_predictions": sum(r["status"] != "predicted" for r in intervals),
                    "cache_semantics_disagreements": sum(
                        c.get("cache_semantics_match") is False for c in http["cohorts"]
                    ),
                    "http_failure_types": dict(
                        Counter(c["failure_type"] for c in http["cohorts"] if c["status"] != "predicted")
                    ),
                    "native_failure_types": dict(
                        Counter(r["failure_type"] for r in intervals if r["status"] != "predicted")
                    ),
                }
            diagnostic_path = getattr(args, f"{profile}_content")
            diagnostic = read(diagnostic_path)
            if diagnostic["logical_plan_sha256"] != sha(paths["plan"]) or not diagnostic["content_coverage_complete"]:
                raise ValueError("content diagnostic is not bound to the admitted plan")
            if diagnostic["logical_run_id"] != plan["run_id"] or diagnostic["missing_cohort_ids"]:
                raise ValueError("content diagnostic run or completeness differs")
            (target / "content-diagnostics.json").write_bytes(diagnostic_path.read_bytes())
            (target / "scope-admission.json").write_bytes(receipt_path.read_bytes())
            content_summary[f"field-{profile}"] = diagnostic["summary"]
            bindings[f"field-{profile}"] = {
                "input_sha256": {k: sha(p) for k, p in paths.items()},
                "admission_sha256": sha(receipt_path),
                "content_diagnostic_sha256": sha(diagnostic_path),
                "comparison_provenance_sha256": sha(target / "replay-provenance.json"),
                "replay_program_sha256": sha(program),
            }
    payload = {
        "schema": "dsv41.corrected-sol-field-verification.v1",
        "correction_fitting": False,
        "model_commit": core["model_commit"],
        "analysis_commit": TOOLS_REF,
        "native_sha256": NATIVE_SHA,
        "core_summary_sha256": sha(root / "summary.json"),
        "source_bindings": bindings,
        "observation_preservation": preservation,
        "coverage_and_cache": coverage,
        "metrics": summaries,
        "per_scenario_metrics": scenarios,
        "content_controls": content_summary,
        "metric_definition": core["metric_definition"],
        "uncertainty": (
            "30 independent trials per profile within separate frozen resource windows; paired "
            "scenario CIs remain in exact comparison outputs. No pooled CI across profiles or corpora "
            "and no claim of 5% precision."
        ),
    }
    (root / "field-summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    consolidated = {
        "schema": "dsv41.corrected-sol-consolidated-verification.v1",
        "model_commit": core["model_commit"],
        "analysis_commit": TOOLS_REF,
        "native_sha256": NATIVE_SHA,
        "source_summary_sha256": {name: sha(root / name) for name in ("summary.json", "field-summary.json")},
        "metrics": core["metrics"] + summaries,
        "qualification": "Distinct scopes remain separate. This is an inventory of metric rows, not a pooled estimate.",
    }
    (root / "consolidated-summary.json").write_text(json.dumps(consolidated, indent=2) + "\n")
    fields = [
        "scope",
        "mode",
        "purpose",
        "metric",
        "planned_points",
        "predicted_points",
        "observed_mean_supported",
        "predicted_mean_supported",
        "mape_percent",
        "wape_percent",
        "weighting",
    ]
    with (root / "field-comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries + scenarios)
    write_readme(root, payload)
    draw(root, summaries)
    paths = [
        root / n
        for n in (
            "replay_field.py",
            "replay_field_on.py",
            "render_field.py",
            "field-summary.json",
            "consolidated-summary.json",
            "field-comparison.csv",
            "field-README.md",
            "field-comparison.png",
            "field-comparison.pdf",
        )
    ]
    paths.extend(p for profile in ("off", "on") for p in (root / f"field-{profile}").iterdir() if p.is_file())
    inventory = {str(p.relative_to(root)): sha(p) for p in sorted(paths)}
    (root / "field-artifact-hashes.json").write_text(json.dumps({"files_sha256": inventory}, indent=2) + "\n")


def write_readme(root, payload):
    lines = [
        "# GB300 TP4 field corpus: separate supplementary OFF and ON strata",
        "",
        (
            "Each profile retains 30 independent trials in its own closed runtime lifecycle and frozen "
            "resource window: four primary scenarios, 120 HTTP cohorts. They cover boundary-129, "
            "heldout-single-192, repeated text, and distinct text. The 10 pilot trials per profile are "
            "excluded from these accuracy aggregates. No heldout fitting or modification of the "
            "original observations occurred."
        ),
        "",
        (
            "The corrected model, native extension, and versioned replicated-indexer tables match the "
            "[core report](README.md). The frozen analysis helper version is identical. Neither corpus "
            "nor replay mode is pooled. Per-scenario paired bootstrap intervals remain in the exact "
            "compressed comparison outputs, conditional on each runtime lifecycle and frozen "
            "calibration. These descriptive aggregates do not claim the original 5% precision target."
        ),
        "",
        (
            "MAPE weights supported cohort/trial pairs equally; WAPE weights by observed value within "
            "the metric. Native intervals remain correlated and are not independent trials. Means and "
            "both errors use identical supported pairs; missing predictions remain in coverage "
            "denominators."
        ),
        "",
        "| Replay | Metric | Mode | Covered | Real mean | Predicted mean | MAPE | WAPE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["metrics"]:
        if row["metric"] not in LABELS:
            continue
        lines.append(
            f"| {row['scope'].removeprefix('field-').upper()} | {LABELS[row['metric']]} | {row['mode']} | "
            f"{row['predicted_points']}/{row['planned_points']} | {row['observed_mean_supported']:.3f} | "
            f"{row['predicted_mean_supported']:.3f} | {row['mape_percent']:.2f}% | {row['wape_percent']:.2f}% |"
        )
    lines += [
        "",
        "![Supplementary field errors with both metrics](field-comparison.png)",
        "",
        "## Coverage and content controls",
        "",
    ]
    for scope, counts in payload["coverage_and_cache"].items():
        lines.append(
            f"- {scope}: {counts['missing_http_predictions']} missing HTTP cohorts; "
            f"{counts['missing_native_predictions']} missing native intervals out of "
            f"{counts['observed_native_intervals']}; {counts['cache_semantics_disagreements']} "
            "predicted cohorts disagree with observed initial cache reuse."
        )
    lines += [
        "",
        (
            "Both repeated-text and distinct-text pairs were cold with respect to observed initial KV "
            "reuse in all 30 trials per profile. Repeated-text sequences and raw 3-gram sets match; "
            "distinct-text pairs differ. Cold pairs arrived in the same first-prefill dispatch in only "
            "21/30 repeated and 16/30 distinct OFF trials, and 20/30 repeated and 17/30 distinct ON "
            "trials. Their decode schedules can also differ. These controls establish input content "
            "and native batching facts; raw n-grams do not establish compressed Engram addresses, "
            "cache hits, or a causal Engram latency effect. KV reset does not flush Engram/HBM/L2 "
            "state."
        ),
        "",
        (
            "[field-summary.json](field-summary.json) includes all metrics, four-scenario breakdowns, "
            "observation-preservation proofs, coverage, source hashes, and content summaries. "
            "[field-comparison.csv](field-comparison.csv) provides the same aggregate/scenario "
            "comparisons. [consolidated-summary.json](consolidated-summary.json) inventories the "
            "unchanged 48 core metric rows and 42 separate supplementary rows for PR reporting."
        ),
        "",
        "## Reproduction",
        "",
        (
            "Use the same native extension, source import paths, and model checkout described in the "
            "core report. Retain each original raw root beside its closed-lifetime admission and "
            "normalized audit. `replay_field.py` uses OFF data; `replay_field_on.py` uses ON data. "
            "Both refuse to overwrite existing output scopes. For OFF, run from a fresh copy of this "
            "report tree after omitting its `field-off` directory; ON also accepts `--output-dir`."
        ),
        "",
        "```bash",
        "python replay_field.py --qualified /path/to/qualified/field-off --raw /path/to/original-field-off",
        "python replay_field_on.py --qualified /path/to/qualified/field-on --raw /path/to/original-field-on",
        (
            "python render_field.py --qualified /path/to/qualified --off-raw "
            "/path/to/original-field-off --on-raw /path/to/original-field-on --off-content "
            "/path/to/field-off-content.json --on-content /path/to/field-on-content.json"
        ),
        "```",
        "",
        (
            "The renderer checks all original core artifact hashes, then independently reconstructs "
            "observation fields from the admitted client/audit inputs and compares every metric and "
            "native interval geometry. It does not fit or regenerate model predictions. Public outputs "
            "retain only source hashes, portable identities, and sanitized observations; internal "
            "cluster files remain separate."
        ),
    ]
    (root / "field-README.md").write_text("\n".join(lines) + "\n")


def draw(root, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for i, profile in enumerate(("field-off", "field-on")):
        for j, metric in enumerate(LABELS):
            ax = axes[i, j]
            rows = [r for r in summaries if r["scope"] == profile and r["metric"] == metric]
            for offset, key, color, label in (
                (-0.18, "mape_percent", "#2374ab", "MAPE"),
                (0.18, "wape_percent", "#d96c06", "WAPE"),
            ):
                ax.bar([k + offset for k in range(3)], [r[key] for r in rows], width=0.34, label=label, color=color)
            ax.set_xticks(range(3), [r["mode"] for r in rows], fontsize=8)
            ax.set_yscale("log")
            ax.set_ylabel("Absolute error (%)")
            ax.set_title(profile.upper() + " · " + LABELS[metric], fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            if i == j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("GB300 TP4 · supplementary field corpus · 30 independent trials per replay profile")
    fig.savefig(root / "field-comparison.png", dpi=180)
    fig.savefig(root / "field-comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
