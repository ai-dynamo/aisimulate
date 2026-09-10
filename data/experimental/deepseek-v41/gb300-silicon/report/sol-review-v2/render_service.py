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


def supplementary_addendum(root, args):
    target = root / "service-addendum"
    target.mkdir(exist_ok=True)
    precision, paired = {}, {}
    for corpus, directory in (("field", "field-notes"), ("service", "service-records")):
        for profile in ("off", "on"):
            raw = getattr(args, f"field_{profile}_raw" if corpus == "field" else f"{profile}_raw")
            stats_path = raw / "strata" / directory / "main-statistics.json"
            statistics = read(stats_path)
            comparison = read(root / f"{corpus}-{profile}" / "e2e-silicon-results.json.gz")
            if statistics["points"] != comparison["client_analysis"]["points"]:
                raise ValueError("observed precision does not match unchanged client observations")
            if statistics["requested_trials"] != 30 or not statistics["client_coverage_complete"]:
                raise ValueError("observed precision has incomplete or changed trial budget")
            points = []
            for purpose, metrics in sorted(statistics["points"].items()):
                for metric in ("ttft_ms", "average_tpot_ms", "output_tokens_per_second"):
                    item = metrics[metric]
                    if item["independent_trials"] != 30:
                        raise ValueError("observed precision lost original independent trials")
                    points.append(
                        {
                            "purpose": purpose,
                            "metric": metric,
                            "observed_mean": item["mean"],
                            "mean_bootstrap_ci95": item["mean_bootstrap_ci95"],
                            "relative_half_width_percent": 100 * item["relative_half_width"],
                            "met_target": item["relative_half_width"] <= 0.05,
                        }
                    )
            if len(points) != 12:
                raise ValueError("predeclared observed precision target set changed")
            precision[f"{corpus}-{profile}"] = {
                "original_statistics_sha256": sha(stats_path),
                "comparison_sha256": sha(root / f"{corpus}-{profile}" / "e2e-silicon-results.json.gz"),
                "trials": 30,
                "required_targets": 12,
                "met_targets": sum(r["met_target"] for r in points),
                "metrics": points,
                "timing_filter_applied": False,
            }
        source = getattr(args, f"{corpus}_output_check")
        document = read(source)
        if (document["paired_trial_count"], document["paired_cohort_count"], document["paired_request_count"]) != (
            30,
            120,
            180,
        ):
            raise ValueError("paired returned-output scope changed")
        if document["sampling_exclusion"] is not False or document["timing_observations_unchanged"] is not True:
            raise ValueError("output checks cannot filter timing observations")
        if set(document["requests_by_status"]) - {"equal", "different"}:
            raise ValueError("paired returned-output evidence has missing requests")
        if (
            sum(document["requests_by_status"].values()) != 180
            or len(document["requests"]) != 180
            or document["corpus_role"] != directory
        ):
            raise ValueError("paired output counts or corpus identity differ")
        for profile in ("off", "on"):
            raw = getattr(args, f"field_{profile}_raw" if corpus == "field" else f"{profile}_raw")
            for name, path in {
                "main-plan": raw / "strata" / directory / "main-plan.json",
                "main-summary": raw / "strata" / directory / "main-client/summary.json",
            }.items():
                if sha(path) != document["input_files_sha256"][profile + ":" + name]:
                    raise ValueError("paired output check detached from frozen service/field observations")
        (target / f"{corpus}-paired-outputs.json").write_bytes(source.read_bytes())
        paired[corpus] = {
            k: document[k]
            for k in (
                "paired_trial_count",
                "paired_cohort_count",
                "paired_request_count",
                "requests_by_status",
                "sampling_exclusion",
                "timing_observations_unchanged",
                "qualification",
                "limitations",
            )
        } | {"original_check_sha256": sha(source)}
    result = {
        "schema": "dsv41.supplemental.observed-precision-output-addendum.v1",
        "observed_precision": precision,
        "paired_outputs": paired,
        "original_core_and_field_reports_modified": False,
    }
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified", type=Path, required=True)
    for profile in ("off", "on"):
        parser.add_argument(f"--{profile}-raw", type=Path, required=True)
        parser.add_argument(f"--{profile}-content", type=Path, required=True)
    for profile in ("off", "on"):
        parser.add_argument(f"--field-{profile}-raw", type=Path, required=True)
    for corpus in ("field", "service"):
        parser.add_argument(f"--{corpus}-output-check", type=Path, required=True)
    args = parser.parse_args()
    repo = next(p for p in root.parents if (p / "Cargo.toml").is_file())
    core = read(root / "summary.json")
    core_inventory = read(root / "artifact-hashes.json")["files_sha256"]
    field_inventory = read(root / "field-artifact-hashes.json")["files_sha256"]
    core_inventory = core_inventory | field_inventory
    for name, digest in core_inventory.items():
        if sha(root / name) != digest:
            raise ValueError("original core artifact changed")
    summaries, scenarios, preservation, coverage, content_summary, bindings = [], [], [], {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="dsv41-service-render-") as temporary:
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
            target = root / f"service-{profile}"
            qualified = args.qualified / f"service-{profile}"
            raw = getattr(args, f"{profile}_raw")
            paths = {
                "audit": qualified / "audit.json",
                "measurement": qualified / "measurement.json",
                "plan": raw / "strata/service-records/main-plan.json",
                "client-summary": raw / "strata/service-records/main-client/summary.json",
                "scheduler-receipt": raw / "scheduler-receipt.json",
            }
            provenance = read(target / "replay-provenance.json")
            program = root / "replay_service.py"
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
                            "scope": f"service-{profile}",
                            "mode": mode,
                            "reference": "unchanged admitted client and native audit inputs",
                        }
                    )
                new_metrics = metrics(
                    forward.error_summary, f"service-{profile}", mode, http["cohorts"], trace["cohorts"]
                )
                if any(new_metrics[-1][key] != trace["summary"][key] for key in ("mape_percent", "wape_percent")):
                    raise ValueError("native aggregate metric differs from frozen helper output")
                summaries.extend(new_metrics)
                for purpose in sorted({c["purpose"] for c in expected_http}):
                    scenarios.extend(
                        metrics(
                            forward.error_summary,
                            f"service-{profile}",
                            mode,
                            [c for c in http["cohorts"] if c["purpose"] == purpose],
                            [c for c in trace["cohorts"] if c["purpose"] == purpose],
                            purpose=purpose,
                        )
                    )
                intervals = [r for c in trace["cohorts"] for r in c["intervals"]]
                coverage[f"service-{profile}-{mode}"] = {
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
            content_summary[f"service-{profile}"] = diagnostic["summary"]
            bindings[f"service-{profile}"] = {
                "input_sha256": {k: sha(p) for k, p in paths.items()},
                "admission_sha256": sha(receipt_path),
                "content_diagnostic_sha256": sha(diagnostic_path),
                "comparison_provenance_sha256": sha(target / "replay-provenance.json"),
                "replay_program_sha256": sha(program),
            }
    addendum = supplementary_addendum(root, args)
    payload = {
        "schema": "dsv41.corrected-sol-service-verification.v1",
        "correction_fitting": False,
        "model_commit": core["model_commit"],
        "analysis_commit": TOOLS_REF,
        "native_sha256": NATIVE_SHA,
        "core_summary_sha256": sha(root / "summary.json"),
        "field_summary_sha256": sha(root / "field-summary.json"),
        "supplementary_addendum": addendum,
        "source_bindings": bindings,
        "observation_preservation": preservation,
        "coverage_and_cache": coverage,
        "metrics": summaries,
        "per_scenario_metrics": scenarios,
        "content_controls": content_summary,
        "metric_definition": core["metric_definition"],
        "uncertainty": (
            "30 independent trials per profile within separate frozen resource windows; paired "
            "scenario CIs remain in exact comparison outputs. No pooled CI across profiles or corpora. "
            "Observed mean-CI precision is recorded separately and is not a prediction-error target."
        ),
    }
    (root / "service-summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    consolidated = {
        "schema": "dsv41.corrected-sol-consolidated-verification.v1",
        "model_commit": core["model_commit"],
        "analysis_commit": TOOLS_REF,
        "native_sha256": NATIVE_SHA,
        "source_summary_sha256": {
            name: sha(root / name) for name in ("summary.json", "field-summary.json", "service-summary.json")
        },
        "metrics": core["metrics"] + read(root / "field-summary.json")["metrics"] + summaries,
        "qualification": "Distinct scopes remain separate. This is an inventory of metric rows, not a pooled estimate.",
    }
    (root / "consolidated-with-service.json").write_text(json.dumps(consolidated, indent=2) + "\n")
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
    with (root / "service-comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries + scenarios)
    write_readme(root, payload)
    draw(root, summaries)
    paths = [
        root / n
        for n in (
            "replay_service.py",
            "render_service.py",
            "service-summary.json",
            "consolidated-with-service.json",
            "service-comparison.csv",
            "service-README.md",
            "service-comparison.png",
            "service-comparison.pdf",
        )
    ]
    paths.extend(p for profile in ("off", "on") for p in (root / f"service-{profile}").iterdir() if p.is_file())
    paths.extend(p for p in (root / "service-addendum").iterdir() if p.is_file())
    inventory = {str(p.relative_to(root)): sha(p) for p in sorted(paths)}
    (root / "service-artifact-hashes.json").write_text(json.dumps({"files_sha256": inventory}, indent=2) + "\n")


def write_readme(root, payload):
    lines = [
        "# GB300 TP4 service corpus: separate supplementary OFF and ON strata",
        "",
        "Each profile contains 30 independent main trials in its own closed runtime lifecycle: four primary "
        "scenarios and 120 HTTP cohorts. The ten pilot trials are excluded. Observations, seeds, requested "
        "lengths, calibration, and the first comparison policy remain frozen. No fitting or timing filtering occurred.",
        "",
        "The corrected model/native/table identities match the [core report](README.md) and "
        "[field report](field-README.md), whose files remain byte-identical. The analysis helpers are pinned to "
        "the same commit. No corpus or replay profile is pooled. Per-scenario paired confidence intervals stay "
        "in the compressed results and remain conditional on their runtime lifecycle and frozen calibration.",
        "",
        "MAPE and WAPE use the same supported pairs. MAPE weights pairs equally; WAPE weights by observed "
        "value in each metric. Missing predictions stay in coverage. Native intervals are correlated and do not "
        "count as independent trials; HTTP mean TPOT is not a tail-ITL metric.",
        "",
        "| Replay | Metric | Mode | Covered | Real mean | Predicted mean | MAPE | WAPE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["metrics"]:
        if row["metric"] in LABELS:
            lines.append(
                f"| {row['scope'].removeprefix('service-').upper()} | {LABELS[row['metric']]} | {row['mode']} | "
                f"{row['predicted_points']}/{row['planned_points']} | {row['observed_mean_supported']:.3f} | "
                f"{row['predicted_mean_supported']:.3f} | {row['mape_percent']:.2f}% | {row['wape_percent']:.2f}% |"
            )
    lines += ["", "![Service corpus prediction errors](service-comparison.png)", "", "## Coverage and content", ""]
    for scope, counts in payload["coverage_and_cache"].items():
        lines.append(
            f"- {scope}: {counts['missing_http_predictions']} missing HTTP predictions; "
            f"{counts['missing_native_predictions']} missing native intervals out of "
            f"{counts['observed_native_intervals']}; {counts['cache_semantics_disagreements']} predicted "
            "cohorts disagree with observed initial cache reuse."
        )
    lines += ["", "Input content and actual batching controls are descriptive:", ""]
    for scope, points in payload["content_controls"].items():
        for purpose, detail in points.items():
            lines.append(
                f"- {scope}, {purpose}: {detail['observed_trials']}/{detail['planned_trials']} trials; "
                f"{detail['equal_sequences']} equal input-sequence pairs; "
                f"{detail['cold_same_first_prefill_dispatch']} cold pairs in the same first-prefill dispatch."
            )
    lines += [
        "",
        "Different first-prefill/decode schedules can confound content comparisons. Raw n-gram "
        "sets do not establish compressed Engram addresses or cache hits. KV resets do not flush "
        "Engram/HBM/L2 state. No causal Engram effect is claimed.",
        "",
        "## Observed precision and returned-output addendum",
        "",
        "These are checks on observed statistics and outputs, separate from prediction accuracy. "
        "The original N=30 remains fixed. The predeclared mean-CI precision target is a 95% bootstrap "
        "relative half-width at most 5%, tested for TTFT, mean TPOT, and throughput in each of four scenarios.",
        "",
        "| Stratum | Targets met | Largest observed CI relative half-width | Failed targets |",
        "| --- | ---: | ---: | --- |",
    ]
    for scope, item in payload["supplementary_addendum"]["observed_precision"].items():
        failed = (
            "; ".join(
                f"{r['purpose']} / {r['metric']}: {r['relative_half_width_percent']:.4f}%"
                for r in item["metrics"]
                if not r["met_target"]
            )
            or "none"
        )
        maximum = max(r["relative_half_width_percent"] for r in item["metrics"])
        lines.append(f"| {scope} | {item['met_targets']}/12 | {maximum:.4f}% | {failed} |")
    lines += [
        "",
        "The paired output check matches all planned input token IDs, seeds and output lengths; it is "
        "descriptive exact returned-token equality, not numerical-equivalence or task-quality validation:",
        "",
    ]
    for corpus, item in payload["supplementary_addendum"]["paired_outputs"].items():
        counts = item["requests_by_status"]
        lines.append(
            f"- {corpus}: {item['paired_trial_count']} trials, {item['paired_cohort_count']} cohorts, "
            f"{item['paired_request_count']} paired requests; {counts.get('equal', 0)} equal outputs, "
            f"{counts.get('different', 0)} different, {counts.get('missing_output_ids', 0)} missing."
        )
    lines += [
        "",
        "Output differences do not filter timing observations or establish a causal mechanism. "
        "The addendum records original response/source hashes while leaving core and field reports unchanged.",
        "",
        "[service-summary.json](service-summary.json) includes all seven metrics and four-scenario "
        "breakdowns. [service-comparison.csv](service-comparison.csv) contains both error metrics and "
        "supported means. [consolidated-with-service.json](consolidated-with-service.json) inventories "
        "132 metric rows: 48 core, 42 field, and 42 service, without pooling them.",
        "",
        "## Reproduction",
        "",
        "Use the recorded corrected native extension and source imports. `replay_service.py off` and "
        "`replay_service.py on` accept their respective `--qualified`, `--raw`, and optional "
        "`--output-dir`; each refuses to overwrite a scope. `render_service.py` accepts `--qualified`, "
        "`--off-raw`, `--on-raw`, `--off-content`, `--on-content`, `--field-off-raw`, `--field-on-raw`, "
        "`--field-output-check`, and `--service-output-check`. Original lifecycle inputs stay private; "
        "published provenance pins their hashes. The renderer checks original core/field file hashes, "
        "re-extracts unchanged observations from admitted inputs, and checks observed CI summaries "
        "against frozen client statistics. It never refits or reruns predictions.",
    ]
    (root / "service-README.md").write_text("\n".join(lines) + "\n")


def draw(root, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for i, profile in enumerate(("service-off", "service-on")):
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
    fig.suptitle("GB300 TP4 · supplementary service corpus · 30 independent trials per replay profile")
    fig.savefig(root / "service-comparison.png", dpi=180)
    fig.savefig(root / "service-comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
