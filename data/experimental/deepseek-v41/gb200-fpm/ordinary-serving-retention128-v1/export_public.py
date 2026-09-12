# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project reviewed private comparison reports; never predict, fit or change observations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

PREDICTION_COMMIT = "3627568d809774eb693abf3551a2ed36ad1dc032"
NATIVE_SHA = "f54a6015f6a97212e018615d909f1814f729d9b25df8b64e86802b7c5630e163"
CONFIG_SHA = "20aaf41eb5fe2e8bb25e85101b862e43cd84645430e863c21949687fec63f767"
FORWARD_ANALYSIS_SHA = "927653ca73334c4e94e9d2e03a62a509920aae4fd701a065decc269125bfc4dd"
STUDY = "ordinary-serving-retention128-v1"
METRICS = (
    "ttft_ms",
    "average_tpot_ms",
    "exact_itl_ms",
    "output_tokens_per_second",
    "request_latency_ms",
    "last_token_latency_ms",
    "native_forward_interval_ms",
)
UUID = re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
PRIVATE_PATH = re.compile(r"(?<![\w])/(?:home|tmp|lustre|campaign|opt|usr)/")
SHA = re.compile(r"^[0-9a-f]{64}$")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def picked(value, keys):
    return {k: deepcopy(value[k]) for k in keys if k in value}


def privacy_check(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    require(
        not UUID.search(text) and not PRIVATE_PATH.search(text),
        "private identifier/path in public projection",
    )


def safe_hashes(mapping):
    require(
        all(
            re.fullmatch(r"[a-zA-Z0-9_.-]+", k) and isinstance(v, str) and SHA.fullmatch(v) for k, v in mapping.items()
        ),
        "unsafe source hash mapping",
    )
    return deepcopy(mapping)


def public_cohort(cohort):
    purpose, trial = cohort["purpose"], cohort["trial_index"]
    require(
        re.fullmatch(r"[A-Za-z0-9-]+", purpose) and type(trial) is int and 0 <= trial < 30,
        "invalid public cohort identity",
    )
    return f"trial-{trial:03d}/{purpose}"


def failure_fields(row):
    if row["status"] == "predicted":
        return {}
    require(row["status"] == "prediction_unavailable", "unknown prediction status")
    fields = picked(row, ("failure_type", "failure"))
    require(
        set(fields) == {"failure_type", "failure"},
        "unavailable prediction lacks reason",
    )
    privacy_check(fields)
    fields["original_failure_sha256"] = hashlib.sha256(row["failure"].encode()).hexdigest()
    return fields


def project_e2e(report, plan):
    require(
        report["schema"] == "dsv41.e2e.comparison.v1"
        and report["complete_requested_study"] is True
        and report["correction_fitting"] is False,
        "E2E report is not complete final comparison",
    )
    targets = {c["cohort_id"]: c for c in plan["cohorts"] if c["comparison_role"] == "primary"}
    require(
        len(targets) == len(report["cohorts"]) == 450,
        "E2E primary cohort denominator differs",
    )
    seen, rows = set(), []
    for c in report["cohorts"]:
        require(
            c["cohort_id"] in targets and c["cohort_id"] not in seen,
            "unexpected/duplicate E2E cohort",
        )
        seen.add(c["cohort_id"])
        target = targets[c["cohort_id"]]
        require(
            all(c[k] == target[k] for k in ("purpose", "trial_index", "trial_seed", "corpus_role")),
            "E2E cohort identity differs",
        )
        row = picked(
            c,
            (
                "purpose",
                "trial_index",
                "trial_seed",
                "declared_coverage_role",
                "corpus_role",
                "observed",
                "status",
                "prediction",
                "signed_error_percent",
                "cache_semantics_match",
                "replay_spec_sha256",
                "native_result_sha256",
            ),
        )
        row["cohort_id"] = public_cohort(c)
        ids = {q["request_id"]: f"{row['cohort_id']}/request-{i:02d}" for i, q in enumerate(target["requests"])}
        if "native_initial_cached_tokens" in c:
            require(
                set(c["native_initial_cached_tokens"]) == set(ids),
                "cache map request coverage differs",
            )
            row["native_initial_cached_tokens"] = {ids[k]: v for k, v in c["native_initial_cached_tokens"].items()}
        if c["status"] == "predicted":
            require(
                {q["request_id"] for q in c["requests"]} == set(ids) and len(c["requests"]) == len(ids),
                "predicted request coverage differs",
            )
            row["requests"] = []
            for q in c["requests"]:
                request = picked(
                    q,
                    (
                        "arrival_time_ms",
                        "ttft_ms",
                        "first_token_ms",
                        "last_token_ms",
                        "terminal_time_ms",
                        "itl_ms",
                        "input_length",
                        "output_length",
                        "reused_input_tokens",
                        "terminal_status",
                        "native_initial_cached_tokens",
                        "cache_reuse_matches_native",
                    ),
                )
                request["request_id"] = ids[q["request_id"]]
                row["requests"].append(request)
        row.update(failure_fields(c))
        rows.append(row)
    result = picked(
        report,
        (
            "complete_requested_study",
            "scope",
            "effective_fmha_quant_mode",
            "capacity_policy",
            "correction_fitting",
            "points",
            "supplementary_metrics",
        ),
    )
    result.update(
        source_schema=report["schema"],
        study=STUDY,
        cohorts=rows,
        original_report_limitations=deepcopy(report["limitations"]),
        current_replay_semantics=(
            "See CURRENT_REPLAY_SEMANTICS.md; vLLM Aggregated retains its first-output decode charge."
        ),
    )
    privacy_check(result)
    return result


def calibration_coordinates(manifest):
    points = {
        (
            phase,
            q["batch_size"],
            q.get("total_prefill_tokens", 0),
            q["total_kv_read_tokens"],
        )
        for phase in ("prefill", "decode")
        for q in manifest[phase]
    }
    require(
        len(points) == 126 and len(manifest["prefill"]) == 100 and len(manifest["decode"]) == 26,
        "frozen measured calibration coordinate set differs",
    )
    return points


def coordinate_region(interval, points):
    scheduled = interval["scheduled_requests"]
    prefill, decode = (
        scheduled["num_prefill_requests"],
        scheduled["num_decode_requests"],
    )
    variance = any(
        interval["native_scheduled_requests"][k] != 0 for k in ("var_prefill_length", "var_decode_kv_tokens")
    )
    if prefill and decode:
        return {
            "region": "mixed_no_single_measured_coordinate",
            "coordinates": None,
            "outside_axes": [],
            "nonzero_native_length_variance": variance,
        }
    require(bool(prefill) != bool(decode), "empty native interval")
    key = (
        (
            "prefill",
            prefill,
            scheduled["sum_prefill_tokens"],
            scheduled["sum_prefill_kv_tokens"],
        )
        if prefill
        else ("decode", decode, 0, scheduled["sum_decode_kv_tokens"])
    )
    phase_points = [p for p in points if p[0] == key[0]]
    names = ("batch_size", "total_prefill_tokens", "total_kv_read_tokens")
    outside = [
        name
        for i, name in enumerate(names, 1)
        if key[i] < min(p[i] for p in phase_points) or key[i] > max(p[i] for p in phase_points)
    ]
    region = (
        "exact_aggregate_calibration_coordinate"
        if key in points
        else ("outside_raw_phase_bounding_box" if outside else "unseen_inside_raw_phase_bounding_box")
    )
    return {
        "region": region,
        "phase": key[0],
        "coordinates": dict(zip(names, key[1:], strict=True)),
        "outside_axes": outside,
        "nonzero_native_length_variance": variance,
    }


def project_trace(report, plan, calibration):
    require(
        report["schema"] == "dsv41.trace.comparison.v1"
        and report["complete_requested_study"] is True
        and report["qualification"] == "qualified_closed_main"
        and report["correction_fitting"] is False
        and report["missing_main_cohort_ids"] == [],
        "native comparison is not complete final report",
    )
    targets = {c["cohort_id"]: c for c in plan["cohorts"] if c["comparison_role"] == "primary"}
    require(len(targets) == len(report["cohorts"]) == 450, "native cohort coverage differs")
    require(
        sum(len(c["intervals"]) for c in report["cohorts"]) == 12531,
        "native observed interval count differs",
    )
    seen, rows = set(), []
    for c in report["cohorts"]:
        require(
            c["cohort_id"] in targets and c["cohort_id"] not in seen,
            "unexpected/duplicate native cohort",
        )
        seen.add(c["cohort_id"])
        require(
            all(c[k] == targets[c["cohort_id"]][k] for k in ("purpose", "trial_index", "trial_seed", "corpus_role")),
            "native trial identity differs",
        )
        row = picked(
            c,
            (
                "purpose",
                "trial_index",
                "trial_seed",
                "declared_coverage_role",
                "corpus_role",
                "summary",
            ),
        )
        row.update(cohort_id=public_cohort(c), intervals=[])
        for i, interval in enumerate(c["intervals"]):
            value = picked(
                interval,
                (
                    "phase",
                    "observed_ms",
                    "axis_bridge",
                    "variance_bridge",
                    "native_scheduled_requests",
                    "scheduled_requests",
                    "status",
                    "predicted_ms",
                    "signed_error_percent",
                    "native_work_role",
                ),
            )
            value.update(
                interval_id=f"{row['cohort_id']}/interval-{i:04d}",
                calibration_coordinate=coordinate_region(interval, calibration),
            )
            value.update(failure_fields(interval))
            row["intervals"].append(value)
        rows.append(row)
    result = picked(
        report,
        (
            "qualification",
            "complete_requested_study",
            "requested_trials",
            "planned_main_cohorts",
            "covered_main_cohorts",
            "missing_main_cohort_ids",
            "observed_target",
            "scope",
            "uncertainty",
            "effective_fmha_quant_mode",
            "correction_fitting",
            "summary",
            "independent_trial_summary",
        ),
    )
    result.update(source_schema=report["schema"], study=STUDY, cohorts=rows)
    privacy_check(result)
    return result


def paired_metrics(observations, pairs, *, metric, purpose, unit):
    require(
        all(math.isfinite(o) and o > 0 and math.isfinite(p) and p > 0 for o, p in pairs),
        "invalid supported pair",
    )
    return {
        "scope": "main",
        "purpose": purpose,
        "metric": metric,
        "coverage_unit": unit,
        "observed_points": len(observations),
        "predicted_points": len(pairs),
        "observed_mean_supported": statistics.mean(o for o, _ in pairs) if pairs else None,
        "predicted_mean_supported": statistics.mean(p for _, p in pairs) if pairs else None,
        "mape_percent": statistics.mean(100 * abs(p / o - 1) for o, p in pairs) if pairs else None,
        "wape_percent": 100 * sum(abs(p - o) for o, p in pairs) / sum(o for o, _ in pairs) if pairs else None,
    }


def metric_rows(e2e, trace):
    rows = []
    for purpose in [None] + sorted({c["purpose"] for c in e2e["cohorts"]}):
        for metric in METRICS:
            if metric == "native_forward_interval_ms":
                observations = [
                    i for c in trace["cohorts"] if purpose is None or c["purpose"] == purpose for i in c["intervals"]
                ]
                pairs = [(r["observed_ms"], r["predicted_ms"]) for r in observations if r["status"] == "predicted"]
                unit = "correlated native intervals"
            else:
                observations = [
                    c
                    for c in e2e["cohorts"]
                    if (purpose is None or c["purpose"] == purpose) and metric in c["observed"]
                ]
                pairs = [
                    (c["observed"][metric], c["prediction"][metric])
                    for c in observations
                    if c["status"] == "predicted" and metric in c["prediction"]
                ]
                unit = "scenario whole-trial cohorts"
            row = paired_metrics(observations, pairs, metric=metric, purpose=purpose, unit=unit)
            if purpose is not None and metric != "native_forward_interval_ms" and pairs:
                stored = e2e["points"][purpose]["metrics"][metric]
                require(
                    stored["independent_trials"] == len(pairs),
                    "stored paired sample count differs",
                )
                for out, key in (
                    ("mape_percent", "mape_percent"),
                    ("wape_percent", "wape_percent"),
                    ("observed_mean_supported", "observed_mean"),
                    ("predicted_mean_supported", "predicted_mean"),
                ):
                    require(
                        math.isclose(row[out], stored[key], rel_tol=1e-12, abs_tol=1e-10),
                        "stored paired arithmetic differs",
                    )
                    row[out] = stored[key]
            rows.append(row)
    return rows


def summarize(e2e, trace, observations, statistical_review, calibration):
    require(
        observations["client_coverage_complete"] is True and observations["requested_trials"] == 30,
        "main observation coverage differs",
    )
    require(
        statistical_review["review_passed"] is True and statistical_review["main_requests_raw_http_requalified"] == 720,
        "independent observation review is not passed",
    )
    misses = deepcopy(statistical_review["precision_target_misses"])
    require(
        {(x["purpose"], x["metric"]) for x in misses} == {("short", "ttft_ms"), ("prefix-reuse-B", "ttft_ms")},
        "observation precision misses changed",
    )
    cache = [c for c in e2e["cohorts"] if c.get("cache_semantics_match") is False]
    require(
        len(cache) == 13 and {c["purpose"] for c in cache} == {"engram-repeated-text"},
        "cache disagreement set differs",
    )
    region_rows = defaultdict(list)
    for c in trace["cohorts"]:
        for row in c["intervals"]:
            region_rows[row["calibration_coordinate"]["region"]].append(row)
    regions = {
        name: paired_metrics(
            rows,
            [(r["observed_ms"], r["predicted_ms"]) for r in rows if r["status"] == "predicted"],
            metric="native_forward_interval_ms",
            purpose=None,
            unit="correlated native intervals",
        )
        for name, rows in sorted(region_rows.items())
    }
    require(
        sum(x["observed_points"] for x in regions.values()) == 12531,
        "calibration coordinate regions lose intervals",
    )
    return {
        "study": STUDY,
        "dataset_role": "independent_verification",
        "correction_fitting": False,
        "calibration_self_queries_in_accuracy": False,
        "physical_lifecycles": 1,
        "independent_trials_per_scenario": 30,
        "coverage": {
            "main_cohorts": 510,
            "main_requests": 720,
            "primary_scenarios": 15,
            "primary_cohorts": 450,
            "primary_requests": 660,
            "primary_native_intervals": 12531,
            "setup_cohorts": 60,
            "setup_requests": 60,
            "pilot_trials_excluded": 10,
            "all_phase_cohorts": 705,
            "all_phase_requests": 994,
            "all_phase_active_dispatches": 17230,
            "all_phase_native_records_including_heartbeats": 17271,
            "predicted_primary_cohorts": sum(c["status"] == "predicted" for c in e2e["cohorts"]),
            "predicted_primary_native_intervals": sum(
                r["status"] == "predicted" for c in trace["cohorts"] for r in c["intervals"]
            ),
        },
        "metrics": metric_rows(e2e, trace),
        "observation_statistics": picked(
            observations,
            (
                "requested_trials",
                "primary_scenarios",
                "planned_cohorts",
                "observed_cohorts",
                "client_coverage_complete",
                "points",
                "qualification",
            ),
        ),
        "observation_precision_target_relative_half_width": 0.05,
        "observation_precision_misses": misses,
        "cache_disagreements": {
            "cohorts": len(cache),
            "observed_primary_cohorts": 450,
            "scenario_cohorts": 30,
            "purpose": "engram-repeated-text",
            "cohort_ids": [c["cohort_id"] for c in cache],
            "requests": sum(r.get("cache_reuse_matches_native") is False for c in cache for r in c["requests"]),
            "selection_policy": "retain every cohort and timing pair",
        },
        "calibration_coordinate_coverage": {
            "measured_points": 126,
            "planned_points": 126,
            "prefill_points": 100,
            "decode_points": 26,
            "coordinates": [
                dict(
                    zip(
                        (
                            "phase",
                            "batch_size",
                            "total_prefill_tokens",
                            "total_kv_read_tokens",
                        ),
                        p,
                        strict=True,
                    )
                )
                for p in sorted(calibration)
            ],
            "region_metrics": regions,
            "classification_is_actual_interpolation_branch": False,
            "interpretation": (
                "Exact aggregate coordinates are not identical content/schedules. Raw bounding-box inclusion is "
                "not interpolation proof. Consumer support is counted separately; native regions do not "
                "classify HTTP replay queries."
            ),
        },
        "runtime_condition": {
            "prefix_cache_retention_interval": 128,
            "observed_prefix_reuse_tokens": 512,
            "main_prefix_reuse_cohorts": 30,
            "max_num_seqs": 2,
            "heterogeneous_http_requests": 3,
            "pressure": picked(
                statistical_review["final_pressure"],
                (
                    "allocation_attempts",
                    "allocation_refusals",
                    "allocation_exceptions",
                    "preemptions",
                    "minimum_free_blocks",
                    "watermark_blocks",
                    "num_physical_blocks",
                ),
            ),
        },
        "canary_output_comparison": picked(
            statistical_review["free_autoregressive_canary"],
            (
                "request_denominator",
                "equal_output_sequences",
                "mismatched_output_sequences",
                "token_position_denominator",
                "mismatched_token_positions",
                "reference_controls_sampling",
                "model_quality_equivalence_established",
                "timings_selected_by_equality",
            ),
        ),
    }


def export(args):
    names = (
        "e2e",
        "trace",
        "measurement",
        "normalization_receipt",
        "plan",
        "observed_main",
        "statistical_review",
        "prediction_config",
        "prepared_image_receipt",
    )
    source_paths = {n: getattr(args, n).resolve(strict=True) for n in names}
    source_paths["calibration_points"] = args.calibration_root / "point-manifest.json"
    source_paths["calibration_admission"] = args.calibration_root / "admission-receipt.json"
    source_hashes = {n: sha(p) for n, p in source_paths.items()}
    values = {n: read(p) for n, p in source_paths.items()}
    e2e, trace, measurement, plan = (values[n] for n in ("e2e", "trace", "measurement", "plan"))
    require(source_hashes["prediction_config"] == CONFIG_SHA, "fixed FP8 FPM config differs")
    require(
        e2e["native_extension_sha256"] == NATIVE_SHA
        and trace["prediction_sources_sha256"]["native_extension"] == NATIVE_SHA,
        "prediction native source differs",
    )
    require(
        trace["prediction_sources_sha256"]["analysis"] == FORWARD_ANALYSIS_SHA,
        "reviewed forward analysis differs",
    )
    require(
        e2e["prediction_config"] == trace["prediction_config"] == values["prediction_config"],
        "prediction configs differ",
    )
    require(
        e2e["systems_identity"] == trace["systems_identity"]
        and e2e["resolved_model_identity"] == trace["resolved_model_identity"],
        "prediction source/model identities differ",
    )
    require(
        e2e["run_id"] == trace["run_id"] == plan["run_id"],
        "comparison lifecycle differs",
    )
    for name in ("plan", "measurement"):
        require(
            e2e["input_sha256"][name] == trace["input_sha256"][name] == source_hashes[name],
            "comparison input SHA differs",
        )
    require(
        e2e["input_sha256"]["audit"] == trace["input_sha256"]["audit"],
        "audited comparison source differs",
    )
    require(
        values["calibration_admission"]["valid"] is True
        and sum(x["measured_points"] for x in values["calibration_admission"]["artifacts"]) == 126,
        "calibration admission count differs",
    )
    require(
        measurement["backend"] == "vllm"
        and measurement["decoder_replay"] is False
        and measurement["execution_mode"] == "eager",
        "unexpected measured runtime",
    )
    require(
        measurement["complete_requested_study"] is True
        and measurement["qualification"]
        == values["normalization_receipt"]["qualification"]
        == "qualified_closed_main_stratum",
        "normalization is not qualified complete main",
    )
    require(
        measurement["source_bindings"]["main_statistics_file_sha256"] == source_hashes["observed_main"],
        "original main statistics differ",
    )
    image = values["prepared_image_receipt"]
    require(
        image["sha256"] == "c42ea19c53004855c25731f2cf5e6b5e3955ed677f8d3a7b854b864b11e32d52"
        and image["bytes"] == 21501927424
        and source_hashes["prepared_image_receipt"]
        == values["normalization_receipt"]["actual_runtime_verification"]["actual_evidence_sha256"]["image"],
        "actual prepared image source differs",
    )
    require(
        {c["corpus_role"] for c in plan["cohorts"] if c["comparison_role"] == "primary"} == {"primary"},
        "normalized corpus role differs",
    )
    for relative, expected in e2e["systems_identity"]["files_sha256"].items():
        path = (args.calibration_root / "systems" / relative).resolve(strict=True)
        require(
            path.is_relative_to((args.calibration_root / "systems").resolve()) and sha(path) == expected,
            "selected calibration overlay differs",
        )
    calibration = calibration_coordinates(values["calibration_points"])
    public_e2e, public_trace = (
        project_e2e(e2e, plan),
        project_trace(trace, plan, calibration),
    )
    public_e2e["original_private_report_sha256"] = source_hashes["e2e"]
    public_trace["original_private_report_sha256"] = source_hashes["trace"]
    summary = summarize(
        public_e2e,
        public_trace,
        values["observed_main"],
        values["statistical_review"],
        calibration,
    )
    identity = picked(
        e2e["measurement_identity"],
        (
            "backend",
            "backend_version",
            "system_name",
            "model_name",
            "decoder_replay",
            "execution_mode",
            "runtime_digest",
            "producer_semantics",
            "producer_source_sha256",
            "observer_source_sha256",
            "model_config_canonical_sha256",
        ),
    )
    identity["source_bindings"] = safe_hashes(e2e["measurement_identity"]["source_bindings"])
    provenance = {
        "study": STUDY,
        "original_private_files_sha256": source_hashes,
        "prediction_commit": PREDICTION_COMMIT,
        "measurement_identity": identity,
        "prediction_native_extension_sha256": NATIVE_SHA,
        "e2e_prediction_sources_sha256": safe_hashes(e2e["prediction_sources"]),
        "trace_prediction_sources_sha256": safe_hashes(trace["prediction_sources_sha256"]),
        "trace_source_sha256": safe_hashes(trace["trace_source_sha256"]),
        "systems_identity": e2e["systems_identity"],
        "resolved_model_identity": e2e["resolved_model_identity"],
        "normalization_builder_sha256": values["normalization_receipt"]["builder_sha256"],
        "projection_policy": (
            "Retained measurement/prediction numbers and CI arrays are preserved; private identifiers are "
            "omitted or replaced by public trial/scenario/order IDs. Original full reports remain private "
            "and hash-bound."
        ),
        "runtime_contract_changed": False,
    }
    provenance["base_arm64_image_digest"] = image["base_arm64_digest"]
    provenance["actual_prepared_squashfs_sha256"] = image["sha256"]
    provenance["actual_prepared_squashfs_bytes"] = image["bytes"]
    for value in (public_e2e, public_trace, summary, provenance):
        privacy_check(value)
    require(
        source_hashes == {n: sha(p) for n, p in source_paths.items()},
        "private source changed during projection",
    )
    root = args.output_root.resolve()
    require(
        not args.private_receipt.resolve().is_relative_to(root),
        "private receipt cannot be inside public bundle",
    )
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "reports/main/e2e.json.gz": public_e2e,
        "reports/main/trace.json.gz": public_trace,
        "reports/main/summary.json.gz": summary,
        "provenance.json": provenance,
    }
    require(
        all(not (root / n).exists() for n in list(outputs) + ["prediction-config.json", "render-inputs.json"]),
        "export refuses overwrite",
    )
    for name, value in outputs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = encoded(value)
        if name.endswith(".gz"):
            with path.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                f.write(blob)
        else:
            with path.open("xb") as f:
                f.write(blob)
    (root / "prediction-config.json").write_bytes(source_paths["prediction_config"].read_bytes())
    render_names = list(outputs) + [
        "prediction-config.json",
        "CURRENT_REPLAY_SEMANTICS.md",
        "export_public.py",
        "render_report.py",
    ]
    inventory = {n: sha(root / n) for n in sorted(render_names)}
    (root / "render-inputs.json").write_bytes(encoded({"files_sha256": inventory}))
    receipt = {
        "source_files_sha256": {str(p): source_hashes[n] for n, p in source_paths.items()},
        "exporter_sha256": sha(__file__),
        "public_inputs_sha256": inventory,
        "retained_measurement_prediction_numbers_preserved": True,
        "predictions_or_fitting_run": False,
    }
    with args.private_receipt.open("xb") as f:
        f.write(encoded(receipt))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "e2e",
        "trace",
        "measurement",
        "normalization-receipt",
        "plan",
        "observed-main",
        "statistical-review",
        "prediction-config",
        "calibration-root",
        "output-root",
        "private-receipt",
        "prepared-image-receipt",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    export(parser.parse_args())
