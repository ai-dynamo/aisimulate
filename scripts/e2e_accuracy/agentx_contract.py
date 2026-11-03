# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX measurement identities, scoring, and public numerical contracts."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.e2e_accuracy.agentx_comparison import validate as validate_comparison
from scripts.e2e_accuracy.benchmark_features import validate as validate_features

INPUT_SCHEMA = "aisimulate.agentx.inputs.v1"
SUMMARY_SCHEMA = "aisimulate.agentx.accuracy.v1"
INDEX = "catalog/e2e/agentx.json"
METRICS = ("tpot_ms", "ttft_ms", "output_tokens_s")
IDENTITY = (
    "id",
    "category",
    "model",
    "hardware",
    "backend",
    "measured_backend_version",
    "precision",
    "tensor_parallel",
    "lanes",
    "initial_input_tokens",
    "cache_state",
    "repeat",
    "benchmark_url",
    "source_run_url",
    "boundary",
    "measured_requests",
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def driver_digest(read_file=None):
    root = Path(__file__).resolve().parents[2]
    read_file = read_file or (lambda name: (root / name).read_bytes())
    names = ("agentx_contract.py", "agentx_predict.py", "run_agentx_accuracy.py", "run_e2e_accuracy.py")
    # Validate historical producers with their original four-file closure.
    if b'"agentx_comparison.py"' in read_file("scripts/e2e_accuracy/agentx_contract.py"):
        names += ("agentx_comparison.py", "agentx_curves.py")
    if b'"benchmark_features.py"' in read_file("scripts/e2e_accuracy/agentx_contract.py"):
        names += ("benchmark_features.py",)
    return hashlib.sha256(
        canonical({name: hashlib.sha256(read_file("scripts/e2e_accuracy/" + name)).hexdigest() for name in names})
    ).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def mean(values):
    values = [value for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def ape(actual, predicted):
    return 100 * abs(predicted - actual) / actual if finite(actual) and actual > 0 and finite(predicted) else None


def validate_catalog(catalog):
    if set(catalog) != {"schema", "runs", "snapshot_id"} or catalog["schema"] != INPUT_SCHEMA:
        raise ValueError("Invalid AgentX input catalog")
    snapshot = hashlib.sha256(canonical({k: v for k, v in catalog.items() if k != "snapshot_id"})).hexdigest()
    if catalog["snapshot_id"] != snapshot or not catalog["runs"]:
        raise ValueError("Invalid AgentX snapshot identity")
    ids, identities = set(), set()
    for run in catalog["runs"]:
        if set(run) != {*IDENTITY, "artifacts"} or not re.fullmatch(r"[a-z0-9-]+", run["id"]):
            raise ValueError("Invalid AgentX run identity")
        if run["id"] in ids:
            raise ValueError("Duplicate AgentX run")
        ids.add(run["id"])
        identity = tuple(
            run[k]
            for k in (
                "category",
                "model",
                "hardware",
                "backend",
                "lanes",
                "initial_input_tokens",
                "cache_state",
                "repeat",
            )
        )
        if identity in identities:
            raise ValueError("Duplicate AgentX workload identity")
        identities.add(identity)
        if run["category"] not in {"agentx", "self_collected"} or type(run["lanes"]) is not int or run["lanes"] < 1:
            raise ValueError("Invalid AgentX category or lane count")
        if run["boundary"] not in {"successful_profile_span", "complete_finite_cohort"}:
            raise ValueError("Unknown measurement boundary")
        for entry in run["artifacts"].values():
            if set(entry) != {"path", "sha256", "bytes"} or not re.fullmatch(
                r"data/e2e/agentx/objects/[0-9a-f]{64}\.(json|jsonl|md)", entry["path"]
            ):
                raise ValueError("Unsafe AgentX artifact path")
            if (
                not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                or type(entry["bytes"]) is not int
                or entry["bytes"] < 0
            ):
                raise ValueError("Invalid AgentX artifact checksum")
            if Path(entry["path"]).stem != entry["sha256"]:
                raise ValueError("Artifact name differs from its checksum")
        if not {"trace", "config", "observations"} <= run["artifacts"].keys():
            raise ValueError("Incomplete AgentX replay inputs")
        if (
            run["category"] == "self_collected"
            and run["boundary"] == "successful_profile_span"
            and "feature_metadata" not in run["artifacts"]
        ):
            raise ValueError("New Weka collections require silicon feature metadata")
    return catalog


def cohort(rows, *, complete=False, expected=None):
    successful = [r for r in rows if r["status"] == "completed"]
    result = {metric: mean(r.get(metric) for r in successful) for metric in ("ttft_ms", "tpot_ms")}
    result["output_tokens_s"] = None
    if not successful or (complete and len(successful) != expected):
        return result
    if all(finite(r.get(k)) for r in successful for k in ("start_ms", "end_ms", "output_length")):
        span = max(r["end_ms"] for r in successful) - min(r["start_ms"] for r in successful)
        if span > 0:
            result["output_tokens_s"] = sum(r["output_length"] for r in successful) * 1000 / span
    return result


def score(observations, native, boundary):
    """Keep all measured observations; source identity never implies equal prompts."""
    references = [row for row in observations if row["phase"] == "profile"]
    measured_keys = Counter(r["key"] for r in references)
    predictions = [row for row in native if row["phase"] == "profile"]
    by_key = defaultdict(list)
    for row in predictions:
        by_key[row["key"]].append(row)
    counts = Counter(measured=len(references), predicted=len(predictions), matched=0)
    pairs = []
    for index, actual in enumerate(references):
        candidates = by_key.get(actual["key"], [])
        predicted = candidates[0] if len(candidates) == 1 else None
        if actual["status"] != "completed":
            failure = actual.get("failure_reason")
            reason = failure if failure in {"context_overflow", "measurement_metrics_unavailable"} else "measurement_failed"
        elif actual.get("identity_valid") is False:
            reason = "missing_measurement_identity"
        elif measured_keys[actual["key"]] > 1:
            reason = "ambiguous_measurement"
        elif len(candidates) > 1:
            reason = "ambiguous_prediction"
        elif predicted is None:
            reason = "missing_prediction"
        elif predicted["status"] != "completed":
            reason = "prediction_failed"
        elif any(type(r.get("output_length")) is not int or r["output_length"] <= 0 for r in (actual, predicted)):
            reason = "invalid_output_length"
        elif actual["output_length"] != predicted["output_length"]:
            reason = "output_length_mismatch"
        else:
            reason = "matched"
        counts[reason] += 1
        pair = {
            "request": index + 1,
            "request_id": hashlib.sha256(actual["key"].encode()).hexdigest()[:16],
            "status": reason,
            "input_length_matches": actual["input_length"] == predicted["input_length"] if predicted else None,
        }
        for metric in ("ttft_ms", "tpot_ms"):
            pair["silicon_" + metric] = actual.get(metric)
            pair["predicted_" + metric] = predicted.get(metric) if predicted else None
            pair[metric + "_ape"] = ape(actual.get(metric), predicted.get(metric)) if reason == "matched" else None
        pairs.append(pair)
    keys = {r["key"] for r in references}
    counts["unmatched_predictions"] = sum(r["key"] not in keys for r in predictions)
    counts["input_length_mismatch"] = sum(p["input_length_matches"] is False for p in pairs)
    actual = cohort(references, complete=boundary == "complete_finite_cohort", expected=len(references))
    predicted = cohort(predictions, complete=boundary == "complete_finite_cohort", expected=len(references))
    request = {}
    for metric in ("ttft_ms", "tpot_ms"):
        errors = [r[metric + "_ape"] for r in pairs if finite(r[metric + "_ape"])]
        request[metric] = {"sum_ape": sum(errors), "count": len(errors), "mape": mean(errors)}
    return {
        "silicon": actual,
        "prediction": predicted,
        "point_errors": {metric: ape(actual[metric], predicted[metric]) for metric in METRICS},
        "request_errors": request,
        "coverage": dict(counts),
        "pairs": pairs,
    }


def aggregate(runs):
    result = {
        "point_errors": {metric: mean(r["point_errors"][metric] for r in runs) for metric in METRICS},
        "request_errors": {},
        "coverage": dict(sum((Counter(r["coverage"]) for r in runs), Counter())),
    }
    for metric in ("ttft_ms", "tpot_ms"):
        count = sum(r["request_errors"][metric]["count"] for r in runs)
        total = sum(r["request_errors"][metric]["sum_ape"] for r in runs)
        result["request_errors"][metric] = {"count": count, "sum_ape": total, "mape": total / count if count else None}
    return result


def public_text(value):
    if not isinstance(value, str) or len(value) > 500 or any(c in value for c in ("<", ">", "\\", "\n")):
        raise ValueError("Invalid public text")
    if re.search(r"(/(?:home|Users|tmp|lustre|mnt)/|https?://|Bearer |hf_[A-Za-z0-9])", value):
        raise ValueError("Private text in public artifact")


def validate_public(value):
    """Strict recursive field vocabulary: unknown raw fields never reach Pages."""
    allowed = set(IDENTITY) | {
        "schema",
        "snapshot",
        "branch",
        "commit_sha",
        "dataset_revision",
        "wheel_sha256",
        "driver_sha256",
        "run_id",
        "run_attempt",
        "completed_at",
        "cohort_sha256",
        "preview",
        "runs",
        "outcome",
        "reason",
        "prediction_backend_version",
        "silicon",
        "prediction",
        "point_errors",
        "request_errors",
        "coverage",
        "pairs",
        "detail_path",
        "report",
        "ttft_ms",
        "tpot_ms",
        "output_tokens_s",
        "sum_ape",
        "count",
        "mape",
        "measured",
        "predicted",
        "matched",
        "measurement_failed",
        "ambiguous_prediction",
        "ambiguous_measurement",
        "missing_measurement_identity",
        "context_overflow",
        "measurement_metrics_unavailable",
        "missing_prediction",
        "prediction_failed",
        "output_length_mismatch",
        "invalid_output_length",
        "input_length_mismatch",
        "unmatched_predictions",
        "request",
        "status",
        "input_length_matches",
        "silicon_ttft_ms",
        "silicon_tpot_ms",
        "predicted_ttft_ms",
        "predicted_tpot_ms",
        "ttft_ms_ape",
        "tpot_ms_ape",
        "metadata",
        "title",
        "framework",
        "backend_version",
        "topology",
        "plotly_python_version",
        "telemetry",
        "completion_status",
        "phases",
        "preparation",
        "profile",
        "start_ms",
        "end_ms",
        "admission_cutoff_ms",
        "requests",
        "samples",
        "lifecycle",
        "uuid",
        "request_id",
        "phase",
        "arrival_time_ms",
        "terminal_time_ms",
        "first_token_ms",
        "input_length",
        "output_length",
        "itl_ms",
        "e2e_latency_ms",
        "terminal_status",
        "routing_history",
        "admission_history",
        "at_ms",
        "pool",
        "worker_id",
        "play_id",
        "conversation_id",
        "agentic",
        "turn_index",
        "admission_count",
        "readmission_count",
        "reused_input_tokens",
        "first_admission_reused_input_tokens",
        "t",
        "start",
        "kind",
        "ordinal",
        "duration",
        "running",
        "waiting",
        "router_pending",
        "cache_pct",
        "workers",
        "host_pools",
        "output_tokens",
        "prefill_tokens",
        "arrivals",
        "completed",
        "accept",
        "accept_count",
        "admission_input",
        "admission_requests",
        "hbm",
        "host",
        "miss",
        "lookup_hits",
        "lookup_queries",
        "lookup_count",
        "scheduler_hits",
        "scheduler_queries",
        "preemptions",
        "active_blocks",
        "inactive_blocks",
        "total_blocks",
        "kv_cache_used_bytes",
        "kv_cache_capacity_bytes",
        "active_cache_usage",
        "physical_cache_usage",
        "running_requests",
        "waiting_requests",
        "role",
        "metric_role",
        "dp_rank",
        "sample_interval_ms",
        "clock",
        "clock_origin_ms",
        "profile_start_ms",
        "units",
        "timestamps",
        "durations",
        "traffic_duration",
        "tokens",
        "cache_blocks",
        "cache_bytes",
        "cache_usage",
        "event",
        "request_uuid",
        "config_summary",
        "name",
        "value",
        "cache_alignment",
        "comparison",
        "op_data",
        "file",
        "sha256",
        "feature_metadata",
    }

    def walk(item, key=None):
        if key == "feature_metadata":
            validate_features(item)
            return
        if key == "comparison":
            validate_comparison(item)
            return
        if isinstance(item, dict):
            if set(item) - allowed:
                raise ValueError("Unexpected public AgentX fields: " + str(sorted(set(item) - allowed)))
            for name, child in item.items():
                walk(child, name)
        elif isinstance(item, list):
            for child in item:
                walk(child, key)
        elif isinstance(item, str):
            if key in {"benchmark_url", "source_run_url"}:
                if not re.fullmatch(
                    r"https://(?:inferencex\.semianalysis\.com/inference/agentic/[0-9]+|github\.com/SemiAnalysisAI/InferenceX/actions/runs/[0-9]+/attempts/[0-9]+)",
                    item,
                ):
                    raise ValueError("Unapproved source link")
            elif key == "detail_path":
                if not re.fullmatch(r"details/[a-z0-9-]+\.json", item):
                    raise ValueError("Unsafe point detail path")
            else:
                public_text(item)
        elif item is not None and type(item) not in (bool, int, float):
            raise ValueError("Invalid public value")
        elif type(item) in (int, float) and not math.isfinite(item):
            raise ValueError("Nonfinite public value")

    walk(value)
    return value


PAIR_STATUSES = {
    "matched",
    "measurement_failed",
    "ambiguous_prediction",
    "ambiguous_measurement",
    "missing_measurement_identity",
    "context_overflow",
    "measurement_metrics_unavailable",
    "missing_prediction",
    "prediction_failed",
    "output_length_mismatch",
    "invalid_output_length",
}


def validate_report(report):
    """Numerical telemetry only; containers and free-form text are not interchangeable."""
    if report is None:
        return
    if set(report) - {"comparison"} != {
        "schema",
        "metadata",
        "prediction",
        "telemetry",
        "phases",
        "requests",
        "samples",
        "lifecycle",
    }:
        raise ValueError("Invalid AgentX replay report")
    if report["schema"] != "aisimulate.replay.dashboard.v1" or report["prediction"] != {}:
        raise ValueError("Unsupported public replay schema")

    def scalars(row, numbers, strings=(), booleans=()):
        if not isinstance(row, dict) or set(row) - set(numbers) - set(strings) - set(booleans):
            raise ValueError("Unexpected numerical report fields")
        for key, value in row.items():
            if value is None:
                continue
            if key in numbers and not finite(value):
                raise ValueError("Report metrics must be numbers")
            if key in strings and (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./ :+-]{1,200}", value)):
                raise ValueError("Invalid report label")
            if key in booleans and type(value) is not bool:
                raise ValueError("Invalid report boolean")

    scalars(
        report["metadata"],
        (),
        ("title", "model", "hardware", "framework", "backend_version", "topology", "plotly_python_version"),
    )
    scalars(
        report["telemetry"],
        ("sample_interval_ms", "clock_origin_ms", "profile_start_ms"),
        ("clock", "completion_status"),
    )
    if set(report["phases"]) - {"preparation", "profile"} or "profile" not in report["phases"]:
        raise ValueError("Invalid report phases")
    for phase in report["phases"].values():
        scalars(phase, ("start_ms", "end_ms", "admission_cutoff_ms"), ("completion_status",))
    for row in report["requests"]:
        scalars(
            {k: v for k, v in row.items() if k not in {"agentic", "routing_history", "admission_history"}},
            (
                "arrival_time_ms",
                "terminal_time_ms",
                "first_token_ms",
                "input_length",
                "output_length",
                "ttft_ms",
                "itl_ms",
                "e2e_latency_ms",
                "admission_count",
                "readmission_count",
                "reused_input_tokens",
                "first_admission_reused_input_tokens",
                "turn_index",
            ),
            ("phase", "terminal_status", "uuid", "request_id"),
        )
        for key in ("uuid", "request_id"):
            if row.get(key) is not None and not re.fullmatch(r"[0-9a-f]{16}", row[key]):
                raise ValueError("Request identifiers must be opaque")
        scalars(row.get("agentic", {}), (), ("play_id", "conversation_id", "request_id"))
        if any(not re.fullmatch(r"[0-9a-f]{16}", v) for v in row.get("agentic", {}).values() if v is not None):
            raise ValueError("Agentic identifiers must be opaque")
        for name in ("routing_history", "admission_history"):
            for history in row.get(name, []):
                scalars(history, ("at_ms", "worker_id"), ("pool",))
    for sample in report["samples"]:
        scalars(
            {k: v for k, v in sample.items() if k not in {"workers", "host_pools"}},
            (
                "t",
                "start",
                "ordinal",
                "duration",
                "running",
                "waiting",
                "router_pending",
                "cache_pct",
                "output_tokens",
                "prefill_tokens",
                "arrivals",
                "completed",
                "accept",
                "accept_count",
                "admission_input",
                "admission_requests",
                "hbm",
                "host",
                "miss",
                "lookup_hits",
                "lookup_queries",
                "lookup_count",
                "scheduler_hits",
                "scheduler_queries",
                "preemptions",
            ),
            ("phase", "kind"),
        )
        if sample.get("host_pools") != []:
            raise ValueError("Host telemetry is outside the initial AgentX contract")
        for worker in sample["workers"]:
            scalars(
                worker,
                (
                    "worker_id",
                    "dp_rank",
                    "active_blocks",
                    "inactive_blocks",
                    "total_blocks",
                    "kv_cache_used_bytes",
                    "kv_cache_capacity_bytes",
                    "active_cache_usage",
                    "physical_cache_usage",
                    "running_requests",
                    "waiting_requests",
                ),
                ("role", "metric_role"),
            )
    for event in report["lifecycle"]:
        scalars(event, ("at_ms",), ("phase", "event", "request_uuid", "play_id"))
        for key in ("request_uuid", "play_id"):
            if event.get(key) is not None and not re.fullmatch(r"[0-9a-f]{16}", event[key]):
                raise ValueError("Lifecycle identifiers must be opaque")


def validate_result(run):
    """Reconcile public point values, per-request errors, and every exclusion."""
    validate_public(run)
    expected = {
        *IDENTITY,
        "outcome",
        "reason",
        "prediction_backend_version",
        "cache_alignment",
        "config_summary",
        "silicon",
        "prediction",
        "point_errors",
        "request_errors",
        "coverage",
        "pairs",
        "report",
    }
    if set(run) - {"op_data", "feature_metadata"} != expected or run["outcome"] not in {
        "completed",
        "failed",
        "unsupported",
    }:
        raise ValueError("Invalid AgentX result fields")
    if "feature_metadata" in run:
        validate_features(run["feature_metadata"], run["id"])
    elif run["id"].startswith("qwen-weka-l"):
        raise ValueError("New Qwen Weka publications require silicon feature metadata")
    for item in run.get("op_data", []):
        if (
            set(item) != {"file", "sha256"}
            or not re.fullmatch(r"[a-z0-9_./-]+\.parquet", item["file"])
            or ".." in item["file"]
            or item["file"].startswith("/")
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("Invalid op-data identity")
    for field in ("silicon", "prediction", "point_errors"):
        if set(run[field]) != set(METRICS) or any(v is not None and not finite(v) for v in run[field].values()):
            raise ValueError("Invalid AgentX point metrics")
    if run["point_errors"] != {m: ape(run["silicon"][m], run["prediction"][m]) for m in METRICS}:
        raise ValueError("Point error does not reconcile")
    if set(run["request_errors"]) != {"ttft_ms", "tpot_ms"}:
        raise ValueError("Invalid request metrics")
    if any(type(n) is not int or n < 0 for n in run["coverage"].values()):
        raise ValueError("Invalid AgentX coverage")
    if (
        set(run["coverage"])
        - PAIR_STATUSES
        - {"measured", "predicted", "unmatched_predictions", "input_length_mismatch"}
    ):
        raise ValueError("Unknown coverage reason")
    if len(run["pairs"]) != run["measured_requests"] or run["coverage"]["measured"] != run["measured_requests"]:
        raise ValueError("Incomplete measured accounting")
    counts = Counter()
    pair_fields = {
        "request",
        "request_id",
        "status",
        "input_length_matches",
        "silicon_ttft_ms",
        "silicon_tpot_ms",
        "predicted_ttft_ms",
        "predicted_tpot_ms",
        "ttft_ms_ape",
        "tpot_ms_ape",
    }
    for index, pair in enumerate(run["pairs"]):
        if set(pair) != pair_fields or pair["request"] != index + 1 or pair["status"] not in PAIR_STATUSES:
            raise ValueError("Invalid request comparison")
        counts[pair["status"]] += 1
        if pair["input_length_matches"] is not None and type(pair["input_length_matches"]) is not bool:
            raise ValueError("Invalid request input comparison")
        for metric in ("ttft_ms", "tpot_ms"):
            actual, predicted = pair["silicon_" + metric], pair["predicted_" + metric]
            if any(v is not None and not finite(v) for v in (actual, predicted)):
                raise ValueError("Invalid request measurement")
            expected_error = (
                ape(actual, predicted) if pair["status"] == "matched" and run["outcome"] == "completed" else None
            )
            if pair[metric + "_ape"] != expected_error:
                raise ValueError("Request error does not reconcile")
    if any(run["coverage"].get(key, 0) != counts[key] for key in PAIR_STATUSES):
        raise ValueError("Request coverage does not reconcile")
    for metric in ("ttft_ms", "tpot_ms"):
        errors = [p[metric + "_ape"] for p in run["pairs"] if p[metric + "_ape"] is not None]
        expected = {"count": len(errors), "sum_ape": sum(errors), "mape": mean(errors)}
        if run["request_errors"][metric] != expected:
            raise ValueError("Request denominator does not reconcile")
    if run["outcome"] != "completed" and any(v is not None for v in run["prediction"].values()):
        raise ValueError("Unavailable runs cannot publish accuracy")
    for item in run["config_summary"]:
        if set(item) != {"name", "value"} or not isinstance(item["name"], str):
            raise ValueError("Invalid configuration projection")
        if item["value"] is not None and type(item["value"]) not in (str, int, float, bool):
            raise ValueError("Configuration must contain scalar settings")
    validate_report(run["report"])
    if run["report"] and run["report"].get("comparison"):
        comparison = run["report"]["comparison"]
        for metric in ("ttft_ms", "tpot_ms"):
            if comparison["phases"]["profile"]["request_scores"][metric] != run["request_errors"][metric]:
                raise ValueError("Chart and overview request scores disagree")
    return run
