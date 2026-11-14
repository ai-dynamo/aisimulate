# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX measurement identities, scoring, and public numerical contracts."""

import hashlib
import json
import math
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.e2e_accuracy.agentx_comparison import validate as validate_comparison
from scripts.e2e_accuracy.agentx_inputs import safe_path
from scripts.e2e_accuracy.benchmark_features import validate as validate_features
from scripts.e2e_accuracy.simulation_features import validate as validate_simulation

INPUT_SCHEMA = "aisimulate.agentx.inputs.v1"
SUMMARY_SCHEMA = "aisimulate.agentx.accuracy.v1"
INDEX = "catalog/e2e/agentx.json"
METRICS = ("tpot_ms", "ttft_ms", "output_tokens_s")
# The original, retained finite Qwen campaign predates complete silicon receipts.
# Keep this compatibility exception finite; new IDs and Weka runs must qualify.
HISTORICAL_SELF_COLLECTED = {
    f"self-p{length}-l{lanes}-{cache}-r{repeat}": (length, lanes, cache, repeat)
    for length in (4096, 32768)
    for lanes in (1, 4, 16)
    for cache in ("cold", "warm")
    for repeat in range(3)
}
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
    if b'"agentx_inputs.py"' in read_file("scripts/e2e_accuracy/agentx_contract.py"):
        names += ("agentx_inputs.py",)
        # Historical producers loaded cache validation from the collection tree.
        if b"importlib.util.spec_from_file_location" in read_file("scripts/e2e_accuracy/agentx_inputs.py"):
            names += ("../../benchmarks/evidence/accuracy/qwen-weka/cache_probe.py",)
    if b'"simulation_features.py"' in read_file("scripts/e2e_accuracy/agentx_contract.py"):
        names += ("simulation_features.py",)
    return hashlib.sha256(
        canonical(
            {
                name: hashlib.sha256(read_file(posixpath.normpath("scripts/e2e_accuracy/" + name))).hexdigest()
                for name in names
            }
        )
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


def native_completion_qualified(native, boundary):
    """Client cutoff completion may retain a canceled, already committed batch."""
    telemetry = native.get("telemetry", {})
    status = telemetry.get("completion_status")
    if status == "completed":
        return True
    if status != "server_unsettled" or boundary != "successful_profile_span":
        return False
    profile = native.get("agentic_profile") or {}
    if (
        profile.get("admission_closed") is not True
        or profile.get("cancel_drain_timed_out") is not False
        or profile.get("client_in_flight_requests") != 0
    ):
        return False
    times = [
        profile.get(key)
        for key in (
            "profile_start_ms",
            "admission_cutoff_ms",
            "response_grace_deadline_ms",
            "finished_at_ms",
            "cancel_drain_deadline_ms",
        )
    ]
    if not all(finite(value) for value in times) or times != sorted(times) or times[0] == times[1]:
        return False
    _, cutoff, grace, finished, _ = times
    boundaries = telemetry.get("phase_boundaries", [])
    if not any(p.get("phase") == "profile" and p.get("end_ms") == finished for p in boundaries) or any(
        p.get("phase") == "preparation" and p.get("completion_status") != "completed" for p in boundaries
    ):
        return False
    requests = [r for r in telemetry.get("requests", []) if r.get("phase") == "profile"]
    ids = {r.get("uuid") for r in requests}
    if not requests or None in ids or len(ids) != len(requests):
        return False
    canceled = {r["uuid"] for r in requests if r.get("terminal_status") == "canceled"}
    completed = [r for r in requests if r.get("terminal_status") == "completed"]
    counts = {
        "issued_requests": len(requests),
        "successful_responses": len(completed),
        "canceled_requests": len(canceled),
    }
    if (
        not canceled
        or len(completed) + len(canceled) != len(requests)
        or any(type(profile.get(key)) is not int or profile[key] != count for key, count in counts.items())
    ):
        return False
    for row in requests:
        arrival, terminal = row.get("dispatched_at_ms"), row.get("terminal_time_ms")
        if not finite(arrival) or not finite(terminal) or not arrival < cutoff or not arrival <= terminal <= finished:
            return False
        if row["uuid"] in canceled and not grace <= terminal:
            return False
    events = [e for e in telemetry.get("lifecycle", []) if e.get("phase") == "profile"]
    settled = {e.get("request_uuid") for e in events if e.get("event") == "quiescent"}
    acknowledged = {
        e.get("request_uuid") for e in events if e.get("event") == "causal_terminal" and e.get("status") == "canceled"
    }
    unsettled = ids - settled
    return (
        bool(unsettled)
        and unsettled <= canceled
        and canceled <= acknowledged
        and all(
            type(profile.get(key)) is int and profile[key] == len(unsettled)
            for key in ("server_unsettled_requests", "unsettled_server_requests")
        )
    )


def validate_catalog(catalog):
    if set(catalog) != {"schema", "runs", "snapshot_id"} or catalog["schema"] != INPUT_SCHEMA:
        raise ValueError("Invalid AgentX input catalog")
    snapshot = hashlib.sha256(canonical({k: v for k, v in catalog.items() if k != "snapshot_id"})).hexdigest()
    if catalog["snapshot_id"] != snapshot or not catalog["runs"]:
        raise ValueError("Invalid AgentX snapshot identity")
    ids = set()
    for run in catalog["runs"]:
        if set(run) != {"id", "category", "configuration_id", "manifest"} or not re.fullmatch(r"[a-z0-9-]+", run["id"]):
            raise ValueError("Invalid AgentX run identity")
        if run["id"] in ids:
            raise ValueError("Duplicate AgentX run")
        ids.add(run["id"])
        if run["category"] not in {"agentx", "self_collected"} or not re.fullmatch(
            r"cfg-[0-9a-f]{64}", run["configuration_id"]
        ):
            raise ValueError("Invalid AgentX configuration identity")
        entry = run["manifest"]
        if (
            set(entry) != {"path", "sha256", "bytes"}
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or type(entry["bytes"]) is not int
            or entry["bytes"] < 0
        ):
            raise ValueError("Invalid AgentX manifest checksum")
        safe_path(Path("."), entry["path"])
        source = "official" if run["category"] == "agentx" else "self-collected"
        if not entry["path"].startswith(f"data/e2e/agentx/{source}/") or not entry["path"].endswith(
            f"/{run['id']}/manifest.json"
        ):
            raise ValueError("Unsafe AgentX manifest path")
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
            reason = (
                failure
                if failure in {"context_overflow", "measurement_metrics_unavailable", "measurement_cancelled"}
                else "measurement_failed"
            )
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
        "measurement_campaign_status",
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
        "measurement_cancelled",
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
        "simulation_features",
    }

    def walk(item, key=None):
        if key == "simulation_features":
            validate_simulation(item)
            return
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
    "measurement_cancelled",
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
    if set(run) - {"op_data", "feature_metadata", "simulation_features"} != expected or run["outcome"] not in {
        "completed",
        "failed",
        "unsupported",
    }:
        raise ValueError("Invalid AgentX result fields")
    historical = HISTORICAL_SELF_COLLECTED.get(run["id"])
    legacy = (
        historical is not None
        and historical == tuple(run[k] for k in ("initial_input_tokens", "lanes", "cache_state", "repeat"))
        and (run["model"], run["hardware"], run["backend"], run["tensor_parallel"], run["boundary"])
        == ("Qwen/Qwen3.5-397B-A17B", "B300", "vllm", 8, "complete_finite_cohort")
    )
    complete_features = run["category"] == "self_collected" and not legacy
    if "feature_metadata" in run:
        validate_features(run["feature_metadata"], run["id"], require_complete=complete_features)
    elif complete_features:
        raise ValueError("New self-collected publications require silicon feature metadata")
    if "simulation_features" in run:
        validate_simulation(run["simulation_features"], run["id"])
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
