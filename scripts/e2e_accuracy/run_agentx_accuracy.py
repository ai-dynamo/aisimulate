# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch, replay, score, and qualify the reviewed AgentX cohort."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.e2e_accuracy.agentx_comparison import build as build_comparison
from scripts.e2e_accuracy.agentx_contract import (
    IDENTITY,
    INDEX,
    SUMMARY_SCHEMA,
    canonical,
    digest,
    driver_digest,
    native_completion_qualified,
    score,
    validate_catalog,
    validate_public,
    validate_result,
)
from scripts.e2e_accuracy.agentx_inputs import (
    CORPORA,
    checked,
    document,
    finite_trace,
    load_runs,
    reconstruct,
    replay_config,
    safe_path,
)
from scripts.e2e_accuracy.simulation_features import compare_speculation
from scripts.e2e_accuracy.simulation_features import unavailable as unavailable_features
from scripts.e2e_accuracy.simulation_features import validate as validate_simulation

REPO = "ai-dynamo/aisim-infx-cache"


def read(path):
    return json.loads(Path(path).read_bytes())


def fetch(revision, output, *, metadata_only=False, campaign_catalog=None, preview=False):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("AgentX requires an immutable HF commit")
    catalog = validate_catalog(read(hf_hub_download(REPO, INDEX, repo_type="dataset", revision=revision)))
    if campaign_catalog:
        if not preview or not re.fullmatch(r"catalog/e2e/agentx/qualifications/[a-z0-9-]+\.json", campaign_catalog):
            raise ValueError("Incomplete measurements require an explicit trusted preview")
        campaign_source = read(hf_hub_download(REPO, campaign_catalog, repo_type="dataset", revision=revision))
        if (
            campaign_source.get("schema") != "aisimulate.agentx.silicon-qualification.v1"
            or campaign_source.get("scored") is not False
            or campaign_source.get("status") not in {"failed", "stopped"}
        ):
            raise ValueError("Not an unscored incomplete silicon campaign")
        catalog = {"schema": catalog["schema"], "runs": [*catalog["runs"], *campaign_source["runs"]]}
        catalog["snapshot_id"] = hashlib.sha256(canonical(catalog)).hexdigest()
        validate_catalog(catalog)
    output.mkdir(parents=True, exist_ok=True)
    if campaign_catalog:
        (output / "measurement-status.json").write_bytes(canonical({"measurement_campaign_status": "partial"}))

    def download(name, entry=None):
        target = safe_path(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=revision))
        shutil.copyfile(source, target)
        if entry is not None:
            checked(output, {**entry, "path": name})
        return target

    registry = read(download(CORPORA))
    if registry.get("schema") != "aisimulate.agentx.corpora.v1":
        raise ValueError("Invalid corpus registry")
    roles = {
        "configuration",
        "workload",
        "observations",
        "overrides",
        "feature_metadata",
        "cache_receipt",
        "runtime_receipt",
        "source_manifest",
        "recipe",
        "feature_point_receipt",
        "feature_harness_receipt",
        "snapshot_receipt",
        "harness_receipt",
    }
    corpora = set()
    for entry in catalog["runs"]:
        path = download(entry["manifest"]["path"], entry["manifest"])
        run = read(path)
        if any(run.get(key) != entry[key] for key in ("id", "category", "configuration_id")):
            raise ValueError("AgentX manifest identity mismatch")
        if run.get("corpus_id") is not None:
            corpora.add(run["corpus_id"])
        for role, artifact_entry in run["artifacts"].items():
            if role in roles or (
                role == "raw" and run.get("boundary") == "complete_finite_cohort" and not metadata_only
            ):
                name = safe_path(path.parent, artifact_entry["path"]).relative_to(output).as_posix()
                download(name, artifact_entry)
        feature_document = read(
            checked(
                output,
                {
                    **run["artifacts"]["feature_metadata"],
                    "path": (path.parent / run["artifacts"]["feature_metadata"]["path"]).relative_to(output).as_posix(),
                },
            )
        )
        configuration = read(
            checked(
                output,
                {
                    **run["artifacts"]["configuration"],
                    "path": (path.parent / run["artifacts"]["configuration"]["path"]).relative_to(output).as_posix(),
                },
            )
        )
        hashes = {ref["sha256"] for row in feature_document["features"] for ref in row["evidence"]}
        hashes.update(
            ref["sha256"] for ref in configuration.get("requested", []) + configuration.get("effective_evidence", [])
        )
        for value in run["artifacts"].values():
            if value["sha256"] in hashes:
                name = safe_path(path.parent, value["path"]).relative_to(output).as_posix()
                if not (output / name).exists():
                    download(name, value)
    for identity in sorted(corpora) if not metadata_only else []:
        descriptor = registry["sources"][identity]
        source = Path(
            hf_hub_download(
                descriptor["repository"], descriptor["filename"], repo_type="dataset", revision=descriptor["revision"]
            )
        )
        target = safe_path(output / "corpora", identity + ".jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        reconstruct(descriptor, source, target)
    (output / INDEX).parent.mkdir(parents=True, exist_ok=True)
    (output / INDEX).write_bytes(canonical(catalog))
    load_runs(output, catalog)


def artifact(root, run, name):
    return checked(root, run["artifacts"][name])


def prepare_trace(root, run, output):
    target = output / "traces.jsonl"
    if run["corpus_id"] is None:
        finite_trace(root, run, target)
        return target
    source = safe_path(root / "corpora", run["corpus_id"] + ".jsonl")
    if "overrides" not in run["artifacts"]:
        shutil.copyfile(source, target)
        return target
    changes = read(artifact(root, run, "overrides"))
    by_play = {}
    identities = set()
    for change in changes:
        key = (change["play"], change["outer"], change["inner"])
        if key in identities:
            raise ValueError("Duplicate observed output substitution")
        identities.add(key)
        by_play.setdefault(change["play"], []).append(change)
    applied = 0
    with source.open("rb") as incoming, target.open("wb") as outgoing:
        for line in incoming:
            play = json.loads(line)
            replacements = by_play.pop(play["id"], [])
            for change in replacements:
                row = play["requests"][change["outer"]]
                if change["inner"] is not None:
                    row = row["requests"][change["inner"]]
                row["out"] = change["output"]
                applied += 1
            outgoing.write(canonical(play) if replacements else line)
    if by_play or applied != len(changes):
        raise ValueError("Observed requests are missing from the replay corpus")
    return target


def normalize_native(native, observations, category, boundary="complete_finite_cohort"):
    phases = {row["key"]: row["phase"] for row in observations}
    rows = native.get("telemetry", {}).get("requests", native.get("per_request", []))
    result = []
    for row in rows:
        key = row.get("request_id") or row.get("agentic", {}).get("request_id")
        phase = row.get("phase", row.get("agentic_phase", "profile"))
        if category == "self_collected" and boundary == "complete_finite_cohort":
            phase = phases.get(key, "profile")
        result.append(
            {
                "key": key,
                "phase": "profile" if phase in {"profile", "profiling"} else "preparation",
                "status": row.get("terminal_status", "unknown"),
                "start_ms": row.get("arrival_time_ms"),
                "end_ms": row.get("terminal_time_ms"),
                "input_length": row.get("input_length"),
                "output_length": row.get("output_length"),
                "ttft_ms": row.get("ttft_ms"),
                "tpot_ms": row.get("itl_ms") if (row.get("output_length") or 0) > 1 else None,
                "cached_tokens": row.get("first_admission_reused_input_tokens"),
            }
        )
    return result


def public_report(payload, observations, category, boundary="complete_finite_cohort"):
    """Explicit projections, including nested telemetry; never publish raw JSON."""

    def opaque(value):
        return hashlib.sha256(str(value).encode()).hexdigest()[:16] if value is not None else None

    request_fields = (
        "phase",
        "arrival_time_ms",
        "terminal_time_ms",
        "first_token_ms",
        "input_length",
        "output_length",
        "ttft_ms",
        "itl_ms",
        "e2e_latency_ms",
        "terminal_status",
        "admission_count",
        "readmission_count",
        "reused_input_tokens",
        "first_admission_reused_input_tokens",
        "turn_index",
    )
    phases = {row["key"]: row["phase"] for row in observations}
    requests = []
    for raw in payload["requests"]:
        row = {key: raw[key] for key in request_fields if key in raw}
        row.update(uuid=opaque(raw.get("uuid")), request_id=opaque(raw.get("request_id")))
        if (
            category == "self_collected"
            and boundary == "complete_finite_cohort"
            and phases.get(raw.get("request_id")) == "preparation"
        ):
            row["phase"] = "warmup"
        agentic = raw.get("agentic", {})
        row["agentic"] = {
            key: opaque(agentic[key]) for key in ("play_id", "conversation_id", "request_id") if key in agentic
        }
        for history in ("routing_history", "admission_history"):
            row[history] = [
                {key: item[key] for key in ("at_ms", "pool", "worker_id") if key in item}
                for item in raw.get(history, [])
            ]
        requests.append(row)
    sample_fields = (
        "t",
        "start",
        "phase",
        "kind",
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
    )
    worker_fields = (
        "worker_id",
        "dp_rank",
        "role",
        "metric_role",
        "active_blocks",
        "inactive_blocks",
        "total_blocks",
        "kv_cache_used_bytes",
        "kv_cache_capacity_bytes",
        "active_cache_usage",
        "physical_cache_usage",
        "running_requests",
        "waiting_requests",
    )
    samples = []
    for raw in payload["samples"]:
        sample = {key: raw[key] for key in sample_fields if key in raw}
        sample["workers"] = [
            {key: worker[key] for key in worker_fields if key in worker} for worker in raw.get("workers", [])
        ]
        sample["host_pools"] = []
        samples.append(sample)
    boundaries = {
        key: {
            name: phase[name]
            for name in ("start_ms", "end_ms", "completion_status", "admission_cutoff_ms")
            if name in phase
        }
        for key, phase in payload["phases"].items()
    }
    if category == "self_collected" and boundary == "complete_finite_cohort":
        profile = [r for r in requests if r["phase"] == "profile"]
        preparation = [r for r in requests if r["phase"] != "profile"]
        cutoff = min(r["arrival_time_ms"] for r in profile) if profile else 0
        if preparation:
            boundaries["preparation"] = {"start_ms": 0, "end_ms": cutoff, "completion_status": "completed"}
            boundaries["profile"]["start_ms"] = cutoff
            # Do not assign a sample spanning the barrier to either phase.
            samples = [s for s in samples if s["t"] * 1000 <= cutoff or s["start"] * 1000 >= cutoff]
            for sample in samples:
                sample["phase"] = "preparation" if sample["t"] * 1000 <= cutoff else "profile"
    metadata = {
        key: payload["metadata"].get(key)
        for key in ("title", "model", "hardware", "framework", "backend_version", "topology", "plotly_python_version")
    }
    telemetry = {
        key: payload["telemetry"].get(key)
        for key in ("completion_status", "sample_interval_ms", "clock", "clock_origin_ms", "profile_start_ms")
    }
    lifecycle = [
        {
            "at_ms": event["at_ms"],
            "phase": event["phase"],
            "event": event["event"],
            "request_uuid": opaque(event.get("request_uuid")),
            "play_id": opaque(event.get("play_id")),
        }
        for event in payload.get("lifecycle", [])
    ]
    return validate_public(
        {
            "schema": payload["schema"],
            "metadata": metadata,
            "prediction": {},
            "telemetry": telemetry,
            "phases": boundaries,
            "requests": requests,
            "samples": samples,
            "lifecycle": lifecycle,
        }
    )


def run_one(root, run, directory, timeout, context):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "feature-context.json").write_bytes(canonical(context))
    observations = read(artifact(root, run, "observations"))
    if sum(r["phase"] == "profile" for r in observations) != run["measured_requests"]:
        raise ValueError("Measured cohort count mismatch")
    trace = prepare_trace(root, run, directory)
    config = replay_config(root, run)
    config["traffic"]["source"]["paths"] = [str(trace.resolve())]
    config_path = directory / "input-config.json"
    config_path.write_bytes(canonical(config))
    with (directory / "replay.log").open("w") as log:
        try:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("agentx_predict.py")), str(config_path), str(directory)],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            outcome = (
                read(directory / "outcome.json")
                if (directory / "outcome.json").exists()
                else {"outcome": "failed", "reason": "replay_process_failed"}
            )
            if process.returncode:
                outcome = {"outcome": "failed", "reason": "replay_process_failed"}
        except subprocess.TimeoutExpired:
            outcome = {"outcome": "failed", "reason": "replay_timeout"}
    normalized, report = [], None
    if outcome["outcome"] == "completed":
        native = read(directory / "native.json")
        normalized = normalize_native(native, observations, run["category"], run["boundary"])
        report = public_report(read(directory / "dashboard.json"), observations, run["category"], run["boundary"])
        if not native_completion_qualified(native, run["boundary"]):
            outcome = {**outcome, "outcome": "failed", "reason": "replay_incomplete"}
    scores = score(observations, normalized, run["boundary"])
    cache_alignment = None
    if run["boundary"] == "complete_finite_cohort" and normalized:
        actual = {r["key"]: r for r in observations}
        cache_alignment = len(normalized) == len(actual) and all(
            r["key"] in actual and r["cached_tokens"] == actual[r["key"]]["cached_tokens"] for r in normalized
        )
        if not cache_alignment:
            outcome = {**outcome, "outcome": "failed", "reason": "cache_alignment_failed"}
    if outcome["outcome"] != "completed":
        unavailable = score(observations, [], run["boundary"])
        scores.update(
            prediction=unavailable["prediction"],
            point_errors=unavailable["point_errors"],
            request_errors=unavailable["request_errors"],
        )
        for pair in scores["pairs"]:
            pair.update(ttft_ms_ape=None, tpot_ms_ape=None)
    reference = document(root, run, "workload").get("silicon_reference")
    if report and reference:
        report["comparison"] = build_comparison(
            reference,
            report,
            read(directory / "dashboard.json"),
            scores,
            outcome["outcome"] == "completed",
        )
    worker = config["engine"]["workers"]["aggregated"]
    settings = {
        "Estimation": "op_level",
        "Timing correction": False,
        "Context length": config["engine"].get("context_length"),
        "Maximum batched tokens": worker["scheduler"].get("max_batched_tokens"),
        "Maximum sequences": worker["scheduler"].get("max_sequences"),
        "KV block size": worker["kv_cache"].get("block_size"),
        "KV blocks": worker["kv_cache"]["capacity"].get("blocks"),
        "Prefix caching": worker["kv_cache"].get("prefix_caching"),
        "Speculative tokens": config["engine"].get("nextn", 0),
        "Accepted length": config["engine"].get("nextn_accepted"),
        "Profile seconds": config["traffic"]["load"].get("agentic_profile", {}).get("duration_seconds"),
    }
    result = {
        "config_summary": [{"name": k, "value": v} for k, v in settings.items()],
        **{key: run[key] for key in IDENTITY},
        **scores,
        **{key: outcome.get(key) for key in ("outcome", "reason", "prediction_backend_version")},
        "cache_alignment": cache_alignment,
        "report": report,
        "op_data": outcome.get("op_data", []),
    }
    if "feature_metadata" in run["artifacts"]:
        result["feature_metadata"] = read(artifact(root, run, "feature_metadata"))
    features_path = directory / "simulation-features.json"
    features = (
        validate_simulation(read(features_path), run["id"])
        if features_path.exists()
        else unavailable_features(
            context, run, "unsupported" if outcome["outcome"] == "unsupported" else "failed_to_record"
        )
    )
    if outcome["outcome"] == "completed" and any(
        row["state"] == "failed_to_record" for role in features["roles"] for row in role["features"]
    ):
        raise ValueError("Newly completed runs require determinate simulation feature settings")
    if "feature_metadata" in result:
        features = compare_speculation(
            features,
            result["feature_metadata"],
            run,
            document(root, run, "recipe") if "recipe" in run["artifacts"] else None,
        )
    result["simulation_features"] = features
    validate_result(result)
    (directory / "result.json").write_bytes(canonical(result))
    trace.unlink()  # Large derived corpora are reproducible from frozen inputs.
    return result


def campaign(args):
    if args.wheel is not None:
        from scripts.e2e_accuracy.run_e2e_accuracy import wheel_identity

        if wheel_identity(args.wheel)["wheel_sha256"] != args.wheel_sha256:
            raise ValueError("AgentX runtime differs from the evaluated wheel")
    elif not args.preview:
        raise ValueError("Published AgentX campaigns require the evaluated wheel")
    catalog = validate_catalog(read(args.inputs / INDEX))
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [
        run for index, run in enumerate(load_runs(args.inputs, catalog)) if index % args.shard_count == args.shard_index
    ]
    evaluated_driver = driver_digest()
    results = [
        run_one(
            args.inputs,
            run,
            args.output / run["id"],
            args.point_timeout,
            {
                "benchmark_id": run["id"],
                "evaluated_revision": args.commit,
                "dataset_revision": args.revision,
            },
        )
        for run in selected
    ]
    snapshot = {
        "branch": args.branch,
        "commit_sha": args.commit,
        "dataset_revision": args.revision,
        "wheel_sha256": args.wheel_sha256,
        "driver_sha256": evaluated_driver,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
        "cohort_sha256": catalog["snapshot_id"],
        "preview": args.preview,
    }
    status = args.inputs / "measurement-status.json"
    if status.exists():
        if not args.preview or read(status) != {"measurement_campaign_status": "partial"}:
            raise ValueError("Partial measurement campaigns cannot publish")
        snapshot.update(read(status))
    (args.output / f"shard-{args.shard_index}.json").write_bytes(canonical({"snapshot": snapshot, "runs": results}))


def qualify(args):
    catalog = validate_catalog(read(args.inputs / INDEX))
    runs, snapshot = [], None
    for index in range(args.shard_count):
        shard = read(args.shards / f"shard-{index}.json")
        if snapshot is not None and snapshot != shard["snapshot"]:
            raise ValueError("AgentX shards evaluated different revisions")
        snapshot = shard["snapshot"]
        runs.extend(shard["runs"])
    if len({r["id"] for r in runs}) != len(runs) or {r["id"] for r in runs} != {r["id"] for r in catalog["runs"]}:
        raise ValueError("Incomplete or duplicate AgentX cohort")
    if snapshot["driver_sha256"] != driver_digest():
        raise ValueError("AgentX qualification driver differs from replay driver")
    if snapshot["cohort_sha256"] != catalog["snapshot_id"]:
        raise ValueError("AgentX cohort digest differs")
    status = args.inputs / "measurement-status.json"
    if snapshot.get("measurement_campaign_status"):
        if (
            snapshot.get("preview") is not True
            or not status.exists()
            or read(status) != {"measurement_campaign_status": "partial"}
        ):
            raise ValueError("Partial measurement campaigns cannot publish")
    elif status.exists():
        raise ValueError("Measurement campaign status differs from replay")
    snapshot["completed_at"] = datetime.now(UTC).isoformat()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    sources = {r["id"]: r for r in load_runs(args.inputs, catalog)}
    for run in runs:
        validate_result(run)
        features = run.get("simulation_features")
        if (
            features is None
            or features["evaluated_revision"] != snapshot["commit_sha"]
            or features["dataset_revision"] != snapshot["dataset_revision"]
        ):
            raise ValueError("Simulation feature evidence differs from the evaluated snapshot")
        if run["outcome"] == "completed" and (
            features["execution_mode"] == "unavailable"
            or features["configuration_sha256"] is None
            or any(
                role["compiled_sha256"] is None or any(row["state"] == "failed_to_record" for row in role["features"])
                for role in features["roles"]
            )
        ):
            raise ValueError("Completed runs require effective simulation feature evidence")
        source = sources[run["id"]]
        if any(run[key] != source[key] for key in IDENTITY):
            raise ValueError("Result identity differs from the reviewed catalog")
        if "feature_metadata" in run and run["feature_metadata"] != document(args.inputs, source, "feature_metadata"):
            raise ValueError("Result silicon features differ from retained evidence")
        relative = "details/" + run["id"] + ".json"
        detail = {"schema": SUMMARY_SCHEMA, "snapshot": snapshot, **run}
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical(detail))
        summaries.append(
            {key: value for key, value in run.items() if key not in {"report", "pairs"}} | {"detail_path": relative}
        )
    summary = {"schema": SUMMARY_SCHEMA, "snapshot": snapshot, "runs": summaries}
    validate_public(summary)
    (args.output / "summary.json").write_bytes(canonical(summary))
    files = {p.relative_to(args.output).as_posix(): digest(p) for p in args.output.rglob("*.json")}
    (args.output / "qualification.json").write_bytes(
        canonical(
            {"schema": "aisimulate.agentx.qualification.v1", "status": "complete", "snapshot": snapshot, "files": files}
        )
    )
    print(json.dumps({"runs": len(runs), "outcomes": dict(Counter(r["outcome"] for r in runs))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "run", "qualify"])
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--campaign-catalog", help="Explicit unscored silicon campaign to include in a partial preview")
    parser.add_argument(
        "--metadata-only", action="store_true", help="Fetch bundle metadata for qualification without replay corpora"
    )
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit")
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--run-id", default="0")
    parser.add_argument("--run-attempt", type=int, default=1)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--point-timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard selection")
    if args.action == "fetch":
        fetch(
            args.revision,
            args.output,
            metadata_only=args.metadata_only,
            campaign_catalog=args.campaign_catalog,
            preview=args.preview,
        )
    elif args.action == "run":
        campaign(args)
    else:
        qualify(args)
