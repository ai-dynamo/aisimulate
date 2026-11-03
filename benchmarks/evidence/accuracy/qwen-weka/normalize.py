#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize retained AgentX observations without fitting clocks or changing prompts."""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import blake3

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.e2e_accuracy.benchmark_features import (
    add_agentx_receipt,
    add_branch_receipt,
    add_preemption_receipt,
    from_vllm_receipt,
    validate as validate_features,
)

CORPUS_SHA = "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def peak(value):
    if not isinstance(value, dict):
        return 0
    values = [value["in"] + value["out"]] if type(value.get("in")) is int and type(value.get("out")) is int else [0]
    for child in value.values():
        if isinstance(child, dict):
            values.append(peak(child))
        elif isinstance(child, list) and child and isinstance(child[0], dict):
            values.extend(peak(item) for item in child)
    return max(values)


def prepare_corpus(source, output):
    output.mkdir(parents=True, exist_ok=False)
    namespaces, excluded = {}, []
    hasher = hashlib.sha256()
    scanned = 0
    with source.open("rb") as incoming, (output / "traces.jsonl").open("xb") as outgoing:
        for line in incoming:
            hasher.update(line)
            scanned += 1
            row = json.loads(line)
            if peak(row) > 262144:
                excluded.append(row["id"])
                continue
            index = len(namespaces)
            namespaces[row["id"]] = "weka:" + blake3.blake3(f"traces.jsonl#{index:06}".encode()).hexdigest()[:16]
            outgoing.write(line)
    if hasher.hexdigest() != CORPUS_SHA or scanned != 393 or len(namespaces) != 220:
        raise ValueError("Corpus checksum or eligibility differs from the reviewed protocol")
    write(output / "namespace-map.json", namespaces)
    write(
        output / "corpus-manifest.json",
        {
            "source_sha256": CORPUS_SHA,
            "dataset_revision": "23f152f6f0f9399a85901b89a6458def0ef16729",
            "filtered_sha256": digest(output / "traces.jsonl"),
            "eligible": list(namespaces),
            "excluded": excluded,
            "context_limit": 262144,
        },
    )


def metric(record, name, unit=None):
    item = record.get("metrics", {}).get(name)
    if item is None:
        return None
    if not isinstance(item, dict) or (unit is not None and item.get("unit") != unit):
        raise ValueError(f"Unexpected metric encoding or unit: {name}")
    value = item.get("value")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    if unit == "tokens":
        return int(value) if float(value).is_integer() else None
    return value


def reference_row(row, namespaces):
    meta = row["metadata"]
    source, outer, inner = (
        meta.get("source_trace_id"),
        meta.get("source_outer_idx"),
        meta.get("source_inner_idx"),
    )
    lineage = source in namespaces and type(outer) is int
    key = (
        f"{namespaces[source]}:request:outer:{outer}" + (f":inner:{inner}" if inner is not None else "")
        if lineage
        else f"unresolved:{meta['x_request_id']}"
    )
    output = metric(row, "output_sequence_length", "tokens")
    input_length = metric(row, "input_sequence_length", "tokens")
    if input_length is None:
        input_length = metric(row, "error_isl", "tokens")
    failed = bool(row.get("error") or meta.get("was_cancelled") or meta.get("context_overflow_skip") or output is None)
    decode = metric(row, "full_decode_duration", "ms")
    phase = (
        "profile" if meta.get("phase_kind", meta.get("benchmark_phase")) in {"profile", "profiling"} else "preparation"
    )
    return {
        "key": key,
        "identity_valid": lineage,
        "request_id": meta.get("x_request_id"),
        "phase": phase,
        "source_trace_id": source,
        "source_outer_idx": outer,
        "source_inner_idx": inner,
        "conversation_id": meta.get("conversation_id"),
        "session_id": meta.get("x_correlation_id"),
        "start_ns": str(meta["request_start_ns"]),
        "agent_depth": meta.get("agent_depth"),
        "end_ns": str(meta["request_end_ns"]) if meta.get("request_end_ns") is not None else None,
        "input_tokens": input_length,
        "output_tokens": output,
        "ttft_ms": metric(row, "time_to_first_token", "ms"),
        "tpot_ms": decode / (output - 1) if decode is not None and output is not None and output > 1 else None,
        "e2e_ms": metric(row, "request_latency", "ms"),
        "cached_tokens": metric(row, "usage_prompt_cache_read_tokens", "tokens"),
        "error": (
            "context_overflow"
            if meta.get("context_overflow_skip")
            else "measurement_metrics_unavailable"
            if output is None and not row.get("error")
            else "measurement_failed"
        )
        if failed
        else None,
    }


def references(profile, namespaces):
    with profile.open() as stream:
        return [reference_row(json.loads(line), namespaces) for line in stream]


def add_stream_observations(references, raw, namespaces):
    """Retain observed token counts/times only when every output token reconciles."""
    by_id = {row["request_id"]: row for row in references}
    if len(by_id) != len(references):
        raise ValueError("Duplicate measured request UUID")
    seen = set()
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["metadata"]["x_request_id"]
            if identity in seen:
                raise ValueError("Raw stream record has duplicate request identity")
            if identity not in by_id:
                retained = reference_row(row, namespaces)
                references.append(retained)
                by_id[identity] = retained
            seen.add(identity)
            reference = by_id[identity]
            chunks = []
            for response in row.get("responses", []):
                for packet in response.get("packets", []):
                    if packet.get("name") != "data" or packet.get("value") in (None, "[DONE]"):
                        continue
                    event = json.loads(packet["value"])
                    usage = event.get("usage") or {}
                    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    if type(cached) is int and cached >= 0 and reference["cached_tokens"] is None:
                        reference["cached_tokens"] = cached
                    count = 0
                    for choice in event.get("choices", []):
                        tokens = choice.get("token_ids", (choice.get("delta") or {}).get("token_ids"))
                        if tokens is not None:
                            if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                                raise ValueError("Invalid recorded output token IDs")
                            count += len(tokens)
                    if count:
                        when = int(reference["start_ns"]) + int(response["perf_ns"]) - int(row["start_perf_ns"])
                        chunks.append({"at_ns": str(when), "count": count})
            reference["chunks"] = (
                chunks if sum(chunk["count"] for chunk in chunks) == reference["output_tokens"] else []
            )
    if seen != set(by_id):
        raise ValueError("Raw streams do not account for every measured/preparation request")


def counter_receipt(path, start, end):
    previous, intervals, excluded = None, [], 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            matches = re.findall(
                r"^vllm:num_preemptions_total\{[^\n]*\} ([0-9.eE+-]+)$", row.get("text", ""), re.MULTILINE
            )
            current = None
            if len(matches) == 1:
                count = float(matches[0])
                if math.isfinite(count) and count.is_integer() and count >= 0:
                    current = (int(row["end_ns"]), int(count))
            if previous is not None and current is not None:
                left, before = previous
                right, after = current
                if left >= start and right <= end and 0 < right - left <= 1_500_000_000 and after >= before:
                    intervals.append({"start_ns": str(left), "end_ns": str(right), "preemptions": after - before})
                elif left < end and right > start:
                    excluded += 1
            previous = current
    return {
        "profile_start_ns": str(start),
        "profile_end_ns": str(end),
        "intervals": intervals,
        "excluded_intervals": excluded,
    }


def retain_cancelled_accounting(references, log, ranges):
    """Account for credits cancelled without a request record; invent no timings or lineage."""
    phases = {}
    for match in re.finditer(r"Received CREDIT_PHASE_COMPLETE[^\n]*PhaseRecordsStats\(([^\n]*)\)", log):
        text = match[1]
        kind = re.search(r"phase_kind='(warmup|profiling)'", text)
        if kind is None:
            continue
        raw_phase = kind[1]
        phase = "profile" if raw_phase == "profiling" else "preparation"
        stats = {}
        for key in (
            "start_ns",
            "sent_end_ns",
            "requests_end_ns",
            "final_requests_completed",
            "final_requests_cancelled",
            "final_request_errors",
        ):
            value = re.search(r"\b" + key + r"=(\d+)(?:,|\))", text + ")")
            if value is None:
                raise ValueError("Missing final harness accounting: " + key)
            stats[key] = int(value[1])
        if stats["start_ns"] != int(ranges[raw_phase]["start_ns"]) or stats["requests_end_ns"] != int(
            ranges[raw_phase]["end_ns"]
        ):
            raise ValueError("Harness credit and metrics phase boundaries differ")
        if phase in phases and phases[phase] != stats:
            raise ValueError("Conflicting final harness accounting")
        phases[phase] = stats
    if "profile" not in phases:
        raise ValueError("Missing final structured profile accounting receipt")
    for match in re.finditer(
        r"Phase (warmup|profiling) \(\1\) complete \| completed=(\d+), cancelled=(\d+), errors=(\d+)", log
    ):
        raw_phase = match[1]
        phase = "profile" if raw_phase == "profiling" else "preparation"
        values = {
            "final_requests_completed": int(match[2]),
            "final_requests_cancelled": int(match[3]),
            "final_request_errors": int(match[4]),
        }
        if phase in phases:
            if any(phases[phase][key] != value for key, value in values.items()):
                raise ValueError("Final phase summaries disagree")
        else:
            phases[phase] = {
                **values,
                "start_ns": int(ranges[raw_phase]["start_ns"]),
                "requests_end_ns": int(ranges[raw_phase]["end_ns"]),
                "sent_end_ns": None,
            }
    if set(phases) != {"profile", "preparation"}:
        raise ValueError("Missing final warmup/profile accounting receipts")
    for phase, stats in phases.items():
        actual = sum(row["phase"] == phase for row in references)
        expected = stats["final_requests_completed"] + stats["final_requests_cancelled"]
        missing = expected - actual
        if missing < 0 or missing > stats["final_requests_cancelled"]:
            raise ValueError("Raw requests do not reconcile with final harness credit accounting")
        stats["observed_request_records"] = actual
        stats["cancelled_without_request_record"] = missing
        for index in range(missing):
            identity = f"unrecorded-cancellation-{phase}-{index + 1}"
            references.append(
                {
                    "key": identity,
                    "identity_valid": False,
                    "request_id": identity,
                    "phase": phase,
                    "source_trace_id": None,
                    "source_outer_idx": None,
                    "source_inner_idx": None,
                    "conversation_id": None,
                    "session_id": None,
                    "agent_depth": None,
                    "start_ns": None,
                    "end_ns": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "ttft_ms": None,
                    "tpot_ms": None,
                    "e2e_ms": None,
                    "cached_tokens": None,
                    "error": "measurement_cancelled",
                    "chunks": [],
                }
            )
    return phases


def normalize(run, corpus, template, output, taxonomy_revision, duration, silicon=None):
    output.mkdir(parents=True, exist_ok=False)
    namespaces = read(corpus / "namespace-map.json")
    raw = run / "benchmark/profile_export.jsonl"
    refs = references(raw, namespaces)
    streams = run / "benchmark/profile_export_raw.jsonl"
    add_stream_observations(refs, streams, namespaces)
    if not any(row["phase"] == "profile" for row in refs):
        raise ValueError("Measured profile is empty")
    export = read(run / "benchmark/server_metrics_export.json")
    ranges = export["summary"]["phase_time_ranges"]
    log = (run / "client.log").read_text()
    accounting = retain_cancelled_accounting(refs, log, ranges)
    write(output / "client-log.json", {"source_sha256": digest(run / "client.log"), "text": log})
    write(output / "phase-accounting.json", {"source_sha256": digest(output / "client-log.json"), "phases": accounting})
    start = int(ranges["profiling"]["start_ns"])
    end = int(ranges["profiling"]["end_ns"])
    harness = read(run / "client-record/resolved-dataset.json")
    elapsed = (end - start) / 1e9
    if harness["phases"][0]["duration"] != duration or not duration - 1 <= elapsed <= duration + 30 + 10 + 1:
        raise ValueError("AgentX profile window or original grace/drain bounds differ")
    receipt = read(run / "cache-receipt.json")
    observations, overrides = [], []
    origin = min(int(r["start_ns"]) for r in refs if r["start_ns"] is not None)
    for row in refs:
        observations.append(
            {
                "key": row["key"],
                "identity_valid": row["identity_valid"],
                "phase": row["phase"],
                "status": "failed" if row["error"] else "completed",
                "failure_reason": row["error"],
                "start_ms": (int(row["start_ns"]) - origin) / 1e6 if row["start_ns"] is not None else None,
                "end_ms": (int(row["end_ns"]) - origin) / 1e6 if row["end_ns"] else None,
                "input_length": row["input_tokens"],
                "output_length": row["output_tokens"],
                "ttft_ms": row["ttft_ms"],
                "tpot_ms": row["tpot_ms"],
                "e2e_ms": row["e2e_ms"],
                "cached_tokens": row["cached_tokens"],
            }
        )
        if row["phase"] == "profile" and not row["error"] and row["identity_valid"]:
            overrides.append(
                {
                    "play": row["source_trace_id"],
                    "outer": row["source_outer_idx"],
                    "inner": row["source_inner_idx"],
                    "output": int(row["output_tokens"]),
                }
            )
    selections, selected_indices = {}, {}
    corpus_order = list(namespaces)
    cuts = [json.loads(line) for line in (run / "client-record/trajectory-cuts.jsonl").read_text().splitlines()]
    for cut in cuts:
        lane = cut["lane_index"]
        if lane is not None and lane not in selections:
            root = cut["root_id"]
            selected_indices[lane] = corpus_order.index(root)
            selections[lane] = {
                "source_play_id": f"{namespaces[root]}:play:{root}",
                "t_star_ms": cut["snapshot"]["t_star_ms"],
            }
    lanes = read(run / "outcome.json")["lanes"]
    if set(selections) != set(range(lanes)):
        raise ValueError("Missing recorded initial lane selections")
    indices = [selected_indices[i] for i in range(lanes)]
    if indices != sorted(set(indices)):
        raise ValueError("Recorded initial selections do not follow the sequential corpus order")
    config = read(template)
    config["prediction_database_version"] = "0.31.0"
    config["engine"].update(estimation_mode="op_level", fallback_policy="deny", database_mode="SILICON")
    config["engine"]["workers"]["aggregated"]["timing"] = {"type": "default"}
    config["engine"].setdefault("estimator_config", {}).setdefault("correction", {})["enabled"] = False
    config["engine"]["context_length"] = 262144
    config["engine"]["workers"]["aggregated"]["kv_cache"] = receipt["manual_kv_cache"]
    config["traffic"] = {
        "source": {
            "type": "trace",
            "format": "weka",
            "paths": ["traces.jsonl"],
            "block_size": 64,
            "nested_timestamp_basis": "absolute",
        },
        "load": {
            "type": "trace_timestamps",
            "agentic_lanes": lanes,
            "agentic_warmup": True,
            "agentic_warmup_policy": "live_trajectory_v1",
            "agentic_snapshot": {
                "seed": 42,
                "corpus_cursor": indices[-1] + 1,
                "initial_selections": [selections[i] for i in range(lanes)],
            },
            "agentic_profile": {
                "duration_seconds": duration,
                "global_idle_cap_seconds": 10,
                "tree_idle_cap_seconds": 300,
                "response_grace_seconds": 30,
                "cancel_drain_seconds": 10,
            },
        },
    }
    taxonomy = {"revision": taxonomy_revision, "sha256": digest(ROOT / "pages/feature-support-matrix/features.toml")}
    identity = f"qwen-weka-l{lanes}"
    features = from_vllm_receipt(identity, receipt, digest(run / "cache-receipt.json"), taxonomy)
    write(output / "request-receipt.json", {"requests": refs})
    write(output / "snapshot-receipt.json", cuts)
    features = add_agentx_receipt(
        features,
        read(run / "client-record/resolved-dataset.json"),
        digest(run / "client-record/resolved-dataset.json"),
        cuts,
        digest(output / "snapshot-receipt.json"),
        refs,
        digest(output / "request-receipt.json"),
    )
    events = counter_receipt(run / "metrics.jsonl", start, end)
    write(output / "metric-event-receipt.json", events)
    features = add_preemption_receipt(features, events, digest(output / "metric-event-receipt.json"))
    summary = run / "benchmark/profile_export_aiperf.json"
    features = add_branch_receipt(features, read(summary), digest(summary))
    write(output / "feature_metadata.json", features)
    write(output / "references.json", refs)
    write(output / "observations.json", observations)
    replacements = defaultdict(set)
    for row in overrides:
        replacements[(row["play"], row["outer"], row["inner"])].add(row["output"])
    # A repeated corpus request can have different observed lengths. Retain every
    # observation, but never choose an arbitrary substitution for those instances.
    write(
        output / "overrides.json",
        [
            {"play": play, "outer": outer, "inner": inner, "output": next(iter(values))}
            for (play, outer, inner), values in replacements.items()
            if len(values) == 1
        ],
    )
    write(output / "config.json", config)
    for source, name in (
        (raw, "raw.jsonl"),
        (streams, "stream-records.jsonl"),
        (run / "metrics.jsonl", "server-scrapes.jsonl"),
        (summary, "harness-results.json"),
        (run / "cache-receipt.json", "runtime-receipt.json"),
        (run / "client-record/resolved-dataset.json", "harness-receipt.json"),
        (run / "benchmark/server_metrics_export.json", "server-metrics.json"),
        (run / "outcome.json", "collection-outcome.json"),
        (run / "client-command.json", "client-command.json"),
        (run / "server-command.json", "server-command.json"),
        (run.parent / "execution.json", "execution.json"),
        (run.parent / "gpus.json", "gpus.json"),
        (run.parent / "provenance.json", "provenance.json"),
    ):
        shutil.copyfile(source, output / name)
    if (run.parent / "collector-source.json").exists():
        shutil.copyfile(run.parent / "collector-source.json", output / "collector-source.json")
    if silicon:
        selected = read(silicon)
        if selected["point_id"] != identity:
            raise ValueError("Selected silicon metrics belong to another benchmark")
        write(
            output / "comparison.json",
            {
                "schema": "aisimulate.agentx.reference.v1",
                "silicon": selected,
                "reference": refs,
                "namespace_map": namespaces,
            },
        )
    write(
        output / "normalization.json",
        {
            "benchmark_id": identity,
            "profile_requested_seconds": duration,
            "profile_observed_seconds": elapsed,
            "response_grace_seconds": 30,
            "cancel_drain_seconds": 10,
            "profile_start_ns": str(start),
            "profile_end_ns": str(end),
            "measured_requests": sum(r["phase"] == "profile" for r in refs),
            "raw_sha256": digest(raw),
            "runtime_receipt_sha256": digest(run / "cache-receipt.json"),
        },
    )
    return validate_features(features, identity)


def manifest(root, corpus):
    """Freeze six complete normalized profiles; intake independently checks every hash."""
    if (root / "manifest.json").exists():
        raise ValueError("Do not overwrite a reviewed campaign manifest")
    shared = root / "corpus"
    shared.mkdir(exist_ok=False)
    shutil.copyfile(corpus / "traces.jsonl", shared / "traces.jsonl")
    shutil.copyfile(corpus / "corpus-manifest.json", shared / "corpus-manifest.json")
    runs = []
    for lanes in (1, 4, 8, 10, 13, 14):
        identity = f"qwen-weka-l{lanes}"
        path = root / identity
        receipt = read(path / "normalization.json")
        elapsed = (int(receipt["profile_end_ns"]) - int(receipt["profile_start_ns"])) / 1e9
        if (
            receipt["benchmark_id"] != identity
            or receipt["profile_requested_seconds"] != 3600
            or not 3599 <= elapsed <= 3641
        ):
            raise ValueError("Incomplete or wrong measured profile")
        files = {
            "trace": shared / "traces.jsonl",
            "corpus_manifest": shared / "corpus-manifest.json",
            "config": path / "config.json",
            "observations": path / "observations.json",
            "overrides": path / "overrides.json",
            "comparison": path / "comparison.json",
            "feature_metadata": path / "feature_metadata.json",
            "runtime_receipt": path / "runtime-receipt.json",
            "harness_receipt": path / "harness-receipt.json",
            "snapshot_receipt": path / "snapshot-receipt.json",
            "request_receipt": path / "request-receipt.json",
            "raw": path / "raw.jsonl",
            "raw_streams": path / "stream-records.jsonl",
            "server_scrapes": path / "server-scrapes.jsonl",
            "metric_event_receipt": path / "metric-event-receipt.json",
            "harness_results": path / "harness-results.json",
            "phase_accounting": path / "phase-accounting.json",
            "client_log": path / "client-log.json",
        }
        for name in (
            "server-metrics",
            "collection-outcome",
            "client-command",
            "server-command",
            "execution",
            "gpus",
            "normalization",
            "provenance",
        ):
            files[name.replace("-", "_")] = path / f"{name}.json"
        if (path / "collector-source.json").exists():
            files["collector_source"] = path / "collector-source.json"
        runs.append(
            {
                "id": identity,
                "category": "self_collected",
                "model": "Qwen/Qwen3.5-397B-A17B",
                "hardware": "B300",
                "backend": "vllm",
                "measured_backend_version": "0.31.0",
                "precision": "BF16",
                "tensor_parallel": 8,
                "lanes": lanes,
                "initial_input_tokens": None,
                "cache_state": "recorded",
                "repeat": 0,
                "benchmark_url": None,
                "source_run_url": None,
                "boundary": "successful_profile_span",
                "measured_requests": receipt["measured_requests"],
                "artifacts": {
                    name: {
                        "path": file.relative_to(root).as_posix(),
                        "sha256": digest(file),
                        "bytes": file.stat().st_size,
                    }
                    for name, file in files.items()
                },
            }
        )
    write(root / "manifest.json", {"schema": "aisimulate.qwen-weka.collection.v1", "runs": runs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-corpus", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--taxonomy-revision")
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--silicon", type=Path, help="Selected numerical intervals from the dataset worker")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest:
        manifest(args.output, args.corpus)
    elif args.prepare_corpus:
        prepare_corpus(args.prepare_corpus, args.output)
    else:
        normalize(
            args.run, args.corpus, args.template, args.output, args.taxonomy_revision, args.duration, args.silicon
        )
