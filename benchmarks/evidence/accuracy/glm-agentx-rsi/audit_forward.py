#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently qualify native forward GPU ranges; never replace AgentX MAPE."""

import argparse
import gzip
import hashlib
import json
import re
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from analyze_timing import analyze_forward


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_forwards(document, device):
    events = document["traceEvents"]
    api = defaultdict(list)
    cpu_annotations = defaultdict(list)
    kernels = defaultdict(list)
    annotations = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("dur", 0) <= 0:
            continue
        category = event.get("cat")
        if category in ("cuda_runtime", "cuda_driver"):
            api[event["pid"], event["tid"]].append(event)
        elif category == "kernel":
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                kernels[correlation].append(event)
        elif category == "gpu_user_annotation":
            annotations[event.get("args", {}).get("External id")].append(event)
        elif category == "user_annotation":
            cpu_annotations[event["pid"], event["tid"]].append(event)
    indexes = {}
    for key, rows in api.items():
        rows.sort(key=lambda row: row["ts"])
        indexes[key] = [row["ts"] for row in rows]
    scopes = sorted((event for event in events if event.get("cat") == "user_annotation"
                     and event.get("ph") == "X" and (re.match(r"step\[\w+\s", event.get("name", ""))
                                                     or event.get("name") in ("draft", "draft_extend"))),
                    key=lambda event: event["ts"])
    results = []
    kernel_summary = defaultdict(lambda: defaultdict(lambda: {"count": 0, "sum_kernel_ms": 0.0}))
    for scope in scopes:
        label = scope["name"]
        step = re.match(r"step\[(\w+)", label)
        stage = step[1] if step else label
        shape = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", label)}
        key = scope["pid"], scope["tid"]
        starts = indexes.get(key, [])
        begin, end = scope["ts"], scope["ts"] + scope["dur"]
        calls = [event for event in api.get(key, [])[bisect_left(starts, begin):bisect_right(starts, end)]
                 if event["ts"] + event["dur"] <= end + 0.001]
        correlations = {event.get("args", {}).get("correlation") for event in calls}
        correlations.discard(None)
        gpu_kernels = [event for correlation in correlations for event in kernels.get(correlation, [])]
        descendants = [event for event in cpu_annotations[key] if begin <= event["ts"]
                       and event["ts"] + event["dur"] <= end + 0.001]
        gpu_ranges = [gpu for event in descendants for gpu in annotations.get(
            event.get("args", {}).get("External id"), [])
                      if event.get("args", {}).get("External id") is not None and gpu["name"] == event["name"]]
        reasons = []
        if not gpu_kernels:
            reasons.append("no explicit CUDA API-to-kernel correlation")
        elif {event.get("args", {}).get("device") for event in gpu_kernels} != {device}:
            reasons.append("correlated kernels belong to an unexpected device")
        if not gpu_ranges or {event["pid"] for event in gpu_ranges} != {device}:
            reasons.append("native GPU annotations are unavailable or on an unexpected device")
        row = {"ordinal": len(results), "label": label, "stage": stage, "shape": shape,
               "cpu_pid": scope["pid"], "cpu_tid": scope["tid"], "cpu_start_us": begin,
               "cpu_span_ms": scope["dur"] / 1000, "correlated_kernel_rows": len(gpu_kernels),
               "native_gpu_range_rows": len(gpu_ranges), "reasons": reasons}
        if gpu_kernels and gpu_ranges:
            gpu_begin = min(event["ts"] for event in gpu_ranges)
            gpu_end = max(event["ts"] + event["dur"] for event in gpu_ranges)
            kernel_begin = min(event["ts"] for event in gpu_kernels)
            kernel_end = max(event["ts"] + event["dur"] for event in gpu_kernels)
            if kernel_begin < gpu_begin - 0.01 or kernel_end > gpu_end + 0.01:
                reasons.append("native GPU annotation does not enclose correlated kernels")
            row.update(gpu_start_us=gpu_begin, gpu_end_us=gpu_end, gpu_span_ms=(gpu_end - gpu_begin) / 1000,
                       kernel_span_ms=(kernel_end - kernel_begin) / 1000,
                       streams=sorted({event.get("args", {}).get("stream", -1) for event in gpu_kernels}))
            for event in gpu_kernels:
                item = kernel_summary[stage][event["name"]]
                item["count"] += 1
                item["sum_kernel_ms"] += event["dur"] / 1000
        call_totals = defaultdict(float)
        for event in calls:
            call_totals[event["name"]] += event["dur"] / 1000
        row["cpu_cuda_api_ms"] = dict(call_totals)
        row["qualified"] = not reasons
        results.append(row)
    return {"forwards": results, "stage_counts": dict(Counter(row["stage"] for row in results)),
            "qualified_forward_count": sum(row["qualified"] for row in results),
            "kernel_summary": dict(kernel_summary),
            "kernel_sum_scope": ("Diagnostic busy-time sum only. Nested draft_extend/DRAFT scopes overlap; "
                                 "do not add their totals or sum TP ranks."),
            "baseTimeNanoseconds": document.get("baseTimeNanoseconds"),
            "category_counts": dict(Counter(event.get("cat", "none") for event in events))}


def distribution(values):
    if not values:
        return {"n": 0, "unavailable": "no qualified observations"}
    return {"n": len(values), "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}


def audit(root, output):
    request_audit = analyze_forward(root, profiler_only=True)
    modes = request_audit["modes"]
    reasons = []
    if not request_audit["collector_completed"] or not modes["instrumented-mtp"]["qualified"]:
        reasons.append("finite request campaign did not independently qualify")
    directory = root / "instrumented-mtp"
    rows_path = directory / "client/requests.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    profiled = {row["context"]: row for row in rows if row["arm"] == "profiled" and row["role"] == "profile"}
    receipt_paths = sorted((directory / "model-receipts").glob("*.json"))
    receipts = [json.loads(path.read_bytes()) for path in receipt_paths]
    if len(receipts) != 16:
        raise ValueError("Require sixteen loaded target/draft model receipts")
    devices = {}
    for rank in range(8):
        selected = [row for row in receipts if row["tp_rank"] == rank]
        if (len(selected) != 2 or {row["draft"] for row in selected} != {False, True}
                or any(row["tp_size"] != 8 for row in selected)
                or len({row["cuda_device"] for row in selected}) != 1):
            raise ValueError("Loaded model rank/role/device receipts are inconsistent")
        devices[rank] = selected[0]["cuda_device"]
    if set(devices.values()) != set(range(8)):
        raise ValueError("Loaded model receipts do not cover eight distinct devices")
    traces = sorted((directory / "client").glob("*.trace.json.gz"))
    expected = {f"rsi-profile-c{context}-profiled-TP-{rank}.trace.json.gz"
                for context in (8192, 393216) for rank in range(8)}
    if {path.name for path in traces} != expected or len(traces) != 16:
        raise ValueError("Native capture file identities are incomplete or duplicated")
    source_hashes = {str(path.relative_to(root)): digest(path) for path in [rows_path, *receipt_paths]}
    for relative in ("runtime.json", "forward-nvtx-instrumentation.json", "completion.json",
                     "instrumented-mtp/command.json", "instrumented-mtp/resolved-server.json",
                     "instrumented-mtp/client/manifest.json"):
        source_hashes[relative] = digest(root / relative)
    source_hashes["allocation/gpus.csv"] = digest(root.parent / "gpus.csv")
    captures = defaultdict(dict)
    for path in traces:
        identity = re.fullmatch(r"rsi-profile-c(\d+)-profiled-TP-(\d+)\.trace\.json\.gz", path.name)
        context, rank = map(int, identity.groups())
        before = digest(path)
        with gzip.open(path, "rt") as stream:
            document = json.load(stream)
        result = project_forwards(document, devices[rank])
        del document
        if digest(path) != before:
            raise ValueError("Native trace changed during audit")
        source_hashes[str(path.relative_to(root))] = before
        selected_pids = {row["pid"] for row in receipts if row["tp_rank"] == rank}
        if any(row["cpu_pid"] not in selected_pids for row in result["forwards"]):
            reasons.append(f"context{context} rank{rank}: CPU trace/model receipt PID mismatch")
        if not result["forwards"] or result["qualified_forward_count"] != len(result["forwards"]):
            reasons.append(f"context{context} rank{rank}: some forward GPU comparisons are unavailable")
        captures[context][rank] = result
        (output / f"context-{context}-rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    summaries = {}
    for context, ranks in captures.items():
        verification = {rank: [row for row in result["forwards"] if row["stage"] == "VERIFY"]
                        for rank, result in ranks.items()}
        expected_rounds = profiled[context]["events"][-1]["meta"].get("spec_verify_ct")
        counts = {rank: len(steps) for rank, steps in verification.items()}
        executed = counts[0]
        valid = (executed >= 11 and len(set(counts.values())) == 1
                 and all(all(row["qualified"] for row in steps) for steps in verification.values()))
        labels = [[row["label"] for row in steps] for steps in verification.values()]
        if not valid or any(names != labels[0] for names in labels):
            reasons.append(f"context{context}: verification rank/shape/count/correlation qualification failed")
            summaries[context] = {"qualified": False, "gpu_projection_qualified": False,
                                  "native_verification_rounds": expected_rounds, "rank_counts": counts}
            continue
        aligned = type(expected_rounds) is int and executed == expected_rounds
        if not aligned:
            reasons.append(
                f"context{context}: {executed} executed VERIFY ranges versus {expected_rounds} reported rounds")
        maxima = [max(steps[i]["gpu_span_ms"] for steps in verification.values()) for i in range(executed)]
        cadence = [max((steps[i + 1]["gpu_start_us"] - steps[i]["gpu_start_us"]) / 1000
                       for steps in verification.values()) for i in range(executed - 1)]
        if any(value <= 0 for value in cadence):
            reasons.append(f"context{context}: GPU verification clock ordering failed")
        phase_descriptions = {}
        for stage in ("draft", "draft_extend", "EXTEND", "DRAFT"):
            by_rank = {rank: [row for row in result["forwards"] if row["stage"] == stage]
                       for rank, result in ranks.items()}
            sizes = [len(steps) for steps in by_rank.values()]
            if len(set(sizes)) != 1 or not sizes[0] or any(
                    not row["qualified"] for steps in by_rank.values() for row in steps):
                phase_descriptions[stage] = {"unavailable": "rank count or GPU projection mismatch",
                                              "rank_counts": dict(zip(by_rank, sizes, strict=True))}
            else:
                phase_descriptions[stage] = distribution([
                    max(steps[i]["gpu_span_ms"] for steps in by_rank.values()) for i in range(sizes[0])])
        summaries[context] = {"qualified": aligned and all(value > 0 for value in cadence),
                              "gpu_projection_qualified": all(value > 0 for value in cadence),
                              "request_round_alignment_qualified": aligned,
                              "native_verification_rounds": expected_rounds,
                              "executed_verification_ranges": executed,
                              "execution_count_minus_reported_rounds": executed - expected_rounds
                              if type(expected_rounds) is int else None,
                              "population": "Every executed GPU range; no subset chosen to match request counters.",
                              "executed_tp_max_verify_span": distribution(maxima),
                              "executed_tp_max_verify_start_cadence": distribution(cadence),
                              "tp_max_phase_descriptions": phase_descriptions,
                              "verify_span_maxima_ms": maxima, "verify_cadence_maxima_ms": cadence,
                              "verify_shapes": [row["shape"] for row in verification[0]],
                              "shape_lookup_available": all("g_sqsk" in row["shape"] for row in verification[0]),
                              "rank_stage_counts": {rank: result["stage_counts"] for rank, result in ranks.items()},
                              "rank_unavailable_forwards": {rank: len(result["forwards"])
                                                            - result["qualified_forward_count"]
                                                            for rank, result in ranks.items()}}
    return {"schema": "aisimulate.glm-rsi.native-forward-audit.v2", "qualified": not reasons, "reasons": reasons,
            "contexts": summaries, "source_sha256": source_hashes, "loaded_models": receipts,
            "auditor_sha256": digest(Path(__file__)),
            "request_auditor_sha256": digest(Path(__file__).with_name("analyze_timing.py")),
            "request_audit": request_audit,
            "scope": "Finite native controls; GPU annotation spans independently checked against CUDA correlations.",
            "rank_rule": "Maxima across matched TP ranks, never an average or sum of ranks.",
            "clock_rule": "Durations and within-rank cadence only; no unverified cross-process absolute clock merge.",
            "limitations": ["GPU spans include gaps and streams; kernel sums are diagnostic.",
                            "Profiling may perturb scheduling; retain before/after controls.",
                            "Delivered counters do not directly establish discarded speculative tails.",
                            "Executed versus reported request-round counts remain separate; no range is removed.",
                            "Exact decode lookup inputs are unavailable when CPU shape mirrors are absent.",
                            "No AgentX request MAPE or historical baseline replacement."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args.root, args.output)
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({"status": "failed", "error_type": type(exc).__name__})
                                                + "\n")
        raise
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"qualified": result["qualified"], "reasons": result["reasons"],
                      "contexts": {context: {key: row.get(key) for key in
                                             ("qualified", "native_verification_rounds", "executed_verification_ranges",
                                              "executed_tp_max_verify_span", "executed_tp_max_verify_start_cadence",
                                              "shape_lookup_available", "tp_max_phase_descriptions")}
                                   for context, row in result["contexts"].items()}}))


if __name__ == "__main__":
    main()
