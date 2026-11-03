#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently reconcile shared-frontend native observations and frozen scores."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

POINTS = {1: "agentx-446063", 8: "agentx-446069", 14: "agentx-446062"}


def read(path):
    return json.loads(path.read_bytes())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    if actual is None or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(f"Numerical reconciliation failed: {actual} != {expected}")


def latencies(row):
    """Client timestamps are milliseconds; zero inter-token latency is defined."""
    first, last, start = (row.get(k) for k in ("first_token_ms", "last_token_ms", "arrival_time_ms"))
    if first is None:
        return {"ttft": None, "tpot": None}
    if first < start or last < first or row["terminal_time_ms"] < last:
        raise ValueError("Client timestamps are not causal")
    ttft = first - start
    close(row["ttft_ms"], ttft)
    output = row["output_length"]
    if type(output) is not int or output < 1 or output != row["requested_output_length"]:
        raise ValueError("Completed request output count is not conserved")
    tpot = (last - first) / (output - 1) if output > 1 else None
    if tpot is not None:
        close(row["itl_ms"], tpot)
    return {"ttft": ttft, "tpot": tpot}


def error(predicted, reference):
    if predicted is None or reference is None or not math.isfinite(reference) or reference <= 0:
        return None
    if not math.isfinite(predicted) or predicted < 0:
        raise ValueError("Invalid predicted latency")
    return 100 * abs(predicted - reference) / reference


def summary(values):
    return {"n": len(values), "mape": math.fsum(values) / len(values) if values else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit output already exists")
    frozen = read(args.frozen)["cohorts"]
    catalog_path = args.inputs / "catalog/e2e/agentx.json"
    catalog = read(catalog_path)
    pooled = defaultdict(list)
    results = []
    for lane, identity in POINTS.items():
        directory = args.candidate / identity
        native_path, score_path = directory / "native.json", directory / "score.json"
        native, score = read(native_path), read(score_path)
        run = next(r for r in catalog["runs"] if r["id"] == identity)
        receipt = run["artifacts"]["observations"]
        observation_path = args.inputs / receipt["path"]
        if checksum(observation_path) != receipt["sha256"] or observation_path.stat().st_size != receipt["bytes"]:
            raise ValueError("Measured observations differ from pinned HF artifact")
        observations = [r for r in read(observation_path) if r["phase"] == "profile"]
        rows = native["telemetry"]["requests"]
        predicted = [r for r in rows if r["phase"] == "profile"]
        profile = native["agentic_profile"]
        close(profile["admission_cutoff_ms"] - profile["profile_start_ms"], 3_600_000.0)
        close(profile["response_grace_deadline_ms"] - profile["admission_cutoff_ms"], 30_000.0)
        if any(r["arrival_time_ms"] < profile["profile_start_ms"] - 1e-6 for r in predicted):
            raise ValueError("Preparation was included in profile requests")
        costs = read(directory / "input-config.json")["engine"]["tokenization"]
        if costs["workers"] != 1:
            raise ValueError("Unexpected tokenizer concurrency")
        previous_end = defaultdict(float)
        for row in sorted(rows, key=lambda r: r.get("tokenized_ms") or -1):
            end = row.get("tokenized_ms")
            if end is None:
                continue
            worker = row["routing_history"][0]["logical_worker_id"]
            start = max(row["dispatched_at_ms"], previous_end[worker])
            close(end, start + costs["base_ms"] + row["input_length"] * costs["per_input_token_us"] / 1000)
            previous_end[worker] = end

        if len(observations) != len(score["pairs"]) or len(observations) != score["coverage"]["measured"]:
            raise ValueError("Measured accounting changed")
        by_key = defaultdict(list)
        for row in predicted:
            by_key[row["request_id"]].append(row)
        measured_counts = Counter(r["key"] for r in observations)
        values, retained, zeros = defaultdict(list), Counter(), Counter()
        frozen_rows = frozen[identity]["dashboard"]
        if len(frozen_rows) != len(observations):
            raise ValueError("Frozen measured accounting changed")
        for actual, saved, original in zip(observations, score["pairs"], frozen_rows, strict=True):
            opaque = hashlib.sha256(actual["key"].encode()).hexdigest()[:16]
            if opaque != saved["request_id"] or opaque != original["request_id"]:
                raise ValueError("Request pairing or order changed")
            candidates = by_key[actual["key"]]
            eligible = (actual["status"] == "completed" and actual.get("identity_valid") is not False
                        and measured_counts[actual["key"]] == 1 and len(candidates) == 1
                        and candidates[0]["terminal_status"] == "completed"
                        and type(actual["output_length"]) is int and actual["output_length"] > 0
                        and candidates[0]["output_length"] == actual["output_length"])
            if (saved["status"] == "matched") != eligible:
                raise ValueError("Saved eligibility disagrees with native/reference accounting")
            numerical = latencies(candidates[0]) if eligible else {"ttft": None, "tpot": None}
            for metric in ("ttft", "tpot"):
                reference = actual[metric + "_ms"]
                if saved["silicon_" + metric + "_ms"] != reference:
                    raise ValueError("Silicon latency changed")
                ape = error(numerical[metric], reference)
                saved_error = saved[metric + "_ms_ape"]
                if ape is None:
                    if saved_error is not None:
                        raise ValueError("Undefined error was scored")
                else:
                    close(saved_error, ape)
                if original[metric + "_ms_ape"] is not None and ape is not None:
                    retained[metric] += 1
                    values[metric].append(ape)
                    zeros[metric] += numerical[metric] == 0
        profile = native["agentic_profile"]
        if profile["canceled_requests"] or profile["server_unsettled_requests"] or profile["client_in_flight_requests"]:
            raise ValueError("Incomplete native profile")
        point = {"lanes": lane, "requests": len(observations),
                 "unscored_preparation_requests": len(rows) - len(predicted), "metrics": {},
                 "checksums": {"native": checksum(native_path), "score": checksum(score_path),
                               "silicon": receipt["sha256"]}}
        for metric in ("ttft", "tpot"):
            count = sum(r[metric + "_ms_ape"] is not None for r in frozen_rows)
            point["metrics"][metric] = {**summary(values[metric]), "frozen": count,
                                         "lost": count - retained[metric], "zeros": zeros[metric]}
            pooled[metric].extend(values[metric])
        results.append(point)
    result = {"schema": "aisimulate.glm-rsi.shared-frontend-audit.v1", "points": results,
              "pooled": {metric: summary(values) for metric, values in pooled.items()},
              "frozen_sha256": checksum(args.frozen), "catalog_sha256": checksum(catalog_path),
              "qualification": "Arithmetic/identity audit only; lost frozen observations prohibit a win."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
