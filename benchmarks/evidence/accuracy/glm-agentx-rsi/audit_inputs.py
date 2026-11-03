#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit observed input/cache geometry without changing frozen accuracy eligibility."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from audit_shared_frontend import latencies

POINTS = {1: "agentx-446063", 4: "agentx-446068", 8: "agentx-446069",
          10: "agentx-446060", 13: "agentx-446061", 14: "agentx-446062"}


def longest_prefix(left, right):
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right))


def read(path):
    return json.loads(path.read_bytes())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(row, cache_field):
    length, reused = row.get("input_length"), row.get(cache_field)
    if length is None or reused is None:
        return None
    if type(length) is not int or type(reused) is not int or not 0 <= reused <= length:
        raise ValueError("Invalid input/cache token counters")
    return length, reused, length - reused


def mean(values):
    return math.fsum(values) / len(values) if values else None


def audit(observations, predicted, frozen):
    """First-admission reuse is comparable to the server's per-request cache receipt."""
    by_key = defaultdict(list)
    measured_counts = Counter(row["key"] for row in observations)
    for row in predicted:
        if row["phase"] == "profile":
            by_key[row["request_id"]].append(row)
    eligible = {row["request_id"]: row for row in frozen}
    if len(eligible) != len(frozen):
        raise ValueError("Duplicate frozen request identity")
    accounting, geometry = Counter(), Counter()
    deltas = defaultdict(list)
    errors = {metric: defaultdict(list) for metric in ("ttft", "tpot")}
    lost = Counter()
    for actual in observations:
        if actual["phase"] != "profile":
            continue
        accounting["measured"] += 1
        opaque = hashlib.sha256(actual["key"].encode()).hexdigest()[:16]
        original = eligible.get(opaque)
        matches = by_key[actual["key"]]
        valid = (actual["status"] == "completed" and measured_counts[actual["key"]] == 1
                 and len(matches) == 1 and matches[0]["terminal_status"] == "completed"
                 and type(actual.get("output_length")) is int and actual["output_length"] > 0
                 and matches[0]["output_length"] == actual["output_length"])
        if not valid:
            accounting["ineligible_or_missing"] += 1
            for metric in errors:
                lost[metric] += original is not None and original[metric + "_ms_ape"] is not None
            continue
        accounting["matched"] += 1
        predicted_row = matches[0]
        accounting["readmitted"] += predicted_row.get("readmission_count", 0) > 0
        input_group = "input_unavailable"
        a, b = actual.get("input_length"), predicted_row.get("input_length")
        if a is not None and b is not None:
            if type(a) is not int or type(b) is not int or min(a, b) < 0:
                raise ValueError("Invalid input token counters")
            geometry["known_input"] += 1
            geometry["input_equal"] += a == b
            deltas["input"].append(b - a)
            input_group = ("input_equal" if a == b else "input_within_64_tokens" if abs(b - a) <= 64
                           else "input_differs_over_64_tokens")
        measured = counts(actual, "cached_tokens")
        simulated = counts(predicted_row, "first_admission_reused_input_tokens")
        geometry["known_measured_cache"] += measured is not None
        geometry["known_predicted_cache"] += simulated is not None
        group = "unavailable_geometry"
        if measured is None or simulated is None:
            geometry[group] += 1
        else:
            for name, a, b in zip(("reuse", "fresh"), measured[1:], simulated[1:], strict=True):
                deltas[name].append(b - a)
                geometry[name + "_equal"] += a == b
            fresh_delta = simulated[2] - measured[2]
            group = ("fresh_within_64_tokens" if abs(fresh_delta) <= 64 else
                     "predicted_more_fresh" if fresh_delta > 0 else "predicted_less_fresh")
            geometry[group] += 1
            geometry["known_geometry"] += 1
            geometry["input_and_reuse_equal"] += measured[:2] == simulated[:2]
        for metric, field in (("ttft", "ttft_ms"), ("tpot", "itl_ms")):
            if original is None or original[metric + "_ms_ape"] is None:
                continue
            reference, value = actual.get(metric + "_ms"), predicted_row.get(field)
            if reference is None or reference <= 0 or not math.isfinite(reference):
                raise ValueError("Frozen metric has an invalid reference")
            if value is None:
                lost[metric] += 1
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid predicted latency")
            ape = 100 * abs(value - reference) / reference
            errors[metric]["all_retained"].append(ape)
            errors[metric][group].append(ape)
            errors[metric][input_group].append(ape)
    result = {"accounting": dict(accounting), "geometry": dict(geometry), "deltas_tokens": {}, "metrics": {}}
    for name, values in deltas.items():
        result["deltas_tokens"][name] = {"n": len(values), "mean_signed": mean(values),
                                        "mean_absolute": mean([abs(v) for v in values]),
                                        "sum_signed": sum(values), "max_absolute": max(map(abs, values))}
    for metric, groups in errors.items():
        expected = sum(row[metric + "_ms_ape"] is not None for row in frozen)
        retained = len(groups["all_retained"])
        if retained + lost[metric] != expected:
            raise ValueError("Frozen metric accounting is incomplete")
        result["metrics"][metric] = {
            "frozen": expected, "retained": retained, "lost": lost[metric],
            "groups": {name: {"n": len(values), "mape": mean(values)} for name, values in groups.items()},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="NAME=retained-replay-directory")
    parser.add_argument("--lanes", type=int, nargs="+", choices=POINTS, default=[1, 8, 14])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit output already exists")
    if len(args.lanes) != len(set(args.lanes)):
        raise ValueError("Duplicate lane selection")
    variants = {}
    for specification in args.variant:
        name, directory = specification.split("=", 1)
        if not name or name in variants:
            raise ValueError("Duplicate/empty variant identity")
        variants[name] = Path(directory)
    catalog_path = args.inputs / "catalog/e2e/agentx.json"
    catalog = read(catalog_path)
    frozen = read(args.frozen)["cohorts"]
    result = {"schema": "aisimulate.glm-rsi.input-cache-audit.v1", "points": [],
              "frozen_sha256": checksum(args.frozen), "catalog_sha256": checksum(catalog_path),
              "lanes": args.lanes,
              "scope": "Explicit selected lanes. Observational strata, not causal attribution or new scoring cohorts. "
                       "Fresh tokens mean input minus reported cache reuse, not measured kernel work. "
                       "Different prompt lengths confound direct reuse-counter comparisons."}
    for lane, identity in POINTS.items():
        if lane not in args.lanes:
            continue
        run = next(row for row in catalog["runs"] if row["id"] == identity)
        receipt = run["artifacts"]["observations"]
        path = args.inputs / receipt["path"]
        if checksum(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
            raise ValueError("Measured observations changed")
        observations = read(path)
        point = {"lanes": lane, "silicon_sha256": receipt["sha256"], "variants": {}}
        for name, directory in variants.items():
            native_path = directory / identity / "native.json"
            native = read(native_path)
            predicted = [row for row in native["telemetry"]["requests"] if row["phase"] == "profile"]
            for row in predicted:
                if row["terminal_status"] == "completed":
                    latencies(row)
            # Native headline throughput counts completed requests only. Keep
            # canceled partial output explicit without adding it to that total.
            total = sum(row["output_length"] for row in predicted if row["terminal_status"] == "completed")
            if total != native["total_output_tokens"] or not math.isclose(
                total * 1000 / native["duration_ms"], native["output_throughput_tok_s"], rel_tol=1e-10, abs_tol=1e-6
            ):
                raise ValueError("Native token/throughput accounting changed")
            phase = native["agentic_profile"]
            if (not math.isclose(phase["admission_cutoff_ms"] - phase["profile_start_ms"], 3_600_000, abs_tol=1e-6)
                    or not math.isclose(phase["response_grace_deadline_ms"] - phase["admission_cutoff_ms"],
                                        30_000, abs_tol=1e-6)):
                raise ValueError("Original profile/grace boundaries changed")
            item = audit(observations, native["telemetry"]["requests"], frozen[identity]["dashboard"])
            item["native_completed_output_tokens"] = total
            item["native_noncompleted_output_tokens"] = sum(
                row["output_length"] for row in predicted if row["terminal_status"] != "completed"
            )
            item["native_timestamp_token_throughput_boundaries_qualified"] = True
            item["native_sha256"] = checksum(native_path)
            point["variants"][name] = item
        result["points"].append(point)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
