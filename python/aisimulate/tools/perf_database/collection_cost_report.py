#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report bounded collection estimates; unavailable evidence remains explicitly unknown."""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tools.perf_database.collection_cost import case_key, compare

# These are workload axes, never geometry, precision, checkpoint or dispatch.
AXES = {
    "batch_size",
    "batch_sizes",
    "num_tokens",
    "seq_lens",
    "seq_len",
    "input_len",
    "prefix_len",
    "sequence_length",
    "kv_cache_length",
}


def clean_case(case):
    return {k: str(v) for k, v in case.items() if not k.startswith("_")}


def identity(sample):
    return sample["system"], sample["backend"], sample["operation"]


def geometry(case):
    return tuple(sorted((k, v) for k, v in clean_case(case).items() if k not in AXES))


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def accepted(samples):
    """Failures/timeouts are evidence of gaps, never cheap successful observations."""
    return [s for s in samples if s["status"] == "passed" and positive(s.get("collection_seconds"))]


def measured_cost(samples):
    """Inclusive point times plus explicitly separate setup once per group.

    Group/direct-call clocks already include setup and forwards. Never add the
    latency or nested phase clocks to an inclusive collection clock.
    """
    groups = defaultdict(list)
    for sample in accepted(samples):
        groups[(*identity(sample), sample["group_id"])].append(sample)
    total = 0.0
    for group in groups.values():
        inclusive = [s for s in group if s["measurement_scope"] in {"group", "direct_call"}]
        if inclusive:
            if len(inclusive) != 1:
                raise ValueError("ambiguous duplicate inclusive group clocks")
            total += inclusive[0]["collection_seconds"]
        else:
            setup = {s["setup_seconds"] for s in group if s.get("setup_seconds") is not None}
            if len(setup) > 1 or any(not positive(v) for v in setup):
                raise ValueError("inconsistent group setup clocks")
            total += sum(s["collection_seconds"] for s in group) + (next(iter(setup)) if setup else 0)
    return total


def predict(case, samples):
    """Exact observations or multilinear interpolation inside measured corners.

    Every non-workload field must match. No extrapolation or proxy-to-time fit.
    Returns an empirical repeated-observation range, not a confidence interval.
    """
    case = clean_case(case)
    matched = [s for s in accepted(samples) if geometry(s["case"]) == geometry(case)]
    if len({s["source_commit"] for s in matched}) > 1 or len({identity(s) for s in matched}) > 1:
        return None
    exact = [s["collection_seconds"] for s in matched if clean_case(s["case"]) == case]
    if exact:
        return min(exact), max(exact)
    axes = sorted(set(case) & AXES)
    if not axes or any(not str(case[a]).isdigit() for a in axes):
        return None
    matched = [s for s in matched if set(clean_case(s["case"])) == set(case)]
    corners = []
    for axis in axes:
        values = sorted({int(s["case"][axis]) for s in matched if str(s["case"][axis]).isdigit()})
        target = int(case[axis])
        if not values or target < values[0] or target > values[-1]:
            return None
        index = bisect.bisect_left(values, target)
        corners.append([target] if values[index] == target else [values[index - 1], values[index]])
    low = high = 0.0
    for point in itertools.product(*corners):
        observations = [
            s["collection_seconds"]
            for s in matched
            if all(int(s["case"][a]) == v for a, v in zip(axes, point, strict=True))
        ]
        if not observations:
            return None
        weight = 1.0
        for axis, value, bounds in zip(axes, point, corners, strict=True):
            if len(bounds) == 2:
                weight *= (
                    (bounds[1] - int(case[axis])) / (bounds[1] - bounds[0])
                    if value == bounds[0]
                    else (int(case[axis]) - bounds[0]) / (bounds[1] - bounds[0])
                )
        low += weight * min(observations)
        high += weight * max(observations)
    return low, high


def validation(samples):
    points = [s for s in samples if s["measurement_scope"] == "point"]
    fit = [s for s in points if s.get("role") != "holdout"]
    holdouts = [s for s in points if s.get("role") == "holdout"]
    errors = []
    for sample in holdouts:
        estimate = predict(sample["case"], fit)
        if sample["status"] != "passed" or estimate is None or not positive(sample.get("collection_seconds")):
            return {"passed": False, "reason": "uncovered_or_failed_holdout", "max_relative_error": None}
        actual = sample["collection_seconds"]
        error = abs(statistics.mean(estimate) - actual) / max(actual, 1e-12)
        errors.append(error)
    return {
        "passed": bool(errors) and max(errors) <= 0.20,
        "reason": "heldout_20_percent" if errors else "no_holdout",
        "max_relative_error": max(errors) if errors else None,
    }


def group_validation(samples):
    """Validate reconstructed point+setup totals against independent group clocks."""
    errors = []
    for group in accepted(samples):
        if group["measurement_scope"] != "group" or group.get("role") != "holdout":
            continue
        points = [
            s
            for s in accepted(samples)
            if s["measurement_scope"] == "point"
            and s["group_id"] == group["group_id"]
            and identity(s) == identity(group)
        ]
        if not points or not any(s.get("setup_seconds") is not None for s in points):
            return {"passed": False, "max_relative_error": None, "reason": "missing_group_members_or_setup"}
        predicted = measured_cost(points)
        errors.append(abs(predicted - group["collection_seconds"]) / max(group["collection_seconds"], 1e-12))
    return {
        "passed": bool(errors) and max(errors) <= 0.15,
        "max_relative_error": max(errors) if errors else None,
        "reason": "group_15_percent" if errors else "no_group_holdout",
    }


def paired_groups(samples, scenario, added, removed, baseline_scenario="baseline"):
    """Exact native old/new group differences, including changed initialization.

    Only explicit member manifests can assign a measured group to inventory
    changes. Repeated pairs form an empirical range; missing pairs stay unknown.
    """
    groups = defaultdict(list)
    for sample in accepted(samples):
        if sample["measurement_scope"] == "group" and sample.get("paired_group_id"):
            groups[sample["paired_group_id"]].append(sample)
    additions, removals, low, high = set(), set(), 0.0, 0.0
    for group in groups.values():
        before = [s for s in group if s.get("scenario") == baseline_scenario]
        after = [s for s in group if s.get("scenario") == scenario]
        if not before or not after or any(not s.get("group_member_cases") for s in before + after):
            continue
        old_sets = [{case_key(clean_case(c)) for c in s["group_member_cases"]} for s in before]
        new_sets = [{case_key(clean_case(c)) for c in s["group_member_cases"]} for s in after]
        if any(keys != old_sets[0] for keys in old_sets) or any(keys != new_sets[0] for keys in new_sets):
            raise ValueError("paired group repeat changed member manifest")
        new, old = new_sets[0] - old_sets[0], old_sets[0] - new_sets[0]
        if not new and not old:
            continue
        if not new <= added or not old <= removed:
            continue
        if new & additions or old & removals:
            raise ValueError("overlapping paired groups double-charge physical cells")
        additions.update(new)
        removals.update(old)
        low += min(s["collection_seconds"] * s.get("participating_gpus", 1) for s in after) - max(
            s["collection_seconds"] * s.get("participating_gpus", 1) for s in before
        )
        high += max(s["collection_seconds"] * s.get("participating_gpus", 1) for s in after) - min(
            s["collection_seconds"] * s.get("participating_gpus", 1) for s in before
        )
    return {
        "covered_added_points": len(additions),
        "covered_removed_points": len(removals),
        "gpu_seconds_range": [low, high],
        "complete": additions == added and removals == removed,
    }


def report_lane(
    baseline, candidate, samples, system, backend, scenario="model", baseline_scenario="baseline", source_commit=None
):
    result = {}
    for table, delta in compare(baseline, candidate).items():
        operation = table.split("_", 2)[2]
        lane = [s for s in samples if identity(s) == (system, backend, operation)]
        buckets = defaultdict(list)
        for sample in lane:
            if sample.get("scenario") == scenario and (
                source_commit is None or sample["source_commit"] == source_commit
            ):
                buckets[(geometry(sample["case"]), sample["source_commit"])].append(sample)
        covered = 0
        low = high = 0.0
        gaps = defaultdict(int)
        for case in delta["added"]:
            matches = [bucket for (shape, _commit), bucket in buckets.items() if shape == geometry(case)]
            estimates = []
            for bucket in matches:
                point_samples = [s for s in bucket if s["measurement_scope"] == "point"]
                estimate = predict(case, point_samples)
                exact = any(clean_case(s["case"]) == clean_case(case) for s in accepted(point_samples))
                if estimate is not None and (exact or validation(bucket)["passed"]):
                    estimates.append(estimate)
            estimate = (min(e[0] for e in estimates), max(e[1] for e in estimates)) if estimates else None
            if estimate is None:
                gaps["unmeasured_geometry_or_unbracketed_shape"] += 1
            else:
                covered += 1
                low += estimate[0]
                high += estimate[1]
        # Setup growth of retained workers is unknown without matched old/new
        # group evidence. Do not silently assume a sunk initialization cost.
        validations = [group_validation(bucket) for bucket in buckets.values()]
        paired = paired_groups(
            lane,
            scenario,
            {case_key(clean_case(c)) for c in delta["added"]},
            {case_key(clean_case(c)) for c in delta["removed"]},
            baseline_scenario,
        )
        result[operation] = {
            "baseline_points": delta["baseline"],
            "candidate_points": delta["candidate"],
            "added_points": len(delta["added"]),
            "removed_points": len(delta["removed"]),
            "covered_added_points": covered,
            "uncovered_added_points": len(delta["added"]) - covered,
            "covered_point_seconds_range": [low, high],
            "uncovered_reasons": dict(gaps),
            "group_validation": validations,
            "strata": [
                {
                    "geometry": dict(shape),
                    "source_commit": commit,
                    "point_validation": validation(bucket),
                    "group_validation": group_validation(bucket),
                }
                for (shape, commit), bucket in buckets.items()
            ],
            "paired_group_measurement": paired,
            "extra_gpu_hours_range": [v / 3600 for v in paired["gpu_seconds_range"]] if paired["complete"] else None,
            "total_status": "measured_group_difference"
            if paired["complete"]
            else "unknown_setup_delta_or_uncovered_points",
            "setup_delta_seconds": None if delta["added"] or delta["removed"] else 0.0,
            "category": "non_history" if operation in {"moe", "kda", "nccl"} else "history",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--source-commit", required=True, help="exact candidate source revision eligible for point interpolation"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    result = {
        "range_kind": "empirical_not_statistical_confidence_interval",
        "allocation_time_included": False,
        "fleet_gpu_hours_range": None,
        "coverage_scope": "declared inventory only; deferred or undeclared model routes remain unmeasured",
        "lanes": {},
    }
    for directory in sorted(args.baseline.glob("*/*")):
        if not directory.is_dir():
            continue
        system, backend = directory.parent.name, directory.name
        result["lanes"][f"{system}/{backend}"] = {
            name: report_lane(
                directory, root / system / backend, samples, system, backend, name, source_commit=args.source_commit
            )
            for name, root in (("shared", args.shared), ("model", args.model))
        }
        result["lanes"][f"{system}/{backend}"]["model_vs_shared"] = report_lane(
            args.shared / system / backend,
            args.model / system / backend,
            samples,
            system,
            backend,
            "model",
            "shared",
            source_commit=args.source_commit,
        )
    result["samples"] = samples
    result["phase_accounting"] = (
        "phase_seconds are supporting diagnostics, may be nested, and are never added to inclusive collection clocks"
    )
    result["sources"] = sorted({s["source_commit"] for s in samples})
    result["sample_status_counts"] = {
        status: sum(s["status"] == status for s in samples) for status in ("passed", "failed", "timeout", "unattempted")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
