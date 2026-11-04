#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify conditional dense point components from observed allocation/timing data.

This deliberately does not forecast collector initialization or fleet totals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TRAIN_GEOMETRIES = {(4, 1), (32, 2)}
HOLDOUT_GEOMETRY = (16, 1)
PHASES = ("eager_warmup", "capture", "graph_warmup", "timed", "cleanup")


def number(mapping, *names):
    for name in names:
        value = mapping.get(name)
        if value is not None:
            value = float(value)
            if math.isfinite(value) and value >= 0:
                return value
            break
    raise ValueError(f"missing/nonphysical observed field: {names[0]}")


def geometry(sample):
    case = sample["case"]
    return int(number(case, "num_heads", "Q")), int(number(case, "num_kv_heads", "KV"))


def workload(sample):
    case = sample["case"]
    batch = int(number(case, "batch_size", "B"))
    length = int(number(case, "input_len", "L"))
    return batch, length


def protocol(sample):
    case = sample["case"]
    actual = sample.get("actual_kernel_source")
    if not isinstance(actual, str) or not actual:
        raise ValueError("actual native dispatch/kernel source missing")
    runtime = sample.get("runtime_protocol")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("observed runtime protocol missing")
    return {
        "system": sample["system"],
        "backend": sample["backend"],
        "operation": sample["operation"],
        "source_commit": sample["source_commit"],
        "audit_driver_sha256": sample["audit_driver_sha256"],
        "observer_sha256": sample["observer_sha256"],
        "runtime_protocol": runtime,
        "actual_kernel_source": actual,
        "planned_dispatch": case.get("dispatch"),
        "model_path": case.get("model_path"),
        "head_dim": int(number(case, "head_dim", "d")),
        "attn_dtype": case["attn_dtype"],
        "kv_cache_dtype": case["kv_cache_dtype"],
        "window_size": number(case, "window_size"),
        "num_runs_executed": number(sample, "num_runs_executed"),
        "requested_num_warmups": number(sample, "requested_num_warmups", "requested_warmups"),
        "repeat_n": number(sample, "repeat_n"),
    }


def observed_components(sample):
    if sample.get("status") != "passed":
        raise ValueError("native point did not pass")
    if not sample.get("allocation_observations"):
        raise ValueError("actual allocation observations missing")
    cache = number(sample, "kv_unique_allocation_bytes")
    inputs = number(sample, "forward_input_unique_storage_bytes")
    if cache <= 0 or inputs <= 0:
        raise ValueError("actual storage byte observations must be positive")
    total = number(sample, "collection_seconds")
    prep = number(sample, "preparation_seconds")
    execution = number(sample, "execution_seconds")
    phases = sample.get("phase_seconds") or {}
    resolved = sum(number(phases, name) for name in PHASES)
    if not math.isclose(resolved, execution, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("execution is not the five disjoint observed helper phases")
    if not math.isclose(prep + execution, total, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("preparation and execution do not partition the native call")
    return prep, execution


def features(sample, component):
    if component == "preparation":
        return [1.0, number(sample, "kv_unique_allocation_bytes"), number(sample, "forward_input_unique_storage_bytes")]
    case = sample["case"]
    batch, length = workload(sample)
    heads, _kv = geometry(sample)
    width = number(case, "head_dim", "d")
    if number(case, "window_size") != 0:
        raise ValueError("only full causal dense work is identified")
    if sample["operation"] == "attention_context":
        pairs = length * (length + 1) / 2
    elif sample["operation"] == "attention_generation":
        pairs = length + 1
    else:
        raise ValueError("not a dense attention operation")
    reps = number(sample, "num_runs_executed") * number(sample, "repeat_n")
    if reps <= 0:
        raise ValueError("observed timed repetitions missing")
    # This is a calibrated timed-work PROXY for all five execution clocks.
    # Warmup/capture invocation counts are not inferred from requested counts.
    return [1.0, batch * heads * width * pairs * reps]


def identify(train, component):
    matrix = np.asarray([features(s, component) for s in train], dtype=float)
    target = np.asarray([observed_components(s)[0 if component == "preparation" else 1] for s in train])
    scales = np.maximum(np.max(np.abs(matrix), axis=0), 1.0)
    scaled = matrix / scales
    coefficients, _residual, rank, singular = np.linalg.lstsq(scaled, target, rcond=1e-10)
    if rank != matrix.shape[1] or singular[-1] <= 0 or singular[0] / singular[-1] > 1e8:
        raise ValueError("rank-deficient or ill-conditioned scaled feature design")
    tolerance = max(1e-10, float(np.max(np.abs(coefficients))) * 1e-8)
    if np.min(coefficients) < -tolerance:
        raise ValueError("negative/nonphysical fitted coefficient")
    coefficients = np.maximum(coefficients, 0) / scales
    return coefficients, np.min(matrix, axis=0), np.max(matrix, axis=0), float(singular[0] / singular[-1])


def fit_component(records, component):
    try:
        train = [s for s in records if s["replicate"] in (1, 2) and s["geometry_role"] == "fit"]
        coefficients, low, high, condition = identify(train, component)
        predictions = []
        for sample in records:
            vector = np.asarray(features(sample, component))
            if np.any(vector < low) or np.any(vector > high):
                raise ValueError("validation features outside observed training bounds")
            actual = observed_components(sample)[0 if component == "preparation" else 1]
            prediction = float(vector @ coefficients)
            predictions.append(
                {
                    "sample_id": sample["sample_id"],
                    "replicate": sample["replicate"],
                    "geometry_role": sample["geometry_role"],
                    "actual_seconds": actual,
                    "predicted_seconds": prediction,
                    "relative_error": abs(prediction - actual) / max(actual, 1e-12),
                }
            )
        heldout = [p for p in predictions if p["geometry_role"] == "whole_geometry_holdout"]
        third = [p for p in predictions if p["replicate"] == 3]
        point_error = max(p["relative_error"] for p in heldout)
        actual_group = sum(p["actual_seconds"] for p in third)
        predicted_group = sum(p["predicted_seconds"] for p in third)
        group_error = abs(predicted_group - actual_group) / max(actual_group, 1e-12)
        passed = point_error <= 0.20 and group_error <= 0.15
        return {
            "qualified": passed,
            "reason": "validated" if passed else "heldout_accuracy_gate_failed",
            "coefficients": coefficients.tolist(),
            "feature_min": low.tolist(),
            "feature_max": high.tolist(),
            "scaled_condition_number": condition,
            "max_whole_geometry_holdout_error": point_error,
            "independent_third_component_error": group_error,
            "third_component_actual_seconds": actual_group,
            "third_component_predicted_seconds": predicted_group,
            "predictions": predictions,
        }
    except (ValueError, KeyError, IndexError) as error:
        return {"qualified": False, "reason": str(error), "coefficients": None}


def grid_check(records):
    for replicate in (1, 2, 3):
        group = [s for s in records if s["replicate"] == replicate]
        if len({s["group_id"] for s in group}) != 1:
            raise ValueError("replicate must identify exactly one independent native worker group")
        expected_lengths = {16384, 32768} if group[0]["operation"] == "attention_context" else {65535, 262143}
        cells = {(geometry(s), *workload(s)) for s in group}
        expected = {
            (g, b, length) for g in (*TRAIN_GEOMETRIES, HOLDOUT_GEOMETRY) for b in (1, 4) for length in expected_lengths
        }
        if cells != expected or len(group) != len(expected):
            raise ValueError("incomplete or duplicate predeclared geometry/batch/length design")
        for s in group:
            role = "whole_geometry_holdout" if geometry(s) == HOLDOUT_GEOMETRY else "fit"
            if s["geometry_role"] != role:
                raise ValueError("whole-geometry holdout leakage or unplanned geometry")
    if len({s["group_id"] for s in records}) != 3:
        raise ValueError("three fresh independent worker groups are required")


def fit_samples(samples):
    if len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("duplicate feature sample IDs")
    pools = defaultdict(list)
    excluded = []
    for sample in samples:
        if sample.get("native_pass_index") != 2:
            excluded.append({"sample_id": sample["sample_id"], "reason": "outside subsequent-in-worker pass-2 scope"})
            continue
        try:
            observed_components(sample)
            key = json.dumps(protocol(sample), sort_keys=True)
            pools[key].append(sample)
        except (ValueError, KeyError, TypeError) as error:
            excluded.append({"sample_id": sample["sample_id"], "reason": str(error)})
    result = []
    for key, records in sorted(pools.items()):
        item = {"protocol": json.loads(key), "sample_ids": [s["sample_id"] for s in records]}
        try:
            grid_check(records)
            item.update({component: fit_component(records, component) for component in ("preparation", "execution")})
        except (ValueError, KeyError, IndexError) as error:
            item["design_qualified"] = False
            item["reason"] = str(error)
        result.append(item)
    return {
        "scope": "conditional native pass 2 only; subsequent-in-worker does not establish compiler-warm state",
        "execution_feature": (
            "causal attention work times observed timed repetitions is a proxy "
            "under identical resolved helper protocol; capture/warmup forward counts are unknown"
        ),
        "whole_worker_gpu_hours": None,
        "cold_setup_gpu_hours": None,
        "fleet_gpu_hours": None,
        "pools": result,
        "excluded": excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.samples.read_bytes()
    samples = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    result = fit_samples(samples)
    result["input_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
