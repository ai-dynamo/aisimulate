# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze explicit homogeneous forwards and project strict module-key coverage.

This is a transport adapter for the native benchmark-point interchange. It
preserves every requested configuration; unsupported heterogeneous requests
raise instead of being converted to homogeneous substitutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def freeze_workloads(payload: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "prefill", "decode"}:
        raise ValueError("native point manifest requires schema_version, prefill and decode")
    if type(payload["schema_version"]) is not int or payload["schema_version"] not in (1, 2, 3):
        raise ValueError("unsupported native point schema")
    cases = []
    for phase in ("prefill", "decode"):
        if not isinstance(payload[phase], list):
            raise ValueError(f"{phase} points must be a list")
        for index, point in enumerate(payload[phase]):
            allowed = {"batch_size", "total_kv_read_tokens"}
            if phase == "prefill":
                allowed.update(("total_prefill_tokens", "rows", "partition"))
            if not isinstance(point, dict) or set(point) - allowed:
                raise ValueError("unknown native point fields")
            batch = _integer(point["batch_size"], "batch_size", 1)
            total_kv = _integer(point["total_kv_read_tokens"], "total_kv_read_tokens", int(phase == "decode"))
            if total_kv % batch:
                raise ValueError("module collection requires homogeneous per-request KV lengths")
            if phase == "prefill":
                total_new = _integer(point["total_prefill_tokens"], "total_prefill_tokens", batch)
                if total_new % batch or point.get("partition") is not None:
                    raise ValueError("module collection requires homogeneous per-request extensions")
                query, prefix = total_new // batch, total_kv // batch
                rows = point.get("rows")
                if rows is not None and (
                    payload["schema_version"] < 3
                    or rows != [[query, prefix] for _ in range(batch)]
                    or any(type(value) is not int for row in rows for value in row)
                ):
                    raise ValueError("explicit rows must exactly match homogeneous request totals")
            else:
                # The measured decode inserts one new token. Seed K-1 real
                # tokens so its attention reads exactly the requested K.
                query, prefix = 1, total_kv // batch - 1
                if prefix < 1:
                    raise ValueError("real-KV decode collection requires KV length >= 2")
            case = {
                "case_id": f"{phase}-{index:04d}",
                "phase": "context" if phase == "prefill" else "generation",
                "batch_size": batch,
                "query": query,
                "prefix": prefix,
            }
            cases.append(case)
    if not cases:
        raise ValueError("empty module workload manifest")
    identities = [(c["phase"], c["batch_size"], c["query"], c["prefix"]) for c in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate logical module workloads must be expressed as repetitions")
    return {
        "schema_version": 1,
        "source_payload": payload,
        "source_sha256": hashlib.sha256(canonical(payload).encode()).hexdigest(),
        "cases": cases,
    }


def coordinates(component, geometry, phase, batch, query, prefix):
    """Mirror the query scope; bounded tails retain their absolute KV end."""
    if component != "attention":
        return 1, 0, batch * query
    if phase == "generation":
        return batch, 0, query + prefix
    if geometry["bounded_prefill"]:
        tail = min(query, geometry["window_size"])
        return batch, prefix + query - tail, tail
    return batch, prefix, query


def baseline_tokens(cases):
    return sorted({case["batch_size"] * case["query"] for case in cases})


def projected_keys(manifest, case):
    keys = set()
    for entry in manifest["phases"][case["phase"]]:
        coordinate = coordinates(
            entry["component"],
            json.loads(entry["geometry"]),
            case["phase"],
            case["batch_size"],
            case["query"],
            case["prefix"],
        )
        keys.add((entry["component"], entry["geometry"], *coordinate))
    return keys


def coverage_report(manifest, calibration, heldout):
    keys = set().union(*(projected_keys(manifest, case) for case in calibration["cases"]))
    curves = {}
    for key in keys:
        curves.setdefault(key[:-1], set()).add(key[-1])
    results = []
    for case in heldout["cases"]:
        expected = projected_keys(manifest, case)
        missing = [k for k in expected if k[:-1] not in curves]
        extrapolated = [
            k for k in expected if k[:-1] in curves and not min(curves[k[:-1]]) <= k[-1] <= max(curves[k[:-1]])
        ]
        results.append(
            {
                **case,
                "module_key_count": len(expected),
                "missing_curve_count": len(missing),
                "extrapolated_key_count": len(extrapolated),
                "missing_curves": [list(k) for k in sorted(missing)],
            }
        )
    return {
        "execution_profile": manifest["execution_profile"],
        "calibration_configuration_count": len(calibration["cases"]),
        "projected_calibration_module_points": len(keys),
        "projected_component_counts": dict(Counter(k[0] for k in keys)),
        "heldout_configuration_count": len(results),
        "heldout_with_complete_interpolation_domain": sum(
            not r["missing_curve_count"] and not r["extrapolated_key_count"] for r in results
        ),
        "heldout_with_missing_curves": sum(bool(r["missing_curve_count"]) for r in results),
        "heldout": results,
    }


def main():
    from collector.sglang.dsv41_contract import build_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    calibration = freeze_workloads(json.loads(Path(args.calibration).read_text()))
    heldout = freeze_workloads(json.loads(Path(args.heldout).read_text()))
    calibration_ids = {tuple(c[k] for k in ("phase", "batch_size", "query", "prefix")) for c in calibration["cases"]}
    if calibration_ids.intersection(
        tuple(c[k] for k in ("phase", "batch_size", "query", "prefix")) for c in heldout["cases"]
    ):
        raise ValueError("calibration and heldout configurations overlap")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration-plan.json").write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    (output / "heldout-plan.json").write_text(json.dumps(heldout, indent=2, sort_keys=True) + "\n")
    tokens = baseline_tokens(calibration["cases"])
    seed_count = sum(c["prefix"] > 0 for c in calibration["cases"])
    report = {
        "status": "planned_unmeasured",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "native_forward_calls_per_profile": (len(calibration["cases"]) + seed_count) * (args.warmup + args.iterations),
        "measured_forward_invocations_per_profile": len(calibration["cases"]) * args.iterations,
        "baseline_token_axis": tokens,
        "baseline_points": len(tokens) * 5,
        "profiles": [coverage_report(build_manifest(4, replay), calibration, heldout) for replay in (False, True)],
    }
    (output / "coverage-projection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "profiles"}, sort_keys=True))
    for profile in report["profiles"]:
        print(json.dumps({k: v for k, v in profile.items() if k != "heldout"}, sort_keys=True))


if __name__ == "__main__":
    main()
