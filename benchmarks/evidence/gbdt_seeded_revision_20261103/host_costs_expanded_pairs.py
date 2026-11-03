#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the expanded host-cost cases sequentially in fresh matched processes."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--baseline-native-sha256", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-native-sha256", required=True)
    parser.add_argument("--shared-harness", required=True, type=Path)
    parser.add_argument("--worker-harness", type=Path, default=Path(__file__).with_name("host_costs_expanded.py"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", choices=("cold2", "cold4", "seeded2", "seeded4", "fallback"))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warm-queries", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.trials < 5 and not args.dry_run:
        parser.error("At least five paired trials required for measurements")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    environment.pop("PYTHONPATH", None)
    builds = {
        "baseline": (args.baseline, args.baseline_native_sha256),
        "candidate": (args.candidate, args.candidate_native_sha256),
    }
    records, timings, identities = [], {}, {}
    cases = args.case or ["seeded2", "seeded4", "fallback"]
    results = {case: {name: [] for name in builds} for case in cases}
    for trial in range(1 if args.dry_run else args.trials):
        case_order = cases if trial % 2 == 0 else list(reversed(cases))
        for case in case_order:
            order = list(builds)
            if trial % 2:
                order.reverse()
            pair = {}
            for name in order:
                root, native = builds[name]
                output = args.output / f"{case}-{name}-{trial}.json"
                invocation = [
                    str(root / "python/aisimulate/.venv/bin/python"),
                    str(args.worker_harness),
                    "--checkout",
                    str(root),
                    "--native-sha256",
                    native,
                    "--shared-harness",
                    str(args.shared_harness),
                    "--case",
                    case,
                    "--trial",
                    str(trial),
                    "--warm-queries",
                    str(args.warm_queries),
                    "--output",
                    str(output),
                ]
                if args.dry_run:
                    invocation.append("--dry-run")
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        invocation, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                result = json.loads(output.read_text())
                if not result["complete"]:
                    raise ValueError(f"Incomplete worker: {output}")
                if name in identities:
                    for field in (
                        "revision",
                        "code_sha256",
                        "native_sha256",
                        "dependency_freeze_sha256",
                        "supplemental_harness_sha256",
                    ):
                        if identities[name][field] != result["identity"][field]:
                            raise ValueError(f"Changed build/environment/harness: {name}: {field}")
                else:
                    identities[name] = result["identity"]
                pair[name] = result
                results[case][name].append(result)
                records.append(
                    {
                        "case": case,
                        "trial": trial,
                        "build": name,
                        "path": output.name,
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    }
                )
                print(json.dumps(records[-1]), flush=True)
            left, right = pair["baseline"], pair["candidate"]
            for field in (
                "synthetic_input_sha256",
                "training_input_sha256",
                "query_input_sha256",
                "work_contract",
                "prediction_evidence",
            ):
                if left[field] != right[field]:
                    raise ValueError(f"Paired source populations/work differ: {field}")
            if left["identity"]["dependency_freeze_sha256"] != right["identity"]["dependency_freeze_sha256"]:
                raise ValueError("Paired dependencies differ")
            if left["stages"].keys() != right["stages"].keys():
                raise ValueError("Paired timed stages differ")
    for case, case_results in results.items():
        timings[case] = {}
        if not args.dry_run:
            for stage in case_results["baseline"][0]["stages"]:
                count = case_results["baseline"][0]["stages"][stage]["operations"]
                for name in case_results:
                    if any(row["stages"][stage]["operations"] != count for row in case_results[name]):
                        raise ValueError("Changed operation counts")
                samples = {
                    name: [row["stages"][stage]["elapsed_ns"] for row in rows] for name, rows in case_results.items()
                }
                ratios = [c / b for b, c in zip(samples["baseline"], samples["candidate"], strict=True)]
                timings[case][stage] = {
                    "operations": count,
                    **{
                        name: {
                            "samples_ns": value,
                            "median_ns": statistics.median(value),
                            "median_ns_per_operation": statistics.median(value) / count,
                            "sample_stdev_ns": statistics.stdev(value),
                        }
                        for name, value in samples.items()
                    },
                    "paired_candidate_over_baseline": {"ratios": ratios, "median_ratio": statistics.median(ratios)},
                }
    value = {
        "schema_version": 1,
        "complete": True,
        "dry_run": args.dry_run,
        "trials_per_build": 1 if args.dry_run else args.trials,
        "identity": identities,
        "execution_order": records,
        "timings": timings,
        "accuracy_qualified": False,
        "interpretation": (
            "Sequential fresh processes, paired AB/BA and case order. Synthetic functional host costs only. "
            "See each worker setup field for offline preparation costs, excluded from steady public calls. "
            "Mixed snapshot support is separately probed because baseline rejects it. "
            "No affinity is available on macOS; inspect trial spread."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
