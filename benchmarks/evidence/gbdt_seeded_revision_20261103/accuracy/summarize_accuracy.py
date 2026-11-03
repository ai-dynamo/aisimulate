#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare fresh pinned Gym traces on identical scoring and admission populations.

The output contains aggregate accuracy/coverage and one-minute learning curves,
not private observation payloads. No acceptance threshold is invented here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(root, runtime_root):
    path = root / "run.json"
    run = json.loads(path.read_text())
    if not run["complete"] or run["failed"]:
        raise ValueError(f"Incomplete qualification: {root}")
    identity = run["identity"]
    expected_root = runtime_root.resolve()
    origins = [identity["native_path"], *(value["root"] for value in identity["runtime_python_sources"].values())]
    if any(not Path(origin).resolve().is_relative_to(expected_root) for origin in origins):
        raise ValueError(f"Qualification imported another checkout: {root}")
    cells = {}
    for entry in run["cells"]:
        result = root / entry["path"]
        if sha(result) != entry["sha256"]:
            raise ValueError(f"Result digest mismatch: {result}")
        data = json.loads(result.read_text())
        trace = result.parent / "predictions.jsonl.gz"
        if sha(trace) != data["trace_sha256"]:
            raise ValueError(f"Trace digest mismatch: {trace}")
        cells[entry["cell_id"]] = (data, trace)
    return run, cells


def records(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def accumulator():
    return {"measured": 0, "predicted": 0, "ape_sum": 0.0, "squared_error_sum": 0.0, "updates": 0}


def add(result, row):
    result["measured"] += 1
    result["updates"] += int(row["update_submitted"])
    predicted, actual = row["predicted_ms"], row["actual_ms"]
    if not math.isfinite(actual) or actual <= 0:
        raise ValueError("Nonpositive or nonfinite measured target")
    if predicted is not None:
        if not math.isfinite(predicted) or predicted <= 0:
            raise ValueError("Nonpositive or nonfinite prediction")
        result["predicted"] += 1
        result["ape_sum"] += abs(predicted - actual) / actual * 100.0
        result["squared_error_sum"] += (predicted - actual) ** 2


def metrics(value):
    count, predicted = value["measured"], value["predicted"]
    return {
        "measured_count": count,
        "predicted_count": predicted,
        "coverage_pct": 100.0 * predicted / count if count else None,
        "mape_pct": value["ape_sum"] / predicted if predicted else None,
        "rmse_ms": math.sqrt(value["squared_error_sum"] / predicted) if predicted else None,
        "admitted_updates": value["updates"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-runtime-root", type=Path, required=True)
    parser.add_argument("--candidate-runtime-root", type=Path, required=True)
    args = parser.parse_args()
    base_run, baseline = load_run(args.baseline, args.baseline_runtime_root)
    candidate_run, candidate = load_run(args.candidate, args.candidate_runtime_root)
    for key in ("evaluator_revision", "evaluator_sources_sha256", "protocol_id", "harness_sha256"):
        if base_run["identity"][key] != candidate_run["identity"][key]:
            raise ValueError(f"Qualification evaluator differs: {key}")
    if base_run["identity"]["dependencies"] != candidate_run["identity"]["dependencies"]:
        raise ValueError("Qualification dependency versions differ")
    if base_run["cases"] != candidate_run["cases"]:
        raise ValueError("Qualification input evidence differs")
    if baseline.keys() != candidate.keys():
        raise ValueError("Different qualification cells")
    results, curves = [], []
    for cell in sorted(baseline):
        before, before_trace = baseline[cell]
        after, after_trace = candidate[cell]
        for key in (
            "scoring_population_sha256",
            "training_population_sha256",
            "online_update_count",
            "online_schedule",
            "test_count",
        ):
            if before["result"][key] != after["result"][key]:
                raise ValueError(f"Changed {key}: {cell}")
        totals = [accumulator(), accumulator()]
        common = [accumulator(), accumulator()]
        by_kind = defaultdict(lambda: [accumulator(), accumulator()])
        windows = defaultdict(lambda: [accumulator(), accumulator()])
        start = None
        differing_predictions = 0
        largest_prediction_delta_ms = 0.0
        for left, right in zip(records(before_trace), records(after_trace), strict=True):
            for key in (
                "observation_id",
                "actual_ms",
                "event_time_ms",
                "worker_id",
                "workload_kind",
                "native_workload_kind",
                "update_submitted",
            ):
                if left[key] != right[key]:
                    raise ValueError(f"Changed prediction/admission population {cell}: {key}")
            if left.get("submitted_rows_sha256") != right.get("submitted_rows_sha256"):
                raise ValueError(f"Changed admitted rank payloads: {cell}")
            if left["update_submitted"] and not left.get("submitted_rows_sha256"):
                raise ValueError(f"Missing admitted rank payload digest: {cell}")
            before_prediction, after_prediction = left["predicted_ms"], right["predicted_ms"]
            before_bits = None if before_prediction is None else float(before_prediction).hex()
            after_bits = None if after_prediction is None else float(after_prediction).hex()
            differing_predictions += before_bits != after_bits
            if before_prediction is not None and after_prediction is not None:
                largest_prediction_delta_ms = max(
                    largest_prediction_delta_ms, abs(before_prediction - after_prediction)
                )
            if start is None:
                start = left["event_time_ms"]
            minute = int((left["event_time_ms"] - start) // 60000)
            for index, row in enumerate((left, right)):
                add(totals[index], row)
                add(by_kind[row["native_workload_kind"]][index], row)
                add(windows[minute][index], row)
                if left["predicted_ms"] is not None and right["predicted_ms"] is not None:
                    add(common[index], row)
        names = ("baseline", "candidate")
        result = {
            "cell_id": cell,
            "differing_prediction_bits": differing_predictions,
            "largest_common_prediction_delta_ms": largest_prediction_delta_ms,
            "reported_status": {
                name: {
                    "status": record["result"]["status"],
                    **{
                        key: record["result"]["metrics"]["all"][key]
                        for key in ("error_count", "tuning_error_count", "unavailable_count")
                    },
                }
                for name, record in zip(names, (before, after), strict=True)
            },
            "all": {name: metrics(value) for name, value in zip(names, totals, strict=True)},
            "common_prediction_population": {name: metrics(value) for name, value in zip(names, common, strict=True)},
            "by_kind": {
                kind: {name: metrics(value) for name, value in zip(names, values, strict=True)}
                for kind, values in sorted(by_kind.items())
            },
        }
        for index, record in enumerate((before, after)):
            expected = record["result"]["metrics"]["all"]
            actual = metrics(totals[index])
            for key in ("measured_count", "predicted_count", "coverage_pct", "mape_pct"):
                if expected[key] is None or actual[key] is None:
                    if expected[key] != actual[key]:
                        raise ValueError(f"Trace and scorer disagree: {cell}: {key}")
                elif not math.isclose(expected[key], actual[key], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"Trace and scorer disagree: {cell}: {key}")
        results.append(result)
        for minute, values in sorted(windows.items()):
            for name, value in zip(names, values, strict=True):
                curves.append({"cell_id": cell, "elapsed_minute": minute, "implementation": name, **metrics(value)})
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "population_and_admission_equivalent": True,
        "all_prediction_bits_identical": all(cell["differing_prediction_bits"] == 0 for cell in results),
        "baseline": {"path": str(args.baseline), "run_sha256": sha(args.baseline / "run.json"), **base_run["identity"]},
        "candidate": {
            "path": str(args.candidate),
            "run_sha256": sha(args.candidate / "run.json"),
            **candidate_run["identity"],
        },
        "cells": results,
        "publication_qualified": False,
        "interpretation": (
            "Fresh recipe comparison; coverage and common-population error are separate. "
            "No hardware bundle qualification is implied."
        ),
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output / "learning-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    print(json.dumps({"cells": len(results), "curve_rows": len(curves), "output": str(args.output)}))


if __name__ == "__main__":
    main()
