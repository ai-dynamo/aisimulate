# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recompute every published MAPE and denominator from the portable CSV pairs."""

import argparse
import csv
import gzip
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from refresh_reported_native import read, require, sha, write


def rows(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return list(csv.DictReader(io.StringIO(raw.decode())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inventory = read(args.results / "artifact-hashes.json")
    for name, expected in inventory.items():
        require(sha(args.results / name) == expected, "artifact hash differs: " + name)
    groups = defaultdict(list)
    pair_count = 0
    for name in (
        "native-predictions.csv.gz",
        "http-predictions.csv.gz",
        "configuration-predictions.csv",
        "diagnostic-predictions.csv",
    ):
        for row in rows(args.results / name):
            if name == "diagnostic-predictions.csv" and row["role"] != "diagnostic_measurement":
                continue
            key = row["scope"], row.get("metric", "native_forward_ms")
            groups[key].append(row)
            if row["status"] == "predicted":
                suffix = "value" if name == "http-predictions.csv.gz" else "ms"
                observed, predicted = float(row["observed_" + suffix]), float(row["predicted_" + suffix])
                error = float(row["signed_error_percent"])
                require(
                    all(math.isfinite(v) for v in (observed, predicted, error)) and observed > 0 and predicted > 0,
                    "nonfinite/nonpositive prediction pair",
                )
                require(
                    math.isclose(error, 100 * (predicted / observed - 1), rel_tol=1e-12, abs_tol=1e-12),
                    "signed error does not match observed/predicted pair",
                )
                pair_count += 1
    summaries = rows(args.results / "summary.csv")
    require(len(summaries) == len(groups), "missing/extra scope summary")
    for record in summaries:
        selected = groups[record["scope"], record["metric"]]
        paired = [r for r in selected if r["status"] == "predicted"]
        require(
            len(selected) == int(record["planned_points"]) and len(paired) == int(record["predicted_points"]),
            "summary denominator differs",
        )
        if paired:
            mape = statistics.mean(abs(float(r["signed_error_percent"])) for r in paired)
            require(math.isclose(mape, float(record["mape_percent"]), rel_tol=1e-12, abs_tol=1e-12), "MAPE differs")
            suffix = "value" if "observed_value" in paired[0] else "ms"
            wape = (
                100
                * sum(abs(float(r["predicted_" + suffix]) - float(r["observed_" + suffix])) for r in paired)
                / sum(float(r["observed_" + suffix]) for r in paired)
            )
            require(math.isclose(wape, float(record["wape_percent"]), rel_tol=1e-12, abs_tol=1e-12), "WAPE differs")
        else:
            require(record["mape_percent"] == "", "missing coverage must not become zero error")
    result = {
        "valid": True,
        "summary_rows": len(summaries),
        "numeric_pairs_checked": pair_count,
        "artifact_hashes_checked": len(inventory),
        "no_predictions_executed": True,
        "script_sha256": sha(__file__),
        "inventory_sha256": sha(args.results / "artifact-hashes.json"),
    }
    if args.output:
        write(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
