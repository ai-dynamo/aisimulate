#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic open-loop replay of recorded arrivals, using a qualified Weka lowering.

Set PYTHONPATH to the frozen scoring-tool archive. This is not a candidate fix:
failed attempts without an observed output length cannot reproduce their GPU work.
"""

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from audit import compare_dashboard
from scripts.e2e_accuracy.agentx_contract import score
from scripts.e2e_accuracy.run_agentx_accuracy import artifact


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--lowered", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 8, 14])
    args = parser.parse_args()
    catalog = read(args.inputs / "catalog/e2e/agentx.json")
    runs = [r for r in catalog["runs"] if r["category"] == "agentx" and r["lanes"] in args.lanes]
    observations = {r["id"]: read(artifact(args.inputs, r, "observations")) for r in runs}
    needed = {row["key"] for rows in observations.values() for row in rows}
    source = {}
    with args.lowered.open() as stream:
        header = json.loads(next(stream))
        if header["block_size"] != 64:
            raise ValueError("Unexpected lowered block size")
        for line in stream:
            row = json.loads(line)
            key = row["request_id"]
            if key in needed:
                if key in source:
                    raise ValueError("Duplicate lowered source identity")
                source[key] = row
    frozen = read(args.frozen)["cohorts"]
    args.output.mkdir(parents=True, exist_ok=False)
    for run in runs:
        output = args.output / run["id"]
        output.mkdir()
        actual = observations[run["id"]]
        counts = Counter(row["key"] for row in actual)
        submitted = []
        exclusions = Counter()
        for row in actual:
            key = row["key"]
            reason = (
                "failed_measurement"
                if row["status"] != "completed"
                else "ambiguous_measurement"
                if counts[key] != 1
                else "missing_source"
                if key not in source
                else "invalid_length"
                if type(row.get("output_length")) is not int or row["output_length"] <= 0
                else None
            )
            if reason:
                exclusions[reason] += 1
                continue
            base = source[key]
            submitted.append(
                {
                    "key": key,
                    "timestamp": row["start_ms"],
                    "input_length": base["input_length"],
                    "output_length": row["output_length"],
                    "hash_ids": base["hash_ids"],
                    "phase": row["phase"],
                }
            )
        if not submitted:
            raise ValueError("No qualified requests")
        origin = min(row["timestamp"] for row in submitted)
        trace = output / "trace.jsonl"
        with trace.open("x") as stream:
            for row in submitted:
                stream.write(
                    json.dumps({k: row[k] for k in ("timestamp", "input_length", "output_length", "hash_ids")}) + "\n"
                )
        config = copy.deepcopy(read(artifact(args.inputs, run, "config")))
        config["traffic"] = {
            "source": {"type": "trace", "format": "mooncake", "block_size": 64, "paths": [str(trace.resolve())]},
            "load": {"type": "trace_timestamps"},
        }
        path = output / "input.json"
        write(path, config)
        with trace.open("rb") as stream:
            trace_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        write(
            output / "intake.json",
            {
                "submitted": len(submitted),
                "retained_observations": len(actual),
                "excluded_from_submission": exclusions,
                "trace_sha256": trace_sha,
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "purpose": "recorded-arrival diagnostic; not a candidate win",
            },
        )
        with (output / "replay.log").open("x") as log:
            subprocess.run(
                [sys.executable, str(args.predictor), str(path), str(output)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        if read(output / "outcome.json")["outcome"] != "completed":
            raise RuntimeError("Native diagnostic failed")
        native = read(output / "native.json")["telemetry"]["requests"]
        index = defaultdict(list)
        for row in native:
            index[(round(row["arrival_time_ms"], 5), row["input_length"], row["requested_output_length"])].append(row)
        expected = Counter(
            (round(row["timestamp"] - origin, 5), row["input_length"], row["output_length"]) for row in submitted
        )
        predicted = []
        for row in submitted:
            identity = (round(row["timestamp"] - origin, 5), row["input_length"], row["output_length"])
            candidates = index[identity]
            if expected[identity] != 1 or len(candidates) != 1:
                exclusions["ambiguous_or_missing_native"] += 1
                continue
            candidate = candidates[0]
            if not math.isclose(candidate["arrival_time_ms"], row["timestamp"] - origin, abs_tol=1e-5):
                raise ValueError("Arrival changed")
            predicted.append(
                {
                    "key": row["key"],
                    "phase": row["phase"],
                    "status": candidate["terminal_status"],
                    "input_length": candidate["input_length"],
                    "output_length": candidate["output_length"],
                    "start_ms": candidate["arrival_time_ms"],
                    "end_ms": candidate["terminal_time_ms"],
                    "ttft_ms": candidate["ttft_ms"],
                    "tpot_ms": candidate["itl_ms"],
                }
            )
        scores = score(actual, predicted, "profiling_window")
        write(output / "score.json", scores)
        comparison = compare_dashboard(frozen[run["id"]]["dashboard"], scores["pairs"])
        write(
            output / "comparison.json",
            {
                "metrics": comparison,
                "submission_and_pairing_exclusions": exclusions,
                "native_requests": len(native),
                "submitted": len(submitted),
                "limitations": [
                    "Recorded arrivals replace the native closed-loop controller.",
                    "Source prompt lengths/hashes are retained; HTTP template token differences remain.",
                    "Failed measurements without observed lengths are not simulated; all remain in accounting.",
                    "Diagnostic only; no headline baseline replacement or causal modeling fix.",
                ],
            },
        )
        print(json.dumps({"id": run["id"], "comparison": comparison, "exclusions": exclusions}), flush=True)


if __name__ == "__main__":
    main()
