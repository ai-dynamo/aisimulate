#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay retained cold native batches; no new GPU work or AgentX headline scoring."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from analyze_timing import span


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = read(args.template)
    args.output.mkdir(parents=True, exist_ok=False)
    all_cases = []
    for mode in ("unprofiled-mtp", "instrumented-mtp", "instrumented-plain"):
        root = args.controls / mode
        settings = read(root / "resolved-server.json")
        if (
            settings["max_running_requests"] != 256 or settings["chunked_prefill_size"] != 8192
            or settings["context_length"] != 1048576 or settings["tp_size"] != 8
        ):
            raise ValueError("Controls do not match the qualified measured recipe")
        raw = (root / "client/requests.jsonl").read_bytes()
        observed = {row["rid"]: row for row in map(json.loads, raw.decode().splitlines())}
        ready = {}
        for path in root.glob("scheduler-*.jsonl"):
            for event in map(json.loads, path.read_text().splitlines()):
                if event["rank"] == 0:
                    ready.setdefault(event["rid"], []).append(event)
        for repeat in range(2):
            for context in (8192, 131072):
                for lanes in (8, 14):
                    ids = [f"rsi-r{repeat}-c{context}-o128-l{lanes}-i{lane}" for lane in range(lanes)]
                    measured = [observed[rid] for rid in ids]
                    for row in measured:
                        final = row["events"][-1]
                        if (
                            row["status"] != "completed" or row["input_tokens"] != context
                            or final["output_tokens"] != 128 or final["meta"].get("cached_tokens") != 0
                            or final["meta"].get("prompt_tokens") != context
                        ):
                            raise ValueError("Native controls fail cold/input/output qualification")
                    origin = min(int(row["start_ns"]) for row in measured)
                    name = f"{mode}-r{repeat}-c{context}-l{lanes}"
                    directory = args.output / name
                    directory.mkdir()
                    trace = directory / "trace.jsonl"
                    with trace.open("x") as stream:
                        for lane, row in enumerate(measured):
                            hashes = list(range(context // 64))
                            hashes[0] = 100000 + lane
                            # Distinct initial pages prevent prefix reuse across lanes,
                            # matching the measured Task-i prompt prefixes.
                            item = {
                                "request_id": row["rid"], "session_id": row["rid"],
                                "timestamp": (int(row["start_ns"]) - origin) / 1e6,
                                "input_length": context, "output_length": 128, "hash_ids": hashes,
                            }
                            stream.write(json.dumps(item) + "\n")
                    config = copy.deepcopy(template)
                    config["engine"].pop("tokenization", None)
                    config["engine"].pop("detokenization", None)
                    config["engine"]["nextn"] = 0 if mode.endswith("plain") else 3
                    if mode.endswith("plain"):
                        config["engine"].pop("nextn_accepted", None)
                    else:
                        config["engine"]["nextn_accepted"] = 1.99
                    config["engine"]["workers"]["aggregated"]["scheduler"]["max_sequences"] = 256
                    config["traffic"] = {
                        "source": {"type": "trace", "format": "mooncake", "block_size": 64,
                                   "paths": [str(trace.resolve())]},
                        "load": {"type": "trace_timestamps"},
                    }
                    path = directory / "input.json"
                    path.write_text(json.dumps(config, indent=2) + "\n")
                    command = [sys.executable, str(args.predictor), str(path), str(directory)]
                    with (directory / "replay.log").open("x") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
                    outcome = read(directory / "outcome.json")
                    native = read(directory / "native.json")
                    predicted = {row["session_id"]: row for row in native["telemetry"]["requests"]}
                    if outcome["outcome"] != "completed" or set(predicted) != set(ids):
                        raise ValueError("Prediction loses or duplicates controlled requests")
                    pairs = []
                    for row in measured:
                        pred = predicted[row["rid"]]
                        if (
                            pred["terminal_status"] != "completed" or pred["output_length"] != 128
                            or pred["input_length"] != context or pred["first_admission_reused_input_tokens"] != 0
                        ):
                            raise ValueError("Prediction input/output/cold accounting differs")
                        pairs.append({
                            "rid": row["rid"], "prediction_ttft_ms": pred["ttft_ms"],
                            "prediction_tpot_ms": pred["itl_ms"], "client": span(row["events"], 128),
                            "server_ready": span(ready.get(row["rid"], []), 128),
                        })
                    evidence = {
                        "case": name, "scope": "Cold native diagnostics; recorded client arrivals, not dispatch clocks",
                        "framework_comparison": "Measured SGLang0.5.20; prediction database0.5.21, cross-version",
                        "source_requests_sha256": hashlib.sha256(raw).hexdigest(),
                        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                        "command": command, "outcome": outcome, "pairs": pairs,
                        "tail_policy": "First attainment of delivered cap; retain generated excess explicitly",
                    }
                    (directory / "comparison.json").write_text(json.dumps(evidence, indent=2) + "\n")
                    all_cases.append(evidence)
                    print(json.dumps({"case": name, "retained_pairs": len(pairs)}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(all_cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
