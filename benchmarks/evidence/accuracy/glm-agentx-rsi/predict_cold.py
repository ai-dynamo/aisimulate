#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh CPU predictions for cold C1 controls; no fitting or headline replacement."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = read(args.template)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for speculation in (True, False):
        for context in (8192, 131072):
            for length in (16, 128):
                directory = args.output / f"{'mtp' if speculation else 'plain'}-c{context}-o{length}"
                directory.mkdir()
                trace = directory / "trace.jsonl"
                trace.write_text(
                    json.dumps(
                        {
                            "timestamp": 0,
                            "input_length": context,
                            "output_length": length,
                            "hash_ids": list(range(context // 64)),
                        }
                    )
                    + "\n"
                )
                config = copy.deepcopy(template)
                config["engine"]["nextn"] = 3 if speculation else 0
                config["engine"]["nextn_accepted"] = 1.99 if speculation else 0
                config["traffic"] = {
                    "source": {
                        "type": "trace",
                        "format": "mooncake",
                        "block_size": 64,
                        "paths": [str(trace.resolve())],
                    },
                    "load": {"type": "trace_timestamps"},
                }
                path = directory / "input.json"
                path.write_text(json.dumps(config, indent=2) + "\n")
                command = [sys.executable, str(args.predictor), str(path), str(directory)]
                with (directory / "replay.log").open("x") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
                outcome = read(directory / "outcome.json")
                if outcome["outcome"] != "completed":
                    raise RuntimeError("Cold control prediction failed")
                rows = read(directory / "native.json")["telemetry"]["requests"]
                if len(rows) != 1 or rows[0]["terminal_status"] != "completed" or rows[0]["output_length"] != length:
                    raise ValueError("Cold control request accounting failed")
                row = rows[0]
                summary.append(
                    {
                        "case": directory.name,
                        "input_tokens": context,
                        "output_tokens": length,
                        "nextn": config["engine"]["nextn"],
                        "accepted_drafts": config["engine"]["nextn_accepted"],
                        "ttft_ms": row["ttft_ms"],
                        "tpot_ms": row["itl_ms"],
                        "reused_tokens": row["first_admission_reused_input_tokens"],
                        "command": command,
                        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "outcome": outcome,
                    }
                )
    payload = {
        "scope": "cold C1 diagnostics; MTP targets forced total acceptance 2.99, not natural 4.0",
        "runtime_comparison": "historical silicon 0.5.20 versus prediction database 0.5.21 is cross-version",
        "cases": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
