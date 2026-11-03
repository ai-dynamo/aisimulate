#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce a long-prefill/decode overlap to numerical Mooncake controls."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def run(template, root, predictor, name, arrival=None):
    output = root / name
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        {
            "request_id": "warm-prefix",
            "timestamp": 0,
            "input_length": 287424,
            "output_length": 1,
            "hash_ids": list(range(4491)),
        },
        {
            "request_id": "victim",
            "timestamp": 100000,
            "input_length": 640,
            "output_length": 23,
            "hash_ids": list(range(10000, 10010)),
        },
    ]
    if arrival is not None:
        rows.append(
            {
                "request_id": "long-prefill",
                "timestamp": arrival,
                "input_length": 361344,
                "output_length": 8,
                "hash_ids": list(range(5646)),
            }
        )
    trace = output / "trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = copy.deepcopy(template)
    config["traffic"] = {
        "source": {"type": "trace", "format": "mooncake", "block_size": 64, "paths": [str(trace.resolve())]},
        "load": {"type": "trace_timestamps"},
    }
    path = output / "input.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run([sys.executable, str(predictor), str(path), str(output)], check=True, timeout=600)
    outcome = json.loads((output / "outcome.json").read_text())
    if outcome["outcome"] != "completed":
        raise RuntimeError("Reduction replay failed")
    native = json.loads((output / "native.json").read_text())
    requests = native["telemetry"]["requests"]
    victim = next(row for row in requests if row["input_length"] == 640 and row["output_length"] == 23)
    return {
        key: victim.get(key)
        for key in (
            "arrival_time_ms",
            "first_token_ms",
            "last_token_ms",
            "itl_ms",
            "ttft_ms",
            "input_length",
            "output_length",
            "terminal_status",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.template.read_text())
    standalone = run(config, args.output, args.predictor, "standalone")
    overlapping = run(config, args.output, args.predictor, "overlap", standalone["first_token_ms"] + 1)
    separated = run(config, args.output, args.predictor, "separated", standalone["last_token_ms"] + 1)
    result = {
        "schema": "aisimulate.glm-rsi.overlap-control.v1",
        "purpose": "diagnostic, not benchmark accuracy",
        "standalone": standalone,
        "overlap": overlapping,
        "separated": separated,
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
