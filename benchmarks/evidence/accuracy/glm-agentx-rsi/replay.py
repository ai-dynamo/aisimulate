#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a one-variable experiment with frozen AgentX intake/scoring and an installed wheel."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tooling", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, nargs="+", required=True)
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tooling))
    from scripts.e2e_accuracy.run_agentx_accuracy import artifact, normalize_native, prepare_trace
    from scripts.e2e_accuracy.agentx_contract import native_completion_qualified, score

    catalog = read(args.inputs / "catalog/e2e/agentx.json")
    for lanes in args.lanes:
        run = next(r for r in catalog["runs"] if r["category"] == "agentx" and r["lanes"] == lanes)
        directory = args.output / run["id"]
        directory.mkdir(parents=True, exist_ok=False)
        config = read(artifact(args.inputs, run, "config"))
        config["traffic"]["source"]["paths"] = [str(prepare_trace(args.inputs, run, directory).resolve())]
        if args.max_sequences is not None:
            config["engine"]["workers"]["aggregated"]["scheduler"]["max_sequences"] = args.max_sequences
        path = directory / "input-config.json"
        write(path, config)
        trace = directory / "traces.jsonl"
        with trace.open("rb") as stream:
            trace_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        command = [sys.executable, str(args.tooling / "scripts/e2e_accuracy/agentx_predict.py"), str(path), str(directory)]
        start = time.monotonic()
        execution = {"status": "running", "command": command, "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "derived_trace": {"sha256": trace_sha, "bytes": trace.stat().st_size,
                                       "retention": "Regenerate from retained frozen HF trace and observed-output overrides"}}
        write(directory / "execution.json", execution)
        with (directory / "replay.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800, check=False)
        execution.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - start)
        outcome = read(directory / "outcome.json")
        execution.update(status=outcome["outcome"], outcome=outcome)
        write(directory / "execution.json", execution)
        if outcome["outcome"] != "completed":
            raise RuntimeError("Native replay failed; retain the experiment before continuing")
        native = read(directory / "native.json")
        if not native_completion_qualified(native, run["boundary"]):
            raise RuntimeError("Native completion did not qualify")
        observations = read(artifact(args.inputs, run, "observations"))
        scores = score(observations, normalize_native(native, observations, run["category"], run["boundary"]), run["boundary"])
        write(directory / "score.json", scores)
        trace.unlink()
        print(json.dumps({"id": run["id"], "metrics": scores["request_errors"], "coverage": scores["coverage"]}), flush=True)


if __name__ == "__main__":
    main()
