#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a one-variable experiment with frozen AgentX intake/scoring and an installed wheel."""

import argparse
import hashlib
import json
import math
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
    parser.add_argument("--host-loop", action="store_true")
    parser.add_argument("--tokenization-profile", type=Path)
    parser.add_argument("--shared-response-gate", action="store_true")
    args = parser.parse_args()
    if args.shared_response_gate and not args.tokenization_profile:
        raise ValueError("Shared delivery requires the fixed measured CPU profile")
    tokenization = None
    profile_sha = None
    if args.tokenization_profile:
        if args.max_sequences is not None or (args.host_loop and not args.shared_response_gate):
            raise ValueError("CPU scheduler composition requires shared delivery; capacity changes stay separate")
        profile_bytes = args.tokenization_profile.read_bytes()
        profile = json.loads(profile_bytes)
        if profile.get("schema") != "aisimulate.glm-rsi.component-tokenization.v1":
            raise ValueError("Unknown measured component profile")
        tokenization = profile["tokenization"]
        rate = tokenization.get("per_input_token_us")
        if (
            set(tokenization) != {"workers", "base_ms", "per_input_token_us"}
            or type(tokenization["workers"]) is not int
            or tokenization["workers"] != 1
            or isinstance(tokenization["base_ms"], bool)
            or tokenization["base_ms"] != 0
            or isinstance(rate, bool)
            or not isinstance(rate, (float, int))
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError("Requires the predeclared positive component rate and one CPU server")
        profile_sha = hashlib.sha256(profile_bytes).hexdigest()
        if args.shared_response_gate:
            tokenization = {**tokenization, "share_with_response": True}
    sys.path.insert(0, str(args.tooling))
    from scripts.e2e_accuracy.agentx_contract import native_completion_qualified, score
    from scripts.e2e_accuracy.run_agentx_accuracy import artifact, normalize_native, prepare_trace

    catalog = read(args.inputs / "catalog/e2e/agentx.json")
    for lanes in args.lanes:
        run = next(r for r in catalog["runs"] if r["category"] == "agentx" and r["lanes"] == lanes)
        directory = args.output / run["id"]
        directory.mkdir(parents=True, exist_ok=False)
        config = read(artifact(args.inputs, run, "config"))
        config["traffic"]["source"]["paths"] = [str(prepare_trace(args.inputs, run, directory).resolve())]
        if args.max_sequences is not None:
            config["engine"]["workers"]["aggregated"]["scheduler"]["max_sequences"] = args.max_sequences
        if args.host_loop:
            config["engine"]["workers"]["aggregated"]["host_loop"] = True
        if tokenization is not None:
            config["engine"]["tokenization"] = tokenization
        path = directory / "input-config.json"
        write(path, config)
        trace = directory / "traces.jsonl"
        with trace.open("rb") as stream:
            trace_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        command = [
            sys.executable,
            str(args.tooling / "scripts/e2e_accuracy/agentx_predict.py"),
            str(path),
            str(directory),
        ]
        start = time.monotonic()
        execution = {
            "status": "running",
            "command": command,
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "derived_trace": {
                "sha256": trace_sha,
                "bytes": trace.stat().st_size,
                "retention": "Regenerate from retained frozen HF trace and observed-output overrides",
            },
        }
        if profile_sha:
            execution["tokenization_profile_sha256"] = profile_sha
            execution["scope"] = (
                "Measured CPU admission and causal zero-service output gate" if args.shared_response_gate
                else "Measured CPU admission only; shared API response blocking remains unmodeled"
            )
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
        if tokenization is not None:
            timed = [row for row in native["per_request"] if row.get("tokenized_ms") is not None]
            if not timed:
                raise RuntimeError("Configured CPU component is absent from native observations")
            for row in timed:
                dispatched = row.get("dispatched_at_ms")
                minimum_ms = row["input_length"] * tokenization["per_input_token_us"] / 1000
                if dispatched is None or row["tokenized_ms"] + 1e-6 < dispatched + minimum_ms:
                    raise RuntimeError("Native tokenization does not conserve measured CPU service")
            execution["tokenization_observed_records"] = len(timed)
            write(directory / "execution.json", execution)
        if not native_completion_qualified(native, run["boundary"]):
            raise RuntimeError("Native completion did not qualify")
        observations = read(artifact(args.inputs, run, "observations"))
        scores = score(
            observations, normalize_native(native, observations, run["category"], run["boundary"]), run["boundary"]
        )
        write(directory / "score.json", scores)
        trace.unlink()
        print(
            json.dumps({"id": run["id"], "metrics": scores["request_errors"], "coverage": scores["coverage"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
