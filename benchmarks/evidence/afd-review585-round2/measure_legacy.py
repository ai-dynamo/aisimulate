# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Five paired fresh legacy CLI estimates exercising Hopper nvfp4 normalization."""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def normalized_report(stdout):
    # Keep every log message and numerical report field; only the known
    # wall-clock/source-location prefix is not part of numerical equality.
    return re.sub(
        r"^\d{2}:\d{2}:\d{2} \[aisimulate\] \[[A-Z]\] \[[^:\]]+\.py:\d+\] ", "[log] ", stdout, flags=re.MULTILINE
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {"baseline": args.baseline.resolve(), "candidate": Path(__file__).resolve().parents[3]}
    command = [
        sys.executable,
        "-m",
        "aisimulate.legacy_cli.main",
        "estimate",
        "--model",
        "Qwen/Qwen3-30B-A3B",
        "--system",
        "h100_sxm",
        "--backend",
        "sglang",
        "--backend-version",
        "0.5.14",
        "--database-mode",
        "SOL",
        "--tp",
        "8",
        "--moe-tp-size",
        "1",
        "--moe-ep-size",
        "8",
        "--gemm-quant-mode",
        "nvfp4",
        "--moe-quant-mode",
        "nvfp4",
        "--isl",
        "1024",
        "--osl",
        "128",
        "--batch-size",
        "8",
    ]
    result = {
        "scope": "full fresh legacy CLI estimate, not request replay",
        "command": command,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": [],
        "sources": {},
    }
    for arm, source in sources.items():
        result["sources"][arm] = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
            "binary_sha256": hashlib.sha256(
                (source / "python/aisimulate/src/aisimulate/_runtime.abi3.so").read_bytes()
            ).hexdigest(),
        }
    for index in range(5):
        pair = {}
        reports = {}
        for arm in ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]:
            source = sources[arm]
            env = {
                **os.environ,
                "PYTHONPATH": str(source / "python/aisimulate/src"),
                "PYTHONHASHSEED": "0",
                **dict.fromkeys(
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"), "1"
                ),
            }
            start = time.perf_counter()
            run = subprocess.run(command, cwd=source, env=env, capture_output=True, text=True)
            pair[arm] = {
                "cli_ms": (time.perf_counter() - start) * 1000,
                "exit_code": run.returncode,
                "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest(),
            }
            if run.returncode:
                pair[arm]["stderr"] = run.stderr
                result["samples"].append(pair)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                raise RuntimeError(run.stderr)
            reports[arm] = run.stdout
        canonical = {arm: normalized_report(value) for arm, value in reports.items()}
        for arm in pair:
            pair[arm]["normalized_report_sha256"] = hashlib.sha256(canonical[arm].encode()).hexdigest()
        if canonical["baseline"] != canonical["candidate"]:
            result["samples"].append(pair)
            result["mismatched_reports"] = reports
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            raise AssertionError("legacy numerical report changed")
        assert canonical["candidate"].strip(), "empty legacy result"
        result["samples"].append(pair)
        result["numeric_report"] = canonical["candidate"]
        result["normalization"] = (
            "Remove only known legacy log timestamp/source-location prefixes; "
            "preserve all messages and numerical fields."
        )
        result["paired_cli_ratio"] = statistics.median(
            p["candidate"]["cli_ms"] / p["baseline"]["cli_ms"] for p in result["samples"]
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(index + 1, result["paired_cli_ratio"], flush=True)


if __name__ == "__main__":
    main()
