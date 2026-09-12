#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time real recommend entrypoints; compare search algorithms, not estimator parity.

Use the variant JSON described in docs/cli/runtime-benchmark.md. Each variant
provides label, kind, root, python, and sources. Unix process groups ensure a
timed-out command does not leave its search workers running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import statistics
import subprocess
import time
from pathlib import Path

from benchmark_migration_runtime import BACKEND_VERSION, MODEL, SYSTEM, attest, digest


def configuration(trials, parallelism, algorithm):
    return {
        "engine": {
            "mode": "aggregated",
            "model": MODEL,
            "hardware": SYSTEM,
            "backend": "vllm",
            "backend_version": BACKEND_VERSION,
            "workers": {"aggregated": {"parallelism": {"preset": "default"}}},
        },
        "optimization": {"target": "throughput_per_gpu", "constraints": {"max_candidate_gpus": 8}},
        "optimizer": {"algorithm": algorithm, "max_trials": trials, "parallelism": parallelism, "seed": 42},
    }


def run(command, env, stdout_path, stderr_path, timeout):
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        start = time.perf_counter()
        child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise RuntimeError(f"command exceeded {timeout}s; inspect {stderr_path}") from None
        elapsed = time.perf_counter() - start
    if code:
        raise RuntimeError(f"command exited {code}; inspect {stdout_path} and {stderr_path}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--systems-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--algorithms", nargs="+", choices=["bayesian", "random"], default=["bayesian", "random"])
    args = parser.parse_args()
    if os.name != "posix" or min(args.rounds, args.trials, args.parallelism, args.timeout) <= 0:
        parser.error("requires a Unix host and positive bounds")
    variants = json.loads(Path(args.variants).read_text())
    if not variants or len({v["label"] for v in variants}) != len(variants):
        parser.error("variants must have distinct labels and cannot be empty")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    systems = Path(args.systems_path).resolve()
    manifest = {
        str(p.relative_to(systems)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(systems.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    evidence = {
        "protocol": 1,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "systems_sha256": digest(manifest),
        "rounds": args.rounds,
        "trials": args.trials,
        "parallelism": args.parallelism,
        "variants": variants,
        "samples": [],
        "scope": (
            "AIC minimum-GPU analytical sizing versus AISimulate aggregated throughput-per-GPU replay search; "
            "results are not equivalent."
        ),
    }
    cases = []
    for variant in variants:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in variant["sources"])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = "1"
        variant["attestation"] = attest(variant, env)
        package = Path(variant["root"]).resolve()
        if variant["kind"] == "aiconfigurator":
            command = [
                variant["python"],
                str(package / ".venv/bin/aiconfigurator"),
                "cli",
                "recommend",
                "--model-path",
                MODEL,
                "--system",
                SYSTEM,
                "--backend",
                "vllm",
                "--backend-version",
                BACKEND_VERSION,
                "--target-concurrency",
                "10",
                "--isl",
                "1024",
                "--osl",
                "128",
                "--ttft",
                "2000",
                "--tpot",
                "30",
                "--systems-paths",
                str(systems),
            ]
            cases.append((variant, "aic", command, env))
        elif variant["kind"] == "aisimulate":
            # Unified CLI uses its bundled systems tree. Require it to match the
            # explicit AIC tree instead of relying on an ignored environment variable.
            bundled = package / "python/aisimulate/src/aiconfigurator_core/systems"
            own = {
                str(p.relative_to(bundled)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(bundled.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts
            }
            if digest(own) != evidence["systems_sha256"]:
                raise RuntimeError(f"{variant['label']} systems data differs from the shared comparison tree")
            # A base checkout can reuse an interpreter with PYTHONPATH pinned to
            # its source, while executing that interpreter's installed entrypoint.
            entrypoint = str(Path(variant["python"]).parent / "aisimulate")
            for algorithm in args.algorithms:
                config = output / f"{variant['label']}-{algorithm}.json"
                config.write_text(json.dumps(configuration(args.trials, args.parallelism, algorithm), indent=2))
                command = [variant["python"], entrypoint, "recommend", "--config", str(config), "--format", "json"]
                cases.append((variant, algorithm, command, env))
        else:
            parser.error(f"unknown variant kind {variant['kind']!r}")
    evidence["configurations"] = {a: configuration(args.trials, args.parallelism, a) for a in args.algorithms}
    for repeat in range(args.rounds):
        for variant, algorithm, base_command, env in cases if repeat % 2 == 0 else list(reversed(cases)):
            label = f"{variant['label']}-{algorithm}-{repeat}"
            destination = output / label
            command = list(base_command)
            if algorithm != "aic":
                command.extend(["--output-dir", str(destination)])
            elapsed = run(command, env, output / f"{label}.stdout", output / f"{label}.stderr", args.timeout)
            row = {
                "variant": variant["label"],
                "algorithm": algorithm,
                "round": repeat,
                "process_wall_s": elapsed,
                "command": command,
            }
            if algorithm != "aic":
                report = json.loads((destination / "recommendation.json").read_text())
                if report["counts"]["failed"] or report["counts"]["timed_out"] or not report["views"]["top_n"]:
                    raise RuntimeError(f"{label} did not produce a successful recommendation")
                row["counts"] = report["counts"]
                row["best_score"] = max(r["score"] for r in report["candidates"] if r["score"] is not None)
                row["report"] = report
            evidence["samples"].append(row)
            (output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(label, round(elapsed, 3), flush=True)
    for variant, algorithm, _, env in cases:
        if attest(variant, env) != variant["attestation"]:
            raise RuntimeError(f"{variant['label']} source/native identity changed during measurement")
        rows = [r for r in evidence["samples"] if r["variant"] == variant["label"] and r["algorithm"] == algorithm]
        print(variant["label"], algorithm, "median_s", statistics.median(r["process_wall_s"] for r in rows))


if __name__ == "__main__":
    main()
