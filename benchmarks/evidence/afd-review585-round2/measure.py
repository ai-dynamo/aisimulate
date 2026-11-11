# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-two paired host costs; reuses the existing nine-call replay worker.

Source: benchmarks/evidence/afd-review585/measure.py at integrated baseline
444c71d55a66b5bc50ea4ddab30e69a024c30525. Modified workload matrix and identities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "benchmarks/evidence/afd-pr-preparation/measure_replay.py"


def helper():
    spec = importlib.util.spec_from_file_location("replay_measurements", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    existing = helper()
    if args.worker:
        print(json.dumps(existing.worker(json.load(sys.stdin)), allow_nan=False))
        return
    rows = [
        ("ordinary_fixed", 16, {}),
        ("ordinary_op_level", 16, {}),
        ("ordinary_op_level", 1024, {}),
        ("afd_four_stage_low", 16, {"concurrency": 1}),
        ("afd_four_stage_busy", 1024, {}),
        ("afd_two_stage", 1024, {}),
        ("afd_ragged", 64, {"a_batch_size": 8, "num_microbatches": 3}),
        ("afd_depth16", 64, {"a_batch_size": 32, "num_microbatches": 16}),
        (
            "afd_restore",
            32,
            {
                "model": "deepseek-ai/DeepSeek-V3.2",
                "tp_a": 2,
                "n_f_nodes": 2,
                "a_batch_size": 4,
                "num_microbatches": 2,
            },
        ),
    ]
    rows.extend(
        [
            (
                "ordinary_gemma4",
                16,
                {
                    "model": "google/gemma-4-26B-A4B",
                    "hardware": "b200_sxm",
                    "database_mode": "SOL",
                    "gemm_quant_mode": "bfloat16",
                    "moe_quant_mode": "bfloat16",
                    "ordinary_tp": 8,
                },
            ),
            (
                "afd_gemma4",
                16,
                {
                    "model": "google/gemma-4-26B-A4B",
                    "hardware": "b200_sxm",
                    "database_mode": "SOL",
                    "gemm_quant_mode": "bfloat16",
                    "moe_quant_mode": "bfloat16",
                },
            ),
            (
                "ordinary_nvfp4_wo_hopper",
                16,
                {
                    "hardware": "h100_sxm",
                    "database_mode": "SOL",
                    "gemm_quant_mode": "nvfp4_wo",
                    "moe_quant_mode": "nvfp4_wo",
                },
            ),
            (
                "afd_nvfp4_hopper",
                16,
                {"hardware": "h100_sxm", "database_mode": "SOL", "gemm_quant_mode": "nvfp4", "moe_quant_mode": "nvfp4"},
            ),
        ]
    )
    sources = {"baseline": args.baseline.resolve(), "candidate": ROOT}
    result = {
        "host": platform.platform(),
        "python": platform.python_version(),
        "pairs": 5,
        "cpu_affinity": "unavailable on macOS",
        "threads": existing.THREADS,
        "candidate_revision": args.candidate_revision,
        "baseline_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.baseline, text=True
        ).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        "rows": [],
    }
    for name, count, overrides in rows:
        kind = "afd_two_stage" if name == "afd_two_stage" else "afd_four_stage" if name.startswith("afd") else name
        raw = existing.configuration(kind, count)
        if name.startswith("afd"):
            topology = raw["engine"]["workers"]["aggregated"]["afd"]
            topology.update(
                {
                    k: v
                    for k, v in overrides.items()
                    if k in {"tp_a", "n_a_nodes", "n_f_nodes", "a_batch_size", "num_microbatches"}
                }
            )
        raw["engine"].update(
            {
                k: v
                for k, v in overrides.items()
                if k in {"model", "hardware", "database_mode", "gemm_quant_mode", "moe_quant_mode"}
            }
        )
        if "ordinary_tp" in overrides:
            raw["engine"]["workers"]["aggregated"]["parallelism"] = {
                "tensor": overrides["ordinary_tp"],
                "moe_tensor": 1,
                "moe_expert": overrides["ordinary_tp"],
            }
        raw["traffic"]["load"]["concurrency"] = overrides.get("concurrency", 32)
        row = {"case": name, "requests": count, "config": raw, "samples": []}
        for index in range(5):
            pair = {}
            for arm in ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]:
                source = sources[arm]
                env = {
                    **os.environ,
                    **existing.THREADS,
                    "PYTHONHASHSEED": "0",
                    "PYTHONPATH": str(source / "python/aisimulate/src"),
                }
                output = subprocess.check_output(
                    [sys.executable, str(Path(__file__).resolve()), "--worker"],
                    input=json.dumps(raw),
                    text=True,
                    env=env,
                    cwd=source,
                )
                sample = json.loads(output)
                runtime = source / "python/aisimulate/src/aisimulate/_runtime.abi3.so"
                assert sample["runtime"] == str(runtime.resolve())
                assert sample["binary_sha256"] == hashlib.sha256(runtime.read_bytes()).hexdigest()
                with tempfile.TemporaryDirectory(prefix="afd-review585-cli-") as tmp:
                    config = Path(tmp) / "config.json"
                    config.write_text(json.dumps(raw))
                    start = time.perf_counter()
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "aisimulate",
                            "predict",
                            "--stack",
                            "engine",
                            "--config",
                            str(config),
                            "--output-dir",
                            str(Path(tmp) / "out"),
                            "--format",
                            "json",
                        ],
                        env=env,
                        cwd=source,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    sample["cli_ms"] = (time.perf_counter() - start) * 1000
                    prediction = json.loads((Path(tmp) / "out/prediction.json").read_text())
                    summary = prediction.get("summary", prediction)
                    for key, expected in {
                        "completed_requests": count,
                        "total_input_tokens": count * 1024,
                        "total_output_tokens": count * 128,
                    }.items():
                        assert summary[key] == expected
                pair[arm] = sample
            assert pair["baseline"]["metrics"] == pair["candidate"]["metrics"]
            row["samples"].append(pair)
        row["ratios"] = {
            metric: statistics.median(
                [
                    (statistics.median(p["candidate"][metric]) / statistics.median(p["baseline"][metric]))
                    if isinstance(p["candidate"][metric], list)
                    else p["candidate"][metric] / p["baseline"][metric]
                    for p in row["samples"]
                ]
            )
            for metric in (
                "compile_ms",
                "empty_native_ms",
                "first_runner_ms",
                "warm_runner_ms",
                "cli_ms",
                "last_native_wall_ms",
            )
        }
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(name, json.dumps(row["ratios"]), flush=True)


if __name__ == "__main__":
    main()
