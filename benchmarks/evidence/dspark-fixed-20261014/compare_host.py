# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare release CLI host cost using frozen repository workloads."""

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import runpy
import statistics
import subprocess
import time
from pathlib import Path

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    roots = {"base": args.base.resolve(), "head": args.head.resolve()}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    harness = roots["head"] / "python/aisimulate/tools/simulation_perf_gate"
    expand_cases = runpy.run_path(str(harness / "cases.py"))["expand_cases"]
    cases = {row["case_id"]: row for row in expand_cases()}
    agentx = copy.deepcopy(cases["agentx-vllm-aggregated"]["config"])
    agentx["traffic"]["source"]["paths"] = [str(harness / "fixtures/agentx.jsonl")]
    agentx["engine"]["model"] = "deepseek-ai/DeepSeek-V4-Flash"
    role = agentx["engine"]["workers"]["aggregated"]
    role["parallelism"].update(tensor=4, moe_expert=4)
    role["kv_cache"].update(block_size=256)
    role["kv_cache"]["capacity"]["blocks"] = 4096
    control = copy.deepcopy(cases["dense-vllm"]["config"])
    control["traffic"]["stop"]["requests"] = 128
    dspark = copy.deepcopy(agentx)
    dspark["engine"]["speculation"] = {
        "kind": "dspark",
        "num_speculative_tokens": 6,
        "accepted_tokens": 2.77,
        "adaptive_verification": False,
        "seed": 42,
        "draft_config": {"dspark_block_size": 5, "dspark_target_layer_ids": [40, 41, 42]},
    }
    configurations = {"control": control, "agentx": agentx, "agentx-dspark": dspark}
    for name, config in configurations.items():
        (out / f"{name}.yaml").write_text(yaml.safe_dump(config))
    env = {
        **os.environ,
        **dict.fromkeys(["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"], "1"),
    }
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "build_profile": "release",
        "threads": 1,
        "trace_sha256": digest(harness / "fixtures/agentx.jsonl"),
        "config_sha256": {name: digest(out / f"{name}.yaml") for name in configurations},
        "identities": {},
        "runs": [],
    }
    for label, root in roots.items():
        native = next((root / "python/aisimulate/src/aisimulate").glob("_runtime*.so"))
        dependencies = subprocess.check_output(
            [
                str(root / "python/aisimulate/.venv/bin/python"),
                "-c",
                "import importlib.metadata,json; print(json.dumps({"
                "d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()},sort_keys=True))",
            ],
            text=True,
        )
        result["identities"][label] = {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)).hexdigest(),
            "native_sha256": digest(native),
            "uv_lock_sha256": digest(root / "python/aisimulate/uv.lock"),
            "installed_dependencies_sha256": hashlib.sha256(dependencies.encode()).hexdigest(),
        }
    assert (
        result["identities"]["base"]["installed_dependencies_sha256"]
        == result["identities"]["head"]["installed_dependencies_sha256"]
    )
    # One discarded warmup per cell, then alternate base/head to reduce drift.
    for name in configurations:
        labels = ["head"] if name == "agentx-dspark" else ["base", "head"]
        for repeat in range(-1, args.repeats):
            for label in labels:
                root = roots[label]
                destination = out / f"{name}-{label}-{repeat}"
                command = [
                    str(root / "python/aisimulate/.venv/bin/aisimulate"),
                    "predict",
                    "--stack",
                    "engine",
                    "-c",
                    str(out / f"{name}.yaml"),
                    "--output-dir",
                    str(destination),
                    "--format",
                    "json",
                ]
                if platform.system() == "Darwin":
                    command = ["/usr/bin/time", "-l", *command]
                start = time.perf_counter()
                run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=600)
                elapsed = time.perf_counter() - start
                destination.mkdir(exist_ok=True)
                (destination / "stdout.log").write_text(run.stdout)
                (destination / "stderr.log").write_text(run.stderr)
                if run.returncode:
                    raise RuntimeError(f"{name}/{label} failed: {run.stderr[-2000:]}")
                report = json.loads(run.stdout)
                metrics = report.get("metrics", report.get("summary", report))
                expected = (129, 114540) if name.startswith("agentx") else (128, 16384)
                observed = (metrics.get("completed_requests"), metrics.get("total_output_tokens"))
                if observed != expected:
                    raise RuntimeError(f"{name}/{label}: incomplete work: {observed}, expected {expected}")
                rss = re.search(r"(\d+)\s+maximum resident set size", run.stderr)
                row = {
                    "case": name,
                    "build": label,
                    "repeat": repeat,
                    "wall_seconds": elapsed,
                    "completed_requests": metrics.get("completed_requests"),
                    "total_output_tokens": metrics.get("total_output_tokens"),
                    "peak_rss_bytes": int(rss[1]) if rss else None,
                }
                result["runs"].append(row)
                print(json.dumps(row), flush=True)
                (out / "host-results.json").write_text(json.dumps(result, indent=2) + "\n")
    result["medians_seconds"] = {
        f"{name}/{label}": statistics.median(
            r["wall_seconds"] for r in result["runs"] if r["case"] == name and r["build"] == label and r["repeat"] >= 0
        )
        for name in configurations
        for label in (["head"] if name == "agentx-dspark" else ["base", "head"])
    }
    (out / "host-results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
