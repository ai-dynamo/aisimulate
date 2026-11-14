# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure matched ordinary replay and candidate AFD scaling on one macOS host."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"), "1")
HOST_KEYS = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}


def configuration(case, count):
    worker = {"kv_cache": {"capacity": {"type": "fixed", "blocks": 65536}}}
    if case.startswith("afd"):
        worker = {
            "type": "afd",
            "afd": {
                "n_a_nodes": 1,
                "n_f_nodes": 1,
                "tp_a": 1,
                "a_batch_size": 18,
                "num_microbatches": 3,
                "pipeline_layout": case[4:],
            },
        }
    elif case == "ordinary_fixed":
        worker["timing"] = {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}
    return {
        "engine": {
            "model": "Qwen/Qwen3-30B-A3B",
            "hardware": "h200_sxm",
            "backend": "sglang",
            "backend_version": "0.5.14",
            "mode": "aggregated",
            "context_length": 2048,
            "workers": {"aggregated": worker},
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 1024, "output_tokens": 128},
            "load": {"type": "concurrency", "concurrency": 32},
            "stop": {"requests": count},
        },
    }


def worker(raw):
    from aisimulate import _runtime
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory, _materialize_engine_execution_spec

    config = CorePredictionConfig.model_validate(raw)
    started = time.perf_counter()
    spec = prediction_to_replay_spec(config)
    compile_ms = (time.perf_counter() - started) * 1000
    execution = _materialize_engine_execution_spec(spec, trace_block_size=512, record_per_request=False)
    execution = execution.get("spec", execution)
    execution["requests"] = []
    # The workload driver is separately encoded at the top level. Remove it for a true empty replay.
    execution.pop("traffic", None)
    started = time.perf_counter()
    _runtime.run_replay_json(json.dumps(execution, allow_nan=False))
    preparation_ms = (time.perf_counter() - started) * 1000
    runner = EngineReplayRunnerFactory().create(0)
    samples = []
    for _ in range(9):
        started = time.perf_counter()
        report = runner.run(spec)
        samples.append((time.perf_counter() - started) * 1000)
    count = raw["traffic"]["stop"]["requests"]
    assert report.metrics["completed_requests"] == count
    assert report.metrics["total_input_tokens"] == count * 1024
    assert report.metrics["total_output_tokens"] == count * 128
    return {
        "runtime": str(Path(_runtime.__file__).resolve()),
        "binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        "compile_ms": compile_ms,
        "empty_native_ms": preparation_ms,
        "first_runner_ms": samples[0],
        "warm_runner_ms": samples[4:],
        "last_native_wall_ms": report.metrics["wall_time_ms"],
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "metrics": {k: v for k, v in report.metrics.items() if k not in HOST_KEYS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin)), allow_nan=False))
        return
    root = Path(__file__).resolve().parents[3]
    sources = {"baseline": args.baseline.resolve(), "candidate": root}
    result = {
        "host": platform.platform(),
        "python": platform.python_version(),
        "cpu_affinity": "unavailable on macOS",
        "pairs": 5,
        "thread_env": THREADS,
        "scope": {
            "empty_native_ms": "native setup with requests=[] and traffic removed",
            "runner_ms": "complete Python runner call including native model construction, memory, replay and reports",
            "first_runner_ms": "first complete runner call after the separately timed empty preparation",
            "warm_runner_ms": (
                "five complete runner calls after the first call and three warmups; "
                "providers are reconstructed each call"
            ),
            "cli_ms": "fresh CLI including supervision, construction, replay and output",
            "rss": "whole fresh worker process peak RSS, not retained cache size",
        },
        "identities": {},
        "rows": [],
    }
    for arm, source in sources.items():
        runtime = source / "python/aisimulate/src/aisimulate/_runtime.abi3.so"
        result["identities"][arm] = {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
            "runtime": str(runtime.resolve()),
            "binary_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        }
    for case in ["ordinary_fixed", "ordinary_op_level", "afd_four_stage", "afd_two_stage"]:
        for count in [16, 1024]:
            row = {"case": case, "requests": count, "samples": []}
            for index in range(5):
                arms = (
                    ["candidate"]
                    if case.startswith("afd")
                    else (["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"])
                )
                pair = {}
                for arm in arms:
                    source = sources[arm]
                    env = {
                        **os.environ,
                        **THREADS,
                        "PYTHONHASHSEED": "0",
                        "PYTHONPATH": str(source / "python/aisimulate/src"),
                    }
                    raw = configuration(case, count)
                    p = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker"],
                        input=json.dumps(raw),
                        env=env,
                        cwd=source,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    sample = json.loads(p.stdout)
                    assert sample["runtime"] == result["identities"][arm]["runtime"]
                    assert sample["binary_sha256"] == result["identities"][arm]["binary_sha256"]
                    with tempfile.TemporaryDirectory(prefix="afd-pr-cli-") as directory:
                        config = Path(directory) / "config.json"
                        config.write_text(json.dumps(raw))
                        started = time.perf_counter()
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
                                str(Path(directory) / "out"),
                                "--format",
                                "json",
                            ],
                            env=env,
                            cwd=source,
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        sample["cli_ms"] = (time.perf_counter() - started) * 1000
                        prediction = json.loads((Path(directory) / "out/prediction.json").read_text())
                        summary = prediction.get("summary", prediction)
                        sample["cli_work_counts"] = {
                            k: summary[k] for k in ("completed_requests", "total_input_tokens", "total_output_tokens")
                        }
                        assert sample["cli_work_counts"] == {
                            "completed_requests": count,
                            "total_input_tokens": count * 1024,
                            "total_output_tokens": count * 128,
                        }
                    pair[arm] = sample
                if "baseline" in pair:
                    assert pair["baseline"]["metrics"] == pair["candidate"]["metrics"], pair
                row["samples"].append(pair)
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(case, count, "complete", flush=True)


if __name__ == "__main__":
    main()
