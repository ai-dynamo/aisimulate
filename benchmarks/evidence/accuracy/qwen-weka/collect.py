#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-server AgentX pilot/profile orchestration with an eight-node-hour ceiling."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3.5-397B-A17B"
LANES = (1, 4, 8, 10, 13, 14)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def poll_metrics(output, stop):
    with output.open("x") as stream:
        while not stop.is_set():
            started = time.time_ns()
            try:
                with urllib.request.urlopen("http://127.0.0.1:8800/metrics", timeout=3) as response:
                    row = {"start_ns": str(started), "end_ns": str(time.time_ns()), "text": response.read().decode()}
            except (OSError, urllib.error.URLError) as exc:
                row = {"start_ns": str(started), "end_ns": str(time.time_ns()), "error": type(exc).__name__}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            stop.wait(max(0, 1 - (time.time_ns() - started) / 1e9))


def server_command(model):
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8800",
        "--language-model-only",
        "--tensor-parallel-size",
        "8",
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--mamba-cache-dtype",
        "bfloat16",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--mamba-cache-mode",
        "align",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--no-async-scheduling",
        "--max-model-len",
        "262144",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--seed",
        "0",
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
        "--worker-extension-cls",
        "receipt_probe.CampaignCacheProbe",
    ]


def wait_ready(server, deadline):
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("vLLM exited before readiness")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8800/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(3)
    raise TimeoutError("vLLM readiness exceeded the allocated startup window")


def run(root, code, client, model, lanes, duration, deadline, pilot=False):
    output = root / (f"pilot-l{lanes}" if pilot else f"qwen-weka-l{lanes}")
    output.mkdir(exist_ok=False)
    command = server_command(model)
    write(output / "server-command.json", command)
    env = dict(os.environ, PYTHONPATH=str(code), VLLM_SERVER_DEV_MODE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    env.update(HF_HOME=str(root.parent / "preparation/hf"), TOKENIZERS_PARALLELISM="false")
    stop = threading.Event()
    metrics = None
    start = time.time_ns()
    outcome = {"lanes": lanes, "profile_seconds": duration, "pilot": pilot, "started_ns": str(start)}
    with (output / "server.log").open("x") as server_log:
        server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            wait_ready(server, min(deadline, time.monotonic() + 1800))
            subprocess.run(
                [sys.executable, str(code / "cache_probe.py"), "--output", str(output / "cache-receipt.json")],
                env=env,
                stdout=(output / "probe.log").open("x"),
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
            metrics = threading.Thread(target=poll_metrics, args=(output / "metrics.jsonl", stop), daemon=True)
            metrics.start()
            command = [
                str(client),
                str(code / "client.py"),
                "--record",
                str(output / "client-record"),
                "--",
                "profile",
                "--url",
                "http://127.0.0.1:8800",
                "--model",
                MODEL,
                "--tokenizer",
                str(model),
                "--concurrency",
                str(lanes),
                "--benchmark-duration",
                str(duration),
                "--benchmark-grace-period",
                "30",
                "--random-seed",
                "42",
                "--cache-bust",
                "none",
                "--unsafe-override",
                "--max-context-length",
                "262144",
                "--tokenizer-trust-remote-code",
                "--server-metrics",
                "http://127.0.0.1:8800/metrics",
                "--output-artifact-dir",
                str(output / "benchmark"),
                "--public-dataset",
                "semianalysis_cc_traces_weka_062126",
                "--export-level",
                "raw",
                "--extra-inputs",
                "return_token_ids:true",
            ]
            write(output / "client-command.json", command)
            client_env = dict(
                env,
                PYTHONPATH=str(code / "client_bootstrap"),
                AGENTX_COLLECTION_RECORD=str(output / "client-record"),
                HF_HUB_OFFLINE="",
                TRANSFORMERS_OFFLINE="",
                HF_DATASETS_OFFLINE="1",
            )
            with (output / "client.log").open("x") as log:
                subprocess.run(
                    command,
                    env=client_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(duration + 1800, max(1, deadline - time.monotonic() - 30)),
                    check=True,
                )
            outcome["outcome"] = "completed"
        except BaseException as exc:
            outcome.update(outcome="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set()
            if metrics:
                metrics.join(timeout=5)
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            outcome["ended_ns"] = str(time.time_ns())
            write(output / "outcome.json", outcome)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--budget-seconds", type=int, default=8 * 3600)
    args = parser.parse_args()
    args.root.mkdir(exist_ok=False)
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(gpus) != 8 or any("B300" not in gpu for gpu in gpus):
        raise ValueError("Collection requires exactly eight B300 GPUs")
    write(args.root / "gpus.json", gpus)
    preparation = args.root.parent / "preparation"
    corpus = json.loads((preparation / "corpus-receipt.json").read_bytes())
    image_sha = (preparation / "image.sha256").read_text().split()[0]
    if image_sha != "35b6881618f1ab48ef054126d659e18aebf2a009dd918b726f4220b32a274eae":
        raise ValueError("Image differs from the verified collection image")
    write(
        args.root / "provenance.json",
        {
            "schema": "aisimulate.qwen-weka.provenance.v1",
            "image_sha256": image_sha,
            "model_revision": "8472618112abcbd45acbcdc58436aff4233c23f7",
            "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
            "dataset_revision": corpus["revision"],
            "harness_version": corpus["harness_version"],
            "vllm_source_revision": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
        },
    )
    code = Path(__file__).parent
    client = args.root.parent / "client/bin/python"
    if not 0 < args.budget_seconds <= 8 * 3600:
        raise ValueError("Remaining budget must be within eight node-hours")
    deadline = time.monotonic() + args.budget_seconds
    write(
        args.root / "execution.json",
        {
            "gpu_budget_seconds": args.budget_seconds,
            "job_id": os.getenv("SLURM_JOB_ID"),
            "code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob("*") if p.is_file()},
        },
    )
    run(args.root, code, client, args.model, 14, 600, deadline, pilot=True)
    print("PILOT_COLLECTION_COMPLETE", flush=True)
    if args.pilot_only:
        return
    gate = args.root / "PILOT_QUALIFIED.json"
    while not gate.exists():
        if time.monotonic() > deadline - 6 * 3600 - 300:
            raise TimeoutError("Pilot qualification did not leave room for six full profiles")
        time.sleep(5)
    qualification = json.loads(gate.read_text())
    if qualification.get("status") != "qualified":
        raise ValueError("Pilot did not qualify")
    for lanes in LANES:
        if deadline - time.monotonic() < 3600 + 180:
            raise TimeoutError("Insufficient GPU budget to start a complete profile")
        run(args.root, code, client, args.model, lanes, 3600, deadline)
    write(args.root / "campaign-complete.json", {"status": "complete", "lanes": list(LANES)})


if __name__ == "__main__":
    main()
