#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-server diagnostic controls with a hard allocation deadline."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sample(url, path, stop):
    with path.open("x") as stream:
        while not stop.is_set():
            try:
                with urllib.request.urlopen(url + "/metrics", timeout=3) as response:
                    value = {"at_ns": str(time.monotonic_ns()), "metrics": response.read().decode()}
            except Exception as exc:
                value = {"at_ns": str(time.monotonic_ns()), "error": type(exc).__name__}
            stream.write(json.dumps(value) + "\n")
            stream.flush()
            stop.wait(1)


def control(root, model, deadline, instrumented, speculation):
    code = Path(__file__).parent
    name = ("instrumented" if instrumented else "unprofiled") + ("-mtp" if speculation else "-plain")
    output = root / name
    output.mkdir()
    url = "http://127.0.0.1:8011"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--served-model-name",
        "nvidia/GLM-5.3-NVFP4",
        "--trust-remote-code",
        "--tensor-parallel-size",
        "8",
        "--expert-parallel-size",
        "1",
        "--quantization",
        "modelopt_fp4",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--context-length",
        "1048576",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "8192",
        "--max-running-requests",
        "256",
        "--mem-fraction-static",
        "0.86",
        "--cuda-graph-max-bs-decode",
        "64",
        "--enable-torch-symm-mem",
        "--enable-metrics",
        "--enable-cache-report",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
        "--host",
        "127.0.0.1",
        "--port",
        "8011",
    ]
    if speculation:
        command += [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
    environment = dict(os.environ, PYTHONPATH=str(code), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for variable, directory in {
        "HF_HOME": "hf",
        "HF_HUB_CACHE": "hf/hub",
        "HUGGINGFACE_HUB_CACHE": "hf/hub",
        "HF_MODULES_CACHE": "hf-modules",
        "XDG_CACHE_HOME": "xdg",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }.items():
        path = root / "cache" / directory
        path.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(path)
    if instrumented:
        environment["RSI_TIMING_ROOT"] = str(output)
    else:
        environment.pop("RSI_TIMING_ROOT", None)
    save(
        output / "command.json",
        {
            "command": command,
            "instrumented": instrumented,
            "speculation": speculation,
            "purpose": "synthetic timing diagnostics; not AgentX accuracy scoring",
        },
    )
    stop = threading.Event()
    sampler = None
    process = None
    started = time.monotonic()
    with (output / "server.log").open("w") as log:
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
            )
            ready_deadline = min(deadline - 120, time.monotonic() + 1200)
            while True:
                if process.poll() is not None or time.monotonic() >= ready_deadline:
                    raise RuntimeError("Server startup failed or exceeded reserved time")
                try:
                    with urllib.request.urlopen(url + "/get_server_info", timeout=3) as response:
                        info = json.loads(response.read())
                    save(output / "resolved-server.json", info)
                    break
                except Exception:
                    time.sleep(2)
            sampler = threading.Thread(target=sample, args=(url, output / "server-metrics.jsonl", stop), daemon=True)
            sampler.start()
            subprocess.run(
                [
                    sys.executable,
                    str(code / "probe.py"),
                    "--url",
                    url,
                    "--model",
                    str(model),
                    "--output",
                    str(output / "client"),
                ],
                check=True,
                timeout=max(1, deadline - time.monotonic() - 90),
                env=environment,
            )
            save(output / "completion.json", {"status": "completed", "elapsed_seconds": time.monotonic() - started})
        finally:
            stop.set()
            if sampler:
                sampler.join(timeout=5)
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 7200:
        raise ValueError("Diagnostic phase cannot exceed two node-hours")
    deadline = time.monotonic() + args.seconds
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "runtime.json",
        {
            "sglang": importlib.metadata.version("sglang"),
            "model_revision": "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf",
            "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
            "source_files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    control(args.output, args.model, deadline, False, True)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("instrument.py"))],
        check=True,
        stdout=(args.output / "instrumentation.json").open("x"),
    )
    control(args.output, args.model, deadline, True, True)
    control(args.output, args.model, deadline, True, False)
    save(args.output / "completion.json", {"status": "completed"})


if __name__ == "__main__":
    main()
