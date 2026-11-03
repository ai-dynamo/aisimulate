#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-server diagnostic controls with a hard allocation deadline."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

GOLDEN_ACCEPTANCE = {
    "SGLANG_SIMULATE_ACC_LEN": "2.99",
    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
}


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


def control(root, model, deadline, instrumented, speculation, acceptance_policy="agentx-golden", suite="timing"):
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
        "--model-loader-extra-config",
        '{"enable_multithread_load":true}',
        "--weight-loader-prefetch-checkpoints",
        "--watchdog-timeout",
        "1800",
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
    environment = dict(
        os.environ,
        PYTHONPATH=str(code),
        PYTHONNOUSERSITE="1",
        SGLANG_TIMEOUT_KEEP_ALIVE="900",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    if acceptance_policy not in ("agentx-golden", "natural"):
        raise ValueError("Unknown acceptance policy")
    for variable in GOLDEN_ACCEPTANCE:
        environment.pop(variable, None)
    if speculation and acceptance_policy == "agentx-golden":
        environment.update(GOLDEN_ACCEPTANCE)
    cache_root = Path(os.environ.get("RSI_CACHE_ROOT", root / "cache"))
    for variable, directory in {
        "HF_HOME": "hf",
        "HF_HUB_CACHE": "hf/hub",
        "HUGGINGFACE_HUB_CACHE": "hf/hub",
        "HF_MODULES_CACHE": "hf-modules",
        "SGLANG_CACHE_DIR": "sglang",
        "XDG_CACHE_HOME": "xdg",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }.items():
        path = cache_root / directory
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
            "acceptance_policy": acceptance_policy if speculation else "disabled",
            "acceptance_environment": {key: environment.get(key) for key in GOLDEN_ACCEPTANCE},
            "suite": suite,
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
                    with urllib.request.urlopen(url + "/ready", timeout=3) as response:
                        response.read()
                    with urllib.request.urlopen(url + "/health_generate", timeout=60) as response:
                        response.read()
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
                    str(code / ("frontend.py" if suite == "frontend" else "probe.py")),
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
    parser.add_argument("--acceptance-policy", choices=("agentx-golden", "natural"), default="agentx-golden")
    parser.add_argument("--suite", choices=("timing", "frontend"), default="timing")
    args = parser.parse_args()
    if not 0 < args.seconds <= 7200:
        raise ValueError("Diagnostic phase cannot exceed two node-hours")
    config_sha = hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
    if args.suite == "frontend":
        preflight = json.loads((args.output.parents[1] / "frontend-template-preflight.json").read_bytes())
        if preflight["model_config_sha256"] != config_sha or preflight["transformers"] != importlib.metadata.version(
            "transformers"
        ):
            raise ValueError("Frontend template preflight does not match model or tokenizer runtime")
        if [case["context_label"] for case in preflight["cases"]] != [8192, 131072, 393216]:
            raise ValueError("Incomplete frontend template preflight")
    deadline = time.monotonic() + args.seconds
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "runtime.json",
        {
            "sglang": importlib.metadata.version("sglang"),
            "packages": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "tokenizers")},
            "cpu": {
                "architecture": platform.machine(),
                "logical_count": os.cpu_count(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "model": next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")
                    ),
                    "unknown",
                ),
                "thread_environment": {
                    key: os.environ.get(key)
                    for key in ("TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS", "OMP_NUM_THREADS")
                },
            },
            "acceptance_policy": args.acceptance_policy,
            "suite": args.suite,
            "cache_root": os.environ.get("RSI_CACHE_ROOT", str(args.output / "cache")),
            "model_revision": "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf",
            "config_sha256": config_sha,
            "source_files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    control(args.output, args.model, deadline, False, True, args.acceptance_policy, args.suite)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("instrument.py"))]
        + (["--frontend"] if args.suite == "frontend" else []),
        check=True,
        stdout=(args.output / "instrumentation.json").open("x"),
    )
    control(args.output, args.model, deadline, True, True, args.acceptance_policy, args.suite)
    if args.suite == "timing":
        control(args.output, args.model, deadline, True, False, args.acceptance_policy)
    save(args.output / "completion.json", {"status": "completed"})


if __name__ == "__main__":
    main()
