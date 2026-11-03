#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-server diagnostic controls with a hard allocation deadline."""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from rsi_dispatch import PREFLIGHT_SOURCE_FILES

GOLDEN_ACCEPTANCE = {
    "SGLANG_SIMULATE_ACC_LEN": "2.99",
    "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
    "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def nsys_command(nsys, output, ranges):
    return [str(nsys), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi", f"--capture-range-end=repeat:{ranges}:sync",
            "--cuda-graph-trace=node:host-only:nvtx-precapture", "--force-overwrite=false",
            "--output=" + str(output)]


def qualify_cuda_trace_tables(kernel_path, projection_path):
    with kernel_path.open() as stream:
        kernels = list(csv.DictReader(stream))
    with projection_path.open() as stream:
        projections = list(csv.DictReader(stream))
    if not kernels or "Device" not in kernels[0]:
        raise ValueError("CUDA probe trace has no recognized device coverage")
    devices = {row["Device"] for row in kernels if float(row["Duration (ns)"]) > 0}
    if len(devices) != 8:
        raise ValueError("CUDA probe did not trace eight devices")
    markers = set()
    for row in projections:
        match = re.search(r"rsi-smoke-(?:capture|replay)-gpu-(\d+)", row.get("Name", ""))
        if match and float(row["Projected Duration (ns)"]) > 0:
            markers.add(int(match[1]))
    if markers != set(range(8)):
        raise ValueError("CUDA probe lacks eight GPU-projected graph/NVTX markers")
    return {"devices": sorted(devices), "gpu_projected_marker_ids": sorted(markers),
            "kernel_rows": len(kernels), "projection_rows": len(projections)}


def cuda_trace_probe(root, nsys, code, deadline):
    root.mkdir()
    command = nsys_command(nsys, root / "smoke", 1) + [
        sys.executable, str(code / "profile_cuda_probe.py"), "--output", str(root / "application.json")]
    save(root / "command.json", {"command": command})
    with (root / "probe.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=max(1, min(180, deadline - time.monotonic() - 120)))
    reports = list(root.glob("*.nsys-rep"))
    if len(reports) != 1:
        raise ValueError("CUDA trace probe did not export exactly one capture")
    with (root / "stats.log").open("x") as log:
        subprocess.run([str(nsys), "stats", "-r", "cuda_gpu_trace,nvtx_gpu_proj_trace", "--format", "csv",
                        "-o", str(root / "coverage"), str(reports[0])],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    kernels = list(root.glob("coverage*cuda_gpu_trace.csv"))
    projections = list(root.glob("coverage*nvtx_gpu_proj_trace.csv"))
    if len(kernels) != 1 or len(projections) != 1:
        raise ValueError("CUDA probe numerical report export is unavailable")
    result = qualify_cuda_trace_tables(kernels[0], projections[0])
    result["report_sha256"] = hashlib.sha256(reports[0].read_bytes()).hexdigest()
    save(root / "qualification.json", result)


def sample_gpu(path, stop):
    with path.open("x") as stream:
        while not stop.is_set():
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=index,clocks.current.sm,clocks.current.memory,power.draw,power.limit",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
                value = {"at_ns": str(time.monotonic_ns()), "returncode": result.returncode,
                         "clock_power_csv": result.stdout}
            except subprocess.TimeoutExpired:
                value = {"at_ns": str(time.monotonic_ns()), "error": "nvidia_smi_timeout"}
            stream.write(json.dumps(value) + "\n")
            stream.flush()
            stop.wait(1)


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


def control(
    root,
    model,
    deadline,
    instrumented,
    speculation,
    acceptance_policy="agentx-golden",
    suite="timing",
    coding_inputs=None,
    profile_nsys=None,
):
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
        "2" if suite == "forward-profile" else "256",
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
    if suite in ("blocking", "matrix", "coding-blocking"):
        dispatch_root = output / "dispatch"
        dispatch_root.mkdir()
        environment["RSI_DISPATCH_ROOT"] = str(dispatch_root)
    else:
        environment.pop("RSI_DISPATCH_ROOT", None)
    if suite == "forward-profile":
        environment.pop("RSI_TIMING_ROOT", None)
        if instrumented:
            environment["RSI_FORWARD_NVTX_ROOT"] = str(output / "model-receipts")
            trace_root = output / "nsys"
            trace_root.mkdir()
            command = nsys_command(profile_nsys, trace_root / "glm", 2) + command
    save(
        output / "command.json",
        {
            "command": command,
            "instrumented": instrumented,
            "speculation": speculation,
            "acceptance_policy": acceptance_policy if speculation else "disabled",
            "acceptance_environment": {key: environment.get(key) for key in GOLDEN_ACCEPTANCE},
            "suite": suite,
            "precision_environment": {
                "SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE": environment.get("SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE")
            },
            "purpose": "synthetic timing diagnostics; not AgentX accuracy scoring",
        },
    )
    stop = threading.Event()
    sampler = None
    gpu_sampler = None
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
            if suite == "forward-profile":
                gpu_sampler = threading.Thread(target=sample_gpu, args=(output / "gpu-clocks.jsonl", stop), daemon=True)
                gpu_sampler.start()
            subprocess.run(
                [
                    sys.executable,
                    str(
                        code
                        / {
                            "frontend": "frontend.py",
                            "blocking": "blocking.py",
                            "timing": "probe.py",
                            "matrix": "matrix_dispatch.py",
                            "coding-blocking": "blocking.py",
                            "forward-profile": "profile_probe.py",
                        }[suite]
                    ),
                    "--url",
                    url,
                    "--model",
                    str(model),
                    "--output",
                    str(output / "client"),
                ]
                + (
                    ["--dispatch-root", str(output / "dispatch")]
                    if suite in ("blocking", "matrix", "coding-blocking")
                    else []
                )
                + (["--coding-inputs", str(coding_inputs)] if suite == "coding-blocking" else [])
                + (["--trace"] if suite == "forward-profile" and instrumented else []),
                check=True,
                timeout=max(1, deadline - time.monotonic() - 90),
                env=environment,
            )
            if suite == "forward-profile" and instrumented:
                reports = list((output / "nsys").glob("*.nsys-rep"))
                if len(reports) != 2 or any(p.stat().st_size == 0 for p in reports):
                    raise ValueError("Both GLM capture ranges must export before qualification")
                receipts = [json.loads(p.read_bytes()) for p in (output / "model-receipts").glob("*.json")]
                for draft in (False, True):
                    role = [r for r in receipts if r["draft"] is draft]
                    if (len(role) != 8 or {r["tp_rank"] for r in role} != set(range(8))
                            or any(r["tp_size"] != 8 for r in role)):
                        raise ValueError("Loaded model receipts require every target and draft TP rank")
            save(output / "completion.json", {"status": "completed", "elapsed_seconds": time.monotonic() - started})
        finally:
            stop.set()
            if sampler:
                sampler.join(timeout=5)
            if gpu_sampler:
                gpu_sampler.join(timeout=5)
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
    parser.add_argument(
        "--suite", choices=("timing", "frontend", "blocking", "matrix", "coding-blocking", "forward-profile"),
        default="timing"
    )
    parser.add_argument("--dispatch-preflight", type=Path)
    parser.add_argument("--coding-inputs", type=Path)
    parser.add_argument("--profile-preflight", type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 7200:
        raise ValueError("Diagnostic phase cannot exceed two node-hours")
    config_sha = hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
    if args.suite in ("frontend", "blocking", "matrix", "coding-blocking"):
        preflight = json.loads((args.output.parents[1] / "frontend-template-preflight.json").read_bytes())
        if preflight["model_config_sha256"] != config_sha or preflight["transformers"] != importlib.metadata.version(
            "transformers"
        ):
            raise ValueError("Frontend template preflight does not match model or tokenizer runtime")
        if [case["context_label"] for case in preflight["cases"]] != [8192, 131072, 393216]:
            raise ValueError("Incomplete frontend template preflight")
    if args.suite in ("blocking", "matrix", "coding-blocking"):
        name = "matrix-dispatch-preflight.json" if args.suite == "matrix" else "blocking-dispatch-preflight.json"
        preflight_path = args.dispatch_preflight or args.output.parents[1] / name
        preflight = json.loads(preflight_path.read_bytes())
        if preflight["status"] != "passed" or preflight["sglang"] != importlib.metadata.version("sglang"):
            raise ValueError("Dispatch preflight does not match runtime")
        if args.suite == "matrix" and preflight.get("prepared_request", {}).get("status") != "passed":
            raise ValueError("Prepared request preflight is unavailable")
        if (
            args.suite in ("matrix", "coding-blocking")
            and preflight.get("dispatch_names", {}).get("status") != "passed"
        ):
            raise ValueError("Dispatch request-name preflight is unavailable")
        if args.suite == "coding-blocking":
            if args.coding_inputs is None or preflight.get("coding_inputs", {}).get("status") != "passed":
                raise ValueError("Actual coding inputs require CPU qualification")
            usage = preflight.get("openai_cache_usage", {})
            if usage.get("status") != "passed":
                raise ValueError("OpenAI cache-zero protocol requires actual-runtime qualification")
            usage_path = (Path(importlib.util.find_spec("sglang").origin).parent
                          / "srt/entrypoints/openai/usage_processor.py")
            if hashlib.sha256(usage_path.read_bytes()).hexdigest() != usage["source_sha256"]:
                raise ValueError("OpenAI cache serializer changed after CPU qualification")
            if set(preflight["coding_inputs"]["files"]) != {"turn-0.json", "turn-128.json"}:
                raise ValueError("Incomplete coding input qualification")
            for name, receipt in preflight["coding_inputs"]["files"].items():
                if hashlib.sha256((args.coding_inputs / name).read_bytes()).hexdigest() != receipt["sha256"]:
                    raise ValueError("Coding input changed after CPU qualification")
        for name in PREFLIGHT_SOURCE_FILES:
            actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            if preflight["source_sha256"].get(name) != actual:
                raise ValueError("Dispatch source changed after CPU preflight")
        origin = importlib.util.find_spec("sglang").origin
        manager = Path(origin).parent / "srt/managers/tokenizer_manager.py"
        if hashlib.sha256(manager.read_bytes()).hexdigest() != preflight["original_sha256"]:
            raise ValueError("Dispatch runtime source differs from CPU preflight")
    profile_nsys = None
    if args.suite == "forward-profile":
        if args.profile_preflight is None:
            raise ValueError("Forward profiling requires an actual-runtime CPU receipt")
        preflight = json.loads(args.profile_preflight.read_bytes())
        if preflight["status"] != "passed_for_gpu_probe" or preflight["sglang"] != importlib.metadata.version("sglang"):
            raise ValueError("Profiler CPU receipt does not qualify this runtime")
        package_root = Path(importlib.util.find_spec("sglang").origin).parent
        for relative, digest in preflight["runtime_sources"].items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe profiler source identity")
            if hashlib.sha256((package_root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Profiler source changed after CPU qualification")
        profile_nsys = preflight["trace_tool"]["nsys"]
        if not profile_nsys or not preflight["trace_tool"]["cuda_graph_node_option"]:
            raise ValueError("Qualified Nsight graph-node capture is unavailable")
        actual_version = subprocess.check_output([profile_nsys, "--version"], text=True, timeout=30).strip()
        if actual_version != preflight["trace_tool"]["version"]:
            raise ValueError("Nsight tool changed after CPU qualification")
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
    if args.suite == "forward-profile":
        cuda_trace_probe(args.output / "cuda-probe", profile_nsys, Path(__file__).parent, deadline)
        control(args.output, args.model, deadline, False, True, args.acceptance_policy,
                args.suite, profile_nsys=profile_nsys)
        subprocess.run([sys.executable, str(Path(__file__).with_name("instrument.py")), "--forward-nvtx"],
                       check=True, stdout=(args.output / "forward-nvtx-instrumentation.json").open("x"))
        control(args.output, args.model, deadline, True, True, args.acceptance_policy,
                args.suite, profile_nsys=profile_nsys)
        save(args.output / "completion.json", {"status": "completed"})
        return
    if args.suite in ("blocking", "matrix", "coding-blocking"):
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("instrument.py")), "--dispatch-only"],
            check=True,
            stdout=(args.output / "dispatch-instrumentation.json").open("x"),
        )
    control(args.output, args.model, deadline, False, True, args.acceptance_policy, args.suite, args.coding_inputs)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("instrument.py"))]
        + (["--frontend"] if args.suite in ("frontend", "blocking", "coding-blocking") else []),
        check=True,
        stdout=(args.output / "instrumentation.json").open("x"),
    )
    control(args.output, args.model, deadline, True, True, args.acceptance_policy, args.suite, args.coding_inputs)
    if args.suite == "timing":
        control(args.output, args.model, deadline, True, False, args.acceptance_policy)
    save(args.output / "completion.json", {"status": "completed"})


if __name__ == "__main__":
    main()
