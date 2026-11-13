#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 AgentX step capture: the e2e serving stack plus per-step FPM records.

Modified from benchmarks/evidence/accuracy/qwen-weka/collect.py at commit
024d22d68 (the harness of e2e job 1002947). Serving and client settings are
unchanged except that the server runs Dynamo's InstrumentedScheduler with the
aisim per-request hooks, an FPM recorder runs beside it, tiers are 20 minutes,
and the client seed is a parameter. A smoke run gates the capture runs.
"""

import argparse
import gzip
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
TIERS = (1, 4, 8, 10, 13, 14, 16, 24, 32, 48, 64)
SEEDS = (42, 7)
TIER_SECONDS = 1200
SMOKE_TIER = 4
FPM_PORT = 20380
SCHEDULER = "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
# Minimum share of non-heartbeat records that must carry consistent per-request lists.
SMOKE_LIST_SHARE = 0.99


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    """The e2e command of job 1002947, plus the FPM scheduler."""
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
        "--scheduler-cls",
        SCHEDULER,
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


def wait_file(path, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError(f"{path.name} writer exited early")
        time.sleep(0.2)
    raise TimeoutError(f"{path.name} did not appear")


def stop_process(process, timeout):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def server_env(code, aisim_src, dynamo_site, root, name):
    hooks = aisim_src / "aisim_core" / "fpm_hooks"
    # Hook directory first: its sitecustomize installs the hooks in every spawned
    # interpreter. receipt_probe (code) and the Dynamo packages follow.
    path = os.pathsep.join(str(p) for p in (hooks, code, aisim_src, dynamo_site))
    env = dict(os.environ, PYTHONPATH=path, VLLM_SERVER_DEV_MODE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    env.update(HF_HOME=str(root.parent / "preparation/hf"), TOKENIZERS_PARALLELISM="false")
    env.update(DYN_FORWARDPASS_METRIC_PORT=str(FPM_PORT), DYN_FPM_WORKER_ID=name)
    return env


def run(root, code, client, model, paths, tier, seed, duration, deadline, name):
    output = root / name
    output.mkdir(exist_ok=False)
    command = server_command(model)
    write(output / "server-command.json", command)
    env = server_env(code, *paths, root, name)
    stop = threading.Event()
    metrics = None
    recorder = None
    outcome = {"tier": tier, "seed": seed, "profile_seconds": duration, "started_ns": str(time.time_ns())}
    recorder_env = dict(env, PYTHONPATH=str(paths[0]))
    recorder_command = [
        sys.executable,
        "-m",
        "aisim_core.fpm_record",
        "--endpoint",
        f"tcp://127.0.0.1:{FPM_PORT}",
        "--output",
        str(output / "steps.jsonl.gz"),
        "--ready-file",
        str(output / "recorder.ready"),
    ]
    with (output / "server.log").open("x") as server_log, (output / "recorder.log").open("x") as recorder_log:
        recorder = subprocess.Popen(
            recorder_command, stdout=recorder_log, stderr=subprocess.STDOUT, env=recorder_env, start_new_session=True
        )
        server = None
        try:
            wait_file(output / "recorder.ready", recorder, 60)
            server = subprocess.Popen(
                command, stdout=server_log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
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
                str(tier),
                "--benchmark-duration",
                str(duration),
                "--benchmark-grace-period",
                "30",
                "--random-seed",
                str(seed),
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
            if server is not None:
                stop_process(server, 30)
            # After the server: every published step is queued; the recorder drains it.
            stop_process(recorder, 60)
            outcome["recorder_returncode"] = recorder.returncode
            outcome["ended_ns"] = str(time.time_ns())
            write(output / "outcome.json", outcome)
    return output


def check_steps(path, summary_path):
    """Smoke gate: records arrive, are numbered without loss, and carry per-request lists."""
    summary = json.loads(Path(summary_path).read_text())
    endpoints = list(summary["endpoints"].values())
    steps = with_lists = 0
    kinds = {"prefill": 0, "decode": 0, "mixed": 0}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            fpm = json.loads(line)["fpm"]
            if not fpm.get("wall_time"):
                continue
            steps += 1
            sched = fpm["scheduled_requests"]
            prefill, decode = sched["num_prefill_requests"], sched["num_decode_requests"]
            kinds["mixed" if prefill and decode else "prefill" if prefill else "decode"] += 1
            ext, past = sched.get("extend_lengths"), sched.get("past_kv_lengths")
            if ext is not None and past is not None and len(ext) == len(past) == prefill + decode:
                with_lists += 1
    checks = {
        "one_endpoint": len(endpoints) == 1,
        "steps_recorded": steps > 0,
        "no_sequence_gaps": all(e["missing_messages"] == 0 and e["out_of_order"] == 0 for e in endpoints),
        "no_undecodable": all(e["undecodable"] == 0 for e in endpoints),
        "per_request_lists": steps > 0 and with_lists / steps >= SMOKE_LIST_SHARE,
        "decode_and_mixed_steps": kinds["decode"] > 0 and kinds["mixed"] > 0,
    }
    return {
        "schema": "aisim.qwen-fpm-capture.smoke-check.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "steps": steps,
        "steps_with_lists": with_lists,
        # Pure prefill steps are rare at the smoke tier; the capture runs must have
        # all three kinds before training (replay needs a store for each).
        "step_kinds": kinds,
        "steps_sha256": sha256(path),
    }


def passed_smoke(path):
    receipt = json.loads(Path(path).read_text())
    if receipt.get("schema") != "aisim.qwen-fpm-capture.smoke-check.v1" or receipt.get("status") != "passed":
        raise ValueError("Smoke check did not pass")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--smoke", action="store_true", help="one 20-minute tier and the smoke check only")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--smoke-check", type=Path, help="passed smoke-check.json; required for a capture run")
    parser.add_argument("--tiers", type=int, nargs="+", choices=TIERS, default=list(TIERS))
    parser.add_argument("--budget-seconds", type=int, default=8 * 3600)
    args = parser.parse_args()
    if args.tiers != sorted(set(args.tiers)):
        parser.error("Tiers must be unique and ordered")
    if args.smoke != (args.seed is None and args.smoke_check is None) or (
        not args.smoke and (args.seed is None or args.smoke_check is None)
    ):
        parser.error("Use --smoke alone, or --seed with --smoke-check")
    smoke = None if args.smoke else passed_smoke(args.smoke_check)
    args.root.mkdir(exist_ok=False)
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(gpus) != 8 or any("B300" not in gpu for gpu in gpus):
        raise ValueError("Collection requires exactly eight B300 GPUs")
    write(args.root / "gpus.json", gpus)
    campaign = args.root.parent
    preparation = campaign / "preparation"
    corpus = json.loads((preparation / "corpus-receipt.json").read_bytes())
    image_sha = (preparation / "image.sha256").read_text().split()[0]
    if image_sha != "35b6881618f1ab48ef054126d659e18aebf2a009dd918b726f4220b32a274eae":
        raise ValueError("Image differs from the verified collection image")
    aisim_src = campaign / "aisim/python/aisim/src"
    dynamo_site = preparation / "dynamo-site"
    hook_files = sorted((aisim_src / "aisim_core/fpm_hooks").glob("*.py")) + [aisim_src / "aisim_core/fpm_record.py"]
    write(
        args.root / "provenance.json",
        {
            "schema": "aisim.qwen-fpm-capture.provenance.v1",
            "image_sha256": image_sha,
            "model_revision": "8472618112abcbd45acbcdc58436aff4233c23f7",
            "model_config_sha256": sha256(args.model / "config.json"),
            "dataset_revision": corpus["revision"],
            "harness_version": corpus["harness_version"],
            "vllm_source_revision": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
            "e2e_harness_commit": "024d22d68",
            "aisim_commit": (campaign / "aisim-commit.txt").read_text().strip(),
            "aisim_capture_files": {p.relative_to(aisim_src).as_posix(): sha256(p) for p in hook_files},
            "dynamo_check": (preparation / "dynamo-check.txt").read_text().strip(),
            "dynamo_site": (preparation / "dynamo-site.txt").read_text().split(),
            "server_packages_sha256": sha256(preparation / "server-packages-after.txt"),
            "scheduler_cls": SCHEDULER,
        },
    )
    code = Path(__file__).parent
    client = campaign / "client/bin/python"
    if not 0 < args.budget_seconds <= 8 * 3600:
        raise ValueError("Budget must be within eight node-hours")
    deadline = time.monotonic() + args.budget_seconds
    write(
        args.root / "execution.json",
        {
            "job_id": os.getenv("SLURM_JOB_ID"),
            "mode": "smoke" if args.smoke else "run",
            "seed": args.seed,
            "tiers": [SMOKE_TIER] if args.smoke else args.tiers,
            "tier_seconds": TIER_SECONDS,
            "smoke_check_sha256": None if args.smoke else sha256(args.smoke_check),
            "smoke_check_steps_sha256": None if smoke is None else smoke["steps_sha256"],
            "code": {
                p.relative_to(code).as_posix(): sha256(p)
                for p in code.rglob("*")
                if p.is_file() and p.suffix in {".py", ".sbatch", ".md"}
            },
        },
    )
    paths = (aisim_src, dynamo_site)
    if args.smoke:
        output = run(
            args.root, code, client, args.model, paths, SMOKE_TIER, 42, TIER_SECONDS, deadline, f"smoke-l{SMOKE_TIER}"
        )
        receipt = check_steps(output / "steps.jsonl.gz", output / "steps.jsonl.gz.summary.json")
        write(args.root / "smoke-check.json", receipt)
        print(f"SMOKE_CHECK_{receipt['status'].upper()}", flush=True)
        return
    for tier in args.tiers:
        if deadline - time.monotonic() < TIER_SECONDS + 1800 + 180:
            raise TimeoutError("Insufficient GPU budget to start a complete tier")
        run(
            args.root,
            code,
            client,
            args.model,
            paths,
            tier,
            args.seed,
            TIER_SECONDS,
            deadline,
            f"seed{args.seed}-l{tier}",
        )
    write(args.root / "capture-complete.json", {"status": "complete", "seed": args.seed, "tiers": args.tiers})


if __name__ == "__main__":
    main()
