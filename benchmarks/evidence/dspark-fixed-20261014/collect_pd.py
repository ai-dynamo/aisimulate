# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The Mooncake request handshake is a modified adaptation of the Apache-2.0
# vLLM example, copyright contributors to the vLLM project:
# https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py
"""Run a fixed DSpark P/D smoke and retain a replayable multi-turn corpus."""

import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

MODEL = "/models/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
OUT = pathlib.Path("/results")
K = 6


def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=300)


def session(index, warmup):
    rows = []
    prompt = [1000 + index * 100 + i % 50 for i in range(128)]
    for turn in range(3):
        if turn:
            time.sleep(0.05)
            prompt += [2000 + index * 100 + turn] * 32
        transfer_id = f"aisim-{uuid.uuid4()}"
        base = {
            "model": "dspark-flash",
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        start = time.perf_counter()
        producer = {
            **base,
            "max_tokens": 1,
            "stream": False,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "transfer_id": transfer_id,
            },
        }
        with request("http://127.0.0.1:8010/v1/completions", producer) as response:
            prefill_result = json.load(response)
        prefill_end = time.perf_counter()
        with request("http://127.0.0.1:8998/query") as response:
            engine_id = json.load(response)["0"]["engine_id"]
        consumer = {
            **base,
            "stream": True,
            "stream_options": {"include_usage": True},
            "kv_transfer_params": {
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "remote_bootstrap_addr": "http://127.0.0.1:8998",
                "remote_engine_id": engine_id,
                "transfer_id": transfer_id,
            },
        }
        tokens, first, last = [], None, None
        usage = None
        with request("http://127.0.0.1:8020/v1/completions", consumer) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    emitted = choice.get("token_ids") or []
                    if emitted:
                        last = time.perf_counter()
                        first = first or last
                        tokens.extend(emitted)
        end = time.perf_counter()
        if len(tokens) != 64 or first is None:
            raise RuntimeError(f"expected 64 explicit output token IDs, got {len(tokens)}; usage={usage}")
        rows.append(
            {
                "session": index,
                "turn": turn,
                "warmup": warmup,
                "prompt_token_ids": list(prompt),
                "output_token_ids": tokens,
                "tool_delay_ms": 50 if turn else 0,
                "ttft_ms": (first - start) * 1000,
                "e2e_ms": (end - start) * 1000,
                "tpot_ms": (last - first) * 1000 / (len(tokens) - 1),
                "prefill_http_ms": (prefill_end - start) * 1000,
                "usage": usage,
                "prefill_usage": prefill_result.get("usage"),
            }
        )
        prompt += tokens
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    print("COLLECTION_START", flush=True)
    subprocess.run(["nvidia-smi"], check=True)
    config = pathlib.Path(MODEL, "config.json").read_bytes()
    common = [
        "--model",
        MODEL,
        "--served-model-name",
        "dspark-flash",
        "--tensor-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "512",
        "--kv-cache-memory-bytes",
        str(8 << 30),
        "--seed",
        "42",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--speculative-config",
        json.dumps(
            {
                "method": "dspark",
                "num_speculative_tokens": K,
                "enable_adaptive_verification": False,
                "draft_sample_method": "probabilistic",
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_length": 3.77,
            }
        ),
    ]
    processes, logs = [], []
    manifest = {
        "model_config_sha256": hashlib.sha256(config).hexdigest(),
        "model": MODEL,
        "draft_tokens": K,
        "accepted_draft_tokens": 2.77,
        "graph_policy": "eager",
        "transfer": "MooncakeConnector/tcp",
        "commands": [],
    }
    try:
        for role, gpus, port in [("producer", "0,1,2,3", 8010), ("consumer", "4,5,6,7", 8020)]:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                *common,
                "--port",
                str(port),
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "MooncakeConnector",
                        "kv_role": "kv_" + role,
                        "kv_connector_extra_config": {"mooncake_protocol": "tcp"},
                    }
                ),
            ]
            manifest["commands"].append(command)
            log = (OUT / f"{role}.log").open("w")
            logs.append(log)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpus,
                "VLLM_MOONCAKE_BOOTSTRAP_PORT": "8998",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        deadline = time.monotonic() + 3600
        for port in [8010, 8020]:
            while True:
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("serving process exited; inspect producer/consumer logs")
                if time.monotonic() > deadline:
                    raise TimeoutError("serving startup exceeded 3600 seconds")
                try:
                    with request(f"http://127.0.0.1:{port}/health"):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(10)
        print("SERVERS_READY", flush=True)
        with (OUT / "measurements.jsonl").open("w") as output:
            for warmup, concurrency in [(True, 1), (False, 1), (False, 4)]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(session, i, warmup) for i in range(concurrency)]
                    for future in futures:
                        for row in future.result():
                            output.write(json.dumps({**row, "concurrency": concurrency}) + "\n")
                            output.flush()
        print("COLLECTION_COMPLETE", flush=True)
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        for log in logs:
            log.close()
        for name in ["producer.log", "consumer.log"]:
            path = OUT / name
            if path.exists():
                print("LOG_TAIL", name, path.read_text(errors="replace")[-16000:], flush=True)


if __name__ == "__main__":
    main()
