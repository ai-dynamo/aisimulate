#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic timing-boundary controls; these are not AgentX benchmark accuracy results."""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def call(url, value=None):
    request = urllib.request.Request(url, data=json.dumps(value).encode() if value is not None else None,
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=600)


def request(url, identity, prompt, output_length, barrier):
    body = {"rid": identity, "input_ids": prompt, "stream": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": output_length, "ignore_eos": True}}
    barrier.wait()
    start = time.monotonic_ns()
    events = []
    try:
        with call(url + "/generate", body) as stream:
            for line in stream:
                now = time.monotonic_ns()
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                payload = json.loads(line.removeprefix(b"data:"))
                meta = payload.get("meta_info", {})
                if meta.get("completion_tokens"):
                    events.append({"at_ns": str(now), "output_tokens": meta["completion_tokens"],
                                   "meta": {key: meta[key] for key in ("id", "prompt_tokens", "cached_tokens", "e2e_latency",
                                            "spec_accept_length", "spec_accept_rate", "spec_verify_ct") if key in meta}})
        status = "completed" if events and events[-1]["output_tokens"] == output_length else "incomplete"
        error = None
    except Exception as exc:
        status, error = "failed", type(exc).__name__
    return {"rid": identity, "input_tokens": len(prompt), "requested_output_tokens": output_length,
            "start_ns": str(start), "end_ns": str(time.monotonic_ns()), "events": events, "status": status, "error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    seed = tokenizer.encode("Explain how a compiler schedules independent arithmetic instructions.\n", add_special_tokens=False)
    with (args.output / "requests.jsonl").open("x") as records:
        for repeat in range(2):
            for context in (8192, 131072):
                for length in (16, 128):
                    for lanes in (1, 8, 14):
                        with call(args.url + "/flush_cache", {}) as response:
                            response.read()
                        barrier = threading.Barrier(lanes)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=lanes) as pool:
                            futures = []
                            for lane in range(lanes):
                                prefix = tokenizer.encode(f"Task {lane}: ", add_special_tokens=False)
                                prompt = (prefix + seed * (context // len(seed) + 1))[:context]
                                identity = f"rsi-r{repeat}-c{context}-o{length}-l{lanes}-i{lane}"
                                futures.append(pool.submit(request, args.url, identity, prompt, length, barrier))
                            for future in futures:
                                row = future.result()
                                row.update(lanes=lanes, repeat=repeat)
                                records.write(json.dumps(row) + "\n")
                                records.flush()
                                if row["status"] != "completed":
                                    raise RuntimeError("Incomplete diagnostic request; stop rather than dropping it")
                        print(f"repeat={repeat} context={context} output={length} lanes={lanes} completed", flush=True)


if __name__ == "__main__":
    main()
