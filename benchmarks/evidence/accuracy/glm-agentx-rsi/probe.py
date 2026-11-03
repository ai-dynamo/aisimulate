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
    request = urllib.request.Request(
        url,
        data=json.dumps(value).encode() if value is not None else None,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=600)


def retain_case(rows, records):
    failed = False
    for row in rows:
        records.write(json.dumps(row) + "\n")
        records.flush()
        failed |= row["status"] != "completed"
    if failed:
        raise RuntimeError("Incomplete diagnostic case; all attempted requests retained")


def request(url, identity, prompt, output_length, barrier, first_output=None):
    body = {
        "rid": identity,
        "input_ids": prompt,
        "stream": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": output_length, "ignore_eos": True},
    }
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
                    if first_output is not None:
                        first_output.set()
                    events.append(
                        {
                            "at_ns": str(now),
                            "output_tokens": meta["completion_tokens"],
                            "meta": {
                                key: meta[key]
                                for key in (
                                    "id",
                                    "prompt_tokens",
                                    "cached_tokens",
                                    "e2e_latency",
                                    "spec_accept_length",
                                    "spec_accept_rate",
                                    "spec_verify_ct",
                                )
                                if key in meta
                            },
                        }
                    )
        status = "completed" if events and events[-1]["output_tokens"] == output_length else "incomplete"
        error = None
    except Exception as exc:
        status, error = "failed", type(exc).__name__
    return {
        "rid": identity,
        "input_tokens": len(prompt),
        "requested_output_tokens": output_length,
        "start_ns": str(start),
        "end_ns": str(time.monotonic_ns()),
        "events": events,
        "status": status,
        "error": error,
    }


def overlap_control(url, seed, records):
    """Use client first-output as the trigger; scheduler evidence establishes actual overlap."""
    prefix = (seed * (361344 // len(seed) + 1))[:361344]
    victim = ([seed[-1]] + seed * (640 // len(seed) + 1))[:640]
    for overlap in (False, True):
        with call(url + "/flush_cache", {}) as response:
            response.read()
        warm = request(url, f"rsi-overlap-{overlap}-warm", prefix[:287424], 1, threading.Barrier(1))
        warm["role"] = "preparation"
        records.write(json.dumps(warm) + "\n")
        if warm["status"] != "completed":
            raise RuntimeError("Incomplete overlap preparation")
        first = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            task = pool.submit(request, url, f"rsi-overlap-{overlap}-victim", victim, 23, threading.Barrier(1), first)
            other = None
            if overlap:
                if not first.wait(timeout=600):
                    raise RuntimeError("Victim produced no observed output")
                time.sleep(0.001)
                other = pool.submit(request, url, "rsi-overlap-interferer", prefix, 8, threading.Barrier(1))
            rows = [task.result()] + ([other.result()] if other else [])
            for row in rows:
                row.update(role="diagnostic_overlap", overlap_requested=overlap)
            retain_case(rows, records)


def chat_controls(url, seed, tokenizer, records):
    """Check the original OpenAI frontend and reasoning parser without saving text."""
    for repeat in range(2):
        for context, length, lanes in ((8192, 16, 1), (8192, 16, 14), (131072, 128, 1)):
            with call(url + "/flush_cache", {}) as response:
                response.read()
            prompt = (seed * (context // len(seed) + 1))[:context]
            text = tokenizer.decode(prompt)
            barrier = threading.Barrier(lanes)

            def one(lane):
                rid = f"rsi-chat-r{repeat}-c{context}-o{length}-l{lanes}-i{lane}"
                body = {
                    "rid": rid,
                    "model": "nvidia/GLM-5.3-NVFP4",
                    "messages": [{"role": "user", "content": text}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "temperature": 0,
                    "max_tokens": length,
                    "ignore_eos": True,
                }
                barrier.wait()
                start = time.monotonic_ns()
                events, usage = [], {}
                error = None
                try:
                    with call(url + "/v1/chat/completions", body) as stream:
                        for line in stream:
                            now = time.monotonic_ns()
                            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                                continue
                            payload = json.loads(line.removeprefix(b"data:"))
                            chars = {
                                key: sum(len(c.get("delta", {}).get(key) or "") for c in payload.get("choices", []))
                                for key in ("content", "reasoning_content")
                            }
                            if any(chars.values()):
                                events.append(
                                    {"at_ns": str(now), **{key + "_chars": value for key, value in chars.items()}}
                                )
                            usage = payload.get("usage") or usage
                except Exception as exc:
                    error = type(exc).__name__
                return {
                    "rid": rid,
                    "role": "diagnostic_chat_frontend",
                    "start_ns": str(start),
                    "end_ns": str(time.monotonic_ns()),
                    "events": events,
                    "lanes": lanes,
                    "repeat": repeat,
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "requested_output_tokens": length,
                    "status": "failed"
                    if error
                    else "completed"
                    if usage.get("completion_tokens") == length
                    else "incomplete",
                    "error": error,
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=lanes) as pool:
                retain_case(pool.map(one, range(lanes)), records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    seed = tokenizer.encode(
        "Explain how a compiler schedules independent arithmetic instructions.\n", add_special_tokens=False
    )
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
                            rows = [dict(future.result(), lanes=lanes, repeat=repeat) for future in futures]
                            retain_case(rows, records)
                        print(f"repeat={repeat} context={context} output={length} lanes={lanes} completed", flush=True)
        overlap_control(args.url, seed, records)
        chat_controls(args.url, seed, tokenizer, records)


if __name__ == "__main__":
    main()
