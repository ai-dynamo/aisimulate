#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolate API CPU work from GPU work and client-side response parsing."""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import queue
import time
import urllib.request
from pathlib import Path

from frontend import measured
from probe import flush_cache, retain_case
from rsi_dispatch import receipt_path
from rsi_frontend import token_digest
from transformers import AutoTokenizer


def tokenize(url, rid, body, expected_count, expected_digest):
    request = urllib.request.Request(url + "/v1/tokenize", data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic_ns()
    row = {
        "rid": rid,
        "role": "cpu_tokenization",
        "start_ns": str(start),
        "pid": os.getpid(),
        "expected_input_tokens": expected_count,
        "expected_input_sha256": expected_digest,
    }
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        received = time.monotonic_ns()
        result = json.loads(raw)
        ids = result["tokens"]
        count = result["count"]
        digest = token_digest(ids)
        row.update(input_tokens=count, input_sha256=digest, body_received_ns=str(received), response_bytes=len(raw))
        if count != len(ids) or count != expected_count or digest != expected_digest:
            raise ValueError("Tokenization identity mismatch")
        row.update(status="completed", error=None)
    except Exception as exc:
        row.update(status="failed", error=type(exc).__name__)
    row["projection_end_ns"] = str(time.monotonic_ns())
    return row


def actor_worker(url, payloads, jobs, results):
    # Separate process: decoding the large token-array response cannot hold the
    # native stream reader's Python lock. Bodies are prepared before measurement.
    results.put({"ready": True, "pid": os.getpid()})
    while True:
        job = jobs.get()
        if job is None:
            return
        rid, kind = job
        body, count, digest = payloads[kind]
        results.put(tokenize(url, rid, body, count, digest))


def wait_dispatch(root, rid, timeout=15):
    deadline = time.monotonic() + timeout
    path = receipt_path(root, rid)
    while time.monotonic() < deadline:
        try:
            row = json.loads(path.read_text())
            if row["rid"] != rid:
                raise ValueError("Wrong dispatch identity")
            return row
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.001)
    raise TimeoutError("No scheduler-submission receipt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dispatch-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    seed = tokenizer.encode(
        "Explain how a compiler schedules independent arithmetic instructions.\n", add_special_tokens=False
    )
    inputs, payloads = {}, {}
    for size in (8192, 131072, 393216):
        text = tokenizer.decode((seed * (size // len(seed) + 1))[:size])
        messages = [{"role": "user", "content": text}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        inputs[size] = ids
        if size != 131072:
            kind = "short" if size == 8192 else "long"
            body = json.dumps({"model": "nvidia/GLM-5.3-NVFP4", "messages": messages}).encode()
            payloads[kind] = (body, len(ids), token_digest(ids))

    context = multiprocessing.get_context("spawn")
    jobs, results = context.Queue(), context.Queue()
    actor = context.Process(target=actor_worker, args=(args.url, payloads, jobs, results))
    actor.start()
    try:
        ready = results.get(timeout=60)
        if not ready.get("ready") or ready["pid"] == os.getpid():
            raise ValueError("Independent actor process failed to start")
        with (args.output / "requests.jsonl").open("x") as stream:
            for kind in ("short", "long"):
                jobs.put((f"rsi-block-preparation-{kind}", kind))
                warm = results.get(timeout=180)
                warm.update(phase="preparation", actor=kind)
                retain_case([warm], stream)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                for size in (8192, 131072):
                    for repeat in range(3):
                        order = ("none", "short", "long") if repeat % 2 == 0 else ("long", "short", "none")
                        for kind in order:
                            flush_cache(args.url)
                            rid = f"rsi-block-c{size}-r{repeat}-{kind}"
                            future = pool.submit(measured, args.url, rid, inputs[size], None, "native")
                            rows, dispatch_error, dispatch = [], None, None
                            try:
                                dispatch = wait_dispatch(args.dispatch_root, rid)
                                if kind != "none":
                                    jobs.put((rid + "-actor", kind))
                                    actor_row = results.get(timeout=180)
                                    actor_row.update(
                                        phase="profile", case=rid, actor=kind, repeat=repeat, context_label=size
                                    )
                                    rows.append(actor_row)
                            except (TimeoutError, ValueError, queue.Empty) as exc:
                                dispatch_error = type(exc).__name__
                            victim = future.result(timeout=600)
                            victim.update(
                                role="gpu_victim",
                                phase="profile",
                                case=rid,
                                actor=kind,
                                repeat=repeat,
                                context_label=size,
                                input_sha256=token_digest(inputs[size]),
                                pid=os.getpid(),
                                dispatch=dispatch,
                                dispatch_error=dispatch_error,
                            )
                            if dispatch_error or victim["cached_tokens"] != 0:
                                victim["status"] = "failed"
                            rows.insert(0, victim)
                            # Write every attempted request before failing a case.
                            retain_case(rows, stream)
                            print(rid + " completed", flush=True)
    finally:
        jobs.put(None)
        actor.join(timeout=15)
        if actor.is_alive():
            actor.terminate()
            actor.join(timeout=5)


if __name__ == "__main__":
    main()
