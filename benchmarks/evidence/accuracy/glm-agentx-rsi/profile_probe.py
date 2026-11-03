#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite native-token controls and two explicit Nsight capture ranges."""

import argparse
import hashlib
import json
import struct
import threading
import time
from pathlib import Path

from probe import call, flush_cache, request
from transformers import AutoTokenizer

CONTEXTS = (8192, 393216)


def encoded(identity, prompt, output):
    return json.dumps({
        "rid": identity, "input_ids": prompt, "stream": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": output, "ignore_eos": True},
    }).encode()


def qualify(row, prompt_tokens, output_tokens, cached_tokens):
    if row["status"] != "completed" or not row["events"]:
        raise ValueError("Profile request did not complete")
    last = row["events"][-1]
    meta = last["meta"]
    actual_cache = meta.get("cached_tokens")
    if (last["output_tokens"] != output_tokens or meta.get("prompt_tokens") != prompt_tokens
            or type(actual_cache) is not int or not 0 <= actual_cache <= prompt_tokens
            or (cached_tokens is not None and actual_cache != cached_tokens)):
        raise ValueError("Profile input/output/cache counters do not match the declared case")
    if any(a["output_tokens"] > b["output_tokens"] for a, b in zip(row["events"], row["events"][1:], strict=False)):
        raise ValueError("Stream output counters decreased")
    return actual_cache


def profiler(url, start, output, identity):
    data = ({"output_dir": str(output), "activities": ["CUDA_PROFILER"],
             "profile_id": identity, "detailed_annotations": True,
             "record_shapes": False, "with_stack": False, "merge_profiles": False}
            if start else {})
    before = time.monotonic_ns()
    with call(url + ("/start_profile" if start else "/stop_profile"), data) as response:
        body = response.read()
        status = response.status
    expected = b"Start profiling.\n" if start else b"Stop profiling. This will take some time.\n"
    if status != 200 or body != expected:
        raise ValueError("Profiler API did not acknowledge the capture boundary")
    # Retain only this control endpoint's acknowledgement, never request text.
    return {"start": start, "begin_ns": str(before), "end_ns": str(time.monotonic_ns()),
            "http_status": status, "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    arms = ("before", "profiled", "after") if args.trace else ("before", "after")
    planned = 1 + len(arms) * 3
    manifest = {"schema": "aisimulate.glm-rsi.forward-profile.v1", "planned_requests": planned,
                "attempted_requests": 0, "qualified_requests": 0, "unattempted_requests": planned,
                "status": "running", "capture_ranges": [], "traced_server": args.trace}
    manifest_path = args.output / "manifest.json"

    def save():
        manifest["unattempted_requests"] = planned - manifest["attempted_requests"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    seed = tokenizer.encode("Explain how a compiler schedules independent arithmetic instructions.\n",
                            add_special_tokens=False)
    prompts = {n: (seed * (n // len(seed) + 1))[:n] for n in (*CONTEXTS, 512)}
    manifest["inputs"] = {str(n): {"tokens": n, "sha256_u32_le": hashlib.sha256(
        struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()} for n, tokens in prompts.items()}
    save()
    with (args.output / "requests.jsonl").open("x") as stream:
        def one(identity, prompt, length, cache, role, context, arm):
            body = encoded(identity, prompt, length)
            row = request(args.url, identity, prompt, length, threading.Barrier(1), encoded_body=body)
            row.update(role=role, context=context, arm=arm, expected_cached_tokens=cache)
            manifest["attempted_requests"] += 1
            try:
                observed_cache = qualify(row, len(prompt), length, cache)
                if context == 393216 and role == "profile":
                    if observed_cache < context - 64 or observed_cache % 64:
                        raise ValueError("Seeded long context did not retain its paged prefix")
                    manifest.setdefault("long_cached_tokens", observed_cache)
                row["qualified"] = True
                manifest["qualified_requests"] += 1
            except ValueError as exc:
                row.update(qualified=False, qualification_error=str(exc))
                raise
            finally:
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                save()

        try:
            flush_cache(args.url)
            one("rsi-profile-warmup", prompts[512], 16, 0, "preparation", 512, "warmup")
            for context in CONTEXTS:
                for arm in arms:
                    flush_cache(args.url)
                    prompt = prompts[context]
                    if context == 393216:
                        one(f"rsi-profile-seed-{arm}", prompt, 1, 0, "preparation", context, arm)
                    cache = manifest.get("long_cached_tokens") if context == 393216 else 0
                    identity = f"rsi-profile-c{context}-{arm}"
                    if arm == "profiled":
                        manifest["capture_ranges"].append(profiler(args.url, True, args.output, identity))
                        save()
                    try:
                        one(identity, prompt, 128, cache, "profile", context, arm)
                    finally:
                        if arm == "profiled":
                            manifest["capture_ranges"].append(profiler(args.url, False, args.output, identity))
                            save()
            manifest["status"] = "completed"
        except Exception as exc:
            manifest.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
