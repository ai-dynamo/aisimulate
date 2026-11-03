#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired native/chat controls with equal tokenized inputs and seeded prefix cache."""

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

from probe import flush_cache, request, retain_case
from rsi_frontend import token_digest
from transformers import AutoTokenizer


def project(payload):
    """Keep all content-bearing boundaries, including tool dispatch, without text."""
    delta = (payload.get("choices") or [{}])[0].get("delta") or {}
    calls = delta.get("tool_calls") or []
    return {
        "content_chars": len(delta.get("content") or ""),
        "reasoning_chars": len(delta.get("reasoning_content") or delta.get("reasoning") or ""),
        "tool_chars": sum(
            len((call.get("function") or {}).get(key) or "") for call in calls for key in ("name", "arguments")
        ),
    }


def cache_receipt(usage, qualified_zero_omission=False):
    value = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if value is not None:
        return value, "explicit OpenAI usage cache count"
    if (
        qualified_zero_omission
        and usage.get("prompt_tokens_details") is None
        and type(usage.get("prompt_tokens")) is int
        and type(usage.get("completion_tokens")) is int
    ):
        return 0, "SGLang0.5.20 zero-omission protocol; requires resolved cache reporting enabled"
    return None, "cache usage unavailable"


def measured(url, rid, ids, messages, endpoint, *, cache_report_qualified=False):
    length = 16
    body = {"rid": rid, "stream": True}
    if endpoint == "native":
        body.update(input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": length, "ignore_eos": True})
        path = "/generate"
    else:
        body.update(
            model="nvidia/GLM-5.3-NVFP4",
            messages=messages,
            stream_options={"include_usage": True},
            temperature=0,
            max_tokens=length,
            ignore_eos=True,
        )
        path = "/v1/chat/completions"
    serialization = time.monotonic_ns()
    encoded = json.dumps(body).encode()
    serialization_ms = (time.monotonic_ns() - serialization) / 1e6
    wire = urllib.request.Request(url + path, data=encoded, headers={"Content-Type": "application/json"})
    events, usage, meta = [], {}, {}
    start = time.monotonic_ns()
    error = None
    try:
        with urllib.request.urlopen(wire, timeout=600) as stream:
            for line in stream:
                at = time.monotonic_ns()
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                payload = json.loads(line.removeprefix(b"data:"))
                if endpoint == "native":
                    meta = payload.get("meta_info") or meta
                    if meta.get("completion_tokens"):
                        events.append({"at_ns": str(at), "output_tokens": meta["completion_tokens"]})
                else:
                    chars = project(payload)
                    if any(chars.values()):
                        events.append({"at_ns": str(at), **chars})
                    usage = payload.get("usage") or usage
    except Exception as exc:
        error = type(exc).__name__
    end = time.monotonic_ns()
    counts = meta if endpoint == "native" else usage
    cached, cache_basis = (
        (counts.get("cached_tokens"), "native meta_info cache count")
        if endpoint == "native"
        else cache_receipt(counts, cache_report_qualified)
    )
    n = counts.get("completion_tokens")
    actual_input = counts.get("prompt_tokens")
    return {
        "rid": rid,
        "role": "paired_frontend",
        "endpoint": endpoint,
        "start_ns": str(start),
        "end_ns": str(end),
        "expected_input_tokens": len(ids),
        "input_tokens": actual_input,
        "output_tokens": n,
        "requested_output_tokens": length,
        "cached_tokens": cached,
        "cache_receipt_basis": cache_basis,
        "events": events,
        "serialization_ms": serialization_ms,
        "serialized_bytes": len(encoded),
        "error": error,
        "status": "completed" if not error and n == length and actual_input == len(ids) and events else "failed",
        "qualification": "requires equal token counts, prefix hashes and reported cache reuse across paired endpoints",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    seed = tokenizer.encode(
        "Explain how a compiler schedules independent arithmetic instructions.\n", add_special_tokens=False
    )
    with (args.output / "requests.jsonl").open("x") as stream:
        for size in (8192, 131072, 393216):
            text = tokenizer.decode((seed * (size // len(seed) + 1))[:size])
            messages = [{"role": "user", "content": text}]
            began = time.monotonic_ns()
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
            tokenize_ms = (time.monotonic_ns() - began) / 1e6
            prefix = (len(ids) // 64 - 1) * 64
            digest = token_digest(ids)
            for repeat in range(3):
                # Alternate order to expose drift; independently reset and seed each endpoint.
                endpoints = ("native", "chat") if repeat % 2 == 0 else ("chat", "native")
                for endpoint in endpoints:
                    identity = f"rsi-frontend-c{size}-r{repeat}-{endpoint}"
                    flush_cache(args.url)
                    warm = request(args.url, identity + "-warm", ids[:prefix], 1, threading.Barrier(1))
                    warm.update(role="preparation", phase="preparation")
                    retain_case([warm], stream)
                    row = measured(args.url, identity, ids, messages, endpoint)
                    row.update(
                        repeat=repeat,
                        context_label=size,
                        phase="profile",
                        prefix_tokens=prefix,
                        input_sha256=digest,
                        local_template_tokenize_ms=tokenize_ms,
                    )
                    retain_case([row], stream)
                    print(identity + " completed", flush=True)


if __name__ == "__main__":
    main()
