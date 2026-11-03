#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cold native batches with prebuilt bodies and retained API submission clocks."""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

from probe import flush_cache, request, retain_case
from rsi_dispatch import receipt_path
from rsi_frontend import token_digest
from transformers import AutoTokenizer


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
    context, length, lanes = 131072, 128, 8
    with (args.output / "requests.jsonl").open("x") as stream:
        for repeat in range(3):
            flush_cache(args.url)
            prepared = []
            for lane in range(lanes):
                prefix = tokenizer.encode(f"Task {lane}: ", add_special_tokens=False)
                ids = (prefix + seed * (context // len(seed) + 1))[:context]
                rid = f"rsi-submit-r{repeat}-c{context}-o{length}-l{lanes}-i{lane}"
                body = json.dumps({
                    "rid": rid, "input_ids": ids, "stream": True,
                    "sampling_params": {"temperature": 0, "max_new_tokens": length, "ignore_eos": True},
                }).encode()
                prepared.append((rid, ids, body, time.monotonic_ns(), token_digest(ids)))
            barrier = threading.Barrier(lanes)
            with concurrent.futures.ThreadPoolExecutor(max_workers=lanes) as pool:
                futures = [pool.submit(request, args.url, rid, ids, length, barrier, encoded_body=body)
                           for rid, ids, body, _at, _digest in prepared]
                rows = [future.result() for future in futures]
            earliest = min(int(row["start_ns"]) for row in rows)
            for row, (rid, _ids, body, at, digest) in zip(rows, prepared, strict=True):
                row.update(
                    role="dispatch_matrix", phase="diagnostic", repeat=repeat, lanes=lanes,
                    context_label=context, prepared_at_ns=str(at), input_sha256=digest,
                    body_sha256=hashlib.sha256(body).hexdigest(), body_bytes=len(body),
                )
                try:
                    dispatch = json.loads(receipt_path(args.dispatch_root, rid).read_bytes())
                    row["dispatch"] = dispatch
                    if not (
                        dispatch["rid"] == rid and dispatch["input_tokens"] == context
                        and at <= earliest <= int(row["start_ns"]) <= int(dispatch["at_ns"])
                        <= int(row["events"][0]["at_ns"])
                        and row["events"][-1]["meta"].get("cached_tokens") == 0
                        and row["events"][-1]["meta"].get("prompt_tokens") == context
                    ):
                        row["status"] = "failed"
                        row["qualification_error"] = "submission, preparation or input/cache receipt differs"
                except (OSError, ValueError, KeyError, IndexError):
                    row["status"] = "failed"
                    row["qualification_error"] = "submission or client receipt unavailable"
            retain_case(rows, stream)
            print(f"dispatch matrix repeat={repeat} retained={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
