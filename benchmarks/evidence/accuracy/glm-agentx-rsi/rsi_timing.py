# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical scheduler output-ready probe; no prompts or token IDs are written."""

import json
import os
import sys
import time

_last = {}
_stream = None


def record(requests):
    global _stream
    root = os.environ.get("RSI_TIMING_ROOT")
    if not root:
        return
    now = time.monotonic_ns()
    torch = sys.modules.get("torch")
    rank = torch.distributed.get_rank() if torch and torch.distributed.is_initialized() else None
    events = []
    for request in requests:
        rid = str(request.rid)
        if not rid.startswith("rsi-"):
            continue
        generated = len(request.output_ids)
        finished_len = request.finished_len
        count = generated if finished_len is None else min(generated, finished_len)
        state = (generated, count, finished_len)
        if _last.get(rid) == state:
            continue
        _last[rid] = state
        events.append(
            {
                "schema": "aisimulate.glm-rsi.output-ready.v2",
                "rid": rid, "at_ns": str(now), "output_tokens": count,
                "generated_output_tokens": generated, "finished_len": finished_len,
                "discarded_tail_tokens": generated - count,
                "batch_requests": len(requests), "rank": rank,
            }
        )
    if events:
        if _stream is None:
            # Keep one stream per process to avoid open/close overhead on every decode step.
            _stream = open(f"{root}/scheduler-{os.getpid()}.jsonl", "a", buffering=1)  # noqa: SIM115
        _stream.write("".join(json.dumps(event) + "\n" for event in events))
