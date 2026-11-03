# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional numerical timer of the known synchronous chat render/encode method."""

import array
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def token_digest(ids):
    values = array.array("I", ids)
    if sys.byteorder != "little":
        values.byteswap()
    if values.itemsize != 4:
        raise ValueError("Expected 32-bit unsigned token storage")
    return hashlib.sha256(values.tobytes()).hexdigest()


def measure_template(function):
    @functools.wraps(function)
    def measured(*args, **kwargs):
        root = os.environ.get("RSI_TIMING_ROOT")
        if not root:
            return function(*args, **kwargs)
        start = time.monotonic_ns()
        cpu_start = time.process_time_ns()
        thread_start = time.thread_time_ns()
        result = function(*args, **kwargs)
        thread_stop = time.thread_time_ns()
        cpu_stop = time.process_time_ns()
        stop = time.monotonic_ns()
        ids = result[1]
        row = {
            "start_ns": str(start),
            "end_ns": str(stop),
            "service_ms": (stop - start) / 1e6,
            "process_cpu_ms": (cpu_stop - cpu_start) / 1e6,
            "thread_cpu_ms": (thread_stop - thread_start) / 1e6,
            "input_tokens": len(ids),
            "input_sha256": token_digest(ids),
            "template_cache_requested": kwargs.get("use_cache"),
            "stage": "synchronous_chat_template_render_encode",
        }
        row["projection_ms"] = (time.monotonic_ns() - stop) / 1e6
        # No prompt, rendered text, token array or exception body is retained.
        with (Path(root) / f"frontend-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return result

    return measured
