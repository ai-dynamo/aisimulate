# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal scheduler-submission receipts for the isolated frontend control."""

import hashlib
import json
import os
import time
from pathlib import Path

PREFLIGHT_SOURCE_FILES = (
    "collect.py", "instrument.py", "rsi_dispatch.py", "blocking.py", "frontend.py", "probe.py",
    "rsi_timing.py", "rsi_frontend.py", "matrix_dispatch.py", "dispatch_preflight.py",
)


def receipt_path(root, rid):
    return Path(root) / (hashlib.sha256(rid.encode()).hexdigest() + ".json")


def record(request):
    root = os.environ.get("RSI_DISPATCH_ROOT")
    rid = request.rid
    if not root or not isinstance(rid, str) or not rid.startswith(("rsi-block-", "rsi-submit-")):
        return
    row = {
        "rid": rid,
        "at_ns": str(time.monotonic_ns()),
        "input_tokens": len(request.input_ids),
        "pid": os.getpid(),
        "boundary": "API scheduler submission returned; not GPU start",
    }
    receipt_path(root, rid).write_text(json.dumps(row) + "\n")
