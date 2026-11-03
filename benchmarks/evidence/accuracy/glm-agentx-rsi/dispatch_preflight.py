#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify the exact-image dispatch hook and separate actor without allocating GPUs."""

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path

from blocking import actor_worker
from instrument import install_dispatch

SOURCE_FILES = ("instrument.py", "rsi_dispatch.py", "blocking.py", "frontend.py", "probe.py")
EXPECTED_MANAGER = "952fdfcf54b13af6851993d4b7599d4b1efeb6aa630a7554584a1a047eda94f2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version("sglang") != "0.5.20":
        raise ValueError("This controlled experiment requires the exact historical runtime")
    receipt = install_dispatch()
    if receipt["original_sha256"] != EXPECTED_MANAGER:
        raise ValueError("Historical tokenizer-manager source differs from archived evidence")
    context = multiprocessing.get_context("spawn")
    jobs, results = context.Queue(), context.Queue()
    child = context.Process(target=actor_worker, args=("http://127.0.0.1:1", {}, jobs, results))
    child.start()
    try:
        ready = results.get(timeout=60)
        if not ready.get("ready") or ready["pid"] == os.getpid():
            raise ValueError("Actor is not an independent process")
        jobs.put(None)
        child.join(timeout=15)
        if child.exitcode != 0:
            raise ValueError("Actor did not shut down cleanly")
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
    receipt.update(
        status="passed",
        sglang=importlib.metadata.version("sglang"),
        actor_pid=ready["pid"],
        parent_pid=os.getpid(),
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES
        },
        scope="CPU source/actor qualification only; no GPU measurements or timing coefficients",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
