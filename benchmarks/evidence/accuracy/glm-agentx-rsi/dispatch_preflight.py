#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify the exact-image dispatch hook and separate actor without allocating GPUs."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import probe
from blocking import actor_worker, load_coding_inputs
from instrument import install_dispatch
from rsi_dispatch import PREFLIGHT_SOURCE_FILES, receipt_path, record
from transformers import AutoTokenizer

EXPECTED_MANAGER = "952fdfcf54b13af6851993d4b7599d4b1efeb6aa630a7554584a1a047eda94f2"


def prepared_request_check():
    body = json.dumps({"rid": "rsi-preflight", "input_ids": [1, 2, 3, 4]}).encode()
    wire = b'data: {"meta_info":{"completion_tokens":1,"prompt_tokens":4,"cached_tokens":0}}\n\n'
    wire += b'data: {"meta_info":{"completion_tokens":3,"prompt_tokens":4,"cached_tokens":0}}\n\n'

    def response(request, timeout):
        if request.data != body or timeout != 600 or request.full_url != "http://127.0.0.1:1/generate":
            raise ValueError("Prepared transport changed the encoded body")
        return io.BytesIO(wire)

    with patch("probe.urllib.request.urlopen", response), patch(
        "probe.json.dumps", side_effect=AssertionError("Serialization occurred inside request measurement")
    ):
        row = probe.request("http://127.0.0.1:1", "rsi-preflight", [1, 2, 3, 4],
                            3, threading.Barrier(1), encoded_body=body)
    if row["status"] != "completed" or row["events"][-1]["output_tokens"] != 3:
        raise ValueError("Prepared transport projection failed")
    return {"status": "passed", "mock_transport": True, "body_sha256": hashlib.sha256(body).hexdigest()}


def dispatch_name_check(root):
    with (
        TemporaryDirectory(prefix="dispatch-preflight-", dir=root) as temporary,
        patch.dict(os.environ, {"RSI_DISPATCH_ROOT": temporary}),
    ):
        for rid in ("rsi-block-preflight", "rsi-submit-preflight"):
            record(SimpleNamespace(rid=rid, input_ids=[1, 2, 3, 4]))
            result = json.loads(receipt_path(temporary, rid).read_bytes())
            if result["rid"] != rid or result["input_tokens"] != 4 or result["pid"] != os.getpid():
                raise ValueError("Dispatch names or scalar input identity changed")
    return {"status": "passed", "tested_prefixes": ["rsi-block-", "rsi-submit-"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coding-inputs", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if importlib.metadata.version("sglang") != "0.5.20":
        raise ValueError("This controlled experiment requires the exact historical runtime")
    receipt = install_dispatch()
    if receipt["original_sha256"] != EXPECTED_MANAGER:
        raise ValueError("Historical tokenizer-manager source differs from archived evidence")
    receipt["prepared_request"] = prepared_request_check()
    receipt["dispatch_names"] = dispatch_name_check(args.output.parent)
    if args.coding_inputs:
        if args.model is None:
            raise ValueError("Coding qualification requires the actual model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
        _, receipt["coding_inputs"] = load_coding_inputs(args.coding_inputs, tokenizer)
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
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in PREFLIGHT_SOURCE_FILES
        },
        scope="CPU source/actor qualification only; no GPU measurements or timing coefficients",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
