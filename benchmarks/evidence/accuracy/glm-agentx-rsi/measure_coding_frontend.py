#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the pinned synchronous component on private, qualified chat bodies."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

SOURCE_SHA = "047e4391dc87816a62da34392eea9379fcb1f93025e70c3b327de8b596f59b09"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpuinfo = Path("/proc/cpuinfo").read_text()
    cpu_model = next(line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name"))
    if "8559C" not in cpu_model or importlib.metadata.version("sglang") != "0.5.20":
        raise ValueError("Requires the qualified CPU model and historical image")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This experiment must not expose GPUs")
    if len(os.sched_getaffinity(0)) != 64:
        raise ValueError("Requires the recorded 64-CPU process affinity")
    for name, expected in json.loads(args.assets.read_text())["model_assets"].items():
        if Path(name).name != name or digest((args.model / name).read_bytes()) != expected:
            raise ValueError("Model/tokenizer asset mismatch")
    source = args.source.read_bytes()
    if digest(source) != SOURCE_SHA:
        raise ValueError("Serving source differs from the exact historical image")
    tree = ast.parse(source)
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "_render_and_encode_chat_template"]
    if len(methods) != 1 or methods[0].decorator_list:
        raise ValueError("Ambiguous or modified component")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), methods[0],
    ], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(args.source), "exec"), namespace)
    measure = namespace["_render_and_encode_chat_template"]
    receipt = json.loads((args.inputs / "receipt.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    encode_kwargs = {"add_special_tokens": False} if tokenizer.encode("") else {}
    owner = SimpleNamespace(tokenizer_manager=SimpleNamespace(tokenizer=tokenizer), _chat_template_cache=OrderedDict())
    cases = []
    for row in receipt["rows"]:
        name = row["private_artifact"]
        if Path(name).name != name:
            raise ValueError("Unsafe primitive artifact name")
        raw = (args.inputs / name).read_bytes()
        if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
            raise ValueError("Primitive checksum mismatch")
        payload = json.loads(raw)["payload"]
        if payload.get("reasoning_effort") or payload.get("chat_template_kwargs"):
            raise ValueError("Additional template settings require separate qualification")
        cases.append((row, payload))
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {
        key: os.environ.get(key) for key in ("TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS", "OMP_NUM_THREADS")
    }
    runtime = {
        "scope": "isolated synchronous render/encode only; not full HTTP, historical replay or headline scoring",
        "source_sha256": SOURCE_SHA, "inputs_receipt_sha256": digest((args.inputs / "receipt.json").read_bytes()),
        "measurement_script_sha256": digest(Path(__file__).read_bytes()),
        "cpu_model": cpu_model, "affinity": sorted(os.sched_getaffinity(0)), "load_average": os.getloadavg(),
        "thread_environment": environment, "encode_kwargs": encode_kwargs, "template_cache": False,
        "packages": {p: importlib.metadata.version(p) for p in ("sglang", "transformers", "tokenizers")},
        "repeats": 7, "prepared_cases": len(cases), "allocated_gpus": 0,
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    with (args.output / "measurements.jsonl").open("x") as stream:
        for repeat in range(7):
            for row, payload in (cases if repeat % 2 == 0 else list(reversed(cases))):
                start, cpu, thread = time.monotonic_ns(), time.process_time_ns(), time.thread_time_ns()
                result = measure(owner, payload["messages"], tools=payload.get("tools"),
                                 template_kwargs={}, encode_kwargs=encode_kwargs, use_cache=False)
                thread_end, cpu_end, end = time.thread_time_ns(), time.process_time_ns(), time.monotonic_ns()
                ids = result[1]
                matched = len(ids) == row["actual_chat_input_tokens"] and digest(
                    json.dumps(ids, separators=(",", ":")).encode()
                ) == row["token_sha256"]
                observation = {
                    "repeat": repeat, "case": row["private_artifact"], "input_tokens": len(ids),
                    "message_count": row["message_count"], "first_case_use": repeat == 0,
                    "start_ns": str(start), "end_ns": str(end), "service_ms": (end - start) / 1e6,
                    "process_cpu_ms": (cpu_end - cpu) / 1e6, "thread_cpu_ms": (thread_end - thread) / 1e6,
                    "identity_matches": matched, "status": "completed" if matched else "failed_identity",
                }
                stream.write(json.dumps(observation) + "\n")
                stream.flush()
                if not matched:
                    raise ValueError("Historical component changes primitive token identity")
    (args.output / "completion.json").write_text(json.dumps({"completed": 7 * len(cases)}) + "\n")


if __name__ == "__main__":
    main()
