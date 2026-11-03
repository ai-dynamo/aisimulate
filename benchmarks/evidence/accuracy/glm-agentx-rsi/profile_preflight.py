#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify the pinned runtime's profiler protocol on CPUs before a GPU probe."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

REQUIRED_FIELDS = {
    "output_dir", "start_step", "num_steps", "activities", "with_stack",
    "record_shapes", "profile_by_stage", "profile_id", "merge_profiles",
    "profile_prefix", "detailed_annotations",
}


def fields(source, name):
    matches = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown runtime class: {name}")
    return {node.target.id for node in matches[0].body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Profiler qualification receipt already exists")
    version = importlib.metadata.version("sglang")
    if version != "0.5.20":
        raise ValueError("This diagnostic requires the historical runtime")
    root = Path(importlib.util.find_spec("sglang").origin).parent
    names = (
        "srt/managers/io_struct.py", "srt/managers/scheduler_components/profiler_manager.py",
        "srt/entrypoints/http_server.py",
    )
    sources = {name: (root / name).read_text() for name in names}
    actual_fields = fields(sources[names[0]], "ProfileReq")
    if not REQUIRED_FIELDS.issubset(actual_fields):
        raise ValueError("Pinned profiler request fields are unavailable")
    manager = sources[names[1]]
    if not all(term in manager for term in ("export_chrome_trace", "CUDA_PROFILER", "record_shapes")):
        raise ValueError("Runtime trace/export contract is unavailable")
    http = sources[names[2]]
    if not all(term in http for term in ('"/start_profile"', '"/stop_profile"')):
        raise ValueError("Profiler HTTP endpoints are unavailable")
    nsys = shutil.which("nsys")
    if not nsys:
        candidates = sorted(
            list(Path("/opt/nvidia").glob("nsight-systems*/bin/nsys"))
            + list(Path("/opt/nvidia").glob("nsight-systems*/*/nsys"))
            + list(Path("/usr/local").glob("cuda*/bin/nsys"))
        )
        nsys = str(candidates[0]) if candidates else None
    trace_tool = {"nsys": nsys, "version": None, "cuda_graph_node_option": False}
    if nsys:
        result = subprocess.run([nsys, "--version"], capture_output=True, text=True, timeout=30, check=True)
        trace_tool["version"] = result.stdout.strip()
        help_result = subprocess.run(
            [nsys, "profile", "--help"], capture_output=True, text=True, timeout=30, check=True
        )
        trace_tool["cuda_graph_node_option"] = "--cuda-graph-trace" in help_result.stdout
    receipt = {
        "schema": "aisimulate.glm-rsi.profile-preflight.v1",
        "status": "passed_for_gpu_probe", "sglang": version,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")},
        "profile_request_fields": sorted(actual_fields),
        "runtime_sources": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
        "trace_tool": trace_tool,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "CPU source/protocol qualification only; CUPTI and graph-node visibility require the GPU probe.",
            "Per-op attribution requires GPU-correlated windows and actual shapes on every required TP rank.",
            "CPU launch ranges and aggregate kernel sums are not forward critical-path measurements.",
        ],
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "trace_tool": trace_tool}))


if __name__ == "__main__":
    main()
