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
import sys
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
    parser.add_argument("--sources-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Profiler qualification receipt already exists")
    version = importlib.metadata.version("sglang")
    if version != "0.5.20":
        raise ValueError("This diagnostic requires the historical runtime")
    root = Path(importlib.util.find_spec("sglang").origin).parent
    if args.sources_dir:
        args.sources_dir.mkdir(parents=True, exist_ok=False)
        (args.sources_dir / "export-context.json").write_text(json.dumps({
            "package_origin": str(root), "python": sys.version,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "status": "source export not yet qualified",
        }, indent=2) + "\n")
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
        trace_tool["profile_help_sha256"] = hashlib.sha256(help_result.stdout.encode()).hexdigest()
        if args.sources_dir:
            (args.sources_dir / "nsys-profile-help.txt").write_text(help_result.stdout)
    extra, licenses = [], []
    if args.sources_dir:
        required = (
            "srt/model_executor/model_runner.py", "srt/model_executor/step_span_utils.py",
            "srt/utils/profile_utils.py", "srt/model_executor/runner/decode_cuda_graph_runner.py",
            "srt/model_executor/model_runner_components/cuda_graph_setup.py",
            "srt/model_executor/forward_batch_info.py", "srt/utils/nvtx_utils.py",
        )
        optional = (
            "srt/models/glm4_moe.py", "srt/models/glm4_moe_nextn.py", "srt/models/glm4_moe_mtp.py",
            "srt/models/glm_moe_dsa.py", "srt/models/glm_moe_dsa_nextn.py",
            "srt/models/deepseek_nextn.py", "srt/speculative/eagle_worker.py",
            "srt/speculative/eagle_worker_v2.py",
        )
        for name in (*names, *required, *optional):
            path = root / name
            if not path.is_file() and name.endswith(".py"):
                package_init = path.with_suffix("") / "__init__.py"
                if package_init.is_file():
                    path = package_init
            if not path.is_file():
                if name in required or name in names:
                    raise ValueError(f"Required runtime trace source missing: {name}")
                extra.append({"relative_path": name, "available": False})
                continue
            data = path.read_bytes()
            if len(data) > 2_000_000:
                raise ValueError("Unexpected runtime source size")
            actual_name = str(path.relative_to(root))
            target = args.sources_dir / actual_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            extra.append({"relative_path": actual_name, "requested_relative_path": name, "available": True,
                          "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        distribution = importlib.metadata.distribution("sglang")
        for entry in distribution.files or ():
            if Path(entry).name.upper().startswith(("LICENSE", "NOTICE", "COPYING")):
                data = distribution.locate_file(entry).read_bytes()
                if len(data) > 1_000_000:
                    raise ValueError("Unexpected license size")
                name = hashlib.sha256(str(entry).encode()).hexdigest()[:12] + "-" + Path(entry).name
                target = args.sources_dir / "licenses" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                licenses.append({"original_path": str(entry), "retained_file": "licenses/" + name,
                                 "sha256": hashlib.sha256(data).hexdigest()})
    receipt = {
        "schema": "aisimulate.glm-rsi.profile-preflight.v1",
        "status": "passed_for_gpu_probe", "sglang": version,
        "python": sys.version,
        "package_origin": str(root),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")},
        "profile_request_fields": sorted(actual_fields),
        "runtime_sources": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
        "trace_tool": trace_tool,
        "retained_runtime_sources": extra,
        "retained_licenses": licenses,
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
