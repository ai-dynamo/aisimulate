#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inventory pinned runtime sources on a CPU allocation; no timing or schema changes."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

FILES = (
    "srt/layers/attention/dsa_backend.py",
    "srt/layers/attention/dsa/dsa_indexer.py",
    "srt/layers/attention/dsa/dsa_metadata_manager.py",
    "kernels/ops/attention/dsa_metadata.py",
    "kernels/ops/attention/dsa/paged_mqa_logits.py",
    "srt/model_executor/model_runner.py",
    "srt/model_executor/runner.py",
    "srt/model_executor/forward_batch_info.py",
    "srt/speculative/eagle_worker_v2.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Inventory output already exists")
    version = importlib.metadata.version("sglang")
    if version != args.expected_version:
        raise ValueError("Pinned runtime version differs")
    spec = importlib.util.find_spec("sglang")
    package = Path(next(iter(spec.submodule_search_locations)))
    result = {"schema": "aisimulate.glm-rsi.runtime-source-inventory.v1", "sglang": version,
              "torch": importlib.metadata.version("torch"), "sources": {},
              "scope": "Source inventory only; not GPU/kernel/module qualification or perf-data promotion."}
    for relative in FILES:
        path = package / relative
        if not path.is_file():
            result["sources"][relative] = {"status": "absent"}
            continue
        raw = path.read_bytes()
        tree = ast.parse(raw)
        result["sources"][relative] = {
            "status": "present", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "functions": [{"name": node.name, "arguments": [arg.arg for arg in node.args.args]}
                          for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and node.name in {"deepgemm_paged_mqa_logits_native", "deepgemm_paged_mqa_logits_split",
                                            "_get_topk_paged", "fused_dsa_target_verify_metadata"}],
        }
    revision = subprocess.run(["git", "-C", str(package.parents[1]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
    result["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    state = subprocess.run(["git", "-C", str(package.parents[1]), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=False)
    result["git_tracked_changes"] = state.stdout.splitlines() if state.returncode == 0 else None
    result["native_wrapper_interfaces_present"] = all(
        result["sources"][relative]["status"] == "present"
        for relative in ("srt/layers/attention/dsa/dsa_indexer.py",
                         "kernels/ops/attention/dsa_metadata.py", "kernels/ops/attention/dsa/paged_mqa_logits.py")
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
