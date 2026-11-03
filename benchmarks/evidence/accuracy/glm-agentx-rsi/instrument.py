#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install one optional output-ready hook in a writable diagnostic container."""

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    package = importlib.metadata.distribution("sglang")
    path = Path(package.locate_file("sglang/srt/managers/scheduler_components/output_streamer.py"))
    source = path.read_text()
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SchedulerOutputStreamer")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "stream_output")
    if not any(arg.arg == "reqs" for arg in method.args.args) or "rsi_timing" in source:
        raise ValueError("Unexpected or already instrumented scheduler output interface")
    lines = source.splitlines(keepends=True)
    lines.insert(method.body[0].lineno - 1, '        __import__("rsi_timing").record(reqs)\n')
    modified = "".join(lines)
    ast.parse(modified)
    path.write_text(modified)
    print(json.dumps({"sglang_version": package.version, "relative_path": "sglang/srt/managers/scheduler_components/output_streamer.py",
                      "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "instrumented_sha256": hashlib.sha256(modified.encode()).hexdigest(),
                      "boundary": "host scheduler output-ready, before detokenization/HTTP; not GPU kernel time"}))


if __name__ == "__main__":
    main()
