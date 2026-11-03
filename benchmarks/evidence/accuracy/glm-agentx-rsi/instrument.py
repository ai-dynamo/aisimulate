#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install one optional output-ready hook in a writable diagnostic container."""

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def inject(source):
    tree = ast.parse(source)
    methods = [
        method
        for owner in tree.body
        if isinstance(owner, ast.ClassDef)
        for method in owner.body
        if isinstance(method, ast.FunctionDef) and method.name == "stream_output"
    ]
    if len(methods) != 1 or "rsi_timing" in source:
        raise ValueError("Unexpected or already instrumented scheduler output interface")
    method = methods[0]
    if not any(arg.arg == "reqs" for arg in method.args.args):
        raise ValueError("Unexpected scheduler request parameter")
    lines = source.splitlines(keepends=True)
    lines.insert(method.body[0].lineno - 1, '        __import__("rsi_timing").record(reqs)\n')
    modified = "".join(lines)
    ast.parse(modified)
    return modified


def main():
    package = importlib.metadata.distribution("sglang")
    specification = importlib.util.find_spec("sglang")
    if specification is None or specification.origin is None:
        raise ValueError("SGLang package origin is unavailable")
    root = Path(specification.origin).parent
    candidates = [
        "srt/managers/scheduler_components/output_streamer.py",
        "srt/managers/scheduler_output_processor_mixin.py",
    ]
    relative = next((name for name in candidates if (root / name).is_file()), None)
    if relative is None:
        raise ValueError(f"Unknown scheduler output layout under {root}")
    path = root / relative
    source = path.read_text()
    modified = inject(source)
    path.write_text(modified)
    print(
        json.dumps(
            {
                "sglang_version": package.version,
                "relative_path": "sglang/" + relative,
                "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "instrumented_sha256": hashlib.sha256(modified.encode()).hexdigest(),
                "boundary": "host scheduler output-ready, before detokenization/HTTP; not GPU kernel time",
            }
        )
    )


if __name__ == "__main__":
    main()
