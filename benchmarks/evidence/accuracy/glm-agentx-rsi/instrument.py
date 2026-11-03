#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install one optional output-ready hook in a writable diagnostic container."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def inject_dispatch(source):
    tree = ast.parse(source)
    methods = [
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_send_one_request"
    ]
    if len(methods) != 1 or "rsi_dispatch" in source:
        raise ValueError("Unexpected or already instrumented scheduler submission interface")
    calls = [
        node
        for node in ast.walk(methods[0])
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_dispatch_to_scheduler"
    ]
    if (
        len(calls) != 1
        or len(calls[0].value.args) != 1
        or not isinstance(calls[0].value.args[0], ast.Name)
        or calls[0].value.args[0].id != "tokenized_obj"
    ):
        raise ValueError("Unknown scheduler submission boundary")
    call = calls[0]
    lines = source.splitlines(keepends=True)
    lines.insert(call.end_lineno, " " * call.col_offset + '__import__("rsi_dispatch").record(tokenized_obj)\n')
    modified = "".join(lines)
    ast.parse(modified)
    return modified


def install_dispatch():
    specification = importlib.util.find_spec("sglang")
    if specification is None or specification.origin is None:
        raise ValueError("SGLang package origin is unavailable")
    path = Path(specification.origin).parent / "srt/managers/tokenizer_manager.py"
    source = path.read_text()
    modified = inject_dispatch(source)
    path.write_text(modified)
    receipt = {
        "relative_path": "sglang/srt/managers/tokenizer_manager.py",
        "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "instrumented_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "boundary": "API scheduler submission returned; not GPU start",
    }
    print(json.dumps(receipt))
    return receipt


def inject_frontend(source):
    tree = ast.parse(source)
    methods = [
        method
        for owner in tree.body
        if isinstance(owner, ast.ClassDef)
        for method in owner.body
        if isinstance(method, ast.FunctionDef) and method.name == "_render_and_encode_chat_template"
    ]
    if len(methods) != 1 or "rsi_frontend" in source:
        raise ValueError("Unexpected or already instrumented chat-template interface")
    method = methods[0]
    if method.decorator_list:
        raise ValueError("Chat-template method unexpectedly has decorators")
    lines = source.splitlines(keepends=True)
    lines.insert(method.lineno - 1, '    @__import__("rsi_frontend").measure_template\n')
    result = "".join(lines)
    ast.parse(result)
    return result


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


def main(frontend=False):
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
    frontend_receipt = None
    if frontend:
        chat = root / "srt/entrypoints/openai/serving_chat.py"
        original = chat.read_text()
        updated = inject_frontend(original)
        chat.write_text(updated)
        frontend_receipt = {
            "relative_path": "sglang/srt/entrypoints/openai/serving_chat.py",
            "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "instrumented_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "boundary": "synchronous chat template render and encode; not the whole HTTP frontend",
        }
    print(
        json.dumps(
            {
                "sglang_version": package.version,
                "relative_path": "sglang/" + relative,
                "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "instrumented_sha256": hashlib.sha256(modified.encode()).hexdigest(),
                "boundary": "host scheduler output-ready, before detokenization/HTTP; not GPU kernel time",
                "frontend": frontend_receipt,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", action="store_true")
    parser.add_argument("--dispatch-only", action="store_true")
    args = parser.parse_args()
    if args.dispatch_only:
        install_dispatch()
    else:
        main(args.frontend)
