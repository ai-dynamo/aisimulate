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
import os
from pathlib import Path


def inject_forward_nvtx(source):
    """Route one existing shaped forward span through our NVTX context."""
    if 'rsi_forward_nvtx' in source:
        raise ValueError("Forward NVTX context is already installed")
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "profile_range" and node.args
             and isinstance(node.args[0], ast.Call) and isinstance(node.args[0].func, ast.Name)
             and node.args[0].func.id == "build_step_span_name"]
    if len(calls) != 1:
        raise ValueError("Unknown shaped ModelRunner forward context")
    function = calls[0].func
    lines = source.splitlines(keepends=True)
    call = calls[0]
    # Apply the later edit first so a one-line span keeps its AST offsets.
    closing = call.end_lineno - 1
    batch = calls[0].args[0].args[0]
    if not isinstance(batch, ast.Name):
        raise ValueError("Unknown forward batch expression")
    lines[closing] = (lines[closing][:call.end_col_offset - 1] + f", runner=self, batch={batch.id}"
                      + lines[closing][call.end_col_offset - 1:])
    line = lines[function.lineno - 1]
    lines[function.lineno - 1] = (line[:function.col_offset]
                                + '__import__("rsi_forward_nvtx").span' + line[function.end_col_offset:])
    result = ("# Modified for AISimulate RSI diagnostic NVTX spans; see THIRD_PARTY_NOTICES.md.\n"
              + "".join(lines))
    ast.parse(result)
    return result


def inject_eagle_geometry(source):
    if "rsi_forward_nvtx" in source:
        raise ValueError("EAGLE geometry context is already installed")
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "execute"
             and isinstance(node.func.value, ast.Attribute)
             and node.func.value.attr in {"cuda_graph_runner", "cuda_graph_runner_for_draft_extend"}]
    if len(calls) != 2 or any(not node.args or not isinstance(node.args[0], ast.Name)
                              or node.args[0].id != "forward_batch" for node in calls):
        raise ValueError("Unknown EAGLE graph entry points")
    lines = source.splitlines(keepends=True)
    for call in sorted(calls, key=lambda node: (node.lineno, node.col_offset), reverse=True):
        stage = "draft" if call.func.value.attr == "cuda_graph_runner" else "draft_extend"
        end = call.end_lineno - 1
        lines[end] = lines[end][:call.end_col_offset - 1] + f', stage="{stage}"' + lines[end][call.end_col_offset - 1:]
        line = lines[call.func.lineno - 1]
        original = line[call.func.col_offset:call.func.end_col_offset]
        lines[call.func.lineno - 1] = (line[:call.func.col_offset]
                                     + '__import__("rsi_forward_nvtx").graph_execute(' + original + ", "
                                     + line[call.func.end_col_offset + 1:])
    result = "# Modified for AISimulate RSI numerical graph geometry; see THIRD_PARTY_NOTICES.md.\n" + "".join(lines)
    ast.parse(result)
    return result


def install_forward_nvtx():
    root = Path(importlib.util.find_spec("sglang").origin).parent
    relative = "srt/model_executor/model_runner.py"
    path = root / relative
    source = path.read_bytes()
    expected = "df38e26ace5deae708f2353b56962afab7f92fac35d1bd4f274b7d6d62356372"
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("ModelRunner differs from the qualified upstream/image source")
    modified = inject_forward_nvtx(source.decode()).encode()
    path.write_bytes(modified)
    receipt = {"relative_path": "sglang/" + relative,
                      "upstream_commit": "94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
                      "original_sha256": expected, "instrumented_sha256": hashlib.sha256(modified).hexdigest(),
                      "boundary": "existing forward label emitted to NVTX; GPU projection still required"}
    if os.environ.get("RSI_FORWARD_GEOMETRY") == "1":
        relative = "srt/speculative/eagle_worker_v2.py"
        path = root / relative
        original = path.read_bytes()
        expected = "8fc9285c458d746874c96da7b16bef2733509dea3d1170cbad9e787bcde67754"
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError("EAGLE graph source differs from the qualified runtime")
        modified = inject_eagle_geometry(original.decode()).encode()
        path.write_bytes(modified)
        receipt["geometry_patch"] = {"relative_path": "sglang/" + relative, "original_sha256": expected,
                                     "instrumented_sha256": hashlib.sha256(modified).hexdigest()}
    print(json.dumps(receipt))


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
    parser.add_argument("--forward-nvtx", action="store_true")
    args = parser.parse_args()
    if args.forward_nvtx:
        if args.dispatch_only or args.frontend:
            parser.error("Forward NVTX collection is separate from dispatch/frontend timing")
        install_forward_nvtx()
    elif args.dispatch_only:
        install_dispatch()
    else:
        main(args.frontend)
