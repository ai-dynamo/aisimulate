#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny eight-device CUDA graph/NVTX probe before loading GLM weights."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool", choices=("nsys", "torch"), default="nsys")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("CUDA probe receipt already exists")
    if torch.cuda.device_count() != 8:
        raise ValueError("CUDA trace probe requires the reserved eight GPUs")
    names = [torch.cuda.get_device_name(i) for i in range(8)]
    if any("B300" not in name for name in names):
        raise ValueError("Unexpected GPU model")
    graphs, buffers, streams = [], [], []
    for device in range(8):
        with torch.cuda.device(device):
            a = torch.ones((32, 32), device=f"cuda:{device}")
            out = a @ a
            torch.cuda.synchronize()
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            torch.cuda.nvtx.range_push(f"rsi-smoke-capture-gpu-{device}")
            with torch.cuda.graph(graph, stream=capture_stream):
                torch.mm(a, a, out=out)
            torch.cuda.nvtx.range_pop()
            streams.append(capture_stream)
            graphs.append(graph)
            buffers.append((a, out))
    def replay():
        for device, graph in enumerate(graphs):
            with torch.cuda.device(device):
                name = f"rsi-smoke-replay-gpu-{device}"
                with torch.profiler.record_function(name):
                    torch.cuda.nvtx.range_push(name)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()

    if args.tool == "torch":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, with_stack=False) as profiler:
            replay()
        profiler.export_chrome_trace(str(args.output.with_name("torch-trace.json")))
    else:
        with torch.cuda.device(0):
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
        replay()
        with torch.cuda.device(0):
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
    # Export is outside the measured capture. Keep tensors alive through it.
    assert all(torch.all(out == 32).item() for _, out in buffers)
    args.output.write_text(json.dumps({
        "schema": "aisimulate.glm-rsi.cuda-trace-probe.v1", "devices": names,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "graphs_constructed_before_capture": True, "results_verified": True,
        "tool": args.tool,
        "qualification": "Application receipt only; actual GPU/NVTX report coverage must also pass.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
