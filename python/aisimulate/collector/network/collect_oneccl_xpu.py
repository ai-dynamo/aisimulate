# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collect oneCCL communication performance data for XPU (Intel GPU).

This script uses the oneCCL benchmark binary (compiled from oneCCL examples)
to measure collective communication latencies on Intel XPU devices.
It produces nccl_perf.txt compatible output for use in projection models.

Prerequisites:
  - Per-op oneCCL benchmark binaries on PATH or /usr/local/bin (see README_oneccl_xpu.md)
  - Intel MPI (mpirun) available on PATH
  - Intel GPU (XPU) devices available

Usage:
  python collector/network/collect_oneccl_xpu.py --oneccl_op all_gather --dtype half --num_gpus 4
  python collector/network/collect_oneccl_xpu.py --oneccl_op reduce_scatter --dtype half --num_gpus 4
  python collector/network/collect_oneccl_xpu.py --oneccl_op all_reduce --dtype half --num_gpus 4
"""

import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import log_perf

OP_NAME_TO_BIN = {
    "all_gather": "allgather_perf",
    "reduce_scatter": "reduce_scatter_perf",
    "all_reduce": "allreduce_perf",
}

DTYPE_TO_CCL = {
    "half": "bf16",
    "int8": "int8",
}

BYTES_PER_ELEMENT = {
    "half": 2,
    "int8": 1,
}


def find_benchmark_binary(oneccl_op: str):
    """Locate the per-op oneCCL benchmark binary (PATH or /usr/local/bin)."""
    bin_name = OP_NAME_TO_BIN[oneccl_op]
    result = subprocess.run(["which", bin_name], capture_output=True, text=True)
    if result.returncode == 0:
        return bin_name
    fallback = f"/usr/local/bin/{bin_name}"
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"oneCCL benchmark binary '{bin_name}' not found. Build it and install to /usr/local/bin "
        "(see README_oneccl_xpu.md)."
    )


def get_oneccl_version():
    """Get installed oneCCL version string (pip wheel, else apt package)."""
    try:
        import importlib.metadata as im

        return im.version("oneccl")
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f", "${Version}", "intel-oneapi-ccl-devel"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown_version"


def get_device_name():
    """Intel GPU device name, matching the other XPU collectors (torch), else sycl-ls."""
    try:
        import torch

        return torch.xpu.get_device_name(0)
    except Exception:
        pass
    try:
        result = subprocess.run(["sycl-ls"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "level_zero:gpu" in line and "Intel" in line:
                parts = line.split(",")
                if len(parts) >= 2:
                    import re

                    match = re.search(r"(Intel\S*\s+Graphics\s+\[0x[0-9a-fA-F]+\])", parts[-1].strip())
                    if match:
                        return match.group(1)
    except Exception:
        pass
    return "Intel GPU"


def oneccl_benchmark(
    dtype: str,
    oneccl_op: str = "all_gather",
    test_range: str = "512,536870913,2",
    num_gpus: int = 2,
    iters: int = 100,
    warmup_iters: int = 20,
):
    """Run the per-op oneCCL benchmark via mpirun and log results."""
    benchmark_bin = find_benchmark_binary(oneccl_op)
    ccl_dtype = DTYPE_TO_CCL[dtype]
    bytes_per_elem = BYTES_PER_ELEMENT[dtype]
    version = get_oneccl_version()
    device_name = get_device_name()

    try:
        min_bytes, max_bytes, ratio = (int(i) for i in test_range.split(","))
    except ValueError as exc:
        raise ValueError("--range must be 'min_bytes,max_bytes,multiplicative_ratio' integers") from exc

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/venv/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK"] = "0"
    env["FI_PROVIDER"] = "tcp"
    env["I_MPI_OFI_PROVIDER"] = "tcp"

    cmd = [
        "mpirun",
        "-n",
        str(num_gpus),
        benchmark_bin,
        "-b",
        str(min_bytes),
        "-e",
        str(max_bytes),
        "-f",
        str(ratio),
        "-g",
        "1",
        "--datatype",
        ccl_dtype,
        "--iters",
        str(iters),
        "--warmup_iters",
        str(warmup_iters),
    ]

    print(
        f"Running oneCCL {oneccl_op}: dtype={dtype}({ccl_dtype}), num_gpus={num_gpus}, "
        f"{min_bytes}..{max_bytes}B x{ratio}"
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {oneccl_op} did not complete within 600s. Skipping.")
        return
    if result.returncode != 0:
        print(f"  Error (exit {result.returncode}): {result.stderr[:500]}")
        return

    items = []
    for line in result.stdout.split("\n"):
        toks = line.split()
        if len(toks) < 9 or not toks[0].isdigit():
            continue
        try:
            latency_ms = float(toks[5]) * 1e-3
            if oneccl_op == "all_gather":
                msg_elements = int(toks[1])
            else:
                msg_elements = int(toks[0]) // bytes_per_elem
        except (ValueError, IndexError):
            continue
        print(f"    {oneccl_op}: {msg_elements} elems, latency={latency_ms:.6f} ms")
        items.append(
            {
                "nccl_dtype": dtype,
                "num_gpus": num_gpus,
                "message_size": msg_elements,
                "latency": latency_ms,
            }
        )

    if items:
        log_perf(
            item_list=items,
            framework="VLLM",
            version=version,
            device_name=device_name,
            op_name=oneccl_op,
            kernel_source="oneCCL",
            perf_filename="oneccl_perf.txt",
        )
    print("Done. Results appended to oneccl_perf.txt")


if __name__ == "__main__":
    parser = ArgumentParser(description="Collect oneCCL communication performance data for XPU")
    parser.add_argument(
        "--oneccl_op",
        "-O",
        default="all_gather",
        choices=["all_gather", "reduce_scatter", "all_reduce"],
        help="oneCCL operation to benchmark",
    )
    parser.add_argument(
        "--dtype",
        "-t",
        default="half",
        choices=["half", "int8"],
        help="Data type for the collective operation",
    )
    parser.add_argument(
        "--range",
        "-r",
        default="512,536870913,2",  # 512B to 512MB, multiply by 2
        help="min_bytes,max_bytes,multiplicative_ratio",
    )
    parser.add_argument("--num_gpus", "-n", default=2, type=int, help="Number of GPUs (MPI ranks)")
    parser.add_argument("--iters", "-i", default=100, type=int, help="Benchmark iterations per size")
    parser.add_argument("--warmup_iters", "-w", default=20, type=int, help="Warmup iterations per size")
    args = parser.parse_args()

    oneccl_benchmark(
        dtype=args.dtype,
        oneccl_op=args.oneccl_op,
        test_range=args.range,
        num_gpus=args.num_gpus,
        iters=args.iters,
        warmup_iters=args.warmup_iters,
    )
