# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure SGLang FP8 activation quantization overhead for static-FP8 GEMM.

One measurement, two tables — the registry entry declares the second one as
``extra_perf_filenames`` and the executor hands both names in, so finalize
binds both to this producer's checkpoint:
  perf_filename            -> computescale_perf.txt  (dynamic per-token quant minus static quant)
  extra_perf_filenames[0]  -> scale_matrix_perf.txt  (the static per-tensor quant alone)

Both quantizations are the calls apply_fp8_linear makes in serving
(python/sglang/srt/layers/quantization/fp8_utils.py at the v0.5.23 tag):
dynamic per-token ``sglang_per_token_quant_fp8`` (:2625, the JIT
per_token_quant_fp8 kernel, kernels/ops/quantization/__init__.py:17-23, 52-60)
and static per-tensor ``static_quant_fp8`` (:2615-2621 and :2494, a Triton
kernel, kernels/ops/quantization/fp8_kernel.py:713-769). Each row's
kernel_source names the kernel it times (the compute_scale row by its dynamic
quant).
"""

# Source-checked against the sglang v0.5.23 tag. One release per file; earlier
# code is `git log -- <this file>` away.
__compat__ = "sglang==0.5.23"

from importlib.metadata import version as _dist_version  # setuptools/pkg_resources is absent from sglang>=0.5.21 images

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8, static_quant_fp8

_OUTSIDE_LOOP_COUNT = 5


def get_computescale_test_cases():
    return plan_cases("compute_scale", "sglang", get_sm_version())


def _setup(m, k, device):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    return device, x


def _bench_dynamic(device, x):
    def kernel_func():
        for _ in range(_OUTSIDE_LOOP_COUNT):
            sglang_per_token_quant_fp8(x)

    with benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1) as results:
        pass
    return results["latency_ms"] / _OUTSIDE_LOOP_COUNT, results["power_stats"]


def _bench_static(device, x):
    static_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    # apply_fp8_linear broadcasts the static scale to one per row only where the
    # per-channel CUTLASS epilogue cannot take a scalar A scale: native_scalar_a_scale
    # holds on SM90/SM100/SM120 (fp8_utils.py:2555-2557, 2615-2621), and the
    # FlashInfer per-tensor path never repeats it (:2494). On SM89 the static
    # quant therefore also writes an [M, 1] scale.
    repeat_scale = get_sm_version() < 90

    def kernel_func():
        for _ in range(_OUTSIDE_LOOP_COUNT):
            static_quant_fp8(x, static_scale, repeat_scale=repeat_scale)

    with benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1) as results:
        pass
    return results["latency_ms"] / _OUTSIDE_LOOP_COUNT, results["power_stats"]


def run_computescale(m, k, *, perf_filename, extra_perf_filenames, device="cuda:0"):
    (scale_matrix_filename,) = extra_perf_filenames
    device, x = _setup(m, k, device)
    dynamic_latency, dynamic_power = _bench_dynamic(device, x)
    static_latency, static_power = _bench_static(device, x)
    compute_scale_latency = max(0.0, dynamic_latency - static_latency)
    version = _dist_version("sglang")

    if not log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": compute_scale_latency}],
        framework="SGLang",
        version=version,
        device_name=torch.cuda.get_device_name(device),
        op_name="compute_scale",
        kernel_source="sglang_jit_per_token_quant_fp8",
        perf_filename=perf_filename,
        power_stats=dynamic_power,
    ):
        raise RuntimeError(f"Failed to persist SGLang compute scale performance row to {perf_filename}")

    if not log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": static_latency}],
        framework="SGLang",
        version=version,
        device_name=torch.cuda.get_device_name(device),
        op_name="scale_matrix",
        kernel_source="sglang_triton_static_quant_fp8",
        perf_filename=scale_matrix_filename,
        power_stats=static_power,
    ):
        raise RuntimeError(f"Failed to persist SGLang scale matrix performance row to {scale_matrix_filename}")
