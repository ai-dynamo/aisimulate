# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure SGLang FP8 activation quantization overhead for static-FP8 GEMM.

One measurement, two tables — the registry entry declares the second one as
``extra_perf_filenames`` and the executor hands both names in, so finalize
binds both to this producer's checkpoint:
  perf_filename            -> computescale_perf.txt  (dynamic per-token quant minus static quant)
  extra_perf_filenames[0]  -> scale_matrix_perf.txt  (the static per-tensor quant alone)
"""

__compat__ = "sglang==0.5.14"

import pkg_resources
import torch
from collector.case_generator import get_compute_scale_case_specs
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from sgl_kernel import sgl_per_token_quant_fp8

_OUTSIDE_LOOP_COUNT = 5


def get_computescale_test_cases():
    if get_sm_version() <= 86:
        return []
    return [[case.m, case.k] for case in get_compute_scale_case_specs()]


def _static_quantize_e4m3_per_tensor(x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    out.copy_((x / scale).clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn))
    return out


def _setup(m, k, device):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    return device, x


def _bench_dynamic(device, x):
    m = x.shape[0]
    dynamic_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    dynamic_scale = torch.empty((m, 1), dtype=torch.float32, device=device)

    def kernel_func():
        for _ in range(_OUTSIDE_LOOP_COUNT):
            sgl_per_token_quant_fp8(x, dynamic_out, dynamic_scale)

    with benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1) as results:
        pass
    return results["latency_ms"] / _OUTSIDE_LOOP_COUNT, results["power_stats"]


def _bench_static(device, x):
    static_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    static_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    def kernel_func():
        for _ in range(_OUTSIDE_LOOP_COUNT):
            _static_quantize_e4m3_per_tensor(x, static_scale, static_out)

    with benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1) as results:
        pass
    return results["latency_ms"] / _OUTSIDE_LOOP_COUNT, results["power_stats"]


def run_computescale(m, k, *, perf_filename, extra_perf_filenames, device="cuda:0"):
    (scale_matrix_filename,) = extra_perf_filenames
    device, x = _setup(m, k, device)
    dynamic_latency, dynamic_power = _bench_dynamic(device, x)
    static_latency, static_power = _bench_static(device, x)
    compute_scale_latency = max(0.0, dynamic_latency - static_latency)
    version = pkg_resources.get_distribution("sglang").version

    if not log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": compute_scale_latency}],
        framework="SGLang",
        version=version,
        device_name=torch.cuda.get_device_name(device),
        op_name="compute_scale",
        kernel_source="sglang",
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
        kernel_source="sglang",
        perf_filename=scale_matrix_filename,
        power_stats=static_power,
    ):
        raise RuntimeError(f"Failed to persist SGLang scale matrix performance row to {scale_matrix_filename}")
