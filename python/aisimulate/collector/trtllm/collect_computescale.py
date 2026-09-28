# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure TensorRT-LLM FP8 compute-scale overhead.

The compute-scale collector compares dynamic FP8 quantization against static
quantization for each YAML-backed shape. The reported latency isolates the
scale-computation portion so support matrix data can track the cost separately
from the static quantize kernel.

One measurement, two tables — the registry entry declares the second one as
``extra_perf_filenames`` and the executor hands both names in, so finalize
binds both to this producer's checkpoint:
  perf_filename            -> computescale_perf.txt  (dynamic quant minus static quant)
  extra_perf_filenames[0]  -> scale_matrix_perf.txt  (the static per-tensor quant alone)
"""

__compat__ = "trtllm>=1.3.0rc20"

import tensorrt_llm
import torch
from collector.case_generator import get_compute_scale_case_specs

from collector.helper import benchmark_with_power, get_sm_version, log_perf

_OUTSIDE_LOOP_COUNT = 5  # to reduce impact of L2 cache hit


def get_computescale_test_cases():
    # compute_scale only applies to fp8 quantization (SM > 86)
    if get_sm_version() <= 86:
        return []

    test_cases = []
    for compute_scale_common_testcase in get_compute_scale_case_specs():
        test_cases.append([compute_scale_common_testcase.m, compute_scale_common_testcase.k])

    return test_cases


def _setup(m, k, device):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    x = torch.randn((m, k), dtype=torch.bfloat16).to(device)
    return device, x


def _bench_dynamic(device, x):
    # dynamic quantization = compute scale + scale matrix
    def kernel_func():
        for _ in range(_OUTSIDE_LOOP_COUNT):
            torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)

    with benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1) as results:
        pass
    return results["latency_ms"] / _OUTSIDE_LOOP_COUNT, results["power_stats"]


def _bench_static(device, x):
    # static quantization = scale matrix only
    scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    def kernel_func():
        for _ in range(_OUTSIDE_LOOP_COUNT):
            torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)

    with benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1) as results:
        pass
    return results["latency_ms"] / _OUTSIDE_LOOP_COUNT, results["power_stats"]


def run_computescale(m, k, *, perf_filename, extra_perf_filenames, device="cuda:0"):
    (scale_matrix_filename,) = extra_perf_filenames
    device, x = _setup(m, k, device)
    dynamic_latency, dynamic_power = _bench_dynamic(device, x)
    static_latency, static_power = _bench_static(device, x)
    compute_scale_latency = max(0.0, dynamic_latency - static_latency)

    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": compute_scale_latency}],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="compute_scale",
        kernel_source="torch_ops",
        perf_filename=perf_filename,
        power_stats=dynamic_power,
    )

    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": static_latency}],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="scale_matrix",
        kernel_source="torch_ops",
        perf_filename=scale_matrix_filename,
        power_stats=static_power,
    )
