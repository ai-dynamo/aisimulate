# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure vLLM FP8 activation quantization overhead for static-FP8 GEMM.

Two registry ops share this module and its case grid, one table each
(the executor's finalize binds every staged table to exactly one checkpoint
producer — a single op writing two tables cannot be finalized):
  compute_scale -> computescale_perf.txt  (dynamic per-token quant minus static quant)
  scale_matrix  -> scale_matrix_perf.txt  (the static per-tensor quant alone)
"""

# B200 0.25.0 qualification (installed vLLM dd10e03f9), job 1968047:
# compute_scale: 8/8 representative cases. The native framework
# builders/selectors remain authoritative; no kernel fallback is introduced.
# The campaign manifest still selects one exact release per run.
__compat__ = "vllm>=0.24.0,<=0.30.0"

import torch
from collector.case_generator import get_compute_scale_case_specs
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.vllm.utils import setup_distributed
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.version import __version__ as vllm_version

_OUTSIDE_LOOP_COUNT = 5


def get_computescale_test_cases():
    if get_sm_version() <= 86:
        return []
    return [[case.m, case.k] for case in get_compute_scale_case_specs()]


def _setup(m, k, device):
    setup_distributed(device)
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    return device, x


def _bench_dynamic(device, x):
    def kernel_func():
        for _ in range(_OUTSIDE_LOOP_COUNT):
            ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)

    with benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1) as results:
        pass
    return results["latency_ms"] / _OUTSIDE_LOOP_COUNT, results["power_stats"]


def _bench_static(device, x):
    static_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    def kernel_func():
        for _ in range(_OUTSIDE_LOOP_COUNT):
            ops.scaled_fp8_quant(
                x,
                scale=static_scale,
                group_shape=(GroupShape.PER_TENSOR.row, GroupShape.PER_TENSOR.col),
            )

    with benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1) as results:
        pass
    return results["latency_ms"] / _OUTSIDE_LOOP_COUNT, results["power_stats"]


def run_computescale(m, k, *, perf_filename, device="cuda:0"):
    device, x = _setup(m, k, device)
    dynamic_latency, dynamic_power = _bench_dynamic(device, x)
    static_latency, _ = _bench_static(device, x)
    compute_scale_latency = max(0.0, dynamic_latency - static_latency)

    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": compute_scale_latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="compute_scale",
        kernel_source="dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant",
        perf_filename=perf_filename,
        power_stats=dynamic_power,
    )


def run_scale_matrix(m, k, *, perf_filename, device="cuda:0"):
    device, x = _setup(m, k, device)
    static_latency, static_power = _bench_static(device, x)

    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": static_latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="scale_matrix",
        kernel_source="static_scaled_fp8_quant",
        perf_filename=perf_filename,
        power_stats=static_power,
    )
