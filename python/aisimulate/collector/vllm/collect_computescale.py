# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure vLLM FP8 activation quantization overhead for static-FP8 GEMM."""

# B200 0.25.0 qualification (installed vLLM dd10e03f9), job 1968047:
# compute_scale: 8/8 representative cases. The native framework
# builders/selectors remain authoritative; no kernel fallback is introduced.
# The campaign manifest still selects one exact release per run.
# 0.30.0 audit (2026-09-30; source tag v0.30.0 == ced6857afa0e vs v0.27.1,
# runtime smoke on GB300 in the GLM-5.3-Flash W1 campaign). ops.
# scaled_fp8_quant's body is byte-identical (_custom_ops.py:1832-1898
# @0.27.1, :1834-1900 @0.30.0): scale=None + per-token ->
# _C.dynamic_per_token_scaled_fp8_quant, a given scale ->
# _C.static_scaled_fp8_quant; GroupShape.PER_TENSOR is unchanged
# (quant_utils.py:135-136). Serving's per-tensor-weight fp8 linear still uses
# a per-token dynamic activation key on SM89+ (fp8.py:289-297) through
# CutlassFP8ScaledMMLinearKernel -> QuantFP8(PER_TOKEN) -> this op when the
# quant_fp8 CustomOp is enabled (input_quant_fp8.py:126-135). Pre-existing
# caveat, unchanged since 0.27.1: under default VLLM_COMPILE+inductor,
# custom_ops="none" (config/vllm.py:1608-1615) runs the inductor-compiled
# forward_native instead of this CUDA kernel. Neither glm53tail overlay
# file is imported by _custom_ops/quant_utils. 0.25.1-0.29.0 are not audited.
__compat__ = "vllm>=0.24.0,<=0.30.0,!=0.25.1,!=0.26.0,!=0.27.0,!=0.27.1,!=0.28.0,!=0.29.0"

import torch
from collector.case_generator import get_compute_scale_case_specs
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.vllm.utils import setup_distributed
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.version import __version__ as vllm_version


def get_computescale_test_cases():
    if get_sm_version() <= 86:
        return []
    return [[case.m, case.k] for case in get_compute_scale_case_specs()]


def run_computescale(m, k, *, perf_filename, device="cuda:0"):
    setup_distributed(device)
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    static_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    outside_loop_count = 5

    def dynamic_kernel_func():
        for _ in range(outside_loop_count):
            ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)

    with benchmark_with_power(device=device, kernel_func=dynamic_kernel_func, repeat_n=1) as dynamic_results:
        pass

    dynamic_latency = dynamic_results["latency_ms"] / outside_loop_count

    def static_kernel_func():
        for _ in range(outside_loop_count):
            ops.scaled_fp8_quant(
                x,
                scale=static_scale,
                group_shape=(GroupShape.PER_TENSOR.row, GroupShape.PER_TENSOR.col),
            )

    with benchmark_with_power(device=device, kernel_func=static_kernel_func, repeat_n=1) as static_results:
        pass

    static_latency = static_results["latency_ms"] / outside_loop_count
    compute_scale_latency = max(0.0, dynamic_latency - static_latency)

    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": compute_scale_latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="compute_scale",
        kernel_source="dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant",
        perf_filename=perf_filename,
        power_stats=dynamic_results["power_stats"],
    )
    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": static_latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="scale_matrix",
        kernel_source="static_scaled_fp8_quant",
        perf_filename="scale_matrix_perf.txt",
        power_stats=static_results["power_stats"],
    )
