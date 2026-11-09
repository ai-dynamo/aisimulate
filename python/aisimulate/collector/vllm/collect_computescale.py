# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure vLLM FP8 activation quantization overhead for static-FP8 GEMM.

One measurement, two tables — the registry entry declares the second one as
``extra_perf_filenames`` and the executor hands both names in, so finalize
binds both to this producer's checkpoint:
  perf_filename            -> computescale_perf.txt  (dynamic per-token quant minus static quant)
  extra_perf_filenames[0]  -> scale_matrix_perf.txt  (the static per-tensor quant alone)

Under torch.compile, vLLM serving folds the per-tensor FP8 activation quant
into Inductor kernels, but these tables time it as a standalone kernel, so
predictions bill a launch and an extra pass over the activation that serving
does not have (observed on l40s, h100 and h200:
benchmarks/evidence/collector/op-probe-2026-10/findings.yaml, entries
full_reprobe_framework_mode_2026_09_24 and sm89_vllm_0300_probe_2026_10_04).
The fold still holds: a per-tensor checkpoint leaves the quant_fp8 custom op
off (only block-quantized weights turn it on, config/vllm.py:1733-1748), so
QuantFP8 runs as a native op that Inductor compiles together with its
neighbours (model_executor/custom_op.py:172-192). Retire it by modelling the
fused quant (fold compute_scale into the gemm row when the deployment
compiles), or by dropping the standalone row for compiled deployments.
"""

# Audited against vLLM v0.32.0 (from v0.30.0): scaled_fp8_quant has the same
# signature and body (_custom_ops.py:1823-1891), GroupShape still lives in
# quant_utils (quantization/utils/quant_utils.py:116-140), and the two CUDA
# kernels it launches, dynamic_per_token_scaled_fp8_quant and
# static_scaled_fp8_quant (csrc/quantization/w8a8/fp8/common.cu:249,467), are
# unchanged in source: v0.31.0 rewrote only the per-tensor dynamic abs-max
# reduction, which this collector does not call, and v0.32.0 moved the files
# out of csrc/libtorch_stable with include-only edits into the same
# _C_stable_libtorch target, built with the same CUDA 13.0.3, torch 2.13.0 and
# arch list. Earlier audits are git history.
__compat__ = "vllm==0.32.0"

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from collector.vllm.utils import setup_distributed
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.version import __version__ as vllm_version

_OUTSIDE_LOOP_COUNT = 5


def get_computescale_test_cases():
    return plan_cases("compute_scale", "vllm", get_sm_version())


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


def run_computescale(m, k, *, perf_filename, extra_perf_filenames, device="cuda:0"):
    (scale_matrix_filename,) = extra_perf_filenames
    device, x = _setup(m, k, device)
    dynamic_latency, dynamic_power = _bench_dynamic(device, x)
    static_latency, static_power = _bench_static(device, x)
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

    log_perf(
        item_list=[{"m": m, "k": k, "quant_dtype": "fp8", "latency": static_latency}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="scale_matrix",
        kernel_source="static_scaled_fp8_quant",
        perf_filename=scale_matrix_filename,
        power_stats=static_power,
    )
