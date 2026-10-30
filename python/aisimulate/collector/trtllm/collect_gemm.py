# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM GEMM collector.

Expands shared GEMM case specs into TRT-LLM Linear-layer benchmarks across BF16,
FP8, FP8 block, and related quantization modes. Shape policy comes from
`case_generator.py`/YAML; this file owns TRT-LLM layer construction, padding,
quantized-weight setup, and perf logging.
"""

__compat__ = "trtllm>=1.3.0rc20"

import ctypes
import math
import os
from collections import defaultdict

import tensorrt_llm
import torch
import torch.nn.functional as F
from collector.case_generator import get_gemm_case_specs
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _compute_fp8_block_weight_scales(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Compute per-block scales from weight tensor (shape: [n, k])."""
    n, k = weight.shape
    n_blocks = math.ceil(n / group_size)
    k_blocks = math.ceil(k / group_size)
    pad_n = n_blocks * group_size - n
    pad_k = k_blocks * group_size - k

    weight_abs = weight.abs()
    if pad_n or pad_k:
        weight_abs = F.pad(weight_abs, (0, pad_k, 0, pad_n))

    block_max = weight_abs.view(n_blocks, group_size, k_blocks, group_size).amax(dim=(1, 3))
    return (block_max / 448.0).clamp_min(1e-6).to(dtype=torch.float32)


# Per-worker-process weight cache for large shapes (outside_loop_count == 1).
# Keyed by (gemm_type, n, k).  Only the most recently used entry is kept so
# that GPU memory is not accumulated across (n, k) groups.
# One worker builds the Linear modules (and their weights) once per
# (gemm_type, n, k) and reuses them for every m that follows on this worker:
# the modules do not depend on m, and the per-case rebuild was the shard's
# cost — nvfp4 unswizzles its block scales on the CPU and the sub-L2 shapes
# build up to 5 copies per case (b300 base run, job 471159002: gemm/nvfp4
# 1.3 GPU-s/row and gemm/fp8_block 1.7 GPU-s/row against 0.04 for bf16).
# get_gemm_test_cases orders the plan by (n, k) then gemm_type then m, so a
# worker sees runs of equal keys. The cache holds one key at a time, so peak
# memory is unchanged.
_gemm_op_cache: dict = {}


def _gemm_ops(gemm_type: str, n: int, k: int, *, quant_config, group_size, outside_loop_count: int, device, dtype, x):
    """The ``outside_loop_count`` Linear copies for (gemm_type, n, k), built on first use."""
    key = (gemm_type, n, k, outside_loop_count)
    op_list = _gemm_op_cache.get(key)
    if op_list is not None:
        return op_list
    _gemm_op_cache.clear()  # release the previous key's modules before building
    torch.cuda.empty_cache()
    op_list = []
    for _i in range(outside_loop_count):
        # nvfp4's per-tensor activation scale is a calibration constant in
        # serving; the first x of this key provides it, and the kernel's
        # timing does not depend on its value.
        weights = _build_weights(gemm_type, n, k, device, dtype, group_size, x)
        gemm = Linear(
            k,
            n,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            force_dynamic_quantization=False,
        )
        gemm.load_weights([weights])
        if gemm_type == "fp8_block" and callable(getattr(gemm, "post_load_weights", None)):
            gemm.post_load_weights()
        gemm.to(torch.device(device))
        op_list.append(gemm)
    _gemm_op_cache[key] = op_list
    return op_list


def _get_l2_cache_bytes(device_id: int = 0) -> int:
    """Query the GPU L2 cache size via the CUDA runtime API."""
    cuda_dev_attr_l2_cache_size = 38
    libcudart = ctypes.CDLL("libcudart.so")
    value = ctypes.c_int()
    ret = libcudart.cudaDeviceGetAttribute(ctypes.byref(value), cuda_dev_attr_l2_cache_size, device_id)
    if ret != 0 or value.value <= 0:
        raise RuntimeError(f"Failed to query L2 cache size (cudaError={ret}, value={value.value})")
    return value.value


def _build_weights(gemm_type: str, n: int, k: int, device, dtype, group_size, x) -> dict:
    """Allocate and return the weight tensors for one GEMM copy."""
    if gemm_type == "fp8":
        return {
            "weight": torch.randn((n, k), dtype=torch.bfloat16, device=device).to(dtype=torch.float8_e4m3fn),
            "weight_scale": torch.randn(1, dtype=torch.float32, device=device),
        }
    elif gemm_type == "fp8_block":
        fp_weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        return {
            "weight": fp_weight.to(dtype=torch.float8_e4m3fn),
            "weight_scale": _compute_fp8_block_weight_scales(fp_weight, group_size),
        }
    elif gemm_type == "nvfp4":
        w = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        w_sf_global = (448 * 6) / w.abs().max().float()
        w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global, 16, False)
        if tensorrt_llm.__version__.startswith("1.3.0"):
            w_sf_block_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(
                w_sf_block.cpu().view(pad_up(n, 128), -1)
            )
        else:
            w_sf_block_unswizzled = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(w_sf_block.cpu().view(k, -1))
        x_sf_global = (448 * 6) / x.abs().max().float()
        return {
            "weight": w_fp4.cpu(),
            "weight_scale": w_sf_block_unswizzled.view(torch.float8_e4m3fn),
            "weight_scale_2": 1.0 / w_sf_global.cpu(),
            "input_scale": 1.0 / x_sf_global.cpu(),
        }
    else:  # bfloat16
        return {"weight": torch.randn((n, k), dtype=torch.bfloat16, device=device)}


def get_gemm_test_cases():
    gemm_list = ["bfloat16"]
    sm_version = get_sm_version()
    if sm_version > 86:
        gemm_list += ["fp8"]
        # SM90 (Hopper) and SM100 (Blackwell) both support fp8_block.
        # SM90 uses CUTLASS backend with FP32 scale.
        # SM100 uses TRTLLM/DeepGEMM backend with UE8M0 scale (MXFP8 style).
        gemm_list += ["fp8_block"]
    if sm_version >= 100:
        gemm_list += ["nvfp4"]

    requested_gemm_types = os.environ.get("AIC_COLLECT_GEMM_TYPES")
    if requested_gemm_types:
        requested = {item.strip() for item in requested_gemm_types.split(",") if item.strip()}
        gemm_list = [gemm_type for gemm_type in gemm_list if gemm_type in requested]

    # Group (x, n, k) triples by (n, k) so that all batch sizes for the same
    # weight shape run consecutively.  This allows the per-process weight cache
    # in run_gemm to amortise the expensive torch.randn weight allocation across
    # all 75 x values instead of reallocating on every call.
    all_cases = get_gemm_case_specs()
    nk_to_x: dict = defaultdict(list)
    for c in all_cases:
        nk_to_x[(c.n, c.k)].append(c.x)

    # Sort: largest (n, k) first so that the most expensive shapes are processed
    # early (same behaviour as before for progress visibility).
    nk_pairs_sorted = sorted(nk_to_x.keys(), key=lambda nk: (-nk[0], -nk[1]))

    test_cases = []
    for n, k in nk_pairs_sorted:
        x_list = sorted(nk_to_x[(n, k)], reverse=True)  # largest batch size first
        for gemm_type in gemm_list:
            if (gemm_type == "nvfp4" or gemm_type == "fp8_block") and (n < 128 or k < 128):
                continue
            if gemm_type == "fp8_block" and n * k >= 2**31:
                continue
            for x in x_list:
                if x * n >= 2**31 or x * k >= 2**31:
                    continue
                # FIXME(kernel-limit): the fp8 small-M giant-projection IMA
                # family (m 1..8, n and k >= 51200) known from rc5/rc10 SM120
                # and rc15 SM89 reproduces on rc20 SM120 (5 recorded IMAs in
                # the 2026-07-25 RTX PRO 6000 full sweep at m5-8,
                # n/k 51200-65536). Cases run and fail classified; the
                # version-scoped skip helper was deleted when this collector
                # pinned trtllm>=1.3.0rc20. Re-verify on the next bump.
                test_cases.append([gemm_type, x, n, k])

    return test_cases


def run_gemm(gemm_type, m, n, k, *, perf_filename, device="cuda:0"):
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    sm_version = get_sm_version()

    if gemm_type == "fp8" and (n % 16 or k % 16):
        # FP8QDQLinearMethod.apply -> torch.ops.trtllm.cublas_scaled_mm / cuda_scaled_mm,
        # which TORCH_CHECK the 16-element alignment of the output row stride (= n)
        # and of mat_b's column stride (= k): cpp/tensorrt_llm/thop/cublasScaledMM.cpp:260
        # and cutlassScaledMM.cpp:152 @1.3.0rc29. The shared GEMM grid carries
        # n in {1, 4, 8, 12, 24} (1,110 cases per system on gb200/gb300).
        raise RuntimeError(
            "FIXME(kernel-limit): TRT-LLM fp8 GEMM (cublas/cutlass scaled_mm) requires n and k to be "
            f"multiples of 16 (out.strides()[0] % 16 == 0 && mat_b.strides()[1] % 16 == 0), got n={n}, k={k}"
        )

    dtype = torch.bfloat16
    x = torch.randn((m, k), dtype=dtype).to(torch.device(device))

    if gemm_type == "fp8":
        group_size = None
        qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    elif gemm_type == "fp8_block":
        group_size = 128
        qc = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size)
    elif gemm_type == "nvfp4":
        # NVFP4 scale blocks are 16 elements (the weights above are quantized with
        # fp4_quantize(..., 16, ...)); 1.3.0rc29 NVFP4LinearMethod.resolve_scaling_vector_size
        # rejects any other declared group_size ("supports NVFP4 scale blocks of (16,) elements,
        # but the checkpoint declares group_size=128" — first seen on B200 2026-10-04; on Hopper
        # the nvfp4 lane is a platform floor and never constructed the module).
        group_size = 16
        qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=group_size)
    else:
        qc = None
        group_size = None

    _l2_cache_bytes = _get_l2_cache_bytes(device.index or 0)
    _bytes_per_elem = {"bfloat16": 2, "fp8": 1, "fp8_block": 1, "nvfp4": 0.5}
    _weight_bytes = int(n * k * _bytes_per_elem[gemm_type])
    outside_loop_count = max(1, min(5, math.ceil(_l2_cache_bytes / _weight_bytes)))
    op_list = _gemm_ops(
        gemm_type,
        n,
        k,
        quant_config=qc,
        group_size=group_size,
        outside_loop_count=outside_loop_count,
        device=device,
        dtype=dtype,
        x=x,
    )
    for op in op_list:
        op.forward(x)  # dry run at this m (lazy init / JIT) before graph capture

    def kernel_func():
        for op in op_list:
            op.forward(x)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        repeat_n=1,  # Already repeating inside kernel_func via op_list
    ) as results:
        pass

    # fp8_block Linear dispatch is is_sm_100f-scoped, not sm>=100: SM100/103
    # take the DeepGEMM fp8_swap_ab_gemm path, while SM120 has its own branch
    # (per_token_quant_and_transform + torch.ops.trtllm.fp8_block_scaling_gemm)
    # and Hopper the fp8_quantize_1x128 + fp8_block_scaling_gemm pair, both
    # labeled torch_flow (modules/linear.py:1111-1133@1.3.0rc20). The previous
    # sm>=100 predicate mislabeled SM120 rows as deepgemm.
    kernel_source = "deepgemm" if gemm_type == "fp8_block" and is_sm_100f(sm_version) else "torch_flow"

    log_perf(
        item_list=[
            {"gemm_dtype": gemm_type, "m": m, "n": n, "k": k, "latency": results["latency_ms"] / outside_loop_count}
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name="gemm",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_gemm_test_cases():
        run_gemm(*test_case, perf_filename=PerfFile.GEMM)
