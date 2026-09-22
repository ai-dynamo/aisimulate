# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared quantization identities without performance-model runtime imports."""

from collections import namedtuple
from enum import Enum

# compute_dtype names the tensor-core pipeline the mode's MMA actually executes
# on ("bfloat16" | "int8" | "fp8" | "fp4"), i.e. which <dtype>_tc_flops entry of
# the system YAML governs its SOL math. Weight-only modes (int8_wo, int4_wo,
# w4a16_*) dequantize to bf16 before the MMA, so they map to "bfloat16". None
# marks memory-only modes (KV cache, comm) that never query FLOPS.
QuantMapping = namedtuple("QuantMapping", ["memory", "compute", "name", "compute_dtype"])


class GEMMQuantMode(Enum):
    """
    GEMM quant mode.
    """

    bfloat16 = QuantMapping(2, 1, "bfloat16", "bfloat16")  # w16a16
    int8_wo = QuantMapping(1, 1, "int8_wo", "bfloat16")  # w8a16
    int4_wo = QuantMapping(0.5, 1, "int4_wo", "bfloat16")  # w4a16
    fp8 = QuantMapping(1, 2, "fp8", "fp8")  # w8fp8
    fp8_static = QuantMapping(
        1, 2, "fp8_static", "fp8"
    )  # fp8 with static quantization (compute_scale/scale_matrix modeled)
    sq = QuantMapping(1, 2, "sq", "int8")  # w8int8
    fp8_block = QuantMapping(1, 2, "fp8_block", "fp8")  # specific for trtllm torch ds fp8
    fp8_ootb = QuantMapping(
        1, 2, "fp8_ootb", "fp8"
    )  # in future, should deprecate this mode as it's specific for trtllm trt backend
    nvfp4 = QuantMapping(9 / 16, 4, "nvfp4", "fp4")  # nvfp4 on blackwell. 1 fp8 scale per 16 nvfp4 weights.
    nvfp4_wo = QuantMapping(9 / 16, 1, "nvfp4_wo", "bfloat16")  # nvfp4 sw dequant to bf16 (non-Blackwell)
    w4a16_nvfp4 = QuantMapping(
        9 / 16, 1, "w4a16_nvfp4", "bfloat16"
    )  # NVFP4 weights + 1 fp8 scale per 16 weights, dequantized into the bf16 MMA lane.


class MoEQuantMode(Enum):
    """
    MoE quant mode.
    """

    bfloat16 = QuantMapping(2, 1, "bfloat16", "bfloat16")  # w16a16
    fp8 = QuantMapping(1, 2, "fp8", "fp8")  # w8fp8
    int4_wo = QuantMapping(0.5, 1, "int4_wo", "bfloat16")  # w4a16
    fp8_block = QuantMapping(1, 2, "fp8_block", "fp8")  # specific for trtllm torch ds fp8
    w4afp8 = QuantMapping(0.5, 2, "w4afp8", "fp8")  # specific for trtllm torch ds w4a8
    nvfp4 = QuantMapping(9 / 16, 4, "nvfp4", "fp4")  # nvfp4 on blackwell. 1 fp8 scale per 16 nvfp4 weights.
    nvfp4_wo = QuantMapping(9 / 16, 1, "nvfp4_wo", "bfloat16")  # nvfp4 sw dequant to bf16 (non-Blackwell)
    w4a16_mxfp4 = QuantMapping(0.5, 1, "w4a16_mxfp4", "bfloat16")  # native data format for gpt oss
    w4a8_mxfp4_mxfp8 = QuantMapping(0.5, 2, "w4a8_mxfp4_mxfp8", "fp8")
    # mxfp4 weights, mxfp8 activations (recommended for Blackwell)
    w4a8_mxfp4_mxfp8_trtllm = QuantMapping(0.5, 2, "w4a8_mxfp4_mxfp8_trtllm", "fp8")
    # Blackwell trtllm-gen fused MoE: MXFP4 (E2M1, block-32) weights x MXFP8 (E4M3)
    # activations -- the kernel DeepSeek-V4-Pro actually runs in prefill on sm100
    # (bmm_MxE4m3_MxE2m1MxE4m3 ... sm100f, flashinfer trtllm_fp4_block_scale_moe).
    # Distinct backend from w4a8_mxfp4_mxfp8 above (flashinfer cutedsl). DSV4 MoE
    # weights are stored MXFP4 (I8-packed E2M1 + E8M0 scales), so sglang dispatches
    # by GPU: sm100 -> this (trtllm-gen); sm90 -> w4a16_mxfp4_cutlass below.
    w4a16_mxfp4_cutlass = QuantMapping(0.5, 1, "w4a16_mxfp4_cutlass", "bfloat16")
    # Hopper (sm90) DeepSeek-V4-Pro MoE: flashinfer cutlass SM90 mixed GEMM
    # (cutlass_fused_moe(use_w4_group_scaling=True)) -- MXFP4 weights x BF16
    # activations (weight-only). Distinct backend from w4a16_mxfp4 above, which is
    # GPT-OSS's triton_kernels mxfp4 path. (DSV4 Hopper silicon data pending.)
    w4a16_nvfp4 = QuantMapping(9 / 16, 1, "w4a16_nvfp4", "bfloat16")
    # Scale-aware NVFP4 weights dequantized into the BF16 MoE compute lane.


class FMHAQuantMode(Enum):
    """
    FMHA quant mode.
    """

    bfloat16 = QuantMapping(2, 1, "bfloat16", "bfloat16")
    float16 = QuantMapping(2, 1, "float16", "bfloat16")  # sglang decode attention uses float16 compute
    fp8 = QuantMapping(1, 2, "fp8", "fp8")
    fp8_block = QuantMapping(1, 2, "fp8_block", "fp8")  # FIXME: specific for sglang wideep


class KVCacheQuantMode(Enum):
    """
    KVCache quant mode.
    """

    bfloat16 = QuantMapping(2, 0, "bfloat16", None)
    int8 = QuantMapping(1, 0, "int8", None)
    fp8 = QuantMapping(1, 0, "fp8", None)


class CommQuantMode(Enum):
    """
    Comm quant mode.
    """

    half = QuantMapping(2, 0, "half", None)
    int8 = QuantMapping(1, 0, "int8", None)
    fp8 = QuantMapping(1, 0, "fp8", None)
