# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang MLA generation BMM micro-collector.

Benchmarks the small FP8/BF16 batched-matrix-multiply kernels used around MLA
generation pre/post processing. It consumes YAML-backed synthetic tensor
shapes, calls the kernels SGLang's absorbed MLA path calls, and logs the
resulting MLA BMM perf rows.

The calls follow serving's DeepseekV2AttentionMLA absorbed path
(models/deepseek_common/attention_forward_methods/forward_mla.py @v0.5.23):
bf16 is ``torch.bmm`` against the dequantized ``w_kc`` / ``w_vc`` (:586, :921-932),
fp8 is ``per_tensor_quant_mla_fp8`` + ``bmm_fp8`` against the per-tensor-requantized
weights (:562-584, :889-914). Serving takes the fp8 branch only for a kv_b_proj that
is not 128x128 block-quantized: a 128x128 block FP8 checkpoint is dequantized to bf16
(models/deepseek_common/deepseek_weight_loader.py:690-712; the DeepGEMM BMM needs
SGLANG_USE_DEEPGEMM_BMM, default off, environ.py:1375).
"""

__compat__ = "sglang==0.5.23"

from importlib.metadata import version as _dist_version  # setuptools/pkg_resources is absent from sglang>=0.5.21 images

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases

# Serving's imports (forward_mla.py:10-13, :95-96): bmm_fp8 is the kernel-registry entry
# that runs flashinfer's cuBLAS bmm_fp8 (kernels/ops/gemm/__init__.py:178-184, :288-298;
# kernels/ops/gemm/flashinfer_bmm_fp8.py), and the Triton per-tensor quant lives under
# sglang.kernels (kernels/ops/quantization/fp8_kernel.py:828).
from sglang.kernels.ops.gemm import bmm_fp8
from sglang.kernels.ops.quantization.fp8_kernel import per_tensor_quant_mla_fp8
from sglang.srt.utils import is_nvidia_cublas_version_ge_12_9

_BMM_FP8_SOURCE = "sglang_flashinfer_bmm_fp8"

# With cuBLAS >= 12.9 serving hands the quant a fresh zero scale per call
# (forward_mla.py:570-577, :897-906: torch.zeros, one fill kernel inside the timed
# step); below it the scale slot comes from the forward's pre-zeroed BumpAllocator.
_FRESH_ZERO_SCALE = is_nvidia_cublas_version_ge_12_9()


def _zero_scale(prezeroed: torch.Tensor) -> torch.Tensor:
    if _FRESH_ZERO_SCALE:
        return torch.zeros((1,), dtype=torch.float32, device=prezeroed.device)
    return prezeroed


def get_mla_gen_pre_test_cases():
    return plan_cases("mla_bmm_gen_pre", "sglang", get_sm_version())


def get_mla_gen_post_test_cases():
    return plan_cases("mla_bmm_gen_post", "sglang", get_sm_version())


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert dtype == "fp8" or dtype == "bfloat16", "only support fp8 and bfloat16"

    qk_nope_head_dim = 128
    kv_lora_rank = 512

    q_nope = torch.randn((num_tokens, num_heads, qk_nope_head_dim), device=device, dtype=torch.bfloat16)

    if dtype == "fp8":
        prezeroed_scale = torch.zeros((1,), dtype=torch.float32, device=device)
        # w_kc as the weight loader leaves it: (heads, qk_nope, kv_lora) over a
        # (heads, kv_lora, qk_nope)-contiguous buffer (deepseek_weight_loader.py:780-786),
        # with the per-tensor scale of block_quant_to_tensor_quant / channel_quant_to_tensor_quant
        # (layers/quantization/fp8_utils.py:2001-2041, :2420).
        w_kc = torch.randn((num_heads, kv_lora_rank, qk_nope_head_dim), dtype=torch.bfloat16, device=device).to(
            dtype=torch.float8_e4m3fn
        )
        w_kc = w_kc.transpose(1, 2)
        w_scale = torch.ones((1,), dtype=torch.float32, device=device)

        def kernel_func():
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(q_nope.transpose(0, 1), _zero_scale(prezeroed_scale))
            bmm_fp8(q_nope_val, w_kc, q_nope_scale, w_scale, torch.bfloat16)
    else:
        w_kc = torch.randn((num_heads, kv_lora_rank, qk_nope_head_dim), dtype=torch.bfloat16, device=device)
        w_kc = w_kc.transpose(1, 2)

        def kernel_func():
            torch.bmm(q_nope.transpose(0, 1), w_kc)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    if not log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=_dist_version("sglang"),
        device_name=torch.cuda.get_device_name(device),
        op_name="mla_gen_pre",
        kernel_source=_BMM_FP8_SOURCE if dtype == "fp8" else "sglang_torch_bmm",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    ):
        raise RuntimeError(f"Failed to persist SGLang MLA generation pre-BMM performance row to {perf_filename}")


def run_mla_gen_post(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert dtype == "bfloat16" or dtype == "fp8", "only support fp8 and bfloat16"

    kv_lora_rank = 512
    v_head_dim = 128

    if dtype == "bfloat16":
        attn_output = torch.randn([num_tokens, num_heads, kv_lora_rank]).bfloat16().to(torch.device(device))
        w_vc = torch.randn([num_heads, v_head_dim, kv_lora_rank]).bfloat16().to(torch.device(device))
        w_vc = w_vc.transpose(1, 2)
        attn_bmm_output = torch.empty(
            (num_tokens, num_heads, v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )

        torch.bmm(
            attn_output.transpose(0, 1),
            w_vc,
            out=attn_bmm_output.transpose(0, 1),
        )

        def kernel_func():
            torch.bmm(
                attn_output.transpose(0, 1),
                w_vc,
                out=attn_bmm_output.transpose(0, 1),
            )
    else:
        attn_output = torch.randn([num_tokens, num_heads, kv_lora_rank], dtype=torch.bfloat16, device=device)
        w_vc = torch.randn([num_heads, v_head_dim, kv_lora_rank], dtype=torch.bfloat16, device=device).to(
            dtype=torch.float8_e4m3fn
        )
        w_vc = w_vc.transpose(1, 2)
        w_scale = torch.ones((1,), dtype=torch.float32, device=device)  # per-tensor, as for w_kc
        prezeroed_scale = torch.zeros((1,), dtype=torch.float32, device=device)

        def kernel_func():
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                _zero_scale(prezeroed_scale),
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                w_vc,
                attn_output_scale,
                w_scale,
                torch.bfloat16,
            )
            # Serving hands o_proj the token-major rows: the transpose+flatten is a
            # copy of the (heads, tokens, v_head_dim) bmm_fp8 output (forward_mla.py:914).
            attn_bmm_output.transpose(0, 1).flatten(1, 2)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    if not log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="SGLang",
        version=_dist_version("sglang"),
        device_name=torch.cuda.get_device_name(device),
        op_name="mla_gen_post",
        kernel_source=_BMM_FP8_SOURCE if dtype == "fp8" else "sglang_torch_bmm",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    ):
        raise RuntimeError(f"Failed to persist SGLang MLA generation post-BMM performance row to {perf_filename}")
