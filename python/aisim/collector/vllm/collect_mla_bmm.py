# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM MLA generation BMM micro-collector.

Benchmarks the batched-matrix-multiply helpers used around MLA generation
pre/post processing (the ``W_UK_T`` absorb and ``W_UV`` projection). On
NVIDIA, vLLM runs BOTH as plain bf16 ``torch.bmm`` whatever the kv_b_proj
quantization: process_weights_after_loading dequantizes kv_b_proj to the model
dtype (split_kv_b_proj, mla_attention.py:407-422) and the fp8/fp4 BMM variants
are ROCm/AITER-gated (``rocm_aiter_ops.is_fp8bmm_enabled``, :711-718):

- stock MLA: mla_attention.py:1022-1027,1087-1088 (``torch.bmm(mqa_q_nope,
  W_UK_T, out=mqa_ql_nope.transpose(0, 1))``) and :1383-1384,1413
  (``torch.bmm(x, self.W_UV, out=out.transpose(0, 1))``);
- Kimi-K3 custom MLA: models/kimi_k3/nvidia/mla.py:696 (BMM1 absorb, into a
  fresh (N, B, L) result) and :481 (W_UV projection) -- the same torch.bmm.
  All @v0.32.0.

The lane therefore collects the ``bfloat16`` dtype only: the plan leaves the
fp8 region out (collector/skips.py). Until this collector existed,
vLLM MLA-BMM queries fell back to the trtllm tables (the SDK's
"low-fidelity fallback rows for mla_bmm_perf" warning).
"""

# 0.27.0 audit: same preflight as collect_mla_module.py (module surface and
# MLA bmm call site unchanged in-image on GB300).
# B200 0.25.0 qualification: both pre/post torch.bmm callbacks passed in
# job 1968047. The NVIDIA bf16 BMM contract and input geometry are unchanged.
__compat__ = "vllm==0.32.0"

from importlib.metadata import (
    version as _dist_version,  # setuptools/pkg_resources is absent from recent framework images
)

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases


def get_mla_gen_pre_test_cases():
    return plan_cases("mla_bmm_gen_pre", "vllm", get_sm_version())


def get_mla_gen_post_test_cases():
    return plan_cases("mla_bmm_gen_post", "vllm", get_sm_version())


def _log_row(*, op_name, dtype, num_tokens, num_heads, results, perf_filename, device):
    if not log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=_dist_version("vllm"),
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source="vllm_torch_bmm",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    ):
        raise RuntimeError(f"Failed to persist vLLM MLA BMM performance row to {perf_filename}")


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    assert dtype == "bfloat16", "NVIDIA vLLM MLA pre-BMM is bf16 torch.bmm only"

    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512

    # BMM1 absorb (N, B, P) x (N, P, L) with serving's operands: q_nope is the
    # nope slice of the q_b_proj output viewed (B, N, P + R) (layers/mla.py:221-225,
    # mla_attention.py:1022-1027), and the result is written through a transposed
    # view of a token-major (B, N, L) buffer (:1087-1088; every backend but CUTLASS
    # MLA, the only one padding q heads, cutlass_mla.py:134). All @v0.32.0.
    # Deviation (every SM): W_UK_T is contiguous. Serving's W_UK_T (and W_UV in
    # run_mla_gen_post) are views of the dequantized kv_b_proj weight
    # (mla_attention.py:1325-1327, split_kv_b_proj :407-422) whose strides depend on
    # how kv_b_proj dequantizes: as stored when it is BF16 or dequantized by scaling,
    # transposed when vLLM dequantizes it through an identity GEMM (DeepGEMM FP8
    # block, NVFP4; get_and_maybe_dequant_weights, quantization/utils/quant_utils.py:
    # 511-571). The row's key does not carry that, and cuBLAS may run another
    # transpose variant. Kimi-K3's BMM1 writes a fresh (N, B, L) result instead of
    # the transposed view. Direction unknown; retiring it needs the checkpoint
    # quantization in the mla_bmm key.
    q = torch.randn((num_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim), device=device, dtype=torch.bfloat16)
    q_nope = q[..., :qk_nope_head_dim].transpose(0, 1)
    w_uk_t = torch.randn((num_heads, qk_nope_head_dim, kv_lora_rank), device=device, dtype=torch.bfloat16)
    ql_nope = torch.empty((num_tokens, num_heads, kv_lora_rank), device=device, dtype=torch.bfloat16)

    def kernel_func():
        torch.bmm(q_nope, w_uk_t, out=ql_nope.transpose(0, 1))

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    _log_row(
        op_name="mla_gen_pre",
        dtype=dtype,
        num_tokens=num_tokens,
        num_heads=num_heads,
        results=results,
        perf_filename=perf_filename,
        device=device,
    )


def run_mla_gen_post(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    assert dtype == "bfloat16", "NVIDIA vLLM MLA post-BMM is bf16 torch.bmm only"

    kv_lora_rank = 512
    v_head_dim = 128

    # W_UV projection: (N, B, L) x (N, L, V) with a transposed-view output, the
    # decode attention output viewed (B, N, L) -- mla_attention.py:1383-1384,1413
    # @v0.32.0 / kimi-k3 mla.py:478-481. W_UV is contiguous (deviation note in
    # run_mla_gen_pre).
    attn_output = torch.randn((num_tokens, num_heads, kv_lora_rank), device=device, dtype=torch.bfloat16)
    w_uv = torch.randn((num_heads, kv_lora_rank, v_head_dim), device=device, dtype=torch.bfloat16)
    attn_bmm_output = torch.empty((num_tokens, num_heads, v_head_dim), device=device, dtype=torch.bfloat16)

    def kernel_func():
        torch.bmm(
            attn_output.transpose(0, 1),
            w_uv,
            out=attn_bmm_output.transpose(0, 1),
        )

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    _log_row(
        op_name="mla_gen_post",
        dtype=dtype,
        num_tokens=num_tokens,
        num_heads=num_heads,
        results=results,
        perf_filename=perf_filename,
        device=device,
    )
