# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM encoder (non-causal) attention collector for multimodal encoders.

Times the kernel vLLM's ``MMEncoderAttention`` runs for the ViT backend its
selector picks: FlashAttention called as ``vit_flash_attn_wrapper`` calls it
eagerly, ``vit_triton_attn_wrapper`` or ``vit_torch_sdpa_wrapper``.

Quant: bf16 only. vLLM upstream supports fp8 ViT FMHA via FLASHINFER;
enabling that path here is left for the future.
"""

# Audited against vLLM v0.32.0 (5df63a72, from v0.30.0 ced6857a); citations
# below are v0.32.0 lines. Every import and wrapper signature used here exists
# unchanged. Selection is unchanged: MMEncoderAttention asks
# get_vit_attn_backend (model_executor/layers/attention/
# mm_encoder_attention.py:352-367 -> models/vision.py:100-113), which on SM80
# and newer takes the first of FLASH_ATTN and TRITON_ATTN that accepts the head
# size and dtype, else TORCH_SDPA (platforms/cuda.py:518-572); FLASHINFER
# (cuDNN) is reachable only through --mm-encoder-attn-backend. The FA version is
# get_flash_attn_version(head_size) (fa_utils.py:74-221): FA3 on SM90, FA4 on
# SM10x up to head_dim 128 (FA2 above, :193-209), FA2 elsewhere. Kernel-side
# changes: vllm-flash-attention moves 506341a1 -> 9cd61de3
# (cmake/external_projects/vllm_flash_attn.cmake:42) and nvidia-cutlass-dsl,
# which compiles FA4, 4.7.1 -> 4.8.0 (requirements/cuda.txt), so the FA rows
# change. Triton's context_attention_fwd narrows its tile only at head_dim 512
# or on ROCm (ops/triton_prefill_attention.py:248-249, :265), which these
# shapes never reach. The SDPA wrapper now copies its output contiguous when
# called without cu_seqlens (ops/vit_attn_wrappers.py:287-288), in serving too.
#
# Deviation (every SM): serving runs the encoder eagerly by default
# (compile_mm_encoder and cudagraph_mm_encoder are False, config/
# compilation.py:520-528), so each layer pays its Python launch path and the
# `.item()` host sync on the max_seqlen tensor the model passes
# (vit_attn_wrappers.py:53-58). The rows time CUDA-graph replays of the same
# kernels: GPU time without launch gaps. They under-predict where the eager
# encoder is launch-bound (small batch x seq_len); size unmeasured. Retire it by
# timing eagerly, which the row must then record.
__compat__ = "vllm==0.32.0"

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.vit_attn_wrappers import (
    vit_torch_sdpa_wrapper,
    vit_triton_attn_wrapper,
)
from vllm.version import __version__ as vllm_version


def get_encoder_attention_test_cases():
    return plan_cases("encoder_attention", "vllm", get_sm_version())


def run_encoder_attention_torch(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    scale = 1.0 / (head_dim**0.5)

    # Same selector MMEncoderAttention uses (cuda.py:get_supported_vit_attn_backends).
    backend = get_vit_attn_backend(head_size=head_dim, dtype=dtype)

    # ViT wrappers expect (B, S, N, D); internal einops rearrange handles varlen.
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)

    # Pre-generate cu_seqlens so the FA/Triton wrappers skip their internal ``torch.arange`` fallback.
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seq_len,
        step=seq_len,
        dtype=torch.int32,
        device=device,
    )

    if backend == AttentionBackendEnum.FLASH_ATTN:
        # mm_encoder_attention.py:363-367.
        fa_version = get_flash_attn_version(head_size=head_dim)
        if fa_version is None:
            raise RuntimeError("vLLM selected FlashAttention for ViT without a concrete FA version")
        # vit_flash_attn_wrapper (mm_encoder_attention.py:580-590 ->
        # vit_attn_wrappers.py:26-86) zero-fills a fresh output buffer whenever
        # it runs inside a CUDA-graph capture (:63-67); eager it hands
        # flash_attn_varlen_func an empty buffer (:68-69). Serving runs the
        # encoder eagerly (see the deviation note above), so time that eager
        # call: the same FlashAttention kernel and arguments with a
        # preallocated output and no zero fill, which a capture of the wrapper
        # would add. The (b s) flattening is the wrapper's view (:60).
        q_flat, k_flat, v_flat = (x.reshape(batch_size * seq_len, num_heads, head_dim) for x in (q, k, v))
        out = torch.empty_like(q_flat)

        def run():
            flash_attn_varlen_func(
                q_flat,
                k_flat,
                v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                dropout_p=0.0,
                causal=False,
                softmax_scale=scale,
                fa_version=fa_version,
                out=out,
            )
    elif backend == AttentionBackendEnum.TRITON_ATTN:

        def run():
            vit_triton_attn_wrapper(
                q,
                k,
                v,
                batch_size=batch_size,
                scale=scale,
                cu_seqlens=cu_seqlens,
            )
    elif backend == AttentionBackendEnum.TORCH_SDPA:

        def run():
            vit_torch_sdpa_wrapper(
                q,
                k,
                v,
                scale=scale,
            )
    else:
        # FlashInfer ViT needs cu_seqlens padding + workspace; not on the ViT default path.
        raise NotImplementedError(f"ViT backend {backend} not supported by collector")

    with benchmark_with_power(
        device=device,
        kernel_func=run,
        num_warmups=3,
        num_runs=6,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    print(f"encoder attn latency: {latency}")

    if backend == AttentionBackendEnum.FLASH_ATTN:
        kernel_source = f"vllm_vit_flash_attn_fa{fa_version}"
    else:
        kernel_source = f"vllm_vit_{backend.name}".lower()

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "attn_dtype": "bfloat16",
                "latency": latency,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="encoder_attention",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_encoder_attention_test_cases()
    for test_case in test_cases:
        try:
            run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
        except Exception as e:
            print(f"[encoder_attention] case {test_case} failed: {type(e).__name__}: {e}")
