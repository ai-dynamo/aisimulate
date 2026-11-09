# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang encoder (non-causal) attention collector for multimodal / omni-modal models.

SM dispatch mirrors ``VisionAttention._determine_attention_backend`` (sglang v0.5.23):

- capability major 9  (SM90, Hopper)            -> FA3 ``flash_attn_varlen_func``
- capability major 10 (SM100/SM103, B200/B300)  -> FA4 ``flash_attn_varlen_func``
- other CUDA (SM<90, SM120)                     -> Triton ``context_attention_fwd``

Serving branches on the capability MAJOR when neither ``--mm-attention-backend`` nor the
model passes a backend (layers/attention/vision.py:1304-1311: ``major == 9 -> fa3``,
``major == 10 -> fa4``, else ``triton_attn``), so SM103 shares the FA4 lane with SM100.
The collector's earlier ``sm == 100`` test sent SM103 down the Triton path (B300 encoder
gate failed) — a wrong-kernel measurement, not a crash.

Each lane calls the kernel function the serving module calls per layer
(VisionFlash3Attention / VisionFlash4Attention / VisionTritonAttention.forward,
vision.py:453-639), with the same arguments.

Quant: bf16 only. SGLang upstream does not support fp8 ViT FMHA.
"""

# One release per file; earlier code is `git log -- <this file>` away.
__compat__ = "sglang==0.5.23"

from importlib.metadata import (
    version as _dist_version,  # setuptools/pkg_resources is absent from recent framework images
)
from typing import NamedTuple

import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases


def serving_vision_backend(sm_version: int) -> str:
    """The vision-attention backend sglang selects for a CUDA capability (vision.py:1304-1311)."""
    major = sm_version // 10
    if major == 9:
        return "fa3"
    if major == 10:
        return "fa4"
    return "triton_attn"


class Timing(NamedTuple):
    mean: float


def get_encoder_attention_test_cases():
    return plan_cases("encoder_attention", "sglang", get_sm_version())


def _build_kernel_runner(
    device: torch.device,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Return ``(run_iter, backend_tag)`` for the given device and shape."""
    if device.type != "cuda":
        raise RuntimeError(f"encoder attention collector requires CUDA device, got {device}")

    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

    # Uniform batch: cu_seqlens = [0, s, 2s, ..., b*s], max_seqlen = s
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seq_len,
        step=seq_len,
        dtype=torch.int32,
        device=device,
    )
    max_seqlen = seq_len
    softmax_scale = head_dim**-0.5

    sm = get_sm_version()  # 90 / 100 / 103 / 120 / ...

    backend = serving_vision_backend(sm)
    if backend == "fa3":
        # VisionFlash3Attention.forward (vision.py:525-582) calls flash_attn_func(q, k, v,
        # **fa_kwargs), which is flash_attention.flash_attn_varlen_func at ver=3
        # (vision.py:54, 62-69).
        from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

        def run_iter():
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                window_size=(-1, -1),
            )

        return run_iter, "flash_attention_v3"

    if backend == "fa4":
        # VisionFlash4Attention.forward (vision.py:585-639) calls flash_attn_func(..., ver=4),
        # which goes straight to flash_attention_v4.flash_attn_varlen_func with only these
        # arguments (vision.py:62-68), not through the generic ver dispatcher.
        from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_varlen_func

        def run_iter():
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
            )

        return run_iter, "flash_attention_v4"

    # every other capability (SM<90, SM120): VisionTritonAttention.forward (vision.py:453-522)
    # calls context_attention_fwd (imported at vision.py:83-85).
    from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd

    seq_lens = torch.full(
        (batch_size,),
        seq_len,
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty_like(q)

    def run_iter():
        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens,
            seq_lens,
            max_seqlen,
            is_causal=False,
            sm_scale=softmax_scale,
        )

    return run_iter, "triton"


def run_encoder_attention_torch(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    *,
    perf_filename,
    device="cuda:0",
):
    from collector.sglang.runtime_compat import ensure_offline_runtime_published

    ensure_offline_runtime_published()  # sglang>=0.5.20 backends read get_exec()/get_parallel()
    torch_device = torch.device(device)
    torch.cuda.set_device(device)

    run_iter, backend_tag = _build_kernel_runner(
        device=torch_device,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
    )

    with benchmark_with_power(
        device=torch_device,
        kernel_func=run_iter,
        num_warmups=3,
        num_runs=20,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]

    if not log_perf(
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
        framework="SGLang",
        version=_dist_version("sglang"),
        device_name=torch.cuda.get_device_name(device),
        op_name="encoder_attention",
        kernel_source=backend_tag,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    ):
        raise RuntimeError(f"Failed to persist SGLang encoder attention performance row to {perf_filename}")

    return Timing(latency * 1e-3)


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_encoder_attention_test_cases():
        try:
            run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
        except Exception as e:
            print(f"[encoder_attention] case {test_case} failed: {type(e).__name__}: {e}")
