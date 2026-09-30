# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KDA (Kimi Delta Attention) Collector for AIConfigurator — SGLang backend.

Kimi-K3's linear-attention layers run KDA, a gated delta-rule attention with
a per-key (full-rank) gate. SGLang serves it through its vendored Triton FLA
kernels (the default `--linear-attn-backend triton` on all SMs; the CuTe /
FlashInfer fast paths are SM100-only and fall back to Triton elsewhere —
kda_backend.py @ kimi-k3 branch).

Context (prefill/extend) phase (kda_backend.forward_extend):
    - causal_conv1d_fn x3: separate Q, K, V causal convolutions
      (kernel_source "causal_conv1d_fn_qkv3": one row = the 3-call sequence)
    - chunk_kda: chunked delta-rule scan with raw per-K gates

Generation (decode) phase (kda_backend.forward_decode):
    - causal_conv1d_update: packed single-step conv over 3P channels
    - fused_recurrent_kda_packed_decode: packed KDA recurrence (T=1)
    - "kda_fused_decode" where serving's attempt-and-verify covered() probe
      accepts the shard (see _fused_decode_module below)

Verify (speculative target-verify) phase (kda_backend._forward_target_verify):
    SM-dispatched like serving (_can_run_dspark_cutedsl_mtp):
    - SM100 + cutlass + draft width 2..8 + 128-dim symmetric bf16 heads:
      "fused_kda_decode_mtp_dspark" — the CuTeDSL fused conv+chain-verify
      recurrence kernel (one row covers the whole verify step). Serving folds
      gated RMSNorm into the kernel only on the TP8 12-head shard
      (_prepare_fused_decode @ models/kimi_k3.py); the collector mirrors that
      per-shard.
    - otherwise, the Triton pair the serving fallback runs:
      - causal_conv1d_update over batch*draft_tokens rows (approximates the
        windowed verify conv; the serving path adds per-draft-token conv-window
        checkpointing bookkeeping on top of the same kernel)
      - fused_sigmoid_gating_delta_rule_update (is_kda=True, chain verify,
        disable_state_update + intermediate state caching)

The in_proj/out_proj/gate GEMMs are standard linear layers modeled by the
existing GEMM infrastructure. This collector focuses on the unique KDA ops.

GLM-5.3-Flash (Glm5NextForConditionalGeneration, SGLang 0.5.20) rows are
routed by model path to the glm5_next serving dispatch (see the GLM section):
one packed q|k|v conv ("causal_conv1d_fn" / "causal_conv1d_update"), the
prefill scan "chunk_kda" (in-kernel bounded gate, raw beta) and the
bounded-gate decode "fused_sigmoid_gating_delta_rule_update".

Output:
    kda_perf.txt — same column layout as gdn_perf (phase, batch_size, seq_len,
    num_tokens, d_model, d_conv, num_k_heads, head_k_dim, num_v_heads,
    head_v_dim, model_name, latency), with seq_len carrying the per-request
    draft-token count for verify rows.
"""

# The kimi-k3 branch build (https://github.com/sgl-project/sglang/tree/kimi-k3)
# reports 0.5.16. 0.5.20 adds GLM-5.3-Flash (glm5_next) KDA, audited against
# the GB300 0.5.20 image (see the GLM section below). The unverified releases
# in between stay excluded.
__compat__ = "sglang>=0.5.16,<=0.5.20,!=0.5.17,!=0.5.18,!=0.5.19"

import gc
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn, causal_conv1d_update

import torch

try:
    from collector.case_generator import get_common_kda_test_cases
    from collector.helper import (
        WORKER_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_kda_test_cases

    from helper import (
        WORKER_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )

aic_debug = int(os.getenv("aic_kda_debug", "0"))  # noqa: SIM112

# KDA safe-gate lower bound: fixed model constant for Kimi-K3
# (config.json linear_attn_config.gate_lower_bound).
KDA_LOWER_BOUND = -5.0


def get_kda_test_cases():
    """
    Generate test cases for KDA kernel benchmarking.

    Returns a list of test case configurations for context (prefill),
    generation (decode) and verify (speculative target-verify) phases.
    """
    test_cases = []

    for common_case in get_common_kda_test_cases():
        test_cases.append(
            [
                common_case.phase,
                common_case.d_model,
                common_case.d_conv,
                common_case.num_k_heads,
                common_case.head_k_dim,
                common_case.num_v_heads,
                common_case.head_v_dim,
                common_case.batch_size_list,
                common_case.seq_len_list,
                common_case.model_name,
            ]
        )

    return test_cases


def _cleanup(tag: str):
    cleanup_errors = []
    for cleanup_name, cleanup_fn in (
        ("gc.collect", gc.collect),
        ("torch.cuda.empty_cache", torch.cuda.empty_cache),
    ):
        try:
            cleanup_fn()
        except Exception as cleanup_error:
            cleanup_errors.append(f"{cleanup_name}: {type(cleanup_error).__name__}: {cleanup_error}")
    if cleanup_errors:
        raise RuntimeError(f"SGLang KDA {tag} cleanup failed: {'; '.join(cleanup_errors)}")


def _format_failures(failures: list[str], limit: int = 8) -> str:
    """Compact per-cell failure evidence for the strict-completeness raise.

    The full list is on stdout; the raised message carries the first ``limit``
    cells so the classified failure record is traceable to shapes without
    ballooning the errors json."""
    if not failures:
        return "<none>"
    shown = "; ".join(failures[:limit])
    extra = len(failures) - limit
    return shown + (f"; ... and {extra} more (see worker stdout)" if extra > 0 else "")


def run_kda_context_benchmark(
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    seq_len_list: list[int],
    model_name: str,
    perf_filename: str,
    sglang_version: str,
    device: str = "cuda:0",
):
    """
    Benchmark KDA context (prefill) kernels: the 3-way Q/K/V causal conv
    sequence and the chunked delta-rule scan (chunk_kda), mirroring
    kda_backend.forward_extend at the kimi-k3 branch.
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim, "KDA requires symmetric Q/K/V head geometry"
    proj_size = num_v_heads * head_v_dim
    conv_channels = 3 * proj_size

    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    q_conv_weight, k_conv_weight, v_conv_weight = conv_weight.split([proj_size] * 3, dim=0)
    successful_points = 0
    failed_points = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            total_tokens = batch_size * seq_len
            if aic_debug:
                print(f"  Benchmarking batch_size={batch_size}, seq_len={seq_len}")

            a = beta = conv_pool = cu_seqlens = has_initial_state = mixed_qkv = None
            q = k = v = recurrent_state = seq_lens_cpu = state_indices = None
            try:
                # Same int32 token-offset overflow class as the GDN conv kernel
                # (causal_conv1d_triton.py:373-379). Although the conv runs per
                # Q/K/V block (proj_size channels), each block is a strided view
                # over the full 3-block mixed_qkv buffer, so the kernel's int32
                # pointer arithmetic spans total_tokens * conv_channels elements.
                # Silicon evidence: cells with total_tokens * conv_channels in
                # [2**31, 3*2**31) crash with cudaErrorIllegalAddress on both
                # Hopper SM90 (2026-07 campaign coverage boundary) and SM100
                # (B200, 2026-07-28), while every cell under this bound passes.
                if total_tokens * conv_channels >= 2**31:
                    raise ValueError(
                        "SGLang causal_conv1d Triton kernel int32 token-offset overflow: "
                        f"total_tokens={total_tokens} * conv_channels={conv_channels} >= 2**31 "
                        "(causal_conv1d_triton.py:373-379; per-block views stride "
                        "across the whole 3-block buffer)"
                    )
                num_warmups = 3
                num_runs = 10
                cu_seqlens = torch.arange(0, total_tokens + 1, seq_len, dtype=torch.int32, device=device)
                seq_lens_cpu = [seq_len] * batch_size
                state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
                has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=device)
                # KDA pool layout is [slots, K-1, dim]; serving transposes to
                # [slots, dim, K-1] before splitting per Q/K/V block
                # (kda_backend.py:573,599).
                conv_pool = torch.zeros(batch_size, d_conv - 1, conv_channels, dtype=dtype, device=device)
                conv_states = conv_pool.transpose(-1, -2)
                q_conv_state, k_conv_state, v_conv_state = conv_states.split([proj_size] * 3, dim=-2)
                recurrent_state = torch.zeros(
                    batch_size, num_v_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device
                )

                mixed_qkv = torch.randn(total_tokens, conv_channels, dtype=dtype, device=device)
                q_in, k_in, v_in = mixed_qkv.transpose(0, 1).split([proj_size] * 3, dim=0)
                # Raw per-K forget gate (f_b_proj output) and beta (pre-sigmoid);
                # chunk_kda consumes raw gates plus A_log/dt_bias in-kernel.
                a = torch.randn(1, total_tokens, num_v_heads, head_k_dim, dtype=dtype, device=device)
                beta = torch.randn(1, total_tokens, num_v_heads, dtype=dtype, device=device)
                a_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
                dt_bias = torch.ones(num_v_heads * head_k_dim, dtype=torch.float32, device=device)

                common_log_data = {
                    "phase": "context",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_tokens": total_tokens,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                def run_conv1d_qkv3():
                    for blk_in, blk_weight, blk_state in (
                        (q_in, q_conv_weight, q_conv_state),
                        (k_in, k_conv_weight, k_conv_state),
                        (v_in, v_conv_weight, v_conv_state),
                    ):
                        causal_conv1d_fn(
                            blk_in,
                            blk_weight,
                            None,
                            activation="silu",
                            conv_states=blk_state,
                            has_initial_state=has_initial_state,
                            cache_indices=state_indices,
                            query_start_loc=cu_seqlens,
                            seq_lens_cpu=seq_lens_cpu,
                        )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv1d_qkv3,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                ) as results:
                    if not log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="SGLang",
                        version=sglang_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="kda",
                        kernel_source="causal_conv1d_fn_qkv3",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    ):
                        raise RuntimeError(f"failed to persist SGLang KDA context row to {perf_filename}")

                q = torch.randn(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device)
                k = torch.randn(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device)
                v = torch.randn(1, total_tokens, num_v_heads, head_v_dim, dtype=dtype, device=device)

                def run_kda_scan():
                    chunk_kda(
                        q=q,
                        k=k,
                        v=v,
                        g=a,
                        beta=beta,
                        initial_state=recurrent_state,
                        initial_state_indices=state_indices,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=cu_seqlens,
                        A_log=a_log,
                        dt_bias=dt_bias,
                        lower_bound=KDA_LOWER_BOUND,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_kda_scan,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                ) as results:
                    if not log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="SGLang",
                        version=sglang_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="kda",
                        kernel_source="chunk_kda",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    ):
                        raise RuntimeError(f"failed to persist SGLang KDA context row to {perf_filename}")
                successful_points += 1

            except Exception as e:
                failed_points += 1
                failures.append(f"batch_size={batch_size} seq_len={seq_len}: {type(e).__name__}: {e}")
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue
            finally:
                a = beta = conv_pool = cu_seqlens = has_initial_state = mixed_qkv = None
                q = k = v = recurrent_state = seq_lens_cpu = state_indices = None
                _cleanup("context")

    summary = f"ok={successful_points} error={failed_points} skip=0"
    print(f"KDA context summary: {summary}")
    if failed_points or successful_points == 0:
        raise RuntimeError(
            f"SGLang KDA context collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def _fused_decode_module():
    """Serving's attempt-and-verify fused decode module: the KDA decode path
    offers every step to sglang.kernels.ops.attention.kda_fused_decode and
    lets its own covered() accept or reject (the kernel is compiled for the
    K3 TP8 12-head/128-dim shard; the model stashes the static args only for
    that shard, _prepare_fused_decode @ models/kimi_k3.py). The collector
    replicates NONE of that shape logic — it hands the constructed tensors to
    the same covered() probe and benchmarks whichever side the framework
    picks, exactly like serving's fallback chain.
    """
    from sglang.kernels.ops.attention import kda_fused_decode

    return kda_fused_decode


def run_kda_generation_benchmark(
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    model_name: str,
    perf_filename: str,
    sglang_version: str,
    device: str = "cuda:0",
):
    """
    Benchmark KDA generation (decode) kernels: packed conv-state update and the
    packed KDA recurrence, mirroring kda_backend.forward_decode (Triton path).
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim, "KDA requires symmetric Q/K/V head geometry"
    proj_size = num_v_heads * head_v_dim
    conv_channels = 3 * proj_size

    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    successful_points = 0
    failed_points = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        if aic_debug:
            print(f"  Benchmarking batch_size={batch_size}")

        a = a_log = b = conv_pool = dt_bias = mixed_qkv = None
        output = recurrent_state = state_indices = None
        try:
            num_warmups = 3
            num_runs = 10
            mixed_qkv = torch.randn(batch_size, conv_channels, dtype=dtype, device=device)
            conv_pool = torch.zeros(batch_size, d_conv - 1, conv_channels, dtype=dtype, device=device)
            state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            recurrent_state = torch.zeros(
                batch_size, num_v_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device
            )
            # a: per-K gate input [B, HV*K]; b: beta [B, HV] (pre-sigmoid).
            a = torch.randn(batch_size, num_v_heads * head_k_dim, dtype=dtype, device=device)
            b = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
            a_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
            dt_bias = torch.ones(num_v_heads * head_k_dim, dtype=torch.float32, device=device)
            output = torch.empty(batch_size, 1, num_v_heads, head_v_dim, dtype=dtype, device=device)

            common_log_data = {
                "phase": "generation",
                "batch_size": batch_size,
                "seq_len": 1,
                "num_tokens": batch_size,
                "d_model": d_model,
                "d_conv": d_conv,
                "num_k_heads": num_k_heads,
                "head_k_dim": head_k_dim,
                "num_v_heads": num_v_heads,
                "head_v_dim": head_v_dim,
                "model_name": model_name,
            }

            # Attempt-and-verify exactly like serving: hand the decode tensors
            # to the fused kernel's own covered() probe and benchmark
            # whichever path it selects (True -> fused conv+recurrence+onorm
            # in one launch; False -> the Triton packed pair, serving's
            # fallback for uncovered shapes). No shard shapes and no SM
            # predicate are replicated here — serving itself has neither:
            # the model stashes the fused args on every NVIDIA GPU (HIP and
            # weight-layout checks only, srt/models/kimi_k3.py:1563-1614 @
            # c6ad1f26), the backend calls the kernel wherever covered()
            # accepts with NO try/except around the JIT load
            # (srt/layers/attention/linear/kda_backend.py:426-476 @
            # c6ad1f26), and the JIT compiles lazily at first call
            # (kernels/ops/attention/kda_fused_decode.py:42-51). Below SM90
            # that call raises (ptxas rejects mbarrier.try_wait.parity:
            # "requires .target sm_90 or higher", L40S/SM89 2026-08-01), so
            # the covered 12-head cells fail CLASSIFIED here — the same
            # crash serving hits; substituting the Triton pair would be an
            # invented fallback (layer_permissions.md).
            fused = _fused_decode_module()
            onorm_g = torch.randn(batch_size, proj_size, dtype=dtype, device=device)
            if fused.covered(mixed_qkv, a, b, conv_pool, recurrent_state, state_indices, onorm_g):
                kda_fused_decode = fused.kda_fused_decode
                # Serving static args (_prepare_fused_decode): per-block
                # transposed fp32 conv weights [d_conv, proj], dense fp32
                # conv bias, fp32 o_norm weight; onorm gate is a per-token
                # bf16 activation.
                conv_weight_f32 = torch.randn(conv_channels, d_conv, dtype=torch.float32, device=device)
                conv_weight_f32_t = conv_weight_f32.t().contiguous()
                w_q_t = conv_weight_f32_t[:, :proj_size].contiguous()
                w_k_t = conv_weight_f32_t[:, proj_size : 2 * proj_size].contiguous()
                w_v_t = conv_weight_f32_t[:, 2 * proj_size :].contiguous()
                conv_bias = torch.zeros(conv_channels, dtype=torch.float32, device=device)
                onorm_weight = torch.randn(head_v_dim, dtype=torch.float32, device=device)
                conv_states = conv_pool  # [B, d_conv-1, conv_channels]

                def run_fused_decode():
                    kda_fused_decode(
                        mixed_qkv,
                        a,
                        b,
                        conv_states,
                        w_q_t,
                        w_k_t,
                        w_v_t,
                        conv_bias,
                        a_log,
                        dt_bias,
                        onorm_g,
                        onorm_weight,
                        recurrent_state,
                        state_indices,
                        scale=head_k_dim**-0.5,
                        onorm_eps=1e-6,
                        lower_bound=KDA_LOWER_BOUND,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_fused_decode,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                ) as results:
                    if not log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="SGLang",
                        version=sglang_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="kda",
                        kernel_source="kda_fused_decode",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    ):
                        raise RuntimeError(f"failed to persist SGLang KDA generation row to {perf_filename}")
                successful_points += 1
                continue

            def run_conv1d_update():
                causal_conv1d_update(
                    mixed_qkv,
                    conv_pool.transpose(-1, -2),
                    conv_weight,
                    None,
                    activation="silu",
                    conv_state_indices=state_indices,
                )

            with benchmark_with_power(
                device=device,
                kernel_func=run_conv1d_update,
                num_warmups=num_warmups,
                num_runs=num_runs,
                repeat_n=1,
            ) as results:
                if not log_perf(
                    item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                    framework="SGLang",
                    version=sglang_version,
                    device_name=torch.cuda.get_device_name(device),
                    op_name="kda",
                    kernel_source="causal_conv1d_update",
                    perf_filename=perf_filename,
                    power_stats=results["power_stats"],
                ):
                    raise RuntimeError(f"failed to persist SGLang KDA generation row to {perf_filename}")

            def run_kda_packed_decode():
                fused_recurrent_kda_packed_decode(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    scale=head_k_dim**-0.5,
                    initial_state=recurrent_state,
                    out=output,
                    ssm_state_indices=state_indices,
                    use_qk_l2norm_in_kernel=True,
                    lower_bound=KDA_LOWER_BOUND,
                )

            with benchmark_with_power(
                device=device,
                kernel_func=run_kda_packed_decode,
                num_warmups=num_warmups,
                num_runs=num_runs,
                repeat_n=1,
            ) as results:
                if not log_perf(
                    item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                    framework="SGLang",
                    version=sglang_version,
                    device_name=torch.cuda.get_device_name(device),
                    op_name="kda",
                    kernel_source="fused_recurrent_kda_packed_decode",
                    perf_filename=perf_filename,
                    power_stats=results["power_stats"],
                ):
                    raise RuntimeError(f"failed to persist SGLang KDA generation row to {perf_filename}")
            successful_points += 1

        except Exception as e:
            failed_points += 1
            failures.append(f"batch_size={batch_size}: {type(e).__name__}: {e}")
            print(f"  Error at batch_size={batch_size}: {e}")
            continue
        finally:
            a = a_log = b = conv_pool = dt_bias = mixed_qkv = None
            output = recurrent_state = state_indices = None
            _cleanup("generation")

    summary = f"ok={successful_points} error={failed_points} skip=0"
    print(f"KDA generation summary: {summary}")
    if failed_points or successful_points == 0:
        raise RuntimeError(
            f"SGLang KDA generation collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def _resolve_verify_kernel(head_k_dim: int, head_v_dim: int, draft_tokens: int):
    """Mirror serving's DSPARK verify dispatch on this device
    (kda_backend._can_run_dspark_cutedsl_mtp, kda_backend.py:858-938 @ kimi-k3
    branch): the fused CuTeDSL kernel engages on SM100 with cutlass importable,
    a fixed dense chain-verify width of 2..8 tokens per request and the K3
    128-dim symmetric bf16 head geometry; everything else runs the Triton pair.

    Returns the fused kernel callable, or None for the Triton fallback pair.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        return None
    if not (2 <= draft_tokens <= 8 and head_k_dim == 128 and head_v_dim == 128):
        return None
    import importlib.util

    try:
        if importlib.util.find_spec("cutlass") is None:
            return None
    except ModuleNotFoundError:
        return None
    from sglang.kernels.ops.kimi_k3.kda_decode_mtp import fused_kda_decode_mtp_dspark

    return fused_kda_decode_mtp_dspark


def run_kda_verify_benchmark(
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    draft_token_list: list[int],
    model_name: str,
    perf_filename: str,
    sglang_version: str,
    device: str = "cuda:0",
):
    """
    Benchmark KDA speculative target-verify kernels: the packed conv update
    over batch*draft_tokens rows and the fused chain-verify recurrence
    (fused_sigmoid_gating_delta_rule_update, is_kda=True,
    disable_state_update=True with intermediate state caching), mirroring
    kda_backend._forward_target_verify → TritonKDAKernel.target_verify.

    seq_len in the persisted row carries the per-request draft-token count
    (dspark block size + 1).
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim, "KDA requires symmetric Q/K/V head geometry"
    proj_size = num_v_heads * head_v_dim
    conv_channels = 3 * proj_size

    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    successful_points = 0
    failed_points = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        for draft_tokens in draft_token_list:
            total_tokens = batch_size * draft_tokens
            if aic_debug:
                print(f"  Benchmarking batch_size={batch_size}, draft_tokens={draft_tokens}")

            a = a_log = b = conv_pool = cu_seqlens = dt_bias = intermediate_states = None
            mixed_qkv = q = k = v = recurrent_state = state_indices = token_state_indices = None
            try:
                num_warmups = 3
                num_runs = 10
                cu_seqlens = torch.arange(0, total_tokens + 1, draft_tokens, dtype=torch.int32, device=device)
                state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
                mixed_qkv = torch.randn(total_tokens, conv_channels, dtype=dtype, device=device)
                conv_pool = torch.zeros(batch_size, d_conv - 1, conv_channels, dtype=dtype, device=device)
                token_state_indices = (
                    torch.arange(total_tokens, dtype=torch.int32, device=device) // draft_tokens
                ).contiguous()
                recurrent_state = torch.zeros(
                    batch_size, num_v_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device
                )
                intermediate_states = torch.zeros(
                    batch_size,
                    draft_tokens,
                    num_v_heads,
                    head_v_dim,
                    head_k_dim,
                    dtype=torch.float32,
                    device=device,
                )
                q = torch.randn(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device)
                k = torch.randn(1, total_tokens, num_k_heads, head_k_dim, dtype=dtype, device=device)
                v = torch.randn(1, total_tokens, num_v_heads, head_v_dim, dtype=dtype, device=device)
                a = torch.randn(1, total_tokens, num_v_heads, head_k_dim, dtype=dtype, device=device)
                b = torch.randn(1, total_tokens, num_v_heads, dtype=dtype, device=device)
                a_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
                dt_bias = torch.ones(num_v_heads * head_k_dim, dtype=torch.float32, device=device)

                common_log_data = {
                    "phase": "verify",
                    "batch_size": batch_size,
                    "seq_len": draft_tokens,
                    "num_tokens": total_tokens,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                fused_verify_kernel = _resolve_verify_kernel(head_k_dim, head_v_dim, draft_tokens)
                if fused_verify_kernel is not None:
                    proj = proj_size
                    x_q_r, x_k_r, x_v_r = (
                        blk.reshape(1, total_tokens, num_v_heads, head_v_dim)
                        for blk in mixed_qkv.split([proj] * 3, dim=-1)
                    )
                    # Serving conv weights are fp32 (checked by
                    # _can_run_dspark_cutedsl_mtp); split per Q/K/V block.
                    conv_weight_f32 = torch.randn(conv_channels, d_conv, dtype=torch.float32, device=device)
                    w_q, w_k, w_v = conv_weight_f32.split([proj] * 3, dim=0)
                    # Committed conv windows per slot, [slots, proj, d_conv-1]
                    # (the pool's transposed per-block view, kda_backend.py:1004-1013).
                    cs_q, cs_k, cs_v = (c.transpose(-1, -2) for c in conv_pool.split([proj] * 3, dim=-1))
                    # Per-draft-token scratch: state snapshots + conv windows
                    # (MambaPool.SpeculativeState layouts, mirrored from the
                    # branch's own kernel test).
                    ic_q = torch.zeros(batch_size, draft_tokens, proj, d_conv - 1, dtype=dtype, device=device)
                    ic_k = torch.zeros_like(ic_q)
                    ic_v = torch.zeros_like(ic_q)
                    # intermediate_states allocated above is [B, draft, HV, V, K] fp32.
                    # Serving folds gated RMSNorm into the kernel only where the
                    # fused-decode static args exist — the TP8 12-head shard
                    # (_prepare_fused_decode seg = 12*128, models/kimi_k3.py:1522-1560).
                    onorm_kwargs = {}
                    if num_v_heads == 12:
                        onorm_kwargs = {
                            "onorm_gate": torch.randn(
                                1, total_tokens, num_v_heads, head_v_dim, dtype=dtype, device=device
                            ),
                            "onorm_weight": torch.randn(head_v_dim, dtype=torch.float32, device=device),
                            "onorm_eps": 1e-6,
                        }

                    # FIXME(kernel-limit): fused_kda_decode_mtp_dspark hit
                    # cudaErrorIllegalAddress at (batch=256, draft_tokens=8,
                    # 96 heads) on B200/SM100 (2026-07-28) while every smaller
                    # cell passed; suspected per-SM resource growth in the
                    # persistent CuTe kernel (its block-size cap is documented
                    # as shared-memory-bound, kda_backend.py:879-880).
                    # Unverified against the kernel source; the cell fails
                    # into the classified log meanwhile.
                    def run_fused_verify():
                        fused_verify_kernel(
                            x_q=x_q_r,
                            x_k=x_k_r,
                            x_v=x_v_r,
                            w_q=w_q,
                            w_k=w_k,
                            w_v=w_v,
                            cs_q=cs_q,
                            cs_k=cs_k,
                            cs_v=cs_v,
                            g=a,
                            beta=b,
                            A_log=a_log,
                            dt_bias=dt_bias,
                            recurrent_state=recurrent_state,
                            intermediate_ssm=intermediate_states,
                            intermediate_state_indices=state_indices,
                            intermediate_conv_q=ic_q,
                            intermediate_conv_k=ic_k,
                            intermediate_conv_v=ic_v,
                            ssm_state_indices=state_indices,
                            cu_seqlens=cu_seqlens,
                            lower_bound=KDA_LOWER_BOUND,
                            scale=head_k_dim**-0.5,
                            **onorm_kwargs,
                        )

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_fused_verify,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                    ) as results:
                        if not log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="SGLang",
                            version=sglang_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="kda",
                            kernel_source="fused_kda_decode_mtp_dspark",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        ):
                            raise RuntimeError(f"failed to persist SGLang KDA verify row to {perf_filename}")
                    successful_points += 1
                    continue

                def run_conv1d_update_verify():
                    causal_conv1d_update(
                        mixed_qkv,
                        conv_pool.transpose(-1, -2),
                        conv_weight,
                        None,
                        activation="silu",
                        conv_state_indices=token_state_indices,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv1d_update_verify,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                ) as results:
                    if not log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="SGLang",
                        version=sglang_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="kda",
                        kernel_source="causal_conv1d_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    ):
                        raise RuntimeError(f"failed to persist SGLang KDA verify row to {perf_filename}")

                def run_kda_target_verify():
                    fused_sigmoid_gating_delta_rule_update(
                        A_log=a_log,
                        dt_bias=dt_bias,
                        q=q,
                        k=k,
                        v=v,
                        a=a,
                        b=b,
                        initial_state_source=recurrent_state,
                        initial_state_indices=state_indices,
                        cu_seqlens=cu_seqlens,
                        use_qk_l2norm_in_kernel=True,
                        softplus_beta=1.0,
                        softplus_threshold=20.0,
                        is_kda=True,
                        disable_state_update=True,
                        intermediate_states_buffer=intermediate_states,
                        intermediate_state_indices=state_indices,
                        cache_steps=draft_tokens,
                        retrieve_parent_token=None,
                        lower_bound=KDA_LOWER_BOUND,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_kda_target_verify,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                ) as results:
                    if not log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="SGLang",
                        version=sglang_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="kda",
                        kernel_source="fused_sigmoid_gating_delta_rule_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    ):
                        raise RuntimeError(f"failed to persist SGLang KDA verify row to {perf_filename}")
                successful_points += 1

            except Exception as e:
                failed_points += 1
                failures.append(f"batch_size={batch_size} draft_tokens={draft_tokens}: {type(e).__name__}: {e}")
                print(f"  Error at batch_size={batch_size}, draft_tokens={draft_tokens}: {e}")
                continue
            finally:
                a = a_log = b = conv_pool = cu_seqlens = dt_bias = intermediate_states = None
                mixed_qkv = q = k = v = recurrent_state = state_indices = token_state_indices = None
                _cleanup("verify")

    summary = f"ok={successful_points} error={failed_points} skip=0"
    print(f"KDA verify summary: {summary}")
    if failed_points or successful_points == 0:
        raise RuntimeError(
            f"SGLang KDA verify collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


# ---------------------------------------------------------------------------
# GLM-5.3-Flash (Glm5NextForConditionalGeneration) KDA — SGLang 0.5.20.
#
# SGLang serves GLM-5.3-Flash KDA through srt/models/glm5_next.py
# ::Glm5NextLinearAttention -> RadixLinearAttention -> KDAAttnBackend
# (srt/layers/attention/linear/kda_backend.py). Source audit (image
# lmsysorg/sglang@sha256:b0d8718a..., /sgl-workspace/sglang, _version.py
# 0.5.20):
#   - arg_groups/fields/exec_.py:397-418  --linear-attn-backend defaults to
#     "triton"; prefill/decode/verify overrides default to None.
#   - arg_groups/attention_hook.py:212-297  the only automatic override is
#     decode -> flashinfer on SM100 when --mamba-ssm-dtype is bfloat16; the
#     GLM baseline keeps the fp32 recurrent state, so every KDA mode stays on
#     TritonKDAKernel (kda_backend.py:73,136).
#   - glm5_next.py:340  do_fuse_qkvbfg is False whenever a quant_config is
#     present (both the FP8 and the NVFP4 checkpoints), so q|k|v, b, f_a/f_b
#     and g_a/g_b are separate GEMMs and mixed_qkv is a dense [T, 3P] tensor.
#   - glm5_next.py:437-448,475-489  fp32 merged q|k|v conv weight [3P, d_conv],
#     no bias, fp32 A_log [1,1,H,1] / dt_bias [P], lower_bound from config.
#   - glm5_next.py:541-552  prefill passes a=[1,T,P] (flat per-K gate) and
#     b=[1,T,H] raw beta; decode passes a=[B,P] and b=[1,B,H].
#   - kda_backend.py:795-908 forward_extend: ONE packed causal_conv1d_fn over
#     q|k|v (:843-853, has_initial_state = extend_prefix_lens > 0 at :822),
#     then chunk_kda with A_log/dt_bias/lower_bound, beta_is_raw=True
#     (flat gate, :860) via TritonKDAKernel.extend (kda_triton.py:219-254).
#   - kda_backend.py:545-793 forward_decode: GLM never stashes the K3
#     fused-decode args (glm5_next.py has no _k3_* attributes), so decode is
#     ONE packed causal_conv1d_update (:729-736) and — because lower_bound is
#     set, the packed T=1 kernel is skipped (:739-742) — the
#     fused_sigmoid_gating_delta_rule_update(is_kda=True, lower_bound) decode
#     (:775-788 -> kda_triton.py:125-157).
#   - configs/glm5_next.py:264-272 + configs/mamba_utils.py:299-321  pool
#     layouts: conv [slots, d_conv-1, 3P] (bf16), temporal [slots, H, V, K]
#     (fp32).
#
# Row boundary: each row times the kernels between the projections and the
# output norm. The projection GEMMs, the gated output RMSNorm (o_norm) and
# the TP all-reduce are NOT in any kda row.
# ---------------------------------------------------------------------------

# Serving architecture of these model paths is Glm5NextForConditionalGeneration.
GLM5_NEXT_KDA_MODEL_PATHS = frozenset({"zai-org/GLM-5.3-Flash", "nvidia/GLM-5.3-Flash-NVFP4"})
# config.json text_config.linear_attn_config.gate_lower_bound (both checkpoints).
GLM5_NEXT_KDA_LOWER_BOUND = -5.0


def _is_glm5_next_kda(model_name: str) -> bool:
    return model_name in GLM5_NEXT_KDA_MODEL_PATHS


def _glm5_next_common(phase, batch_size, seq_len, d_model, d_conv, nh, hd, model_name):
    return {
        "phase": phase,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_tokens": batch_size * seq_len,
        "d_model": d_model,
        "d_conv": d_conv,
        "num_k_heads": nh,
        "head_k_dim": hd,
        "num_v_heads": nh,
        "head_v_dim": hd,
        "model_name": model_name,
    }


def _glm5_next_log(common, results, kernel_source, perf_filename, sglang_version, device):
    if not log_perf(
        item_list=[{**common, "latency": results["latency_ms"]}],
        framework="SGLang",
        version=sglang_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="kda",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    ):
        raise RuntimeError(f"failed to persist SGLang GLM-5.3-Flash KDA row to {perf_filename}")


def _glm5_next_layer(num_slots, nh, hd, d_conv, device):
    """Per-layer parameters and a non-zero state pool (rows model requests
    whose prefix is already cached, i.e. has_initial_state=True)."""
    proj = nh * hd
    conv_weight = torch.randn(3 * proj, d_conv, dtype=torch.float32, device=device)
    a_log = torch.zeros(1, 1, nh, 1, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(proj, dtype=torch.float32, device=device)
    conv_pool = 0.1 * torch.randn(num_slots, d_conv - 1, 3 * proj, dtype=torch.bfloat16, device=device)
    ssm_states = 0.01 * torch.randn(num_slots, nh, hd, hd, dtype=torch.float32, device=device)
    return conv_weight, a_log, dt_bias, conv_pool, ssm_states


def run_glm5_next_kda_context(
    d_model,
    d_conv,
    nh,
    hd,
    batch_size_list,
    seq_len_list,
    model_name,
    perf_filename,
    sglang_version,
    device="cuda:0",
):
    """GLM-5.3-Flash extend (every request has a cached prefix): packed q|k|v
    causal_conv1d_fn row and the chunk_kda row (kda_backend.forward_extend)."""
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    proj = nh * hd
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            nt = batch_size * seq_len
            conv_weight = a_log = dt_bias = conv_pool = ssm_states = None
            mixed_qkv = a = b = conv_out = q = k = v = None
            try:
                conv_weight, a_log, dt_bias, conv_pool, ssm_states = _glm5_next_layer(
                    batch_size, nh, hd, d_conv, device
                )
                cu = torch.arange(0, nt + 1, seq_len, dtype=torch.int32, device=device)
                idx = torch.arange(batch_size, dtype=torch.int32, device=device)
                has_init = torch.ones(batch_size, dtype=torch.bool, device=device)
                seq_lens_cpu = [seq_len] * batch_size
                conv_states = conv_pool.transpose(-1, -2)  # kda_backend.py:813
                mixed_qkv = torch.randn(nt, 3 * proj, dtype=dtype, device=device)
                a = torch.randn(1, nt, proj, dtype=dtype, device=device)  # f_b_proj out, unsqueezed
                b = torch.randn(1, nt, nh, dtype=dtype, device=device)  # b_proj out, unsqueezed
                common = _glm5_next_common("context", batch_size, seq_len, d_model, d_conv, nh, hd, model_name)

                def run_conv():
                    # kda_backend.py:843-853
                    return causal_conv1d_fn(
                        mixed_qkv.transpose(0, 1),
                        conv_weight,
                        None,
                        activation="silu",
                        conv_states=conv_states,
                        has_initial_state=has_init,
                        cache_indices=idx,
                        query_start_loc=cu,
                        seq_lens_cpu=seq_lens_cpu,
                    ).transpose(0, 1)

                with benchmark_with_power(
                    device=device, kernel_func=run_conv, num_warmups=3, num_runs=10, repeat_n=1
                ) as results:
                    _glm5_next_log(common, results, "causal_conv1d_fn", perf_filename, sglang_version, device)

                conv_out = run_conv()
                q, k, v = (x.unflatten(-1, (-1, hd)).unsqueeze(0) for x in conv_out.split(proj, dim=-1))
                g = a.unflatten(-1, (-1, hd))  # kda_backend.py:860-862

                def run_chunk():
                    # kda_backend.py:893-906 -> kda_triton.py:237-254
                    chunk_kda(
                        q=q,
                        k=k,
                        v=v,
                        g=g,
                        beta=b,
                        initial_state=ssm_states,
                        initial_state_indices=idx,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=cu,
                        A_log=a_log,
                        dt_bias=dt_bias,
                        lower_bound=GLM5_NEXT_KDA_LOWER_BOUND,
                        beta_is_raw=True,
                        output_intermediate_states=False,
                        track_state=None,
                        track_chunk_idx=None,
                    )

                with benchmark_with_power(
                    device=device, kernel_func=run_chunk, num_warmups=3, num_runs=10, repeat_n=1
                ) as results:
                    _glm5_next_log(common, results, "chunk_kda", perf_filename, sglang_version, device)
                ok += 1
            except Exception as e:
                err += 1
                failures.append(f"batch_size={batch_size} seq_len={seq_len}: {type(e).__name__}: {e}")
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue
            finally:
                conv_weight = a_log = dt_bias = conv_pool = ssm_states = None
                mixed_qkv = a = b = conv_out = q = k = v = None
                _cleanup("glm5_next context")

    summary = f"ok={ok} error={err} skip=0"
    print(f"GLM-5.3-Flash KDA context summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"SGLang GLM-5.3-Flash KDA context collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_glm5_next_kda_generation(
    d_model,
    d_conv,
    nh,
    hd,
    batch_size_list,
    model_name,
    perf_filename,
    sglang_version,
    device="cuda:0",
):
    """GLM-5.3-Flash decode: packed q|k|v causal_conv1d_update row and the
    bounded-gate fused_sigmoid_gating_delta_rule_update row
    (kda_backend.forward_decode, non-packed branch)."""
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    proj = nh * hd
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        conv_weight = a_log = dt_bias = conv_pool = ssm_states = None
        mixed_qkv = a = b = conv_out = q = k = v = None
        try:
            conv_weight, a_log, dt_bias, conv_pool, ssm_states = _glm5_next_layer(batch_size, nh, hd, d_conv, device)
            cu = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
            idx = torch.arange(batch_size, dtype=torch.int32, device=device)
            mixed_qkv = torch.randn(batch_size, 3 * proj, dtype=dtype, device=device)
            a = torch.randn(batch_size, proj, dtype=dtype, device=device)  # decode: not unsqueezed
            b = torch.randn(1, batch_size, nh, dtype=dtype, device=device)
            common = _glm5_next_common("generation", batch_size, 1, d_model, d_conv, nh, hd, model_name)

            def run_conv_update():
                # kda_backend.py:729-736
                return causal_conv1d_update(
                    mixed_qkv,
                    conv_pool.transpose(-1, -2),
                    conv_weight,
                    None,
                    activation="silu",
                    conv_state_indices=idx,
                )

            with benchmark_with_power(
                device=device, kernel_func=run_conv_update, num_warmups=3, num_runs=10, repeat_n=10
            ) as results:
                _glm5_next_log(common, results, "causal_conv1d_update", perf_filename, sglang_version, device)

            conv_out = run_conv_update()
            q, k, v = (x.unflatten(-1, (-1, hd)).unsqueeze(0) for x in conv_out.split(proj, dim=-1))

            def run_decode():
                # kda_backend.py:775-788 -> kda_triton.py:141-157
                fused_sigmoid_gating_delta_rule_update(
                    A_log=a_log,
                    dt_bias=dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    a=a,
                    b=b,
                    initial_state_source=ssm_states,
                    initial_state_indices=idx,
                    cu_seqlens=cu,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    is_kda=True,
                    lower_bound=GLM5_NEXT_KDA_LOWER_BOUND,
                )

            with benchmark_with_power(
                device=device, kernel_func=run_decode, num_warmups=3, num_runs=10, repeat_n=10
            ) as results:
                _glm5_next_log(
                    common, results, "fused_sigmoid_gating_delta_rule_update", perf_filename, sglang_version, device
                )
            ok += 1
        except Exception as e:
            err += 1
            failures.append(f"batch_size={batch_size}: {type(e).__name__}: {e}")
            print(f"  Error at batch_size={batch_size}: {e}")
            continue
        finally:
            conv_weight = a_log = dt_bias = conv_pool = ssm_states = None
            mixed_qkv = a = b = conv_out = q = k = v = None
            _cleanup("glm5_next generation")

    summary = f"ok={ok} error={err} skip=0"
    print(f"GLM-5.3-Flash KDA generation summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"SGLang GLM-5.3-Flash KDA generation collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_glm5_next_kda_torch(
    phase, d_model, d_conv, num_k_heads, head_k_dim, num_v_heads, head_v_dim, batch_size_list, seq_len_list, **kwargs
):
    """Phase router for GLM-5.3-Flash KDA rows (called from run_kda_torch).

    SGLang runs every non-verify extend batch — one-token requests included —
    through forward_extend, so context seq_len=1 cells stay on the prefill
    kernels (unlike vLLM)."""
    if num_k_heads != num_v_heads or head_k_dim != head_v_dim:
        raise ValueError("GLM-5.3-Flash KDA has symmetric q/k/v heads (glm5_next.py:326-331)")
    shape = dict(d_model=d_model, d_conv=d_conv, nh=num_v_heads, hd=head_v_dim)
    if phase == "context":
        if not seq_len_list:
            raise ValueError("SGLang KDA context collection requires at least one sequence length")
        run_glm5_next_kda_context(batch_size_list=batch_size_list, seq_len_list=seq_len_list, **shape, **kwargs)
    elif phase == "generation":
        run_glm5_next_kda_generation(batch_size_list=batch_size_list, **shape, **kwargs)
    elif phase == "verify":
        # GLM-5.3-Flash is modeled with nextn=0 (MTP off); its target-verify
        # dispatch (kda_backend.py:955-1203, fused chain-verify gate) has not
        # been audited for collection.
        raise NotImplementedError(
            "GLM-5.3-Flash KDA verify (MTP target-verify) is not collected: the GLM "
            "baseline runs without speculative decoding"
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")


def run_kda_torch(
    phase: str,
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    seq_len_list: list[int] | None,
    model_name: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """
    Main entry point for KDA benchmarking using SGLang's Triton FLA kernels.

    Routes to the appropriate benchmark function based on phase.
    Imports the target SGLang kernels at runtime. GLM-5.3-Flash rows are
    routed to the glm5_next dispatch (run_glm5_next_kda_torch).
    """
    import contextlib
    from importlib.metadata import version as _get_version

    if _is_glm5_next_kda(model_name):
        run_glm5_next_kda_torch(
            phase,
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            batch_size_list,
            seq_len_list,
            model_name=model_name,
            perf_filename=perf_filename,
            sglang_version=_get_version("sglang"),
            device=device,
        )
        return WORKER_RESTART

    with (
        open(os.devnull, "w") as _devnull_file,
        contextlib.redirect_stdout(_devnull_file),
        contextlib.redirect_stderr(_devnull_file),
    ):
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_kda_packed_decode,
        )
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
        from sglang.kernels.ops.attention.fla.kda import chunk_kda
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn, causal_conv1d_update

    sglang_version = _get_version("sglang")

    globals().update(
        {
            "causal_conv1d_fn": causal_conv1d_fn,
            "causal_conv1d_update": causal_conv1d_update,
            "chunk_kda": chunk_kda,
            "fused_recurrent_kda_packed_decode": fused_recurrent_kda_packed_decode,
            "fused_sigmoid_gating_delta_rule_update": fused_sigmoid_gating_delta_rule_update,
        }
    )

    if phase == "context":
        run_kda_context_benchmark(
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            seq_len_list=seq_len_list,
            model_name=model_name,
            perf_filename=perf_filename,
            sglang_version=sglang_version,
            device=device,
        )
    elif phase == "generation":
        run_kda_generation_benchmark(
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            model_name=model_name,
            perf_filename=perf_filename,
            sglang_version=sglang_version,
            device=device,
        )
    elif phase == "verify":
        run_kda_verify_benchmark(
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            draft_token_list=seq_len_list,
            model_name=model_name,
            perf_filename=perf_filename,
            sglang_version=sglang_version,
            device=device,
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return WORKER_RESTART


if __name__ == "__main__":
    import sys
    from importlib.metadata import version as _get_ver

    from collector.registry_types import PerfFile

    print(f"KDA Collector - SGLang {_get_ver('sglang')}")
    print(f"SM Version: {get_sm_version()}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    test_cases = get_kda_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    last_exit_code = 0
    for i, test_case in enumerate(test_cases):
        (
            phase,
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            batch_size_list,
            seq_len_list,
            model_name,
        ) = test_case

        print(f"\n[{i + 1}/{len(test_cases)}] {model_name} - {phase}")
        print(
            f"  d_model={d_model}, num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}, d_conv={d_conv}"
        )
        print(f"  batch_sizes={batch_size_list}")
        if seq_len_list is not None:
            print(f"  seq_lens={seq_len_list}")

        last_exit_code = run_kda_torch(
            phase=phase,
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            seq_len_list=seq_len_list,
            model_name=model_name,
            perf_filename=PerfFile.KDA,
        )

    sys.exit(last_exit_code)
