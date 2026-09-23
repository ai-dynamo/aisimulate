# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "trtllm>=1.3.0rc20"

"""
GDN (Gated DeltaNet) Collector for AIConfigurator — TensorRT-LLM backend.

TRT-LLM added Qwen3.5 support (PR #12242, merged) using FLA-derived GDN kernels.
This collector benchmarks the same GDN operations as the vLLM and SGLang collectors,
using TRT-LLM's bundled causal_conv1d and vendored FLA kernels for GDN scan/update.

Context (prefill) phase:
    - causal_conv1d_fn: Causal 1D convolution over key channels
    - chunk_gated_delta_rule: GDN chunked scan (Q, K, V, g, beta) via vendored FLA

Generation (decode) phase:
    - causal_conv1d_update: Single-step conv state update
    - fused_recurrent_gated_delta_rule: Single-step GDN recurrence via vendored FLA

The in_proj and out_proj GEMMs are standard linear layers modeled by the
existing GEMM infrastructure. This collector focuses on the unique GDN ops.

GDN Layer Flow:
    in_proj (GEMM) → Conv1D (keys) → GDN Scan/Update → out_proj (GEMM)
    ^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
    Use GEMM model          Benchmarked here            Use GEMM model

Usage:
    python collect_gdn.py

Output:
    gdn_perf.txt - Performance data for GDN Conv1D + scan operations
"""

import gc
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
    from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule
    from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

import torch

try:
    from case_generator import get_common_gdn_test_cases

    from helper import (
        EXIT_CODE_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_gdn_test_cases

    from helper import (
        EXIT_CODE_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )

aic_debug = int(os.getenv("aic_gdn_debug", "0"))  # noqa: SIM112
# Use cached inputs (same data each iteration) instead of randomized inputs
aic_cached_inputs = int(os.getenv("AIC_GDN_CACHED_INPUTS", "0"))


def get_gdn_test_cases():
    """
    Generate test cases for GDN kernel benchmarking.

    Returns a list of test case configurations for both context (prefill)
    and generation (decode) phases.
    """
    test_cases = []

    for common_case in get_common_gdn_test_cases():
        if common_case.phase == "context":
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
        else:
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
                    None,  # seq_len_list not used for generation
                    common_case.model_name,
                ]
            )

    return test_cases


def _make_input_pool(
    shapes: dict[str, tuple[int, ...]],
    count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    """Pre-generate a pool of random input tensors for randomized benchmarking."""
    return {
        name: [torch.randn(*shape, dtype=dtype, device=device) for _ in range(count)] for name, shape in shapes.items()
    }


def run_gdn_context_benchmark(
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
    trtllm_version: str,
    device: str = "cuda:0",
):
    """
    Benchmark GDN operations for context (prefill) phase using TRT-LLM runtime.

    Benchmarks:
    1. causal_conv1d_fn  — Conv1D over key channels (TRT-LLM bundled)
    2. chunk_gated_delta_rule — GDN scan (Q, K, V, g, beta) via vendored FLA
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    # key channels only go through conv; q/v bypass conv
    conv_channels = num_k_heads * head_k_dim

    if aic_debug:
        print(
            f"GDN Context: d_model={d_model}, num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}, d_conv={d_conv}"
        )

    # Conv weights (key channels)
    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    conv_bias = torch.randn(conv_channels, dtype=dtype, device=device)

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            if aic_debug:
                print(f"  Benchmarking batch_size={batch_size}, seq_len={seq_len}")

            try:
                num_warmups = 3
                num_runs = 10
                total_iters = num_warmups + num_runs

                # Conv state: (batch, channels, d_conv - 1)
                conv_state = torch.randn(batch_size, conv_channels, d_conv - 1, dtype=dtype, device=device)

                common_log_data = {
                    "phase": "context",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_tokens": batch_size * seq_len,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                if aic_cached_inputs:
                    k_input = torch.randn(batch_size, conv_channels, seq_len, dtype=dtype, device=device)
                    q = torch.randn(batch_size, seq_len, num_k_heads, head_k_dim, dtype=dtype, device=device)
                    k = torch.randn(batch_size, seq_len, num_k_heads, head_k_dim, dtype=dtype, device=device)
                    v = torch.randn(batch_size, seq_len, num_v_heads, head_v_dim, dtype=dtype, device=device)
                    g = torch.nn.functional.logsigmoid(
                        torch.randn(batch_size, seq_len, num_v_heads, dtype=dtype, device=device)
                    )
                    beta = torch.sigmoid(torch.randn(batch_size, seq_len, num_v_heads, dtype=dtype, device=device))

                    # --- Benchmark causal_conv1d_fn ---
                    torch.cuda.synchronize()
                    causal_conv1d_fn(k_input, conv_weight, conv_bias, activation="silu", conv_states=conv_state)
                    torch.cuda.synchronize()

                    def run_conv1d(_ki=k_input, _cs=conv_state):
                        causal_conv1d_fn(_ki, conv_weight, conv_bias, activation="silu", conv_states=_cs)

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_conv1d,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="TRTLLM",
                            version=trtllm_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="gdn",
                            kernel_source="causal_conv1d_fn",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                    # --- Benchmark chunk_gated_delta_rule ---
                    torch.cuda.synchronize()
                    chunk_gated_delta_rule(q, k, v, g, beta)
                    torch.cuda.synchronize()

                    def run_gdn_scan(_q=q, _k=k, _v=v, _g=g, _beta=beta):
                        chunk_gated_delta_rule(_q, _k, _v, _g, _beta)

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_gdn_scan,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="TRTLLM",
                            version=trtllm_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="gdn",
                            kernel_source=_CHUNK_KERNEL_SOURCE,
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                else:
                    input_pool = _make_input_pool(
                        {
                            "k_input": (batch_size, conv_channels, seq_len),
                            "q": (batch_size, seq_len, num_k_heads, head_k_dim),
                            "k": (batch_size, seq_len, num_k_heads, head_k_dim),
                            "v": (batch_size, seq_len, num_v_heads, head_v_dim),
                            "g": (batch_size, seq_len, num_v_heads),
                            "beta": (batch_size, seq_len, num_v_heads),
                        },
                        total_iters,
                        dtype,
                        device,
                    )
                    for i in range(total_iters):
                        input_pool["g"][i] = torch.nn.functional.logsigmoid(input_pool["g"][i])
                        input_pool["beta"][i] = torch.sigmoid(input_pool["beta"][i])

                    # --- Benchmark causal_conv1d_fn ---
                    torch.cuda.synchronize()
                    causal_conv1d_fn(
                        input_pool["k_input"][0], conv_weight, conv_bias, activation="silu", conv_states=conv_state
                    )
                    torch.cuda.synchronize()

                    conv1d_iter_idx = [0]

                    def run_conv1d(_pool=input_pool, _cs=conv_state, _idx=conv1d_iter_idx):
                        idx = _idx[0] % total_iters
                        _idx[0] += 1
                        causal_conv1d_fn(
                            _pool["k_input"][idx], conv_weight, conv_bias, activation="silu", conv_states=_cs
                        )

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_conv1d,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="TRTLLM",
                            version=trtllm_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="gdn",
                            kernel_source="causal_conv1d_fn",
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                    # --- Benchmark chunk_gated_delta_rule ---
                    torch.cuda.synchronize()
                    chunk_gated_delta_rule(
                        input_pool["q"][0],
                        input_pool["k"][0],
                        input_pool["v"][0],
                        input_pool["g"][0],
                        input_pool["beta"][0],
                    )
                    torch.cuda.synchronize()

                    gdn_iter_idx = [0]

                    def run_gdn_scan(_pool=input_pool, _idx=gdn_iter_idx):
                        idx = _idx[0] % total_iters
                        _idx[0] += 1
                        chunk_gated_delta_rule(
                            _pool["q"][idx],
                            _pool["k"][idx],
                            _pool["v"][idx],
                            _pool["g"][idx],
                            _pool["beta"][idx],
                        )

                    with benchmark_with_power(
                        device=device,
                        kernel_func=run_gdn_scan,
                        num_warmups=num_warmups,
                        num_runs=num_runs,
                        repeat_n=1,
                        allow_graph_fail=True,
                    ) as results:
                        log_perf(
                            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                            framework="TRTLLM",
                            version=trtllm_version,
                            device_name=torch.cuda.get_device_name(device),
                            op_name="gdn",
                            kernel_source=_CHUNK_KERNEL_SOURCE,
                            perf_filename=perf_filename,
                            power_stats=results["power_stats"],
                        )

                # Cleanup
                if aic_cached_inputs:
                    del k_input, q, k, v, g, beta, conv_state
                else:
                    del input_pool, conv_state
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue


def _bench_serving_decode_entry(
    *,
    batch_size: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    dtype,
    device,
    common_log_data: dict,
    trtllm_version: str,
    perf_filename: str,
    num_warmups: int,
    num_runs: int,
) -> None:
    """Measure the entry serving actually calls for pure decode.

    gdn_mixer.forward_decode (_torch/modules/mamba/gdn_mixer.py:670 @1.3.0rc23)
    calls fused_sigmoid_gating_delta_rule_update with the raw gating inputs
    (A_log, dt_bias, a, b), the SSM state pool + slot indices and
    cu_seqlens; that function dispatches to FlashInfer's bf16-state GDN decode
    kernel when _can_use_flashinfer_gdn_decode holds (SM90/SM100, K==V==128,
    bf16 state, one token per sequence) and to the FLA Triton kernel otherwise
    (fla/fused_sigmoid_gating_recurrent.py:205-238,452-458). kernel_source
    records which branch ran. Builds without this entry (pre-rc23) skip it —
    their serving truth is the fused_recurrent row above.
    """
    fn = globals().get("fused_sigmoid_gating_delta_rule_update")
    gate = globals().get("_can_use_flashinfer_gdn_decode")
    if fn is None:
        return
    B, H, K, V = batch_size, num_v_heads, head_k_dim, head_v_dim
    q = torch.randn(1, B, num_k_heads, K, dtype=dtype, device=device)
    k = torch.randn(1, B, num_k_heads, K, dtype=dtype, device=device)
    v = torch.randn(1, B, H, V, dtype=dtype, device=device)
    a = torch.randn(B, H, dtype=dtype, device=device)
    b = torch.randn(B, H, dtype=dtype, device=device)
    A_log = torch.zeros(H, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(H, dtype=torch.float32, device=device)
    ssm_states = torch.randn(B, H, K, V, dtype=torch.bfloat16, device=device)
    slot_indices = torch.arange(B, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(B + 1, dtype=torch.int64, device=device)
    uses_flashinfer = bool(gate(ssm_states, K, V, B, B)) if gate is not None else False

    def run_serving_decode():
        fn(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=ssm_states, initial_state_indices=slot_indices,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0, softplus_threshold=20.0,
        )

    torch.cuda.synchronize()
    run_serving_decode()
    torch.cuda.synchronize()
    with benchmark_with_power(
        device=device,
        kernel_func=run_serving_decode,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
        allow_graph_fail=True,
    ) as results:
        log_perf(
            item_list=[{**common_log_data, "latency": results["latency_ms"]}],
            framework="TRTLLM",
            version=trtllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name="gdn",
            kernel_source=("fused_sigmoid_gating_delta_rule_update_flashinfer" if uses_flashinfer
                           else "fused_sigmoid_gating_delta_rule_update_triton"),
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )


def run_gdn_generation_benchmark(
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    batch_size_list: list[int],
    model_name: str,
    perf_filename: str,
    trtllm_version: str,
    device: str = "cuda:0",
):
    """
    Benchmark GDN operations for generation (decode) phase using TRT-LLM runtime.

    Benchmarks:
    1. causal_conv1d_update  — Single-step conv state update (TRT-LLM bundled)
    2. fused_recurrent_gated_delta_rule — Single-step GDN recurrence via vendored FLA
    """
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dtype = torch.bfloat16

    conv_channels = num_k_heads * head_k_dim

    if aic_debug:
        print(
            f"GDN Generation: d_model={d_model}, num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_heads={num_v_heads}, head_v_dim={head_v_dim}, d_conv={d_conv}"
        )

    conv_weight = torch.randn(conv_channels, d_conv, dtype=dtype, device=device)
    conv_bias = torch.randn(conv_channels, dtype=dtype, device=device)

    for batch_size in batch_size_list:
        if aic_debug:
            print(f"  Benchmarking batch_size={batch_size}")

        try:
            num_warmups = 3
            num_runs = 10
            total_iters = num_warmups + num_runs

            conv_state = torch.randn(batch_size, conv_channels, d_conv - 1, dtype=dtype, device=device)
            # GDN state: [batch, num_v_heads, head_k_dim, head_v_dim] stored as BF16
            gdn_state = torch.randn(batch_size, num_v_heads, head_k_dim, head_v_dim, dtype=dtype, device=device)

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

            # serving's pure-decode entry (dispatch-labeled); the rows below keep
            # the conv update and the FLA recurrent fallback lane
            _bench_serving_decode_entry(
                batch_size=batch_size, num_k_heads=num_k_heads, head_k_dim=head_k_dim,
                num_v_heads=num_v_heads, head_v_dim=head_v_dim, dtype=dtype, device=device,
                common_log_data=common_log_data, trtllm_version=trtllm_version,
                perf_filename=perf_filename, num_warmups=num_warmups, num_runs=num_runs,
            )

            if aic_cached_inputs:
                k_input = torch.randn(batch_size, conv_channels, dtype=dtype, device=device)
                q = torch.randn(batch_size, 1, num_k_heads, head_k_dim, dtype=dtype, device=device)
                k = torch.randn(batch_size, 1, num_k_heads, head_k_dim, dtype=dtype, device=device)
                v = torch.randn(batch_size, 1, num_v_heads, head_v_dim, dtype=dtype, device=device)
                g = torch.nn.functional.logsigmoid(torch.randn(batch_size, 1, num_v_heads, dtype=dtype, device=device))
                beta = torch.sigmoid(torch.randn(batch_size, 1, num_v_heads, dtype=dtype, device=device))

                # --- Benchmark causal_conv1d_update ---
                torch.cuda.synchronize()
                causal_conv1d_update(k_input, conv_state, conv_weight, conv_bias, activation="silu")
                torch.cuda.synchronize()

                def run_conv1d_update(_k=k_input, _cs=conv_state):
                    causal_conv1d_update(_k, _cs, conv_weight, conv_bias, activation="silu")

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv1d_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="TRTLLM",
                        version=trtllm_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="causal_conv1d_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

                # --- Benchmark fused_recurrent_gated_delta_rule ---
                torch.cuda.synchronize()
                fused_recurrent_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=gdn_state,
                    output_final_state=True,
                )
                torch.cuda.synchronize()

                def run_gdn_update(_q=q, _k=k, _v=v, _g=g, _beta=beta, _state=gdn_state):
                    fused_recurrent_gated_delta_rule(
                        _q,
                        _k,
                        _v,
                        _g,
                        _beta,
                        initial_state=_state,
                        output_final_state=True,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_gdn_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="TRTLLM",
                        version=trtllm_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="fused_recurrent_gated_delta_rule",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

            else:
                input_pool = _make_input_pool(
                    {
                        "k_input": (batch_size, conv_channels),
                        "q": (batch_size, 1, num_k_heads, head_k_dim),
                        "k": (batch_size, 1, num_k_heads, head_k_dim),
                        "v": (batch_size, 1, num_v_heads, head_v_dim),
                        "g": (batch_size, 1, num_v_heads),
                        "beta": (batch_size, 1, num_v_heads),
                    },
                    total_iters,
                    dtype,
                    device,
                )
                for i in range(total_iters):
                    input_pool["g"][i] = torch.nn.functional.logsigmoid(input_pool["g"][i])
                    input_pool["beta"][i] = torch.sigmoid(input_pool["beta"][i])

                # --- Benchmark causal_conv1d_update ---
                torch.cuda.synchronize()
                causal_conv1d_update(input_pool["k_input"][0], conv_state, conv_weight, conv_bias, activation="silu")
                torch.cuda.synchronize()

                conv1d_iter_idx = [0]

                def run_conv1d_update(_pool=input_pool, _cs=conv_state, _idx=conv1d_iter_idx):
                    idx = _idx[0] % total_iters
                    _idx[0] += 1
                    causal_conv1d_update(_pool["k_input"][idx], _cs, conv_weight, conv_bias, activation="silu")

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv1d_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="TRTLLM",
                        version=trtllm_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="causal_conv1d_update",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

                # --- Benchmark fused_recurrent_gated_delta_rule ---
                torch.cuda.synchronize()
                fused_recurrent_gated_delta_rule(
                    input_pool["q"][0],
                    input_pool["k"][0],
                    input_pool["v"][0],
                    input_pool["g"][0],
                    input_pool["beta"][0],
                    initial_state=gdn_state,
                    output_final_state=True,
                )
                torch.cuda.synchronize()

                gdn_iter_idx = [0]

                def run_gdn_update(_pool=input_pool, _state=gdn_state, _idx=gdn_iter_idx):
                    idx = _idx[0] % total_iters
                    _idx[0] += 1
                    fused_recurrent_gated_delta_rule(
                        _pool["q"][idx],
                        _pool["k"][idx],
                        _pool["v"][idx],
                        _pool["g"][idx],
                        _pool["beta"][idx],
                        initial_state=_state,
                        output_final_state=True,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_gdn_update,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                    allow_graph_fail=True,
                ) as results:
                    log_perf(
                        item_list=[{**common_log_data, "latency": results["latency_ms"]}],
                        framework="TRTLLM",
                        version=trtllm_version,
                        device_name=torch.cuda.get_device_name(device),
                        op_name="gdn",
                        kernel_source="fused_recurrent_gated_delta_rule",
                        perf_filename=perf_filename,
                        power_stats=results["power_stats"],
                    )

            # Cleanup
            if aic_cached_inputs:
                del k_input, q, k, v, g, beta, conv_state, gdn_state
            else:
                del input_pool, conv_state, gdn_state
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error at batch_size={batch_size}: {e}")
            continue


def run_gdn_torch(
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
    Main entry point for GDN benchmarking using TRT-LLM runtime.

    Routes to appropriate benchmark function based on phase.
    Uses TRT-LLM's bundled causal_conv1d and vendored FLA kernels for GDN scan/update.
    """
    import contextlib

    with (
        open(os.devnull, "w") as _devnull_file,
        contextlib.redirect_stdout(_devnull_file),
        contextlib.redirect_stderr(_devnull_file),
    ):
        import tensorrt_llm
        from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule
        from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

        # Serving dispatch, never the vendored kernel directly: the GDN mixer
        # resolves chunk_gated_delta_rule to FlashInfer's cute-DSL chunk kernel on
        # SM90/SM100 (TLLM_USE_FLASHINFER_GDN_PREFILL=1 default) and to the FLA
        # Triton kernel elsewhere (_torch/modules/mamba/gdn_mixer.py:57-70
        # @1.3.0rc23; arch gate _utils.is_flashinfer_gdn_supported_arch). Older
        # builds have no resolver and only the Triton kernel — import that.
        chunk_label = "chunk_gated_delta_rule"
        try:
            from tensorrt_llm._torch.modules.mamba.gdn_mixer import (
                _resolve_chunk_gated_delta_rule,
                chunk_gated_delta_rule,
            )

            if "flashinfer" in getattr(_resolve_chunk_gated_delta_rule(), "__module__", ""):
                chunk_label = "chunk_gated_delta_rule_flashinfer"
        except ImportError:
            from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
        # Pure-decode serving entry (gdn_mixer.py:670): fused sigmoid gating +
        # recurrent update, itself dispatching to FlashInfer's bf16-state GDN
        # decode kernel when _can_use_flashinfer_gdn_decode holds
        # (fla/fused_sigmoid_gating_recurrent.py:205-238,452-458 @1.3.0rc23).
        try:
            from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
                _can_use_flashinfer_gdn_decode,
                fused_sigmoid_gating_delta_rule_update,
            )
        except ImportError:
            _can_use_flashinfer_gdn_decode = None
            fused_sigmoid_gating_delta_rule_update = None

    globals().update(
        {
            "tensorrt_llm": tensorrt_llm,
            "causal_conv1d_fn": causal_conv1d_fn,
            "causal_conv1d_update": causal_conv1d_update,
            "chunk_gated_delta_rule": chunk_gated_delta_rule,
            "_CHUNK_KERNEL_SOURCE": chunk_label,
            "fused_recurrent_gated_delta_rule": fused_recurrent_gated_delta_rule,
            "fused_sigmoid_gating_delta_rule_update": fused_sigmoid_gating_delta_rule_update,
            "_can_use_flashinfer_gdn_decode": _can_use_flashinfer_gdn_decode,
        }
    )

    trtllm_version = tensorrt_llm.__version__

    if phase == "context":
        run_gdn_context_benchmark(
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
            trtllm_version=trtllm_version,
            device=device,
        )
    elif phase == "generation":
        run_gdn_generation_benchmark(
            d_model=d_model,
            d_conv=d_conv,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            batch_size_list=batch_size_list,
            model_name=model_name,
            perf_filename=perf_filename,
            trtllm_version=trtllm_version,
            device=device,
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Return EXIT_CODE_RESTART to signal that a process restart would be
    # desirable for GPU memory cleanup.  collect.py's orchestrator previously
    # relied on this function calling sys.exit(EXIT_CODE_RESTART) directly,
    # which killed the worker process after each task so the OS reclaimed GPU
    # memory before the next task started.  That also prevented the __main__
    # for-loop from completing more than one case when run standalone.
    #
    # The sys.exit has been moved outside the loop in __main__ so that all
    # test cases run in sequence.  When invoked via collect.py the worker
    # process no longer restarts between GDN tasks; if GPU OOM is observed in
    # that path, restoring per-task process recycling here would fix it.
    return EXIT_CODE_RESTART


if __name__ == "__main__":
    import sys

    import tensorrt_llm
    from collector.registry_types import PerfFile

    print(f"GDN Collector - TensorRT-LLM {tensorrt_llm.__version__}")
    print(f"SM Version: {get_sm_version()}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    test_cases = get_gdn_test_cases()
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

        if phase == "context":
            print(f"  batch_sizes={batch_size_list}")
            print(f"  seq_lens={seq_len_list}")
        else:
            print(f"  batch_sizes={batch_size_list}")

        last_exit_code = run_gdn_torch(
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
            perf_filename=PerfFile.GDN,
        )

    sys.exit(last_exit_code)
