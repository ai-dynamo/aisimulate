# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure vLLM Mamba2 convolution, scan, and FlashInfer state updates.

Calls follow vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6 (v0.31.0),
model_executor/layers/mamba/mamba_mixer2.py and ops/ssu_dispatch.py.
Prefill uses the Triton scan even when the decode backend is FlashInfer.
The registry keeps this collector unverified until a GPU qualification run.
"""

__compat__ = "vllm==0.31.0"

from itertools import cycle, product
from math import ceil
from types import SimpleNamespace

from collector.case_generator import _required_base_common_case_values, get_common_mamba2_test_cases


def _decode_query_start_loc(token_width: int, starts):
    """Decode cumulative lengths exactly as serving passes them.

    vLLM db9527a4 leaves ``query_start_loc_d`` None for non-speculative decode
    (v1/attention/backends/mamba_attn.py:533,599-601) and forwards it to
    ``causal_conv1d_update`` and ``selective_state_update``
    (model_executor/layers/mamba/mamba_mixer2.py:1043,1150). FlashInfer then
    runs the single-token kernel for width one and the MTP kernel otherwise.
    """
    return starts if token_width > 1 else None


def get_mamba2_test_cases():
    sweep = _required_base_common_case_values("mamba2")
    cases = []
    for case in get_common_mamba2_test_cases():
        widths = [1] if case.phase == "context" else sweep["vllm_token_widths"]
        lengths = case.seq_len_list if case.phase == "context" else [1]
        for tp, width, batch, length in product(case.vllm_tensor_parallel_sizes, widths, case.batch_size_list, lengths):
            # vLLM's constructor rejects these tensor partitions (v0.31.0).
            if case.nheads % tp or (case.n_groups % tp and case.n_groups != 1):
                raise ValueError(f"Invalid declared Mamba partition: {case.model_name}, TP{tp}")
            cases.append(
                [
                    case.phase,
                    case.d_model,
                    case.d_state,
                    case.d_conv,
                    case.nheads // tp,
                    case.head_dim,
                    max(1, case.n_groups // tp),
                    case.chunk_size,
                    tp,
                    width,
                    batch,
                    length,
                    case.model_name,
                    sweep["vllm_state_dtype"],
                    sweep["vllm_stochastic_rounding"],
                    sweep["vllm_philox_rounds"],
                ]
            )
    return cases


def run_mamba2(
    phase,
    d_model,
    d_state,
    d_conv,
    nheads,
    head_dim,
    n_groups,
    chunk_size,
    tp,
    token_width,
    batch_size,
    seq_len,
    model_name,
    state_dtype,
    stochastic_rounding,
    philox_rounds,
    perf_filename,
    device="cuda:0",
):
    import torch
    from collector.helper import benchmark_with_power, log_perf
    from vllm.config.mamba import MambaConfig
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined_varlen
    from vllm.model_executor.layers.mamba.ops.ssu_dispatch import FlashInferSSUBackend
    from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.version import __version__

    if phase not in ("context", "generation") or token_width < 1:
        raise ValueError("Mamba collection requires context or generation and positive token_width")
    if phase == "context" and token_width != 1:
        raise ValueError("Prefill token_width must be one")
    if state_dtype not in ("float16", "bfloat16", "float32") or philox_rounds not in (5, 10):
        raise ValueError("Invalid Mamba state dtype or Philox rounds")
    if stochastic_rounding and state_dtype != "float16":
        raise ValueError("Stochastic rounding requires float16 state")
    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    state_type = getattr(torch, state_dtype)
    length = seq_len if phase == "context" else token_width
    tokens = batch_size * length
    conv_dim = nheads * head_dim + 2 * n_groups * d_state
    rand = lambda *shape: torch.randn(*shape, dtype=dtype, device=device)
    # mamba_mixer2.py:709-713 slices X/B/C and dt from the packed projection.
    projected = rand(tokens, nheads * head_dim + conv_dim + nheads)
    xbc, dt = projected[:, nheads * head_dim :].split([conv_dim, nheads], dim=-1)
    weights, bias = rand(conv_dim, d_conv), rand(conv_dim)
    # Block zero is reserved for padding in vLLM.
    slots = batch_size * token_width + 1
    indices = torch.arange(1, slots, dtype=torch.int32, device=device).reshape(batch_size, token_width)
    # mamba_mixer2.py:733-740 uses SD storage unless DS is explicitly selected.
    conv_state = rand(slots, d_conv - 1 + token_width - 1, conv_dim).transpose(-1, -2)
    # Rotate recurrent states beyond L2 so graph replay does not time one hot state.
    state_bytes = (
        batch_size * token_width * nheads * head_dim * d_state * torch.empty((), dtype=state_type).element_size()
    )
    state_repeats = (
        max(1, ceil(2 * torch.cuda.get_device_properties(device).L2_cache_size / state_bytes))
        if phase == "generation"
        else 1
    )
    state = torch.randn(1 + (slots - 1) * state_repeats, nheads, head_dim, d_state, device=device, dtype=state_type)
    state_indices = cycle([indices + repeat * (slots - 1) for repeat in range(state_repeats)])
    a = -torch.rand(nheads, device=device) - 1
    d = torch.ones(nheads, dtype=dtype, device=device)
    dt_bias = torch.full((nheads,), -4.0, dtype=dtype, device=device)
    out = torch.empty(tokens, nheads, head_dim, dtype=dtype, device=device)
    starts_cpu = torch.arange(0, tokens + 1, length, dtype=torch.int32, device="cpu")
    starts = starts_cpu.to(device)
    row = dict(
        phase=phase,
        batch_size=batch_size,
        seq_len=seq_len,
        num_tokens=tokens,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        nheads=nheads,
        head_dim=head_dim,
        n_groups=n_groups,
        chunk_size=chunk_size,
        tp=tp,
        token_width=token_width,
        state_dtype=state_dtype,
        stochastic_rounding=stochastic_rounding,
        philox_rounds=philox_rounds if stochastic_rounding else 0,
        model_name=model_name,
        use_replayssm=False,
        conv_state_layout="SD",
    )
    if phase == "context":
        nums, batch_ptr, offsets = compute_causal_conv1d_metadata(starts_cpu, device=device)
        metadata = SimpleNamespace(nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=offsets)
        has_initial = torch.ones(batch_size, dtype=torch.bool, device=device)
        cu_chunks, last_chunks, seq_idx = compute_varlen_chunk_metadata(starts, chunk_size)

        def conv():
            return causal_conv1d_fn(
                xbc.T,
                weights,
                bias,
                conv_state,
                starts,
                cache_indices=indices[:, 0],
                has_initial_state=has_initial,
                activation="silu",
                metadata=metadata,
            )

        def ssm():
            # mamba_mixer2.py:885-890 includes the cached-state gather and mask.
            initial = torch.where(has_initial[:, None, None, None], state[indices[:, 0]], 0)
            final = mamba_chunk_scan_combined_varlen(
                x,
                dt,
                a,
                b,
                c,
                chunk_size,
                starts,
                cu_chunks,
                last_chunks,
                seq_idx,
                out,
                D=d,
                dt_bias=dt_bias,
                initial_states=initial,
                dt_softplus=True,
                state_dtype=state_type,
            )
            state[indices[:, 0]] = final

        sources = ("vllm_causal_conv1d_fn", "vllm_mamba_chunk_scan_combined_varlen")
    else:
        accepted = torch.ones(batch_size, dtype=torch.int32, device=device) if token_width > 1 else None
        decode_starts = _decode_query_start_loc(token_width, starts)
        backend = FlashInferSSUBackend(
            MambaConfig(
                backend="flashinfer",
                enable_stochastic_rounding=stochastic_rounding,
                stochastic_rounding_philox_rounds=philox_rounds,
            )
        )
        # mamba_mixer2.py:1053-1063 preserves the zero-stride A expansion.
        a_decode = a[:, None, None].expand(nheads, head_dim, d_state)
        dt_decode = dt[:, :, None].expand(tokens, nheads, head_dim)
        d_decode = d[:, None].expand(nheads, head_dim)
        bias_decode = dt_bias[:, None].expand(nheads, head_dim)

        def conv():
            return causal_conv1d_update(
                xbc,
                conv_state,
                weights,
                bias,
                "silu",
                conv_state_indices=indices,
                num_accepted_tokens=accepted,
                query_start_loc=decode_starts,
                max_query_len=token_width,
            )

        def ssm():
            current_indices = next(state_indices)
            backend(
                state,
                x,
                dt_decode,
                a_decode,
                b,
                c,
                d_decode,
                bias_decode,
                dt_softplus=True,
                state_batch_indices=current_indices,
                dst_state_batch_indices=current_indices,
                out=out,
                num_accepted_tokens=accepted,
                cu_seqlens=decode_starts,
                is_blackwell=torch.cuda.get_device_capability(device)[0] == 10,
            )

        sources = ("vllm_causal_conv1d_update", "vllm_flashinfer_selective_state_update")
    # mamba_mixer2.py:871-873,1047-1049 keeps the convolution output strides.
    convolved = conv()
    packed = convolved.T if phase == "context" else convolved
    x, b, c = packed.split([nheads * head_dim, n_groups * d_state, n_groups * d_state], dim=-1)
    x = x.view(tokens, nheads, head_dim)
    b, c = b.view(tokens, n_groups, d_state), c.view(tokens, n_groups, d_state)
    for source, kernel in zip(sources, (conv, ssm), strict=True):
        kernel()
        torch.cuda.synchronize(device)
        repeats = state_repeats if source == "vllm_flashinfer_selective_state_update" else 1
        with benchmark_with_power(
            device=device, kernel_func=kernel, num_warmups=3, num_runs=10, repeat_n=repeats
        ) as result:
            log_perf(
                item_list=[{**row, "state_buffer_repetitions": repeats, "latency": result["latency_ms"]}],
                framework="vLLM",
                version=__version__,
                device_name=torch.cuda.get_device_name(device),
                op_name="mamba2",
                kernel_source=source,
                perf_filename=perf_filename,
                power_stats=result["power_stats"],
            )
