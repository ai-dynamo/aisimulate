# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KDA (Kimi Delta Attention) Collector for AIConfigurator — vLLM backend.

Kimi-K3 support lives on the vLLM `kimi-k3` branch (official preview image
vllm/vllm-openai:kimi-k3, version 0.1.dev19262+gb6bbf29dd, CUDA 13 build).
The K3 KDA layer (vllm/models/kimi_k3/nvidia/kda.py) dispatches differently
from SGLang's Triton-only path, so this collector mirrors vLLM's own
dispatch on the target SM and records the actually-invoked kernel:

Context (prefill) phase (KimiK3DeltaAttention._forward prefill branch):
    - causal_conv1d_fn x3: separate Q/K/V causal convolutions
      (kernel_source "causal_conv1d_fn_qkv3")
    - prefill core, dispatched like serving (resolve_kda_prefill_backend):
        * "flashkda_fwd" — FlashKDA CUDA extension (vllm._flashkda_C),
          the SM90/SM100/SM120 default for bf16 head_dim 128 with a gate
          lower bound; or
        * "chunk_kda_with_fused_gate" — Triton fallback.

Generation (decode) phase:
    - "fused_kda_decode" — the CUDA fused conv+recurrence+gated-RMSNorm
      decode kernel (CUDA>=13 builds; probed via is_fused_kda_decode_supported
      exactly like serving). This is the NOSPEC serving fast path and
      includes the conv update and output norm.
    - "causal_conv1d_update" + "fused_recurrent_kda_packed_decode" — the
      fallback pair, and the path serving uses whenever speculative decoding
      is enabled (num_spec != 0 permanently disables the fused kernel).

Verify (speculative target-verify, DSPARK/MTP) phase:
    - "causal_conv1d_update" (spec form: query_start_loc + num_accepted_tokens)
    - "fused_recurrent_kda" — chain verify with per-draft-token state
      checkpointing via 2-hd ssm_state_indices [num_seqs, num_spec_tokens].

The in_proj/out_proj/gate GEMMs are standard linear layers modeled by the
existing GEMM infrastructure. Tensor constructions mirror the branch's own
tests (tests/models/kimi_k3/test_kda.py).

GLM-5.3-Flash (Glm5NextForConditionalGeneration, vLLM 0.30.0) rows are
routed by model path to the glm5next layer's own dispatch (see the GLM
section): one merged q|k|v conv ("causal_conv1d_fn" / "causal_conv1d_update"),
the prefill core ("flashkda_fwd" = state gather + FlashKDA + state scatter,
or the Triton "chunk_kda_with_fused_gate") and the decode recurrence
("fused_recurrent_kda", in-kernel bounded gate).

Output:
    kda_perf.txt — same column layout as the sglang kda collector.
"""

# Serve-parity `metadata=` argument validated on the 0.27.0 release image
# (Kimi-K3 landed upstream there). The manifest kda family pin stays on the
# kimi-k3 preview build until the SDK consumer routes 0.27.0's split decode
# (see collector/framework_manifest.yaml), so this module exactly pins the
# preview version; it runs on either image — the DS-layout probe
# (is_fused_kda_decode_supported) yields fused_kda_decode rows on the
# preview and the packed conv-update + recurrence pair on 0.27.0.
# The file-level range spans the two audited routes (PEP 440 clauses are
# conjunctive, so it cannot name the two releases alone). Each KDA
# architecture is additionally gated at runtime to its own audited release by
# _KDA_ARCHITECTURE_COMPAT: Kimi-K3 only on the 0.1.dev19262 preview, GLM-5.3-
# Flash only on 0.30.0 (installed 0.30.0+glm53tail.eb4704514fdf); every
# release in between raises KdaRuntimeNotAuditedError.
__compat__ = "vllm>=0.1.dev19262,<=0.30.0"

import gc
import os

import torch

try:
    from collector.case_generator import get_common_kda_test_cases
    from collector.helper import (
        WORKER_RESTART,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
    from collector.version_resolver import _check_compat
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_kda_test_cases
    from version_resolver import _check_compat

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

NUM_WARMUPS = 3
NUM_RUNS = 10


def get_kda_test_cases():
    """Test cases for KDA kernel benchmarking (context/generation/verify)."""
    test_cases = []
    for c in get_common_kda_test_cases():
        test_cases.append(
            [
                c.phase,
                c.d_model,
                c.d_conv,
                c.num_k_heads,
                c.head_k_dim,
                c.num_v_heads,
                c.head_v_dim,
                c.batch_size_list,
                c.seq_len_list,
                c.model_name,
            ]
        )
    return test_cases


def _cleanup(tag: str):
    cleanup_errors = []
    for name, fn in (("gc.collect", gc.collect), ("torch.cuda.empty_cache", torch.cuda.empty_cache)):
        try:
            fn()
        except Exception as e:
            cleanup_errors.append(f"{name}: {type(e).__name__}: {e}")
    if cleanup_errors:
        raise RuntimeError(f"vLLM KDA {tag} cleanup failed: {'; '.join(cleanup_errors)}")


def _log(common, latency_ms, kernel_source, perf_filename, vllm_version, device, power_stats):
    if not log_perf(
        item_list=[{**common, "latency": latency_ms}],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="kda",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise RuntimeError(f"failed to persist vLLM KDA row to {perf_filename}")


def _resolve_prefill_kernel(dtype: torch.dtype):
    """Mirror serving's resolve_kda_prefill_backend on this device: FlashKDA
    when supported AND importable, else the Triton chunk kernel
    (vllm/models/kimi_k3/nvidia/kda.py resolve_kda_prefill_backend)."""
    from vllm.models.kimi_k3.nvidia.kda import is_flashkda_supported

    if is_flashkda_supported(128, dtype, KDA_LOWER_BOUND):
        try:
            import vllm._flashkda_C  # noqa: F401
            from vllm.models.kimi_k3.nvidia.kda import _flashkda_prefill

            return "flashkda_fwd", _flashkda_prefill
        except ImportError:
            pass
    return "chunk_kda_with_fused_gate", None


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
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    seq_len_list,
    model_name,
    perf_filename,
    vllm_version,
    device="cuda:0",
):
    """Context (prefill): 3-way Q/K/V causal conv + prefill core kernel,
    dispatched like KimiK3DeltaAttention._forward on this SM."""
    if any(seq_len <= 1 for seq_len in seq_len_list):
        raise ValueError(
            "vLLM KDA context collection requires seq_len > 1; "
            "run_kda_torch routes seq_len=1 through the serving decode path"
        )

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim
    nh, hd, cw = num_v_heads, head_v_dim, d_conv
    proj = nh * hd

    prefill_source, flashkda_prefill = _resolve_prefill_kernel(dtype)
    conv_weight = torch.randn(3 * proj, cw, dtype=torch.float32, device=device)
    q_w, k_w, v_w = conv_weight.split([proj] * 3, dim=0)
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            nt = batch_size * seq_len
            try:
                # Resolved at the 0.27.0 era bump: the former FIXME
                # kernel-limit guard (nt * proj >= 2**31 rejected) is
                # deleted. 0.27.0's causal_conv1d keeps token addressing in
                # int64 end-to-end
                # (vllm/model_executor/layers/mamba/ops/causal_conv1d.py:40,48
                # `stride_x_token: tl.int64` plus explicit .to(tl.int64)
                # casts), establishing no int32 token-offset limit; GB300
                # silicon at 0.27.0 confirms the two tightest formerly-vetoed
                # cells pass with SOL-sane timings (nv12 bs64 s32768: conv
                # 3-way 15.8ms for ~116GB traffic ≈ 7.3TB/s; nv96 bs8 s32768:
                # 14.3ms for the same traffic — both within the 8TB/s byte
                # model margin).
                cu = torch.arange(0, nt + 1, seq_len, dtype=torch.int32, device=device)
                idx = torch.arange(batch_size, dtype=torch.int32, device=device)
                has_init = torch.zeros(batch_size, dtype=torch.bool, device=device)
                conv_state = torch.zeros(batch_size, 3 * proj, cw - 1, dtype=dtype, device=device)
                q_cs, k_cs, v_cs = conv_state.split([proj] * 3, dim=-2)
                mixed = torch.randn(nt, 3 * proj, dtype=dtype, device=device)
                q_in, k_in, v_in = mixed.transpose(0, 1).split([proj] * 3, dim=0)

                common = {
                    "phase": "context",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_tokens": nt,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                # Serving precomputes the conv metadata once per step in the
                # KDA attention-metadata builder (KimiK3KDAMetadata subclasses
                # GDNAttentionMetadata;
                # vllm/models/kimi_k3/nvidia/kda_metadata.py @0.27.0 image era)
                # and passes it into every layer's prefill conv call
                # (vllm/models/kimi_k3/nvidia/kda.py::_prefill_conv passes
                # metadata=m). Omitting metadata takes the non-serving branch:
                # causal_conv1d_fn recomputes nums/offset lists with
                # np.repeat + nums.sum().item() (D2H) inside EVERY timed
                # call, adding ~0.25ms host overhead per conv (a flat
                # ~0.8ms/iter floor on GB300 that dominated the 0.1.dev19262
                # re-collect, conv1d SOL ~15%). Build it per-shape and pass
                # it through like collect_gdn.py does. The entry point routes
                # seq_len=1 through the decode benchmark before reaching this
                # function, matching KimiK3KDAMetadataBuilder's
                # split_decodes_and_prefills(..., decode_threshold=1).
                # Pinned serving provenance at vLLM v0.27.0:
                # - kda_metadata.py:261-279 derives num_spec_decodes and all
                #   prefill/decode request and token counts.
                # - kda_metadata.py:403-418 computes nums_dict, batch_ptr, and
                #   token_chunk_offset_ptr from non_spec_query_start_loc_cpu
                #   only when num_prefills > 0.
                # - kda_metadata.py:466-485 constructs KimiK3KDAMetadata with
                #   those fields and num_actual_tokens=m.num_actual_tokens.
                nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
                    torch.arange(0, nt + 1, seq_len, dtype=torch.int32),
                    device=device,
                )
                conv_metadata = GDNAttentionMetadata(
                    num_prefills=batch_size,
                    num_prefill_tokens=nt,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_spec_decodes=0,
                    num_spec_decode_tokens=0,
                    num_actual_tokens=nt,
                    nums_dict=nums_dict,
                    batch_ptr=batch_ptr,
                    token_chunk_offset_ptr=token_chunk_offset_ptr,
                )

                def run_conv_qkv3():
                    for x, w, cs in ((q_in, q_w, q_cs), (k_in, k_w, k_cs), (v_in, v_w, v_cs)):
                        causal_conv1d_fn(
                            x,
                            w,
                            None,
                            conv_states=cs,
                            query_start_loc=cu,
                            cache_indices=idx,
                            has_initial_state=has_init,
                            activation="silu",
                            metadata=conv_metadata,
                        )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv_qkv3,
                    num_warmups=NUM_WARMUPS,
                    num_runs=NUM_RUNS,
                    # With the metadata handed in, production launches this op
                    # eagerly but back-to-back in a deep queue, so the row must
                    # hold GPU service time (same as the GDN precedent in
                    # collect_gdn.py — repeat graph replay, not a sync-bounded
                    # eager loop).
                    repeat_n=10,
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "causal_conv1d_fn_qkv3",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )

                q = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                k = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                v = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                raw_g = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                raw_beta = torch.randn(1, nt, nh, dtype=dtype, device=device)
                a_log = torch.zeros(nh, dtype=torch.float32, device=device)
                dt_bias = 0.1 * torch.randn(nh * hd, dtype=torch.float32, device=device)
                init_state = torch.zeros(batch_size, nh, hd, hd, dtype=torch.float32, device=device)

                if prefill_source == "flashkda_fwd":

                    def run_prefill():
                        flashkda_prefill(
                            q,
                            k,
                            v,
                            raw_g,
                            raw_beta,
                            a_log,
                            dt_bias,
                            KDA_LOWER_BOUND,
                            init_state,
                            cu,
                        )

                else:

                    def run_prefill():
                        chunk_kda_with_fused_gate(
                            q,
                            k,
                            v,
                            raw_g,
                            raw_beta,
                            a_log,
                            dt_bias,
                            initial_state=init_state,
                            output_final_state=True,
                            lower_bound=KDA_LOWER_BOUND,
                            use_qk_l2norm_in_kernel=True,
                            cu_seqlens=cu,
                        )

                with benchmark_with_power(
                    device=device, kernel_func=run_prefill, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=1
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        prefill_source,
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )
                ok += 1
            except Exception as e:
                err += 1
                failures.append(f"batch_size={batch_size} seq_len={seq_len}: {type(e).__name__}: {e}")
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue
            finally:
                # Drop the per-iteration tensor references before empty_cache
                # (same pattern as the sglang kda collector's finally blocks).
                cu = idx = has_init = conv_state = q_cs = k_cs = v_cs = None
                mixed = q_in = k_in = v_in = q = k = v = raw_g = raw_beta = None
                a_log = dt_bias = init_state = None
                _cleanup("context")

    summary = f"ok={ok} error={err} skip=0"
    print(f"KDA context summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"vLLM KDA context collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_kda_generation_benchmark(
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    model_name,
    perf_filename,
    vllm_version,
    row_phase="generation",
    device="cuda:0",
):
    """Generation (decode): the CUDA fused_kda_decode fast path (probed like
    serving) plus the packed conv-update + Triton recurrence fallback pair
    (which is also the serving path under speculative decoding).

    ``row_phase`` remains ``context`` for one-token cells originating from the
    shared context grid; the recorded kernels still follow vLLM's decode path.
    """
    if row_phase not in {"context", "generation"}:
        raise ValueError(f"Unsupported KDA decode row phase: {row_phase}")

    import vllm._custom_ops as ops
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.kimi_k3.nvidia.kda import is_fused_kda_decode_supported
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent_kda_packed_decode

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim
    nh, hd, cw = num_v_heads, head_v_dim, d_conv
    dim = nh * hd

    fused_ok = is_fused_kda_decode_supported(nh, hd, cw, num_spec=0, input_dtype=dtype, conv_state_dtype=dtype)
    conv_weight = torch.randn(3 * dim, cw, dtype=torch.float32, device=device)
    fused_weight = conv_weight.reshape(3, dim, cw).transpose(1, 2).contiguous()
    norm_weight = torch.randn(hd, dtype=torch.float32, device=device)
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        try:
            nb = batch_size
            x = torch.randn(nb, 3 * dim, dtype=dtype, device=device)
            # KDA conv cache: [slots, 3*dim, cw-1] with stride(1)==1 ("SD" layout)
            conv_state = torch.zeros(nb, cw - 1, 3 * dim, dtype=dtype, device=device).transpose(1, 2)
            raw_g = torch.randn(1, nb, nh, hd, dtype=dtype, device=device)
            raw_beta = torch.randn(1, nb, nh, dtype=dtype, device=device)
            a_log = torch.zeros(nh, dtype=torch.float32, device=device)
            dt_bias_hd = 0.1 * torch.randn(nh, hd, dtype=torch.float32, device=device)
            state = torch.zeros(nb, nh, hd, hd, dtype=torch.float32, device=device)
            idx = torch.arange(nb, dtype=torch.int32, device=device)
            output_gate = torch.randn(nb, nh, hd, dtype=dtype, device=device)
            conv_out = torch.empty_like(x)

            common = {
                "phase": row_phase,
                "batch_size": nb,
                "seq_len": 1,
                "num_tokens": nb,
                "d_model": d_model,
                "d_conv": d_conv,
                "num_k_heads": num_k_heads,
                "head_k_dim": head_k_dim,
                "num_v_heads": num_v_heads,
                "head_v_dim": head_v_dim,
                "model_name": model_name,
            }

            if fused_ok:

                def run_fused_decode():
                    ops.fused_kda_decode(
                        x=x,
                        weight=fused_weight,
                        bias=None,
                        conv_state=conv_state,
                        raw_g=raw_g,
                        raw_beta=raw_beta,
                        A_log=a_log,
                        dt_bias=dt_bias_hd.reshape(-1),
                        state_indices=idx,
                        state=state,
                        lower_bound=KDA_LOWER_BOUND,
                        output_gate=output_gate,
                        norm_weight=norm_weight,
                        norm_eps=1e-5,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_fused_decode,
                    num_warmups=NUM_WARMUPS,
                    num_runs=NUM_RUNS,
                    repeat_n=1,
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "fused_kda_decode",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )

            def run_conv_update():
                causal_conv1d_update(
                    x,
                    conv_state,
                    conv_weight,
                    None,
                    activation="silu",
                    conv_state_indices=idx,
                    validate_data=True,
                    out=conv_out,
                )

            with benchmark_with_power(
                device=device, kernel_func=run_conv_update, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=1
            ) as results:
                _log(
                    common,
                    results["latency_ms"],
                    "causal_conv1d_update",
                    perf_filename,
                    vllm_version,
                    device,
                    results["power_stats"],
                )

            def run_packed_decode():
                fused_recurrent_kda_packed_decode(
                    mixed_qkv=x,
                    raw_g=raw_g,
                    raw_beta=raw_beta,
                    A_log=a_log,
                    dt_bias=dt_bias_hd,
                    lower_bound=KDA_LOWER_BOUND,
                    initial_state=state,
                    state_indices=idx,
                )

            with benchmark_with_power(
                device=device, kernel_func=run_packed_decode, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=1
            ) as results:
                _log(
                    common,
                    results["latency_ms"],
                    "fused_recurrent_kda_packed_decode",
                    perf_filename,
                    vllm_version,
                    device,
                    results["power_stats"],
                )
            ok += 1
        except Exception as e:
            err += 1
            failures.append(f"batch_size={batch_size}: {type(e).__name__}: {e}")
            print(f"  Error at batch_size={batch_size}: {e}")
            continue
        finally:
            _cleanup("generation")

    summary = f"ok={ok} error={err} skip=0"
    print(f"KDA {row_phase} decode-path summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"vLLM KDA {row_phase} decode-path collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_kda_verify_benchmark(
    d_model,
    d_conv,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    batch_size_list,
    draft_token_list,
    model_name,
    perf_filename,
    vllm_version,
    device="cuda:0",
):
    """Speculative target-verify: spec-form packed conv update + the
    fused_recurrent_kda chain-verify kernel with per-draft-token state
    checkpointing (2-hd ssm_state_indices). num_accepted_tokens is set to the
    full draft width (cost is token-count dominated)."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent_kda

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    assert num_k_heads == num_v_heads and head_k_dim == head_v_dim
    nh, hd, cw = num_v_heads, head_v_dim, d_conv
    dim = nh * hd

    conv_weight = torch.randn(3 * dim, cw, dtype=torch.float32, device=device)
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        for ns in draft_token_list:
            nb = batch_size
            nt = nb * ns
            try:
                x = torch.randn(nt, 3 * dim, dtype=dtype, device=device)
                # spec conv cache carries num_spec extra columns
                conv_state = torch.zeros(nb, cw - 1 + ns, 3 * dim, dtype=dtype, device=device).transpose(1, 2)
                cu = torch.arange(0, nt + 1, ns, dtype=torch.int32, device=device)
                idx = torch.arange(nb, dtype=torch.int32, device=device)
                accepted = torch.full((nb,), ns, dtype=torch.int32, device=device)
                conv_out = torch.empty_like(x)

                q = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                k = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                v = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                raw_g = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                raw_beta = torch.randn(1, nt, nh, dtype=dtype, device=device)
                a_log = torch.zeros(nh, dtype=torch.float32, device=device)
                dt_bias_hd = 0.1 * torch.randn(nh, hd, dtype=torch.float32, device=device)
                state = torch.zeros(nb * (ns + 1), nh, hd, hd, dtype=torch.float32, device=device)
                ssm_idx = torch.arange(nb * ns, dtype=torch.int32, device=device).view(nb, ns)
                out = torch.empty(1, nt, nh, hd, dtype=dtype, device=device)

                common = {
                    "phase": "verify",
                    "batch_size": nb,
                    "seq_len": ns,
                    "num_tokens": nt,
                    "d_model": d_model,
                    "d_conv": d_conv,
                    "num_k_heads": num_k_heads,
                    "head_k_dim": head_k_dim,
                    "num_v_heads": num_v_heads,
                    "head_v_dim": head_v_dim,
                    "model_name": model_name,
                }

                def run_conv_update_spec():
                    causal_conv1d_update(
                        x,
                        conv_state,
                        conv_weight,
                        None,
                        activation="silu",
                        conv_state_indices=idx,
                        num_accepted_tokens=accepted,
                        query_start_loc=cu,
                        max_query_len=ns,
                        validate_data=False,
                        out=conv_out,
                    )

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_conv_update_spec,
                    num_warmups=NUM_WARMUPS,
                    num_runs=NUM_RUNS,
                    repeat_n=1,
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "causal_conv1d_update",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )

                def run_verify():
                    fused_recurrent_kda(
                        q=q,
                        k=k,
                        v=v,
                        raw_g=raw_g,
                        raw_beta=raw_beta,
                        A_log=a_log,
                        dt_bias=dt_bias_hd,
                        lower_bound=KDA_LOWER_BOUND,
                        initial_state=state,
                        cu_seqlens=cu,
                        ssm_state_indices=ssm_idx,
                        num_accepted_tokens=accepted,
                        out=out,
                    )

                with benchmark_with_power(
                    device=device, kernel_func=run_verify, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=1
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "fused_recurrent_kda",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )
                ok += 1
            except Exception as e:
                err += 1
                failures.append(f"batch_size={batch_size} draft_tokens={ns}: {type(e).__name__}: {e}")
                print(f"  Error at batch_size={batch_size}, draft_tokens={ns}: {e}")
                continue
            finally:
                _cleanup("verify")

    summary = f"ok={ok} error={err} skip=0"
    print(f"KDA verify summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"vLLM KDA verify collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


# ---------------------------------------------------------------------------
# GLM-5.3-Flash (Glm5NextForConditionalGeneration) KDA — vLLM 0.30.0.
#
# vLLM serves GLM-5.3-Flash KDA through its own layer,
# vllm/models/glm5next/nvidia/kda.py::Glm5NextLinearAttention (a
# GatedDeltaNetAttention subclass driven by GDNAttentionMetadata), NOT the
# Kimi-K3 layer above: one merged q|k|v causal conv instead of three, the
# glm5next FLA fork of fused_recurrent_kda (in-kernel bounded gate) for
# decode, and gather/scatter of the recurrent state around the prefill core.
# Source audit (GB300 image vllm/vllm-openai@sha256:4864d466..., installed
# 0.30.0+glm53tail.eb4704514fdf; glm5next/nvidia/kda.py sha256 37745b45...
# is byte-identical in the stock 0.30.0 image and the glm53tail overlay):
#   - kda.py:127-150  _resolve_kda_prefill_backend: FlashKDA on CUDA SM9x/10x/12x
#                     for bf16 + head_dim 128 + bounded gate, else Triton
#                     chunk_kda_with_fused_gate ("auto" is the serving default,
#                     kda.py:321-329). The collector asks this same function.
#   - kda.py:247-273,505-515  q/k/v conv weights are fp32 params merged once
#                     into one [3P, d_conv] weight; bias is None (bias=False).
#   - kda.py:493-497  conv state is (…, dim, width-1); SD layout is a
#                     transposed view of the pool (is_conv_state_dim_first).
#   - kda.py:569-582  prefill: ONE causal_conv1d_fn over q|k|v with the
#                     step GDNAttentionMetadata (metadata=...).
#   - kda.py:644-692  prefill core: gather_initial_states -> FlashKDA fwd
#                     (kda.py:355-398, .contiguous() copies of q/k/v/g, raw
#                     bf16 beta) or chunk_kda_with_fused_gate -> scatter_states.
#   - kda.py:583-596,693-720  decode: ONE causal_conv1d_update over q|k|v, then
#                     glm5next fused_recurrent_kda(compute_gate=True,
#                     sigmoid_beta=True, lower_bound) writing into the layer
#                     output buffer.
#   - gdn_attn.py:250-252  split_decodes_and_prefills(decode_threshold=1):
#                     one-token non-spec requests are decodes.
#   - gdn_attn.py:399-417  has_initial_state = num_computed_tokens > 0 and the
#                     causal-conv metadata from the CPU query_start_loc.
#   - mamba_utils.py:133-149,298-321  conv state bf16 (model dtype),
#                     recurrent state fp32 (mamba_ssm_cache_dtype auto),
#                     shapes (3P, d_conv-1) / (H, D, D).
#
# Row boundary: each row times the kernels between the projections and the
# output norm. The fused
# in_proj_qkvbfg_a, f_b_proj, g_b_proj and o_proj GEMMs, the gated output
# RMSNorm (o_norm, a vLLM CustomOp) and the TP all-reduce are NOT in any
# kda row.
# ---------------------------------------------------------------------------

# Serving architecture of these model paths is Glm5NextForConditionalGeneration
# (config.json "architectures"); vLLM routes it to glm5next/nvidia/kda.py.
GLM5_NEXT_KDA_MODEL_PATHS = frozenset({"zai-org/GLM-5.3-Flash", "nvidia/GLM-5.3-Flash-NVFP4"})
# config.json text_config.linear_attn_config.gate_lower_bound (both checkpoints).
GLM5_NEXT_KDA_LOWER_BOUND = -5.0


def _is_glm5_next_kda(model_name: str) -> bool:
    return model_name in GLM5_NEXT_KDA_MODEL_PATHS


KIMI_K3_KDA_ARCHITECTURE = "KimiK3ForConditionalGeneration"
GLM5_NEXT_KDA_ARCHITECTURE = "Glm5NextForConditionalGeneration"
# Per-architecture audited vllm releases for the KDA dispatch this module
# replicates; any other installed release raises KdaRuntimeNotAuditedError (a
# classified failure) instead of timing a possibly different kernel path.
_KDA_ARCHITECTURE_COMPAT = {
    KIMI_K3_KDA_ARCHITECTURE: "vllm==0.1.dev19262",
    GLM5_NEXT_KDA_ARCHITECTURE: "vllm==0.30.0",
}


class KdaRuntimeNotAuditedError(RuntimeError):
    """The installed vllm release is not audited for this architecture's KDA dispatch."""


def _kda_architecture(model_name: str) -> str:
    """GLM-5.3-Flash paths run the glm5next layer; every other KDA row runs the
    (unchanged) Kimi-K3 layer path of this module."""
    return GLM5_NEXT_KDA_ARCHITECTURE if _is_glm5_next_kda(model_name) else KIMI_K3_KDA_ARCHITECTURE


def _require_audited_runtime(model_name: str, runtime_version: str) -> None:
    architecture = _kda_architecture(model_name)
    compat = _KDA_ARCHITECTURE_COMPAT[architecture]
    if not _check_compat(compat, runtime_version):
        raise KdaRuntimeNotAuditedError(
            f"vllm {runtime_version} is not an audited KDA runtime for {architecture} (audited: {compat})"
        )


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


def _glm5_next_state_pool(num_slots, nh, hd, d_conv, device):
    """Allocate the per-layer KDA pool exactly like serving: shapes/dtypes from
    the framework's own calculators (mamba_utils.py:133-149,298-321) and the
    conv-state orientation from is_conv_state_dim_first (kda.py:493-497).
    Both states are filled with non-zero values: rows model requests whose
    prefix is already cached (has_initial_state=True)."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
        is_conv_state_dim_first,
    )

    conv_dtype, state_dtype = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")
    conv_shape, state_shape = MambaStateShapeCalculator.kda_state_shape(1, nh, hd, conv_kernel_size=d_conv)
    conv_pool = 0.1 * torch.randn(num_slots, *conv_shape, device=device).to(conv_dtype)
    conv_state = conv_pool if is_conv_state_dim_first() else conv_pool.transpose(-1, -2)
    recurrent_state = 0.01 * torch.randn(num_slots, *state_shape, dtype=state_dtype, device=device)
    return conv_state, recurrent_state


def _glm5_next_layer_params(nh, hd, d_conv, device):
    """fp32 merged conv weight [3P, d_conv] (kda.py:247-273,505-515), A_log
    [1,1,H,1] and dt_bias [P] fp32 (kda.py:241-243,279-282)."""
    proj = nh * hd
    conv_weight = torch.randn(3 * proj, d_conv, dtype=torch.float32, device=device).contiguous()
    a_log = torch.zeros(1, 1, nh, 1, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(proj, dtype=torch.float32, device=device)
    return conv_weight, a_log, dt_bias


def _glm5_next_projected(num_tokens, nh, hd, device):
    """in_proj_qkvbfg_a output [T, 3P + H + 2D] (kda.py:214-232) and its
    serving splits (kda.py:407-423): qkv [T, 3P] and raw bf16 beta [1, T, H]
    are strided column views of the one GEMM output."""
    proj = nh * hd
    projected = torch.randn(num_tokens, 3 * proj + nh + 2 * hd, dtype=torch.bfloat16, device=device)
    qkv, beta_raw, _f_a, _g_a = projected.split([3 * proj, nh, hd, hd], dim=-1)
    return qkv, beta_raw.unsqueeze(0)


def run_glm5_next_kda_context(
    d_model,
    d_conv,
    nh,
    hd,
    batch_size_list,
    seq_len_list,
    model_name,
    perf_filename,
    vllm_version,
    device="cuda:0",
):
    """GLM-5.3-Flash prefill (every request has a cached prefix): merged q|k|v
    causal_conv1d_fn row, then the prefill-core row (state gather + FlashKDA
    or Triton chunk + state scatter), dispatched by the framework's resolver."""
    if any(seq_len <= 1 for seq_len in seq_len_list):
        raise ValueError("GLM-5.3-Flash KDA context collection requires seq_len > 1 (seq_len=1 is a decode)")

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import gather_initial_states
    from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
    from vllm.models.glm5next.nvidia.kda import _cast_sigmoid, _resolve_kda_prefill_backend
    from vllm.models.glm5next.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    proj = nh * hd
    # Serving selection (kda.py:321-329 -> :127-150) with the default "auto".
    prefill_backend = _resolve_kda_prefill_backend("auto", hd, dtype, GLM5_NEXT_KDA_LOWER_BOUND)
    prefill_source = {"flashkda": "flashkda_fwd", "triton": "chunk_kda_with_fused_gate"}[prefill_backend]
    if prefill_backend == "flashkda":
        import vllm._flashkda_C  # noqa: F401  (kda.py:333-334 imports it at layer init)
    conv_weight, a_log, dt_bias = _glm5_next_layer_params(nh, hd, d_conv, device)
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        for seq_len in seq_len_list:
            nt = batch_size * seq_len
            conv_state = recurrent_state = qkv = beta = g1 = conv_out = None
            q = k = v = core_out = final_state = workspace = conv_metadata = None
            try:
                cu_cpu = torch.arange(0, nt + 1, seq_len, dtype=torch.int32)
                cu = cu_cpu.to(device)
                idx = torch.arange(batch_size, dtype=torch.int32, device=device)
                has_init = torch.ones(batch_size, dtype=torch.bool, device=device)
                conv_state, recurrent_state = _glm5_next_state_pool(batch_size, nh, hd, d_conv, device)
                qkv, beta = _glm5_next_projected(nt, nh, hd, device)
                # f_b_proj output reshaped to [1, T, H, D] (kda.py:424-425).
                g1 = torch.randn(1, nt, nh, hd, dtype=dtype, device=device)
                # Pure-prefill GDNAttentionMetadata as populated by
                # gdn_attn.py:250-260 (no spec) and :399-417 (prefill fields).
                nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(cu_cpu, device=device)
                conv_metadata = GDNAttentionMetadata(
                    num_prefills=batch_size,
                    num_prefill_tokens=nt,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_spec_decodes=0,
                    num_spec_decode_tokens=0,
                    num_actual_tokens=nt,
                    has_initial_state=has_init,
                    non_spec_query_start_loc=cu,
                    non_spec_state_indices_tensor=idx,
                    nums_dict=nums_dict,
                    batch_ptr=batch_ptr,
                    token_chunk_offset_ptr=token_chunk_offset_ptr,
                )
                common = _glm5_next_common("context", batch_size, seq_len, d_model, d_conv, nh, hd, model_name)

                def run_conv():
                    # kda.py:571-581
                    return causal_conv1d_fn(
                        qkv.transpose(0, 1),
                        conv_weight,
                        None,
                        activation="silu",
                        conv_states=conv_state,
                        has_initial_state=has_init,
                        cache_indices=idx,
                        query_start_loc=cu,
                        metadata=conv_metadata,
                    ).transpose(0, 1)

                with benchmark_with_power(
                    device=device, kernel_func=run_conv, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=10
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        "causal_conv1d_fn",
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )

                conv_out = run_conv()
                q, k, v = (x.reshape(1, -1, nh, hd) for x in conv_out.split(proj, dim=-1))
                core_out = torch.empty(1, nt, nh, hd, dtype=dtype, device=device)

                if prefill_backend == "flashkda":
                    # kda.py:336-353 workspace sizing; :372-398 the fwd call.
                    workspace = torch.empty(
                        torch.ops._flashkda_C.get_workspace_size(nt, nh, batch_size), dtype=torch.uint8, device=device
                    )
                    final_state = torch.empty_like(recurrent_state)

                    def run_prefill_core():
                        initial_state = gather_initial_states(recurrent_state, idx, has_init)
                        torch.ops._flashkda_C.fwd(
                            q.contiguous(),
                            k.contiguous(),
                            v.contiguous(),
                            g1.contiguous(),
                            beta,
                            hd**-0.5,
                            core_out,
                            workspace,
                            a_log.view(-1),
                            dt_bias.view(-1, hd),
                            GLM5_NEXT_KDA_LOWER_BOUND,
                            initial_state.contiguous(),
                            final_state,
                            cu.contiguous(),
                            None,
                            None,
                        )
                        scatter_states(recurrent_state, final_state, idx)

                else:

                    def run_prefill_core():
                        initial_state = gather_initial_states(recurrent_state, idx, has_init)
                        _, last_state = chunk_kda_with_fused_gate(
                            q=q,
                            k=k,
                            v=v,
                            raw_g=g1,
                            beta=_cast_sigmoid(beta.squeeze(0)).unsqueeze(0),
                            A_log=a_log,
                            g_bias=dt_bias,
                            initial_state=initial_state,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                            cu_seqlens=cu,
                            safe_gate=True,
                            lower_bound=GLM5_NEXT_KDA_LOWER_BOUND,
                        )
                        scatter_states(recurrent_state, last_state, idx)

                with benchmark_with_power(
                    device=device,
                    kernel_func=run_prefill_core,
                    num_warmups=NUM_WARMUPS,
                    num_runs=NUM_RUNS,
                    repeat_n=1,
                ) as results:
                    _log(
                        common,
                        results["latency_ms"],
                        prefill_source,
                        perf_filename,
                        vllm_version,
                        device,
                        results["power_stats"],
                    )
                ok += 1
            except Exception as e:
                err += 1
                failures.append(f"batch_size={batch_size} seq_len={seq_len}: {type(e).__name__}: {e}")
                print(f"  Error at batch_size={batch_size}, seq_len={seq_len}: {e}")
                continue
            finally:
                conv_state = recurrent_state = qkv = beta = g1 = conv_out = None
                q = k = v = core_out = final_state = workspace = conv_metadata = None
                _cleanup("glm5_next context")

    summary = f"ok={ok} error={err} skip=0"
    print(f"GLM-5.3-Flash KDA context summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"vLLM GLM-5.3-Flash KDA context collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_glm5_next_kda_decode(
    d_model,
    d_conv,
    nh,
    hd,
    batch_size_list,
    model_name,
    perf_filename,
    vllm_version,
    row_phase="generation",
    device="cuda:0",
):
    """GLM-5.3-Flash non-spec decode: merged q|k|v causal_conv1d_update row and
    the glm5next fused_recurrent_kda row (kda.py:583-596,693-720).

    ``row_phase`` stays ``context`` for seq_len=1 cells of the shared context
    grid; vLLM classifies them as decodes (gdn_attn.py:250-252)."""
    if row_phase not in {"context", "generation"}:
        raise ValueError(f"Unsupported GLM-5.3-Flash KDA decode row phase: {row_phase}")

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

    device = torch.device(device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    proj = nh * hd
    conv_weight, a_log, dt_bias = _glm5_next_layer_params(nh, hd, d_conv, device)
    ok = err = 0
    failures: list[str] = []

    for batch_size in batch_size_list:
        conv_state = recurrent_state = qkv = beta = g1 = conv_out = core_out = None
        try:
            nb = batch_size
            cu = torch.arange(0, nb + 1, dtype=torch.int32, device=device)
            idx = torch.arange(nb, dtype=torch.int32, device=device)
            conv_state, recurrent_state = _glm5_next_state_pool(nb, nh, hd, d_conv, device)
            qkv, beta = _glm5_next_projected(nb, nh, hd, device)
            g1 = torch.randn(1, nb, nh, hd, dtype=dtype, device=device)
            # Layer output buffer [1, T, H, D] the decode kernel writes into
            # (kda.py:431-435,607-611,700-701).
            core_out = torch.empty(1, nb, nh, hd, dtype=dtype, device=device)
            common = _glm5_next_common(row_phase, nb, 1, d_model, d_conv, nh, hd, model_name)

            def run_conv_update():
                # kda.py:588-595
                return causal_conv1d_update(
                    qkv,
                    conv_state,
                    conv_weight,
                    None,
                    activation="silu",
                    conv_state_indices=idx,
                )

            with benchmark_with_power(
                device=device, kernel_func=run_conv_update, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=10
            ) as results:
                _log(
                    common,
                    results["latency_ms"],
                    "causal_conv1d_update",
                    perf_filename,
                    vllm_version,
                    device,
                    results["power_stats"],
                )

            conv_out = run_conv_update()
            q, k, v = (x.reshape(1, -1, nh, hd) for x in conv_out.split(proj, dim=-1))

            def run_recurrent():
                # kda.py:702-720
                fused_recurrent_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g1,
                    beta=beta,
                    initial_state=recurrent_state,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu,
                    ssm_state_indices=idx,
                    out=core_out,
                    sigmoid_beta=True,
                    a_log=a_log,
                    g_bias=dt_bias,
                    compute_gate=True,
                    lower_bound=GLM5_NEXT_KDA_LOWER_BOUND,
                )

            with benchmark_with_power(
                device=device, kernel_func=run_recurrent, num_warmups=NUM_WARMUPS, num_runs=NUM_RUNS, repeat_n=10
            ) as results:
                _log(
                    common,
                    results["latency_ms"],
                    "fused_recurrent_kda",
                    perf_filename,
                    vllm_version,
                    device,
                    results["power_stats"],
                )
            ok += 1
        except Exception as e:
            err += 1
            failures.append(f"batch_size={batch_size}: {type(e).__name__}: {e}")
            print(f"  Error at batch_size={batch_size}: {e}")
            continue
        finally:
            conv_state = recurrent_state = qkv = beta = g1 = conv_out = core_out = None
            _cleanup("glm5_next decode")

    summary = f"ok={ok} error={err} skip=0"
    print(f"GLM-5.3-Flash KDA {row_phase} decode-path summary: {summary}")
    if err or ok == 0:
        raise RuntimeError(
            f"vLLM GLM-5.3-Flash KDA {row_phase} decode-path collection failed strict completeness: {summary}; "
            f"failed cells: {_format_failures(failures)}"
        )


def run_glm5_next_kda_torch(
    phase, d_model, d_conv, num_k_heads, head_k_dim, num_v_heads, head_v_dim, batch_size_list, seq_len_list, **kwargs
):
    """Phase router for GLM-5.3-Flash KDA rows (called from run_kda_torch)."""
    if num_k_heads != num_v_heads or head_k_dim != head_v_dim:
        raise ValueError("GLM-5.3-Flash KDA has symmetric q/k/v heads (kda.py:196-203)")
    shape = dict(d_model=d_model, d_conv=d_conv, nh=num_v_heads, hd=head_v_dim)
    if phase == "context":
        if not seq_len_list:
            raise ValueError("vLLM KDA context collection requires at least one sequence length")
        if any(seq_len < 1 for seq_len in seq_len_list):
            raise ValueError(f"vLLM KDA context sequence lengths must be positive: {seq_len_list}")
        if 1 in seq_len_list:
            run_glm5_next_kda_decode(batch_size_list=batch_size_list, row_phase="context", **shape, **kwargs)
        prefill_seq_len_list = [seq_len for seq_len in seq_len_list if seq_len > 1]
        if prefill_seq_len_list:
            run_glm5_next_kda_context(
                batch_size_list=batch_size_list, seq_len_list=prefill_seq_len_list, **shape, **kwargs
            )
    elif phase == "generation":
        run_glm5_next_kda_decode(batch_size_list=batch_size_list, **shape, **kwargs)
    elif phase == "verify":
        # GLM-5.3-Flash is modeled with nextn=0 (MTP off); its spec-decode
        # path (kda.py:549-565,612-636) has not been audited for collection.
        raise NotImplementedError(
            "GLM-5.3-Flash KDA verify (MTP target-verify) is not collected: the GLM "
            "baseline runs without speculative decoding"
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")


def run_kda_torch(
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
    *,
    perf_filename,
    device="cuda:0",
):
    """Main entry point: routes phases and reports the installed vLLM version."""
    from vllm.version import __version__ as vllm_version

    _require_audited_runtime(model_name, vllm_version)
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
            vllm_version=vllm_version,
            device=device,
        )
        return WORKER_RESTART

    kwargs = dict(
        d_model=d_model,
        d_conv=d_conv,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        model_name=model_name,
        perf_filename=perf_filename,
        vllm_version=vllm_version,
        device=device,
    )
    if phase == "context":
        if seq_len_list is None:
            raise ValueError("vLLM KDA context collection requires seq_len_list")
        if not seq_len_list:
            raise ValueError("vLLM KDA context collection requires at least one sequence length")
        if any(seq_len < 1 for seq_len in seq_len_list):
            raise ValueError(f"vLLM KDA context sequence lengths must be positive: {seq_len_list}")

        # vLLM 0.27.0 classifies one-token non-spec batches as pure decodes
        # (kda_metadata.py:274-278) and invokes causal_conv1d_update plus
        # fused_recurrent_kda_packed_decode (kda.py:733-760). Keep the shared
        # context grid identity, but dispatch that cell through the serving
        # decode kernels instead of manufacturing prefill metadata.
        if 1 in seq_len_list:
            run_kda_generation_benchmark(batch_size_list=batch_size_list, row_phase="context", **kwargs)

        prefill_seq_len_list = [seq_len for seq_len in seq_len_list if seq_len > 1]
        if prefill_seq_len_list:
            run_kda_context_benchmark(
                batch_size_list=batch_size_list,
                seq_len_list=prefill_seq_len_list,
                **kwargs,
            )
    elif phase == "generation":
        run_kda_generation_benchmark(batch_size_list=batch_size_list, **kwargs)
    elif phase == "verify":
        run_kda_verify_benchmark(batch_size_list=batch_size_list, draft_token_list=seq_len_list, **kwargs)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return WORKER_RESTART


if __name__ == "__main__":
    import sys

    from collector.registry_types import PerfFile
    from vllm.version import __version__ as _v

    print(f"KDA Collector - vLLM {_v}")
    print(f"SM Version: {get_sm_version()}")
    print(f"Device: {torch.cuda.get_device_name()}")

    last = 0
    cases = get_kda_test_cases()
    print(f"Total test cases: {len(cases)}")
    for i, tc in enumerate(cases):
        print(f"\n[{i + 1}/{len(cases)}] {tc[9]} - {tc[0]} heads={tc[5]}")
        last = run_kda_torch(
            phase=tc[0],
            d_model=tc[1],
            d_conv=tc[2],
            num_k_heads=tc[3],
            head_k_dim=tc[4],
            num_v_heads=tc[5],
            head_v_dim=tc[6],
            batch_size_list=tc[7],
            seq_len_list=tc[8],
            model_name=tc[9],
            perf_filename=PerfFile.KDA,
        )
    sys.exit(last)
