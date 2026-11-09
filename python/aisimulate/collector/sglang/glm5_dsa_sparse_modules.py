# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5 DSA sparse-attention kernel-level collector for SGLang.

GLM-5 analogue of ``deepseekv4_sparse_modules.py``. Collects the three GLM-5
NSA/DSA sparse sub-kernels at the kernel level. Each task times the points its
op-plan row lists (op_plans._sglang_glm5_sparse_rows): the DSA module plan's
points of the checkpoint. The kernel is benched standalone with synthetic
fp8/bf16 inputs (no real weights).

Sub-kernels (GLM-5 DSA prefill path, in order):
    1. ``deep_gemm.fp8_mqa_logits``  (mqa)      indexer scoring, NON-paged ragged
                                                kv over the FULL context. 32 idx
                                                heads x head_dim 128.
    2. ``fast_topk_transform_fused`` (sgl_kernel) (topk)
                                                top-2048 plus the paged output
                                                transform over the mqa logits
                                                (FULL length, NOT /4); decode
                                                points run the top-k v2 kernel
                                                (``topk_transform_v2``).
    3. ``flash_mla_sparse_fwd``      (dsa_attn) sparse FMLA over topk-selected
                                                positions; d_qk=576 (kv_lora 512
                                                + rope 64), d_v=512.

GLM-5 is uniform DSA (no compress_ratios): one attention kind, full-context.
CSV schema matches the aic module CSVs (``isl``=M, ``step``=past_kv).
"""

from __future__ import annotations

# 0.5.21 (H20/sm90 collector port: op_smoke + path gates in the v0.5.21 image;
# findings hopper_sglang_collector_port_0514_to_0521_2026_10_01).
# One release per file; earlier code is `git log -- <this file>` away.
__compat__ = "sglang==0.5.23"

import os

import torch

# Generic (kernel-agnostic) infra reused from the DSV4 sparse collector.
from collector.op_plans import plan_cases
from collector.sglang.deepseekv4_sparse_modules import (
    _bench_cuda_graph,
    _guarded_bench,
)
from collector.sglang.deepseekv4_sparse_modules import (
    _dsv4_cfg_int as _cfg_int,
)
from collector.sglang.deepseekv4_sparse_modules import (
    _dsv4_model_config as _model_config,
)

try:
    from collector.sglang.helper import get_sm_version, log_perf
except ModuleNotFoundError:
    from helper import get_sm_version, log_perf

__all__ = [
    "get_glm5_dsa_attn_test_cases",
    "get_glm5_mqa_test_cases",
    "get_glm5_topk_test_cases",
    "run_glm5_dsa_sparse_kernel_worker",
]

GLM5_ARCHITECTURE = "GlmMoeDsaForCausalLM"


def _glm5_sparse_config(model_path: str):
    from types import SimpleNamespace

    cfg = _model_config(model_path)
    kv_lora = _cfg_int(cfg, "kv_lora_rank")  # 512
    rope = _cfg_int(cfg, "qk_rope_head_dim")  # 64
    return SimpleNamespace(
        num_attention_heads=_cfg_int(cfg, "num_attention_heads"),  # 64
        index_n_heads=_cfg_int(cfg, "index_n_heads"),  # 32
        index_head_dim=_cfg_int(cfg, "index_head_dim"),  # 128
        index_topk=_cfg_int(cfg, "index_topk"),  # 2048
        kv_lora_rank=kv_lora,  # FMLA d_v = 512
        d_rope=rope,  # 64
        d_qk=kv_lora + rope,  # FMLA d_qk = 576
        compress_ratio=1,  # uniform DSA, full context
    )


KERNEL_TO_OP_NAME = {
    "mqa": "glm5_mqa_logits_module",
    "topk": "glm5_topk_module",
    "dsa_attn": "glm5_dsa_attn_module",
}
KERNEL_TO_KERNEL_SOURCE = {
    "mqa": "deep_gemm.fp8_mqa_logits",
    "topk": "fast_topk_v2",
    "dsa_attn": "flash_mla_sparse_fwd",
}


def _make_perf_filename(kernel: str, output_path: str) -> str:
    if os.path.isdir(output_path) or not output_path.endswith(".txt"):
        return os.path.join(output_path, f"{KERNEL_TO_OP_NAME[kernel]}_perf.txt")
    return output_path


def _write_row(
    perf_filename,
    *,
    kernel,
    bs,
    isl,
    past_kv,
    tp_size,
    native_heads,
    latency_ms,
    device_name,
    model_path,
    score_mode=None,
    kernel_source=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(perf_filename)) or ".", exist_ok=True)
    mla_dtype = "bfloat16" if kernel == "dsa_attn" else "fp8_e4m3"
    item = {
        "model": model_path,
        "architecture": GLM5_ARCHITECTURE,
        "mla_dtype": mla_dtype,
        "kv_cache_dtype": "fp8_e4m3",
        "gemm_type": "fp8_block",
        "num_heads": native_heads,
        "batch_size": bs,
        "isl": isl,
        "tp_size": tp_size,
        "step": past_kv,
        "compress_ratio": 1,
        "latency": f"{latency_ms:.6f}",
    }
    if score_mode is not None:
        item["score_mode"] = score_mode
    if not log_perf(
        item_list=[item],
        framework="SGLang",
        version="kernel-level",
        device_name=device_name,
        op_name=KERNEL_TO_OP_NAME[kernel],
        kernel_source=kernel_source or KERNEL_TO_KERNEL_SOURCE[kernel],
        perf_filename=perf_filename,
    ):
        raise RuntimeError(f"failed to persist {GLM5_ARCHITECTURE} sparse row to {perf_filename}")


# ═══════════════════════════════════════════════════════════════════════
# Kernel benches (standalone, synthetic inputs)
# ═══════════════════════════════════════════════════════════════════════
def _glm5_score_rows_per_chunk(num_q, num_k, device):
    """Bound one standalone FP32 score workspace by query rows.

    This keeps MQA logits and context top-k on the same production-derived
    *standalone* policy.  It is not SGLang serving's complete policy: a real
    Indexer also applies finalized ``mem_fraction_static`` state and caches the
    first non-capture budget while the model and KV pool are resident.  The
    kernel collector has none of that state, so it deliberately derives a
    fresh, device-capacity-neutral bound rather than hard-coding an H20 budget.
    """
    if num_q * num_k < 8_000_000:
        return num_q

    from sglang.srt.environ import envs

    free_mem, total_mem = torch.cuda.mem_get_info(device)
    free_mem_fraction = envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()
    budget_bytes = min(int(free_mem * free_mem_fraction), int(total_mem * 0.30))
    rows = min(num_q, max(1, budget_bytes // (num_k * 4)))
    if rows < num_q:
        chunks = (num_q + rows - 1) // rows
        print(
            "  standalone score chunks: "
            f"Q={num_q} K={num_k} rows={rows} chunks={chunks} "
            f"budget_bytes={budget_bytes} free_bytes={free_mem} total_bytes={total_mem}"
        )
    return rows


def _bench_glm5_mqa(M, past_kv, isl, *, index_n_heads, index_head_dim, device):  # noqa: N803
    """deep_gemm.fp8_mqa_logits — ragged batch of bs = M // isl requests.
    M = bs*isl query tokens over a CONCATENATED per-request KV cache (bs
    segments of past_kv + isl); ks/ke are absolute [start, end) into that
    cache, matching sglang's dsa_indexer (per-token k_start / k_end). Each
    request r local pos p scans causal [r*seg, r*seg + past_kv + p + 1).

    Deviation (decode points, isl == 1, every SM): serving's decode indexer scores
    with DeepGEMM fp8_paged_mqa_logits over the page-64 index-K cache
    (Indexer._get_topk_paged, dsa/dsa_indexer.py:1219-1449 @v0.5.23, through
    deepgemm_paged_mqa_logits_split, kernels/ops/attention/dsa/paged_mqa_logits.py:
    69-92, with get_paged_mqa_logits_metadata built at metadata time,
    dsa_backend.py:960-973, and batches above the SM count split per chunk); these
    rows time the ragged prefill kernel of a one-token extend instead (the op plan
    declares them decode steps, op_plans.Glm5SparseCase). The direction of the error
    is not measured. The only reader, the CP prefill composition
    (crates/core/src/perfmodel/operators/dsa.rs:429-490), interpolates across them.
    Retire by timing fp8_paged_mqa_logits for isl == 1 once the owner settles which
    kernel these points mean.
    """
    from deep_gemm import fp8_mqa_logits

    bs = max(1, M // isl)
    seg = past_kv + isl  # per-request KV length
    full_s = max(1, bs * seg)  # CONCATENATED kv: bs segments of (past_kv + isl)
    q = torch.randn(M, index_n_heads, index_head_dim, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k_fp8 = torch.randn(full_s, index_head_dim, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k_scale = torch.ones(full_s, dtype=torch.float32, device=device)
    weights = torch.randn(M, index_n_heads, dtype=torch.float32, device=device)
    # Absolute [ks, ke) into the concatenated kv (sglang dsa_indexer k_start/k_end):
    # token (request r, local pos p) attends its own segment
    # [r*seg, r*seg + past_kv + p + 1).
    seg_start = torch.repeat_interleave(torch.arange(bs, dtype=torch.int32, device=device) * seg, isl)
    causal = torch.arange(1, isl + 1, dtype=torch.int32, device=device).repeat(bs)
    ks = seg_start
    ke = (seg_start + past_kv + causal).clamp(max=full_s)

    # Match SGLang Indexer._get_topk_ragged (dsa/dsa_indexer.py:1540-1700 @v0.5.23): fp8_mqa_logits
    # materializes a float32 [query_rows, concatenated_kv_rows] workspace, so
    # large ragged batches must be split along Q. Without this loop the
    # standalone collector can request terabytes even though the serving path
    # runs the same shape in bounded chunks.
    query_rows_per_chunk = _glm5_score_rows_per_chunk(M, full_s, device)

    def kernel_fn():
        logits = None
        for start in range(0, M, query_rows_per_chunk):
            end = min(start + query_rows_per_chunk, M)
            logits = fp8_mqa_logits(
                q[start:end],
                (k_fp8, k_scale),
                weights[start:end],
                ks[start:end],
                ke[start:end],
                clean_logits=False,
            )
        return logits

    return _bench_cuda_graph(kernel_fn, allow_graph_fail=False, device=device)


def _glm5_fuse_topk_enabled() -> bool:
    """Whether SGLang runs the FUSED topk+index-transform (env default = True)."""
    from sglang.srt import environ as _environ

    return bool(_environ.envs.SGLANG_DSA_FUSE_TOPK.get())


def _glm5_topk_v2_enabled() -> bool:
    """DSATopKBackend.should_use_topk_v2 for serving's default sgl-kernel top-k backend
    (dsa/dsa_topk_backend.py:55-56 @v0.5.23; SGLANG_OPT_USE_TOPK_V2 defaults to True)."""
    from sglang.srt import environ as _environ

    return bool(_environ.envs.SGLANG_OPT_USE_TOPK_V2.get())


def _glm5_topk_metadata(bs, isl, past_kv):
    """Return causal row lengths, ragged KV offsets, and full KV width."""
    max_seqlen_k = max(1, past_kv + isl)
    lengths = [past_kv + token_idx + 1 for _ in range(bs) for token_idx in range(isl)]
    cu_seqlens_k = [request_idx * max_seqlen_k for request_idx in range(bs + 1)]
    topk_indices_offset = [cu_seqlens_k[request_idx] for request_idx in range(bs) for _ in range(isl)]
    return lengths, topk_indices_offset, max_seqlen_k


def _make_glm5_topk_scores(mode, lengths, seq, device, generator, topk_k):
    """DSV4-style score distributions for the topk DELTA calibration.

    * ``flat``     — all zeros: degenerate worst case (every element ties, so
                     the kernel does maximal tie-break work).
    * ``top_last`` — background ~-5 with the last ``topk_k`` positions ~+5:
                     representative (clear winners, the common real case).

    Each row's winners occupy its own causal ``[length-k:length)`` span.
    This helper is used by paged decode.  SGLang's page size and DeepGEMM
    block size are 64, so its MQA logits stride is ``ceil(seq / 64) * 64``.
    """
    rows = lengths.numel()
    pad = ((seq + 63) // 64) * 64
    if mode == "flat":
        return torch.zeros(rows, pad, dtype=torch.float32, device=device)
    if mode == "top_last":
        s = -5.0 + 0.05 * torch.randn(rows, pad, dtype=torch.float32, device=device, generator=generator)
        counts = lengths.to(torch.int64).clamp(min=0, max=min(topk_k, seq))
        offsets = torch.arange(topk_k, device=device)
        columns = lengths.to(torch.int64).unsqueeze(1) - counts.unsqueeze(1) + offsets.unsqueeze(0)
        valid = offsets.unsqueeze(0) < counts.unsqueeze(1)
        row_ids = torch.arange(rows, device=device).unsqueeze(1).expand_as(columns)
        selected = int(valid.sum().item())
        s[row_ids[valid], columns[valid]] = 5.0 + torch.randn(
            selected, dtype=torch.float32, device=device, generator=generator
        )
        return s.contiguous()
    raise ValueError(f"unknown topk score mode: {mode}")


def _bench_glm5_topk(M, past_kv, isl, bs, *, topk, device):  # noqa: N803
    """GLM-5 indexer top-k — the kernel SGLang's DSA backend ACTUALLY runs,
    benched as a FLAT/TOP_LAST DELTA calibration (DSV4-style).

    Kernel selection at SGLang v0.5.23 (``SGLANG_DSA_FUSE_TOPK`` and
    ``SGLANG_OPT_USE_TOPK_V2`` both default True, sgl-kernel top-k backend):
    context uses the PAGED ``fast_topk_transform_fused`` output transform with
    row_starts (dsa/dsa_topk_backend.py:204-211), decode the top-k v2 kernel
    (``topk_transform_v2``, :123-133).  The emitted FP8 profile selects
    ``flashmla_kv`` on SM90 and ``trtllm`` on SM100/103
    (arg_groups/overrides.py:714-720), both PAGED (dsa_backend.py:4101-4124).
    SGLang uses the RAGGED transform only for an FP8 + ``flashmla_sparse``
    EXTEND profile, which this collector does not emit and must not merge into
    the same persisted contract.  Fuse-topk disabled uses plain ``fast_topk_v2``.

    Context still uses the production concatenated-ragged MQA score geometry
    and absolute row starts; PAGED describes how selected local indices are
    mapped through the request page table, not the score layout.  Decode keeps
    this release collector's separate compact paged calibration.

    topk timing is DATA-DEPENDENT (measured 3-22% spread by score distribution),
    so instead of guessing the real logit distribution we bench two anchors —
    ``flat`` (degenerate worst-case) and ``top_last`` (representative) — and
    return both as ``[("flat", lat), ("top_last", lat)]``; the SDK applies the
    DELTA as the data-dependent correction.  Trivial when the full per-request
    context ``<= topk`` (select-all, no data-dependent cost) → both ``0``.

    Context rows use production's concatenated ragged-K geometry and bound the
    synthetic FP32 score along Q.  Their reported latency is the sum of
    independently warmed steady-state chunk-kernel measurements.  It excludes
    score construction, Python orchestration, and power, and is not an
    end-to-end timing of production's interleaved MQA/top-k/copy loop.  Decode
    is a single-kernel paged calibration on the kernel serving's decode runs.

    Returns ``(results, kernel_source)``.
    """
    from sgl_kernel import (
        fast_topk_transform_fused,
        fast_topk_v2,
    )

    fused = _glm5_fuse_topk_enabled()
    kernel_src = "fast_topk_transform_fused" if fused else "fast_topk_v2"

    # GLM-5 is uniform DSA (ratio=1): per-request full context = past_kv + isl.
    length_values, topk_offset_values, max_seqlen_k = _glm5_topk_metadata(bs, isl, past_kv)
    if max_seqlen_k <= topk:
        # nothing to select -> no data-dependent cost; DELTA is 0.
        return [("flat", 0.0), ("top_last", 0.0)], kernel_src

    lengths = torch.tensor(length_values, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)

    if isl > 1:
        # SGLang's MQA output is [Q, sum(request KV lengths)], not one compact
        # per-request score row. ``topk_offset_values`` is the absolute start
        # of each request's valid span in that concatenated score.
        full_k = bs * max_seqlen_k
        row_starts = torch.tensor(topk_offset_values, dtype=torch.int32, device=device)
        rows_per_chunk = _glm5_score_rows_per_chunk(M, full_k, device)
        mode_latency_ms = {"flat": 0.0, "top_last": 0.0}
        chunked = rows_per_chunk < M
        if fused:
            page_table = torch.arange(full_k, dtype=torch.int32, device=device).view(bs, max_seqlen_k)
            request_for_row = torch.repeat_interleave(torch.arange(bs, dtype=torch.int64, device=device), isl)
            base_cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=device) * isl

        for start in range(0, M, rows_per_chunk):
            end = min(start + rows_per_chunk, M)
            lengths_chunk = lengths[start:end]
            starts_chunk = row_starts[start:end]
            score = torch.empty((end - start, full_k), dtype=torch.float32, device=device)
            if fused and chunked:
                # Match Indexer._get_topk_ragged's PAGED chunk path: each query
                # row is a length-one sequence and selects its request's page
                # table row through token_to_batch_idx.
                page_table_chunk = page_table[request_for_row[start:end]]
                cu_seqlens_q_chunk = torch.arange(end - start + 1, dtype=torch.int32, device=device)
            elif fused:
                # The unchunked production call keeps the original request
                # metadata rather than expanding one page-table row per token.
                page_table_chunk = page_table
                cu_seqlens_q_chunk = base_cu_seqlens_q

            # Reuse one score allocation for the two calibration anchors.  A
            # benchmark closure must be released before the buffer is changed
            # because CUDA Graph capture fixes its input address and lifetime.
            for mode in ("flat", "top_last"):
                if mode == "flat":
                    score.zero_()
                else:
                    score.normal_(mean=-5.0, std=0.05, generator=generator)
                    counts = lengths_chunk.to(torch.int64).clamp(min=0, max=topk)
                    offsets = torch.arange(topk, dtype=torch.int64, device=device)
                    columns = (
                        starts_chunk.to(torch.int64).unsqueeze(1)
                        + lengths_chunk.to(torch.int64).unsqueeze(1)
                        - counts.unsqueeze(1)
                        + offsets.unsqueeze(0)
                    )
                    valid = offsets.unsqueeze(0) < counts.unsqueeze(1)
                    row_ids = torch.arange(end - start, device=device).unsqueeze(1).expand_as(columns)
                    selected = int(valid.sum().item())
                    winners = torch.empty(selected, dtype=torch.float32, device=device)
                    winners.normal_(mean=5.0, std=1.0, generator=generator)
                    score[row_ids[valid], columns[valid]] = winners

                if fused:
                    kernel_fn = lambda: fast_topk_transform_fused(
                        score=score,
                        lengths=lengths_chunk,
                        page_table_size_1=page_table_chunk,
                        cu_seqlens_q=cu_seqlens_q_chunk,
                        topk=topk,
                        row_starts=starts_chunk,
                    )
                else:
                    kernel_fn = lambda: fast_topk_v2(
                        score,
                        lengths_chunk,
                        topk,
                        row_starts=starts_chunk,
                    )
                # Fail closed if any context chunk cannot use the declared
                # independent CUDA-graph benchmark boundary.  Mixing eager and
                # graph timings inside one additive row would be ambiguous.
                measured = _bench_cuda_graph(kernel_fn, allow_graph_fail=False, device=device)
                mode_latency_ms[mode] += measured["latency_ms"]
                del kernel_fn
            del score
            if fused:
                # Advanced indexing materializes the per-row page table for a
                # real chunk. Release it before the next score/table pair is
                # allocated; retaining one prior chunk can triple transient
                # workspace at batch size one.
                del page_table_chunk, cu_seqlens_q_chunk

        return [(mode, round(mode_latency_ms[mode], 6)) for mode in ("flat", "top_last")], kernel_src

    # Decode: Indexer._get_topk_paged calls metadata.topk_transform(logits, index_topk)
    # without row_starts (dsa/dsa_indexer.py:1448-1449 @v0.5.23). With the default
    # SGLANG_OPT_USE_TOPK_V2=True (environ.py:1486) and the sgl-kernel top-k backend
    # (arg_groups/fields/exec_.py:386-393), that PAGED decode shape goes to the
    # DeepSeek-V4 top-k v2 kernel over the indexer's page-64 real_page_table
    # (dsa/dsa_topk_backend.py:123-133 -> _topk_transform_v2_paged :312-381); its plan
    # is built once per forward when the metadata is (dsa_backend.py:975-989), not in
    # the layer. sgl-kernel's fast_topk_transform_fused without row_starts runs only
    # when v2 is off.
    if not fused:

        def make_fn(score):
            return lambda: fast_topk_v2(score, lengths, topk)
    elif _glm5_topk_v2_enabled() and topk <= 2048:
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

        page_size = 64  # dsa/dsa_indexer.py:1247: the DSA index-K pool page
        pages = (max_seqlen_k + page_size - 1) // page_size
        real_page_table = torch.arange(M * pages, dtype=torch.int32, device=device).view(M, pages)
        plan = plan_topk_v2(lengths)
        kernel_src = "topk_transform_v2"

        def make_fn(score):
            def run():
                # _topk_transform_v2_paged allocates its output per call (:378).
                out = score.new_empty((M, topk), dtype=torch.int32)
                topk_transform_paged_v2(score, lengths, real_page_table, out, page_size, plan)
                return out

            return run
    else:
        pt = torch.arange(max_seqlen_k, dtype=torch.int32, device=device)
        page_table_size_1 = pt.view(1, max_seqlen_k).repeat(M, 1).contiguous()
        cu_seqlens_q = torch.arange(M + 1, dtype=torch.int32, device=device)

        def make_fn(score):
            return lambda: fast_topk_transform_fused(
                score=score,
                lengths=lengths,
                page_table_size_1=page_table_size_1,
                cu_seqlens_q=cu_seqlens_q,
                topk=topk,
            )

    results = []
    for mode in ("flat", "top_last"):
        score = _make_glm5_topk_scores(mode, lengths, max_seqlen_k, device, generator, topk)
        r = _bench_cuda_graph(make_fn(score), allow_graph_fail=False, device=device)
        results.append((mode, round(r["latency_ms"], 6)))
    return results, kernel_src


def _bench_glm5_dsa_attn(M, past_kv, isl, *, native_heads, d_qk, d_v, topk, device):  # noqa: N803
    """flash_mla_sparse_fwd — sparse FMLA, ragged batch of bs = M // isl reqs.

    The call serving's flashmla_sparse prefill makes (layers/attention/dsa_backend.py:
    2464-2471 -> _forward_flashmla_sparse :2805-2878 @v0.5.23): the sgl-flash-mla
    wheel's legacy kernel through kernels/ops/attention/flash_mla_interface.py:144-168,
    which replaced sgl_kernel.flash_mla in serving at v0.5.23.
    q (M=bs*isl, heads->64 on SM90 / pad128 on SM100+, d_qk), kv = CONCATENATED bs segments of
    (past_kv + isl); indices (M, 1, K) are absolute into each token's own
    segment. The production top-k transform always returns ``topk`` columns:
    token p of a request sees past_kv + p + 1 positions, selects
    min(past_kv + p + 1, topk) of them and pads the rest with -1. Serving passes
    that count as ``topk_length`` (dsa_cache_seqlens_int32 = compute_dsa_seqlens,
    dsa/utils.py:159-161, dsa_backend.py:1353-1360) so the kernel skips the padded
    tail."""
    from sglang.kernels.ops.attention.flash_mla_interface import flash_mla_sparse_fwd

    bs = max(1, M // isl)
    seg = past_kv + isl
    full_s = max(1, bs * seg)  # concatenated bs segments of (past_kv + isl)
    sm_major, _ = torch.cuda.get_device_capability(device)
    q_heads = 128 if sm_major >= 10 else native_heads
    q = torch.randn(M, q_heads, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(full_s, 1, d_qk, dtype=torch.bfloat16, device=device)
    # each token selects positions inside its own causal window [r*seg, r*seg + past_kv + p + 1)
    seg_start = torch.repeat_interleave(torch.arange(bs, dtype=torch.int32, device=device) * seg, isl)
    causal = torch.arange(1, isl + 1, dtype=torch.int32, device=device).repeat(bs)
    topk_length = (past_kv + causal).clamp(max=topk).to(torch.int32)
    columns = torch.arange(topk, dtype=torch.int32, device=device)
    indices = torch.where(
        columns.view(1, topk) < topk_length.view(M, 1),
        seg_start.view(M, 1) + columns.view(1, topk),
        torch.full((), -1, dtype=torch.int32, device=device),
    ).view(M, 1, topk)
    sm_scale = 1.0 / (d_qk**0.5)

    def kernel_fn():
        return flash_mla_sparse_fwd(q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v, topk_length=topk_length)

    return _bench_cuda_graph(kernel_fn, allow_graph_fail=False, device=device)


def _bench_glm5_sparse_kernel_shape(kernel, prefix, isl, bs, sc, device):
    M = max(bs * isl, 1)  # noqa: N806
    if kernel == "topk":
        # flat/top_last DELTA calibration -> two rows; reports the fused/unfused
        # kernel it actually ran.
        results, kernel_source = _bench_glm5_topk(M, prefix, isl, bs, topk=sc.index_topk, device=device)
        return kernel_source, results
    if kernel == "mqa":
        r = _bench_glm5_mqa(
            M, prefix, isl, index_n_heads=sc.index_n_heads, index_head_dim=sc.index_head_dim, device=device
        )
    elif kernel == "dsa_attn":
        r = _bench_glm5_dsa_attn(
            M,
            prefix,
            isl,
            native_heads=sc.num_attention_heads,
            d_qk=sc.d_qk,
            d_v=sc.kv_lora_rank,
            topk=sc.index_topk,
            device=device,
        )
    else:
        raise ValueError(f"unknown glm5 kernel={kernel}")
    return KERNEL_TO_KERNEL_SOURCE[kernel], [(None, r["latency_ms"])]


def run_glm5_dsa_sparse_kernel_worker(
    model_path,
    kernel,
    batch_size,
    points,
    *,
    perf_filename,
    device="cuda:0",
):
    """One (model, batch) row of the op plan: bench each listed ``[prefix, isl]`` point."""
    if kernel not in KERNEL_TO_OP_NAME:
        raise ValueError(f"unknown kernel={kernel}; expected one of {list(KERNEL_TO_OP_NAME)}")
    bs_only = int(batch_size)
    shapes = [(int(prefix), int(isl), bs_only) for prefix, isl in points]
    if not shapes:
        raise RuntimeError(f"glm5-sparse {kernel} bs={bs_only}: the row lists no points")
    sc = _glm5_sparse_config(model_path)
    output_dir = os.path.dirname(perf_filename) or os.getcwd()
    perf_path = _make_perf_filename(kernel, output_dir)
    device_name = torch.cuda.get_device_name(device)
    print(f"[glm5-sparse {kernel} bs={bs_only}] {len(shapes)} shapes -> {perf_path}")
    n_ok = 0
    failures = []
    for prefix, isl, bs in shapes:
        shape_label = f"bs={bs} isl={isl} past_kv={prefix}"
        out, error = _guarded_bench(
            lambda: _bench_glm5_sparse_kernel_shape(kernel, prefix, isl, bs, sc, device),
            shape_label,
        )
        if error is not None:
            failures.append(f"{shape_label}: {type(error).__name__}: {error}")
            continue
        kernel_source, results = out
        expected_modes = {"flat", "top_last"} if kernel == "topk" else {None}
        if len(results) != len(expected_modes) or {score_mode for score_mode, _ in results} != expected_modes:
            message = f"expected score modes {expected_modes}, got {results}"
            print(f"  incomplete result at {shape_label}: {message}")
            failures.append(f"{shape_label}: RuntimeError: {message}")
            continue
        for score_mode, latency_ms in results:
            _write_row(
                perf_path,
                kernel=kernel,
                bs=bs,
                isl=isl,
                past_kv=prefix,
                tp_size=1,
                native_heads=sc.num_attention_heads,
                latency_ms=latency_ms,
                device_name=device_name,
                model_path=model_path,
                score_mode=score_mode,
                kernel_source=kernel_source,
            )
        n_ok += 1
    error_count = len(failures)
    summary = f"ok={n_ok} error={error_count} skip=0 total={len(shapes)}"
    print(f"  {kernel}: {summary}")
    if not n_ok or failures:
        details = "\n- ".join(failures) if failures else "no inner shapes produced a row"
        raise RuntimeError(f"glm5-sparse {kernel} bs={bs_only}: {summary}; failures:\n- {details}")


def get_glm5_mqa_test_cases():
    return plan_cases("glm5_mqa_logits_module", "sglang", get_sm_version())


def get_glm5_topk_test_cases():
    return plan_cases("glm5_topk_module", "sglang", get_sm_version())


def get_glm5_dsa_attn_test_cases():
    return plan_cases("glm5_dsa_attn_module", "sglang", get_sm_version())
