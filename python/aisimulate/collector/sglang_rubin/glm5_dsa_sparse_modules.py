# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 DSA kernel collectors for the frozen Rubin SGLang image.

Fork of this project's ``sglang/glm5_dsa_sparse_modules.py``. Shapes and row
contracts remain the existing GLM sparse operation contracts. Metadata comes
from the serving scheduler, not a reconstructed FlashMLA input layout.

Audited source: NVIDIA SGLang 02c5a855aceb968c310e6fbc6632270e26edc84b:
https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b
All source citations below are relative to ``python/sglang`` at that commit.
"""

from __future__ import annotations

__compat__ = "sglang==0.5.18+nvinternal.rubin.0.8full.66997102"

import argparse
import json
import sys

from collector.sglang.runtime_limits import required_kv_alloc_tokens
from collector.sglang_rubin.collect_mla_module import (
    ARCHITECTURE,
    MODEL_PATH,
    _check_model,
    _dsa_context_derived_shapes,
    _dsa_generation_derived_shapes,
    _model_config,
    _pilot_model_spec,
    _prepare_batch,
    _run_subprocess,
    _validate_runtime,
    load_model_runner,
)

GLM5_ARCHITECTURE = ARCHITECTURE
KERNEL_TO_OP_NAME = {
    "mqa": "glm5_mqa_logits_module",
    "topk": "glm5_topk_module",
    "dsa_attn": "glm5_dsa_attn_module",
}


def _selected_glm5_models():
    _pilot_model_spec()
    return [MODEL_PATH]


def _glm5_sparse_kernel_cases(kernel):
    _pilot_model_spec()
    shapes = _dsa_context_derived_shapes(MODEL_PATH) + _dsa_generation_derived_shapes(MODEL_PATH)
    return [[MODEL_PATH, kernel, bs] for bs in sorted({shape[2] for shape in shapes})]


def get_glm5_mqa_test_cases():
    return _glm5_sparse_kernel_cases("mqa")


def get_glm5_topk_test_cases():
    return _glm5_sparse_kernel_cases("topk")


def get_glm5_dsa_attn_test_cases():
    return _glm5_sparse_kernel_cases("dsa_attn")


def _bench(call, device):
    from collector.helper import benchmark_with_power

    with benchmark_with_power(
        device=device,
        kernel_func=call,
        num_warmups=5,
        num_runs=20,
        repeat_n=4,
        use_cuda_graph=True,
        allow_graph_fail=False,
    ) as result:
        pass
    if not result.get("used_cuda_graph"):
        raise RuntimeError("Sparse kernel measurement did not use the required CUDA graph")
    return result


def _indexer(runner):
    indexer = runner.model.model.layers[0].self_attn.indexer
    return getattr(indexer, "_module", indexer)


def _require_full_indexer_path(runner, forward):
    indexer = _indexer(runner)
    # dsa_indexer.py:391-409,1601-1620 owns the K-only decision. Dense MHA
    # requests no indices without speculation (forward_mha.py:52-74), and
    # the K-only path returns before top-k (dsa_indexer.py:1280-1294).
    skip_logits = indexer._should_skip_logits_computation(forward) and not indexer.dsa_enable_prefill_cp
    if runner.attn_backend.use_mha or skip_logits:
        selected = "dense MHA" if runner.attn_backend.use_mha else "K-only indexer"
        raise RuntimeError(
            f"Raw MQA/top-k measurement does not support the framework-selected {selected} path; "
            "the full DSA module collector measures this serving path"
        )
    return indexer


def _mqa_rows_per_chunk(indexer, num_q, num_k, device):
    import torch

    # dsa_indexer.py:958-1009 applies the serving static-memory and observed
    # free-memory budget. Reuse that policy with this runner's resident state.
    needed, budget = indexer._should_chunk_mqa_logits(num_q, num_k, torch.device(device).index)
    return min(num_q, max(1, budget // (num_k * 4))) if needed else num_q


def _bench_mqa(runner, forward, metadata, *, is_prefill, device):
    indexer = _require_full_indexer_path(runner, forward)

    import deep_gemm
    import torch
    from sglang.kernels.ops.attention.dsa import deepgemm_paged_mqa_logits_split

    num_q = forward.input_ids.numel()
    q = torch.randn(num_q, indexer.n_heads, indexer.head_dim, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(num_q, indexer.n_heads, dtype=torch.float32, device=device)
    if is_prefill:
        # dsa_indexer.py:1045-1130: concatenated ragged KV, absolute causal
        # ks/ke, FP8 Q/K with per-token FP32 scales; head padding is framework-owned.
        num_k = int(metadata.get_indexer_seq_len_cpu().sum())
        k = torch.randn(num_k, indexer.head_dim, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        scales = torch.ones(num_k, dtype=torch.float32, device=device)
        ks, ke = metadata.get_indexer_kvcache_range()
        rows = _mqa_rows_per_chunk(indexer, num_q, num_k, device)
        q, weights, _ = indexer._pad_heads_for_deep_gemm(q, weights)

        def call():
            result = None
            # Same kernel/chunk boundary as dsa_indexer.py:1153-1191.
            with indexer._with_real_sm_count():
                for start in range(0, num_q, rows):
                    end = min(start + rows, num_q)
                    result = deep_gemm.fp8_mqa_logits(
                        q[start:end],
                        (k, scales),
                        weights[start:end],
                        ks[start:end],
                        ke[start:end],
                        clean_logits=False,
                    )
            return result

        return "deep_gemm.fp8_mqa_logits", _bench(call, device)

    # Paged decode is a different kernel from ragged prefill. Honor the
    # framework selector rather than silently timing ragged MQA for decode.
    # paged_mqa_logits_backend.py:23-45 and dsa_indexer.py:784-945.
    if not indexer.paged_mqa_logits_backend.is_deepgemm():
        raise RuntimeError(f"Pilot paged MQA selector returned {indexer.paged_mqa_logits_backend}")
    block_table = metadata.get_page_table_64()
    lengths = metadata.get_seqlens_int32().unsqueeze(-1)
    schedule = metadata.paged_mqa_schedule_metadata
    if schedule is None:
        schedule = deep_gemm.get_paged_mqa_logits_metadata(lengths, runner.page_size, indexer.sm_count)
    # The pool owns the index-K byte layout, including FP8 values / FP32 scales.
    cache = runner.token_to_kv_pool.get_index_k_with_scale_buffer(0).view(-1, runner.page_size, 1, 132)

    def call():
        return deepgemm_paged_mqa_logits_split(
            deep_gemm.fp8_paged_mqa_logits,
            q,
            cache,
            weights,
            lengths,
            block_table,
            schedule,
            block_table.shape[1] * runner.page_size,
            q_offset=num_q,
        )

    return "deep_gemm.fp8_paged_mqa_logits", _bench(call, device)


def _score_anchors(rows, width, lengths, starts, topk, device):
    """Existing flat/top-last calibration anchors, with causal valid spans."""
    import torch

    # Decode v2 requires an FP32 row stride divisible by four
    # (dsa_topk_backend.py:281-290), as produced by DeepGEMM.
    stride = (width + 3) // 4 * 4
    score = torch.zeros((rows, stride), dtype=torch.float32, device=device)[:, :width]
    yield "flat", score
    score.fill_(-5)
    for row in range(rows):
        count = min(int(lengths[row]), topk)
        end = int(starts[row]) + int(lengths[row])
        score[row, end - count : end] = torch.linspace(1, 2, count, device=device)
    yield "top_last", score


def _topk_source(metadata, *, is_prefill):
    from sglang.srt.environ import envs

    backend = metadata.topk_backend
    if not envs.SGLANG_DSA_FUSE_TOPK.get() or metadata.force_unfused_topk:
        return f"sglang.{backend.value}.topk_unfused"
    if not is_prefill and backend.should_use_topk_v2():
        return "sglang.topk_transform_512_v2"
    return f"sglang.{backend.value}.topk_transform_{metadata.topk_transform_method.name.lower()}"


def _bench_topk(runner, forward, metadata, *, is_prefill, device):
    indexer = _require_full_indexer_path(runner, forward)

    import torch

    num_q = forward.input_ids.numel()
    lengths = metadata.get_seqlens_expanded()
    if is_prefill:
        starts, _ = metadata.get_indexer_kvcache_range()
        width = int(metadata.get_indexer_seq_len_cpu().sum())
        rows = _mqa_rows_per_chunk(indexer, num_q, width, device)
    else:
        starts = torch.zeros_like(lengths)
        width = metadata.get_page_table_64().shape[1] * runner.page_size
        rows = num_q
    totals = {"flat": 0.0, "top_last": 0.0}
    for start in range(0, num_q, rows):
        end = min(start + rows, num_q)
        for mode, score in _score_anchors(
            end - start,
            width,
            lengths[start:end].cpu().tolist(),
            starts[start:end].cpu().tolist(),
            indexer.index_topk,
            device,
        ):
            kwargs = {}
            if is_prefill:
                kwargs["ks"] = starts[start:end]
                if rows < num_q:
                    # dsa_indexer.py:1189-1214 and dsa_indexer_metadata.py:128-175:
                    # chunked PAGED top-k supplies one query per sequence plus
                    # a token-to-request map. The framework computes cu_seqlens.
                    kwargs.update(
                        cu_seqlens_q=torch.ones(end - start, dtype=torch.int32, device=device),
                        ke_offset=lengths[start:end],
                        batch_idx_list=metadata.get_token_to_batch_idx()[start:end],
                    )

            def call():
                return metadata.topk_transform(score, indexer.index_topk, **kwargs)

            totals[mode] += _bench(call, device)["latency_ms"]
    return _topk_source(metadata, is_prefill=is_prefill), list(totals.items())


def _trtllm_sequence_lengths(metadata, *, is_prefill):
    # dsa_backend.py:1904-1911 passes clipped sparse lengths for prefill;
    # 2204-2212 passes the full cache lengths for decode. They are different
    # serving contracts even when both invoke the same FlashInfer entrypoint.
    return metadata.dsa_cache_seqlens_int32 if is_prefill else metadata.cache_seqlens_int32


def _bench_trtllm_attention(runner, forward, metadata, *, is_prefill, device):
    import flashinfer.decode
    import torch
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.trtllm_mla_backend import grow_multi_ctas_kv_counter_buffer_if_needed

    backend = runner.attn_backend
    selected = backend.dsa_prefill_impl if is_prefill else backend.dsa_decode_impl
    if selected != "trtllm" or backend.use_mha:
        raise RuntimeError(
            f"Sparse TRTLLM kernel is not selected for this shape: backend={selected}, dense={backend.use_mha}; "
            "the full DSA module collector measures the selected dense attention path"
        )
    attn = runner.model.model.layers[0].self_attn
    radix = attn.attn_mqa
    num_q = forward.input_ids.numel()
    # dsa_backend.py:3184-3274: FP8 absorbed Q, page-size-64 FP8 KV, and
    # physical page-size-1 sparse indices. Use serving metadata and actual
    # cache buffers; do not substitute BF16 flash_mla_sparse_fwd.
    query = torch.randn(num_q, 1, attn.num_local_heads, radix.head_dim, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    # Synthetic top-k: choose the earliest valid causal positions. This is
    # the physical-index transform documented by dsa_topk_backend.py:252-264,
    # using serving page tables and clipped per-query lengths. It avoids a
    # Q x full-context score allocation just to measure the attention kernel.
    positions = torch.arange(backend.dsa_index_topk, dtype=torch.int64, device=device)
    requests = metadata.get_token_to_batch_idx() if is_prefill else torch.arange(num_q, device=device)
    pages = metadata.get_page_table_64()
    columns = (positions // runner.page_size).clamp(max=pages.shape[1] - 1)
    physical = pages[requests[:, None], columns[None, :]] * runner.page_size + positions[None, :] % runner.page_size
    selected_indices = torch.where(
        positions[None, :] < backend.forward_metadata.dsa_cache_seqlens_int32[:, None], physical, -1
    ).to(torch.int32)
    if not backend.use_fused_topk:
        raise RuntimeError("Standalone Rubin sparse attention requires the framework's fused physical-index path")
    cache = runner.token_to_kv_pool.get_key_buffer(0).view(-1, 1, runner.page_size, backend.kv_cache_dim)
    counter = grow_multi_ctas_kv_counter_buffer_if_needed(
        backend._multi_ctas_kv_counter_buffer,
        torch.device(device),
        attn.num_local_heads,
        num_q,
    )
    block_table = selected_indices.unsqueeze(1)

    def call():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=cache,
            workspace_buffer=backend.workspace_buffer,
            qk_nope_head_dim=backend.qk_nope_head_dim,
            kv_lora_rank=backend.kv_lora_rank,
            qk_rope_head_dim=backend.qk_rope_head_dim,
            block_tables=block_table,
            seq_lens=_trtllm_sequence_lengths(backend.forward_metadata, is_prefill=is_prefill),
            max_seq_len=backend.forward_metadata.max_seq_len_k,
            sparse_mla_top_k=backend.dsa_index_topk,
            bmm1_scale=(radix.k_scale_float if radix.k_scale_float is not None else 1.0) * radix.scaling,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            multi_ctas_kv_counter_buffer=counter,
        )

    return "flashinfer.trtllm_batch_decode_with_kv_cache_mla:trtllm-gen", _bench(call, device)


def _write_row(perf_filename, *, kernel, bs, isl, prefix, num_heads, latency, model_path, source, device, mode):
    import torch

    from collector.helper import log_perf

    item = {
        "model": model_path,
        "architecture": ARCHITECTURE,
        "mla_dtype": "fp8_e4m3",
        "kv_cache_dtype": "fp8_e4m3",
        "gemm_type": "fp8_block",
        "num_heads": num_heads,
        "batch_size": bs,
        "isl": isl,
        "tp_size": 4,
        "step": prefix,
        "compress_ratio": 1,
        "latency": f"{latency:.6f}",
    }
    if mode is not None:
        item["score_mode"] = mode
    if not log_perf(
        item_list=[item],
        framework="SGLang",
        version="kernel-level",
        device_name=torch.cuda.get_device_name(device),
        op_name=KERNEL_TO_OP_NAME[kernel],
        kernel_source=source,
        perf_filename=perf_filename,
    ):
        raise RuntimeError(f"Failed to persist {kernel} to {perf_filename}")


def _run_glm5_dsa_sparse_kernel(
    model_path,
    kernel,
    bs_only,
    *,
    perf_filename,
    device="cuda:0",
    architecture=ARCHITECTURE,
    op_name_map=None,
    label="glm5",
):
    import torch

    _check_model(model_path)
    _validate_runtime()
    if architecture != ARCHITECTURE or op_name_map not in (None, KERNEL_TO_OP_NAME):
        raise ValueError("Rubin sparse collectors only support the GLM-5.2 pilot contract")
    if kernel not in KERNEL_TO_OP_NAME:
        raise ValueError(f"Unknown sparse kernel: {kernel}")
    torch.cuda.set_device(device)
    shapes = _dsa_context_derived_shapes(model_path) + _dsa_generation_derived_shapes(model_path)
    shapes = [shape for shape in dict.fromkeys(shapes) if shape[2] == bs_only]
    if not shapes:
        raise RuntimeError(f"Queued {kernel} case at batch {bs_only} resolved no shapes")
    if "--smoke" in sys.argv and len(shapes) > 8:
        shapes = [shapes[index] for index in sorted({round(i * (len(shapes) - 1) / 7) for i in range(8)})]
    # The pilot is TP4. Unlike FlashMLA's padded-head kernel, TRTLLM really
    # depends on local heads; existing num_heads/tp_size columns record that.
    local_heads = int(_model_config(model_path)["num_attention_heads"]) // 4
    max_tokens = max(
        required_kv_alloc_tokens(bs, isl if isl > 1 else prefix, prefix if isl > 1 else 0, 64, is_prefill=isl > 1)
        for prefix, isl, bs in shapes
    )
    runner = load_model_runner(
        model_path, local_heads, device=device, target_tp_size=4, max_total_tokens=max_tokens + 1024
    )
    with torch.inference_mode():
        for prefix, isl, bs in shapes:
            try:
                forward = _prepare_batch(runner, prefix, isl, bs, is_prefill=isl > 1)
                metadata = runner.attn_backend.get_indexer_metadata(0, forward)
                if kernel == "topk":
                    source, rows = _bench_topk(runner, forward, metadata, is_prefill=isl > 1, device=device)
                else:
                    measure = _bench_mqa if kernel == "mqa" else _bench_trtllm_attention
                    source, result = measure(runner, forward, metadata, is_prefill=isl > 1, device=device)
                    rows = [(None, result["latency_ms"])]
                for mode, latency in rows:
                    _write_row(
                        perf_filename,
                        kernel=kernel,
                        bs=bs,
                        isl=isl,
                        prefix=prefix,
                        num_heads=local_heads,
                        latency=latency,
                        model_path=model_path,
                        source=source,
                        device=device,
                        mode=mode,
                    )
            except Exception as error:
                # Keep shape context; a failure is observable data, never a
                # substituted kernel or success with an empty row set.
                raise RuntimeError(
                    f"{label} {kernel} failed at batch={bs}, isl={isl}, past_kv={prefix}: "
                    f"{type(error).__name__}: {error}"
                ) from error


def run_glm5_dsa_sparse_kernel_worker(
    model_path,
    kernel,
    bs_only,
    *,
    perf_filename,
    device="cuda:0",
    architecture=ARCHITECTURE,
    op_name_map=None,
    label="glm5",
):
    _run_subprocess(
        "collector.sglang_rubin.glm5_dsa_sparse_modules",
        dict(
            model_path=model_path,
            kernel=kernel,
            bs_only=bs_only,
            perf_filename=perf_filename,
            device="cuda:0",
            architecture=architecture,
            op_name_map=op_name_map,
            label=label,
        ),
        device,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    _run_glm5_dsa_sparse_kernel(**json.loads(args.payload))


if __name__ == "__main__":
    main()
