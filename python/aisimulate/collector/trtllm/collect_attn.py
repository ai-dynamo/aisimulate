# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM dense attention collector.

Constructs a single TRT-LLM torch-flow attention layer and synthetic metadata to
benchmark context and generation attention. This file owns TRT-LLM cache manager
setup, quantization flags and perf-row formatting; the cases come from the
shared attention plan (collector/op_plans.py).
"""

import os

import tensorrt_llm
import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from collector.registry_types import PerfFile
from tensorrt_llm._torch.attention.backends import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

__compat__ = "trtllm==1.3.0rc29"


def _deployment_tokens_per_block() -> int:
    """KV page the deployment actually serves with, not a collector constant.

    The generator renders every trtllm engine with ``kv_cache_config.tokens_per_block: 32``
    (TRT-LLM's own KvCacheConfig default); the collector used to pin 64 here, which on
    Blackwell selects a different FMHA cubin family — the sm100 kernel names carry the page
    (fmhaSm100a…PagedKvCausalP64… vs serving's …P32…), so the sm100 attention gates
    DIVERGED on nothing but the page while the sm90 names hide it (B200, 2026-10-04,
    opharness finding sm100_sglang_trtllm_gate_verdicts_2026_10_04). Ask the framework
    for its default; ``AIS_TRTLLM_TOKENS_PER_BLOCK`` is the A/B hook.
    """
    override = os.environ.get("AIS_TRTLLM_TOKENS_PER_BLOCK")
    if override:
        return int(override)
    try:
        from tensorrt_llm.llmapi import KvCacheConfig

        return int(KvCacheConfig().tokens_per_block)
    except Exception:  # older builds without the pydantic config
        return 32


def run_attention_torch(
    batch_size,
    input_len,
    num_heads,
    num_key_value_heads,  # keep same as num_heads for MHA
    head_dim,
    attention_window_size,
    use_fp8_kv_cache,
    use_fp8_context_fmha,
    is_context_phase,
    attn_backend_name="TRTLLM",
    *,
    perf_filename,
    device="cuda:0",
):
    device = torch.device(device)
    # No torch.set_default_device(device): serving never sets a default device, and
    # every tensor this collector creates names its device.
    torch.cuda.set_device(device)

    # TRT-LLM serves every collected dense-attention model on its default TRTLLM
    # backend (TorchLlmArgs.attn_backend, llmapi/llm_args.py).
    if attn_backend_name != "TRTLLM":
        raise ValueError(f"Unsupported TRT-LLM dense attention backend: {attn_backend_name}")
    # if XQA JIT is enabled, the context phase will also trigger XQA prepare which causes the error
    # with specifc q/kv head and seq setting.
    # NOTE (1.3.0rc23, verified 2026-09-24): this env no longer exists in the
    # binaries (only TRTLLM_FORCE_XQA / TRTLLM_XQA_BLOCKS_PER_SEQUENCE do).
    # XQA vs MMHA for decode is the op's own heuristic ("JIT XQA is not
    # used: maybe no performance gain" at short KV; XQA selected at long KV) —
    # the collector follows it per (batch, kv_len) cell exactly as serving
    # does, which path_diff confirmed at kv 4095 (XQA) vs kv 1 (MMHA). Kept
    # for older builds that still read it.
    if is_context_phase:
        os.environ["TRTLLM_ENABLE_XQA_JIT"] = "0"
    else:
        os.environ["TRTLLM_ENABLE_XQA_JIT"] = "1"

    layer_idx = 0
    world_size = 1
    tp_size = 1
    tokens_per_block = _deployment_tokens_per_block()
    warming_up = 10
    test_ite = 6
    output_len = 1
    if use_fp8_context_fmha:
        assert use_fp8_kv_cache
        quant_algo = QuantAlgo.FP8
    else:
        quant_algo = None

    # out_scale mirrors serving's Attention._use_quantize_output()
    # (_torch/modules/attention.py:648-670,758-761@1.3.0rc20): an fp8-quantized
    # model supplies o_proj.inv_input_scale; a model with no weight/activation
    # quant (``not has_any_quant(exclude_kv_cache=True)``) supplies None, fp8 KV
    # cache or not. Passing a scale is not inert: the backend allocates fp8
    # attention output whenever out_scale is present (is_quantize_output,
    # attention_backend/trtllm.py:1452) and the op then takes the fp8-output path
    # (is_fp8_out, thop/attentionOp.cpp:1155:
    # ``is_fp8_out || is_fp4_out || (hasFp8KvCache() && use_paged_context_fmha)``).
    # Per phase:
    # - context: the fp8-model flavor is the dedicated use_fp8_context_fmha case,
    #   so the fp8-KV-only case means "unquantized model + fp8 KV" -> None. On
    #   SM90 the framework then aborts on its own ("attention output scale should
    #   be provided", common/attentionOp.cpp:681) because fp8-KV plus the forced
    #   paged-context FMHA (trtllm.py:1434, nvbug 5624818) select the fp8-output
    #   path with no scale — serving aborts identically for that config on
    #   SM90/rc20, so the case fails classified rather than being measured under
    #   the wrong label (the row records attn_dtype=bfloat16).
    # - generation: the sweep has no separate fp8-model case, so the fp8-KV row
    #   is the fp8-model deployment; its serving path provides the scale and
    #   writes fp8 attention output on every SM. A dummy unit scale mirrors
    #   o_proj.inv_input_scale for the bare-layer benchmark.
    if use_fp8_context_fmha or (use_fp8_kv_cache and not is_context_phase):
        out_scale = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )
    else:
        out_scale = None

    if use_fp8_kv_cache:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    else:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16

    # Serving hands pos_embd_params to create_attention when the backend fuses
    # RoPE (modules/attention.py), which the TRTLLM backend does.
    pos_embd_params = PositionalEmbeddingParams(type=PositionEmbeddingType.rope_gpt_neox, rope=RopeParams(dim=128))

    quant_config = QuantConfig(
        quant_algo=quant_algo,  # fp8 fmha
        kv_cache_quant_algo=QuantAlgo.FP8 if use_fp8_kv_cache else None,  # fp8 kv,
        group_size=128,
        smoothquant_val=0.5,
        clamp_val=None,
        use_meta_recipe=False,
        has_zero_point=False,
        pre_quant_scale=False,
        exclude_modules=None,
    )

    # A head size the kernels lack aborts inside AttentionOp::initialize before
    # Python can catch it ("Head size N is not supported by MMHA" was seen above
    # head_dim 256); the worker restart records it. Gemma 4's head_dim 512 rows
    # run here, as serving runs them.
    attn = create_attention(
        backend_name=attn_backend_name,
        layer_idx=layer_idx,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_key_value_heads,
        pos_embd_params=pos_embd_params,
        quant_config=quant_config,
        is_mla_enable=False,
    )
    # A row must time the backend it is labeled with: verify the constructed class.
    if type(attn).__name__ != "TrtllmAttention":
        raise RuntimeError(
            f"create_attention resolved {type(attn).__name__} for backend {attn_backend_name} "
            "(expected TrtllmAttention)"
        )

    total_num_tokens = (input_len + output_len) * batch_size

    mapping = Mapping(world_size=world_size, rank=0, tp_size=tp_size)

    num_hidden_layers = 1

    synthetic_max_seq_len = input_len + output_len + 1
    if attention_window_size > 0:
        synthetic_max_seq_len = max(synthetic_max_seq_len, attention_window_size + output_len + 1)

    kv_cache_config = KvCacheConfig(
        max_tokens=int((synthetic_max_seq_len - 1) / tokens_per_block + 1)
        * tokens_per_block
        * batch_size
        * 2,  # num_bloacks * block_size
        enable_block_reuse=False,
    )

    # Serving allocates dense-attention KV cache as CacheType.SELF
    # (pyexecutor/_util.py:1539@1.3.0rc20); SELFKONLY is reserved for MLA
    # (_util.py:1636). Regular GQA/MHA has separate K and V tensors, so both
    # TRTLLM and FlashInfer paths must benchmark the full serving layout.
    kv_cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
    kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=kv_cache_type,
        num_layers=num_hidden_layers,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=synthetic_max_seq_len,  # +1 for the magic fixme mentioned in trtllm xqa JIT path impl.
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    input_seq_lens = [input_len for _ in range(batch_size)]
    total_seq_lens = [input_len + output_len for _ in range(batch_size)]
    request_ids = list(range(batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, total_seq_lens)

    if is_context_phase:
        metadata_kwargs = dict(
            seq_lens=torch.tensor(input_seq_lens, dtype=torch.int32, device="cpu"),
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0 for _ in range(batch_size)],
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=False, cache_reuse=False, has_speculative_draft_tokens=False
            ),
        )
    else:
        gen_seq_lens = [1 for _ in range(batch_size)]
        metadata_kwargs = dict(
            seq_lens=torch.tensor(gen_seq_lens, dtype=torch.int32, device="cpu"),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=input_seq_lens,
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
        )
    metadata_kwargs.update(
        max_num_requests=batch_size,
        max_num_tokens=total_num_tokens,
        kv_cache_manager=kv_cache_manager,
        mapping=mapping,
        enable_flash_mla=False,
        position_ids=None,
        cross=None,
        request_ids=request_ids,
        # Serving populates prompt_lens with the current-chunk length for
        # context requests and request.py_prompt_len for generation requests
        # (pyexecutor/model_engine.py:3015,3180,3227,3277,3353,3760@1.3.0rc20);
        # the field is part of the shared AttentionMetadata contract
        # (attention_backend/interface.py:134-138@1.3.0rc20) that every
        # backend inherits — FlashInferAttentionMetadata does not consume it
        # directly. With chunked_prefill=False the current chunk IS the full
        # prompt, so input_seq_lens matches both phases here.
        prompt_lens=input_seq_lens,
        all_rank_num_tokens=None,
    )
    attn_metadata = TrtllmAttentionMetadata(
        **metadata_kwargs,
        workspace=torch.tensor([], device=device, dtype=torch.int8),
    )

    attn_metadata.prepare()

    if is_context_phase:
        num_tokens = input_len * batch_size
    else:
        num_tokens = batch_size

    sinks = torch.randn(num_heads, dtype=torch.float32, device=device) if head_dim == 64 else None
    q = torch.randn([num_tokens, num_heads * head_dim], dtype=torch.bfloat16, device=device)
    k = torch.randn([num_tokens, num_key_value_heads * head_dim], dtype=torch.bfloat16, device=device)
    v = torch.randn([num_tokens, num_key_value_heads * head_dim], dtype=torch.bfloat16, device=device)
    forward_args = (torch.concat([q, k, v], dim=-1), None, None)
    forward_kwargs = {
        "attention_window_size": attention_window_size if attention_window_size > 0 else None,
        "attention_sinks": sinks,
        "out_scale": out_scale,
    }
    # FIXME(kernel-limit): H20 campaign 2026-10-02 (1.3.0rc29, TRTLLM backend,
    # attention/backends/fmha/fallback.py:122): the context FMHA faults with
    # cudaErrorIllegalAddress exactly when the LAST sequence's packed Q/O byte
    # offset (batch-1) * input_len * num_heads * head_dim * 2 >= 2**31 —
    # verified cell-by-cell on the (96,*,256), (64,*,256) and (96,8,128) grids
    # (b16 s3072 h96 d256 crashes, b8 s6144 passes at the same token count),
    # i.e. a 32-bit per-sequence offset inside the kernel. Serving reaches such
    # batches only with max_num_tokens >= ~44k tokens per context forward.
    # Unverified against the kernel source; cases above the boundary keep
    # failing into the classified log until the claim is confirmed (or a bump
    # fixes it). Generation has a second family: fp8 KV with GQA ratio 24 or
    # 32 (96/4 at d64/128/256; 32/1, 64/2, 128/4 at d256) fails with
    # cudaErrorLaunchFailure / IMA at every batch and KV length while the same
    # shapes pass with a bf16 KV cache.
    attn.forward(*forward_args, attn_metadata, **forward_kwargs)

    # Use benchmark_with_power context manager
    def kernel_func():
        attn.forward(*forward_args, attn_metadata, **forward_kwargs)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=test_ite,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]

    # write result
    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"
    kv_cache_dtype_str = "bfloat16"
    if use_fp8_kv_cache:
        kv_cache_dtype_str = "fp8"
    if use_fp8_context_fmha:
        dtype_str = "fp8"
    else:
        dtype_str = "bfloat16"

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "window_size": attention_window_size,
                "beam_width": 1,
                "attn_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "step": step,
                "latency": latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source="torch_flow",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
    kv_cache_manager.shutdown()


def get_context_attention_test_cases():
    return plan_cases("attention_context", "trtllm", get_sm_version())


def get_generation_attention_test_cases():
    return plan_cases("attention_generation", "trtllm", get_sm_version())


if __name__ == "__main__":
    test_cases = get_context_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case, perf_filename=PerfFile.CONTEXT_ATTENTION)

    test_cases = get_generation_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case, perf_filename=PerfFile.GENERATION_ATTENTION)
