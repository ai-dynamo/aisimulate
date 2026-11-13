# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM dense attention collector.

Constructs a single TRT-LLM torch-flow attention layer and synthetic metadata to
benchmark context and generation attention. This file owns TRT-LLM cache manager
setup, quantization flags and perf-row formatting; the cases come from the
shared attention plan (collector/op_plans.py).

The timed call is TrtllmAttention.forward, the call serving's Attention._attn_impl
makes (_torch/attention/attention.py:988-1011). Inside it the FMHA manager picks the
kernel library per request exactly as in serving (attention/backends/trtllm.py:2563-2580):
FlashInferTrtllmGenFmha on SM100/103 (fmha/flashinfer_trtllm_gen.py:398-420) ahead of
FallbackFmha, the thop attention op (fmha/registry.py:42-54); the fallback serves
SM89/90/120, small-batch BF16-KV context on SM100/103
(flashinfer_trtllm_gen.py:627-652) and every request FlashInfer declines (e.g. a
heads ratio above 32 in generation, :774-779). No environment variable steers the
choice, and the decode kernel inside the thop op (XQA or MMHA) is the op's own
per-(batch, kv_len) heuristic (decoderXQARunner.cpp:79-90).
"""

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

__compat__ = "trtllm==1.4.0rc2"


def _deployment_tokens_per_block() -> int:
    """KV page the deployment actually serves with, not a collector constant.

    The generator renders trtllm engines without ``kv_cache_config.tokens_per_block``,
    so serving runs TRT-LLM's own KvCacheConfig default (llmapi/llm_args.py:5040, 32;
    dense models keep it, pyexecutor/py_executor_creator.py:589-608). The page is part
    of the kernel identity on Blackwell: the sm100 kernel names carry it
    (fmhaSm100a…PagedKvCausalP64… vs serving's …P32…), and a pinned 64 once made the
    sm100 attention gates diverge on nothing but the page while the sm90 names hide it
    (B200, opharness finding sm100_sglang_trtllm_gate_verdicts_2026_10_04).
    """
    return int(KvCacheConfig().tokens_per_block)


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
    # backend (TorchLlmArgs.attn_backend, llmapi/llm_args.py:6664-6667;
    # get_attention_backend, attention/backends/utils.py:33-57).
    if attn_backend_name != "TRTLLM":
        raise ValueError(f"Unsupported TRT-LLM dense attention backend: {attn_backend_name}")

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

    # out_scale mirrors serving's Attention._attn_impl (attention/attention.py:975-979):
    # a model with fp8 weights and activations passes o_proj.inv_input_scale, a model
    # without weight/activation quant passes None, fp8 KV cache or not
    # (_use_quantize_output, attention.py:869-872). The backend writes fp8 output only
    # when its own quant config has such weights and an fp8 KV cache
    # (get_quantize_output_dtype, attention/backends/trtllm.py:2030-2040, read by
    # create_output via forward, :2286-2298), and the thop op reads the scale only on
    # that fp8 path (thop/attentionOp.cpp:807-809, 1382-1384). Per case:
    # - fp8_kv_cache_and_context_fmha (context only): the fp8-model deployment;
    #   quant_algo FP8 above plus a dummy unit o_proj.inv_input_scale -> fp8 output.
    # - fp8_kv_cache, both phases: an unquantized model with an fp8 KV cache, the
    #   deployment the serving gates probe (bf16 Llama-3.1-8B, kv fp8) -> None and
    #   bf16 output. With paged context FMHA off (no chunked prefill, block reuse or
    #   draft tokens, as the generator renders it) the context FMHA runs in bf16 on
    #   SM89/90/120 and writes the fp8 KV cache; on SM100/103 FlashInfer trtllm-gen
    #   quantizes Q to fp8 for an fp8 KV cache (flashinfer_trtllm_gen.py:731-735, 830-843).
    # Deviation (generation, every SM): the sweep has no fp8-model generation case.
    # Serving's fp8 and nvfp4 checkpoints with an fp8 KV cache decode with fp8 (or
    # nvfp4) attention output, which on SM100/103 also quantizes Q to fp8 in generation
    # (flashinfer_trtllm_gen.py:731-735) and on SM89/90/120 builds the fp8-output XQA
    # (common/attentionOp.cpp:315); these rows time the bf16-Q, bf16-output decode.
    # Direction unmeasured; decode reads the same fp8 KV either way. Retire with an
    # fp8-model precision case in cases/base_ops/attention_generation.yaml.
    if use_fp8_context_fmha:
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
    # RoPE (attention/attention.py:749-750, 768-777), which the TRTLLM backend does
    # (support_fused_rope, attention/backends/trtllm.py:2599-2601), so RoPE runs inside the
    # op's QKV preprocessing, or inside the SM90 fp8-KV XQA kernel when the rotary dim is a
    # multiple of 16 no larger than head_dim and the type is neox/gptj/longrope
    # (decoderXQAImplJIT/kernelUtils.cpp:85-102).
    # Deviation (every SM): one RoPE for every shape, neox with dim 128; serving takes type
    # and dim from the model (RopeParams.from_config, attention/backends/interface.py:784-830;
    # GPT-OSS: yarn, dim 64, models/modeling_gpt_oss.py:55-58). Only the rotated element
    # count and the SM90 in-kernel placement depend on it; head_dim 64 rows (dim 128 >
    # head_dim) take the separate preprocessing kernel, as GPT-OSS's yarn does in serving.
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
    # (pyexecutor/_util.py:3326-3327, 3867-3892); SELFKONLY is reserved for MLA
    # (_util.py:3575-3585). Regular GQA/MHA has separate K and V tensors, so both
    # TRTLLM and FlashInfer paths must benchmark the full serving layout.
    # Deviation (every SM): serving's KvCacheConfig.use_kv_cache_manager_v2 now defaults to
    # True (llmapi/llm_args.py:5057-5064), so dense models get KVCacheManagerV2
    # (_non_hybrid_kv_cache_manager_cls, pyexecutor/_util.py:123-128); this builds the V1
    # KVCacheManager. The timed kernels and their selection read the cache only through
    # the block offsets, pool pointers and page size that prepare() copies from either
    # manager (attention/backends/trtllm.py:1082-1114), so the rows are expected to match;
    # retire by building the class get_kv_cache_manager_cls resolves after an A/B on one
    # system per SM generation.
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
        # (pyexecutor/engine/runners/decoder/inputs.py:1012, 1368, 1452, 1667,
        # assigned at :2055); the field is part of the shared AttentionMetadata
        # contract (attention/backends/interface.py:151-155). With
        # chunked_prefill=False the current chunk IS the full prompt, so
        # input_seq_lens matches both phases here.
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
    # FIXME(kernel-limit): H20 campaign at the previous pin (SM90, FallbackFmha,
    # attention/backends/fmha/fallback.py:311): the context FMHA faults with
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
    # shapes pass with a bf16 KV cache. Re-checked by source at this pin: the XQA
    # kernel sources (cpp/kernels/xqa) are unchanged, and the SM90 fmha_v2 changes
    # (cpp/kernels/fmha_v2) are SageAttention paths, so both observations still apply
    # until a targeted SM90 run says otherwise.
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
