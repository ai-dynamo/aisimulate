# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM MLA attention collector (mla_context, mla_generation).

Times the attention calls of serving's MLA module (_torch/attention/mla.py) on the
backends serving builds for them with create_attention (MLA.__init__, mla.py:599-652).
The context row times MLA.mha.forward of forward_context_default (mla.py:1196-1207);
the generation row times MLA.mqa.mla_rope_generation plus MLA.mqa.forward of
forward_absorption_generation (mla.py:1691-1704, 1764-1782). The absorption BMMs
around them are collect_mla_bmm's rows.

Inside TrtllmAttention.forward the FMHA manager picks the kernel library per request
exactly as in serving (attention/backends/trtllm.py:2562-2585; library order
fmha/registry.py:42-56):
- context, every SM: the thop attention op. FlashInfer TRTLLM-Gen declines MLA
  context (fmha/flashinfer_trtllm_gen.py:682-683) and the CuTe DSL library is
  generation-only. With an FP8 KV cache the op quantizes Q/K/V to FP8 itself on
  SM90/100/103/107/120 (thop/attentionOp.cpp:1466-1470).
- generation, SM100/103: CuteDslMlaFmha for a BF16 KV cache at 6, 12, 16 or 96 local
  heads, and for an FP8 KV cache at the (local heads, batch) of its measured table
  (fmha/cute_dsl_mla.py:382-437). FlashInfer TRTLLM-Gen declines DeepSeek dimensions
  on 32-token pages and 64 < heads < 128 (flashinfer_trtllm_gen.py:378-386, 539-556),
  so every other case runs the thop op's trtllm-gen MLA kernels
  (common/attentionOp.h:623, common/attentionOp.cpp:3193-3229).
- generation, SM90: FlashMLA on 64-token pages (thop/attentionOp.cpp:1474,
  thop/dsv3RopeOp.cpp:204).
- generation, SM120: the op's XQA MLA kernel (common/attentionOp.cpp:3192).
The CuTe DSL decode is an AutoTuner op (custom_ops/cute_dsl_custom_ops.py:12901-12913),
but it has one tactic per batch bucket, the one default_tactic returns on a cache
miss (:12165-12178, 12301-12304, 12474-12495), so serving's warmup tuning pass would
not change what these rows time.
"""

__compat__ = "trtllm==1.4.0rc2"

import math
from dataclasses import dataclass

import tensorrt_llm
import torch
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan_cases
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _mla_tokens_per_block() -> int:
    """KV page serving uses for DeepSeek-dim MLA (kv_lora_rank + rope = 576) on this SM.

    SM90 serving enables FlashMLA for it (ModelConfig.enable_flash_mla,
    _torch/model_config.py:466-475) and forces 64-token pages
    (pyexecutor/py_executor_creator.py:859-873); the attention op runs FlashMLA only
    on that page (thop/attentionOp.cpp:1474). Every other SM keeps TRT-LLM's
    KvCacheConfig default (llmapi/llm_args.py:5040-5041), 32.
    """
    return 64 if get_sm_version() == 90 else int(KvCacheConfig().tokens_per_block)


def get_context_mla_test_cases():
    return plan_cases("mla_context", "trtllm", get_sm_version())


def get_generation_mla_test_cases():
    return plan_cases("mla_generation", "trtllm", get_sm_version())


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 128
    num_kv_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    hidden_size: int = 7168
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 64


@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = (
        {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
    )
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def run_mla(
    input_len,
    batch_size,
    output_len,
    kv_cache_dtype,
    num_heads,
    world_size,
    tp_size,
    tokens_per_block,
    warming_up,
    test_ite,
    is_context_phase,
    *,
    perf_filename,
    device="cuda:0",
):
    # collector/skips.py leaves SM80-89 out: AttentionOp aborts there on every case.
    if get_sm_version() < 90:
        raise ValueError("FIXME(kernel-limit): TRT-LLM MLA has no pre-Hopper FMHA kernel (collector/skips.py)")
    num_generation_steps = 0 if is_context_phase else 1
    # Synthetic table capacity, not a checkpoint context-limit override: the RoPE
    # table length (RopeParams.max_positions, attention/backends/interface.py:817) is
    # separate from the YaRN original length (:844-847), and the backend grows the
    # table to the cache's max_seq_len anyway (attention/backends/trtllm.py:2135, 2877).
    scenario = Scenario(
        max_position_embeddings=max(
            Scenario.max_position_embeddings, input_len + (num_generation_steps + 1) * output_len
        )
    )
    q_lora_rank = scenario.q_lora_rank
    kv_lora_rank = scenario.kv_lora_rank
    qk_nope_head_dim = scenario.qk_nope_head_dim
    qk_rope_head_dim = scenario.qk_rope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling={
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        },
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    kv_cache_tokens_per_block = tokens_per_block
    dtype = scenario.dtype

    assert num_heads % tp_size == 0, "num_heads != N * tp_size"
    num_heads = num_heads // tp_size
    num_kv_heads = num_heads
    context_sequence_lengths = [input_len for _ in range(batch_size)]

    _run_attn_for_backend(
        "TRTLLM",
        num_heads,
        num_kv_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_config,
        kv_cache_tokens_per_block,
        device,
        dtype,
        kv_cache_dtype,
        context_sequence_lengths,
        output_len,
        num_generation_steps,
        world_size,
        tp_size,
        warming_up,
        test_ite,
        is_context_phase,
        perf_filename,
    )


def _run_attn_for_backend(
    backend_name,
    num_heads,
    num_kv_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    rope_config,
    kv_cache_tokens_per_block,
    device,
    dtype,
    kv_cache_dtype,
    context_sequence_lengths,
    generation_seq_len_q,
    num_generation_steps,
    world_size,
    tp_size,
    warming_up,
    test_ite,
    is_context_phase,
    perf_filename,
):
    max_context_sequence_length = max(context_sequence_lengths)
    max_num_contexts = len(context_sequence_lengths)

    torch.cuda.set_device(device)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # DeepseekV3Attention's positional embedding (models/modeling_deepseekv3.py:786-790).
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    # MLA.__init__ folds the YaRN mscale into q_scaling (mla.py:552-559).
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = pos_embd_params.rope.mscale_all_dim
    scaling_factor = pos_embd_params.rope.scale
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    quant_config = None
    if kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)

    mla_kwargs = dict(
        pos_embd_params=pos_embd_params,
        quant_config=quant_config,
        q_scaling=q_scaling,
        is_mla_enable=True,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        predicted_tokens_per_seq=1,
    )
    if is_context_phase:
        # MLA.mha (mla.py:633-652).
        attn_mla = create_attention(
            backend_name,
            0,
            num_heads,
            head_dim=qk_head_dim,
            num_kv_heads=num_kv_heads,
            v_head_dim=v_head_dim,
            **mla_kwargs,
        )
    else:
        # MLA.mqa (mla.py:599-623): one latent KV head of kv_lora_rank + rope, whose
        # value is the latent itself.
        attn_mla = create_attention(
            backend_name,
            0,
            num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            num_kv_heads=1,
            v_head_dim=kv_lora_rank,
            hidden_size=rope_config.hidden_size,
            dtype=dtype,
            rope_append=True,
            **mla_kwargs,
        )
    # A row must time the backend it is labeled with: verify the constructed class.
    if type(attn_mla).__name__ != "TrtllmAttention":
        raise RuntimeError(
            f"create_attention resolved {type(attn_mla).__name__} for backend {backend_name} (expected TrtllmAttention)"
        )

    mapping = Mapping(world_size=world_size, tp_size=tp_size, rank=0)
    max_tokens = (
        (
            max_context_sequence_length
            + (num_generation_steps + 1) * generation_seq_len_q
            + kv_cache_tokens_per_block
            - 1
        )
        // kv_cache_tokens_per_block
        * kv_cache_tokens_per_block
        * max_num_contexts
    )
    # Serving builds the MLA cache as SELFKONLY, one head of kv_lora_rank + rope
    # (pyexecutor/_util.py:3575-3607).
    # Deviation (every SM): serving's KvCacheConfig.use_kv_cache_manager_v2 defaults to True
    # (llmapi/llm_args.py:5057-5064), so MLA models get KVCacheManagerV2
    # (get_kv_cache_manager_cls, pyexecutor/_util.py:123-128, 448); this builds the V1
    # KVCacheManager, whose impl.add_token grows a generation request below. The timed
    # kernels read the cache only through the block offsets, pool pointers and page size
    # that prepare() copies from either manager (attention/backends/trtllm.py:1082-1114),
    # and the CuTe DSL decode addresses both pool layouts (fmha/cute_dsl_mla.py:600-626),
    # so the rows are expected to match. Retire by building KVCacheManagerV2 after an A/B
    # on one system per SM generation.
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        tokens_per_block=kv_cache_tokens_per_block,
        max_seq_len=max(context_sequence_lengths) + (num_generation_steps + 1) * generation_seq_len_q,
        max_batch_size=len(context_sequence_lengths),
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    # Register the context sequences through the framework's add_dummy_requests
    # warmup path, like the sibling attention collectors. It returns None when the
    # pool cannot hold them, which would leave the block table unset.
    if (
        kv_cache_manager.add_dummy_requests(
            list(range(len(context_sequence_lengths))),
            token_nums=list(context_sequence_lengths),
        )
        is None
    ):
        kv_cache_manager.shutdown()
        raise RuntimeError("FIXME(kernel-limit): does not fit: the KV cache pool cannot hold the case's requests")

    # Serving enables FlashMLA metadata for DeepSeek-dim MLA on SM90 in both phases
    # (build_attention_metadata, pyexecutor/engine/metadata.py:86; ModelConfig.enable_flash_mla,
    # _torch/model_config.py:466-475).
    enable_flash_mla = get_sm_version() == 90
    if is_context_phase:
        attn_metadata = attn_mla.Metadata(
            seq_lens=torch.tensor(context_sequence_lengths, dtype=torch.int),
            request_ids=list(range(len(context_sequence_lengths))),
            max_num_requests=len(context_sequence_lengths),
            num_contexts=len(context_sequence_lengths),
            prompt_lens=context_sequence_lengths,
            max_num_tokens=max(context_sequence_lengths),
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0 for _ in context_sequence_lengths],
            ),
            mapping=mapping,
            enable_flash_mla=enable_flash_mla,
        )
    else:
        for req_id in range(len(context_sequence_lengths)):
            for _ in range(generation_seq_len_q):
                kv_cache_manager.impl.add_token(req_id)
        attn_metadata = attn_mla.Metadata(
            seq_lens=torch.tensor([generation_seq_len_q] * len(context_sequence_lengths), dtype=torch.int),
            request_ids=list(range(len(context_sequence_lengths))),
            max_num_requests=len(context_sequence_lengths),
            num_contexts=0,
            prompt_lens=context_sequence_lengths,
            max_num_tokens=max(context_sequence_lengths),
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=list(context_sequence_lengths),
            ),
            mapping=mapping,
            enable_flash_mla=enable_flash_mla,
        )
    attn_metadata.prepare()

    if is_context_phase:
        # Packed context token count; case grids use uniform context lengths,
        # so this matches the previous last-length * num-contexts sizing.
        total_ctx_tokens = sum(context_sequence_lengths)
        ctx_compressed_kv = torch.randn(
            [total_ctx_tokens, kv_lora_rank],
            dtype=dtype,
            device=device,
        )

        ctx_k_pe = torch.randn(
            [total_ctx_tokens, qk_rope_head_dim],
            dtype=dtype,
            device=device,
        )

        ctx_q = torch.randn(
            [total_ctx_tokens, num_heads * qk_head_dim],
            dtype=dtype,
            device=device,
        )

        # forward_context_default's V is the v half of a [tokens, H * (nope + v)] buffer
        # on every path (mla.py:1168-1171, 1179-1183), and its K a [tokens, H * qk_head_dim]
        # buffer (:1151).
        ctx_kv = torch.randn(
            [
                total_ctx_tokens,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
            ],
            dtype=dtype,
            device=device,
        )
        ctx_k_nope, ctx_v = ctx_kv.split([num_kv_heads * qk_nope_head_dim, num_kv_heads * v_head_dim], dim=-1)
        ctx_k_nope = ctx_k_nope.view(-1, num_kv_heads, qk_nope_head_dim)
        ctx_k = torch.cat(
            [ctx_k_nope, ctx_k_pe.view(-1, 1, qk_rope_head_dim).expand(-1, num_kv_heads, -1)],
            dim=-1,
        )
        ctx_k = ctx_k.view(-1, num_kv_heads * qk_head_dim)

        q = ctx_q
        k = ctx_k
        v = ctx_v
        compressed_kv = ctx_compressed_kv
        k_pe = ctx_k_pe

        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
        attn_mla.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
        )
    else:
        num_tokens = generation_seq_len_q * len(context_sequence_lengths)
        compressed_kv = torch.randn(
            [num_tokens, kv_lora_rank],
            dtype=dtype,
            device=device,
        )

        k_pe = torch.randn(
            [num_tokens, qk_rope_head_dim],
            dtype=dtype,
            device=device,
        )

        fused_q = torch.randn(
            [
                num_tokens,
                num_heads * (kv_lora_rank + qk_rope_head_dim),
            ],
            dtype=dtype,
            device=device,
        )

        # q_pe is the rope slice of the q_b_proj output (mla.py:1606-1608); the rope
        # kernel reads it through its strides.
        q = torch.randn([num_tokens, num_heads * qk_head_dim], dtype=dtype, device=device)
        _q_nope, q_pe = q.view([-1, num_heads, qk_head_dim]).split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

        num_seqs = len(context_sequence_lengths)
        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)

        # Serving allocates the FP8 scale and quantized-Q buffers when the MQA backend
        # has an FP8 KV cache (forward_absorption_generation, mla.py:1631-1646), read from
        # TrtllmAttention.update_quant_config (attention/backends/trtllm.py:1836-1839).
        # A missing attribute or a state that disagrees with the requested KV dtype would
        # set up the other path under this row's label, so both raise.
        if not hasattr(attn_mla, "has_fp8_kv_cache"):
            raise RuntimeError(
                "TrtllmAttention.has_fp8_kv_cache missing; refusing to default the MLA generation quant path to BF16"
            )
        has_fp8_kv_cache = attn_mla.has_fp8_kv_cache
        expected_fp8_kv = kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8
        if bool(has_fp8_kv_cache) != expected_fp8_kv:
            raise RuntimeError(
                f"MLA generation quant-state mismatch: backend has_fp8_kv_cache="
                f"{has_fp8_kv_cache}, requested kv_cache_dtype={kv_cache_dtype}"
            )
        if has_fp8_kv_cache:
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
            quant_q_buffer = torch.empty(
                num_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim), dtype=torch.uint8, device=device
            )
        else:
            mla_bmm1_scale = None
            mla_bmm2_scale = None
            quant_q_buffer = None

        # Call mla_rope_generation before forward
        attn_mla.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            attn_metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )
        attn_mla.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=latent_cache,
            q_pe=q_pe,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
        )

    # Use benchmark_with_power context manager
    def kernel_func():
        if is_context_phase:
            attn_mla.forward(
                q,
                k,
                v,
                attn_metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
            )
        else:
            attn_mla.mla_rope_generation(
                fused_q,
                q_pe,
                latent_cache,
                attn_metadata,
                cu_q_seqlens,
                cu_kv_seqlens,
                fmha_scheduler_counter,
                mla_bmm1_scale,
                mla_bmm2_scale,
                quant_q_buffer,
            )
            attn_mla.forward(
                fused_q,
                None,
                None,
                attn_metadata,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=fmha_scheduler_counter,
                mla_bmm1_scale=mla_bmm1_scale,
                mla_bmm2_scale=mla_bmm2_scale,
                quant_q_buffer=quant_q_buffer,
            )

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
        isl = max_context_sequence_length
        step = 0
    else:
        isl = 1
        step = max_context_sequence_length

    dtype_str = "bfloat16"
    if kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8:
        dtype_str = "fp8"

    log_perf(
        item_list=[
            {
                # The attention op quantizes context Q/K/V to FP8 itself for an FP8 KV
                # cache on these SMs (thop/attentionOp.cpp:1466-1470); the BF16 Python
                # inputs do not describe the FMHA compute precision.
                "mla_dtype": (
                    dtype_str if is_context_phase and get_sm_version() in (90, 100, 103, 107, 120) else "bfloat16"
                ),
                "kv_cache_dtype": dtype_str,
                "num_heads": num_heads,
                "batch_size": len(context_sequence_lengths),
                "isl": isl,
                "tp_size": tp_size,
                "step": step,
                "latency": latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"mla_{'context' if is_context_phase else 'generation'}",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    kv_cache_manager.shutdown()
