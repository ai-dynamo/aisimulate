# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one place that keeps cases out of a collection.

An entry in ``SKIPS`` skips a case when every key matches:

- ``backend``, ``op`` and ``sm``: the run;
- ``model``: the ``architecture`` of the case file a model row comes from;
- ``dtype``: the case's dtype or quant mode;
- ``dispatch``: a kernel the op plan picks for the backend; the entry leaves
  the case out on that kernel only, and it still runs on any other;
- any other key: the case field (or property) of that name.

A list matches any of its values; a callable is a predicate on the value. An
entry without ``sm`` applies on every SM. A ``model`` entry drops that file's
rows before cases are merged, so a shape another model also needs is still
collected; besides ``model`` it may name only ``backend``, ``op`` and ``sm``.

Add an entry only when the cases would be mismeasured, would hang, would fail
as a whole block, or come from a model the backend cannot serve. Write the
reason as a comment above the entry, and edit or delete the entry when the
reason stops being true. Everything else runs, and its failures are recorded.

SM lists name the SMs the collector targets: 80, 86, 89, 90, 100, 103, 120.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

_DSA_MODULE_OPS = [
    "dsa_context_module",
    "dsa_generation_module",
    "dsa_context_module_skip_indexer",
    "dsa_generation_module_skip_indexer",
]
_GLM5_SPARSE_OPS = ["glm5_mqa_logits_module", "glm5_topk_module", "glm5_dsa_attn_module"]
_DSA_AND_SPARSE_OPS = [
    *_DSA_MODULE_OPS,
    "dsv4_paged_mqa_logits_module",
    "dsv4_csa_topk_calib",
    *_GLM5_SPARSE_OPS,
]
_DSV4_ATTENTION_OPS = [
    "dsv4_csa_context_module",
    "dsv4_csa_generation_module",
    "dsv4_hca_context_module",
    "dsv4_hca_generation_module",
]
_MSA_MODULE_OPS = ["msa_context_module", "msa_generation_module"]
_ATTENTION_OPS = ["attention_context", "attention_generation"]

SKIPS: list[dict] = [
    # SGLang shards DeepSeek-V4 attention by its published parallel context only
    # (MQALayer reads get_parallel().attn_tp_size; srt/models/deepseek_v4.py), which a
    # single-process collector cannot publish, so collect_dsv4_attn refuses every TP>1
    # row before building the model (_refuse_tp_emulation): the whole TP>1 block fails.
    # Earlier tables timed the TP=1 module under the TP>1 label. Delete this entry when
    # the collector times one rank of a TP>1 module.
    {"backend": "sglang", "op": _DSV4_ATTENTION_OPS, "tp_size": lambda tp: int(tp) > 1},
    # The sparse-attention (DSA) modules and the DeepSeek-V4 and GLM-5 sparse
    # kernels (DeepGEMM fp8_mqa_logits, FlashMLA sparse, JIT top-k) need Hopper
    # (SM90) or newer.
    {"op": _DSA_AND_SPARSE_OPS, "sm": [80, 86, 89]},
    # DeepSeek-V4 attention runs only on FlashMLA (SM90, SM10x) and SM100+ paths;
    # on SM89 serving fails the same way.
    {"op": _DSV4_ATTENTION_OPS, "sm": [80, 86, 89]},
    # compute_scale times FP8 quantization, which needs SM89 or newer.
    {"op": "compute_scale", "sm": [80, 86]},
    # FP8 tensor cores start at Ada (SM89).
    {"dtype": ["fp8", "fp8_static", "fp8_block"], "sm": [80, 86]},
    # MXFP4 weight-only MoE starts at SM89: SGLang maps no runner below it and
    # TRT-LLM runs it on SM90 only.
    {"dtype": "w4a16_mxfp4", "sm": [80, 86]},
    # FP4 tensor cores, which NVFP4 and MXFP8 x MXFP4 need, start at Blackwell (SM100).
    # Below it vLLM serves NVFP4 checkpoints weight-only (Marlin W4A16; Humming on
    # SM90, kernels/linear/nvfp4/humming.py), which is not an NVFP4 measurement.
    {"dtype": ["nvfp4", "w4a8_mxfp4_mxfp8"], "sm": [80, 86, 89, 90]},
    # SGLang: the MoE collector has no per-tensor fp8 path, so every case would
    # fail. Serving runs ModelOpt FP8 checkpoints through ModelOptFp8MoEMethod.
    {"backend": "sglang", "op": "moe", "dtype": "fp8", "sm": [89, 90, 100, 103, 120]},
    # SGLang: a MoE case runs on the runner serving picks, pinned per quant mode
    # and SM in cases/base_ops/moe.yaml (moe_sglang.backends). int4_wo has no pin
    # on SM80/86, so these cases would time a guessed runner.
    {"backend": "sglang", "op": "moe", "dtype": "int4_wo", "sm": [80, 86]},
    # SGLang serves GPT-OSS MXFP4 on SM120 on flashinfer_mxfp4
    # (arg_groups/model_overrides/gpt_oss.py:69-74), whose SM120 leaf quantizes
    # activations to MXFP8 (layers/quantization/mxfp4.py:426-427,
    # moe_runner/flashinfer_cutlass.py:541-548): the w4a8_mxfp4_mxfp8 cases. A
    # W4A16 case would time a mode serving does not run.
    {
        "backend": "sglang",
        "op": "moe",
        "dtype": "w4a16_mxfp4",
        "sm": [120],
        "model_name": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    },
    # SGLang serves GPT-OSS MXFP4 on SM100/103 on flashinfer_mxfp4 at every EP
    # (model_overrides/gpt_oss.py:64-75) at the default precision, which quantizes
    # activations to MXFP8 on the trtllm-gen leaf (arg_groups/fields/exec_.py:892-897,
    # moe_runner/flashinfer_trtllm.py:1145-1166); the B200 serving probe of
    # gpt-oss-120b ran bmm_*_MxE2m1MxE4m3 GEMMs (2026-11-08). A W4A16 case would time
    # a mode serving does not run; the w4a8_mxfp4_mxfp8 cases are serving's.
    {
        "backend": "sglang",
        "op": "moe",
        "dtype": "w4a16_mxfp4",
        "sm": [100, 103],
        "model_name": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    },
    # SGLang serves Kimi-K3 on SM120 with BF16 activations: its flashinfer_mxfp4
    # default is SM100/103/107 only (model_overrides/kimi_k3.py:172-190), and auto
    # resolves Mxfp4MoEMethod to the Triton runner on upcast BF16 experts
    # (mxfp4.py:1443-1450, 1115-1133). An MXFP8-activation case would time a mode
    # serving does not run.
    {
        "backend": "sglang",
        "op": "moe",
        "dtype": "w4a8_mxfp4_mxfp8",
        "sm": [120],
        "model_name": "moonshotai/Kimi-K3",
    },
    # SGLang serves Kimi-K3 on SM100/103 on flashinfer_mxfp4 (kimi_k3.py:172-190)
    # at the default precision, which quantizes activations to MXFP8 on the
    # trtllm-gen leaf (arg_groups/fields/exec_.py:892-897,
    # moe_runner/flashinfer_trtllm.py:1145-1166): the w4a8_mxfp4_mxfp8 cases. A
    # W4A16 case (BF16 activations) would time a mode serving does not run.
    {
        "backend": "sglang",
        "op": "moe",
        "dtype": "w4a16_mxfp4",
        "sm": [100, 103],
        "model_name": "moonshotai/Kimi-K3",
    },
    # TRT-LLM has no MoE implementation for int4_wo (W4A16) on any SM: building
    # the layer rejects every case.
    {"backend": "trtllm", "op": "moe", "dtype": "int4_wo", "sm": [80, 86, 89, 90, 100, 103, 120]},
    # TRT-LLM: on SM89 and SM120 only CutlassFusedMoE serves these formats, and
    # it runs fp8_block on SM90/120 only and w4a16_mxfp4 on SM90 only, so
    # building the layer rejects every case.
    {"backend": "trtllm", "op": "moe", "dtype": "fp8_block", "sm": [89]},
    {"backend": "trtllm", "op": "moe", "dtype": "w4a16_mxfp4", "sm": [89, 120]},
    # TRT-LLM cannot serve Llama 4 fp8_block MoE on SM100/103: AUTO resolves
    # FP8_BLOCK_SCALES to TRTLLM there (ModelConfig.resolve_moe_backend,
    # _torch/model_config.py), resolution picks TrtllmTrtllmGenFp8BlockScalesImpl
    # (MoEProblem carries no apply_router_weight_on_input), and create_moe_backend
    # rejects Llama4MoE's apply_router_weight_on_input=True (moe/fused_moe/create_moe.py;
    # TRTLLM_GEN_CAPABILITIES, trtllm_gen/identity.py). CUTLASS has no SM100 fp8_block
    # leaf; only an explicit DEEPGEMM backend serves it. Every case fails at construction.
    {
        "backend": "trtllm",
        "op": "moe",
        "sm": [100, 103],
        "dtype": "fp8_block",
        "model_name": ["meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"],
    },
    # vLLM never splits routed experts by TP and EP together: with EP on it
    # turns the whole TP x DP world into EP, so a case with tp > 1 and ep > 1
    # would time that EP layout under the wrong key.
    {"backend": "vllm", "op": "moe", "parallel": "tp_ep", "sm": [80, 86, 89, 90, 100, 103, 120]},
    # vLLM serves Llama 4 MoE on SM100/103 on FlashInfer TRT-LLM monolithic kernels
    # that route from logits (oracle/unquantized.py _get_priority_backends;
    # experts/trtllm_bf16_moe.py TrtLlmBf16ExpertsMonolithic and trtllm_nvfp4_moe.py
    # TrtLlmNvFp4ExpertsMonolithic, RoutingMethodType.Llama4). Below 4 tokens FlashInfer
    # routes Llama 4 in routingIndicesWarpKernel (trtllm_fused_moe_routing_llama4.cu),
    # which with EP > 1 reserves GEMM tiles for non-local experts but writes
    # permuted_idx_to_token_idx for local experts only, so FC1 gathers stale token ids.
    # The illegal memory access at 1 or 2 tokens poisons the CUDA context and ends the
    # case: almost every EP > 1 Llama 4 case of these dtypes kept only its 1-token row.
    # Serving runs the same kernel at 1-3 tokens. TP-only cases have no non-local
    # experts and fp8_block routes in vLLM (top-k ids), so neither is affected. Delete
    # when the pinned FlashInfer's warp kernel initializes the non-local tiles' rows.
    {
        "backend": "vllm",
        "op": "moe",
        "sm": [100, 103],
        "dtype": ["bfloat16", "nvfp4"],
        "parallel": "ep",
        "model_name": ["meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"],
    },
    # vLLM serves MXFP4 MoE on SM120 without MXFP8 activations (that backend
    # runs only on explicit request), so these cases would time a path serving
    # does not take.
    {"backend": "vllm", "op": "moe", "dtype": "w4a8_mxfp4_mxfp8", "sm": [120]},
    # vLLM's collector builds the MXFP8-activation MoE path for the DeepSeek-V4
    # fp4-expert checkpoints only and rejects every other packaged config, and
    # vLLM serves the other MXFP4 checkpoints with BF16 activations by default,
    # so these cases fail at construction on every run.
    {
        "backend": "vllm",
        "op": "moe",
        "dtype": "w4a8_mxfp4_mxfp8",
        "model_name": lambda model_name: not model_name.startswith("deepseek-ai/DeepSeek-V4"),
    },
    # vLLM's MoE collector reads a packaged HF config for every model, and none
    # is packaged for this NVFP4 alias, so its cases fail at construction on
    # every run.
    {"backend": "vllm", "op": "moe", "model_name": "RadixArk/Qwen3.8-2.4T-A95B-NVFP4"},
    # Large-EP MoE (moe_ep): SGLang's collector drives DeepEPMoE with FP8
    # activations on the DeepGEMM runner, so block-scaled FP8 is the one quant
    # mode it times. This keeps the coverage collected before.
    {"backend": "sglang", "op": "moe_ep", "dtype": "nvfp4"},
    # SGLang's moe_ep gives every routed expert one slot (no EPLB redundant-expert
    # layout), so an EPLB case would time that layout under the case's num_slots.
    {"backend": "sglang", "op": "moe_ep", "use_eplb": True},
    # SGLang's moe_ep loads the model's checkpoint (dummy weights keep its
    # quantization config), so a quant mode the checkpoint does not plan would time
    # the checkpoint's own MoE under another label.
    {"backend": "sglang", "op": "moe_ep", "own_quant": False},
    # TRT-LLM: the wide-EP fp8_block grouped GEMM is DeepGEMM-backed even under the
    # CUTLASS runner (CutlassFp8BlockScaleGemmRunner::moeGemm), and DeepGEMM's JIT
    # compiles for Hopper only (deep_gemm/compiler.cuh), so every case fails off SM90.
    {"backend": "trtllm", "op": "moe_ep", "dtype": "fp8_block", "sm": [89, 100, 103, 120]},
    # TRT-LLM has no MLA FMHA kernel below Hopper: AttentionOp asserts "Deepseek
    # should be supported by fmha" (attentionOp.cpp) and the worker aborts on every
    # case, as serving does.
    {
        "backend": "trtllm",
        "op": ["mla_context", "mla_generation", "mla_context_module", "mla_generation_module"],
        "sm": [80, 86, 89],
    },
    # TRT-LLM's MLA decode FMHA on SM100/103 tiles 16 query heads per CTA and rejects
    # more than 16 heads per rank unless they are a multiple of 16 ("numHeadsPerCta=16").
    {
        "backend": "trtllm",
        "op": "mla_generation",
        "sm": [100, 103],
        "local_heads": lambda heads: heads > 16 and heads % 16 != 0,
    },
    # vLLM runs both MLA generation BMMs as bf16 torch.bmm on NVIDIA GPUs (its
    # fp8 BMMs are ROCm only), so an fp8 row would label a kernel vLLM never runs.
    {"backend": "vllm", "op": ["mla_bmm_gen_pre", "mla_bmm_gen_post"], "dtype": "fp8", "sm": [89, 90, 100, 103, 120]},
    # SGLang: the MLA collector builds each case on the attention backend
    # serving picks, which it maps for SM89 and newer only.
    {"backend": "sglang", "op": ["mla_context", "mla_generation"], "dtype": "bfloat16", "sm": [80, 86]},
    # MLA modules: of the two backends that time them, only vLLM takes FP8 prefill
    # queries, and only on device capability family 100
    # (backend_supports_prefill_query_quantization in
    # vllm/model_executor/layers/attention/mla_attention.py). Elsewhere vLLM falls
    # back to BF16 queries, which the collector refuses to record as FP8.
    {"op": "mla_context_module", "compute_dtype": "fp8", "sm": [89, 90, 120]},
    # TRT-LLM's MLA attention takes BF16 inputs and computes in FP8 by itself when
    # the KV cache is FP8 (mFP8ContextMLA in thop/attentionOp.cpp); its FP8-KV rows
    # record that, so an FP8-compute case has no kernel of its own to time.
    {"backend": "trtllm", "op": "mla_context_module", "compute_dtype": "fp8"},
    # vLLM: FP8-KV MLA modules on SM89 were not collected before, and this keeps
    # that coverage. Triton MLA would serve them there (its FP8 KV cache needs
    # SM89+, vllm/v1/attention/backends/mla/triton_mla.py).
    {"backend": "vllm", "op": ["mla_context_module", "mla_generation_module"], "kv_cache_dtype": "fp8", "sm": [89]},
    # TRT-LLM's DSA attention reads an FP8 KV cache in the trtllm-gen sparse MLA
    # kernels of the SM100 family, and FP8-KV cases were not collected on SM90
    # before.
    {"backend": "trtllm", "op": ["dsa_context_module", "dsa_generation_module"], "kv_cache_dtype": "fp8", "sm": [90]},
    # TRT-LLM runs DSA and DeepSeek-V4 sparse MLA on SM120 only on the packed
    # fp8_ds_mla KV pool: TrtllmAttention refuses any other pool ("requires
    # kv_cache_config.dtype='fp8_ds_mla'", attention/backends/trtllm.py). The DSA
    # module builds BF16 and FP8 pools, and the DeepSeek-V4 module builds fp8_ds_mla
    # on SM90 only, so every case fails, as an RTX PRO 6000 run of the DSA modules
    # showed.
    {"backend": "trtllm", "op": ["dsa_context_module", "dsa_generation_module", *_DSV4_ATTENTION_OPS], "sm": [120]},
    # DSA modules on SM100/103 run trtllm-gen sparse-MLA kernels, which start at 8
    # query heads per rank (TRT-LLM's StoreSmemP.h tile, FlashInfer's
    # trtllm_fmha_kernel_launcher.cu), so fewer heads per rank fail on every case,
    # as serving does.
    {
        "backend": ["trtllm", "vllm"],
        "op": ["dsa_context_module", "dsa_generation_module"],
        "sm": [100, 103],
        "num_heads": lambda heads: heads < 8,
    },
    # vLLM serves DSA on an FP8 KV cache on SM100/103 through FLASHINFER_MLA_SPARSE,
    # whose auto backend has no kernel for 64 < heads per rank < 128: trtllm-gen
    # rejects the head range and cute-dsl the sparse top-k (flashinfer mla/_core.py).
    {
        "backend": "vllm",
        "op": ["dsa_context_module", "dsa_generation_module"],
        "sm": [100, 103],
        "kv_cache_dtype": "fp8",
        "num_heads": lambda heads: 64 < heads < 128,
    },
    # SGLang loads a checkpoint with its own quantization (load_model_runner), so a
    # DSA module on another projection GEMM would time the checkpoint's GEMM under
    # the case's gemm_type.
    {"backend": "sglang", "op": _DSA_MODULE_OPS, "own_gemm": False},
    # SGLang did not collect DSA modules with NVFP4 projections (DeepSeek-V3.2-NVFP4)
    # before; this keeps that coverage.
    {"backend": "sglang", "op": _DSA_MODULE_OPS, "gemm_type": "nvfp4"},
    # SGLang loads NVFP4 checkpoints on Blackwell only (load_model_runner has no
    # NVFP4 loader below SM100).
    {"backend": "sglang", "op": _DSA_MODULE_OPS, "checkpoint_quant": "nvfp4", "sm": [80, 86, 89, 90]},
    # SGLang: a one-token DSA prefill, timed eagerly, illegal-accesses in the indexer
    # (runtime_limits.dsa_indexer_prefill_shape_is_supported), and the sticky CUDA
    # error fails the rest of the task.
    {"backend": "sglang", "op": ["dsa_context_module", "dsa_context_module_skip_indexer"], "seq_len": 1},
    # SGLang's GLM-5 sparse kernels read only the checkpoint's geometry, which GLM-5
    # and its NVFP4 checkpoints share. Hopper rows are recorded under the BF16 GLM-5
    # checkpoint and Blackwell rows under the NVFP4 one, whose longer context covers
    # GLM-5's range. This keeps the coverage collected before.
    {"backend": "sglang", "op": _GLM5_SPARSE_OPS, "sm": [100, 103, 120], "model_path": "zai-org/GLM-5"},
    {
        "backend": "sglang",
        "op": _GLM5_SPARSE_OPS,
        "sm": [80, 86, 89, 90],
        "model_path": ["nvidia/GLM-5.2-NVFP4", "nvidia/GLM-5-NVFP4"],
    },
    # The TRT-LLM MSA collector builds a BF16 KV cache only (run_msa_module), so
    # every FP8-KV case would fail. TRT-LLM serves FP8 and NVFP4 KV on the MSA
    # backend on SM100/103 (modeling_minimaxm3.py main_kv_is_fp8/main_kv_is_nvfp4),
    # so this is a collector gap.
    {"backend": "trtllm", "op": _MSA_MODULE_OPS, "kv_cache_dtype": "fp8"},
    # TRT-LLM runs MSA off SM100/103 on its Triton reference (triton_backend.py),
    # a masked dense FP32 attention whose cost grows with isl x (isl + prefix): an
    # isl-32768 prefill takes 18 s per forward on h200, and a longer one runs out of
    # memory on its index scores. A 16K chunk stays within that up to a 32K prefix,
    # so these SMs stop there.
    {
        "backend": "trtllm",
        "op": "msa_context_module",
        "sm": [80, 86, 89, 90, 120],
        "prefix_len": lambda prefix_len: prefix_len > 32768,
    },
    # SGLang loads MiniMax-M3 as its BF16 checkpoint, and SGLang quantizes M3
    # projections only to MXFP8, so the collector refuses fp8_block and nvfp4 cases.
    {"backend": "sglang", "op": _MSA_MODULE_OPS, "gemm_type": ["fp8_block", "nvfp4"]},
    # SGLang: FP8-KV MSA modules on SM89 were not collected before, and this keeps
    # that coverage.
    {"backend": "sglang", "op": _MSA_MODULE_OPS, "kv_cache_dtype": "fp8", "sm": [89]},
    # TRT-LLM and vLLM did not collect DeepSeek-V4 modules with BF16 projections
    # before (every DeepSeek-V4 checkpoint runs FP8 block-scaled ones); this keeps
    # that coverage.
    {"backend": ["trtllm", "vllm"], "op": _DSV4_ATTENTION_OPS, "gemm_type": "bfloat16"},
    # SGLang loads a native DeepSeek-V4 checkpoint (deepseek-ai/*) with its FP8
    # projections only together with its FP4 routed experts, which need Blackwell,
    # so these modules would fail as a whole below SM100. The converted
    # sgl-project FP8 checkpoints load there.
    {
        "backend": "sglang",
        "op": _DSV4_ATTENTION_OPS,
        "sm": [90],
        "gemm_type": "fp8_block",
        "model_path": lambda model_path: model_path.startswith("deepseek-ai/"),
    },
    # SGLang: the attention collector maps a serving backend for SM89 and newer
    # only (SGLang's default attention backend table), so every SM80/86 case
    # would fail.
    {"backend": "sglang", "op": _ATTENTION_OPS, "sm": [80, 86]},
    # SGLang: fa3 (head_dim <= 256) and trtllm_mha quantize Q to FP8 themselves
    # when the KV cache is FP8 (flashattention_backend.py, trtllm_mha_backend.py),
    # so a BF16-compute prefill on an FP8 KV cache does not exist there: the row
    # would time FP8 compute under a bfloat16 label. It is the FP8-compute case.
    {
        "backend": "sglang",
        "op": "attention_context",
        "dispatch": "trtllm_mha",
        "kv_cache_dtype": "fp8",
        "attn_dtype": "bfloat16",
    },
    {
        "backend": "sglang",
        "op": "attention_context",
        "dispatch": "fa3",
        "head_dim": lambda head_dim: head_dim <= 256,
        "kv_cache_dtype": "fp8",
        "attn_dtype": "bfloat16",
    },
    # SGLang: flashinfer, triton and fa4 prefill run BF16 Q against the FP8 KV cache
    # and have no FP8-compute path (flashinfer_backend.py; triton_backend.py
    # forward_extend never quantizes Q; flashattention_backend.py
    # prepare_paged_mha_query skips the Q cast for fa4 prefill), so an
    # FP8-compute row would time BF16 compute under an fp8 label.
    {
        "backend": "sglang",
        "op": "attention_context",
        "dispatch": ["flashinfer", "triton", "fa4"],
        "attn_dtype": "fp8",
    },
    # SGLang: fa4 prefill leaves Q in BF16 on an FP8 KV cache
    # (flashattention_backend.py prepare_paged_mha_query skips the Q cast for fa4
    # prefill) and the FA4 kernel requires Q and K to share a dtype
    # (kernels/ops/attention/flash_attn/cute/interface.py _flash_attn_fwd asserts
    # it), so serving cannot run an FP8-KV prefill on fa4 (MiMo-V2 on SM100/103);
    # every such case fails (B200 attention smoke: 119 of 119).
    {"backend": "sglang", "op": "attention_context", "dispatch": "fa4", "kv_cache_dtype": "fp8"},
    # vLLM: the attention collector has no FP8-compute (FP8 FMHA) input, so an
    # FP8-compute row would re-time the FP8-KV case under an fp8 label.
    {"backend": "vllm", "op": "attention_context", "attn_dtype": "fp8"},
    # TRT-LLM off the SM100 family cannot run head_dim 512 attention: every
    # non-MLA AttentionOp checks mmha_supported(head_size) when it is initialized
    # (cpp/tensorrt_llm/common/attentionOp.cpp AttentionOp::initialize, "Head size
    # %d is not supported by MMHA"), so serving refuses these layers too, and the
    # uncaught C++ exception kills the collector worker at every case (h200
    # attention_context shard: 1,122 cases, 1,122 worker restarts). SM100/103
    # run them (b200 attention shards: no failures).
    {"backend": "trtllm", "op": _ATTENTION_OPS, "sm": [80, 86, 89, 90, 120], "head_dim": 512},
    # TRT-LLM on SM90: FP8-KV decode with 96 query heads over 4 KV heads aborts
    # the worker (launch failure, then SIGABRT) at every head_dim, batch and KV
    # length (h200 attention_generation shard: 512 cases, none passing).
    {
        "backend": "trtllm",
        "op": "attention_generation",
        "sm": [90],
        "kv_cache_dtype": "fp8",
        "num_heads": 96,
        "num_kv_heads": 4,
    },
    # TRT-LLM on SM90: FP8-KV full-attention decode at head_dim 256 with 32 query
    # heads per KV head aborts the worker once the KV length reaches 255 (same
    # shard: 169 cases); shorter KV lengths pass at every batch.
    {
        "backend": "trtllm",
        "op": "attention_generation",
        "sm": [90],
        "kv_cache_dtype": "fp8",
        "head_dim": 256,
        "window_size": 0,
        "q_per_kv": 32,
        "input_len": lambda input_len: input_len >= 255,
    },
    # SGLang: the mHC collector has no GLM-5.3-Flash route (the GLM call sites
    # were never ported to SGLang's current mHC module), so every case would fail.
    {"backend": "sglang", "op": "mhc_module", "model": "Glm5NextForConditionalGeneration"},
    # vLLM: mhc_module_perf has no architecture in its key, and DeepSeek-V4-Flash
    # shares (hc_mult, hidden_size) with GLM-5.3-Flash, whose rows vLLM
    # collects, so DeepSeek-V4 rows would merge with GLM's.
    {"backend": "vllm", "op": "mhc_module", "model": "DeepseekV4ForCausalLM"},
    # TRT-LLM cannot serve MiMo: it registers no model class for it.
    {"backend": "trtllm", "model": "MiMoForCausalLM"},
    # TRT-LLM cannot serve MiMo-V2-Flash: it registers no model class for it.
    # On SM100/103 its head_dim 192 attention also has no kernel.
    {"backend": "trtllm", "model": "MiMoV2FlashForCausalLM"},
    # TRT-LLM cannot serve Muse Glimmer: it registers no model class for it.
    {"backend": "trtllm", "model": "MuseGlimmerForConditionalGeneration"},
    # TRT-LLM cannot serve GLM-5.3-Flash on the release kda is pinned to: it
    # registers no model class for it. The release gemm and moe are pinned to
    # registers it (_torch/models/modeling_glm5_next.py:400,
    # modeling_glm5_next_vision.py:995) and the MoE collector builds its layer
    # (Glm5NextMoE), so neither is listed; drop kda when its pin moves to a
    # release with the class.
    {"backend": "trtllm", "op": "kda", "model": "Glm5NextForConditionalGeneration"},
    # TRT-LLM's mHC collector drives only DeepSeek-V4's sites and its rows carry
    # no architecture: GLM-5.3-Flash pre/post cases would be timed as DeepSeek-V4
    # rows, and fused_post_pre / expand / contract fail as a block. Serving runs
    # them through the same mHC module (modeling_glm5_next.py
    # Glm5NextModel._forward_single_phase_fused_hc, glm5_next_expand_streams,
    # glm5_next_hyper_head).
    {"backend": "trtllm", "op": "mhc_module", "model": "Glm5NextForConditionalGeneration"},
    # TRT-LLM registers no Step3p7FlashForCausalLM (the Step3p7 text model
    # registers as "Step3p5ForCausalLM", _torch/models/modeling_step3p7.py:1557),
    # so serving cannot load it; the MoE collector refuses the architecture.
    {"backend": "trtllm", "op": "moe", "model": "Step3p7FlashForCausalLM"},
    # TRT-LLM loads every MXFP4 checkpoint as W4A8_MXFP4_MXFP8 on SM100 and newer
    # unless the MoE backend is TRITON (ModelConfig.get_mxfp4_quant_algo,
    # _torch/model_config.py:730-741); AUTO resolves GPT-OSS and Kimi-K3 to TRTLLM
    # there (Kimi-K3's TRTLLM-Gen SiTu also takes only NVFP4 or W4A8_MXFP4_MXFP8,
    # KimiK3MoERuntime._check_trtllm_situ_quant, modeling_kimi_linear.py). The B200
    # serving probe of gpt-oss-120b ran bmm_*_MxE2m1MxE4m3 GEMMs (2026-11-08). A W4A16
    # row would time a mode serving does not run.
    {"backend": "trtllm", "op": "moe", "dtype": "w4a16_mxfp4", "sm": [100, 103]},
    # TRT-LLM serves fp8_block MoE on SM120 through run_triton_fp8_block_scale_moe,
    # whose moe_align_block_size launches _moe_prefix_kernel with NUM_EXPERTS = the
    # rank's expert count as a tl.arange length, which Triton requires to be a power
    # of two (_torch/moe/fused_moe/fused_moe_triton_fp8_block_scale.py:37-60, 95-160).
    # These models' local expert counts are not a power of two at any EP (288, 160 and
    # 384 experts), so every case fails at compile ("arange's range must be a power of
    # 2", GLM-5.3-Flash on RTX PRO, smoke 2026-11-08) and serving cannot run them.
    {
        "backend": "trtllm",
        "op": "moe",
        "sm": [120],
        "moe_type": "fp8_block",
        "model_name": [
            "zai-org/GLM-5.3-Flash",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "moonshotai/Kimi-K2-Instruct",
            "sgl-project/DeepSeek-V4-Pro-FP8",
        ],
    },
]

_RUN_KEYS = frozenset(("backend", "op", "sm"))
_MISSING = object()

# Case attribute names that carry a dtype or quant mode.
_DTYPE_FIELDS = ("moe_type", "gemm_type", "dtype", "kv_cache_dtype", "compute_dtype")
# Boolean case attributes that imply FP8.
_FP8_FLAG_FIELDS = ("use_fp8_kv_cache", "use_fp8_context_fmha")


def _check(skips: list[dict]) -> None:
    for index, entry in enumerate(skips):
        where = f"SKIPS[{index}]"
        if not isinstance(entry, dict) or not entry:
            raise ValueError(f"{where} must be a non-empty dict")
        for key, value in entry.items():
            if value in ([], (), "", None):
                raise ValueError(f"{where}.{key} must not be empty")
        if "model" in entry:
            extra = sorted(set(entry) - _RUN_KEYS - {"model"})
            if extra:
                raise ValueError(f"{where}: a model entry may name only backend, op and sm besides model, not {extra}")
            if not all(isinstance(model, str) for model in _values(entry["model"])):
                raise ValueError(f"{where}.model must name case-file architectures")
        if "sm" in entry and not all(type(sm) is int for sm in _values(entry["sm"])):
            raise ValueError(f"{where}.sm must be SM versions")


def _values(want) -> list:
    return list(want) if isinstance(want, (list, tuple, set, frozenset)) else [want]


def _matches(want, value) -> bool:
    if callable(want):
        return bool(want(value))
    return value in want if isinstance(want, (list, tuple, set, frozenset)) else value == want


def _applies(entry: dict, backend: str | None, op: str | None, sm: int | None) -> bool:
    """Whether an entry's run keys match this backend, op and SM."""
    for key, value in (("backend", backend), ("op", op), ("sm", sm)):
        if key in entry and (value is None or not _matches(entry[key], value)):
            return False
    return True


def model_entries(backend: str, op: str | None, sm: int | None) -> list[tuple[dict, frozenset[str]]]:
    """The ``model`` entries that apply to this backend, op and SM, each with the architectures it names.

    With ``op`` None, only the entries that leave a model out of every op.
    """
    return [
        (entry, frozenset(_values(entry["model"])))
        for entry in SKIPS
        if "model" in entry and _applies(entry, backend, op, sm)
    ]


def models(backend: str, op: str | None, sm: int | None) -> frozenset[str]:
    """The case-file architectures whose model rows this backend leaves out of an op on an SM."""
    return frozenset().union(*(architectures for _entry, architectures in model_entries(backend, op, sm)))


def split(
    cases: Iterable,
    backend: str,
    op: str,
    sm: int | None,
    dtypes_of: Callable[[object], Iterable[str]],
    *,
    dispatch_entries: bool | None = None,
) -> tuple[list, list[tuple[object, dict, str]]]:
    """Split cases into the kept ones and ``(case, entry, region)`` for each skipped one.

    ``dtypes_of`` gives a case's dtypes for the ``dtype`` key; ``region`` names
    what the entry matched, such as ``dtype=fp8_block`` or ``op=compute_scale``.
    ``dispatch_entries`` applies only the entries that name a ``dispatch``
    (True), only the others (False), or all of them (None).
    """
    entries = [
        entry
        for entry in SKIPS
        if "model" not in entry
        and _applies(entry, backend, op, sm)
        and (dispatch_entries is None or ("dispatch" in entry) == dispatch_entries)
    ]
    cases = list(cases)
    if not entries:
        return cases, []
    keys = sorted({key for entry in entries for key in entry} - _RUN_KEYS)
    found: dict[tuple, tuple[dict, str] | None] = {}
    kept, skipped = [], []
    for case in cases:
        values = tuple(tuple(dtypes_of(case)) if key == "dtype" else getattr(case, key, _MISSING) for key in keys)
        try:
            match = found[values]
        except KeyError:
            match = found[values] = _first_match(entries, keys, values, op)
        except TypeError:  # an unhashable field value: match without the cache
            match = _first_match(entries, keys, values, op)
        if match is None:
            kept.append(case)
        else:
            skipped.append((case, *match))
    return kept, skipped


def _first_match(entries: list[dict], keys: list[str], values: tuple, op: str) -> tuple[dict, str] | None:
    by_key = dict(zip(keys, values, strict=True))
    for entry in entries:
        region = []
        for key in entry:
            if key in _RUN_KEYS:
                continue
            want, value = entry[key], by_key[key]
            if key == "dtype":
                value = next((dtype for dtype in value if _matches(want, dtype)), _MISSING)
            elif value is not _MISSING and not _matches(want, value):
                value = _MISSING
            if value is _MISSING:
                break
            region.append(f"{key}={value}")
        else:
            return entry, ",".join(region) or f"op={op}"
    return None


def case_dtypes(test_case) -> list[str]:
    """Return the dtype/quant names a case requires (typed attribute access)."""
    dtypes = []
    for field in _DTYPE_FIELDS:
        value = getattr(test_case, field, None)
        if isinstance(value, str) and value:
            dtypes.append(value)
    for field in _FP8_FLAG_FIELDS:
        if getattr(test_case, field, False):
            dtypes.append("fp8")
    return dtypes


def filter_cases(test_cases, backend: str, op: str, sm_version: int | None):
    """Drop the cases an entry skips before queueing; return ``(kept, skipped)``.

    Cases are matched on their typed attributes, so a positional (list) case is
    skipped only by an entry that names no case field.
    """
    kept, skipped = split(test_cases, backend, op, sm_version, case_dtypes, dispatch_entries=False)
    if skipped:
        counts = Counter(repr(entry) for _case, entry, _region in skipped)
        summary = "; ".join(f"{count}x {entry}" for entry, count in sorted(counts.items()))
        logger.info(f"{op}: skipped {len(skipped)}/{len(kept) + len(skipped)} cases before queueing ({summary})")
    return kept, skipped


def detect_sm_version() -> int | None:
    """Detect the SM version of the local CUDA device, or None (e.g. XPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return major * 10 + minor
    except Exception:
        pass
    return None


_check(SKIPS)
