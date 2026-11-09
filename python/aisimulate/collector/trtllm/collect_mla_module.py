# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# FP8-KV DSA runs on the SM100 family only (trtllm-gen sparseMla): the SM90
# DSA core runs FlashMLA sparse kernels, whose sparse_prefill_fwd asserts
# kv.dtype()==kBFloat16 (flashmla-src/csrc/pybind.cpp), so collector/skips.py
# leaves FP8-KV DSA out on SM90. FP8-KV MLA modules run on SM90, SM100,
# SM103 and SM120 (trtllm-gen MLA on the SM100 family, the framework's
# non-trtllm-gen dispatch on SM120); collector/skips.py leaves TRT-LLM MLA
# out below SM90. SM121 is not hardware-validated.

# SM100/103 DSA with fewer than 8 local heads (high-TP shards) has no
# trtllm-gen sparse-MLA kernel: the generator requires the row dim, the local
# q-head count, to be a multiple of 8 (StoreSmemP.h), and serving passes the
# same local count, so it fails identically. collector/skips.py leaves those
# cases out; SM90 runs FlashMLA and is not affected.

# The DSA path uses the cache manager's sparse kwargs and the lowered
# SparseMetadataParams attention metadata.
__compat__ = "trtllm==1.4.0rc2"

"""
MLA Module Collector for TRT-LLM — unified MLA and DSA benchmarking.

Profiles the complete attention module forward pass (projections + attention +
output), not just the bare attention kernel.  Uses TRT-LLM's own modeling code
to construct a single-layer mock model with dummy weights, then extracts the
attention module for benchmarking.

Supported models, attention types, and micro-sweeps are defined in collector v2
YAML and loaded through collector.case_generator. Model dimensions are loaded
from the HF config.json via from_pretrained().

Usage:
    # MLA context phase (DeepSeek-V3)
    python collect_mla_module.py --mode context --model deepseek-ai/DeepSeek-V3

    # DSA generation phase (DeepSeek-V3.2)
    python collect_mla_module.py --mode generation --model deepseek-ai/DeepSeek-V3.2

    # All models, context phase
    python collect_mla_module.py --mode context

    # Quick single-point test
    python collect_mla_module.py --mode context --model deepseek-ai/DeepSeek-V3
    --quick --batch-size 4 --seq-len 2048 --num-heads 64

    # FP8 KV cache only
    python collect_mla_module.py --mode context --model deepseek-ai/DeepSeek-V3 --kv-cache-dtype fp8
"""

import argparse
import dataclasses
import gc
import os
import sys
import time
import traceback
import weakref

import tensorrt_llm
import torch
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer,
)
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.model_loader import initialize_dummy_weights
from tensorrt_llm._torch.utils import AuxStreamType, get_model_extra_attrs, model_extra_attrs
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collector.case_generator import get_mla_module_model_specs
from collector.helper import (
    PeriodicGarbageCollector,
    _resolve_local_model_path,
    benchmark_with_power,
    get_sm_version,
    log_perf,
)
from collector.op_plans import plan, plan_cases
from collector.registry_types import PerfFile
from collector.trtllm.module_series import run_points

# ═══════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════


def get_mla_context_module_test_cases():
    return plan_cases("mla_context_module", "trtllm", get_sm_version())


def get_mla_generation_module_test_cases():
    return plan_cases("mla_generation_module", "trtllm", get_sm_version())


def get_dsa_context_module_test_cases():
    return plan_cases("dsa_context_module", "trtllm", get_sm_version())


def get_dsa_generation_module_test_cases():
    return plan_cases("dsa_generation_module", "trtllm", get_sm_version())


# ═══════════════════════════════════════════════════════════════════════
# Layer Construction
# ═══════════════════════════════════════════════════════════════════════


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _ceil_div(a, b) * b


# One 32 MiB GPU allocation unit for each pool of a DSA layer (MLA KV, indexer
# K); see create_kv_cache_and_metadata.
_DSA_POOL_ROUNDING_BYTES = 2 * (32 << 20)


def _replace_quant_config(qc, **kwargs):
    """Replace fields in quant_config; supports both dataclass and Pydantic BaseModel."""
    if dataclasses.is_dataclass(qc):
        return dataclasses.replace(qc, **kwargs)
    # Pydantic BaseModel (TRT-LLM >= rc9 uses StrictBaseModel / Pydantic v2)
    return qc.model_copy(update=kwargs)


def _set_quant_config(model_config, new_qc):
    """Set quant_config, bypassing ModelConfig freeze if necessary."""
    try:
        model_config.quant_config = new_qc
    except AttributeError:
        object.__setattr__(model_config, "quant_config", new_qc)


def _apply_gemm_type_quant(model_config, gemm_type: str, use_fp8_kv_cache: bool):
    """Apply GEMM quantization to model_config.quant_config.

    Replaces the quant_config so that linear-layer GEMMs (and MLA absorption
    BMMs where supported) run at the requested precision.
    """
    kv_algo = QuantAlgo.FP8 if use_fp8_kv_cache else None

    if gemm_type == "bfloat16":
        _set_quant_config(
            model_config,
            _replace_quant_config(
                model_config.quant_config,
                quant_algo=None,
                kv_cache_quant_algo=kv_algo,
                exclude_modules=None,
            ),
        )
    elif gemm_type == "fp8_block":
        _set_quant_config(
            model_config,
            _replace_quant_config(
                model_config.quant_config,
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                group_size=128,
                kv_cache_quant_algo=kv_algo,
                # Serving ALWAYS excludes these from FP8_BLOCK_SCALES: every
                # translation of an fp8 weight_block_size checkpoint appends
                # ["*kv_b_proj*", "*k_b_proj*", "*eh_proj"] to exclude_modules
                # (_torch/model_config.py:774-782, merged with the HF
                # modules_to_not_convert; same list at :720-726 for the
                # TRTLLM-MoE default), because 128x128 block boundaries need
                # not align with per-head dims (GLM-5 qk_nope_head_dim=192),
                # and the weight loader then routes kv_b_proj through the
                # dequant path so k_b_proj_trans stays bf16
                # (models/modeling_deepseekv3.py:395-405). Passing None
                # here made k_b_proj_trans fp8 — a bmm path serving never
                # takes for these checkpoints — which crashed every GLM
                # fp8_block DSA context case on H20 ("Scale tensor size
                # mismatch", expected/got = 192-per-head blocks 2 vs 1.5)
                # and silently mis-measured the DeepSeek ones that passed.
                exclude_modules=["*kv_b_proj*", "*k_b_proj*", "*eh_proj"],
            ),
        )
    elif gemm_type == "nvfp4":
        _set_quant_config(
            model_config,
            _replace_quant_config(
                model_config.quant_config,
                quant_algo=QuantAlgo.NVFP4,
                # NVFP4 scale blocks are 16 elements; without an explicit value the replaced
                # config keeps the checkpoint's/fp8_block group_size=128 and
                # NVFP4LinearMethod.resolve_scaling_vector_size (modules/linear.py:1603-1611)
                # rejects module construction ("supports NVFP4 scale blocks of (16,) elements,
                # but the checkpoint declares group_size=128"; nvfp4 is a platform floor on Hopper).
                group_size=16,
                kv_cache_quant_algo=kv_algo,
                exclude_modules=None,
            ),
        )
    else:
        raise ValueError(f"Unknown gemm_type: {gemm_type!r}")


def create_attention_layer(
    model_path: str,
    num_heads: int = 128,
    use_fp8_kv_cache: bool = False,
    gemm_type: str = "bfloat16",
    device: str = "cuda:0",
):
    """
    Create a single attention layer from TRT-LLM's own modeling code.

    Uses the official config.json from the HF model repo via from_pretrained()
    so the layer matches real inference behavior.  MLA vs DSA is determined by
    the model's config.json automatically (model_type, sparse_attention_config).
    """
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    # Resolve the model id to a local, auto_map-stripped config dir before any
    # transformers/TRT-LLM load, mirroring the sglang/vllm module collectors
    # (collector/sglang/collect_mla_module.py:1091). The raw HF id would send
    # get_config_dict()/ModelConfig.from_pretrained() through the trust_remote_code
    # path, whose ~/.cache/huggingface/modules/_remote_code.lock serializes the 8
    # parallel workers and caused mass file-lock Timeouts on the *_module ops. The
    # bundled config (src/aisimulate_core/model_configs/) carries every dimension the
    # bare-layer build needs, including the sparse index_* fields.
    model_path = _resolve_local_model_path(model_path)

    import transformers

    # Capture the original HF architecture *before* from_pretrained() which
    # may remap the model_type.  We return it separately because ModelConfig
    # may be frozen and cannot accept new attrs.
    _cfg_dict, _ = transformers.PretrainedConfig.get_config_dict(model_path)
    original_architecture = _cfg_dict.get("architectures", [_cfg_dict.get("model_type", "unknown")])[0]

    # No sparse_attention_config is passed, as in a serving deployment that sets none:
    # from_pretrained builds the DSA config from the checkpoint's index_* fields for both
    # DeepseekV32ForCausalLM and GlmMoeDsaForCausalLM (_torch/model_config.py:1202-1262,
    # the no-user-config branch), and loads a glm_moe_dsa config through the registered
    # DeepseekV3Config after dropping its HF-only layer_types
    # (pyexecutor/config_utils.py:930, 1017-1027).
    model_config = ModelConfig.from_pretrained(
        model_path,
        mapping=mapping,
        enable_min_latency=False,
        use_cuda_graph=False,
        force_dynamic_quantization=False,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=131072,
        max_seq_len=163840,
        moe_max_num_tokens=None,
        moe_load_balancer=None,
        lora_config=None,
        allreduce_strategy=AllReduceStrategy.AUTO,
        mm_encoder_only=False,
        attn_backend="TRTLLM",
        moe_backend="CUTLASS",
        moe_disable_finalize_fusion=False,
        use_low_precision_moe_combine=False,
        skip_create_weights_in_init=True,
    )

    pretrained_config = model_config.pretrained_config

    # DeepseekV3DecoderLayer builds DeepseekV32Attention (the DSA module) only for
    # model_type "deepseek_v32" (models/modeling_deepseekv3.py:1348-1360). Serving's
    # DeepseekV3ForCausalLM.__init__ remaps glm_moe_dsa to it first (:2025-2028); this
    # bare-layer build skips that constructor, so it applies the same remap.
    if pretrained_config.model_type == "glm_moe_dsa":
        pretrained_config.model_type = "deepseek_v32"

    # Override for single-layer, single-GPU benchmark
    pretrained_config.num_hidden_layers = 1
    pretrained_config.num_attention_heads = num_heads
    pretrained_config.num_key_value_heads = num_heads

    _apply_gemm_type_quant(model_config, gemm_type, use_fp8_kv_cache)

    aux_stream = torch.cuda.Stream(device=device)
    aux_stream_dict = {
        AuxStreamType.Attention: aux_stream,
        AuxStreamType.MoeShared: aux_stream,
        AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(device=device),
    }

    layer = DeepseekV3DecoderLayer(
        model_config=model_config,
        layer_idx=0,
        aux_stream_dict=aux_stream_dict,
    )

    # Serving applies QuantConfig.exclude_modules in
    # DecoderModelForCausalLM.__post_init__ via
    # apply_quant_config_exclude_modules() (_torch/models/
    # modeling_utils.py:438-485, 714-720): every named module matching an
    # exclusion pattern has its quant_config replaced by a fresh QuantConfig
    # carrying only kv_cache_quant_algo, before weights are (re)created.
    # This bare-layer build skips __post_init__, which left kv_b_proj
    # quantized even though every fp8 weight_block_size checkpoint
    # translation force-excludes *kv_b_proj*/*k_b_proj*
    # (_torch/model_config.py:774-782, GLM-5's qk_nope_head_dim=192
    # cannot tile into 128x128 scale blocks) — so MLA.create_weights
    # (attention/mla.py:720-752) built an fp8 k_b_proj_trans and routed
    # the absorption bmm down a path serving never takes for these
    # checkpoints. Mirror the pass here, before create_weights, so weight
    # dtypes come out serving-identical. The serving loop's fused-gate_up/
    # fused-qkv alias candidates are irrelevant for this op family: its
    # exclusion patterns only name plain Linears.
    quant_config = model_config.quant_config
    if quant_config is not None and quant_config.exclude_modules is not None:
        excluded_replacement = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for module_name, module in layer.named_modules():
            if getattr(module, "quant_config", None) is None:
                continue
            if quant_config.is_module_excluded_from_quantization(module_name):
                module.quant_config = excluded_replacement

    for module in layer.modules():
        if callable(getattr(module, "create_weights", None)):
            module.create_weights()
    layer.to(device)

    initialize_dummy_weights(layer)
    for module in layer.modules():
        if hasattr(module, "post_load_weights") and not getattr(module, "_weights_removed", False):
            module.post_load_weights()

    layer.eval()
    layer.requires_grad_(False)

    next_ln = RMSNorm(
        hidden_size=pretrained_config.hidden_size,
        eps=pretrained_config.rms_norm_eps,
        dtype=pretrained_config.torch_dtype,
    ).to(device)
    next_ln.requires_grad_(False)
    initialize_dummy_weights(next_ln)
    layer.next_layer_layernorm = next_ln

    attn_module = layer.self_attn
    # DSA too: its tunable ops are Linear projections as well. Besides the MLA projections
    # (attention/backends/sparse/dsa/module.py:258, 272) the indexer adds one, wq_b
    # (dsa/indexer.py:846, called at :2327) with the layer's GEMM quant. Its fused
    # wk + weights_proj GEMM is an F.linear (:2383, :2391), and no AutoTuner op runs
    # anywhere under attention/backends/sparse/dsa/ or modules/top_k.py: the DeepGEMM and
    # CuTe DSL MQA logits, top-k and the sparse MLA kernels have no tactics to tune.
    _autotune_projections(attn_module, _SERVING_WARMUP_NUM_TOKENS, device)
    return attn_module, model_config, original_architecture


# Serving tunes every tunable op in one autotune() pass at warmup: a context forward of
# curr_max_num_tokens tokens (max_num_tokens, 8192 by default, llmapi/llm_args.py:5932-5933,
# unless the KV pool holds fewer), then a one-token generation forward
# (pyexecutor/engine/runners/decoder/warmup.py:1050-1068, 1098-1159). In the MLA module the
# tunable ops are the projection GEMMs (collect_gemm.py explains each GEMM type: nvfp4 tunes
# its backend per power-of-2 bucket up to the pass's M; fp8_block on SM100/103 tunes its quant
# kernel at the pass's M only and JITs DeepGEMM; bf16 calls no tunable op), and each projection
# sees the forward's new-token count. The CuTe DSL MLA decode is tunable too, but its one tactic
# per bucket is the cache-miss default (collect_mla.py). So every projection is tuned at
# serving's two warmup token counts when the layer is built, and a context point with more new
# tokens than that first gets a pass at its own count, which a deployment that runs it
# (max_num_tokens >= that count) tunes at warmup. On the cached-KV context path kv_b_proj also
# sees the cached prefix (attention/mla.py:1241); serving tunes it only at new-token counts too.
_SERVING_WARMUP_NUM_TOKENS: tuple[int, ...] = (8192, 1)
# The largest token count each built attention module is tuned at.
_TUNED_NUM_TOKENS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _autotune_projections(attn_module, num_tokens: tuple[int, ...], device: str) -> None:
    """Forward every projection of ``attn_module`` at each count of ``num_tokens`` under
    autotune(), as serving's warmup pass does; the tuned tactics stay in the worker's
    AutoTuner cache for the timed forwards."""
    start = time.perf_counter()
    projections = [module for module in attn_module.modules() if isinstance(module, Linear)]
    with torch.inference_mode(), autotune():
        for tokens in num_tokens:
            for projection in projections:
                projection.forward(
                    torch.randn(
                        (tokens, projection.in_features),
                        dtype=projection.dtype or torch.bfloat16,
                        device=device,
                    )
                )
    torch.cuda.synchronize()
    _TUNED_NUM_TOKENS[attn_module] = max(max(num_tokens), _TUNED_NUM_TOKENS.get(attn_module, 0))
    print(
        f"  [MLA module autotune] {len(projections)} projections at num_tokens={list(num_tokens)}: "
        f"{time.perf_counter() - start:.1f} s"
    )


# ═══════════════════════════════════════════════════════════════════════
# KV Cache + Metadata
# ═══════════════════════════════════════════════════════════════════════


# One attention layer per (model, heads, kv dtype, gemm type) per worker. The layer
# build (ModelConfig.from_pretrained, DeepseekV3DecoderLayer, dummy weights,
# post_load_weights) does not depend on the point: create_attention_layer sizes
# the config with fixed max_num_tokens / max_seq_len, so one layer serves every
# (batch, seq, prefix) of its key exactly as serving reuses one module. The
# per-case KV cache manager, metadata, dry run, CUDA-graph capture and timing are
# untouched. Same pattern as collect_dsv4_attn._cached_dsv4_attention_module.
# A b300 base run (job 471159002) measured
# dsa_context_module 1.2-2.4 GPU-s and dsa_generation_module 1.8 GPU-s per
# one-row case, most of it this build. At most one layer is retained.
_LAYER_CACHE: dict = {}


def _cached_attention_layer(model_path: str, num_heads: int, use_fp8_kv_cache: bool, gemm_type: str, device: str):
    key = (model_path, int(num_heads), bool(use_fp8_kv_cache), gemm_type, device)
    hit = _LAYER_CACHE.get(key)
    if hit is not None:
        return hit
    if _LAYER_CACHE:
        _LAYER_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()
    entry = create_attention_layer(
        model_path=model_path,
        num_heads=num_heads,
        use_fp8_kv_cache=use_fp8_kv_cache,
        gemm_type=gemm_type,
        device=device,
    )
    _LAYER_CACHE[key] = entry
    return entry


def create_kv_cache_and_metadata(
    model_config: ModelConfig,
    attn_module,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    prefix_len: int = 0,
    use_fp8_kv_cache: bool = False,
    device: str = "cuda:0",
):
    """
    Create KV cache manager and attention metadata using framework utilities.

    Follows the same pattern as TRT-LLM's ``layer_wise_benchmarks/runner_utils.py``.
    """
    config = model_config.pretrained_config
    mapping = model_config.mapping

    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim
    head_dim = kv_lora_rank + qk_rope_head_dim

    # SM90 serving runs FlashMLA for head_dim 576 (ModelConfig.enable_flash_mla,
    # _torch/model_config.py:466-475) and forces 64-token pages for it
    # (pyexecutor/py_executor_creator.py:859-873); the attention op runs FlashMLA only
    # on that page (thop/attentionOp.cpp:1474). Every other case keeps TRT-LLM's
    # KvCacheConfig default (llmapi/llm_args.py:5040-5041), 32.
    tokens_per_block = 64 if model_config.enable_flash_mla else int(KvCacheConfig().tokens_per_block)

    prefix_len = int(prefix_len) if is_context else 0

    if is_context:
        max_seq = prefix_len + seq_len + 1
        total_tokens = seq_len * batch_size
        seq_len_q = seq_len
        kv_cache_len = prefix_len
    else:
        max_seq = seq_len + 1
        total_tokens = batch_size
        seq_len_q = 1
        kv_cache_len = seq_len

    # --- KV Cache Manager ---
    # The pool is sized for this case's requests; its size is not a timing input
    # (MLA and the DSA indexer read KV through the block table). A DSA layer
    # adds an indexer K-cache buffer (DSACacheManagerV2._extra_buffers_per_layer,
    # attention/backends/sparse/dsa/cache_manager.py), a second pool in the same
    # pool group, and V2 splits the quota with every pool rounded up to the GPU
    # allocation unit, 2 MiB growing to 32 MiB with the quota
    # (StorageManager::computeSlotCountForLevel, ratioToSlotCountList,
    # kv_cache_manager_v2/storageManager.cpp). On a one-layer pool that rounding
    # left the MLA pool short of one request: max_seq_len was clamped
    # (clamp_max_seq_len_for_mem) and add_dummy_requests overran the page-index
    # buffer ("base page indices is too short", every DSA case). DSA quotas
    # therefore carry one largest unit per pool, converted to tokens at the
    # smallest per-token footprint (head_dim bytes, an FP8 row); dense MLA has
    # one pool and keeps the exact size.
    kv_tokens = batch_size * _round_up(max_seq, tokens_per_block)
    if model_config.sparse_attention_config is not None:
        kv_tokens += _round_up(-(-_DSA_POOL_ROUNDING_BYTES // head_dim), tokens_per_block)
    kv_cache_config = KvCacheConfig(
        max_tokens=kv_tokens,
        enable_block_reuse=False,
    )
    # Serving's class: KvCacheConfig.use_kv_cache_manager_v2 defaults to True
    # (llmapi/llm_args.py:5057-5064), so a dense MLA model resolves KVCacheManagerV2
    # (pyexecutor/_util.py:123-128, 447-448) and a DSA model its sparse V2 manager (:438-446).
    kv_cache_manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
    kv_cache_dtype = DataType.FP8 if use_fp8_kv_cache else torch_dtype_to_binding(torch.bfloat16)

    layer_mask = [True]  # single layer
    kv_cache_manager = kv_cache_manager_cls(
        kv_cache_config,
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
        layer_mask=layer_mask,
        # Serving passes sparse_attention_config + pretrained_config when
        # constructing MLA cache managers (pyexecutor/_util.py:3575-3607); the DSA
        # cache managers require both, the dense managers swallow them via kwargs.
        sparse_attention_config=model_config.sparse_attention_config,
        pretrained_config=config,
        model_config=model_config,
    )

    request_ids = list(range(batch_size))
    token_nums = [prefix_len + seq_len_q] * batch_size if is_context else [max_seq] * batch_size
    # add_dummy_requests returns None when the pool cannot hold the requests, which
    # would leave their block table unset.
    if kv_cache_manager.add_dummy_requests(request_ids, token_nums) is None:
        kv_cache_manager.shutdown()
        raise RuntimeError("FIXME(kernel-limit): does not fit: the KV cache pool cannot hold the case's requests")

    # --- Attention Metadata ---
    attention_cls = get_attention_backend(
        model_config.attn_backend,
        model_config.sparse_attention_config,
    )

    # Metadata classes take the lowered SparseMetadataParams, not the raw
    # sparse_attention_config (build_attention_metadata, pyexecutor/engine/metadata.py:72-77).
    sparse_metadata_params = (
        model_config.sparse_attention_config.to_sparse_metadata_params(pretrained_config=config)
        if model_config.sparse_attention_config is not None
        else None
    )

    attn_metadata = attention_cls.Metadata(
        max_num_requests=batch_size,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_manager,
        mapping=mapping,
        # build_attention_metadata (pyexecutor/engine/metadata.py:86).
        enable_flash_mla=model_config.enable_flash_mla,
        seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int32),
        position_ids=None,
        num_contexts=batch_size if is_context else 0,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[kv_cache_len] * batch_size,
        ),
        cross=None,
        request_ids=request_ids,
        # Serving sets a context request's prompt_lens to the CURRENT-CHUNK
        # token count only — prompt_tokens = request.get_tokens_range(0,
        # begin_compute, end_compute); prompt_lengths.append(len(prompt_tokens)) —
        # while the cached prefix travels separately via num_cached_tokens_per_seq
        # (pyexecutor/engine/runners/decoder/inputs.py:1000-1014). kv_lens is then
        # derived as cached_token_lens + seq_lens_kv (attention/backends/trtllm.py:1047),
        # so prompt_lens must NOT be inflated by the prefix: it feeds
        # host_context_lengths, which the SM100 MLA context rope kernel
        # (applyMLARopeAndAssignQKVKernelOptContext via
        # AttentionOp::enqueueContext) uses to walk the new-tokens-sized
        # dense q/latent input. Passing prefix+seq here made that kernel
        # read prefix tokens past the input (compute-sanitizer-verified OOB
        # on B200, every prefix>0 DSA context case); SM90 was
        # blind to it only because FlashMLA doesn't walk the dense input by
        # host_context_lengths.
        prompt_lens=[seq_len_q if is_context else kv_cache_len] * batch_size,
        # Serving computes enable_context_mla_with_cached_kv = is_mla &&
        # (cache_reuse || chunked_prefill) (pyexecutor/engine/metadata.py:64-68);
        # a prefix-cached context request only exists under cache reuse, so
        # mirror that state here. Without the flag the DSA indexer takes the
        # no-cache alias branch for slot_mapping_*_fullkv (dsa.py
        # "_need_full_kv_gathering") and its table — sized for new tokens
        # only — overflows once batch*(prefix+seq) exceeds it.
        enable_context_mla_with_cached_kv=bool(is_context and prefix_len > 0),
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=False,
            cache_reuse=bool(is_context and prefix_len > 0),
        ),
        all_rank_num_tokens=None,
        workspace=torch.tensor([], device=device, dtype=torch.int8),
        sparse_metadata_params=sparse_metadata_params,
    )

    # DSA needs a reference to the indexer for prepare()
    if hasattr(attn_module, "indexer") and attn_module.indexer is not None:
        attn_metadata.indexer = attn_module.indexer

    attn_metadata.prepare()

    return kv_cache_manager, attn_metadata


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


def run_mla_module(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    prefix_len: int = 0,
    *,
    model_path: str,
    attn_type: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
    dry_run: bool = True,
):
    """Run a single MLA / DSA module-level benchmark point.

    ``dry_run`` False skips the checking forward before timing: a series has
    already run it for an earlier point of this seq_len (module_series).
    """
    # collector/skips.py leaves out SM80-89, where AttentionOp aborts on every case,
    # and SM120 DSA, which TRT-LLM runs only on the packed fp8_ds_mla KV pool that
    # this module does not build.
    if attn_type == "mla" and get_sm_version() < 90:
        raise ValueError("FIXME(kernel-limit): TRT-LLM MLA has no pre-Hopper FMHA kernel (collector/skips.py)")
    # FIXME(kernel-limit): SM120 MLA context expands k_nope through kv_b_proj and
    # maybe_compiled_copy_ into k (forward_context_default, attention/mla.py:1178-1188;
    # SM100/103 with a bf16 kv_b_proj write k_nope with per-head BMMs instead, :1152-1177,
    # _use_kvbproj_strided_out :1099-1121), and that copy fails under torch-inductor
    # autotune at huge extents: h=128 cases with ~>=49k total tokens die with
    # CUDA "invalid argument" (deterministic repro on RTX PRO 6000,
    # CUDA_LAUNCH_BLOCKING=1: inductor copy kernel launched with
    # xnumel=2147483648 for b=4/s=32768/h=128). ~72/5856 context cases fail
    # classified; run-and-record.
    torch.cuda.set_device(device)
    torch_device = torch.device(device)

    use_fp8_kv_cache = kv_cache_dtype == "fp8"

    is_context = "context" in perf_filename
    prefix_len = int(prefix_len) if is_context else 0
    phase = "context" if is_context else "generation"
    variant = attn_type.upper()
    print(
        f"\n[{variant} module] {phase} b={batch_size}, s={seq_len}, "
        f"prefix={prefix_len}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    # Per-phase wall time of this case, printed after cleanup so the campaign
    # log shows where a module case spends its seconds (layer cache hit vs
    # KV manager + metadata vs graph capture + timing vs teardown).
    _t_case = _t_last = time.perf_counter()
    phase_timing: dict[str, float] = {}

    def _mark(name: str) -> None:
        nonlocal _t_last
        now = time.perf_counter()
        phase_timing[name] = now - _t_last
        _t_last = now

    # 1. Create attention layer (from_pretrained reads config.json)
    attn_module, model_config, original_architecture = _cached_attention_layer(
        model_path, num_heads, use_fp8_kv_cache, gemm_type, device
    )
    # A context point past serving's warmup token count tunes its projections at
    # its own count first (_SERVING_WARMUP_NUM_TOKENS).
    new_tokens = batch_size * seq_len if is_context else batch_size
    if new_tokens > _TUNED_NUM_TOKENS.get(attn_module, 0):
        _autotune_projections(attn_module, (new_tokens,), device)
    _mark("layer")

    # The KV pool is released on every exit path: a series keeps running after a
    # failed point, and a leaked pool would make the next points fail too.
    kv_cache_manager = None
    try:
        # 2. Create KV cache + metadata
        kv_cache_manager, attn_metadata = create_kv_cache_and_metadata(
            model_config=model_config,
            attn_module=attn_module,
            batch_size=batch_size,
            seq_len=seq_len,
            is_context=is_context,
            prefix_len=prefix_len,
            use_fp8_kv_cache=use_fp8_kv_cache,
            device=device,
        )

        _mark("kv_meta")

        # 3. Input tensors
        hidden_size = model_config.pretrained_config.hidden_size
        if is_context:
            num_tokens = seq_len * batch_size
            position_ids = (
                torch.arange(prefix_len, prefix_len + seq_len, device=torch_device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .reshape(-1)
                .contiguous()
            )
        else:
            num_tokens = batch_size
            position_ids = torch.full(
                (batch_size,),
                seq_len - 1,
                device=torch_device,
                dtype=torch.long,
            )

        hidden_states = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=torch_device,
        )

        _mark("inputs")

        # 4. Dry run (once per seq_len in a series)
        if dry_run:
            with model_extra_attrs(model_config.extra_attrs):
                get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
                try:
                    with torch.inference_mode():
                        attn_module.forward(position_ids, hidden_states, attn_metadata)
                except Exception:
                    # Never swallow a failed case: raising lets the executor record a
                    # classified failure. A silent return here left green runs with
                    # missing perf rows (e.g. FlashMLA sparse rejecting FP8 KV probes
                    # surfaced as "passed" with no output).
                    print("  Dry run failed:")
                    traceback.print_exc()
                    raise

            _mark("dry_run")

        # 5. Benchmark
        import tensorrt_llm._torch.utils as _trtllm_utils

        _trtllm_utils._model_extra_attrs.attrs = model_config.extra_attrs
        _trtllm_utils._model_extra_attrs.attrs["attention_metadata"] = weakref.ref(attn_metadata)

        # FIXME(kernel-limit): same 32-bit per-sequence offset boundary as
        # collect_attn.py (H20 campaign, rc29): DeepSeek-V3 context cells
        # with (batch-1) * input_len * local_heads * 192 * 2 >= 2**31 (tp1: 81920+
        # total tokens) fault with cudaErrorIllegalAddress while the TP-sharded rows
        # of the same (batch, seq) pass. Unverified against the kernel source; left
        # failing into the classified log.
        def kernel_func():
            attn_module.forward(position_ids, hidden_states, attn_metadata)

        with benchmark_with_power(
            device=torch_device,
            kernel_func=kernel_func,
            num_warmups=warming_up,
            # A dry run of this seq_len already executed this forward; three eager
            # passes cover lazily initialised paths before capture.
            num_eager_warmups=3,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=False,
        ) as results:
            pass

        latency = results["latency_ms"]
        _mark("bench")

        # 6. Log results
        if is_context:
            isl = seq_len
            step = prefix_len
        else:
            isl = 1
            step = seq_len

        op_name = f"{attn_type}_{phase}_module"

        architecture = original_architecture

        # TRT-LLM's MLA context attention quantizes Q/K/V to FP8 itself when the
        # KV cache is FP8 on SM90/100/103/107/120 (thop/attentionOp.cpp:1466-1470), so
        # those rows time FP8 compute although the inputs are BF16. Same label as
        # collect_mla.py.
        mla_dtype = compute_dtype
        if (
            attn_type == "mla"
            and is_context
            and kv_cache_dtype == "fp8"
            and get_sm_version() in (90, 100, 103, 107, 120)
        ):
            mla_dtype = "fp8"

        log_perf(
            item_list=[
                {
                    "model": model_path,
                    "architecture": architecture,
                    "mla_dtype": mla_dtype,
                    "kv_cache_dtype": "bfloat16" if kv_cache_dtype == "bfloat16" else kv_cache_dtype,
                    "gemm_type": "bfloat16" if gemm_type == "bfloat16" else gemm_type,
                    "num_heads": num_heads,
                    "batch_size": batch_size,
                    "isl": isl,
                    "tp_size": 1,
                    "step": step,
                    "latency": f"{latency:.4f}",
                }
            ],
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name=op_name,
            kernel_source="default",
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )

        print(
            f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
            f"prefix={prefix_len}, gemm={gemm_type}, compute={compute_dtype}, "
            f"kv={kv_cache_dtype}: {latency:.4f} ms"
        )
    finally:
        _cleanup(kv_cache_manager)
    _mark("cleanup")
    print(
        f"  [{variant} module timing] {phase} b={batch_size}, s={seq_len}, prefix={prefix_len}, "
        f"heads={num_heads}, gemm={gemm_type}, kv={kv_cache_dtype}: "
        + " ".join(f"{name}={seconds:.2f}s" for name, seconds in phase_timing.items())
        + f" total={time.perf_counter() - _t_case:.2f}s"
    )
    return latency


def run_mla_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py."""
    return run_mla_module(
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        prefix_len=prefix_len,
        perf_filename=perf_filename,
        model_path=model_path,
        attn_type=attn_type,
        device=device,
    )


def run_mla_module_series(
    points,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    *,
    model_path: str,
    attn_type: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
) -> list[float]:
    """Measure the (seq_len, prefix_len) points of one MLA / DSA module key in one task.

    Every point runs through run_mla_module, so its rows are those of a one-point
    case: the layer comes from the layer cache, and the KV cache manager and
    metadata are built per point (module_series explains why).
    """

    def run_point(seq_len: int, prefix_len: int, dry_run: bool) -> float:
        return run_mla_module(
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype=compute_dtype,
            gemm_type=gemm_type,
            perf_filename=perf_filename,
            prefix_len=prefix_len,
            model_path=model_path,
            attn_type=attn_type,
            device=device,
            warming_up=warming_up,
            test_ite=test_ite,
            dry_run=dry_run,
        )

    label = f"{attn_type.upper()} module series b={batch_size} heads={num_heads} gemm={gemm_type} kv={kv_cache_dtype}"
    return run_points(points, run_point, label=label)


def run_mla_module_series_worker(
    points,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker entry for one series task: run_mla_module_worker's positional fields, with
    the list of (seq_len, prefix_len) points in place of seq_len and prefix_len."""
    return run_mla_module_series(
        points,
        batch_size,
        num_heads,
        kv_cache_dtype,
        compute_dtype,
        gemm_type,
        perf_filename,
        model_path=model_path,
        attn_type=attn_type,
        device=device,
    )


# gc.collect() after every case cost 0.17 s of a ~1 s module case; collect periodically instead.
_CASE_GC = PeriodicGarbageCollector()


def _cleanup(kv_cache_manager):
    if kv_cache_manager is not None:
        kv_cache_manager.shutdown()
    torch.cuda.empty_cache()
    _CASE_GC.maybe_collect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _perf_file_for(attn_type: str, mode: str) -> str:
    """Return the canonical PerfFile for (attn_type, mode)."""
    _map = {
        ("mla", "context"): PerfFile.MLA_CONTEXT_MODULE,
        ("mla", "generation"): PerfFile.MLA_GENERATION_MODULE,
        ("dsa", "context"): PerfFile.DSA_CONTEXT_MODULE,
        ("dsa", "generation"): PerfFile.DSA_GENERATION_MODULE,
    }
    return _map[(attn_type, mode)]


def main():
    all_model_specs = get_mla_module_model_specs(apply_model_filter=False)
    model_names = [spec.model_path for spec in all_model_specs]

    parser = argparse.ArgumentParser(
        description="MLA/DSA module-level collector for TRT-LLM",
    )
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=model_names,
        help=f"Model to benchmark. If not specified, runs all: {model_names}",
    )
    parser.add_argument("--num-heads", type=int, default=None, help="Filter by number of heads")
    parser.add_argument("--batch-size", type=int, default=None, help="Single batch size (for --quick)")
    parser.add_argument("--seq-len", type=int, default=None, help="Single seq len (for --quick)")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["bfloat16", "fp8"],
        default=None,
        help="KV cache dtype (default: run both bfloat16 and fp8 when GPU supports it)",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        choices=["bfloat16"],
        default=None,
        help="Compute dtype for attention (default: bfloat16; reserved for future FP8 support)",
    )
    parser.add_argument(
        "--gemm-type",
        type=str,
        choices=["bfloat16", "fp8_block", "nvfp4"],
        default=None,
        help="GEMM quantisation type for linear layers (default: run all supported by GPU)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="Quick single-point test")
    args = parser.parse_args()

    # Select models to run
    if args.model:
        model_specs_to_run = [spec for spec in all_model_specs if spec.model_path == args.model]
    else:
        model_specs_to_run = all_model_specs

    if not args.quick:
        # The full sweep is the op plan's rows; --model targets that checkpoint, as collect.py --model-path does.
        if args.model:
            os.environ["COLLECTOR_MODEL_PATH"] = args.model
        for attn_type in sorted({spec.attention_type for spec in model_specs_to_run} & {"mla", "dsa"}):
            perf_filename = _perf_file_for(attn_type, args.mode)
            rows = plan(f"{attn_type}_{args.mode}_module", "trtllm", get_sm_version()).rows
            for points, b, h, kv_dtype, compute, gemm, model_path, _attn in rows:
                if args.num_heads is not None and h != args.num_heads:
                    continue
                if args.kv_cache_dtype is not None and kv_dtype != args.kv_cache_dtype:
                    continue
                if args.compute_dtype is not None and compute != args.compute_dtype:
                    continue
                if args.gemm_type is not None and gemm != args.gemm_type:
                    continue
                print(f"\n[{attn_type.upper()}] {model_path} b={b} heads={h} gemm={gemm} kv={kv_dtype}: {len(points)}p")
                try:
                    run_mla_module_series(
                        points,
                        b,
                        h,
                        kv_dtype,
                        compute,
                        gemm,
                        perf_filename,
                        model_path=model_path,
                        attn_type=attn_type,
                        device=args.device,
                    )
                except Exception as e:
                    print(f"  FAILED: b={b}, heads={h}, gemm={gemm}, compute={compute}, kv={kv_dtype}: {e}")
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                    gc.collect()
        return

    for model_spec in model_specs_to_run:
        model_path = model_spec.model_path
        attn_type = model_spec.attention_type
        print(f"\n{'=' * 60}")
        print(f"Model: {model_path}  |  Attention: {attn_type.upper()}")
        print(f"{'=' * 60}")

        b = args.batch_size or 4
        s = args.seq_len or 2048
        h = args.num_heads or 128
        run_mla_module(
            seq_len=s,
            batch_size=b,
            num_heads=h,
            kv_cache_dtype=args.kv_cache_dtype or "bfloat16",
            compute_dtype=args.compute_dtype or "bfloat16",
            gemm_type=args.gemm_type or "bfloat16",
            perf_filename=_perf_file_for(attn_type, args.mode),
            model_path=model_path,
            attn_type=attn_type,
            device=args.device,
        )


if __name__ == "__main__":
    main()
