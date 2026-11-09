# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MiniMax-M3 sparse attention (algorithm "minimax_m3") at TensorRT-LLM 1.4.0rc2: the
# MSA backend on SM100/103 and the Triton reference elsewhere, on the M3 KV cache
# manager V2.
__compat__ = "trtllm==1.4.0rc2"

"""
MSA Module Collector for TRT-LLM — MiniMax-M3 sparse-attention benchmarking.

Profiles the complete MiniMax-M3 sparse-attention module forward pass
(qkv/index projections + per-head QK norm + partial RoPE + index attention +
top-k block selection + sparse GQA + output projection), not a bare kernel.
Uses TRT-LLM's own modeling code (``MiniMaxM3DecoderLayer`` →
``MiniMaxM3Attention``) to construct a single sparse layer with dummy
weights, then extracts the attention module for benchmarking — the same
framework-builder approach as ``collect_mla_module.py``.

The M3 sparse backend has two framework implementations, selected by
``MiniMaxM3SparseAttentionConfig.implementation`` through the framework's own
dispatch (attention/backends/sparse/registry.py:60-75@v1.4.0rc2):

* MSA (``implementation="msa"``, SM100/103 only):
  ``MiniMaxM3MsaSparseAttention`` on the TrtllmAttention stack
  (sparse/minimax_m3/msa_backend.py). Context rows run fmha_sm100 on plans
  prepare() builds; generation rows run the dedicated decode kernels (CuTe DSL
  index scorer, top-k selection, Triton block-sparse decode;
  fmha/msa_decode.py). Its metadata is a ``TrtllmAttentionMetadata`` subclass
  with flat CUDA-graph-stable MSA buffers.
* Triton reference (``implementation="triton"``, any SM):
  ``MiniMaxM3SparseRuntimeBackend`` (sparse/minimax_m3/triton_backend.py),
  whose metadata contract is the prebuilt ``attn_metadata.minimax_m3`` dict.

``create_kv_cache_and_metadata`` detects the resolved Metadata class (never
the version string) and validates the matching contract; ``kernel_source``
records which backend actually ran.

Supported models and micro-sweeps come from collector v2 YAML
(``cases/models/MiniMaxM3ForCausalLM_cases.yaml`` `mla_module` rows with
``attention_type: msa`` + ``cases/base_ops/mla_module.yaml``).

Usage:
    python collect_msa_module.py --mode context --model MiniMaxAI/MiniMax-M3
    python collect_msa_module.py --mode generation --quick --batch-size 4 --seq-len 2048 --num-heads 64
"""

import argparse
import gc
import os
import sys
import time
import traceback
import weakref

import tensorrt_llm
import torch
import transformers
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3DecoderLayer
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls

# ═══════════════════════════════════════════════════════════════════════
# Config registry patch — TRT-LLM's _CONFIG_REGISTRY has no "minimax_m3"
# entry (pyexecutor/config_utils.py:920-932@v1.4.0rc2) and transformers'
# AutoConfig does not know the model_type either (load_pretrained_config
# would raise in its AutoConfig fallback, config_utils.py:1160-1163).
# Serving loads the checkpoint via trust_remote_code; the bundled config dir
# has auto_map stripped (see helper._resolve_local_model_path), so mirror the
# collect_mla_module.py glm_moe_dsa precedent and route through a config
# class instead (the registry branch, config_utils.py:1119-1122).  The M3
# modeling code needs attribute access only — its own VL path materialises a
# bare PretrainedConfig the same way (_wrap_dict_as_config,
# modeling_minimaxm3.py:291-304) — so the base PretrainedConfig is the
# faithful offline stand-in.  This changes HOW the config object is
# constructed, never which kernel runs (the M3 sparse dispatch keys on
# sparse_attention_config / model_type, both preserved verbatim).
# ═══════════════════════════════════════════════════════════════════════
from tensorrt_llm._torch.pyexecutor.config_utils import _CONFIG_REGISTRY
from tensorrt_llm._torch.pyexecutor.model_loader import initialize_dummy_weights
from tensorrt_llm._torch.utils import AuxStreamType, get_model_extra_attrs, model_extra_attrs
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# LazyConfigDict resolves entries via getattr(tensorrt_llm._torch.configs,
# name) (config_utils.py:913-918), so expose the base class there before
# registering the model_type.
import tensorrt_llm._torch.configs as _trtllm_configs
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

if not hasattr(_trtllm_configs, "PretrainedConfig"):
    _trtllm_configs.PretrainedConfig = transformers.PretrainedConfig
if "minimax_m3" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["minimax_m3"] = "PretrainedConfig"


# ═══════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════


def _msa_implementation(sm_version: int) -> str:
    """Which MiniMax-M3 sparse implementation this collector times on this SM.

    "msa" on the SM100 family, the Triton reference elsewhere. The config
    default is "triton" (llm_args.py:1171-1179@v1.4.0rc2), but the model warns
    on it and recommends "msa" on SM100/SM103 (modeling_minimaxm3.py:402-408),
    which is the only family the MSA backend runs on: ensure_msa_available
    raises off SM100/103, SM120 included (msa_availability.py:24-51), when
    the dispatch selects it (sparse/registry.py:60-75). collector/skips.py
    leaves the Triton reference's long-prefix points out.
    """
    return "msa" if sm_version in (100, 103) else "triton"


def get_msa_context_module_test_cases():
    return plan_cases("msa_context_module", "trtllm", get_sm_version())


def get_msa_generation_module_test_cases():
    return plan_cases("msa_generation_module", "trtllm", get_sm_version())


# ═══════════════════════════════════════════════════════════════════════
# Layer Construction
# ═══════════════════════════════════════════════════════════════════════


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _ceil_div(a, b) * b


def _shard_head_counts(native_q: int, native_kv: int, native_index_heads: int, num_heads: int):
    """(tp, KV heads, index heads) of the rank a ``num_heads``-query-head shard runs on.

    num_heads is one rank's query heads of a tp = native_q / num_heads shard. KV
    heads split across ranks and replicate once tp exceeds them, and index heads
    follow their KV-head group: a rank holds native_index_heads / min(tp,
    native_kv) of them (index_head_range,
    attention/backends/sparse/minimax_m3/common.py:35-56@v1.4.0rc2, read by the
    attention layer, modeling_minimaxm3.py:1256-1259, the MSA backend,
    MiniMaxM3SparseConfig.from_sparse_params, common.py:172-201, and the MSA
    metadata, common.py:114-116).
    """
    if num_heads <= 0 or native_q % num_heads:
        raise ValueError(f"num_heads={num_heads} is not a TP shard of the checkpoint's {native_q} query heads")
    if native_index_heads % native_kv:
        raise RuntimeError(f"{native_index_heads} index heads do not group under {native_kv} KV heads")
    tp_size = native_q // num_heads
    num_kv_heads = max(1, native_kv // tp_size)
    return tp_size, num_kv_heads, num_kv_heads * (native_index_heads // native_kv)


def _apply_gemm_type_quant(model_config, gemm_type: str):
    """Apply GEMM quantization to model_config.quant_config.

    M3 sparse attention keeps the KV cache in bf16 (run_msa_module), so
    kv_cache_quant_algo stays None on every combo. The index_qk projection is
    built with ``quant_config=None`` regardless (modeling_minimaxm3.py:1260-1278@v1.4.0rc2),
    so GEMM quant here covers qkv_proj / o_proj only. ModelConfig lets
    quant_config change after from_pretrained froze it (model_config.py:367-382).
    """
    if gemm_type == "bfloat16":
        update = dict(quant_algo=None)
    elif gemm_type == "fp8_block":
        update = dict(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
    elif gemm_type == "nvfp4":
        update = dict(quant_algo=QuantAlgo.NVFP4)
    else:
        raise ValueError(f"Unknown gemm_type: {gemm_type!r}")
    model_config.quant_config = model_config.quant_config.model_copy(
        update={**update, "kv_cache_quant_algo": None, "exclude_modules": None}
    )


def create_msa_attention_layer(
    model_path: str,
    num_heads: int = 64,
    gemm_type: str = "bfloat16",
    device: str = "cuda:0",
):
    """Create a single MiniMax-M3 sparse attention layer from TRT-LLM's own
    modeling code.

    Builds ``MiniMaxM3DecoderLayer`` — the same constructor serving uses for
    every layer (modeling_minimaxm3.py:2796-2799@v1.4.0rc2; the layer reads
    ``is_sparse_attention_layer`` / ``disable_index_value`` from
    ``get_sparse_layer_ids`` / ``get_sparse_disable_index_value_layer_ids``,
    :2515-2518) — and extracts ``layer.self_attn``.  The single benchmark layer
    is made sparse by truncating the config's per-layer pattern arrays to one
    sparse entry; every scalar sparse field (index dim, block size, top-k) is
    taken from the checkpoint config verbatim, and the head counts follow the
    TP shard ``num_heads`` stands for.
    """
    mapping = tensorrt_llm.mapping.Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    # Resolve to the bundled auto_map-stripped config dir first (see
    # collect_mla_module.create_attention_layer for why: the remote-code
    # path's file lock serializes parallel workers).
    model_path = _resolve_local_model_path(model_path)

    _cfg_dict, _ = transformers.PretrainedConfig.get_config_dict(model_path)
    original_architecture = _cfg_dict.get("architectures", [_cfg_dict.get("model_type", "unknown")])[0]
    # The VL checkpoint nests the text model under text_config; the module
    # benchmark models the text decoder only, mirroring
    # modeling_minimaxm3.get_text_config (:307-337@v1.4.0rc2).
    text_cfg_dict = _cfg_dict.get("text_config", _cfg_dict)

    sparse_cfg_dict = text_cfg_dict.get("sparse_attention_config")
    if not sparse_cfg_dict:
        raise ValueError(
            f"model {model_path} has no sparse_attention_config; the MSA module "
            "collector only supports MiniMax-M3 sparse-attention checkpoints"
        )

    def _required_sparse_field(key: str) -> int:
        # Unresolvable artifact configuration fails loudly: substituting a
        # literal here would benchmark a different sparse geometry and persist
        # it under this model's identity.
        if key not in sparse_cfg_dict:
            raise RuntimeError(
                f"checkpoint sparse_attention_config is missing '{key}' "
                f"(present: {sorted(sparse_cfg_dict)}); refusing to default the sparse geometry"
            )
        return int(sparse_cfg_dict[key])

    # NOTE: sparse_attention_config is intentionally NOT passed as a
    # from_pretrained kwarg. load_pretrained_config forwards every kwarg into
    # the registered config class (config_utils.py:1119-1122@v1.4.0rc2), and
    # PretrainedConfig.from_dict setattr's any kwarg whose name matches an
    # existing config attribute — the M3 checkpoint's config.json carries a
    # `sparse_attention_config` dict, so the typed object would clobber the dict
    # the modeling code reads (MiniMaxM3Attention.__init__,
    # modeling_minimaxm3.py:1222-1229). It is set on the ModelConfig after
    # loading instead; the attention-backend dispatch reads it at layer
    # construction time, which happens later.
    model_config = ModelConfig.from_pretrained(
        model_path,
        mapping=mapping,
        enable_min_latency=False,
        use_cuda_graph=False,
        force_dynamic_quantization=False,
        spec_config=None,
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

    # The registry-patched load returns a base PretrainedConfig, which keeps
    # torch_dtype as the JSON string; the typed remote-code config serving
    # loads carries a torch.dtype. Normalize so downstream dtype= kwargs
    # (RMSNorm/Linear construction) see the same object serving sees.
    if isinstance(getattr(pretrained_config, "torch_dtype", None), str):
        pretrained_config.torch_dtype = getattr(torch, pretrained_config.torch_dtype)

    # Single-layer benchmark surgery: one SPARSE layer with a dense MLP so
    # layer.self_attn is the MSA module under test. get_sparse_layer_ids /
    # get_moe_layer_ids validate len(freq) == num_hidden_layers
    # (modeling_minimaxm3.py:442-488@v1.4.0rc2), so the per-layer pattern
    # arrays are truncated consistently.
    pretrained_config.num_hidden_layers = 1
    sparse_cfg = dict(pretrained_config.sparse_attention_config)
    sparse_cfg["sparse_attention_freq"] = [1]
    if "sparse_disable_index_value" in sparse_cfg:
        flags = sparse_cfg["sparse_disable_index_value"]
        sparse_cfg["sparse_disable_index_value"] = [flags[-1] if isinstance(flags, list) else flags]
    if getattr(pretrained_config, "moe_layer_freq", None) is not None:
        pretrained_config.moe_layer_freq = [0]

    # TP-shard emulation: the layer is built unsharded with one rank's head counts
    # (_shard_head_counts), so the framework's own sharding at tp 1 reproduces them.
    tp_size, num_kv_heads, num_index_heads = _shard_head_counts(
        int(pretrained_config.num_attention_heads),
        int(pretrained_config.num_key_value_heads),
        _required_sparse_field("sparse_num_index_heads"),
        num_heads,
    )
    pretrained_config.num_attention_heads = num_heads
    pretrained_config.num_key_value_heads = num_kv_heads
    sparse_cfg["sparse_num_index_heads"] = num_index_heads
    pretrained_config.sparse_attention_config = sparse_cfg

    # Serving constructs the M3 runtime with an explicit
    # MiniMaxM3SparseAttentionConfig (llm_args.py:1084-1256@v1.4.0rc2;
    # _validate_sparse_attention_runtime_config, modeling_minimaxm3.py:385-439,
    # rejects a model built without it), and ModelConfig.from_pretrained only
    # builds sparse configs for the DSA/CSA2/DeepSeek-V4 checkpoints. Build it
    # from the checkpoint's sparse dict — the same fields MiniMaxM3Attention
    # reads (modeling_minimaxm3.py:1222-1229) — with this shard's index heads.
    # indexer_kv_dtype and fuse_qkv_index_projection keep their defaults (bf16,
    # unfused), the BF16-KV configuration; the fused FP8 producer needs an FP8
    # or NVFP4 KV cache (modeling_minimaxm3.py:421-439).
    sparse_attention_config = MiniMaxM3SparseAttentionConfig(
        sparse_num_index_heads=num_index_heads,
        sparse_index_dim=_required_sparse_field("sparse_index_dim"),
        sparse_block_size=_required_sparse_field("sparse_block_size"),
        sparse_topk_blocks=_required_sparse_field("sparse_topk_blocks"),
        sparse_init_blocks=_required_sparse_field("sparse_init_block"),
        sparse_local_blocks=_required_sparse_field("sparse_local_block"),
        sparse_disable_index_value=True,
        implementation=_msa_implementation(get_sm_version()),
    )
    # from_pretrained returns a frozen ModelConfig that refuses this field
    # (model_config.py:367-382, frozen at :1558).
    object.__setattr__(model_config, "sparse_attention_config", sparse_attention_config)

    _apply_gemm_type_quant(model_config, gemm_type)

    aux_stream = torch.cuda.Stream(device=device)
    aux_stream_dict = {
        AuxStreamType.Attention: aux_stream,
        AuxStreamType.MoeShared: aux_stream,
        AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(device=device),
    }

    layer = MiniMaxM3DecoderLayer(
        model_config=model_config,
        layer_idx=0,
        aux_stream_dict=aux_stream_dict,
    )

    # Mirror serving's exclude_modules pass (see
    # collect_mla_module.create_attention_layer for the full rationale);
    # M3 combos currently set exclude_modules=None so this is a no-op kept
    # for parity with future quant translations.
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

    attn_module = layer.self_attn
    if not getattr(attn_module, "is_sparse_attention_layer", False):
        raise RuntimeError("benchmark layer came out dense; sparse_attention_freq surgery did not take")
    if attn_module.sparse_num_index_heads != num_index_heads:
        raise RuntimeError(
            f"the layer holds {attn_module.sparse_num_index_heads} index heads, "
            f"not the {num_index_heads} of a tp={tp_size} shard"
        )

    if sparse_attention_config.implementation == "msa":
        # Serving's warmup compiles the MSA indexer's KV-split proxy variants, which a
        # prefill of more than 64 tokens over a cached prefix of 4096 or more selects
        # and no warmup forward plans (_prewarm_minimax_m3_msa_indexer,
        # pyexecutor/engine/runners/decoder/warmup.py:1477-1511@v1.4.0rc2;
        # msa_indexer.py:163-305). Without it the first such point compiles inline,
        # for over a minute.
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
            prewarm_split_proxy_variants,
        )

        with torch.inference_mode():
            prewarm_split_proxy_variants(torch.bfloat16, num_index_heads, 128)
        torch.cuda.synchronize()

    _autotune_projections(attn_module, _SERVING_WARMUP_NUM_TOKENS, device)
    return attn_module, model_config, original_architecture


# Serving tunes every tunable op in one autotune() pass at warmup: a context forward of
# curr_max_num_tokens tokens (max_num_tokens, 8192 by default, unless the KV pool holds
# fewer), then a one-token generation forward (_run_autotuner_warmup,
# pyexecutor/engine/runners/decoder/warmup.py:1050-1104, 1157-1159@v1.4.0rc2). In the M3
# attention module the only tunable ops are its projection GEMMs (collect_gemm.py
# explains each GEMM type: nvfp4 tunes its backend per power-of-2 bucket up to the
# pass's M; fp8_block on SM100/103 tunes its quant kernel at the pass's M only and
# JITs DeepGEMM; bf16 calls no tunable op): nothing under
# attention/backends/sparse/minimax_m3, fmha/msa_prefill.py or fmha/msa_decode.py
# calls the AutoTuner. Each projection sees the forward's new-token count, so every
# projection is tuned at serving's two warmup token counts when the layer is built,
# and a context point with more new tokens than that first gets a pass at its own
# count, which a deployment that runs it (max_num_tokens >= that count) tunes at
# warmup. Same procedure as collect_mla_module._autotune_projections.
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
        f"  [MSA module autotune] {len(projections)} projections at num_tokens={list(num_tokens)}: "
        f"{time.perf_counter() - start:.1f} s"
    )


# ═══════════════════════════════════════════════════════════════════════
# KV Cache + Metadata
# ═══════════════════════════════════════════════════════════════════════


def _resolve_attention_backend_cls(model_config: ModelConfig):
    """Resolve the sparse attention backend class exactly like serving.

    Serving lowers the llm_args config into SparseParams and passes
    ``sparse_params`` (pyexecutor/model_engine.py:561-566@v1.4.0rc2); the
    factory routes ``implementation == "msa"`` to MiniMaxM3MsaSparseAttention
    after ensure_msa_available and everything else to the Triton reference
    (attention/backends/sparse/registry.py:60-75).
    """
    sparse_params = model_config.sparse_attention_config.to_sparse_params(
        pretrained_config=model_config.pretrained_config
    )
    return get_attention_backend(model_config.attn_backend, sparse_params=sparse_params)


# One MiniMax-M3 sparse attention layer per (model, heads, gemm type) per worker —
# same pattern and rationale as collect_mla_module._cached_attention_layer /
# collect_dsv4_attn._cached_dsv4_attention_module: the layer build does not depend
# on the point; the per-case KV cache manager, metadata, dry run and timing are
# untouched (b300 base run: msa_context_module 1.8 GPU-s per one-row case).
_LAYER_CACHE: dict = {}


def _cached_msa_attention_layer(model_path: str, num_heads: int, gemm_type: str, device: str):
    key = (model_path, int(num_heads), gemm_type, device)
    hit = _LAYER_CACHE.get(key)
    if hit is not None:
        return hit
    if _LAYER_CACHE:
        _LAYER_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()
    entry = create_msa_attention_layer(model_path=model_path, num_heads=num_heads, gemm_type=gemm_type, device=device)
    _LAYER_CACHE[key] = entry
    return entry


# One 32 MiB GPU allocation unit for each pool of the M3 pool group (main K/V and
# index-K); see _kv_pool_tokens.
_M3_POOL_ROUNDING_BYTES = 2 * (32 << 20)


def _kv_pool_tokens(
    request_tokens: int, kv_bytes_per_token: int, index_bytes_per_token: int, tokens_per_block: int
) -> int:
    """KvCacheConfig.max_tokens for a pool that holds ``request_tokens`` (whole pages).

    The pool is sized for one case's requests; its size is not a timing input. The
    indexer reads index-K through a zero-copy pool view resolved when the manager is
    built (cache_manager.py:394-417@v1.4.0rc2, get_index_k_buffer :722-738;
    MiniMaxM3MsaSparseAttentionMetadata.msa_idx_k_cache, msa_backend.py:1209-1211),
    and the sparse kernels reach K/V through the block table. Two corrections to the
    request size:

    * max_tokens becomes a byte quota at the K/V bytes per token alone
      (KVCacheManagerV2._get_quota_from_max_tokens_impl and get_layer_bytes_per_token,
      kv_cache_manager_v2.py:1658-1670, 2623-2640, 6389-6438), while the pool group
      also stores every sparse layer's index-K page
      (MiniMaxM3KVCacheManagerV2._extra_buffers_per_layer, cache_manager.py:672-690).
      The token count is scaled up by that ratio: at one KV head the bare request
      size holds two thirds of the requests.
    * Past one KV head an index-K page is smaller than a K/V page and sits in a pool
      of its own in the same pool group, and V2 splits the quota with every pool
      rounded up to the GPU allocation unit, 2 MiB growing to 32 MiB with the quota
      (StorageManager::computeSlotCountForLevel,
      kv_cache_manager_v2/storageManager.cpp:2310-2318;
      CacheLevelStorage::ratioToSlotCountList, storage/core.cpp:927-1010), the
      rounding that left collect_mla_module's DSA pools short of one request. One
      largest unit per pool covers it.
    """
    tokens = _ceil_div(request_tokens * (kv_bytes_per_token + index_bytes_per_token), kv_bytes_per_token)
    return _round_up(tokens + _ceil_div(_M3_POOL_ROUNDING_BYTES, kv_bytes_per_token), tokens_per_block)


def create_kv_cache_and_metadata(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    prefix_len: int = 0,
    device: str = "cuda:0",
):
    """Create the M3 KV cache manager and attention metadata via framework
    utilities, mirroring serving's construction.

    Cache manager: get_kv_cache_manager_cls routes every minimax_m3 model to
    MiniMaxM3KVCacheManagerV2, whatever use_kv_cache_manager_v2 says
    (pyexecutor/_util.py:438-446@v1.4.0rc2 -> sparse/registry.py:24-56; the
    manager is shared by the Triton and MSA backends). Kwargs mirror the generic
    branch serving builds it in (_util.py:3867-3893): CacheType.SELF, per-layer
    num_kv_heads list, sparse_attention_config + pretrained_config forwarded.
    The single benchmark layer is declared sparse explicitly
    (sparse_layer_ids=[0]) — the manager's num_layers-based default assumes the
    full checkpoint's "first 3 dense" convention (sparse/minimax_m3/
    cache_manager.py:336-340) which does not apply to a one-layer build.

    Metadata: the resolved backend's Metadata class, with the constructor
    kwargs of build_attention_metadata (pyexecutor/engine/metadata.py:48-90),
    builds the sparse runtime metadata inside prepare() from the standard
    AttentionMetadata fields, populated as the decoder input builder populates
    them (pyexecutor/engine/runners/decoder/inputs.py): seq_lens carries the
    current chunk's new-token counts, the cached prefix travels via
    kv_cache_params.num_cached_tokens_per_seq (:1012-1014 context, :1383
    decode), and kv_lens = cached + seq_lens is derived internally.
    Two framework contracts exist, discriminated by the resolved Metadata
    class (never the version string):

    * Triton (implementation="triton"): prepare() attaches the prebuilt
      runtime metadata + out_cache_loc as the ``attn_metadata.minimax_m3``
      dict (triton_metadata.py:771-866), which the model layer's
      _triton_sparse_attention_core reads (modeling_minimaxm3.py:2392-2497).
    * MSA (implementation="msa"): the Metadata subclasses
      TrtllmAttentionMetadata; __post_init__ reads sparse_metadata_params,
      allocates flat CUDA-graph-stable buffers and settles the decode kernels'
      geometry (msa_backend.py:222-227, 368-543); prepare() (:660-670) fills
      the cache-write buffers (_build_msa_fields, :1028-1207, setting
      _msa_fields_ready), describes a step's generation rows as
      msa_decode_span (:686-735) and builds fmha_sm100 plans for its context
      rows only (_build_step_plans, :897-1026). The model layer consumes it
      via write_layer_caches + run_indexer + the inherited TrtllmAttention
      forward (modeling_minimaxm3.py:2117-2183).

    Returns (kv_cache_manager, attn_metadata, kernel_source) where
    kernel_source records the resolved backend: "msa_fmha_sm100" for the MSA
    path, "default" for the Triton reference (the label the SM90 rc20 dataset
    already carries). The manager is shut down when this raises.
    """
    config = model_config.pretrained_config
    mapping = model_config.mapping
    sparse_attention_config = model_config.sparse_attention_config

    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    num_kv_heads = int(config.num_key_value_heads)
    # Serving forces the KV page size per implementation: 128 for "msa"
    # (the fmha_sm100 kernels and the CuTe DSL scorer take 128-token pages
    # only), 32 for the Triton reference (py_executor_creator.py:604-608@v1.4.0rc2).
    tokens_per_block = 128 if sparse_attention_config.implementation == "msa" else 32

    prefix_len = int(prefix_len) if is_context else 0

    if is_context:
        max_seq = prefix_len + seq_len + 1
        total_tokens = seq_len * batch_size
        seq_len_q = seq_len
        kv_cache_len = prefix_len
    else:
        # Generation KV coordinate contract: the getter's `kv_cache_len` is
        # the CACHED length, the current token decodes at position seq_len,
        # and the measured total is seq_len + 1 — exactly the coordinate the
        # loader keys the row at (s = isl + step = 1 + seq_len). An earlier
        # revision cached seq_len - 1 tokens, which measured total seq_len
        # while persisting the seq_len + 1 coordinate (one-token key skew;
        # shipped rows from that revision were re-keyed step -> step - 1).
        max_seq = seq_len + 2
        total_tokens = batch_size
        seq_len_q = 1
        kv_cache_len = seq_len

    # --- KV cache manager ---
    # bf16 K/V (run_msa_module) and an index-K of indexer_kv_dtype (bf16 by default).
    kv_bytes_per_token = 2 * num_kv_heads * head_dim * 2
    index_elem_bytes = 1 if sparse_attention_config.indexer_kv_dtype == "fp8" else 2
    kv_tokens = _kv_pool_tokens(
        batch_size * _round_up(max_seq, tokens_per_block),
        kv_bytes_per_token,
        sparse_attention_config.sparse_index_dim * index_elem_bytes,
        tokens_per_block,
    )
    kv_cache_config = KvCacheConfig(
        max_tokens=kv_tokens,
        tokens_per_block=tokens_per_block,
        # The sparse-attention path serves with block reuse disabled (the
        # curated config examples/configs/curated/minimax-m3-throughput.yaml).
        enable_block_reuse=False,
    )
    # Capacity is refused before allocation: the manager commits GPU memory for its
    # whole quota when it is built (GpuSlotPool, kv_cache_manager_v2/storage/core.cpp:
    # 322-356) and zeroes it (cache_manager.py:419-436), kv_tokens K/V bytes over
    # max_util_for_resume (kv_cache_manager_v2.py:1658-1670).
    pool_bytes = kv_tokens * kv_bytes_per_token / kv_cache_config.max_util_for_resume
    free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
    if pool_bytes > free_bytes:
        raise RuntimeError(
            f"FIXME(kernel-limit): does not fit: the KV cache pool needs {pool_bytes / 2**30:.1f} GiB, "
            f"{free_bytes / 2**30:.1f} GiB free"
        )
    kv_cache_manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
    # Deviation (MSA generation rows, SM100/103): max_seq_len is this request's
    # length, where serving passes the deployment's. A pure-decode step scores and
    # selects blocks over the manager's worst-case block extent
    # (msa_worst_case_max_k_tiles, msa_backend.py:78-102, 328-337, read by
    # run_indexer :1522-1528@v1.4.0rc2), and that extent picks the selection
    # kernel (launchMinimaxM3SelectBlocks, cpp/tensorrt_llm/kernels/
    # minimaxM3SelectBlocks.cu:939-978): warp kernels up to 128 blocks, a histogram
    # kernel beyond. A deployment with max_seq_len above 16K tokens runs the
    # histogram kernel for every row, where rows below 16K tokens here run a warp
    # kernel. One small kernel per layer; direction and size are unmeasured. The
    # Triton path sizes by the step's own lengths. Retire it by timing at a
    # deployment envelope once the table keys on one.
    kv_cache_manager = kv_cache_manager_cls(
        kv_cache_config,
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=[num_kv_heads],
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=torch_dtype_to_binding(torch.bfloat16),
        sparse_attention_config=sparse_attention_config,
        pretrained_config=config,
        sparse_layer_ids=[0],
        disable_index_value_layer_ids=[0],
        sparse_index_dim=sparse_attention_config.sparse_index_dim,
    )

    try:
        attn_metadata, kernel_source = _prepare_metadata(
            model_config,
            kv_cache_manager,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            kv_cache_len=kv_cache_len,
            total_tokens=total_tokens,
            is_context=is_context,
        )
    except BaseException:
        kv_cache_manager.shutdown()
        raise
    return kv_cache_manager, attn_metadata, kernel_source


def _prepare_metadata(
    model_config: ModelConfig,
    kv_cache_manager,
    *,
    batch_size: int,
    seq_len_q: int,
    kv_cache_len: int,
    total_tokens: int,
    is_context: bool,
):
    """Register the case's requests, then build and prepare the backend's metadata."""
    config = model_config.pretrained_config
    request_ids = list(range(batch_size))
    # token_nums = past_kv_len + input_len (add_dummy_requests docstring,
    # kv_cache_manager_v2.py:5892-5935@v1.4.0rc2); is_gen marks decode requests so
    # the committed-history hint matches a decode step's cache state. It returns
    # None, with nothing left allocated, when the pool cannot hold the requests.
    token_nums = [kv_cache_len + seq_len_q] * batch_size
    if kv_cache_manager.add_dummy_requests(request_ids, token_nums=token_nums, is_gen=not is_context) is None:
        raise RuntimeError("FIXME(kernel-limit): does not fit: the KV cache pool cannot hold the case's requests")

    metadata_cls = _resolve_attention_backend_cls(model_config).Metadata
    is_msa = issubclass(metadata_cls, MiniMaxM3MsaSparseAttentionMetadata)

    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    attn_metadata = metadata_cls(
        # Capacity bounds: serving passes the engine's (build_attention_metadata,
        # engine/metadata.py:48-90@v1.4.0rc2); one prepared batch needs only its own.
        max_num_requests=batch_size,
        max_num_tokens=total_tokens,
        # max_batch_size * max_beam_width, beam width 1 (metadata.py:80).
        max_num_sequences=batch_size,
        # The GQA ratio (_get_max_num_heads_per_kv, metadata.py:29-45), of the shard here.
        max_num_heads_per_kv=num_q_heads // num_kv_heads,
        # The lowered sparse geometry the MSA metadata's __post_init__ reads
        # (metadata.py:72-76; msa_backend.py:222-227).
        sparse_metadata_params=model_config.sparse_attention_config.to_sparse_metadata_params(pretrained_config=config),
        kv_cache_manager=kv_cache_manager,
        mapping=model_config.mapping,
        # Per-request new-token counts (decoder/inputs.py:1011 context; one per
        # decode row).
        seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int32),
        position_ids=None,
        num_contexts=batch_size if is_context else 0,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[kv_cache_len] * batch_size,
        ),
        cross=None,
        request_ids=request_ids,
        # Current-chunk token count for a context request (decoder/inputs.py:1012),
        # the prompt length for a decode request (:1368); read by the generic
        # attention op only, not by the M3 sparse paths.
        prompt_lens=[seq_len_q if is_context else kv_cache_len] * batch_size,
        # The engine's runtime features (py_executor_creator.py:717-722): block reuse
        # off on this path, so a cached prefix is chunked prefill.
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=bool(is_context and kv_cache_len > 0),
            cache_reuse=False,
        ),
        all_rank_num_tokens=None,
    )

    attn_metadata.prepare()
    if is_msa:
        # MSA contract: prepare() must have filled the cache-write buffers
        # (_build_msa_fields sets _msa_fields_ready, msa_backend.py:1028-1207@v1.4.0rc2).
        if not attn_metadata._msa_fields_ready:
            raise RuntimeError(
                "MiniMaxM3MsaSparseAttentionMetadata.prepare() did not build the MSA "
                "cache-write buffers; the KV cache manager is not the M3 sparse manager"
            )
        # A pure decode step runs every row on the dedicated decode kernels, with a
        # span from row 0 and no fmha_sm100 plan; a pure prefill step has no span
        # and runs fmha_sm100 on the plans prepare() built (_set_decode_span,
        # _build_step_plans, msa_backend.py:686-735, 897-1026).
        span = attn_metadata.msa_decode_span
        if is_context:
            planned = span is None and attn_metadata.msa_prefill_proxy_plan is not None
        else:
            planned = span is not None and span.row_first == 0 and attn_metadata.msa_prefill_proxy_plan is None
        if not planned:
            raise RuntimeError(
                f"MSA metadata does not describe a pure {'prefill' if is_context else 'decode'} step "
                f"(decode span {span}); the dispatch would split it between fmha_sm100 and the decode kernels"
            )
        kernel_source = "msa_fmha_sm100"
    else:
        # Triton contract: prepare() attaches the prebuilt runtime metadata dict
        # (triton_metadata.py:863-866@v1.4.0rc2).
        if getattr(attn_metadata, "minimax_m3", None) is None:
            raise RuntimeError(
                "MiniMaxM3AttentionMetadata.prepare() did not build the minimax_m3 "
                "attachment; the KV cache manager is not the M3 sparse manager"
            )
        # Keep the label the SM90 rc20 dataset already carries for this path.
        kernel_source = "default"

    return attn_metadata, kernel_source


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


def run_msa_module(
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
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
    dry_run: bool = True,
):
    """Run a single MiniMax-M3 MSA module-level benchmark point.

    ``dry_run`` False skips the checking forward before timing: a series has
    already run it for an earlier point of this seq_len (module_series).
    """
    torch.cuda.set_device(device)
    torch_device = torch.device(device)

    if kv_cache_dtype != "bfloat16" or compute_dtype != "bfloat16":
        raise ValueError(
            f"this collector builds the M3 sparse attention on a bf16 KV cache only "
            f"(skips.py leaves FP8-KV cases out); got compute={compute_dtype}, kv={kv_cache_dtype}"
        )

    is_context = "context" in perf_filename
    prefix_len = int(prefix_len) if is_context else 0
    phase = "context" if is_context else "generation"
    print(
        f"\n[MSA module] {phase} b={batch_size}, s={seq_len}, "
        f"prefix={prefix_len}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    attn_module, model_config, original_architecture = _cached_msa_attention_layer(
        model_path, num_heads, gemm_type, device
    )
    # A context point past serving's warmup token count tunes its projections at
    # its own count first (_SERVING_WARMUP_NUM_TOKENS).
    new_tokens = batch_size * seq_len if is_context else batch_size
    if new_tokens > _TUNED_NUM_TOKENS.get(attn_module, 0):
        _autotune_projections(attn_module, (new_tokens,), device)

    kv_cache_manager, attn_metadata, kernel_source = create_kv_cache_and_metadata(
        model_config=model_config,
        batch_size=batch_size,
        seq_len=seq_len,
        is_context=is_context,
        prefix_len=prefix_len,
        device=device,
    )

    import tensorrt_llm._torch.utils as _trtllm_utils

    _prev_extra_attrs = getattr(_trtllm_utils._model_extra_attrs, "attrs", None)
    try:
        # Cross-check: the metadata contract validated above must match the
        # backend instance the layer actually dispatches to — the model layer
        # branches on isinstance(self.attn, MiniMaxM3MsaSparseAttention)
        # (_dispatch_attention_backend, modeling_minimaxm3.py:2070-2115@v1.4.0rc2).
        # A mismatch would benchmark one path against the other's metadata and
        # mislabel kernel_source.
        layer_backend_name = type(getattr(attn_module, "attn", None)).__name__
        layer_is_msa = layer_backend_name == "MiniMaxM3MsaSparseAttention"
        if layer_is_msa != (kernel_source == "msa_fmha_sm100"):
            raise RuntimeError(
                f"metadata contract ({kernel_source}) does not match the layer's attention backend "
                f"({layer_backend_name})"
            )

        hidden_size = model_config.pretrained_config.hidden_size
        # Positions are an int32 [1, num_tokens] model input
        # (pyexecutor/engine/runners/decoder/inputs.py:1948, 2397@v1.4.0rc2).
        if is_context:
            num_tokens = seq_len * batch_size
            position_ids = (
                torch.arange(prefix_len, prefix_len + seq_len, device=torch_device, dtype=torch.int32)
                .repeat(batch_size)
                .unsqueeze(0)
            )
        else:
            num_tokens = batch_size
            position_ids = torch.full((1, batch_size), seq_len, device=torch_device, dtype=torch.int32)

        hidden_states = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=torch_device,
        )

        if dry_run:
            with model_extra_attrs(model_config.extra_attrs):
                get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
                try:
                    with torch.inference_mode():
                        attn_module.forward(position_ids, hidden_states, attn_metadata)
                except Exception:
                    print("  Dry run failed:")
                    traceback.print_exc()
                    raise

        _trtllm_utils._model_extra_attrs.attrs = model_config.extra_attrs
        _trtllm_utils._model_extra_attrs.attrs["attention_metadata"] = weakref.ref(attn_metadata)

        def kernel_func():
            attn_module.forward(position_ids, hidden_states, attn_metadata)

        # Measurement mode mirrors serving's execution mode per backend/phase.
        # MSA prefill runs eagerly in serving: its fmha_sm100 plans are per-step
        # host tuples built in prepare(), and the metadata refuses a CUDA graph
        # step that carries context rows (_check_capture_is_pure_decode,
        # msa_backend.py:737-750@v1.4.0rc2; _build_step_plans :897-1026) — so MSA
        # context is measured eagerly. MSA decode runs the graph-safe decode
        # kernels on graph-stable buffers and keeps graph-mode measurement, as
        # does the Triton path for both phases (unchanged from the SM90 rc20
        # collection).
        use_cuda_graph = not (kernel_source == "msa_fmha_sm100" and is_context)

        with benchmark_with_power(
            device=torch_device,
            kernel_func=kernel_func,
            num_warmups=warming_up,
            # Graph-timed cases: a dry run of this seq_len already executed this
            # forward, three eager passes suffice before capture. Eager-timed MSA
            # prefill keeps the full warmup since it IS the measurement's warmup.
            num_eager_warmups=3 if use_cuda_graph else warming_up,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=use_cuda_graph,
        ) as results:
            pass

        latency = results["latency_ms"]

        if is_context:
            isl = seq_len
            step = prefix_len
        else:
            isl = 1
            step = seq_len

        op_name = f"msa_{phase}_module"

        if not log_perf(
            item_list=[
                {
                    "model": model_path,
                    "architecture": original_architecture,
                    "mla_dtype": compute_dtype,
                    "kv_cache_dtype": kv_cache_dtype,
                    "gemm_type": gemm_type,
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
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        ):
            # A failed write with a normal return would report the case as
            # successful while silently shrinking the dataset.
            raise RuntimeError(f"failed to persist MSA row to {perf_filename}")

        print(
            f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
            f"prefix={prefix_len}, gemm={gemm_type}, backend={kernel_source}: "
            f"{latency:.4f} ms"
        )

        return latency
    finally:
        # Restore the process-global extra-attrs slot and always release the
        # KV pool: the CLI loop continues past a failed case, so a leaked
        # pool would surface as spurious OOM in later cases, and a stale
        # weakref here would point at the shut-down manager.
        if _prev_extra_attrs is not None:
            _trtllm_utils._model_extra_attrs.attrs = _prev_extra_attrs
        elif hasattr(_trtllm_utils._model_extra_attrs, "attrs"):
            del _trtllm_utils._model_extra_attrs.attrs
        _cleanup(kv_cache_manager)


def run_msa_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py."""
    return run_msa_module(
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        prefix_len=prefix_len,
        perf_filename=perf_filename,
        model_path=model_path,
        device=device,
    )


def run_msa_module_series(
    points,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    *,
    model_path: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
) -> list[float]:
    """Measure the (seq_len, prefix_len) points of one MSA module key in one task.

    Every point runs through run_msa_module, so its rows are those of a one-point
    case. The KV cache manager stays per point: the manager's max_seq_len sets the
    block extent of a decode step's top-k selection (msa_worst_case_max_k_tiles,
    msa_backend.py:78-102, 328-337@v1.4.0rc2), which picks the selection kernel
    (launchMinimaxM3SelectBlocks, cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu),
    so a manager sized for the largest point would change what the smaller points
    measure.
    """

    def run_point(seq_len: int, prefix_len: int, dry_run: bool) -> float:
        return run_msa_module(
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype=compute_dtype,
            gemm_type=gemm_type,
            perf_filename=perf_filename,
            prefix_len=prefix_len,
            model_path=model_path,
            device=device,
            warming_up=warming_up,
            test_ite=test_ite,
            dry_run=dry_run,
        )

    label = f"MSA module series b={batch_size} heads={num_heads} gemm={gemm_type} kv={kv_cache_dtype}"
    return run_points(points, run_point, label=label)


def run_msa_module_series_worker(
    points,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker entry for one series task: run_msa_module_worker's positional fields, with
    the list of (seq_len, prefix_len) points in place of seq_len and prefix_len."""
    return run_msa_module_series(
        points,
        batch_size,
        num_heads,
        kv_cache_dtype,
        compute_dtype,
        gemm_type,
        perf_filename,
        model_path=model_path,
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


def _perf_file_for(mode: str) -> str:
    return PerfFile.MSA_CONTEXT_MODULE if mode == "context" else PerfFile.MSA_GENERATION_MODULE


def main():
    all_model_specs = get_mla_module_model_specs(attention_type="msa", apply_model_filter=False)
    model_names = [spec.model_path for spec in all_model_specs]

    parser = argparse.ArgumentParser(description="MiniMax-M3 MSA module-level collector for TRT-LLM")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--model", type=str, default=None, choices=model_names)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--gemm-type", type=str, choices=["bfloat16", "fp8_block", "nvfp4"], default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    perf_filename = _perf_file_for(args.mode)
    if not args.quick:
        # The full sweep is the op plan's rows; --model targets that checkpoint, as collect.py --model-path does.
        if args.model:
            os.environ["COLLECTOR_MODEL_PATH"] = args.model
        rows = plan(f"msa_{args.mode}_module", "trtllm", get_sm_version()).rows
        rows = [
            row
            for row in rows
            if (args.num_heads is None or row[2] == args.num_heads)
            and (args.gemm_type is None or row[5] == args.gemm_type)
            and (args.batch_size is None or row[1] == args.batch_size)
        ]
        print(f"Running {len(rows)} {args.mode} MSA module series...")
        num_failed = 0
        for points, b, h, kv_dtype, compute, gemm, model_path in rows:
            print(f"\n[MSA] {model_path} b={b} heads={h} gemm={gemm}: {len(points)}p")
            try:
                run_msa_module_series(
                    points, b, h, kv_dtype, compute, gemm, perf_filename, model_path=model_path, device=args.device
                )
            except Exception as e:
                print(f"  FAILED: b={b}, heads={h}, gemm={gemm}: {e}")
                traceback.print_exc()
                num_failed += 1
                torch.cuda.empty_cache()
                gc.collect()
        if num_failed:
            # Exit non-zero so an operator (and CI) can see failures without
            # scraping the log; per-point detail is printed above.
            print(f"{num_failed}/{len(rows)} series failed")
            sys.exit(1)
        return

    model_specs_to_run = (
        [spec for spec in all_model_specs if spec.model_path == args.model] if args.model else all_model_specs
    )
    for model_spec in model_specs_to_run:
        model_path = model_spec.model_path
        print(f"\n{'=' * 60}\nModel: {model_path}  |  Attention: MSA\n{'=' * 60}")
        run_msa_module(
            seq_len=args.seq_len or 2048,
            batch_size=args.batch_size or 4,
            num_heads=args.num_heads or 64,
            kv_cache_dtype="bfloat16",
            compute_dtype="bfloat16",
            gemm_type=args.gemm_type or "bfloat16",
            prefix_len=args.prefix_len,
            perf_filename=perf_filename,
            model_path=model_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()
