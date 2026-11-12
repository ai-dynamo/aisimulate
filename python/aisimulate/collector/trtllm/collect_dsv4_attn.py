# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 CSA/HCA attention module collector for TensorRT-LLM.

Benchmarks the full DeepSeek-V4 attention module — q/kv/o projections,
rotary, compressor, indexer (top-k selection) and the sparse MLA kernels —
for the four module tables the SDK's ``ContextDeepSeekV4AttentionModule`` /
``GenerationDeepSeekV4AttentionModule`` consume:

    dsv4_csa_context_module     dsv4_hca_context_module
    dsv4_csa_generation_module  dsv4_hca_generation_module

TRT-LLM ships the model class (``_torch/models/modeling_deepseekv4.py``) and
the sparse attention stack (``_torch/attention/backends/sparse/deepseek_v4/``:
DeepseekV4TrtllmAttention, DeepseekV4Indexer, DeepseekV4CacheManager,
Compressor). ``ModelConfig.from_pretrained`` auto-builds
``DeepSeekV4SparseAttentionConfig`` for architecture ``DeepseekV4ForCausalLM``
(_torch/model_config.py:1273-1365),
so this collector constructs ``DeepseekV4Attention`` through the
framework's own config/dispatch path — CSA vs HCA is expressed exclusively by
the checkpoint config's per-layer ``compress_ratios`` value (4 = CSA c4 +
indexer top-k; 128 = HCA c128), mirroring the sglang/vllm module collectors.

Row semantics follow the unified convention (same as sglang/vllm):
``num_heads`` is the RANK-LOCAL head count with a mandatory ``tp_size``
column (the loader derives native = num_heads * tp_size and validates the
local semantics, aic-core .../operations/dsv4.py); context rows carry
isl = fresh chunk tokens / step = cached prefix, generation rows carry
isl = 1 / step = past KV length. TP shards are simulated by shrinking the
config head/o_groups counts (num_attention_heads = native // tp,
o_groups = max(1, o_groups // tp)) in a single process, matching the vllm
collector; collective latency is therefore excluded, like both references.

Weights are dummy. The cached history's index K is written before timing
(collect_mla_module._write_indexer_history), so the CSA indexer top-k selects
among distinct scores and the rows need no top-k correction: no trtllm
``dsv4_csa_topk_calib`` table is produced, and the SDK's topK DELTA is then a
no-op ((calib or {}).get(...) -> delta 0.0, operations/dsv4.py).
"""

from __future__ import annotations

__compat__ = "trtllm==1.4.0rc2"

import contextlib
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import weakref

import torch

try:
    from registry_types import PerfFile

    from helper import benchmark_with_power, get_sm_version, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from registry_types import PerfFile

    from helper import benchmark_with_power, get_sm_version, log_perf


from collector.op_plans import plan_cases

ARCHITECTURE = "DeepseekV4ForCausalLM"

ATTN_KIND_TO_COMPRESS_RATIO = {"csa": 4, "hca": 128}

_MODEL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "aisimulate_core",
    "model_configs",
)


# ═══════════════════════════════════════════════════════════════════════
# Case population: the op plans' rows (collector/op_plans.py)
# ═══════════════════════════════════════════════════════════════════════


def get_dsv4_csa_context_test_cases():
    return plan_cases("dsv4_csa_context_module", "trtllm", get_sm_version())


def get_dsv4_hca_context_test_cases():
    return plan_cases("dsv4_hca_context_module", "trtllm", get_sm_version())


def get_dsv4_csa_generation_test_cases():
    return plan_cases("dsv4_csa_generation_module", "trtllm", get_sm_version())


def get_dsv4_hca_generation_test_cases():
    return plan_cases("dsv4_hca_generation_module", "trtllm", get_sm_version())


# ═══════════════════════════════════════════════════════════════════════
# Module construction (framework dispatch — no manual backend pinning)
# ═══════════════════════════════════════════════════════════════════════

# Size-1 cache of the last constructed attention module, keyed by
# (model_path, attn_kind, tp_size). collect.py workers are persistent and the
# case order groups shapes into long same-geometry runs, so consecutive tasks
# reuse one module and skip the ~30s construction (from_pretrained + weight
# creation + fp8 post-processing) that dominated the 42s/case cost — the
# per-case KV cache manager, metadata, CUDA-graph capture and benchmark are
# untouched, and reusing one module across batches is exactly what serving
# does.
_MODULE_CACHE: dict = {}
# Lower bound on the per-case KV token quota (see create_dsv4_kv_cache_and_metadata).
_KV_POOL_MIN_TOKENS = 2 * 1024 * 1024


def _cached_dsv4_attention_module(
    model_path: str, attn_kind: str, tp_size: int, device: str, kv_cache_dtype: str = "fp8"
):
    key = (model_path, attn_kind, int(tp_size), device, kv_cache_dtype)
    hit = _MODULE_CACHE.get(key)
    if hit is not None:
        return hit
    # Evict the previous geometry before building the next (bounded memory:
    # at most one module's weights are ever retained).
    if _MODULE_CACHE:
        _MODULE_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()
    entry = create_dsv4_attention_module(
        model_path=model_path, attn_kind=attn_kind, tp_size=tp_size, device=device, kv_cache_dtype=kv_cache_dtype
    )
    _MODULE_CACHE[key] = entry
    return entry


def generation_request_geometry(seq_len: int) -> dict[str, int]:
    """Decode-request state triple, single-sourced so the serving invariant
    cannot drift between construction sites.

    Serving population (pyexecutor/engine/runners/decoder/inputs.py, ordinary
    decode of a request without a previous tensor):
    ``past_seen_token_num = request.max_beam_num_tokens - 1`` (:1616),
    ``position_id = past_seen_token_num`` (:1625) and
    ``num_cached_tokens_per_seq.append(past_seen_token_num - ...)``
    (:1663-1665) — i.e. the new token's POSITION equals the PAST-SEEN/CACHED
    token count. (The draft-token branch at :1410-1422 starts its positions at
    the same past-seen count; this collector's cases are ordinary decode.)
    The collector's decode dummy request registers ``seq_len + 1`` beam tokens
    (``request_tokens = max_seq = seq_len + 1``), so past-seen == cached ==
    position == ``seq_len``, and the persisted row ``step`` (past-KV length,
    the unified row semantics) is the same value.
    """
    return {"num_cached_tokens": seq_len, "position": seq_len, "persisted_step": seq_len}


def _patched_config_dir(model_path: str, compress_ratio: int, tp_size: int) -> tuple[str, dict]:
    """Write a single-layer DSV4 config with the requested compress ratio and
    TP-local head counts into a temp dir.

    Uses the AIC-packaged config (no HF download / trust_remote_code lock —
    same rationale as collect_mla_module._resolve_local_model_path).
    ``compress_ratios=[compress_ratio]`` expresses CSA (4) vs HCA (128)
    exactly like the vllm collector (collect_dsv4_attn.py:133);
    ModelConfig.from_pretrained reads it into
    DeepSeekV4SparseAttentionConfig (_torch/model_config.py:1281-1365).
    """
    cfg_fname = model_path.replace("/", "--") + "_config.json"
    config_file = os.path.join(_MODEL_CONFIG_DIR, cfg_fname)
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"AIC packaged config not found for {model_path!r}: expected {config_file}")
    with open(config_file) as f:
        config = json.load(f)

    native_heads = int(config["num_attention_heads"])
    if native_heads % tp_size != 0:
        raise ValueError(f"tp_size={tp_size} does not divide native head count {native_heads} of {model_path}")
    local_heads = native_heads // tp_size
    # o_groups shards with TP like the SDK model's
    # local_o_groups = max(1, o_groups // tp_size)
    # (aic-core .../sdk/models/deepseek_v4.py:116).
    if "o_groups" not in config:
        # Fail loudly (case_authoring.md "unresolvable declarations"): a
        # defaulted value would silently benchmark a different attention
        # geometry, and o_groups is not part of the logged row. Both packaged
        # DSV4 artifacts declare it (Flash 8, Pro 16).
        raise KeyError(
            f"AIC packaged config for {model_path!r} omits 'o_groups'; the TP-local geometry cannot be resolved"
        )
    o_groups = int(config["o_groups"])
    local_o_groups = max(1, o_groups // tp_size)

    config.pop("auto_map", None)
    config["architectures"] = [ARCHITECTURE]
    # The DSV4 release artifacts tag model_type "deepseek_ref" (remote-code
    # config class). TRT-LLM's native config class registers as
    # "deepseek_v4" (_torch/configs/__init__.py:84, written into transformers'
    # CONFIG_MAPPING at import, :55-141) and its defaults supply the DSV4 dims the
    # artifact json omits (kv_lora_rank=448, v_head_dim=512,
    # qk_nope_head_dim=448 — configs/deepseekv4.py:35-39), so
    # rewrite the type instead of pulling remote code.
    config["model_type"] = "deepseek_v4"
    config["num_hidden_layers"] = 1
    config["compress_ratios"] = [compress_ratio]
    config["num_attention_heads"] = local_heads
    config["o_groups"] = local_o_groups

    tmp_dir = tempfile.mkdtemp(prefix=f"aic_dsv4_{model_path.replace('/', '_')}_")
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return tmp_dir, {"native_heads": native_heads, "local_heads": local_heads}


def create_dsv4_attention_module(
    model_path: str,
    attn_kind: str,
    tp_size: int,
    device: str = "cuda:0",
    kv_cache_dtype: str = "fp8",
):
    """Build DeepseekV4Attention through TRT-LLM's own config path.

    Mirrors collect_mla_module.create_attention_layer: framework-constructed
    module, dummy weights, serving quant-exclusion pass. The attention module
    (an MLA subclass) contains the projections, indexer and compressor — the
    unit the SDK module tables bill.
    """
    # Registers model_type "deepseek_v4" -> DeepseekV4Config into transformers
    # CONFIG_MAPPING at import (_torch/configs/__init__.py:84, 141); required
    # before ModelConfig.from_pretrained resolves the patched config.
    import tensorrt_llm._torch.configs  # noqa: F401
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Attention
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    try:
        from .collect_mla_module import (
            _apply_gemm_type_quant,
            _give_index_scores_magnitude,
            initialize_dummy_weights,
        )
    except ImportError:
        from collect_mla_module import _apply_gemm_type_quant, _give_index_scores_magnitude, initialize_dummy_weights

    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    config_dir, head_info = _patched_config_dir(model_path, compress_ratio, tp_size)

    try:
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        model_config = ModelConfig.from_pretrained(
            config_dir,
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
            mm_encoder_only=False,
            attn_backend="TRTLLM",
            moe_backend="CUTLASS",
            skip_create_weights_in_init=True,
        )
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)

    if (
        model_config.sparse_attention_config is None
        or getattr(model_config.sparse_attention_config, "algorithm", None) != "deepseek_v4"
    ):
        raise RuntimeError(
            "ModelConfig.from_pretrained did not build a deepseek_v4 sparse_attention_config "
            f"for {model_path!r} — got {model_config.sparse_attention_config!r}. "
            "The framework dispatch contract changed; re-audit against the runtime version."
        )

    # The KV quant algo follows the lane: fp8 (the serving pool, op_plans._trtllm_dsv4_rows) sets
    # kv_cache_quant_algo=FP8 and the module runs the E4m3 FMHA cubins; a bf16 lane leaves it unset
    # so the module runs the QkvBfloat16 cubins — with FP8 pinned here a bf16 pool still produced E4m3
    # kernels (B200 capture).
    _apply_gemm_type_quant(model_config, "fp8_block", use_fp8_kv_cache=(kv_cache_dtype == "fp8"))

    # Provenance: print the RESOLVED kernel-selection knobs (same auto-build
    # code path default serving takes when the user sets no
    # sparse_attention_config; _torch/model_config.py:1116-1365). Audits
    # compare these against serving defaults. use_cute_dsl_paged_mqa_logits
    # defaults to True (llmapi/llm_args.py:1339-1344, inherited by the V4
    # config), so on the SM100 family the decode indexer runs the CuTe DSL paged
    # MQA logits kernel instead of DeepGEMM (sparse/dsa/indexer.py:92-100);
    # cute-dsl top-k stays off.
    _sc = model_config.sparse_attention_config
    print(
        "[dsv4-collector] resolved sparse config: "
        + ", ".join(
            f"{k}={getattr(_sc, k, None)}"
            for k in (
                "indexer_k_dtype",
                "use_cute_dsl_topk",
                "use_cute_dsl_paged_mqa_logits",
                "enable_heuristic_topk",
                "q_split_threshold",
                "skip_indexer_for_short_seqs",
                "index_topk",
                "compress_ratios",
            )
        )
    )

    if get_sm_version() == 90:
        # Serving publishes the requested pool dtype to the attention modules through the
        # model's extra attrs (pyexecutor/model_loader.py:2113-2114 extra_attrs['kv_cache_dtype']
        # = kv_cache_config.dtype; read by attention/mla.py:345). On Hopper that is
        # 'fp8_ds_mla' (see the KvCacheConfig below) and it also switches the DSV4 module
        # OFF the fused Q-FP8/KV-norm prologue (sparse/deepseek_v4/module.py:580-592
        # _is_fused_q_fp8_quant_enabled: "fp8_ds_mla ... does not use the fused Q FP8 path").
        # Without it the module takes the fused path, hands the footer-scale cache a
        # strided raw latent and mla_rope_inplace rejects it ("data must be contiguous",
        # H20 ctx capture) — a path serving never runs on Hopper.
        model_config.extra_attrs["kv_cache_dtype"] = "fp8_ds_mla"

    # DeepseekV4Attention takes the model's per-purpose stream dict, as
    # DeepseekV4DecoderLayer passes it (models/modeling_deepseekv4.py:1463-1470, 1970-1975).
    from tensorrt_llm._torch.utils import AuxStreamType

    aux_stream = torch.cuda.Stream(device=device)
    attn_module = DeepseekV4Attention(
        model_config=model_config,
        layer_idx=0,
        aux_stream_dict={
            AuxStreamType.Attention: aux_stream,
            AuxStreamType.MoeShared: aux_stream,
            AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(device=device),
        },
    )

    # Serving applies QuantConfig.exclude_modules before weight creation
    # (apply_quant_config_exclude_modules, _torch/models/modeling_utils.py:438-485, 714-720);
    # mirror it like collect_mla_module does (see the long citation there).
    quant_config = model_config.quant_config
    if quant_config is not None and quant_config.exclude_modules is not None:
        excluded_replacement = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for module_name, module in attn_module.named_modules():
            if getattr(module, "quant_config", None) is None:
                continue
            if quant_config.is_module_excluded_from_quantization(module_name):
                module.quant_config = excluded_replacement

    for module in attn_module.modules():
        if callable(getattr(module, "create_weights", None)):
            module.create_weights()
    attn_module.to(device)

    initialize_dummy_weights(attn_module)
    _give_index_scores_magnitude(attn_module)
    for module in attn_module.modules():
        if hasattr(module, "post_load_weights") and not getattr(module, "_weights_removed", False):
            module.post_load_weights()

    attn_module.eval()
    attn_module.requires_grad_(False)
    _autotune_dsv4_module(attn_module, model_config, _SERVING_AUTOTUNE_PASSES, device, kv_cache_dtype)
    return attn_module, model_config, head_info


# ═══════════════════════════════════════════════════════════════════════
# KV cache + metadata
# ═══════════════════════════════════════════════════════════════════════


def create_dsv4_kv_cache_and_metadata(
    model_config,
    attn_module,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    prefix_len: int = 0,
    device: str = "cuda:0",
    kv_cache_dtype_str: str = "fp8",
):
    """DSV4 cache manager + attention metadata, following the serving
    construction path with a DIRECT pinned citation on every hand-set field
    (layer_permissions.md metadata-parity rule; TensorRT-LLM paths under
    tensorrt_llm/_torch/, re-checked at the module's pin):

    - manager construction  -> pyexecutor/_util.py:3575-3609 (is_mla branch)
    - Metadata construction -> pyexecutor/engine/metadata.py:48-90
      (build_attention_metadata)
    - per-batch metadata assignment -> pyexecutor/engine/runners/decoder/
      inputs.py (seq_lens :2031-2035; request_ids/prompt_lens/num_contexts
      :2054-2056; KVCacheParams(use_cache=True, num_cached_tokens_per_seq)
      :2088-2095)
    - context per-request population -> inputs.py: chunk slicing and
      position_ids.extend(range(begin_compute, ...)) :985-1001, prompt length
      and past-seen prefix accounting :1012-1014
    - ordinary decode per-request population -> inputs.py:1610-1667
      (past_seen/position/cached/prompt_len)
    - decode dummy-request prompt_len proof -> pyexecutor/kv_cache/
      kv_cache_manager_v2.py:6052-6055 (is_gen: req.prompt_len = token_num - 1;
      py_prompt_len = prompt_len), consumed at inputs.py:1667
    """
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionRuntimeFeatures,
        KVCacheParams,
    )
    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType

    # DeepseekV4CacheManager derives from KVCacheManagerV2, which consumes the
    # llm-args pydantic KvCacheConfig (kv_cache_manager_v2.py reads fields
    # like enable_swa_scratch_reuse) — the config object serving forwards from
    # TorchLlmArgs, not the C++ executor binding the V1 managers take.
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    try:
        from .collect_mla_module import _write_indexer_history, get_kv_cache_manager_cls
    except ImportError:
        from collect_mla_module import _write_indexer_history, get_kv_cache_manager_cls

    config = model_config.pretrained_config
    mapping = model_config.mapping

    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim
    head_dim = kv_lora_rank + qk_rope_head_dim

    # DeepSeek-V4 serving default page size: the model's kv_cache_config default is
    # tokens_per_block=128 on every SM, 256 only for an explicit fp8_ds_mla pool off
    # Hopper (models/modeling_deepseekv4.py:2914-2925); DeepseekV4CacheManager accepts
    # 128 or 256 (attention/backends/sparse/deepseek_v4/cache_manager.py:363-366).
    tokens_per_block = 128

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
        kv_cache_len = generation_request_geometry(seq_len)["num_cached_tokens"]
    # Serving's max_seq_len is the ENGINE envelope, not the request length:
    # the DSV4 metadata sizes the HCA indices as
    # next_pow2(ceil(max_seq_len/128)) (sparse/deepseek_v4/metadata.py:244-253),
    # num_sparse_topk = window(128) + that, and the trtllmGen fmha kernel asserts
    # it is a multiple of 4
    # — request-sized envelopes (< 257) yield pow2 1/2 -> 129/130 and crashed
    # every tiny-KV HCA decode case on B200 (smoke round 3). Floor ONLY the
    # envelope; per-request state (add_dummy_requests token_nums, metadata
    # kv lens) stays the real request size, exactly as in serving — flooring
    # token_nums too registered 512-token dummy KV against 4-token metadata
    # and turned the crash into an IMA (smoke round 4).
    request_tokens = prefix_len + seq_len_q if is_context else max_seq
    max_seq = max(max_seq, 512)

    # KVCacheManagerV2 requires an explicit quota (max_tokens or
    # max_gpu_total_bytes; pyexecutor/kv_cache/kv_cache_manager_v2.py:1651-1674
    # "Quota not set")
    # — in serving, free_gpu_memory_fraction is converted to a byte quota by
    # the executor's KV estimation before the manager is built, so pass the
    # equivalent byte quota directly (half of currently-free device memory,
    # matching the DSV4 example's free_gpu_memory_fraction=0.5,
    # examples/models/core/deepseek_v4/README.md:148-151). A
    # collector-computed main-KV-shaped byte cap under-counted on V2 (the
    # byte quota spans main KV + SWA + compressor/indexer caches) and made
    # large-KV shapes fail dummy-request allocation ("Request ID not found in
    # IndexMapper", B200 smoke) — hence the token cap below
    # is handed to the manager's own arithmetic, never computed here.
    free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
    # Pool size is a per-case cost, not a measurement input: the manager
    # sizes its GPU-tier pools from the whole quota at construction
    # (cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.cpp:269-277)
    # and this collector
    # builds and tears one down per case. Cap the quota at what this batch
    # can touch via kv_cache_config.max_tokens — the manager converts it with
    # its own DSV4 arithmetic (DeepseekV4CacheManager._get_quota_from_max_tokens,
    # sparse/deepseek_v4/cache_manager.py:951-1006: non-sliding KV +
    # SWA per token/request + indexer/compress pools + per-type padding) and
    # takes min(max_gpu_total_bytes, quota_from_max_tokens) (pyexecutor/kv_cache/
    # kv_cache_manager_v2.py:1651-1670). The envelope is doubled and rounded
    # to whole pages so dummy-request allocation never sits at the edge; a
    # pool that is still too small fails allocation loudly (classified),
    # never silently. The free-memory half stays as the upper bound, matching
    # the DSV4 example's free_gpu_memory_fraction=0.5.
    planned_tokens = batch_size * (max(max_seq, request_tokens) + tokens_per_block)
    planned_tokens = -(-2 * planned_tokens // tokens_per_block) * tokens_per_block
    # Floor: the pool must stay large enough that inductor emits the SAME
    # kernels serving runs. footer_scale_kv.dequant_gather (fp8_ds_mla path,
    # @maybe_compile(dynamic=True)) indexes the whole footer-scale pool; when
    # that pool extent is < 2**31 bytes inductor generates int32-indexed
    # triton kernels (xnumel i32), above it int64 — serving pools are tens of
    # GiB, so serving always runs the int64 variant. H20 calibration
    # quota from 1,048,576
    # tokens = 11.4 GiB -> i32 (53.9 ms), 1,572,864 tokens = 17.1 GiB -> i64
    # (45.8 ms); 2M tokens (22.7 GiB on Flash, i64 verified in the same sweep)
    # sits 1.33x past the threshold, and every other DSV4 geometry carries more
    # footer bytes per planned token than Flash, so the margin only grows. The
    # footer row format is fixed (584 B/token), so the threshold is in tokens,
    # not model geometry. 4M was tried first: on DeepSeek-V4-Pro it reproduced
    # the campaign's free*0.5 pool size and left 20 GiB for the 262k-295k token
    # context cells, whose forward allocates a 32 GiB intermediate -> OOM.
    planned_tokens = max(planned_tokens, _KV_POOL_MIN_TOKENS)
    kv_cache_kwargs = dict(
        tokens_per_block=tokens_per_block,
        max_gpu_total_bytes=int(free_bytes * 0.5),
        max_tokens=int(planned_tokens),
        enable_block_reuse=False,
        # Deviation (every SM; also collect_mla_module.py and collect_msa_module.py): the
        # manager is built without a host tier. Serving leaves host_cache_size unset, and
        # KVCacheManagerV2 then adds a host tier sized from the GPU quota
        # (pyexecutor/kv_cache/kv_cache_manager_v2.py:1718-1764@v1.4.0rc2) that a background
        # thread prefaults and registers with CUDA (kv_cache_manager_v2/storage/core.cpp:392
        # waitForFill=false, utils/hostMem.cpp:187-216). That tier holds offloaded blocks; the
        # attention kernels timed here read the GPU pool. These collectors build a manager per
        # point, so the fill ran while the point was timed: it stalled the timed loop at random
        # (B200 and B300 rows up to 2,700x their neighbours) and cost about 17 s per point here.
        # Without the tier the latencies are the same (B200 CSA context, 1,235 points: median
        # 0.99x, 1st to 99th percentile 0.96-1.04x) and nine rows take 17 minutes instead of 6
        # hours. Retire it when a row reuses one manager for all its points.
        host_cache_size=0,
    )
    if get_sm_version() == 90:
        # On Hopper DeepseekV4CacheManager refuses any pool but the packed FP8
        # sparse-MLA cache (attention/backends/sparse/deepseek_v4/cache_manager.py:368-374
        # "DeepSeek-V4 on Hopper requires kv_cache_config.dtype='fp8_ds_mla'"), and the
        # model's serving default sets that dtype there (models/modeling_deepseekv4.py:
        # 2920-2921; the option llmapi/llm_args.py:4985-4989 documents for SM90/SM120/SM121).
        kv_cache_config = KvCacheConfig(dtype="fp8_ds_mla", **kv_cache_kwargs)
    else:
        kv_cache_config = KvCacheConfig(**kv_cache_kwargs)
    kv_cache_manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
    # fp8 KV rows -> DataType.FP8, matching serving's kv_cache_dtype
    # resolution from kv_cache_quant_algo (set by _apply_gemm_type_quant).
    kv_cache_dtype = DataType.BF16 if kv_cache_dtype_str == "bfloat16" else DataType.FP8

    # Serving construction site: pyexecutor/_util.py:3575-3609
    # (is_mla branch): CacheType.SELFKONLY :3585, num_kv_heads=1 :3587,
    # head_dim=kv_lora_rank+qk_rope_head_dim :3588-3591, dtype=kv_cache_dtype :3596,
    # sparse_attention_config :3603, pretrained_config :3604, layer_mask :3607
    # (single-layer here). vocab_size only keys multimodal block-reuse events;
    # serving passes None for a text-only engine (:3442-3449), and no kernel reads it.
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
        vocab_size=config.vocab_size,
        layer_mask=[True],
        sparse_attention_config=model_config.sparse_attention_config,
        pretrained_config=config,
        model_config=model_config,
    )

    # From here on the manager owns device memory: release it on ANY
    # failure before re-raising, so a failed case cannot leak a
    # batch x max_seq fp8 pool into the worker's next cases (which would
    # then OOM for the wrong reason and pollute the failure log).
    try:
        request_ids = list(range(batch_size))
        token_nums = [request_tokens] * batch_size
        # add_dummy_requests is serving's own warmup-request mechanism
        # (pyexecutor/engine/runners/decoder/dummy_requests.py:234-251 register
        # dummy requests the same way before capture/warmup); is_gen mirrors the
        # request phase (generation metadata below declares num_contexts=0
        # with cached KV). The V2 manager returns None, with nothing left allocated,
        # when the pool cannot hold the requests (pyexecutor/kv_cache/
        # kv_cache_manager_v2.py:5927-5931); their block tables would stay unset.
        if kv_cache_manager.add_dummy_requests(request_ids, token_nums, is_gen=not is_context) is None:
            raise RuntimeError("FIXME(kernel-limit): does not fit: the KV cache pool cannot hold the case's requests")

        attention_cls = get_attention_backend(
            model_config.attn_backend,
            model_config.sparse_attention_config,
        )

        sparse_metadata_params = model_config.sparse_attention_config.to_sparse_metadata_params(
            pretrained_config=config
        )

        # Constructor kwargs mirror build_attention_metadata
        # (pyexecutor/engine/metadata.py:48-90); per-batch fields mirror the
        # inputs.py population sites cited per field below.
        attn_metadata = attention_cls.Metadata(
            # max_num_requests/max_num_tokens are CAPACITY BOUNDS
            # (engine/metadata.py:78-80 passes engine-lifetime capacity).
            # This collector prepares exactly one batch, so the tight bound
            # (= this batch's request/token counts) satisfies the same
            # buffer-sizing contract; the parity test asserts
            # max_num_tokens >= sum(seq_lens).
            max_num_requests=batch_size,
            max_num_tokens=total_tokens,
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            # seq_lens: per-request current-step token counts — context
            # appends the fresh chunk length, decode appends 1; assigned as a
            # tensor at inputs.py:2031-2035.
            seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int32),
            # position_ids: interface dataclass default None
            # (attention/backends/interface.py:110) — serving never assigns it
            # on this metadata; positions travel as a model input (the tensor
            # built in the runner below, mirroring inputs.py:1001 / :1625).
            position_ids=None,
            # num_contexts = scheduled context-request count
            # (inputs.py:2056); ordinary decode batches carry 0.
            num_contexts=batch_size if is_context else 0,
            # num_cached_tokens_per_seq: context = begin_compute prefix
            # (inputs.py:1013-1014); decode = past_seen - compressed (:1663-1665,
            # compressed offset 0 for these single-layer dummy weights);
            # wrapped into KVCacheParams(use_cache=True, ...) exactly as
            # inputs.py:2088-2095 does.
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[kv_cache_len] * batch_size,
            ),
            # cross: interface default None (interface.py:145) — cross
            # metadata exists only for encoder-decoder attention.
            cross=None,
            request_ids=request_ids,
            # prompt_lens: context = chunk-local fresh length (inputs.py:1000,
            # 1012 prompt slicing; the SM100 cached-KV walker consumes chunk-local
            # semantics); decode = py_prompt_len, which for is_gen dummy
            # requests IS the past-KV length: add_dummy_requests sets
            # req.prompt_len = token_num - 1 (kv_cache_manager_v2.py:6052-6055)
            # and we register token_num = seq_len + 1, so py_prompt_len ==
            # seq_len == kv_cache_len, appended at inputs.py:1667.
            prompt_lens=[seq_len_q if is_context else kv_cache_len] * batch_size,
            # cached-KV context flag: is_mla AND cache_reuse|chunked_prefill
            # (engine/metadata.py:65-68) — true here exactly for
            # prefix-carrying context cases.
            enable_context_mla_with_cached_kv=bool(is_context and prefix_len > 0),
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=False,
                cache_reuse=bool(is_context and prefix_len > 0),
            ),
            # all_rank_num_tokens stays None: attention-DP only
            # (inputs.py:459-460); single-process collection.
            all_rank_num_tokens=None,
            # workspace: optional attention-kernel scratch
            # (attention/backends/trtllm.py:129, lazily allocated at :470-475); an
            # empty tensor is the pre-warmup serving state.
            workspace=torch.tensor([], device=device, dtype=torch.int8),
            # sparse_metadata_params: same to_sparse_metadata_params call
            # serving makes (engine/metadata.py:73-77).
            sparse_metadata_params=sparse_metadata_params,
        )

        if hasattr(attn_module, "indexer") and attn_module.indexer is not None:
            attn_metadata.indexer = attn_module.indexer

        attn_metadata.prepare()
        # CSA only (HCA builds no indexer): the cached history's index K, which the shared DSA
        # indexer path scores and selects from. The reason is at _write_indexer_history.
        if getattr(attn_module, "indexer", None) is not None:
            with torch.inference_mode():
                _write_indexer_history(attn_module.indexer, attn_metadata, device)
        return kv_cache_manager, attn_metadata, attention_cls
    except Exception:
        kv_cache_manager.shutdown()
        raise


# ═══════════════════════════════════════════════════════════════════════
# Inputs, forward and serving's warm-up autotuning
# ═══════════════════════════════════════════════════════════════════════

# Serving tunes every tunable op in one autotune() pass at warmup, forwarding the whole
# model on a context batch of curr_max_num_tokens tokens (max_num_tokens, 8192 by
# default, llmapi/llm_args.py:5932-5933, unless the KV pool holds fewer) and then on one
# one-token generation request (pyexecutor/engine/runners/decoder/warmup.py:1050-1068,
# 1098-1159). The DSV4 module has tunable ops beyond its Linear projections: on the SM100
# family o_a_proj runs the autotuned CuTe DSL FP8 BMM (attention/backends/sparse/
# deepseek_v4/module.py:301-322, custom_ops/cute_dsl_custom_ops.py:5559-5590) and the
# indexer may run the autotuned fused FP8 Q GEMM (sparse/deepseek_v4/indexer.py:133-169),
# so the module itself is forwarded under autotune() at both passes when it is built, each
# on its own KV pool and metadata. The tuned tactics stay in the worker's AutoTuner cache.
# A context point with more new tokens than the context pass is first forwarded under
# autotune() at its own count, which a deployment that runs it (max_num_tokens >= that
# count) tunes at warmup; collect_mla_module tunes its projections the same way.
_SERVING_AUTOTUNE_PASSES: tuple[tuple[bool, int], ...] = ((True, 8192), (False, 1))
# The largest new-token count each built attention module is tuned at.
_TUNED_NUM_TOKENS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _dsv4_inputs(model_config, batch_size: int, seq_len: int, prefix_len: int, is_context: bool, device: str):
    """(num_tokens, position_ids, hidden_states) of one uniform batch."""
    torch_device = torch.device(device)
    hidden_size = model_config.pretrained_config.hidden_size
    # int32 position_ids: serving populates them int32 and the DSV4 rope ops assert it
    # (cpp/tensorrt_llm/thop/mlaRopeInplaceOp.cpp:52, compressorOp.cpp:121
    # "position_ids must be int32").
    if is_context:
        num_tokens = seq_len * batch_size
        position_ids = (
            torch.arange(prefix_len, prefix_len + seq_len, device=torch_device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        num_tokens = batch_size
        # position == past-seen/cached count, NOT seq_len - 1: see
        # generation_request_geometry (ordinary decode, inputs.py:1616, 1625,
        # 1663-1665). Off-by-one here rotates rope one step early; it does not
        # change kernel shapes or measured cost.
        position_ids = torch.full(
            (batch_size,),
            generation_request_geometry(seq_len)["position"],
            device=torch_device,
            dtype=torch.int32,
        )
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=torch_device)
    return num_tokens, position_ids, hidden_states


def _forward_dsv4(attn_module, model_config, attn_metadata, position_ids, hidden_states) -> None:
    """One eager forward with the metadata published the way the model publishes it."""
    from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs

    with model_extra_attrs(model_config.extra_attrs):
        get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
        with torch.inference_mode():
            attn_module.forward(position_ids, hidden_states, attn_metadata)


def _autotune_dsv4_module(attn_module, model_config, passes, device: str, kv_cache_dtype: str) -> None:
    """Forward ``attn_module`` under autotune() on one request per (is_context, tokens)
    pass of ``passes``, as serving's warmup does (_SERVING_AUTOTUNE_PASSES)."""
    from tensorrt_llm._torch.autotuner import autotune

    try:
        from .collect_mla_module import _cleanup
    except ImportError:
        from collect_mla_module import _cleanup

    start = time.perf_counter()
    for is_context, num_tokens in passes:
        kv_cache_manager = None
        try:
            kv_cache_manager, attn_metadata, _ = create_dsv4_kv_cache_and_metadata(
                model_config=model_config,
                attn_module=attn_module,
                batch_size=1,
                seq_len=num_tokens,
                is_context=is_context,
                device=device,
                kv_cache_dtype_str=kv_cache_dtype,
            )
            _, position_ids, hidden_states = _dsv4_inputs(model_config, 1, num_tokens, 0, is_context, device)
            with autotune():
                _forward_dsv4(attn_module, model_config, attn_metadata, position_ids, hidden_states)
            torch.cuda.synchronize()
        finally:
            _cleanup(kv_cache_manager)
    _TUNED_NUM_TOKENS[attn_module] = max(
        [num_tokens for is_context, num_tokens in passes if is_context] + [_TUNED_NUM_TOKENS.get(attn_module, 0)]
    )
    print(f"  [DSV4 module autotune] passes={list(passes)}: {time.perf_counter() - start:.1f} s")


# ═══════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════


def run_dsv4_attn(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    prefix_len: int = 0,
    *,
    mode: str,
    perf_filename: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
    dry_run: bool = True,
):
    """Run one DeepSeek-V4 attention-module point.

    ``dry_run`` False skips the checking forward before timing: a series has
    already run it for an earlier point of this seq_len (module_series).
    """
    import tensorrt_llm
    import tensorrt_llm._torch.utils as _trtllm_utils
    from tensorrt_llm._torch.autotuner import autotune

    try:
        from .collect_mla_module import _cleanup
    except ImportError:
        from collect_mla_module import _cleanup

    if attn_kind not in ATTN_KIND_TO_COMPRESS_RATIO:
        raise ValueError(f"unsupported DSV4 attn_kind: {attn_kind}")
    if kv_cache_dtype not in ("fp8", "bfloat16"):
        raise ValueError(f"DSV4 module rows are fp8-KV or bfloat16-KV only (got {kv_cache_dtype!r})")
    if compute_dtype != "bfloat16":
        raise ValueError(f"DSV4 module rows are bfloat16-compute only (got {compute_dtype!r})")
    if gemm_type != "fp8_block":
        raise ValueError(f"DSV4 module rows are fp8_block-gemm only (got {gemm_type!r})")

    is_context = mode == "context"
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)

    attn_module, model_config, head_info = _cached_dsv4_attention_module(
        model_path, attn_kind, tp_size, device, kv_cache_dtype
    )

    # Ownership: the KV pool and the process-global extra-attrs slot are
    # released/restored on EVERY exit path (success, dry-run failure,
    # benchmark failure) — workers survive a failed case and run the next
    # one, so a leaked pool would turn later cases into bogus OOM records
    # and a stale attention_metadata weakref would outlive the case.
    saved_extra_attrs = getattr(_trtllm_utils._model_extra_attrs, "attrs", None)
    kv_cache_manager = None
    try:
        kv_cache_manager, attn_metadata, attention_cls = create_dsv4_kv_cache_and_metadata(
            model_config=model_config,
            attn_module=attn_module,
            batch_size=batch_size,
            seq_len=seq_len,
            is_context=is_context,
            prefix_len=prefix_len,
            device=device,
            kv_cache_dtype_str=kv_cache_dtype,
        )

        num_tokens, position_ids, hidden_states = _dsv4_inputs(
            model_config, batch_size, seq_len, prefix_len, is_context, device
        )

        # A point past the tuned token count runs its checking forward under
        # autotune() (_SERVING_AUTOTUNE_PASSES).
        tune = num_tokens > _TUNED_NUM_TOKENS.get(attn_module, 0)
        if dry_run or tune:
            try:
                with autotune() if tune else contextlib.nullcontext():
                    _forward_dsv4(attn_module, model_config, attn_metadata, position_ids, hidden_states)
            except Exception:
                print("  Dry run failed:")
                traceback.print_exc()
                raise  # the finally below releases the KV pool
            if tune:
                _TUNED_NUM_TOKENS[attn_module] = num_tokens

        _trtllm_utils._model_extra_attrs.attrs = model_config.extra_attrs
        _trtllm_utils._model_extra_attrs.attrs["attention_metadata"] = weakref.ref(attn_metadata)

        def kernel_func():
            attn_module.forward(position_ids, hidden_states, attn_metadata)

        with benchmark_with_power(
            device=torch_device,
            kernel_func=kernel_func,
            num_warmups=warming_up,
            # A dry run of this seq_len already executed this forward (this point or
            # an earlier one of the series); three eager passes cover lazily
            # initialised paths before capture, as in the MLA and DSA modules.
            num_eager_warmups=3,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=False,
        ) as results:
            pass

        latency = results["latency_ms"]

        if is_context:
            isl, step = seq_len, prefix_len
        else:
            isl, step = 1, generation_request_geometry(seq_len)["persisted_step"]

        log_perf(
            item_list=[
                {
                    "model": model_path,
                    "architecture": ARCHITECTURE,
                    "mla_dtype": compute_dtype,
                    "kv_cache_dtype": kv_cache_dtype,
                    "gemm_type": gemm_type,
                    # Rank-LOCAL heads + mandatory tp_size (unified convention; the
                    # SDK loader derives native = num_heads * tp_size and
                    # validates local semantics).
                    "num_heads": head_info["local_heads"],
                    "batch_size": batch_size,
                    "isl": isl,
                    "tp_size": tp_size,
                    "step": step,
                    "compress_ratio": ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                    "latency": f"{latency:.4f}",
                }
            ],
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name=f"dsv4_{attn_kind}_{mode}_module",
            # Ground truth: the attention backend class TRT-LLM's own selector
            # returned for this sparse config (get_attention_backend); the many
            # internal kernels (indexer deepgemm, sparse MLA, compressor) are
            # not observable from one label — see kernel_source_backends.yaml.
            kernel_source=attention_cls.__name__,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )

        print(
            f"  [dsv4_{attn_kind}_{mode}] b={batch_size} s={seq_len} prefix={prefix_len} "
            f"tp={tp_size} local_heads={head_info['local_heads']}: {latency:.4f} ms"
        )

        return latency
    finally:
        _trtllm_utils._model_extra_attrs.attrs = saved_extra_attrs
        # Shuts the KV pool down, empties the CUDA cache and collects garbage
        # periodically (a full collection per point cost 0.2-0.4 s).
        _cleanup(kv_cache_manager)


def run_dsv4_attn_worker(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    """collect.py worker: one task = one shape = one row.

    Mode is derived from the target perf filename (same convention as the
    vllm collector, collect_dsv4_attn.py:968).
    """
    run_dsv4_attn(
        seq_len=seq_len,
        batch_size=batch_size,
        tp_size=tp_size,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        model_path=model_path,
        attn_kind=attn_kind,
        prefix_len=prefix_len,
        mode=_worker_mode(perf_filename),
        perf_filename=perf_filename,
        device=device,
    )


def _worker_mode(perf_filename: str) -> str:
    return "context" if "context" in os.path.basename(perf_filename) else "generation"


def run_dsv4_attn_series(
    points,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    *,
    mode: str,
    perf_filename: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
) -> list[float]:
    """Measure the (seq_len, prefix_len) points of one DeepSeek-V4 module key in one task.

    Every point runs through run_dsv4_attn, so its rows are those of a one-point
    case: the module comes from the module cache, and the KV cache manager and
    metadata are built per point. The manager's max_seq_len sizes the
    compressed-index buffers of the HCA path, so a manager sized for the largest
    point would change what the smaller points measure (module_series).
    """
    try:
        from collector.trtllm.module_series import run_points
    except ModuleNotFoundError:  # run as a script from collector/trtllm
        from module_series import run_points

    def run_point(seq_len: int, prefix_len: int, dry_run: bool) -> float:
        return run_dsv4_attn(
            seq_len=seq_len,
            batch_size=batch_size,
            tp_size=tp_size,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype=compute_dtype,
            gemm_type=gemm_type,
            model_path=model_path,
            attn_kind=attn_kind,
            prefix_len=prefix_len,
            mode=mode,
            perf_filename=perf_filename,
            device=device,
            warming_up=warming_up,
            test_ite=test_ite,
            dry_run=dry_run,
        )

    label = f"dsv4_{attn_kind}_{mode} series b={batch_size} tp={tp_size} kv={kv_cache_dtype}"
    return run_points(points, run_point, label=label)


def run_dsv4_attn_series_worker(
    points,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    """Worker entry for one series task (op_plans._trtllm_dsv4_rows): run_dsv4_attn_worker's
    positional fields, with the list of (seq_len, prefix_len) points in place of seq_len and
    prefix_len."""
    run_dsv4_attn_series(
        points,
        batch_size,
        tp_size,
        kv_cache_dtype,
        compute_dtype,
        gemm_type,
        model_path,
        attn_kind,
        mode=_worker_mode(perf_filename),
        perf_filename=perf_filename,
        device=device,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect TRT-LLM DeepSeek-V4 CSA/HCA attention module latency.")
    parser.add_argument("--mode", choices=["context", "generation"], default="context")
    parser.add_argument("--attn-kind", choices=["csa", "hca"], default="csa")
    parser.add_argument("--model-path", default="sgl-project/DeepSeek-V4-Flash-FP8")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    perf_name = PerfFile[f"DSV4_{args.attn_kind.upper()}_{args.mode.upper()}_MODULE"].value
    perf_filename = os.path.join(args.output_path, perf_name) if args.output_path else perf_name
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    run_dsv4_attn(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        tp_size=args.tp_size,
        kv_cache_dtype="fp8",
        compute_dtype="bfloat16",
        gemm_type="fp8_block",
        model_path=args.model_path,
        attn_kind=args.attn_kind,
        prefix_len=args.prefix,
        mode=args.mode,
        perf_filename=perf_filename,
        device=args.device,
        warming_up=3,
        test_ite=3,
    )


if __name__ == "__main__":
    main()
