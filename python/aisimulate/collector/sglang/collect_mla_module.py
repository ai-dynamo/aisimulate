# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MLA Module Collector for SGLang — unified MLA and DSA benchmarking.

Profiles the complete attention module forward pass at the model-runner level,
using SGLang's own ServerArgs → ModelRunner → ForwardBatch pipeline with dummy
weights. Op names and data schema are aligned with vLLM and TRT-LLM
collect_mla_module.py so that perf_database queries work across frameworks.

Supported models, attention types, and micro-sweeps are defined in collector v2
YAML and loaded through collector.case_generator.

Usage:
    # DSA context phase (DeepSeek-V3.2 style)
    SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
        python collect_mla_module.py --mode context --attn-type dsa

    # MLA generation phase (DeepSeek-V3 style)
    SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
        python collect_mla_module.py --mode generation --attn-type mla
"""

# 0.5.21 (H20/sm90 collector port: op_smoke + path gates in the v0.5.21 image;
# findings hopper_sglang_collector_port_0514_to_0521_2026_10_01).
# One release per file; earlier code is `git log -- <this file>` away.
__compat__ = "sglang==0.5.23"

import argparse
import gc
import json
import os

# Collector only ever benches a single M per point; skip DeepGEMM's upfront
# full-M precompile sweep (compiles every M in 1..16384 x each GEMM = ~32818
# kernels, all but one unused). With this off, DeepGEMM JIT-compiles only the
# actual M on demand during the per-point warmup (absorbed, not in timing).
# setdefault so an explicit env can still override.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
from importlib.metadata import version as get_version

import torch
from collector.op_plans import plan, plan_cases
from collector.sglang.runtime_compat import resolved_arg  # 0.5.21: derived server args live behind the resolving view

try:
    from helper import (
        PhaseClock,
        benchmark_with_power,
        get_sm_version,
        log_perf,
        render_phase_totals,
        run_measurement_subprocess,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import (
        PhaseClock,
        benchmark_with_power,
        get_sm_version,
        log_perf,
        render_phase_totals,
        run_measurement_subprocess,
    )

try:
    from collector.case_generator import (
        get_mla_module_model_specs,
        get_mla_module_precision_specs,
        get_mla_module_sweep_spec,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_mla_module_model_specs, get_mla_module_precision_specs, get_mla_module_sweep_spec

try:
    from collector.sglang.runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from collector.sglang.runtime_limits import (
        kv_pool_page_size,
        request_slots,
        required_kv_alloc_tokens,
        required_kv_tokens,
        required_prefill_extend_tokens,
        sglang_dsa_mqa_logits_chunking_supported,
    )
    from collector.sglang.runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
    )
    from collector.sglang.runtime_limits import (
        temporarily_chunked_alloc_extend as _temporarily_chunked_alloc_extend,
    )
except ModuleNotFoundError:
    from runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from runtime_limits import (
        kv_pool_page_size,
        request_slots,
        required_kv_alloc_tokens,
        required_kv_tokens,
        required_prefill_extend_tokens,
        sglang_dsa_mqa_logits_chunking_supported,
    )
    from runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
    )
    from runtime_limits import (
        temporarily_chunked_alloc_extend as _temporarily_chunked_alloc_extend,
    )


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Perf-database-compatible dtype strings → SGLang ServerArgs kv_cache_dtype values.
# The perf DB uses enum names like "fp8"; SGLang uses "fp8_e4m3".
SGLANG_KV_DTYPE: dict[str, str] = {
    "bfloat16": "bfloat16",
    "fp8": "fp8_e4m3",
}
SGLANG_DSA_PAGE_SIZE = 64

# AIC's cached HuggingFace model configs — avoids HF downloads in CI.
_MODEL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "aisimulate_core",
    "model_configs",
)
_GLM5_DSA_ARCHITECTURE = "GlmMoeDsaForCausalLM"
_NATIVE_NVFP4_SMS = {100, 103, 120}


# Per-subprocess sum of the point phase times printed by run_mla_module.
_POINT_PHASE_TOTALS: dict[str, float] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _is_glm5_dsa_model(model_id: str) -> bool:
    return _module_model_architecture(model_id) == _GLM5_DSA_ARCHITECTURE


# Set by run_mla_module() at the start of each benchmark subprocess from its
# skip_indexer arg (threaded down from run_mla_module_worker, which receives it
# from the registry entry's run_kwargs). Process-local: each subprocess runs exactly one
# run_mla_module, so this is never shared across cases. Replaces the old
# AIC_DSA_SKIP_INDEXER env that previously crossed the subprocess boundary.
_SKIP_INDEXER_PASS = False


def _dsa_skip_indexer_enabled(attn_type: str, model_path: str) -> bool:
    """Whether THIS case collects the skip-indexer (reuse-layer) variant: a
    skip-indexer pass (run_mla_module(skip_indexer=True), recorded in the
    _SKIP_INDEXER_PASS process global) AND a GLM-5-family DSA model that actually
    shares its topk index across layers (index_topk_freq>1). For such a 'skip'
    layer the per-layer indexer (mqa logits + topk + index-K store) is patched
    out (skip_topk=True, reuse prev layer's indices), so the captured cost is the
    reuse-layer cost. Rows land in the MAIN dsa_*_module_perf.txt file, tagged
    by an op_name "_skip_indexer" suffix (not a separate file)."""
    return (
        _SKIP_INDEXER_PASS
        and attn_type == "dsa"
        and _is_glm5_dsa_model(model_path)
        and _model_shares_dsa_index(model_path)
    )


def _generation_cuda_graph_batch_size(model_runner, num_tokens: int) -> int | None:
    """Match SGLang decode CUDA graph coverage for the decode microbench.

    SGLang captures decode CUDA graphs for batch sizes up to
    server_args.cuda_graph_max_bs (or the resolved cuda_graph_bs list); mirror
    that coverage using sglang's own settings -- no AIC env override -- so the
    decode benchmark uses graph timing exactly where serve would.
    """
    decode_config = resolved_arg(model_runner.server_args, "cuda_graph_config").decode
    if decode_config.backend == "disabled":
        return None
    capture_bs = decode_config.bs
    if capture_bs:
        # DecodeCudaGraphRunner.can_run_graph/load_batch: padding selects the
        # smallest captured batch at least as large as the real request batch.
        if resolved_arg(model_runner.server_args, "disable_cuda_graph_padding", False):
            return int(num_tokens) if int(num_tokens) in capture_bs else None
        return next((int(batch) for batch in sorted(capture_bs) if int(batch) >= num_tokens > 0), None)
    max_bs = decode_config.max_bs or 256
    return int(num_tokens) if 0 < int(num_tokens) <= int(max_bs) else None


def _generation_cuda_graph_enabled_for_tokens(model_runner, num_tokens: int) -> bool:
    return _generation_cuda_graph_batch_size(model_runner, num_tokens) is not None


def _padded_dsa_decode_batch(model_runner, forward_batch, padded_batch_size: int):
    """Use serving's buffer registry to populate dummy graph slots, keeping the real row key.

    SGLang DecodeCudaGraphRunner.load_batch fills this registry before replay.
    Native seq-length sentinels and cache/request-slot padding are essential:
    padding with extra full-history requests would measure a different batch.
    """
    from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry

    real_batch_size = forward_batch.batch_size
    fill_value = model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
    registry = build_decode_registry(
        device=forward_batch.input_ids.device,
        max_bs=padded_batch_size,
        max_num_token=padded_batch_size,
        seq_len_fill_value=fill_value,
        cache_loc_dtype=forward_batch.out_cache_loc.dtype,
        register_global_num_tokens=False,
    )
    registry.fill_from(
        forward_batch,
        raw_bs=real_batch_size,
        padded_bs=padded_batch_size,
        raw_num_tokens=real_batch_size,
        padded_num_tokens=padded_batch_size,
    )
    padded = registry.extract_buffer(
        padded_bs=padded_batch_size,
        padded_num_tokens=padded_batch_size,
        forward_batch_template=forward_batch,
    )
    padded.seq_lens_sum = forward_batch.seq_lens_sum + (padded_batch_size - real_batch_size) * fill_value
    return padded


def _initialize_dsa_history(model_runner, attention_module, forward_batch, history_length: int) -> None:
    """Populate every referenced historical slot through SGLang's writers.

    The allocator only assigns locations; clearing it does not clear or fill
    either cache. Seed post-RoPE synthetic activations outside the measurement,
    including partial pages, rather than exposing zeros or an earlier case's
    contents. These inputs represent no particular checkpoint distribution.

    Native contracts at SGLang v0.5.23: dsa/dsa_indexer.py:1806-1885
    (Indexer._store_index_k_cache) owns index-K quantization and opaque layout
    (its pool comes from ForwardContext, :1827); dsa_backend.py:3879 writes
    latent/rotary KV through set_mla_kv_buffer with the attention layer;
    memory_pool.py:5026-5051 and :5279+ (DSATokenToKVPool) own BF16/FP8 MLA
    layouts and their scale fields. Paths above are relative to
    python/sglang/srt/layers/attention except memory_pool.py, which is in
    python/sglang/srt/mem_cache.
    """
    if history_length < 0:
        raise ValueError("DSA historical length must be nonnegative")
    if not history_length:
        return

    # The act_quant serving hands the index-K writer on CUDA
    # (dsa/dsa_indexer.py:1936-1938, kernels/ops/quantization/act_quant_triton.py:88 @v0.5.23).
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant

    forward_context_type, forward_context = _import_sglang_forward_context()
    pool = model_runner.token_to_kv_pool
    locations = (
        model_runner.req_to_token_pool.req_to_token[forward_batch.req_pool_indices.long(), :history_length]
        .flatten()
        .long()
        .contiguous()
    )
    expected = int(forward_batch.batch_size) * history_length
    if locations.numel() != expected:
        raise ValueError(f"DSA history has {locations.numel()} locations; expected {expected}")
    generator = torch.Generator(device="cpu").manual_seed(0)
    for start in range(0, expected, 8192):
        slots = locations[start : start + 8192]

        def random_activations(*shape):
            return torch.randn(shape, generator=generator, dtype=torch.float32).to(
                device=slots.device, dtype=torch.bfloat16
            )

        # Normalized latent K and post-RoPE K are inputs to the native cache
        # writer. Do not fill the packed cache bytes or scale fields directly.
        pool.set_mla_kv_buffer(
            attention_module.attn_mqa,
            slots,
            random_activations(slots.numel(), 1, attention_module.kv_lora_rank),
            random_activations(slots.numel(), 1, attention_module.qk_rope_head_dim),
        )
        # The native fused store obtains its pool from ForwardContext, just
        # as it does during serving Indexer.forward (dsa_indexer.py:1827 @v0.5.23).
        with forward_context(forward_context_type(attn_backend=model_runner.attn_backend)):
            attention_module.indexer._store_index_k_cache(
                forward_batch,
                attention_module.layer_id,
                random_activations(slots.numel(), pool.index_head_dim),
                act_quant=act_quant,
                out_cache_loc=slots,
            )


def _dsa_forward_input_relay(model_runner, batch):
    """Own the native scheduler relay needed by SGLang input staging.

    At v0.5.23 prepare_for_extend leaves input_ids None and stages
    prefill_input_ids_cpu (schedule_batch.py:3069-3070); resolve_forward_inputs
    materializes them (managers/overlap_utils.py:113-153), and the scheduler runs
    it and FutureMap.stash even without overlap (scheduler.py:3858-3860,
    4681-4916). Attention-only execution skips embeddings, but ForwardBatch still
    needs these IDs to derive the actual token count (forward_batch_info.py:1364-1382).
    The relay is built as the scheduler builds it (scheduler.py:1667-1686).
    """
    from sglang.srt.managers.overlap_utils import decide_needs_confidence_relay, decide_needs_cpu_seq_lens

    if not hasattr(model_runner, "_aic_dsa_input_relay"):
        model_runner._aic_dsa_input_relay = batch.spec_algorithm.create_future_map(
            batch.device,
            model_runner.req_to_token_pool,
            needs_cpu_seq_lens=decide_needs_cpu_seq_lens((model_runner.attn_backend,)),
            needs_confidence_relay=decide_needs_confidence_relay(),
        )
    return model_runner._aic_dsa_input_relay


def _radix_disabled_tree_cache(model_runner):
    """The prefix cache serving runs with --disable-radix-cache (load_model_runner sets it).

    v0.5.23 deleted ChunkCache (sglang #43201). A disabled cache on a model with
    full attention only is a UnifiedRadixCache with ``disable=True`` and the FULL
    component (mem_cache/registry.py:91-99 -> create_unified_radix_cache :149-191;
    CacheInitParams from mem_cache/kv_cache_builder.py:389-402 @v0.5.23, page_size
    the allocator's without DCP). It matches and inserts nothing, and evicts nothing
    (unified_radix_cache.py:619-665).
    """
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return UnifiedRadixCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            page_size=model_runner.token_to_kv_pool_allocator.page_size,
            tree_components=(ComponentType.FULL,),
        )
    )


def _new_schedule_batch(model_runner, reqs, tree_cache):
    """A non-overlap, non-speculative batch, as the scheduler builds one (ScheduleBatch.init_new)."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def _validate_dsa_forward_tokens(forward_batch, expected_tokens: int) -> None:
    """Reject missing scheduler inputs before they alter native dispatch."""
    for name in ("input_ids", "positions", "out_cache_loc"):
        value = getattr(forward_batch, name)
        if value is None or value.numel() != expected_tokens:
            actual = None if value is None else value.numel()
            raise ValueError(f"DSA {name} has {actual} tokens; expected {expected_tokens}")
    # The host token count (forward_batch_info.py:707, set at :1382 @v0.5.23)
    if forward_batch.global_num_token_non_padded_cpu != expected_tokens:
        raise ValueError(
            f"DSA global_num_token_non_padded_cpu={forward_batch.global_num_token_non_padded_cpu}; "
            f"expected {expected_tokens}"
        )


def _resolve_local_model_path(model_id: str) -> str:
    """Resolve a HuggingFace model ID to a local config directory.

    Uses AIC's cached model configs from src/aisimulate_core/model_configs/
    so that the collection pipeline never needs HuggingFace network access.
    The function patches model_type and architectures for sglang AutoConfig
    compatibility and strips auto_map to prevent any remote code download.

    SGLang's AutoConfig.from_pretrained() doesn't recognize "deepseek_v32" or
    "glm_moe_dsa" model types.  The workaround (matching sglang's own
    _load_deepseek_v32_model approach) is to present these as "deepseek_v3".
    DSA-specific fields (index_topk, index_head_dim, etc.) are preserved in
    the config and sglang uses those for DSA detection via is_deepseek_nsa().

    Falls back to the original HF model ID if local config is not found.
    """
    config_file = os.path.join(_MODEL_CONFIG_DIR, f"{model_id.replace('/', '--')}_config.json")
    if not os.path.exists(config_file):
        return model_id

    with open(config_file) as f:
        config = json.load(f)

    # Strip auto_map to prevent transformers from attempting to download
    # custom config/model classes from HuggingFace when trust_remote_code=True.
    config.pop("auto_map", None)

    # GLM-5.2's cached checkpoint config uses the pre-Transformers-v5 name
    # ``deepseek_sparse_attention``.  The Transformers build shipped in the
    # pinned SGLang 0.5.14 image validates layer_types before SGLang can apply
    # its GLM DSA compatibility path and accepts the canonical CSA name only.
    # This field is descriptive for GlmMoeDsaForCausalLM; DSA dispatch itself
    # is selected from architecture + index_topk, which remain unchanged.
    if isinstance(config.get("layer_types"), list):
        config["layer_types"] = [
            "compressed_sparse_attention" if layer_type == "deepseek_sparse_attention" else layer_type
            for layer_type in config["layer_types"]
        ]

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_sglang_config_{model_id.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)

    quant_config_file = os.path.join(
        _MODEL_CONFIG_DIR,
        f"{model_id.replace('/', '--')}_hf_quant_config.json",
    )
    if os.path.exists(quant_config_file):
        shutil.copyfile(quant_config_file, os.path.join(tmp_dir, "hf_quant_config.json"))

    return tmp_dir


# ═══════════════════════════════════════════════════════════════════════
# Precision Combos
# ═══════════════════════════════════════════════════════════════════════


def _get_precision_combos(phase: str):
    """Return YAML-backed operator precision triples for one phase and SM."""
    return [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs(
            "sglang",
            phase=phase,
            sm_version=get_sm_version(),
        )
    ]


def _get_backends(attn_type: str):
    """Return the attention backend string to use for a given attention type.

    For DSA: returns SGLang's "dsa" attention backend (sub-backends resolve in load_model_runner).
    For MLA: returns the best available backend based on SM version.
    Aligns with sglang's _get_default_attn_backend() in server_args.py.
    """
    sm = get_sm_version()
    if attn_type == "dsa":
        return "dsa"
    else:
        if sm >= 100:
            return "trtllm_mla"
        elif sm >= 90:
            return "fa3"
        else:
            # sglang defaults MLA to "triton" on SM < 90; flashinfer MLA
            # (BatchMLAPagedAttentionWrapper) is not validated on these GPUs.
            return "triton"


def _get_mla_backend_list() -> list[str]:
    """Return all MLA backends to sweep for wideep MLA collection.

    Per-architecture backends:
      SM >= 100: ["trtllm_mla"]  — flashinfer MLA is not supported on Blackwell;
                  sglang auto-promotes to trtllm_mla and then fails kv_cache_dtype
                  validation.  Existing B200 perf data contains only trtllm_mla.
      SM >= 90:  ["flashinfer", "fa3"]
      SM < 90:   []  — sglang defaults MLA to "triton" on SM < 90 (A100 etc.);
                  flashinfer MLA kernels (BatchMLAPagedAttentionWrapper) are not
                  validated on SM < 90 and crash the subprocess.  Skip wideep MLA
                  collection on these GPUs; kernel-level collectors (collect_mla.py)
                  already capture per-kernel latency via mocks.
    """
    sm = get_sm_version()
    if sm >= 100:
        return ["trtllm_mla"]
    elif sm >= 90:
        return ["flashinfer", "fa3"]
    else:
        return []


# ═══════════════════════════════════════════════════════════════════════
# Test Case Generation
# ═══════════════════════════════════════════════════════════════════════


def _module_model_architecture(model_path: str) -> str:
    """Return the YAML-declared architecture for a module benchmark model."""
    for spec in get_mla_module_model_specs(apply_model_filter=False):
        if spec.model_path == model_path:
            return spec.architecture
    return "unknown"


def _module_model_native_heads(model_path: str) -> int:
    """Return native attention heads for TP-sim filtering."""
    for spec in get_mla_module_model_specs(apply_model_filter=False):
        if spec.model_path == model_path:
            return spec.native_num_heads
    return 128


def get_context_test_cases(attn_type: str):
    """Context-phase test cases.

    Returns list of [seq_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type].
    """
    sweep = get_mla_module_sweep_spec("sglang")
    cases = []
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("context"):
        for num_heads in sweep.inner_sweep_head_counts:
            for batch_size in sweep.context_batch_sizes:
                for seq_len in sweep.context_sequence_lengths:
                    if batch_size * seq_len > sweep.context_max_tokens:
                        continue
                    if (
                        seq_len >= sweep.context_large_sequence_min
                        and batch_size > sweep.context_large_sequence_max_batch_size
                    ):
                        continue
                    cases.append([seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type])
    return cases


def get_generation_test_cases(attn_type: str):
    """Generation-phase test cases.

    Returns list of [kv_cache_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type].
    """
    sweep = get_mla_module_sweep_spec("sglang")
    cases = []
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("generation"):
        for num_heads in sweep.inner_sweep_head_counts:
            for batch_size in sweep.generation_batch_sizes:
                for seq_len in sweep.generation_sequence_lengths:
                    if batch_size * seq_len > sweep.generation_max_tokens:
                        continue
                    if (
                        seq_len >= sweep.generation_large_sequence_min
                        and batch_size > sweep.generation_large_sequence_max_batch_size
                    ):
                        continue
                    cases.append([seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type])
    return cases


def _model_max_position_embeddings(model_id: str) -> int | None:
    """Return the model's max context length (RoPE table size) from its config.

    This is exactly the value SGLang uses to size the DSA/NSA indexer's rotary
    cos/sin cache (``nsa_indexer`` builds it with
    ``max_position=max_position_embeddings``) and to admit requests
    (``ServerArgs.context_length`` defaults to it).  Read from AIC's cached
    model config — the same file :func:`_resolve_local_model_path` loads — so
    no HuggingFace network access is needed.  Returns ``None`` if unavailable
    (caller then applies no context cap).
    """
    config_file = os.path.join(_MODEL_CONFIG_DIR, f"{model_id.replace('/', '--')}_config.json")
    if not os.path.exists(config_file):
        return None
    try:
        with open(config_file) as f:
            value = json.load(f).get("max_position_embeddings")
        return int(value) if value else None
    except Exception:
        return None


def _model_dsa_index_topk_freq(model_id: str) -> int:
    """Return the model's ``index_topk_freq`` (how many consecutive layers share
    one DSA topk index), or 1 if absent.

    GLM-5.2 sets ``index_topk_freq=4`` -> 1 layer computes the indexer, 3 reuse
    it. GLM-5 / DSV3.2 omit it (==1, every layer computes its own index).
    """
    config_file = os.path.join(_MODEL_CONFIG_DIR, f"{model_id.replace('/', '--')}_config.json")
    if not os.path.exists(config_file):
        return 1
    try:
        with open(config_file) as f:
            value = json.load(f).get("index_topk_freq")
        return int(value) if value else 1
    except Exception:
        return 1


def _model_shares_dsa_index(model_id: str) -> bool:
    """Whether the model reuses a shared DSA topk index across layers
    (``index_topk_freq > 1``). Only such models have a distinct skip-indexer
    per-layer cost worth collecting separately."""
    return _model_dsa_index_topk_freq(model_id) > 1


def _model_native_gemm_quant(model_id: str) -> str | None:
    """Return the checkpoint loader's native weight quantization, or ``None``.

    Different GLM-5 / DeepSeek checkpoints ship at different precisions —
    ``nvidia/GLM-5-NVFP4`` (ModelOpt NVFP4), ``zai-org/GLM-5-FP8`` (block-fp8),
    ``zai-org/GLM-5`` (bf16) — and they are *separate models*, not one model
    reinterpreted at several precisions. This value therefore selects SGLang's
    checkpoint loader; it is not automatically the persisted module
    ``gemm_type``. In particular, the NVFP4 checkpoint's timed DSA projections
    remain BF16 and are logged as ``bfloat16``.

    Reads the same AIC cached config files :func:`_resolve_local_model_path`
    loads:

    * ModelOpt ``*_hf_quant_config.json`` → ``quantization.quant_algo``
      (``NVFP4`` → ``"nvfp4"``, ``FP8`` → ``"fp8_block"``)
    * HF ``*_config.json`` ``quantization_config`` → ``quant_method`` +
      ``weight_block_size`` (block-fp8 → ``"fp8_block"``)
    * ModelOpt ``MIXED_PRECISION`` export (one lane per module in
      ``quantized_layers``), checked after the two above → ``"nvfp4"`` when a
      lane is NVFP4 or W4A16_NVFP4

    Returns one of ``"nvfp4"`` / ``"fp8_block"`` / ``None``. Raises
    ``ValueError`` for a ``MIXED_PRECISION`` export without an NVFP4 lane:
    SGLang resolves it to ``w4afp8``, which this collector has no path for.
    It also raises when a ``self_attn`` lane is listed: the ``bfloat16`` DSA label assumes BF16 attention.
    For the mixed recipe a ``compression_config`` in ``*_config.json`` takes precedence over the sidecar, as
    in SGLang, and a ``quantization_config``, ``compression_config`` or ``quantization`` section that is not
    an object raises.
    """
    base = os.path.join(_MODEL_CONFIG_DIR, model_id.replace("/", "--"))
    quant_file = f"{base}_hf_quant_config.json"
    if os.path.exists(quant_file):
        try:
            with open(quant_file) as f:
                algo = (json.load(f).get("quantization") or {}).get("quant_algo")
            if algo:
                algo = str(algo).upper()
                if "FP4" in algo:
                    return "nvfp4"
                if "FP8" in algo:
                    return "fp8_block"
        except Exception:
            pass
    config_file = f"{base}_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                quant_cfg = json.load(f).get("quantization_config") or {}
            method = str(quant_cfg.get("quant_method", "")).lower()
            if method == "fp8" and quant_cfg.get("weight_block_size"):
                return "fp8_block"
            if "fp4" in method:
                return "nvfp4"
        except Exception:
            pass
    # ModelOpt MIXED_PRECISION export: one quantization lane per module in ``quantized_layers``.
    # Citations are file:line@release-tag under python/sglang/srt/. SGLang gates the ``config.json``
    # ``quantization_config`` on ``quant_method`` (configs/model_config.py:1034-1044), reads
    # ``compression_config`` next (:1044-1046), and parses ``hf_quant_config.json`` as ModelOpt unconditionally
    # (:1117-1123). The gate is applied to both sections here because ModelOpt writes no other ``quant_method``
    # into that file.
    # * 0.5.14, the pin: every MIXED_PRECISION export except NemotronH parses as ``w4afp8``
    #   (configs/model_config.py:1145-1151). ``modelopt_fp4`` is then rejected inside ServerArgs, so a
    #   targeted case fails loudly: the compatibility map allows only ``modelopt`` and the check raises
    #   (configs/model_config.py:1299-1302, 1346-1375, reached from server_args.py:2533, 3497).
    #   ``None`` would run W4AFp8Config, whose Fp8LinearMethod makes the BF16 attention projections block-FP8
    #   (layers/quantization/w4afp8.py:99-104, layers/quantization/fp8.py:354-355), and those rows
    #   would be labelled ``bfloat16``.
    # * From v0.5.15: an export with an NVFP4 or W4A16_NVFP4 lane parses as ``modelopt_mixed``
    #   (configs/model_config.py:1153-1163) and an explicit ``modelopt_fp4`` is overridden to it
    #   (layers/quantization/modelopt_quant.py:649-654, configs/model_config.py:1466-1475).
    # So SGLang's lane test decides here: an NVFP4 lane returns "nvfp4" (the SM guard and the ``modelopt_fp4``
    # loader mapping apply). Without one SGLang resolves the export to ``w4afp8`` (NemotronH aside at 0.5.14), a
    # path this collector lacks, so the helper raises.
    # Past 0.5.14 SGLang stops rejecting ``modelopt_fp4`` and
    # overrides it to ``modelopt_mixed``, so this path goes from a loud failure to live data: re-derive the
    # ``bfloat16`` label premise (no attention lane), the ``ModelOptMixedPrecisionConfig`` methods for unlisted
    # modules and the KV cache, the section precedence including ``compression_config``, the ``quant_method`` gate,
    # the exact match and the lane tuple, the NemotronH exception and the ``w4afp8`` fallback.
    # A parse error of an existing file propagates and a section that is not an object raises: ``None`` would
    # become the ``bfloat16`` DSA label.
    quant_cfg = None
    compression_cfg = None
    section = "config.json quantization_config"
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
        quant_cfg = config.get("quantization_config")
        compression_cfg = config.get("compression_config")
    if quant_cfg is not None and not isinstance(quant_cfg, dict):
        raise ValueError(f"{model_id}: {section} must be an object, got {type(quant_cfg).__name__}")
    if compression_cfg is not None and not isinstance(compression_cfg, dict):
        raise ValueError(
            f"{model_id}: config.json compression_config must be an object, got {type(compression_cfg).__name__}"
        )
    # SGLang resolves ``quantization_config``, then ``compression_config``, then the sidecar
    # (configs/model_config.py:1029, 1046 and 1117): a checkpoint that declares
    # ``compression_config`` reaches the mixed recipe only through its own ``quantization_config``,
    # never through the sidecar.
    if quant_cfg is None and compression_cfg is None and os.path.exists(quant_file):
        section = "hf_quant_config.json quantization"
        with open(quant_file) as f:
            quant_cfg = json.load(f).get("quantization")
        if quant_cfg is not None and not isinstance(quant_cfg, dict):
            raise ValueError(f"{model_id}: {section} must be an object, got {type(quant_cfg).__name__}")
    if (
        isinstance(quant_cfg, dict)
        and quant_cfg.get("quant_method", "modelopt") == "modelopt"
        and quant_cfg.get("quant_algo") == "MIXED_PRECISION"
    ):
        lanes_map = quant_cfg.get("quantized_layers")
        if lanes_map is None:
            lanes_map = {}
        elif not isinstance(lanes_map, dict):
            raise ValueError(
                f"{model_id}: {section} quantized_layers must be an object, got {type(lanes_map).__name__}"
            )
        attention_lanes = sorted(
            name for name, lane in lanes_map.items() if isinstance(lane, dict) and "self_attn" in str(name)
        )
        if attention_lanes:
            raise ValueError(
                f"{model_id} is a ModelOpt MIXED_PRECISION export that quantizes attention modules "
                f"({attention_lanes[:3]}): the DSA module label assumes BF16 attention projections "
                "and cannot describe it"
            )
        has_fp4_lane = any(
            str(lane.get("quant_algo", "")).upper() in {"NVFP4", "W4A16_NVFP4"}
            for lane in lanes_map.values()
            if isinstance(lane, dict)
        )
        if has_fp4_lane:
            return "nvfp4"
        raise ValueError(
            f"{model_id} is a ModelOpt MIXED_PRECISION export with no NVFP4 lane in its {section}: "
            "SGLang resolves such an export to w4afp8, which this collector has no path for"
        )
    return None


def _build_wideep_mla_test_cases(mode: str):
    """Build test cases for wideep MLA collection (backward-compatible).

    Output format: [seq_len, batch_size, num_heads, kv_cache_dtype,
                    compute_dtype, gemm_type, model_path,
                    attn_type, attention_backend]

    Matches the old collect_wideep_attn.py behavior:
    - Single precision combo (bfloat16 run, logged as fp8_block/fp8)
    - Sweeps multiple attention backends per SM version
    - Only DeepSeek-V3 (the MLA model), not V3.2/GLM-5 (DSA models)

    perf_filename is supplied by collect.py via functools.partial as a keyword
    argument, so it is not included in the test case tuple.
    """
    if _skip_sm120_deepgemm_attention_modules():
        return []
    model_specs = get_mla_module_model_specs(attention_type="mla", wideep_mla=True)
    if get_sm_version() not in _NATIVE_NVFP4_SMS:
        model_specs = [spec for spec in model_specs if _model_native_gemm_quant(spec.model_path) != "nvfp4"]
    sweep = get_mla_module_sweep_spec("sglang")
    backends = _get_mla_backend_list()
    cases = []
    seen_consumer_keys = set()
    for model_spec in model_specs:
        for backend in backends:
            for num_heads in sweep.inner_sweep_head_counts:
                if num_heads > model_spec.native_num_heads:
                    continue
                # WideEP MLA consumers do not key on model/checkpoint path.
                # Full collection keeps the first V3-family representative;
                # an explicit COLLECTOR_MODEL_PATH is filtered before here.
                consumer_key = (backend, num_heads)
                if consumer_key in seen_consumer_keys:
                    continue
                seen_consumer_keys.add(consumer_key)
                # Single precision: run with bfloat16, log as fp8_block/fp8
                cases.append(
                    [
                        0,
                        0,
                        num_heads,
                        "bfloat16",
                        "bfloat16",
                        "bfloat16",
                        model_spec.model_path,
                        "mla",
                        backend,
                    ]
                )
    return cases


def _skip_sm120_deepgemm_attention_modules() -> bool:
    """Return True when SGLang's DeepGEMM-backed module path is unsupported.

    This skip originated from an SM120 SGLang 0.5.10 hardware run in
    ``37826f10``. Forcing these module collectors produced no usable rows:
    - WideEP MLA context/generation subprocesses failed after every shape was
      skipped, ending with "MLA module ... produced no perf rows".
    - DSA context/generation failed the same way, and the underlying DeepGEMM
      calls reported unsupported Blackwell paths, e.g.
      attention.hpp:136 and gemm.hpp:376 "Unsupported architecture".

    The exact 0.5.14 image is source-derived only here: SGLang forces
    TRTLLM-GEN for DSA while the bundled capability check accepts exact SM100,
    not SM120, and the bundled FlashMLA binary has no SM120 target. Keep the
    skip until 0.5.14 is hardware-validated on SM120. Developers can set
    COLLECTOR_FORCE_DEEPGEMM_ATTENTION_MODULES=1 to repro or validate it.
    """
    return os.environ.get("COLLECTOR_FORCE_DEEPGEMM_ATTENTION_MODULES") != "1" and get_sm_version() >= 120


def get_wideep_mla_context_test_cases():
    """collect.py entrypoint for wideep MLA context collection."""
    return _build_wideep_mla_test_cases(mode="context")


def get_wideep_mla_generation_test_cases():
    """collect.py entrypoint for wideep MLA generation collection."""
    return _build_wideep_mla_test_cases(mode="generation")


def get_dsa_context_module_test_cases():
    return plan_cases("dsa_context_module", "sglang", get_sm_version())


def get_dsa_generation_module_test_cases():
    return plan_cases("dsa_generation_module", "sglang", get_sm_version())


def get_dsa_context_module_skip_indexer_test_cases():
    return plan_cases("dsa_context_module_skip_indexer", "sglang", get_sm_version())


def get_dsa_generation_module_skip_indexer_test_cases():
    return plan_cases("dsa_generation_module_skip_indexer", "sglang", get_sm_version())


# ═══════════════════════════════════════════════════════════════════════
# SGLang Helpers
# ═══════════════════════════════════════════════════════════════════════


def cleanup_distributed():
    """Clean up SGLang distributed environment if it exists."""
    import sglang.srt.distributed.parallel_state as parallel_state

    try:
        parallel_state.destroy_model_parallel()
    except Exception:
        pass
    for var_name in [
        "_TP",
        "_PP",
        "_MOE_EP",
        "_MOE_TP",
        "_MOE_DP",
        "_WORLD",
        "_PDMUX_PREFILL_TP_GROUP",
        "_ATTN_CP",
        "_ATTN_TP",
        "_ATTN_DP",
    ]:
        if hasattr(parallel_state, var_name):
            setattr(parallel_state, var_name, None)

    from collector.sglang.runtime_compat import reset_expert_location_metadata

    reset_expert_location_metadata()


def _ensure_fp8_block_quant_config(hf_cfg) -> None:
    """Populate hf_config.quantization_config with weight_block_size for fp8_block.

    After _resolve_local_model_path rewrites the model_type to deepseek_v3, the
    JSON's ``quantization_config`` section may not be preserved as an attribute
    on the HF config object. sglang's _get_quantization_config then falls back
    to ``Fp8Config()`` with no weight_block_size, which flips Fp8LinearMethod
    to the channel-FP8 path — that path transposes the weight post-load and
    breaks downstream DeepseekV2 post_load_weights. Re-inject the block-scale
    fields so block_quant=True fires.
    """
    default_qc = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "fmt": "e4m3",
        "quant_algo": "FP8",
    }
    qc = getattr(hf_cfg, "quantization_config", None)
    if qc is None:
        hf_cfg.quantization_config = default_qc
        return
    if isinstance(qc, dict):
        for k, v in default_qc.items():
            if qc.get(k) is None:
                qc[k] = v
        return
    for k, v in default_qc.items():
        if getattr(qc, k, None) is None:
            setattr(qc, k, v)


def _patch_nsa_rope_contiguity(model_runner):
    """Workaround for sglang rope contiguity bugs on Blackwell (SM>=100).

    The NSA Indexer's _get_k_bf16 calls torch.split on q/k, producing
    non-contiguous views.  The JIT RoPE kernel rejects these:
      - <=0.5.9  rope.cuh:498 check_cuda_contiguous  (q/k assertion)
      - 0.5.14  rope.cuh:299 TensorMatcher verify    (positions stride check)

    We monkey-patch rotary_emb.forward on each NSA Indexer layer to call
    .contiguous() on positions, q, and k before the kernel.
    """
    if get_sm_version() < 100:
        return

    for layer in model_runner.model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            continue
        # MultiPlatformOp wraps the actual module
        actual = getattr(indexer, "_module", indexer)
        rotary_emb = getattr(actual, "rotary_emb", None)
        if rotary_emb is None:
            continue

        original_forward = rotary_emb.forward

        def _make_contiguous_forward(orig):
            def _forward(positions, query, key, fused_set_kv_buffer_arg=None):
                # by keyword: the 4th positional of forward_cuda is offsets
                # (rotary_embedding/base.py:380-387)
                kwargs = {} if fused_set_kv_buffer_arg is None else {"fused_set_kv_buffer_arg": fused_set_kv_buffer_arg}
                return orig(
                    positions.contiguous(),
                    query.contiguous(),
                    key.contiguous() if key is not None else key,
                    **kwargs,
                )

            return _forward

        rotary_emb.forward = _make_contiguous_forward(original_forward)
        print(f"Patched rope contiguity for layer {layer}")


def _patch_nsa_indexer_compile_for_module_cuda_graph(attention_module) -> None:
    """Use eager indexer gate helpers during module-level CUDA graph capture.

    SGLang's piecewise runner compiles/captures the whole model path.  The AIC
    module collector captures just ``self_attn.forward``.  In that narrower
    capture, the NSA indexer's small ``@torch.compile`` gate helpers can lazily
    invoke inductor inside ``torch.cuda.graph(...)`` and fail on CPU->CUDA
    copies.  Binding the original wrapped functions keeps the same math while
    avoiding compile work inside graph capture.
    """
    indexer = getattr(attention_module, "indexer", None)
    if indexer is None:
        return
    actual_indexer = getattr(indexer, "_module", indexer)
    for name in (
        "_project_and_scale_head_gates",
        "_get_logits_head_gate",
        "_apply_q_scale_and_softmax_scale",
    ):
        bound = getattr(actual_indexer, name, None)
        if bound is None:
            continue
        raw = getattr(bound, "__wrapped__", None)
        if raw is None:
            raw = getattr(getattr(type(actual_indexer), name, None), "__wrapped__", None)
        if raw is not None:
            setattr(actual_indexer, name, types.MethodType(raw, actual_indexer))


class CudaIllegalAccessError(RuntimeError):
    """Stop the current subprocess after a CUDA illegal access poisons context."""


class MeasurementBoundaryError(RuntimeError):
    """Native dispatch ran, but its module timing boundary is not qualified."""


class PerfLogWriteError(RuntimeError):
    """Fail the subprocess when a measured row was not durably persisted."""


def _expect_module_attr(module, attr_name: str, expected: int, module_name: str) -> None:
    value = getattr(module, attr_name, None)
    if value != expected:
        raise RuntimeError(f"{module_name}.{attr_name}={value}, expected {expected}")


def _validate_dsa_tp_module_shapes(model_runner, local_num_heads: int, target_tp_size: int) -> None:
    """Verify the single-GPU DSA module emulates one target TP rank.

    The collector cannot launch a full distributed TP group for every
    micro-benchmark.  Instead it loads a single local-rank attention module with
    local ``num_attention_heads``.  That must also localize the TP-sharded
    projection GEMMs, not just the value logged as ``num_heads``.
    """
    if target_tp_size <= 1:
        return

    try:
        attn = model_runner.model.model.layers[0].self_attn
    except AttributeError as exc:
        raise RuntimeError("failed to locate DSA self_attn module for TP shape validation") from exc

    _expect_module_attr(attn, "num_heads", local_num_heads, "self_attn")
    _expect_module_attr(attn, "num_local_heads", local_num_heads, "self_attn")

    qk_nope_head_dim = int(attn.qk_nope_head_dim)
    qk_rope_head_dim = int(attn.qk_rope_head_dim)
    v_head_dim = int(attn.v_head_dim)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    q_b_proj = getattr(attn, "q_b_proj", None)
    if q_b_proj is not None:
        q_b_out = local_num_heads * qk_head_dim
        _expect_module_attr(q_b_proj, "output_size", q_b_out, "q_b_proj")
        _expect_module_attr(q_b_proj, "output_size_per_partition", q_b_out, "q_b_proj")

    kv_b_proj = getattr(attn, "kv_b_proj", None)
    if kv_b_proj is not None:
        kv_b_out = local_num_heads * (qk_nope_head_dim + v_head_dim)
        _expect_module_attr(kv_b_proj, "output_size", kv_b_out, "kv_b_proj")
        _expect_module_attr(kv_b_proj, "output_size_per_partition", kv_b_out, "kv_b_proj")

    o_proj = getattr(attn, "o_proj", None)
    if o_proj is not None:
        o_proj_in = local_num_heads * v_head_dim
        _expect_module_attr(o_proj, "input_size", o_proj_in, "o_proj")
        _expect_module_attr(o_proj, "input_size_per_partition", o_proj_in, "o_proj")
        _expect_module_attr(o_proj, "output_size", int(attn.hidden_size), "o_proj")


def _import_sglang_forward_context():
    """Return SGLang's forward-context wrapper across runtime API versions."""
    try:
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    except ModuleNotFoundError:
        from contextlib import nullcontext

        class ForwardContext:
            def __init__(self, *args, **kwargs):
                pass

        def forward_context(_context):
            return nullcontext()

    return ForwardContext, forward_context


def load_model_runner(
    model_path: str,
    head_num: int,
    kv_cache_dtype: str,
    attention_backend: str,
    dsa_prefill_backend: str | None = None,
    device: str = "cuda:0",
    tp_rank: int = 0,
    gemm_type: str = "bfloat16",
    target_tp_size: int = 1,
    enable_piecewise_cuda_graph: bool = False,
    max_total_tokens: int | None = None,
    chunked_prefill_size: int | None = None,
    max_running_requests: int | None = None,
):
    """Load SGLang ModelRunner with dummy weights.

    Args:
        model_path: HuggingFace model path (e.g. "deepseek-ai/DeepSeek-V3.2").
        head_num: Number of local attention heads to benchmark for one target TP rank.
        kv_cache_dtype: Perf-DB-compatible string ("bfloat16" or "fp8").
            Mapped to SGLang-native string via SGLANG_KV_DTYPE.
        attention_backend: Backend string for ServerArgs (e.g. "dsa", "fa3").
        gemm_type: Expected Perf-DB label for the timed DSA projections. The
            checkpoint's own quantization metadata independently selects the
            SGLang loader; callers pre-filter this label to the executed path.

    Environment variables:
        SGLANG_TEST_NUM_LAYERS: Number of layers to load (default 2).
        SGLANG_LOAD_FORMAT: Weight format (default "dummy").
    """
    native_quant = _model_native_gemm_quant(model_path)
    sm_version = get_sm_version()
    if native_quant == "nvfp4" and sm_version not in _NATIVE_NVFP4_SMS:
        raise ValueError(
            f"SGLang has no collector-supported native NVFP4 model-loader backend on SM{sm_version}; "
            "Marlin is valid only for INT4-WO"
        )

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()

    device_str = str(device)
    if ":" in device_str:
        gpu_id = int(device_str.split(":")[-1])
    else:
        gpu_id = tp_rank

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")

    # Map perf-DB dtype to SGLang-native dtype.
    # trtllm_mla accepts "bf16" but rejects "bfloat16" in its validation.
    sglang_kv_dtype = SGLANG_KV_DTYPE.get(kv_cache_dtype, kv_cache_dtype)
    if attention_backend == "trtllm_mla" and sglang_kv_dtype == "bfloat16":
        sglang_kv_dtype = "bf16"

    # Use AIC's local model configs to avoid HF downloads while preserving
    # SGLang 0.5.14's native model identity.
    local_model_path = _resolve_local_model_path(model_path)
    load_quantization = {"fp8_block": "fp8", "nvfp4": "modelopt_fp4", None: None}[native_quant]

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        # Deviation (DSA prefill rows, every SM): v0.5.23 serves DSA prefill under the
        # breakable prefill CUDA graph on CUDA (model_executor/cuda_graph_config.py:119-123),
        # capturing passes up to min(8192, chunked_prefill_size) tokens for an MLA backend
        # (arg_groups/memory_hook.py:233-238; no DSA rule in cuda_graph_hook.py:170-255), i.e.
        # every one-shot point this collector times. Attention runs eagerly between the graph
        # segments (dsa_backend.py:3258-3261), but the dispatch differs from this eager timing:
        # inside BCG set_dsa_prefill_impl forces use_mha=False (dsa_backend.py:4050-4053), so
        # serving never takes MHA_ONE_SHOT, while eager takes it for max_kv_len <=
        # SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=2048 (:4057-4078; rows labelled
        # sglang_dsa_dense_mha_*); the indexer runs as one split op and computes the logits it
        # skips eagerly (dsa/dsa_indexer.py:1966-1970, 2030-2072). The 0.5.21 tables were timed
        # the same way. Retire by porting dsa_prefill_graph.py to the breakable runner (its
        # module docstring) once the measurement method for prefill graphs is decided.
        disable_prefill_cuda_graph=True,
        kv_cache_dtype=sglang_kv_dtype,
        max_total_tokens=max_total_tokens,
        # The request slots of the timed batches, not the count SGLang derives from the
        # dummy's token capacity (runtime_limits.request_slots).
        max_running_requests=max_running_requests,
        chunked_prefill_size=chunked_prefill_size,
        quantization=load_quantization,
    )

    server_args.attention_backend = attention_backend
    # Do NOT warm up gemm at sglang ModelRunner init (that is the separate
    # front-of-timeline gemm autotune phase, done over the FULL model). The DSA
    # module gemm tuning is instead absorbed into the dsa-module warmup below
    # (run_mla_module, under `with autotune(True)`), so it is part of the module
    # init we actually measure -- not a global init phase.
    server_args.disable_flashinfer_autotune = True
    # Match serving's FP8-KV dispatch: trtllm on capability major >= 10, flashmla_kv on
    # Hopper, for both phases (arg_groups/overrides.py:714-720 @v0.5.23). The GLM SM12x FP8
    # branch (flashinfer_sparse_mla, :637-642, 665-675) is not collected: the registry parks
    # SM120. BF16 KV stays on the runtime default (flashmla_sparse prefill, trtllm/fa3
    # decode, :721-726) because flashmla_kv rejects BF16.
    if attention_backend == "dsa" and sglang_kv_dtype == "fp8_e4m3":
        if dsa_prefill_backend is None:
            dsa_prefill_backend = "trtllm" if get_sm_version() >= 100 else "flashmla_kv"
        server_args.dsa_prefill_backend = dsa_prefill_backend
        server_args.dsa_decode_backend = dsa_prefill_backend
    print(
        f"Using attention backend: {attention_backend}, kv_cache_dtype: {sglang_kv_dtype}, "
        f"gpu_id: {gpu_id}, chunked_prefill_size={server_args.chunked_prefill_size}, "
        f"max_prefill_tokens={server_args.max_prefill_tokens}, max_total_tokens={max_total_tokens}, "
        f"max_running_requests={max_running_requests}"
    )

    if num_layers > 0 and load_format == "dummy":
        override_args = {
            "num_hidden_layers": num_layers,
            "num_attention_heads": head_num,
            "num_key_value_heads": head_num,
        }
        server_args.json_model_override_args = json.dumps(override_args)

    _set_envs_and_config(server_args)

    from collector.sglang.runtime_compat import free_local_port

    nccl_port = free_local_port()

    model_config = ModelConfig.from_server_args(server_args)
    expected_architecture = _module_model_architecture(model_path)
    actual_architecture = (model_config.hf_config.architectures or [None])[0]
    if actual_architecture != expected_architecture:
        raise RuntimeError(
            f"SGLang loaded architecture={actual_architecture!r} for {model_path}, expected {expected_architecture!r}"
        )

    # Bug A fix: ensure hf_config.quantization_config carries weight_block_size
    # so sglang constructs Fp8Config with block_quant=True. Without this, the
    # Fp8LinearMethod post-load path at fp8.py:660 transposes kv_b_proj.weight
    # from (out=7168, in=512) to (in=512, out=7168), and then
    # deepseek_weight_loader.py:555 unflatten(dim0=512, 448) fails with
    # `448 ∤ 512`. Re-inject the cached checkpoint metadata after
    # AutoConfig parsing so the block shape remains explicit on hf_config.
    if native_quant == "fp8_block":
        _ensure_fp8_block_quant_config(model_config.hf_config)

    # parallel geometry / config publish / distributed bootstrap: one helper per sglang generation
    from collector.sglang.runtime_compat import init_runtime_config

    _runner_parallel_kwargs = init_runtime_config(
        server_args, gpu_id, nccl_port=nccl_port, model_config=model_config, tp_rank=gpu_id, tp_size=server_args.tp_size
    )
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        nccl_port=nccl_port,
        server_args=server_args,
        **_runner_parallel_kwargs,
    )

    model_runner.alloc_memory_pool()
    model_runner.init_attention_backends()

    _patch_nsa_rope_contiguity(model_runner)
    _validate_dsa_tp_module_shapes(model_runner, head_num, target_tp_size)

    return model_runner


# ═══════════════════════════════════════════════════════════════════════
# Core Benchmarking
# ═══════════════════════════════════════════════════════════════════════


def _validate_mla_projection_precision(attention_module, gemm_type: str) -> None:
    """Reject rows whose single GEMM label cannot describe the loaded projections."""
    for name in ("fused_qkv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj"):
        projection = getattr(attention_module, name)
        method = projection.quant_method
        config = getattr(method, "quant_config", None)
        if projection.weight.dtype == torch.bfloat16:
            actual = "bfloat16"
        elif projection.weight.dtype == torch.float8_e4m3fn and getattr(config, "weight_block_size", None) is not None:
            actual = "fp8_block"
        else:
            raise ValueError(f"Cannot represent {name} precision with the ordinary MLA module key")
        if actual != gemm_type:
            raise ValueError(f"{name} executes {actual}, but the requested module GEMM label is {gemm_type}")


def run_attention_torch(
    model_runner,
    test_cases,
    head_num: int,
    test_layer: int,
    num_warmup: int,
    num_iterations: int,
    device: str,
    output_path: str | None,
    *,
    attn_type: str,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    target_tp_size: int = 1,
    use_module_cuda_graph: bool = False,
    dsa_prefill_backend: str | None = None,
    ordinary_mla: bool = False,
):
    """Run attention benchmark for both prefill and decode phases.

    Args:
        test_cases: List of (batch_size, seq_length, is_prefill) tuples.
        kv_cache_dtype: Perf-DB-compatible string for logging.
        compute_dtype: Perf-DB-compatible string for logging.
        gemm_type: Perf-DB-compatible string for logging.
    """
    attention_module = model_runner.model.model.layers[test_layer].self_attn
    architecture = _module_model_architecture(model_path)
    backend_name = model_runner.server_args.attention_backend
    resolved_dsa_prefill_backend = getattr(model_runner.server_args, "dsa_prefill_backend", None) or dsa_prefill_backend
    resolved_dsa_decode_backend = getattr(model_runner.server_args, "dsa_decode_backend", None) or dsa_prefill_backend
    version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)

    # Wideep MLA backward-compatibility: the old collect_wideep_attn.py logged
    # mla_dtype="fp8_block", kv_cache_dtype="fp8" for all runs, used the raw
    # backend name as kernel_source, and different op_name / filename patterns.
    # perf_database loaders (load_wideep_*_mla_data) expect these conventions.
    is_wideep_mla = attn_type == "mla" and not ordinary_mla

    if is_wideep_mla:
        log_mla_dtype = "fp8_block"
        log_kv_dtype = "fp8"
        log_gemm_type = "fp8_block"
    else:
        # DSA: log dtype strings that match the common.*QuantMode enum member
        # names expected by perf_database loaders (e.g. "bfloat16", "fp8").
        log_mla_dtype = compute_dtype
        log_kv_dtype = kv_cache_dtype
        log_gemm_type = gemm_type

    # QKV latent dimensions for AttentionInputs
    q_lora_rank = getattr(attention_module, "q_lora_rank", 1536) or 1536
    kv_lora_rank = getattr(attention_module, "kv_lora_rank", 512)
    qk_rope_head_dim = getattr(attention_module, "qk_rope_head_dim", 64)
    qkv_latent_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim

    def dummy_qkv_latent_func(h, fb):
        return torch.randn(h.shape[0], qkv_latent_dim, dtype=h.dtype, device=h.device)

    if ordinary_mla:
        if attn_type != "mla" or any(not case[2] for case in test_cases):
            raise ValueError("ordinary_mla collects MLA context modules only")
        _validate_mla_projection_precision(attention_module, gemm_type)
        log_mla_dtype = "fp8" if backend_name == "trtllm_mla" and kv_cache_dtype == "fp8" else "bfloat16"

    if ordinary_mla or attn_type == "dsa":
        # SGLang deepseek_v2.py:2525-2550,2690 @v0.5.23 supplies the
        # native projection to AttentionInputs for both MLA and DSA. Keep its fused/quantized
        # dispatch; the module perf row includes this down-projection.
        dummy_qkv_latent_func = attention_module.prepare_qkv_latent

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    logged_count = 0
    boundary_failures = []
    for test_case in test_cases:
        if len(test_case) == 4:
            batch_size, seq_length, is_prefill, prefix_len = test_case
        else:
            batch_size, seq_length, is_prefill = test_case
            prefix_len = 0

        if is_prefill:
            try:
                logged_count += int(
                    _run_prefill(
                        model_runner=model_runner,
                        attention_module=attention_module,
                        batch_size=batch_size,
                        seq_length=seq_length,
                        head_num=head_num,
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                        device=device,
                        output_path=output_path,
                        dummy_qkv_latent_func=dummy_qkv_latent_func,
                        attn_type=attn_type,
                        model_path=model_path,
                        architecture=architecture,
                        backend_name=backend_name,
                        version=version,
                        device_name=device_name,
                        log_mla_dtype=log_mla_dtype,
                        log_kv_dtype=log_kv_dtype,
                        log_gemm_type=log_gemm_type,
                        target_tp_size=target_tp_size,
                        prefix_len=prefix_len,
                        use_module_cuda_graph=use_module_cuda_graph,
                        dsa_prefill_backend=resolved_dsa_prefill_backend,
                        ordinary_mla=ordinary_mla,
                    )
                )
            except MeasurementBoundaryError as exc:
                failure = {
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "prefix_len": prefix_len,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                boundary_failures.append(failure)
                print(f"DSA measurement boundary failure: {json.dumps(failure, sort_keys=True)}")
        else:
            logged_count += int(
                _run_decode(
                    model_runner=model_runner,
                    attention_module=attention_module,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    head_num=head_num,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                    device=device,
                    output_path=output_path,
                    dummy_qkv_latent_func=dummy_qkv_latent_func,
                    attn_type=attn_type,
                    model_path=model_path,
                    architecture=architecture,
                    backend_name=backend_name,
                    version=version,
                    device_name=device_name,
                    log_mla_dtype=log_mla_dtype,
                    log_kv_dtype=log_kv_dtype,
                    log_gemm_type=log_gemm_type,
                    target_tp_size=target_tp_size,
                    use_module_cuda_graph=use_module_cuda_graph,
                    dsa_prefill_backend=resolved_dsa_decode_backend,
                )
            )
    if boundary_failures:
        raise MeasurementBoundaryError(
            "DSA prefill collection incomplete: "
            + json.dumps({"logged_count": logged_count, "boundary_failures": boundary_failures}, sort_keys=True)
        )
    return logged_count


def _run_prefill(
    model_runner,
    attention_module,
    batch_size: int,
    seq_length: int,
    head_num: int,
    num_warmup: int,
    num_iterations: int,
    device: str,
    output_path: str | None,
    dummy_qkv_latent_func,
    attn_type: str,
    model_path: str,
    architecture: str,
    backend_name: str,
    version: str,
    device_name: str,
    log_mla_dtype: str,
    log_kv_dtype: str,
    log_gemm_type: str,
    target_tp_size: int,
    prefix_len: int = 0,
    use_module_cuda_graph: bool = False,
    dsa_prefill_backend: str | None = None,
    ordinary_mla: bool = False,
):
    """Run prefill (context) benchmark for a single (batch_size, seq_length) point."""
    from array import array

    is_wideep_mla = attn_type == "mla" and not ordinary_mla
    # layers/layer_boundary/adapters/attention.py:44-70,113-114,140-141 @v0.5.23
    from sglang.srt.layers.layer_boundary.adapters.attention import AttentionInputs, get_attn_tp_context
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.common import checkpoint_kv_cache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.utils import BumpAllocator

    forward_context_type, forward_context = _import_sglang_forward_context()

    print(f"\nPrefill: batch_size={batch_size}, seq_length={seq_length}, prefix_len={prefix_len}")
    point_clock = PhaseClock()

    native_graph_context = None
    try:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()

        full_length = prefix_len + seq_length

        reqs = []
        for i in range(batch_size):
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=list(torch.randint(0, 10000, (full_length,)).tolist()),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.logprob_start_len = 0
            reqs.append(req)
        point_clock.mark("reqs")

        tree_cache = _radix_disabled_tree_cache(model_runner)
        if prefix_len:
            # The cached context is the KV of the chunks before this one. With the
            # radix cache disabled serving matches nothing, so a prefix lives only in
            # the request's own row from its previous chunk (mem_cache/allocation.py:
            # 363-371 @v0.5.23). Build that row as serving does: an extend over
            # [0, prefix_len) allocates it (schedule_batch.py:2847-2919), then the
            # scheduler's checkpoint advances prefix_len to its end (mem_cache/common.py:
            # 271-280); the next round extends [prefix_len, extend_end)
            # (schedule_batch.py:1105-1106, 1218-1220).
            for req in reqs:
                req.extend_end = prefix_len
            prefix_batch = _new_schedule_batch(model_runner, reqs, tree_cache)
            with _temporarily_chunked_alloc_extend(model_runner, batch_size * prefix_len):
                prefix_batch.prepare_for_extend()
            for req in reqs:
                checkpoint_kv_cache(req, tree_cache)
        for req in reqs:
            req.extend_end = full_length
        batch = _new_schedule_batch(model_runner, reqs, tree_cache)
        with _temporarily_chunked_alloc_extend(model_runner, batch_size * seq_length):
            batch.prepare_for_extend()
        point_clock.mark("batch")
        if attn_type == "dsa":
            from sglang.srt.managers.overlap_utils import resolve_forward_inputs

            resolve_forward_inputs(batch, _dsa_forward_input_relay(model_runner, batch))
        # sglang 0.5.16 made return_hidden_states_before_norm a required
        # keyword (ForwardBatch.init_new); older pins do not accept it.
        import inspect as _inspect

        _fb_kw = (
            {"return_hidden_states_before_norm": False}
            if "return_hidden_states_before_norm" in _inspect.signature(ForwardBatch.init_new).parameters
            else {}
        )
        forward_batch = ForwardBatch.init_new(batch, model_runner, **_fb_kw)
        if attn_type == "dsa":
            _validate_dsa_forward_tokens(forward_batch, batch_size * seq_length)
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        point_clock.mark("metadata")
        if attn_type == "dsa":
            _initialize_dsa_history(model_runner, attention_module, forward_batch, prefix_len)
            point_clock.mark("history")

        hidden_states = torch.randn(
            batch_size * seq_length,
            model_runner.model.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        positions = (
            torch.arange(prefix_len, prefix_len + seq_length, device="cuda")
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
            .flatten()
        )
        zero_allocator = BumpAllocator(buffer_size=256, dtype=torch.float32, device="cuda")

        attn_inputs = AttentionInputs(hidden_states, forward_batch, dummy_qkv_latent_func)
        get_attn_tp_context().set_attn_inputs(attn_inputs)

        # Preparation starts eagerly. Below, eligible DSA batches enter the
        # native TC piecewise runner with an attention-only model adapter.
        use_module_piecewise_replay = False
        use_full_model_piecewise_replay = False
        use_module_cuda_graph = False
        use_module_piecewise_context = False
        # Eager DSA timing: proper forward context but NO piecewise CUDA graph.
        # Used when the prefill chunk exceeds the piecewise graph's captured token
        # ceiling (see the fallback just below).
        use_module_eager_dsa_context = False
        # If an earlier case in this subprocess found piecewise CUDA graph
        # capture unsupported (e.g. sglang 0.5.13), time the module EAGERLY.
        if getattr(model_runner, "_aic_module_piecewise_fallback_eager", False):
            use_module_eager_dsa_context = True
        token_count = batch_size * seq_length
        # Production chunks prefill to chunked_prefill_size; a one-shot forward with
        # more query tokens than that needs more shared memory than the GPU has
        # (kernel falls back to a low-smem path that illegal-memory-accesses).
        # These (bs*seq > chunk) shapes are multi-chunk in serve; the single-chunk
        # shape (bs*seq <= chunk) is collected separately. Skip to avoid the crash.
        # Follow sglang: prefill is bounded only by chunked_prefill_size (sglang
        # chunks longer prefills). FlashMLA metadata is sized by num_sm_parts
        # (flashmla_backend.py), NOT a bs*seq shared-memory cap -- no smem limit.
        _chunk_cap = _runtime_chunk_size(model_runner)
        if token_count > _chunk_cap:
            if ordinary_mla:
                raise ValueError(f"Single module has {token_count} tokens, exceeding runtime chunk size {_chunk_cap}")
            print(
                f"  SKIP oversized prefill: token_count(bs*seq)={token_count} > "
                f"chunked_prefill_size={_chunk_cap}; serve chunks this, not a one-shot shape.",
                flush=True,
            )
            return 0
        # BUG (TODO: fix cuda-graph capture at large context): at large isl (e.g.
        # 16384) + prefix > 65536, the module CUDA-graph / piecewise capture
        # crashes -- "Module CUDA graph capture failed" / "RopeQuantize ... invalid
        # argument" / "Offset increment outside graph capture". Root cause is a
        # cuda-graph-capture incompatibility at large context (sglang/flashinfer),
        # not yet fixed. Force EAGER for prefix > 65536 (covers use_module_cuda_graph
        # too, which the token gate below does NOT) so these shapes still collect.
        if prefix_len > 65536:
            use_module_piecewise_replay = False
            use_module_cuda_graph = False
            use_module_piecewise_context = False
            use_module_eager_dsa_context = True
        # SGLang 0.5.14 removed the legacy piecewise-prefill ServerArgs fields.
        # This collector intentionally times the eager module path above, so no
        # piecewise token ceiling applies here.
        max_piecewise_tokens = 0
        if use_module_piecewise_replay and max_piecewise_tokens and token_count > max_piecewise_tokens:
            # piecewise_cuda_graph_max_tokens is an sglang serving knob (2048 by
            # default for MLA backends, to avoid kernel-dispatch regression) — NOT
            # a limit on valid prefill sizes. A larger chunk simply has no captured
            # graph, so we must still collect it by timing the module EAGERLY with
            # the correct DSA forward context. (Production likewise runs MLA prefill
            # chunks above this ceiling without a piecewise graph.) The piecewise
            # switch decides graph-or-not; it must never drop the data point.
            print(
                "  Module piecewise CUDA graph disabled: "
                f"num_tokens={token_count} exceeds piecewise max={max_piecewise_tokens}; "
                "timing eager (no graph), still collected"
            )
            use_module_piecewise_replay = False
            use_module_cuda_graph = False
            use_module_piecewise_context = False
            use_module_eager_dsa_context = True
        if use_module_piecewise_replay:
            print("  Module piecewise CUDA graph replay enabled")
            use_full_model_piecewise_replay = False
            use_module_cuda_graph = False
            use_module_piecewise_context = False
        if use_full_model_piecewise_replay:
            print("  Full model piecewise CUDA graph replay enabled")
            use_module_cuda_graph = False
            use_module_piecewise_context = False
        if use_module_piecewise_context:
            from sglang.srt.compilation.piecewise_context_manager import (
                enable_piecewise_cuda_graph,
            )
            from sglang.srt.compilation.piecewise_context_manager import (
                set_forward_context as set_piecewise_forward_context,
            )

            if use_module_cuda_graph:
                print("  Module CUDA graph disabled; using piecewise CUDA graph context")
                use_module_cuda_graph = False

        if use_module_cuda_graph:
            _patch_nsa_indexer_compile_for_module_cuda_graph(attention_module)

        if use_module_piecewise_replay:
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            replay_token_count = getattr(model_runner, "_aic_module_piecewise_replay_token_count", None)
            if (
                getattr(model_runner, "_aic_module_piecewise_replay_initialized", False)
                and replay_token_count != token_count
            ):
                # The module-only wrapper compiles against the current token
                # shape. Reusing it for a different token_count can trigger
                # runtime recompilation outside the PCG capture stream, which
                # SGLang rejects with "PCG capture stream is not set".
                model_runner.piecewise_cuda_graph_runner = None
                model_runner._aic_module_piecewise_replay_initialized = False
                model_runner._aic_module_piecewise_replay_token_count = None

            if getattr(model_runner, "_aic_module_piecewise_replay_initialized", False):
                print("  Module piecewise runner=True")
            else:
                original_piecewise_tokens = getattr(model_runner.server_args, "piecewise_cuda_graph_tokens", None)
                original_piecewise_max_tokens = getattr(
                    model_runner.server_args, "piecewise_cuda_graph_max_tokens", None
                )
                model_runner.server_args.piecewise_cuda_graph_tokens = [token_count]
                model_runner.server_args.piecewise_cuda_graph_max_tokens = token_count
                module_hidden_states = torch.randn(
                    token_count,
                    model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                module_logits = torch.empty(
                    module_hidden_states.shape[0],
                    1,
                    dtype=torch.float32,
                    device="cuda",
                )

                class _AttentionOnlyLayer(torch.nn.Module):
                    def __init__(self, self_attn):
                        super().__init__()
                        self.self_attn = self_attn

                class _AttentionOnlyLanguageModel(torch.nn.Module):
                    def __init__(self, self_attn, hidden_buffer, logits_buffer):
                        super().__init__()
                        self.layers = torch.nn.ModuleList([_AttentionOnlyLayer(self_attn)])
                        self.hidden_states = hidden_buffer
                        self.logits = logits_buffer

                    def forward(
                        self,
                        input_ids: torch.Tensor,
                        positions: torch.Tensor,
                        forward_batch,
                        **kwargs,
                    ):
                        hidden = self.hidden_states[: input_ids.shape[0]]
                        attn_inputs = AttentionInputs(hidden, forward_batch, dummy_qkv_latent_func)
                        get_attn_tp_context().set_attn_inputs(attn_inputs)
                        out = self.layers[0].self_attn(
                            positions=positions,
                            hidden_states=hidden,
                            forward_batch=forward_batch,
                            zero_allocator=zero_allocator,
                        )
                        if isinstance(out, tuple):
                            out = out[0]
                        return LogitsProcessorOutput(
                            next_token_logits=self.logits[: input_ids.shape[0]],
                            hidden_states=out,
                        )

                class _AttentionOnlyModel(torch.nn.Module):
                    def __init__(self, self_attn):
                        super().__init__()
                        self.model = _AttentionOnlyLanguageModel(self_attn, module_hidden_states, module_logits)

                    def forward(
                        self,
                        input_ids: torch.Tensor,
                        positions: torch.Tensor,
                        forward_batch,
                        **kwargs,
                    ):
                        return self.model.forward(input_ids, positions, forward_batch, **kwargs)

                original_model = model_runner.model
                original_num_layers = model_runner.model_config.num_hidden_layers
                import sglang.srt.model_executor.piecewise_cuda_graph_runner as pcg_runner_mod
                from sglang.srt.compilation.piecewise_context_manager import (
                    set_forward_context as set_piecewise_forward_context,
                )
                from sglang.srt.layers.dp_attention import (
                    DpPaddingMode,
                    set_dp_buffer_len,
                    set_is_extend_in_batch,
                )

                original_install_torch_compiled = pcg_runner_mod.install_torch_compiled
                original_warmup_compile = pcg_runner_mod.PiecewiseCudaGraphRunner.warmup_compile
                original_capture_one_batch_size = pcg_runner_mod.PiecewiseCudaGraphRunner.capture_one_batch_size

                def _install_torch_compiled_with_module_dims(module, *args, **kwargs):
                    if isinstance(module, _AttentionOnlyLanguageModel) and kwargs.get("dynamic_arg_dims") is None:
                        kwargs["dynamic_arg_dims"] = {
                            "input_ids": 0,
                            "positions": 0,
                        }
                    return original_install_torch_compiled(module, *args, **kwargs)

                def _run_module_piecewise_target(runner, target_forward_batch):
                    static_forward_batch = runner.replay_prepare(target_forward_batch)
                    if static_forward_batch.dp_padding_mode is None:
                        static_forward_batch.dp_padding_mode = DpPaddingMode.get_default_mode_in_cuda_graph()
                    runner.model_runner.attn_backend.init_forward_metadata(target_forward_batch)
                    static_forward_batch.dp_local_start_pos = None
                    static_forward_batch.dp_local_num_tokens = None
                    set_dp_buffer_len(
                        None,
                        len(static_forward_batch.input_ids),
                        static_forward_batch.dp_padding_mode.is_max_len(),
                    )
                    set_is_extend_in_batch(False)
                    with (
                        forward_context(forward_context_type(attn_backend=runner.model_runner.attn_backend)),
                        set_piecewise_forward_context(
                            static_forward_batch,
                            runner.attention_layers,
                            runner.quant_config,
                            runner.moe_layers,
                            runner.moe_fusions,
                            dsa_indexers=runner.dsa_indexers,
                        ),
                    ):
                        return runner.model_runner.model.forward(
                            static_forward_batch.input_ids,
                            static_forward_batch.positions,
                            static_forward_batch,
                        )

                def _warmup_compile_with_module_batch(runner, num_tokens: int):
                    target_forward_batch = getattr(
                        runner.model_runner,
                        "_aic_module_piecewise_forward_batch",
                        None,
                    )
                    if target_forward_batch is not None and num_tokens in runner.capture_num_tokens:
                        return _run_module_piecewise_target(runner, target_forward_batch)
                    return original_warmup_compile(runner, num_tokens)

                def _capture_one_batch_size_with_module_batch(runner, num_tokens: int):
                    target_forward_batch = getattr(
                        runner.model_runner,
                        "_aic_module_piecewise_forward_batch",
                        None,
                    )
                    if target_forward_batch is not None and num_tokens in runner.capture_num_tokens:
                        for _ in range(2):
                            runner.device_module.synchronize()
                            runner.model_runner.tp_group.barrier()
                            _run_module_piecewise_target(runner, target_forward_batch)
                        return
                    return original_capture_one_batch_size(runner, num_tokens)

                module_model = _AttentionOnlyModel(attention_module).to("cuda")
                module_model.config = getattr(original_model, "config", None)
                module_model.quant_config = getattr(original_model, "quant_config", None)
                module_model.model.config = getattr(
                    getattr(original_model, "model", original_model), "config", module_model.config
                )
                model_runner.model = module_model
                model_runner.model_config.num_hidden_layers = 1
                # sglang >= 0.5.13: PiecewiseCudaGraphRunner.replay_prepare reads
                # len(forward_batch.input_ids); the module-only attention forward
                # leaves it None (compute is driven by hidden_states via attn_inputs).
                # Populate a dummy id tensor of the right token count so replay_prepare
                # does not crash (the attention-only model ignores input_ids).
                if getattr(forward_batch, "input_ids", None) is None:
                    forward_batch.input_ids = torch.zeros(token_count, dtype=torch.int64, device="cuda")
                model_runner._aic_module_piecewise_forward_batch = forward_batch
                try:
                    pcg_runner_mod.install_torch_compiled = _install_torch_compiled_with_module_dims
                    pcg_runner_mod.PiecewiseCudaGraphRunner.warmup_compile = _warmup_compile_with_module_batch
                    pcg_runner_mod.PiecewiseCudaGraphRunner.capture_one_batch_size = (
                        _capture_one_batch_size_with_module_batch
                    )
                    with torch.no_grad():
                        model_runner.init_piecewise_cuda_graphs()
                except Exception as _pw_exc:
                    # Piecewise CUDA graph capture unsupported in this collector
                    # context (e.g. sglang >= 0.5.13: concat_mla_absorb_q launches
                    # on the device default stream, illegal under the piecewise
                    # graph context -> cudaErrorInvalidValue). Clear the non-sticky
                    # error and fall back to EAGER module timing for this and all
                    # subsequent cases in this subprocess.
                    print(
                        f"  Module piecewise CUDA graph unsupported "
                        f"({type(_pw_exc).__name__}: {_pw_exc}); falling back to EAGER"
                    )
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    model_runner._aic_module_piecewise_fallback_eager = True
                    model_runner.piecewise_cuda_graph_runner = None
                    model_runner.model = original_model
                finally:
                    pcg_runner_mod.install_torch_compiled = original_install_torch_compiled
                    pcg_runner_mod.PiecewiseCudaGraphRunner.warmup_compile = original_warmup_compile
                    pcg_runner_mod.PiecewiseCudaGraphRunner.capture_one_batch_size = original_capture_one_batch_size
                    model_runner.model_config.num_hidden_layers = original_num_layers
                    if hasattr(model_runner, "_aic_module_piecewise_forward_batch"):
                        delattr(model_runner, "_aic_module_piecewise_forward_batch")
                    model_runner.server_args.piecewise_cuda_graph_tokens = original_piecewise_tokens
                    model_runner.server_args.piecewise_cuda_graph_max_tokens = original_piecewise_max_tokens
                if not getattr(model_runner, "_aic_module_piecewise_fallback_eager", False):
                    model_runner._aic_module_piecewise_replay_initialized = True
                    model_runner._aic_module_piecewise_replay_token_count = token_count
                    print(f"  Module piecewise runner={model_runner.piecewise_cuda_graph_runner is not None}")

        # Current case: piecewise capture just fell back -> time it eagerly.
        if getattr(model_runner, "_aic_module_piecewise_fallback_eager", False) and use_module_piecewise_replay:
            use_module_piecewise_replay = False
            use_module_eager_dsa_context = True

        # skip_indexer collection: GLM-5.2 (index_topk_freq>1) reuse layers run
        # proj GEMMs + DSA attention but NOT the per-layer indexer (mqa logits +
        # topk + index-K store). SGLang carries the producer layer's real topk as
        # the second self-attention return value when next_skip_topk=True. Capture
        # that production value from warmup[0], then force skip_topk=True and feed
        # it back as prev_topk_indices. Warmup[1:] and every timed run therefore
        # execute the reuse-layer path. Short prefill uses MHA_ONE_SHOT and never
        # runs the indexer, so there is intentionally no topk tensor to carry.
        _skip_indexer = _dsa_skip_indexer_enabled(attn_type, model_path)
        # The skip-indexer pass only takes effect through call_attention_module,
        # which threads _skip_kwargs() (skip_topk + reused prev_topk_indices) into
        # the module forward. The full/module piecewise REPLAY path calls
        # model_runner.forward() and cannot bypass the indexer, so it would record
        # full-indexer latency for a skip row. Force the eager module path (which
        # still uses the piecewise CONTEXT when available) whenever skipping.
        if _skip_indexer:
            use_full_model_piecewise_replay = False
            use_module_piecewise_replay = False
        _skip_uses_dense_mha = False
        if _skip_indexer:
            from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
                AttnForwardMethod,
            )

            with forward_context(forward_context_type(attn_backend=model_runner.attn_backend)):
                _skip_uses_dense_mha = (
                    attention_module.dispatch_attn_forward_method(forward_batch) == AttnForwardMethod.MHA_ONE_SHOT
                )
        _skip_state = {"prev_topk": None}
        if _skip_indexer and getattr(attention_module, "indexer", None) is None:
            raise RuntimeError(
                f"skip_indexer requested for {attn_type} but the attention module has no indexer; "
                "refusing to record a skip row with full-indexer latency."
            )
        original_next_skip_topk = getattr(attention_module, "next_skip_topk", None)
        if _skip_indexer:
            attention_module.skip_topk = _skip_uses_dense_mha
            if _skip_uses_dense_mha:
                print("  Skip-indexer reuse uses MHA_ONE_SHOT; no topk tensor is produced or required")
            else:
                attention_module.next_skip_topk = True  # return the producer topk from warmup[0]

        executed_dsa_source = None
        if attn_type == "dsa":
            dsa_backend = model_runner.attn_backend
            if dsa_backend.use_mha:
                dense_leaf = "trtllm_ragged" if dsa_backend.device_sm_major >= 10 else "fa3"
                executed_dsa_source = f"sglang_dsa_dense_mha_{dense_leaf}"
            else:
                indexer_mode = "skip_indexer" if _skip_indexer else "indexer"
                executed_dsa_source = f"sglang_dsa_{indexer_mode}_{dsa_backend.dsa_prefill_impl}"

        def _skip_kwargs():
            # Once a real topk index is captured, bypass the indexer for all
            # subsequent (timed) forwards by forcing skip_topk + reusing it.
            if _skip_indexer and (_skip_uses_dense_mha or _skip_state["prev_topk"] is not None):
                attention_module.skip_topk = True
                if _skip_state["prev_topk"] is not None:
                    return {"prev_topk_indices": _skip_state["prev_topk"]}
            return {}

        def call_attention_module():
            if ordinary_mla or attn_type == "dsa":
                # AttentionInputs caches QKV (layer_boundary/adapters/attention.py:61-70);
                # serving builds fresh inputs per layer/forward
                # (layer_boundary/prepare.py:254-264 @v0.5.23).
                # Reset inside EVERY timed iteration, including warmup, so the
                # down-projection is measured rather than reused from warmup.
                get_attn_tp_context().set_attn_inputs(
                    AttentionInputs(hidden_states, forward_batch, dummy_qkv_latent_func)
                )
            with forward_context(forward_context_type(attn_backend=model_runner.attn_backend)):
                if use_module_piecewise_context:
                    with (
                        enable_piecewise_cuda_graph(),
                        set_piecewise_forward_context(
                            forward_batch,
                            getattr(model_runner, "attention_layers", []),
                            getattr(model_runner.model, "quant_config", None),
                            getattr(model_runner, "moe_layers", []),
                            getattr(model_runner, "moe_fusions", []),
                            dsa_indexers=getattr(model_runner, "dsa_indexers", None),
                        ),
                    ):
                        # Keep DSA dispatch consistent with SGLang's
                        # piecewise path even when we do not replay a
                        # captured CUDA graph for this token count.
                        return attention_module(
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            zero_allocator=zero_allocator,
                            **_skip_kwargs(),
                        )
                elif use_module_eager_dsa_context:
                    # Prefill chunk exceeds the piecewise graph ceiling: time the
                    # module EAGERLY in the non-piecewise "original mode" — exactly
                    # what production runs for >max-token MLA prefill chunks (sglang
                    # caps piecewise at 2048 for MLA precisely to avoid the piecewise
                    # kernel-dispatch regression). No piecewise forward context, no
                    # graph: the standard forward_context entered above plus the
                    # attention metadata is all the NSA indexer needs (its metadata
                    # is built in init_forward_metadata, not via the forward context).
                    # Metadata was planned once for this batch above. Serving
                    # does this before model.forward, not once per layer:
                    # model_executor/runner/eager_runner.py:296-339 then :352-366 @v0.5.23.
                    # Retain the native ForwardContext while timing the layer.
                    return attention_module(
                        positions=positions,
                        hidden_states=hidden_states,
                        forward_batch=forward_batch,
                        zero_allocator=zero_allocator,
                        **_skip_kwargs(),
                    )
                else:
                    return attention_module(
                        positions=positions,
                        hidden_states=hidden_states,
                        forward_batch=forward_batch,
                        zero_allocator=zero_allocator,
                        **_skip_kwargs(),
                    )

        point_clock.mark("setup")
        last_can_run_graph = None

        def call_full_model_piecewise_replay():
            nonlocal last_can_run_graph
            output = model_runner.forward(forward_batch)
            last_can_run_graph = getattr(output, "can_run_graph", None)

        call_target = (
            call_full_model_piecewise_replay
            if (use_full_model_piecewise_replay or use_module_piecewise_replay)
            else call_attention_module
        )

        if attn_type == "dsa":
            from collector.sglang.dsa_prefill_graph import dsa_prefill_graph, graph_token_bucket

            if graph_token_bucket(model_runner, token_count) is not None:
                graph_context = dsa_prefill_graph(
                    model_runner,
                    attention_module,
                    forward_batch,
                    hidden_states,
                    zero_allocator,
                    skip_indexer=_skip_indexer,
                )
                call_target = graph_context.__enter__()
                native_graph_context = graph_context
                # Native piecewise metadata disables MHA_ONE_SHOT even for
                # small inputs (under a breakable prefill graph,
                # dsa_backend.py:4050-4053 @v0.5.23).
                # Label the path selected after native load_batch, not the
                # earlier eager preparation or the requested backend.
                if model_runner.attn_backend.use_mha:
                    raise RuntimeError("Native piecewise DSA unexpectedly selected dense MHA")
                indexer_mode = "skip_indexer" if _skip_indexer else "indexer"
                executed_dsa_source = f"sglang_dsa_{indexer_mode}_{model_runner.attn_backend.dsa_prefill_impl}"
                # Native dispatch/capture above is observed, not predicted from
                # the requested shape. The compiled Inner enters model-level
                # Dynamo on every replay, so its timing cannot represent one
                # attention layer.
                # Preserve the native path and its failure; never emit a row
                # with this known measurement-boundary error or fall back to eager.
                raise MeasurementBoundaryError(
                    f"Unqualified SGLang DSA native TC prefill measurement boundary: {executed_dsa_source}; "
                    "per-model Dynamo entry remains inside per-layer timing; no perf row was written"
                )

        # Warmup — run UNDER the flashinfer autotune context so the fp4_gemm
        # autotuning is absorbed into this module warmup (tuned here, cached for
        # the timed region) instead of running as a separate model-init phase.
        import contextlib

        try:
            from flashinfer.autotuner import autotune as _fi_autotune

            _tune_ctx = _fi_autotune(True) if native_graph_context is None else contextlib.nullcontext()
        except Exception:
            _tune_ctx = contextlib.nullcontext()
        try:
            with _tune_ctx:
                for _ in range(num_warmup):
                    with torch.no_grad():
                        warmup_output = call_target()
                    torch.cuda.synchronize()
                    if (
                        _skip_indexer
                        and _skip_state["prev_topk"] is None
                        and isinstance(warmup_output, tuple)
                        and len(warmup_output) == 2
                        and warmup_output[1] is not None
                    ):
                        _skip_state["prev_topk"] = warmup_output[1].detach()
                        attention_module.next_skip_topk = original_next_skip_topk
        finally:
            if _skip_indexer:
                attention_module.next_skip_topk = original_next_skip_topk
        if use_full_model_piecewise_replay or use_module_piecewise_replay:
            print(f"  Piecewise can_run_graph={last_can_run_graph}")

        if (
            _skip_indexer
            and native_graph_context is None
            and not _skip_uses_dense_mha
            and _skip_state["prev_topk"] is None
        ):
            raise RuntimeError(
                f"skip_indexer pass for {attn_type} captured no topk index during warmup; "
                "refusing to record a skip row with full-indexer latency."
            )

        point_clock.mark("warmup")
        module_cuda_graph = None
        if use_module_cuda_graph:
            try:
                torch.cuda.synchronize()
                module_cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(module_cuda_graph):
                    call_target()
                torch.cuda.synchronize()
                print("  Module CUDA graph capture enabled")
            except Exception:
                print("  Module CUDA graph capture failed")
                raise

        point_clock.mark("capture")
        # Timed runs — run ALL iterations back-to-back, ONE sync at the end, then
        # divide by count. Per-iteration cuda.synchronize would insert GPU-idle
        # bubbles + per-launch latency between modules (NOT representative); the
        # amortized back-to-back timing matches the serve continuous pipeline.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            for i in range(num_iterations):
                if module_cuda_graph is not None:
                    module_cuda_graph.replay()
                else:
                    call_target()
        end_event.record()
        torch.cuda.synchronize()
        avg_time_ms = start_event.elapsed_time(end_event) / num_iterations
        point_clock.mark("timed")

        # Log perf — wideep MLA uses old filename/op_name/kernel_source conventions
        try:
            if is_wideep_mla:
                perf_fname = "wideep_context_mla_perf.txt"
                op_name = "mla_context"
                kernel_source = backend_name
            else:
                _skip_sfx = "_skip_indexer" if _dsa_skip_indexer_enabled(attn_type, model_path) else ""
                # full and skip share ONE perf file; the op_name column (with the
                # _skip_indexer suffix) is what distinguishes skip rows. No extra column.
                perf_fname = f"{attn_type}_context_module_perf.txt"
                op_name = f"{attn_type}_context_module{_skip_sfx}"
                kernel_source = executed_dsa_source or f"{attn_type}_{backend_name}"
            perf_filename = _resolve_perf_path(output_path, perf_fname)
            if not log_perf(
                item_list=[
                    {
                        "model": model_path,
                        "architecture": architecture,
                        "mla_dtype": log_mla_dtype,
                        "kv_cache_dtype": log_kv_dtype,
                        "gemm_type": log_gemm_type,
                        "num_heads": head_num,
                        "batch_size": batch_size,
                        "isl": seq_length,
                        "tp_size": target_tp_size,
                        "step": prefix_len,
                        "latency": f"{avg_time_ms:.4f}",
                    }
                ],
                framework="SGLang",
                version=version,
                device_name=device_name,
                op_name=op_name,
                kernel_source=kernel_source,
                perf_filename=perf_filename,
            ):
                raise PerfLogWriteError(f"failed to persist prefill row to {perf_filename}")
        except PerfLogWriteError:
            raise
        except Exception as e:
            print(f"  Warning: failed to log prefill metrics: {e}")
            return False

        print(f"  Prefill: {avg_time_ms:.3f} ms (back-to-back avg over {num_iterations} iters)")
        return True

    except (MeasurementBoundaryError, PerfLogWriteError):
        raise
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).lower()
        if "out of memory" in error_str:
            print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
            torch.cuda.empty_cache()
            return False
        if "cuda" in error_str and "illegal" in error_str:
            print("  CUDA illegal access detected — stopping this subprocess to preserve prior rows")
            raise CudaIllegalAccessError(f"CUDA illegal access at b={batch_size}, s={seq_length}") from e
        print("  Skipping this configuration...")
        return False
    finally:
        cleanup_errors = []
        cleanup_actions = []
        if native_graph_context is not None:
            cleanup_actions.append(
                ("native_prefill_graph.close", lambda: native_graph_context.__exit__(None, None, None))
            )
        cleanup_actions.extend(
            (
                ("req_to_token_pool.clear", model_runner.req_to_token_pool.clear),
                ("token_to_kv_pool_allocator.clear", model_runner.token_to_kv_pool_allocator.clear),
                ("torch.cuda.empty_cache", torch.cuda.empty_cache),
            )
        )
        for cleanup_name, cleanup_fn in cleanup_actions:
            try:
                cleanup_fn()
            except Exception as cleanup_exc:
                cleanup_errors.append(f"{cleanup_name}: {type(cleanup_exc).__name__}: {cleanup_exc}")
        if cleanup_errors:
            raise RuntimeError(f"DSA/MLA prefill cleanup failed: {'; '.join(cleanup_errors)}")
        point_clock.mark("cleanup")
        point_clock.add_to(_POINT_PHASE_TOTALS)
        print(
            f"  [DSA point timing] prefill b={batch_size}, s={seq_length}, prefix={prefix_len}: {point_clock.render()}"
        )


def _run_decode(
    model_runner,
    attention_module,
    batch_size: int,
    seq_length: int,
    head_num: int,
    num_warmup: int,
    num_iterations: int,
    device: str,
    output_path: str | None,
    dummy_qkv_latent_func,
    attn_type: str,
    model_path: str,
    architecture: str,
    backend_name: str,
    version: str,
    device_name: str,
    log_mla_dtype: str,
    log_kv_dtype: str,
    log_gemm_type: str,
    target_tp_size: int,
    use_module_cuda_graph: bool = False,
    dsa_prefill_backend: str | None = None,
):
    """Run decode (generation) benchmark for a single (batch_size, kv_cache_length) point."""
    from array import array

    is_wideep_mla = attn_type == "mla"
    # layers/layer_boundary/adapters/attention.py:44-70,113-114,140-141 @v0.5.23
    from sglang.srt.layers.layer_boundary.adapters.attention import AttentionInputs, get_attn_tp_context
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner import model_capture_mode
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.utils import BumpAllocator

    forward_context_type, forward_context = _import_sglang_forward_context()

    print(f"\nDecode: batch_size={batch_size}, kv_cache_length={seq_length}")
    point_clock = PhaseClock()

    try:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()

        reqs = []
        for i in range(batch_size):
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=list(torch.randint(0, 10000, (seq_length,)).tolist()),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            # The prompt is one extend over [prefix_len=0, extend_end) (schedule_batch.py:1105-1106 @v0.5.23).
            req.extend_end = len(req.origin_input_ids)
            req.logprob_start_len = 0
            req.cached_tokens = 0
            req.already_computed = 0
            reqs.append(req)
        point_clock.mark("reqs")

        batch = _new_schedule_batch(model_runner, reqs, _radix_disabled_tree_cache(model_runner))
        # Allocate KV cache slots, then switch to decode
        batch.prepare_for_extend()
        if attn_type == "dsa":
            from sglang.srt.managers.overlap_utils import resolve_forward_inputs

            input_relay = _dsa_forward_input_relay(model_runner, batch)
            resolve_forward_inputs(batch, input_relay)
        for req in batch.reqs:
            req.output_ids.append(0)
        if attn_type == "dsa":
            # Match the non-overlap scheduler's sampled-token handoff
            # (scheduler.py:4588-4597, 4688-4696). The synthetic next
            # token is the same zero appended to each Req.
            from sglang.srt.managers.overlap_utils import RelayPayload

            input_relay.stash(
                batch.req_pool_indices,
                RelayPayload(bonus_tokens=torch.zeros(batch_size, dtype=torch.int64, device=batch.device)),
            )
            batch.input_ids = None
        batch.prepare_for_decode()
        point_clock.mark("batch")
        if attn_type == "dsa":
            resolve_forward_inputs(batch, input_relay)
        import inspect as _inspect

        _fb_kw = (
            {"return_hidden_states_before_norm": False}
            if "return_hidden_states_before_norm" in _inspect.signature(ForwardBatch.init_new).parameters
            else {}
        )  # sglang 0.5.16 required keyword (see the context path above)
        forward_batch_decode = ForwardBatch.init_new(batch, model_runner, **_fb_kw)
        if attn_type == "dsa":
            _validate_dsa_forward_tokens(forward_batch_decode, batch_size)
        model_runner.attn_backend.init_forward_metadata(forward_batch_decode)
        point_clock.mark("metadata")
        if attn_type == "dsa":
            _initialize_dsa_history(model_runner, attention_module, forward_batch_decode, seq_length)
            point_clock.mark("history")

        graph_batch_size = (
            _generation_cuda_graph_batch_size(model_runner, batch_size) if attn_type == "dsa" else batch_size
        )
        if graph_batch_size is not None and graph_batch_size != batch_size:
            forward_batch_decode = _padded_dsa_decode_batch(model_runner, forward_batch_decode, graph_batch_size)
            model_runner.attn_backend.init_forward_metadata(forward_batch_decode)
        if attn_type == "dsa" and graph_batch_size is not None:
            # A graph-covered decode step runs on the backend's CUDA-graph metadata:
            # DecodeCudaGraphRunner allocates it (init_cuda_graph_state,
            # model_executor/runner/decode_cuda_graph_runner.py:429), builds it at capture
            # (init_forward_metadata_out_graph(in_capture=True), :1400) and refills it before
            # every replay (init_forward_metadata_out_graph, :1812). Its page tables span
            # req_to_token's full width (DSABackend.init_cuda_graph_state,
            # layers/attention/dsa_backend.py), which the indexer passes to top-k as the
            # max sequence length (dsa/dsa_indexer_kpool.py _get_topk_paged): past the
            # cluster floor (topk_v2.cuh get_static_cluster_floor) serving runs
            # topk_small_batch_cluster_kernel, where batch-sized eager metadata ran
            # topk_main_kernel.
            model_runner.attn_backend.init_cuda_graph_state(graph_batch_size, graph_batch_size)
            model_runner.attn_backend.init_forward_metadata_out_graph(forward_batch_decode, in_capture=True)
            model_runner.attn_backend.init_forward_metadata_out_graph(forward_batch_decode)
        execution_batch_size = forward_batch_decode.batch_size

        decode_hidden = torch.randn(
            execution_batch_size,
            model_runner.model.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        decode_positions = (
            forward_batch_decode.positions
            if attn_type == "dsa"
            else torch.full((batch_size,), seq_length, device="cuda")
        )
        zero_allocator = BumpAllocator(buffer_size=2048, dtype=torch.float32, device="cuda")

        attn_inputs_decode = AttentionInputs(decode_hidden, forward_batch_decode, dummy_qkv_latent_func)
        get_attn_tp_context().set_attn_inputs(attn_inputs_decode)
        point_clock.mark("inputs")
        if use_module_cuda_graph or _env_flag("AIC_ENABLE_MODULE_CUDA_GRAPH"):
            _patch_nsa_indexer_compile_for_module_cuda_graph(attention_module)

        use_benchmark_cuda_graph = graph_batch_size is not None

        # skip_indexer (see _run_prefill): obtain the real producer-layer topk
        # through SGLang's next_skip_topk return contract, then reuse it so the
        # timed runs (and any captured CUDA graph) exclude the per-layer indexer.
        _skip_indexer = _dsa_skip_indexer_enabled(attn_type, model_path)
        _skip_state = {"prev_topk": None}
        if _skip_indexer and getattr(attention_module, "indexer", None) is None:
            raise RuntimeError(
                f"skip_indexer requested for {attn_type} but the attention module has no indexer; "
                "refusing to record a skip row with full-indexer latency."
            )
        original_next_skip_topk = getattr(attention_module, "next_skip_topk", None)
        if _skip_indexer:
            attention_module.skip_topk = False
            attention_module.next_skip_topk = True

        executed_dsa_source = None
        if attn_type == "dsa":
            indexer_mode = "skip_indexer" if _skip_indexer else "indexer"
            executed_dsa_source = f"sglang_dsa_{indexer_mode}_{model_runner.attn_backend.dsa_decode_impl}"

        def _skip_kwargs():
            if _skip_indexer and _skip_state["prev_topk"] is not None:
                attention_module.skip_topk = True
                return {"prev_topk_indices": _skip_state["prev_topk"]}
            return {}

        def kernel_func():
            # SGLang's DecodeCudaGraphRunner wraps capture in
            # model_capture_mode(); DSA uses that state to select its
            # production dual-stream indexer path.
            from contextlib import nullcontext

            capture_context = model_capture_mode() if use_benchmark_cuda_graph else nullcontext()
            with capture_context, forward_context(forward_context_type(attn_backend=model_runner.attn_backend)):
                if attn_type == "dsa":
                    # SGLang layer_boundary/adapters/attention.py:61-70,
                    # prepare.py:254-264 @v0.5.23: QKV is cached within one AttentionInputs only.
                    # Refresh for every forward so capture includes the native
                    # projection instead of reusing the warmup's latent.
                    get_attn_tp_context().set_attn_inputs(
                        AttentionInputs(decode_hidden, forward_batch_decode, dummy_qkv_latent_func)
                    )
                return attention_module(
                    positions=decode_positions,
                    hidden_states=decode_hidden,
                    forward_batch=forward_batch_decode,
                    zero_allocator=zero_allocator,
                    **_skip_kwargs(),
                )

        # Pre-warm JIT / autotuning before CUDA graph capture.
        # DSA decode on Blackwell calls DeepGEMM fp8_paged_mqa_logits and
        # flashinfer trtllm_batch_decode_with_kv_cache_mla, both of which
        # JIT or autotune on first call to a new (heads, bs, kv_len) shape.
        # If that work spills into the graph-capture window inside
        # benchmark_with_power, it emits cudaMemcpy-like ops that aren't
        # permitted during capture and the whole sweep silently skips
        # (Issue #3 — reproduces only at reduced heads ∈ {8, 16, 32} where
        # the warmup-time JIT cache from heads=64 doesn't satisfy the new
        # template instantiation). A few extra eager kernel_func calls
        # with explicit syncs in between flush that path before capture.
        try:
            if torch.cuda.is_available():
                for _ in range(5):
                    warmup_output = kernel_func()
                    torch.cuda.synchronize()
                    if (
                        _skip_indexer
                        and _skip_state["prev_topk"] is None
                        and isinstance(warmup_output, tuple)
                        and len(warmup_output) == 2
                        and warmup_output[1] is not None
                    ):
                        _skip_state["prev_topk"] = warmup_output[1].detach()
                        attention_module.next_skip_topk = original_next_skip_topk
        finally:
            if _skip_indexer:
                attention_module.next_skip_topk = original_next_skip_topk

        point_clock.mark("prewarm")
        print(
            f"  Decode module CUDA graph: {use_benchmark_cuda_graph} "
            f"(real_tokens={batch_size}, graph_tokens={execution_batch_size})"
        )

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
            use_cuda_graph=use_benchmark_cuda_graph,
        ) as results:
            pass
        point_clock.mark("bench")
        point_clock.absorb(results.get("phase_seconds"), prefix="bench_")

        if _skip_indexer and _skip_state["prev_topk"] is None:
            raise RuntimeError(
                f"skip_indexer pass for {attn_type} captured no topk index during warmup; "
                "refusing to record a skip row with full-indexer latency."
            )

        avg_time_ms = results["latency_ms"]
        power_stats = results["power_stats"]

        # Log perf — wideep MLA uses isl=seq_len, step=0 (old convention).
        # DSA uses isl=1, step=seq_len, where seq_len is past-KV length.
        # The wideep generation loader computes s = isl + step, so both
        # conventions yield the same effective key when step=0 → s=seq_len.
        try:
            if is_wideep_mla:
                perf_fname = "wideep_generation_mla_perf.txt"
                op_name = "mla_generation"
                kernel_source = backend_name
                log_isl = seq_length
                log_step = 0
            else:
                _skip_sfx = "_skip_indexer" if _dsa_skip_indexer_enabled(attn_type, model_path) else ""
                # full + skip share ONE perf file; op_name (with _skip_indexer) tags skip.
                perf_fname = f"{attn_type}_generation_module_perf.txt"
                op_name = f"{attn_type}_generation_module{_skip_sfx}"
                kernel_source = executed_dsa_source or f"{attn_type}_{backend_name}"
                log_isl = 1
                log_step = seq_length
            perf_filename = _resolve_perf_path(output_path, perf_fname)
            if not log_perf(
                item_list=[
                    {
                        "model": model_path,
                        "architecture": architecture,
                        "mla_dtype": log_mla_dtype,
                        "kv_cache_dtype": log_kv_dtype,
                        "gemm_type": log_gemm_type,
                        "num_heads": head_num,
                        "batch_size": batch_size,
                        "isl": log_isl,
                        "tp_size": target_tp_size,
                        "step": log_step,
                        "latency": f"{avg_time_ms:.4f}",
                    }
                ],
                framework="SGLang",
                version=version,
                device_name=device_name,
                op_name=op_name,
                kernel_source=kernel_source,
                perf_filename=perf_filename,
                power_stats=power_stats,
            ):
                raise PerfLogWriteError(f"failed to persist decode row to {perf_filename}")
        except PerfLogWriteError:
            raise
        except Exception as e:
            print(f"  Warning: failed to log decode metrics: {e}")
            return False

        print(f"  Decode: {avg_time_ms:.3f} ms")
        return True

    except PerfLogWriteError:
        raise
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).lower()
        if "out of memory" in error_str:
            print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
            torch.cuda.empty_cache()
            return False
        if "cuda" in error_str and "illegal" in error_str:
            print("  CUDA illegal access detected — stopping this subprocess to preserve prior rows")
            raise CudaIllegalAccessError(f"CUDA illegal access at b={batch_size}, s={seq_length}") from e
        print("  Skipping this configuration...")
        return False
    finally:
        cleanup_errors = []
        for cleanup_name, cleanup_fn in (
            ("req_to_token_pool.clear", model_runner.req_to_token_pool.clear),
            ("token_to_kv_pool_allocator.clear", model_runner.token_to_kv_pool_allocator.clear),
            ("torch.cuda.empty_cache", torch.cuda.empty_cache),
        ):
            try:
                cleanup_fn()
            except Exception as cleanup_exc:
                cleanup_errors.append(f"{cleanup_name}: {type(cleanup_exc).__name__}: {cleanup_exc}")
        if cleanup_errors:
            raise RuntimeError(f"DSA/MLA decode cleanup failed: {'; '.join(cleanup_errors)}")
        point_clock.mark("cleanup")
        point_clock.add_to(_POINT_PHASE_TOTALS)
        print(f"  [DSA point timing] decode b={batch_size}, kv={seq_length}: {point_clock.render()}")


def _resolve_perf_path(output_path: str | None, filename: str) -> str:
    """Resolve the full path for a perf output file."""
    if output_path is not None:
        return os.path.join(output_path, filename)
    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(collector_dir, filename)


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════

_KERNEL_LIMIT = "FIXME(kernel-limit)"


def _over_chunk(case: tuple, chunk: int) -> str | None:
    """Why serving would not run this prefill point as one forward on this device, or None."""
    batch_size, seq_len, _is_prefill, _prefix_len = case
    tokens = required_prefill_extend_tokens(batch_size, seq_len)
    if tokens <= chunk:
        return None
    return (
        f"{_KERNEL_LIMIT}: does not fit: {tokens} new tokens exceed the chunked_prefill_size={chunk} "
        "SGLang uses on this device, and serving chunks this prefill"
    )


def _over_kv_pool(case: tuple, page_size: int, capacity: int) -> str | None:
    """Why this point's KV cache cannot fit SGLang's KV pool on this device, or None."""
    batch_size, seq_len, is_prefill, prefix_len = case
    needed = required_kv_alloc_tokens(batch_size, seq_len, prefix_len, page_size, is_prefill=is_prefill)
    if needed <= capacity:
        return None
    return f"{_KERNEL_LIMIT}: does not fit: the point needs {needed} KV tokens, the KV pool holds {capacity}"


def _refuse(cases: list, refused: dict, reason_for) -> None:
    """Record the reason of every case not yet refused that ``reason_for`` refuses."""
    for case in cases:
        if case not in refused:
            reason = reason_for(case)
            if reason is not None:
                refused[case] = reason


def _refusal_summary(refused: dict, total: int) -> str:
    """One classified line for the points this device refused (the subprocess's last error line)."""
    shown = "; ".join(
        f"b={b} s={s} prefix={p}: {reason.removeprefix(f'{_KERNEL_LIMIT}: ')}"
        for (b, s, _ip, p), reason in list(refused.items())[:3]
    )
    more = f"; and {len(refused) - 3} more" if len(refused) > 3 else ""
    return f"{_KERNEL_LIMIT}: {len(refused)}/{total} point(s) refused: {shown}{more}"


def run_mla_module(
    attn_type: str,
    head_num: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
    attention_backend: str | None = None,
    target_tp_size: int = 1,
    dsa_prefill_backend: str | None = None,
    skip_indexer: bool = False,
    ordinary_mla: bool = False,
    chunked_prefill_size: int | None = None,
    points=(),
):
    """Run MLA/DSA module benchmark — called inside a subprocess.

    Sets up the model runner for the given configuration and runs every point
    for the specified phase: a DSA case's (batch_size, seq_length, prefix_len)
    ``points`` from its plan row, or the MLA sweep of the precision combo.

    ``skip_indexer`` selects the GLM-5.2 reuse-layer variant (the per-layer
    indexer is patched out and rows are tagged ``_skip_indexer``). The worker
    receives it from the registry entry's run_kwargs and passes it down.
    Recorded in the ``_SKIP_INDEXER_PASS`` process global so the
    existing ``_dsa_skip_indexer_enabled`` call sites need no signature change.
    """
    global _SKIP_INDEXER_PASS
    if chunked_prefill_size is not None and chunked_prefill_size <= 0:
        raise ValueError("chunked_prefill_size must be positive for one-shot module collection")
    _SKIP_INDEXER_PASS = skip_indexer
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    if attention_backend is None:
        attention_backend = _get_backends(attn_type)
    phase_name = "Context" if is_prefill else "Generation"

    # run_attention_torch takes (batch_size, seq_length, is_prefill, prefix_len) cases.
    if attn_type == "dsa":
        if not points:
            raise RuntimeError(
                f"DSA module {phase_name.lower()} has no points: its plan row lists them; "
                f"model={model_path}, heads={head_num}, kv={kv_cache_dtype}, gemm={gemm_type}"
            )
        cases = [(int(b), int(s), is_prefill, int(p)) for b, s, p in points]
    else:
        all_cases = get_context_test_cases(attn_type) if is_prefill else get_generation_test_cases(attn_type)
        # Test case format: [seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type]
        cases = [
            (tc[1], tc[0], is_prefill, 0)
            for tc in all_cases
            if tc[3] == kv_cache_dtype and tc[4] == compute_dtype and tc[5] == gemm_type and tc[2] == head_num
        ]
    print(f"\n{'=' * 60}")
    print(
        f"{attn_type.upper()} Module {phase_name}: model={model_path}, backend={attention_backend}, "
        f"head_num={head_num}, target_tp={target_tp_size}, kv={kv_cache_dtype}, "
        f"compute={compute_dtype}, gemm={gemm_type}, GPU={gpu_id}"
    )
    print(f"Test cases: {len(cases)}")
    print(f"{'=' * 60}")

    use_module_cuda_graph = False
    enable_runner_piecewise_cuda_graph = False

    if not cases:
        raise RuntimeError(
            f"{attn_type.upper()} module {phase_name.lower()} has no runnable cases; "
            f"model={model_path}, heads={head_num}, kv={kv_cache_dtype}, gemm={gemm_type}"
        )

    # A DSA point this device cannot run is refused with its reason before anything is
    # allocated for it, and the ModelRunner is sized for the points that remain.
    refused: dict[tuple, str] = {}
    if attn_type == "dsa" and is_prefill:
        # SGLang derives its prefill chunk from device memory (the same tiering as
        # ServerArgs._handle_gpu_memory_settings); serving chunks a longer prefill.
        try:
            from collector.sglang.deepseekv4_sparse_modules import _sglang_chunked_prefill_size
        except ModuleNotFoundError:
            from deepseekv4_sparse_modules import _sglang_chunked_prefill_size
        chunk = _sglang_chunked_prefill_size() if chunked_prefill_size is None else chunked_prefill_size
        _refuse(cases, refused, lambda case: _over_chunk(case, chunk))
    runnable = [case for case in cases if case not in refused]
    if not runnable:
        raise RuntimeError(_refusal_summary(refused, len(cases)))

    max_total_tokens = max(
        (
            required_kv_alloc_tokens(bs, seq_len, prefix_len, SGLANG_DSA_PAGE_SIZE, is_prefill=ip)
            if attn_type == "dsa"
            else required_kv_tokens(bs, seq_len, prefix_len, is_prefill=ip)
        )
        for (bs, seq_len, ip, prefix_len) in runnable
    )
    if max_total_tokens > 0:
        max_total_tokens += max(1024, max_total_tokens // 20)

    if use_module_cuda_graph:
        print("[DSA] Module CUDA graph capture enabled for GLM-5 DSA piecewise parity")
    if enable_runner_piecewise_cuda_graph:
        print("[DSA] SGLang piecewise CUDA graph enabled (tokens=SGLang default)")
    print(f"[DSA] SGLang max_total_tokens capped for collector at {max_total_tokens}")

    cleanup_distributed()
    torch.cuda.empty_cache()

    try:
        model_runner = load_model_runner(
            model_path=model_path,
            head_num=head_num,
            kv_cache_dtype=kv_cache_dtype,
            attention_backend=attention_backend,
            dsa_prefill_backend=dsa_prefill_backend,
            device=device,
            gemm_type=gemm_type,
            target_tp_size=target_tp_size,
            enable_piecewise_cuda_graph=enable_runner_piecewise_cuda_graph,
            max_total_tokens=max_total_tokens,
            chunked_prefill_size=chunked_prefill_size,
            max_running_requests=request_slots(bs for bs, _seq_len, _ip, _prefix_len in runnable),
        )

        if attn_type == "dsa":
            if is_prefill:
                runtime_chunk = _runtime_chunk_size(model_runner)
                print(f"[DSA] SGLang runtime chunked_prefill_size={runtime_chunk}")
                _refuse(cases, refused, lambda case: _over_chunk(case, runtime_chunk))
            kv_capacity = _kv_pool_capacity_tokens(model_runner)
            print(f"[DSA] SGLang KV pool holds {kv_capacity} tokens")
            if kv_capacity is not None:
                page_size = kv_pool_page_size(model_runner)
                _refuse(cases, refused, lambda case: _over_kv_pool(case, page_size, kv_capacity))
            if is_prefill:
                if not sglang_dsa_mqa_logits_chunking_supported():
                    raise RuntimeError(
                        "FIXME(kernel-limit): this SGLang build's DSA indexer does not chunk MQA logits, "
                        "so a prefill workspace has no bound; the collector relies on that chunking"
                    )
                print("[DSA] SGLang MQA logits chunk path detected; no heuristic workspace pre-filter")
            runnable = [case for case in cases if case not in refused]
            if not runnable:
                raise RuntimeError(_refusal_summary(refused, len(cases)))

        logged_count = run_attention_torch(
            model_runner=model_runner,
            test_cases=runnable,
            head_num=head_num,
            test_layer=0,
            num_warmup=8,
            num_iterations=10,
            device=device,
            output_path=output_path,
            attn_type=attn_type,
            model_path=model_path,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype=compute_dtype,
            gemm_type=gemm_type,
            target_tp_size=target_tp_size,
            use_module_cuda_graph=use_module_cuda_graph,
            dsa_prefill_backend=dsa_prefill_backend,
            ordinary_mla=ordinary_mla,
        )
        if logged_count == 0:
            raise RuntimeError(
                f"{attn_type.upper()} module {phase_name.lower()} persisted no rows; "
                f"model={model_path}, heads={head_num}, kv={kv_cache_dtype}, gemm={gemm_type}"
            )
        error_count = len(runnable) - logged_count
        summary = f"ok={logged_count} error={error_count} refused={len(refused)} total={len(cases)}"
        print(f"[{attn_type.upper()}] {phase_name.lower()} {summary}")
        print(f"[{attn_type.upper()} timing] {phase_name.lower()}: {render_phase_totals(_POINT_PHASE_TOTALS)}")
        if error_count > 0:
            raise RuntimeError(
                f"{attn_type.upper()} module {phase_name.lower()} failed strict completeness: {summary}; "
                f"model={model_path}, heads={head_num}, kv={kv_cache_dtype}, gemm={gemm_type}"
            )
        if refused:
            raise RuntimeError(_refusal_summary(refused, len(cases)))
    finally:
        cleanup_distributed()
        torch.cuda.empty_cache()
        gc.collect()


def _run_mla_subprocess(
    attn_type: str,
    head_num: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
    attention_backend: str | None = None,
    target_tp_size: int = 1,
    dsa_prefill_backend: str | None = None,
    skip_indexer: bool = False,
    ordinary_mla: bool = False,
    chunked_prefill_size: int | None = None,
    points=(),
):
    """Run MLA/DSA benchmark in a subprocess with CUDA_VISIBLE_DEVICES isolation."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    phase = "context" if is_prefill else "generation"
    output_repr = f'"{output_path}"' if output_path else "None"
    backend_repr = f'"{attention_backend}"' if attention_backend else "None"
    dsa_backend_repr = f'"{dsa_prefill_backend}"' if dsa_prefill_backend else "None"
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_mla_module import run_mla_module\n"
        f'run_mla_module("{attn_type}", {head_num}, "{model_path}", '
        f'"{kv_cache_dtype}", "{compute_dtype}", "{gemm_type}", {is_prefill}, '
        f"0, {output_repr}, {backend_repr}, {target_tp_size}, "
        f"{dsa_backend_repr}, skip_indexer={skip_indexer}, ordinary_mla={ordinary_mla}, "
        f"chunked_prefill_size={chunked_prefill_size!r}, points={[list(point) for point in points]!r})\n"
    )

    # Per-subprocess timeout is OFF by default. A DSA context subprocess times
    # every (isl, prefix) point of its row for one batch_size and can legitimately
    # run for many minutes; a wall-clock cap silently truncated the sweep (killed mid-grid,
    # kept the partial rows, logged a WARNING not an error) and left the perf DB
    # with short prefix coverage, forcing AIC to extrapolate. Unset = block until
    # the subprocess finishes on its own; the executor's stall watchdog still sees
    # liveness, because the wait beats while the child prints its points, and a
    # reclaimed worker takes the child down with it (helper.run_measurement_subprocess).
    # Users can still opt into a cap by setting AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC.
    _timeout_env = os.environ.get("AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC")
    subprocess_timeout = int(_timeout_env) if _timeout_env else None
    try:
        returncode, stdout = run_measurement_subprocess(
            [sys.executable, "-c", code],
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=subprocess_timeout,
        )
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as exc:
        if exc.output:
            print(exc.output.decode("utf-8", errors="replace"))
        raise RuntimeError(
            f"{attn_type.upper()} module {phase} subprocess timed out "
            f"after {subprocess_timeout}s "
            f"(heads={head_num}, model={model_path}, kv={kv_cache_dtype}, "
            f"gemm={gemm_type}, target_tp={target_tp_size})"
        ) from exc

    if returncode != 0:
        # Include last lines of subprocess output in the error so the cause
        # is visible in the error group (subprocess stderr is merged into stdout).
        tail = ""
        lines = []
        if stdout:
            lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
            tail = "\n".join(lines[-30:])  # last 30 lines should include traceback
        # Points the device refused end the subprocess on one classified exception
        # line, the first unindented line of its last traceback (output printed at
        # interpreter exit may follow it); keep its kernel-limit classification.
        # torch.distributed prefixes each line of a rank's output with "[rankN]: ".
        plain = [line.split("]: ", 1)[1] if line.startswith("[rank") and "]: " in line else line for line in lines]
        tracebacks = [index for index, line in enumerate(plain) if line.startswith("Traceback ")]
        error = (
            next((line for line in plain[tracebacks[-1] + 1 :] if not line.startswith(" ")), "") if tracebacks else ""
        )
        if error.startswith(f"RuntimeError: {_KERNEL_LIMIT}"):
            raise RuntimeError(error.removeprefix("RuntimeError: "))
        raise RuntimeError(
            f"{attn_type.upper()} module {phase} subprocess failed (exit code {returncode})\n"
            f"--- subprocess output (last 30 lines) ---\n{tail}"
        )


def run_mla_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    attention_backend: str | None = None,
    target_tp_size: int = 1,
    dsa_prefill_backend: str | None = None,
    points=(),
    *,
    perf_filename: str,
    device: str = "cuda:0",
    chunked_prefill_size: int | None = None,
    skip_indexer: bool = False,
):
    """Worker-compatible wrapper used by collector/collect.py.

    Each call runs every point of one (attn_type, num_heads, precision, model)
    combo in one subprocess (one ModelRunner). A DSA case lists its
    [batch_size, seq_len, prefix_len] ``points`` (the op plan's row; a context
    row holds one batch_size). A wideep MLA case sweeps its own grid, and
    batch_size and seq_len are placeholders.

    For wideep MLA test cases, attention_backend is the 9th positional
    element specifying which backend to benchmark (e.g. "flashinfer", "fa3").
    For DSA test cases, it defaults to None and _get_backends() is used.

    perf_filename and device are keyword-only arguments supplied by
    collect.py via functools.partial and the worker dispatch loop;
    skip_indexer is bound the same way from the registry entry's run_kwargs
    (dsa_*_module_skip_indexer ops), so the full and skip ops share one
    table and one run_func and differ only in this explicit argument.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    is_prefill = "context" in perf_filename

    # skip_indexer ops route through the SAME worker/run path as the full DSA
    # module; the only difference is (1) the per-layer indexer (mqa+topk) is
    # patched out via skip_topk in _run_prefill/_run_generation and (2) the rows
    # are tagged with an op_name "_skip_indexer" suffix in the same perf file.
    # Threaded down to the subprocess as an explicit arg; no env var crosses
    # the subprocess boundary and nothing is inferred from the table name.
    skip_indexer = bool(skip_indexer)

    print(f"\n{'=' * 60}")
    print(
        f"{attn_type.upper()} Module {'Context' if is_prefill else 'Generation'}: "
        f"model={model_path}, heads={num_heads}, target_tp={target_tp_size}, kv={kv_cache_dtype}, "
        f"compute={compute_dtype}, gemm={gemm_type}, "
        f"backend={attention_backend or 'auto'}, batch={batch_size or 'all'}, points={len(points)}, GPU={gpu_id}"
    )
    print(f"{'=' * 60}")

    # Resolve output directory for perf data files.  When collect.py provides
    # a bare filename (e.g. "dsa_context_module_perf.txt"), write to CWD so
    # the perf files land next to vllm's output — matching artifact collection.
    output_path = os.path.dirname(perf_filename) or os.getcwd()

    # The case builder records configured backend buckets, not every leaf
    # kernel selected by an individual shape: flashmla_kv on Hopper and
    # trtllm on validated SM100. SM103 mirrors an upstream selector whose
    # bundled TRTLLM-GEN kernel is unavailable; registry maturity markers park
    # SM103/SM120. BF16 leaves the sub-backend unset so SGLang applies its
    # serving default.
    _run_mla_subprocess(
        attn_type=attn_type,
        head_num=num_heads,
        model_path=model_path,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        is_prefill=is_prefill,
        gpu_id=gpu_id,
        output_path=output_path,
        attention_backend=attention_backend,
        target_tp_size=target_tp_size,
        dsa_prefill_backend=dsa_prefill_backend,
        skip_indexer=skip_indexer,
        ordinary_mla=os.path.basename(perf_filename) == "mla_context_module_perf.txt",
        chunked_prefill_size=chunked_prefill_size,
        points=points,
    )


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _run_dsa_plan_rows(args, gpu_id: int) -> int:
    """Run the DSA op plan's rows on this SM (--model targets that checkpoint, as
    collect.py --model-path does); return how many failed."""
    if args.model:
        os.environ["COLLECTOR_MODEL_PATH"] = args.model
    failed = 0
    dispatched = 0
    for row in plan(f"dsa_{args.mode}_module", "sglang", get_sm_version()).rows:
        _seq, _batch, heads, kv_dtype, compute, gemm, model_path, _attn, _backend, target_tp, pinned, points = row
        if args.num_heads is not None and heads != args.num_heads:
            continue
        if args.target_tp_size is not None and target_tp != args.target_tp_size:
            continue
        if args.kv_cache_dtype and kv_dtype != args.kv_cache_dtype:
            continue
        dispatched += 1
        try:
            run_mla_module(
                attn_type="dsa",
                head_num=heads,
                model_path=model_path,
                kv_cache_dtype=kv_dtype,
                compute_dtype=compute,
                gemm_type=gemm,
                is_prefill=args.mode == "context",
                gpu_id=gpu_id,
                output_path=args.output_path,
                target_tp_size=target_tp,
                dsa_prefill_backend=pinned,
                chunked_prefill_size=args.chunked_prefill_size,
                points=points,
            )
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            traceback.print_exc()
    if dispatched == 0:
        raise RuntimeError(
            "no DSA plan row matches --num-heads/--target-tp-size/--kv-cache-dtype/--model on this SM; nothing ran"
        )
    return failed


def main():
    parser = argparse.ArgumentParser(description="SGLang MLA/DSA Module Benchmark")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--attn-type", choices=["mla", "dsa"], default=None, help="If not set, runs both")
    parser.add_argument(
        "--ordinary-mla",
        action="store_true",
        help="Collect the complete MLA context module, including down-projection, using its ordinary table",
    )
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model path")
    parser.add_argument("--num-heads", type=int, default=None, help="Filter by head count")
    parser.add_argument("--target-tp-size", type=int, default=None, help="Serving TP size; derives per-rank heads")
    parser.add_argument("--kv-cache-dtype", choices=["bfloat16", "fp8"], default=None)
    parser.add_argument("--chunked-prefill-size", type=int, default=None, help="Serving prefill chunk limit in tokens")
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()
    if args.ordinary_mla and (args.mode != "context" or args.attn_type != "mla"):
        parser.error("--ordinary-mla requires --mode context --attn-type mla")
    if args.target_tp_size is not None and args.target_tp_size <= 0:
        parser.error("--target-tp-size must be positive")
    if args.num_heads is not None and args.num_heads <= 0:
        parser.error("--num-heads must be positive")

    # Determine which attn_types to run
    if args.attn_type:
        attn_types = [args.attn_type]
    else:
        attn_types = sorted({spec.attention_type for spec in get_mla_module_model_specs(apply_model_filter=False)})

    failed_dispatches = 0
    gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    for attn_type in attn_types:
        if attn_type == "dsa":
            failed_dispatches += _run_dsa_plan_rows(args, gpu_id)
            continue
        # Determine models
        if args.model:
            models = [args.model]
        else:
            models = [
                spec.model_path
                for spec in get_mla_module_model_specs(attention_type=attn_type, apply_model_filter=False)
            ]

        for model_path in models:
            print(f"\n{'=' * 60}")
            print(f"Model: {model_path}  |  Attention: {attn_type.upper()}  |  Mode: {args.mode}")
            print(f"{'=' * 60}")

            native_heads = _module_model_native_heads(model_path)
            head_nums = (
                [args.num_heads]
                if args.num_heads
                else [h for h in get_mla_module_sweep_spec("sglang").inner_sweep_head_counts if h <= native_heads]
            )
            if args.target_tp_size is not None:
                if native_heads % args.target_tp_size:
                    parser.error("--target-tp-size must divide the model's native head count")
                local_heads = native_heads // args.target_tp_size
                if args.num_heads is not None and args.num_heads != local_heads:
                    parser.error("--num-heads must equal native heads / --target-tp-size")
                head_nums = [local_heads]
            if args.ordinary_mla and any(native_heads % heads for heads in head_nums):
                parser.error("ordinary MLA per-rank heads must divide the model's native head count")

            for compute_dtype, kv_dtype, gemm_type in _get_precision_combos(args.mode):
                if args.kv_cache_dtype and kv_dtype != args.kv_cache_dtype:
                    continue

                for head_num in head_nums:
                    target_tp_size = args.target_tp_size or (native_heads // head_num if args.ordinary_mla else 1)
                    is_prefill = args.mode == "context"
                    try:
                        run_mla_module(
                            attn_type=attn_type,
                            head_num=head_num,
                            model_path=model_path,
                            kv_cache_dtype=kv_dtype,
                            compute_dtype=compute_dtype,
                            gemm_type=gemm_type,
                            is_prefill=is_prefill,
                            gpu_id=gpu_id,
                            output_path=args.output_path,
                            target_tp_size=target_tp_size,
                            ordinary_mla=args.ordinary_mla,
                            chunked_prefill_size=args.chunked_prefill_size,
                        )
                    except Exception as e:
                        failed_dispatches += 1
                        print(f"  FAILED: {e}")
                        traceback.print_exc()

    print(f"\n{'=' * 50}")
    if failed_dispatches:
        print(f"COLLECTION FAILED: {failed_dispatches} dispatches failed")
        raise SystemExit(1)
    print("ALL TESTS COMPLETED")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
