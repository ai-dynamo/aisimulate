# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from vLLM dd10e03f95f94edbea1975c67ace3a35ec9a8a40.
# Copyright contributors to the vLLM project. Modified for collector execution.

# Versions are upgraded IN PLACE: this file always
# targets the manifest pin; prior-version code is `git log -- <this file>`
# away and prior-version DATA stays permanent under its version key. Every
# vLLM citation below without a version is @v0.32.0 (paths under vllm/,
# model paths under vllm/models/minimax_m3/).
__compat__ = "vllm==0.32.0"

"""
MSA Module Collector for vLLM — MiniMax-M3 sparse-attention benchmarking.

Profiles the complete MiniMax-M3 sparse-attention module forward pass
(fused qkv+index projection, fused per-head Gemma QK norm + partial RoPE +
paged KV/index-cache insert, lightning-indexer top-k block selection,
block-sparse GQA attend, output projection), not a bare kernel. Uses vLLM's
own modeling code (``MiniMaxM3SparseAttention``, the exact module
``MiniMaxM3DecoderLayer`` builds for sparse layers — nvidia/model.py:712-720)
with dummy weights, the same framework-builder approach as
``collect_mla_module.py``.

Kernel dispatch is the framework's own: the attend backend and impl are chosen
by ``select_main_backend_and_impl_cls`` (Triton off the SM100 family; the
fmha_sm100 "MSA" attend on it for top-k 4/8/16/32 and a non-e5m2 KV cache —
common/sparse_attention.py:525-587) and the indexer impl by
``select_indexer_impl_cls`` (fmha_sm100 score + top-k on the SM100 family at
top-k 16, Triton elsewhere — common/indexer.py:495-550), both at module
construction (nvidia/model.py:518-550). ``kernel_source`` records the
actually-selected attend impl class. Per-SM validation state lives in the
registry's ``unverified_sms``.

Supported models and micro-sweeps come from collector v2 YAML
(``cases/models/MiniMaxM3ForCausalLM_cases.yaml`` `mla_module` rows with
``attention_type: msa`` + ``cases/base_ops/mla_module.yaml`` vllm override).

Usage:
    python collect_msa_module.py --mode context --model MiniMaxAI/MiniMax-M3
    python collect_msa_module.py --mode generation --quick --batch-size 4 --seq-len 2048 --num-heads 64
    python collect_msa_module.py --mode context --quick --kv-cache-dtype fp8
"""

import argparse
import gc
import json
import os
import sys
import traceback

import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context

# ═══════════════════════════════════════════════════════════════════════
# Config registry shim — the MiniMax-M3 HF release config carries
# model_type "minimax_m3", but vLLM's _CONFIG_REGISTRY only maps
# "minimax_m3_vl" / "minimax_m3_mtp" (transformers_utils/config.py:114-115,
# resolved lazily by LazyConfigDict :71-78)
# and transformers doesn't know the model_type either, so the
# AutoConfig fallback fails offline. Serving loads such checkpoints via
# trust_remote_code; the bundled config dir has auto_map stripped (see
# helper._resolve_local_model_path), so route the model_type to vLLM's own
# text-backbone config class ``MiniMaxM3TextConfig``
# (transformers_utils/configs/minimax_m3.py:8-105), whose fields
# 1:1 mirror the checkpoint config (every bundled-config key is a named
# __init__ arg or **kwargs passthrough). This changes HOW the config
# object is constructed, never which kernel runs — sparse dispatch keys on
# sparse_attention_config / head geometry, preserved verbatim from JSON.
# ═══════════════════════════════════════════════════════════════════════
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.version import __version__ as vllm_version

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collector.case_generator import get_mla_module_model_specs
from collector.helper import _resolve_local_model_path, benchmark_with_power, get_sm_version, log_perf
from collector.op_plans import plan, plan_cases
from collector.registry_types import PerfFile
from collector.vllm.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
    enable_engine_fused_ops,
    setup_distributed,
    with_exit_stack,
)

if "minimax_m3" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["minimax_m3"] = "MiniMaxM3TextConfig"


# ═══════════════════════════════════════════════════════════════════════
# Test Cases — aligned with TRT-LLM's collect_msa_module.py
# ═══════════════════════════════════════════════════════════════════════


# Native MiniMax-M3 GQA ratio: 64 q heads / 4 kv heads (bundled config
# num_attention_heads / num_key_value_heads). Used by the TP-shard emulation
# and to refuse generation points whose KV cache cannot fit the device.
_NATIVE_GQA_RATIO = 16
_M3_HEAD_DIM = 128
_M3_INDEX_DIM = 128

_MEMORY_BUDGET_SAFETY_FACTOR = 0.9

# Values drawn per host sample, which bounds the host temporary.
_SAMPLE_CHUNK_ELEMENTS = 1 << 20
# Distinct history tokens: a larger index cache repeats them, so writing it costs one
# copy per this many tokens and no further sampling. A request scores only its own
# blocks, so a block's keys recur within a row only for a request longer than this:
# two to four times at the plan's 524288- and 1048575-token histories, against the
# thousands of candidate blocks of such a row.
_HISTORY_TOKENS = 1 << 18
# The worker's sampled history keys by (device, block size, key width).
_HISTORY_KEYS: dict = {}


def _emulated_kv_heads(num_heads: int) -> int:
    return max(1, num_heads // _NATIVE_GQA_RATIO)


def _generation_kv_footprint_bytes(total_tokens: int, num_heads: int, kv_cache_dtype: str) -> int:
    """Lower bound of a generation case's peak device footprint.

    run_msa_module allocates the paged main K/V cache (2 x kv_heads x 128
    elems/token; 2 B/elem for bf16, 1 B/elem for the fp8 cache's uint8
    storage — STR_DTYPE_TO_TORCH_DTYPE["fp8"], utils/torch_utils.py:39)
    plus the indexer's key side-cache (128 elems/token, always
    bf16). Module weights, metadata, and the top-k buffer are deliberately
    excluded so the estimate stays a provable lower bound: a case refused on
    it cannot fit on the device, on any platform.
    """
    kv_heads = _emulated_kv_heads(num_heads)
    main_bytes_per_elem = 1 if kv_cache_dtype == "fp8" else 2
    entry_bytes = 2 * kv_heads * _M3_HEAD_DIM * main_bytes_per_elem + _M3_INDEX_DIM * 2
    return total_tokens * entry_bytes


def _kv_does_not_fit(total_tokens: int, num_heads: int, kv_cache_dtype: str, total_memory: int) -> str | None:
    """Why a generation point's KV cache cannot fit the device, or None when it can.

    The module plans do not depend on the device, so the run refuses such a
    point before allocating it.
    """
    needed = _generation_kv_footprint_bytes(total_tokens, num_heads, kv_cache_dtype)
    allowed = int(total_memory * _MEMORY_BUDGET_SAFETY_FACTOR)
    if needed <= allowed:
        return None
    return (
        f"FIXME(kernel-limit): does not fit: the KV cache needs {needed / 2**30:.1f} GiB, "
        f"the device allows {allowed / 2**30:.1f} GiB ({_MEMORY_BUDGET_SAFETY_FACTOR:g} of its memory)"
    )


def get_msa_context_module_test_cases():
    return plan_cases("msa_context_module", "vllm", get_sm_version())


def get_msa_generation_module_test_cases():
    return plan_cases("msa_generation_module", "vllm", get_sm_version())


# ═══════════════════════════════════════════════════════════════════════
# Module Construction
# ═══════════════════════════════════════════════════════════════════════


def _ceil_div(a, b):
    return (a + b - 1) // b


def _create_gemm_quant_config(gemm_type: str):
    """Create the vLLM QuantizationConfig for a given gemm_type.

    Returns None for bfloat16 (unquantised GEMMs). For fp8_block / nvfp4,
    returns a serialized-checkpoint config so dummy weights are created in
    the quantized layout and processed by ``process_weights_after_loading``
    — same rationale and citations as collect_mla_module.py.
    """
    if gemm_type == "bfloat16":
        return None
    if gemm_type == "fp8_block":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        # vLLM requires is_checkpoint_fp8_serialized=True for block-scaled
        # FP8 (fp8.py raises ValueError otherwise). This routes through
        # Fp8LinearMethod (block_quant=True) → W8A8BlockFp8LinearOp →
        # DeepGEMM on SM>=89.
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    if gemm_type == "nvfp4":
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        return ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
    raise ValueError(f"Unknown gemm_type: {gemm_type!r}")


_ATTN_LAYER_NAME = "model.layers.0.self_attn.attn"
_INDEX_CACHE_LAYER_NAME = "model.layers.0.self_attn.attn.index_cache"


def _move_to_device(module, device: str):
    """Move the module to the device with the rotary table its constructor computed.

    Module.to cannot copy meta parameters and to_empty reallocates every tensor, the
    buffers included, so a module moved with to_empty would hold an uninitialized
    cos_sin_cache (a non-persistent buffer, RotaryEmbeddingBase.__init__,
    model_executor/layers/rotary_embedding/base.py:58-63). The tables are held aside
    and put back.
    """
    if not any(p.is_meta for p in module.parameters()):
        return module.to(device)
    tables = [
        (owner, owner.cos_sin_cache)
        for owner in module.modules()
        if isinstance(getattr(owner, "cos_sin_cache", None), torch.Tensor)
    ]
    module = module.to_empty(device=torch.device(device))
    for owner, table in tables:
        owner.cos_sin_cache = table.to(device)
    return module


def _give_index_projection_values(attn_module) -> None:
    """Give the index query and key projections non-zero rows that differ between heads.

    The fused projection emits [q | k | v | index_q | index_k] (MinimaxM3QKVParallelLinear
    WithIndexer.output_sizes, model_executor/layers/linear.py:1416-1422), so the index
    branch is its last index_q_size + idx_head_dim output rows (MiniMaxM3SparseAttention
    .__init__, nvidia/model.py:437-455; one rank here, so the rows are the layer's own).
    Under the dummy fill those rows were one constant (bf16 GEMMs: every index head got
    the same query) or zero (FP8 and NVFP4 GEMMs: index query and key were zero), and
    every block scored the same. The block selection that follows depends on the scores
    (sparse_topk_select on the SM100 family, nvidia/indexer_msa.py:347-355).

    Seeded N(0, 0.02) values go into the rows of a bf16 or FP8 weight. A packed NVFP4
    weight holds two E2M1 values per byte with one E4M3 scale per 16 inputs (KNvfp4Static
    .create_weights, model_executor/layers/quantization/modelopt.py:1992-2035): every
    byte is a valid pair, so its rows take seeded bytes and a block scale of 1. The Gemma
    norms that follow bring either to unit magnitude (x * rsqrt(mean(x^2) + eps) *
    (1 + w), csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:135-146). The values
    stand for no checkpoint; no kernel is chosen by them. The q, k and v rows keep the
    dummy fill.
    """
    projection = attn_module.qkv_proj
    rows = attn_module.index_q_size + attn_module.idx_head_dim
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        weight = projection.weight[-rows:]
        if weight.dtype == torch.uint8:
            values = torch.randint(0, 256, weight.shape, generator=generator, dtype=torch.uint8, device="cpu")
            scale = projection.weight_scale[-rows:]
            ones = torch.ones(scale.shape, dtype=torch.float32, device="cpu")
            scale.copy_(ones.to(device=scale.device, dtype=scale.dtype))
        else:
            values = torch.randn(weight.shape, generator=generator, dtype=torch.float32, device="cpu") * 0.02
        weight.copy_(values.to(device=weight.device, dtype=weight.dtype))


def _create_msa_attention_module(
    model_path: str,
    num_heads: int,
    gemm_type: str,
    kv_cache_dtype: str,
    max_seq_len: int,
    max_batch_size: int,
    max_query_len: int,
    is_context: bool,
    device: str = "cuda:0",
):
    """Create a ``MiniMaxM3SparseAttention`` module from vLLM's own modeling
    code.

    ``max_seq_len`` is the case's KV length (cached prefix + new tokens) and
    ``max_query_len`` its per-request new-token count (context ``seq_len``;
    ``1`` for generation).

    Loads the real HF config from model_path, applies the TP-shard head
    emulation in-memory, and constructs the exact module serving builds for
    sparse layers (``MiniMaxM3DecoderLayer.__init__``,
    nvidia/model.py:712-720). Backend selection happens inside the
    module constructor exactly as in serving (select_main_backend_and_impl_cls /
    select_indexer_impl_cls); ``kv_cache_dtype`` must therefore be baked
    into cache_config before construction — "fp8" here models serving's
    global ``--kv-cache-dtype fp8``.
    """
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        # This collector benchmarks the NVIDIA path (models/minimax_m3/
        # __init__.py:16-27 routes ROCm to amd/model.py, a different
        # module with different kernels).
        raise RuntimeError("collect_msa_module benchmarks the NVIDIA MiniMax-M3 path; ROCm is unsupported")

    local_model_path = _resolve_local_model_path(model_path)

    with open(os.path.join(local_model_path, "config.json")) as f:
        cfg_dict = json.load(f)
    original_architecture = cfg_dict.get("architectures", [cfg_dict.get("model_type", "unknown")])[0]
    if not cfg_dict.get("sparse_attention_config"):
        raise ValueError(
            f"model {model_path} has no sparse_attention_config; the MSA module "
            "collector only supports MiniMax-M3 sparse-attention checkpoints"
        )

    native_q = int(cfg_dict["num_attention_heads"])
    native_kv = int(cfg_dict["num_key_value_heads"])
    gqa_ratio = max(1, native_q // native_kv)
    num_kv_heads = max(1, num_heads // gqa_ratio)

    max_model_len = max(max_seq_len, 4096)

    vllm_config = create_vllm_config(
        model_name=local_model_path,
        max_model_len=max_model_len,
        # No --block-size: serving derives the page from the layers' backends
        # after the model is built (_serving_kv_block_size). An explicit value
        # would mark the config user-specified and skip that derivation.
        block_size=None,
        # cache_config.num_gpu_blocks is not read on this path; an upper bound
        # at the smallest page.
        num_gpu_blocks=1 + _ceil_div(max_seq_len + 1, 16) * max_batch_size,
        max_num_seqs=max_batch_size,
        # The serving scheduler budgets NEW tokens per step, not cached ones
        # (token_budget = max_num_batched_tokens, debited by each request's
        # num_new_tokens: v1/core/sched/scheduler.py:137-141,611,856,1366).
        # Here the budget only sizes persistent buffers whose per-step views
        # the kernels read: the shared top-k buffer (nvidia/model.py:831-844)
        # and, on the SM100 indexer, a [max_num_batched_tokens, index heads,
        # 8192] fp32 score buffer (nvidia/indexer_msa.py:132-138), which cost
        # up to ~140 GB at a 1M-token prefix when the budget counted cached
        # tokens too. Keep the existing 131072-token floor for context steps.
        max_num_batched_tokens=max(max_batch_size * max_query_len, 131072) if is_context else max_batch_size,
        # Serving's global --kv-cache-dtype: use_fp8_kv_cache=True sets
        # cache_config.cache_dtype "fp8" (= fp8_e4m3 on CUDA), False leaves
        # "auto" (model dtype, bf16). The module reads it at construction
        # (self.kv_cache_dtype = cache_config.cache_dtype, nvidia/model.py:
        # 493-498) and selects the attend impl off it
        # (select_main_backend_and_impl_cls(kv_cache_dtype=...),
        # nvidia/model.py:518-534), so the dtype must be set before the module
        # is built for dispatch to match serving.
        use_fp8_kv_cache=(kv_cache_dtype == "fp8"),
        trust_remote_code=False,
        # The HF release config declares architectures
        # ["MiniMaxM3ForCausalLM"], which vLLM's model registry does
        # not know; the registered text-backbone architecture is
        # "MiniMaxM3SparseForCausalLM" (model_executor/models/registry.py
        # 163-166) — the same class vLLM's own VL wrapper routes
        # the text backbone through (nvidia/model.py:1171-1176). Map it via
        # hf_overrides, vLLM's first-class --hf-overrides mechanism
        # (config/model.py:605-676 applies it before arch resolution). The
        # architecture also auto-enables breakable CUDA graphs
        # (config/vllm.py:74-102, 812-840): see the timing note in
        # run_msa_module.
        hf_overrides={"architectures": ["MiniMaxM3SparseForCausalLM"]},
        # TP-shard emulation: shrink the head axes exactly like serving's TP
        # sharding would. Per-rank q heads = total/tp and kv heads =
        # max(1, total_kv/tp) (MiniMaxM3SparseAttention.__init__,
        # nvidia/model.py:419-433), so with tp_size=1 here we bake
        # the sharded counts into the config.
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    # Override quant_config to control linear-layer GEMM precision (the
    # BF16 checkpoint ships no quant config, so None means bf16 GEMMs).
    vllm_config.quant_config = _create_gemm_quant_config(gemm_type)

    hf_config = vllm_config.model_config.hf_text_config

    # Index-head TP emulation: index_q has the same head count as the KV
    # heads and shards identically — including replication when
    # tp > total_kv (num_idx_heads = num_kv_heads, nvidia/model.py:435-442;
    # MinimaxM3QKVParallelLinearWithIndexer asserts total_num_index_heads ==
    # total_num_kv_heads and rides the KV-head sharding/replication path,
    # model_executor/layers/linear.py:1352-1430; index_k is a single head
    # replicated to every rank). Natively sparse_num_index_heads (4) ==
    # num_key_value_heads (4), so the per-rank emulated counts are reproduced
    # by setting the config's index-head count to the emulated kv-head count.
    # The indexer metadata builder derives the same per-rank count from this
    # field (common/indexer.py:225-235), and the shared top-k buffer as well
    # (nvidia/model.py:831-844).
    sparse_cfg = dict(hf_config.sparse_attention_config)
    sparse_cfg["sparse_num_index_heads"] = num_kv_heads
    hf_config.sparse_attention_config = sparse_cfg

    # Reserved top-k indices buffer shared by the indexer and the attend —
    # mirrors MiniMaxM3Model.__init__ (nvidia/model.py:826-844): token-major
    # [tokens padded to a multiple of 4 for build_k2q_csr's int4 loads,
    # num_index_heads, sparse_topk_blocks], int32. The indexers write
    # buf[:num_tokens] (nvidia/indexer_msa.py:347-355; the Triton indexer
    # writes its transposed view, common/indexer.py:430-433).
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    topk_indices_buffer = torch.empty(
        *_msa_topk_buffer_shape(num_kv_heads, max_num_batched_tokens, sparse_cfg["sparse_topk_blocks"]),
        dtype=torch.int32,
        device=device,
    )

    # Build the module inside set_current_vllm_config():
    # MiniMaxM3SparseAttention.__init__ reads the current vllm config for
    # attention_config.indexer_kv_dtype and registers itself (and its
    # indexer side cache) in compilation_config.static_forward_context
    # (nvidia/model.py:491-557, common/indexer.py:136-139).
    from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
        attn_module = MiniMaxM3SparseAttention(
            config=hf_config,
            layer_id=0,
            quant_config=vllm_config.quant_config,
            prefix="model.layers.0.self_attn",
            cache_config=vllm_config.cache_config,
            topk_indices_buffer=topk_indices_buffer,
        )

    # Serialized quant configs create weight params on meta device; to()
    # cannot copy meta tensors, so use to_empty() when needed.
    attn_module = _move_to_device(attn_module, device)
    attn_module.eval()
    attn_module.requires_grad_(False)

    # Deterministic dummy weights: fp8/uint8 → 0, fp32 scales → 0.5, rest 0.01.
    # The GEMM, norm and attend kernels depend on shapes and dtypes, not values.
    # The indexer's block selection depends on the scores, so the RoPE table is
    # left as built and the index projections get values of their own.
    with torch.no_grad():
        for name, tensor in list(attn_module.named_parameters()) + list(attn_module.named_buffers()):
            if tensor.is_meta:
                continue
            if name.endswith("cos_sin_cache"):
                # The RoPE table is the constructor's (get_rope in
                # MiniMaxM3SparseAttention.__init__, nvidia/model.py:473-480), not
                # a weight. Filled with a constant it rotated every position alike,
                # so no index query or key depended on its position; get_rope also
                # hands the same module to every later build in the process
                # (_ROPE_DICT, model_executor/layers/rotary_embedding/__init__.py:
                # 84-85,351), so the fill was permanent.
                continue
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                tensor.data.zero_()
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.data.fill_(0.5)
            else:
                tensor.data.fill_(0.01)
    _give_index_projection_values(attn_module)

    # Process weights, mimicking vLLM's model loader (quantized-layout
    # conversion; no-op for unquantized linears).
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    with set_current_vllm_config(vllm_config):
        for _, module in attn_module.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module)

    return attn_module, vllm_config, original_architecture


# ═══════════════════════════════════════════════════════════════════════
# KV Cache + Metadata
# ═══════════════════════════════════════════════════════════════════════


def _serving_kv_block_size(vllm_config, attn_module) -> int:
    """The KV page serving runs the sparse layer and its index cache with.

    After loading the model, serving sizes the cache-manager block with the
    platform's update_block_size_for_backend (v1/executor/uniproc_executor.py:75,
    multiproc_executor.py:701): without --block-size, the smallest size every
    non-SSM attention backend accepts (platforms/interface.py:750-794,
    _preferred_block_size_for_backends :700-748). Each group's kernel page is
    select_common_block_size of that block, its backend and spec
    (v1/worker/utils.py:284-300). Both M3 sparse backends accept only 128
    (common/sparse_attention.py:162-165, common/indexer.py:93-95), and the
    dense layers' backend, which serving also counts, accepts 128 too. One
    block table serves both caches here, as serving's two groups share the
    block size.
    """
    from vllm.platforms import current_platform
    from vllm.v1.worker.utils import select_common_block_size

    current_platform.update_block_size_for_backend(vllm_config)
    pages = {
        int(
            select_common_block_size(
                vllm_config.cache_config.block_size, [layer.get_attn_backend()], [layer.get_kv_cache_spec(vllm_config)]
            )
        )
        for layer in (attn_module, attn_module.indexer.index_cache)
    }
    if len(pages) != 1:
        raise RuntimeError(f"MSA main and index caches resolved different KV pages {sorted(pages)}")
    return pages.pop()


def _serving_kv_cache_layout(vllm_config, attn_module):
    """The physical KV-cache layout serving resolves for MiniMax-M3.

    The engine resolves one layout for the whole model from every attention
    backend's supported set (v1/engine/core.py:296-300 ->
    resolve_kv_cache_layout, v1/attention/backends/utils.py:209-244,259-349).
    Besides the sparse layers' two backends (no declared set) that includes
    the dense layers 0-2, whose generic Attention picks its backend with
    get_attn_backend(head_dim, model dtype, kv_cache_dtype)
    (nvidia/model.py:355-362, model_executor/layers/attention/attention.py:
    348-360): FLASHINFER on the SM100 family declares LBHNC/BLHNC
    (v1/attention/backends/flashinfer.py:603-610), so the model resolves
    LBHNC there; elsewhere nothing declares, and the main and index caches'
    different head shapes keep only block-compact layouts of the default
    preference, so it resolves LBNHC (utils.py:195-202). The kernels read the
    cache through its strides (fused insert csrc/
    fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:1053-1123, Triton
    common/ops/sparse_attn.py).
    """
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
    from vllm.v1.attention.selector import get_attn_backend

    model_config = vllm_config.model_config
    index_cache = attn_module.indexer.index_cache
    with set_current_vllm_config(vllm_config):
        dense_backend = get_attn_backend(
            model_config.hf_text_config.head_dim, model_config.dtype, vllm_config.cache_config.cache_dtype
        )
    backends = [dense_backend, attn_module.get_attn_backend(), index_cache.get_attn_backend()]
    supported = [layout.name for layout in get_supported_kv_cache_layouts(backends)]
    specs = [attn_module.get_kv_cache_spec(vllm_config), index_cache.get_kv_cache_spec(vllm_config)]
    return resolve_kv_cache_layout(vllm_config, [supported], specs)


def _layer_kv_cache(spec, num_blocks: int, layout, device):
    """One layer's zeroed cache: the logical [B, H, N, C] view over memory in ``layout``.

    Serving views a flat byte buffer per layer as [B, H, N, C] with C in bytes
    (compute_layer_kv_cache_shape_bytes, v1/kv_cache_interface.py:319-335) in
    the resolved physical axis order, then as spec.dtype (create_kv_cache_views,
    :377-441). For a single layer that is this permuted allocation.
    """
    from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes

    logical = compute_layer_kv_cache_shape_bytes(spec, num_blocks)
    order = layout.layer_view_order
    physical = torch.zeros([logical[axis] for axis in order], dtype=torch.uint8, device=device)
    return physical.permute(*(order.index(axis) for axis in range(len(order)))).view(spec.dtype)


def _write_index_key_history(index_kv_cache) -> None:
    """Write the index keys of the cached history the indexer selects blocks from.

    _layer_kv_cache zero-fills the index cache and nothing here runs the forwards that
    would have written the history, so every history block scored exactly 0.0 and the
    block selection was timed on ties. Serving stores a token's index key when it
    computes it: the fused insert writes the normed, rotated index_k as one bf16 row at
    its slot, index_cache[slot * 128 + dim] (ops.fused_minimax_m3_qknorm_rope_kv_insert
    from MiniMaxM3SparseAttention.forward, nvidia/model.py:625-650; csrc/
    fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:753-760), and the indexers read the
    cache as those rows ([num_blocks, 128, 128], nvidia/indexer_msa.py:325, common/
    indexer.py:425-481). That kernel has no key-only entry, so seeded N(0, 1) rows, the
    magnitude the Gemma norm gives a key, fill the cache directly through its logical
    [num_blocks, 1, block_size, 128] view. The fill is outside the measurement and the
    timed forward writes its own tokens over it, as in serving. The rows stand for no
    particular checkpoint distribution. Slots past a request's length hold them too,
    where serving has whatever an earlier request left: at most one partial block per
    request, beyond every query's causal bound. Same treatment as the SGLang and
    TensorRT-LLM MSA collectors.

    Off the SM100 family the Triton indexer's top-k is a bitonic merge network
    (_topk_index_kernel, common/ops/index_topk.py:208-306), whose work does not depend
    on the scores: the fill changes no timing there.
    """
    num_blocks, _, block_size, index_dim = index_kv_cache.shape
    distinct = max(1, _HISTORY_TOKENS // block_size)
    # Sampled on the host once per worker and kept on the device: a point then pays one
    # device copy, where sampling it again would cost a point more than its measurement.
    memo = (str(index_kv_cache.device), block_size, index_dim)
    keys = _HISTORY_KEYS.get(memo)
    if keys is None:
        keys = torch.empty(
            (distinct, 1, block_size, index_dim), dtype=index_kv_cache.dtype, device=index_kv_cache.device
        )
        chunk = max(1, _SAMPLE_CHUNK_ELEMENTS // (block_size * index_dim))
        generator = torch.Generator(device="cpu").manual_seed(1234)
        for start in range(0, distinct, chunk):
            part = keys[start : start + chunk]
            part.copy_(torch.randn(part.shape, generator=generator, dtype=torch.float32, device="cpu"))
        _HISTORY_KEYS[memo] = keys
    for start in range(0, num_blocks, distinct):
        part = index_kv_cache[start : start + distinct]
        part.copy_(keys[: part.shape[0]])


def _rebase_block_table_and_slots(common_attn_metadata, block_size: int):
    """Reserve block 0 as the null block and derive the slot mapping from the
    block table.

    Serving computes each group's slot mapping from that group's block table
    as ``block_id * block_size + intra_block_offset`` for the tokens being
    written (_compute_slot_mappings_kernel, v1/worker/gpu/block_table.py:
    341-348), with block 0 taken as the null block (v1/core/block_pool.py:
    187). create_common_attn_metadata's arange path starts at block 0 and
    fills slot_mapping with a plain arange, so rebase both here.
    """
    block_table = common_attn_metadata.block_table_tensor
    block_table += 1

    query_start_loc = common_attn_metadata.query_start_loc_cpu
    # seq_lens - query_lens on device (CommonAttentionMetadata
    # .compute_num_computed_tokens, v1/attention/backend.py:486-491)
    context_lens = common_attn_metadata.compute_num_computed_tokens().cpu()
    slot_mapping = common_attn_metadata.slot_mapping
    device = slot_mapping.device
    for i in range(common_attn_metadata.num_reqs):
        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        token_offsets = torch.arange(end - start, dtype=torch.long, device=device) + int(context_lens[i])
        block_ids = block_table[i, token_offsets // block_size].to(torch.long)
        slot_mapping[start:end] = block_ids * block_size + token_offsets % block_size
    return int(block_table.max().item()) + 1  # blocks needed incl. null block


def _msa_topk_buffer_shape(num_heads: int, max_tokens: int, topk: int) -> tuple[int, int, int]:
    """Serving's allocation: token-major, tokens padded to a multiple of 4
    (MiniMaxM3Model.__init__, nvidia/model.py:826-844 @v0.32.0)."""
    padded_tokens = (max_tokens + 3) // 4 * 4
    return padded_tokens, num_heads, topk


def _msa_query_positions(batch_size: int, seq_len: int, is_context: bool, prefix_len: int = 0) -> list[int]:
    """Absolute positions of only the current query tokens, not cached tokens."""
    start = prefix_len if is_context else seq_len
    width = seq_len if is_context else 1
    return [position for _ in range(batch_size) for position in range(start, start + width)]


def _create_kv_caches_and_metadata(
    vllm_config,
    attn_module,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    prefix_len: int = 0,
    device: str = "cuda:0",
):
    """Create the main + indexer KV caches and both metadata objects via the
    framework's own specs and metadata builders, mirroring serving.

    Metadata: serving (Model Runner V2, the default: config/vllm.py:703-738)
    builds one metadata per KV-cache group via the layer's backend builder
    over CommonAttentionMetadata (build_attn_metadata,
    v1/worker/gpu/attn_utils.py:457-581) and hands the model a
    ``{layer_name: metadata}`` dict plus a per-layer slot-mapping dict
    (build_slot_mappings_by_layer, :446-455, passed into set_forward_context
    at v1/worker/gpu/model_runner.py:2015-2024). The cached prefix travels as
    num_computed_tokens (context_lens) inside CommonAttentionMetadata; the
    builders derive decode/prefill splits themselves
    (split_decodes_and_prefills, common/sparse_attention.py:288-294,
    common/indexer.py:270-276, nvidia/indexer_msa.py:152-158).

    Deviation (generation, Triton indexer: SM89/90/120): serving replays a
    uniform decode step of up to its capture ceiling (min(2 * max_num_seqs,
    1024 on the SM100 family else 512), config/vllm.py:2354-2366) as one FULL
    CUDA graph whose metadata was built at capture with max_seq_len =
    max_model_len (v1/worker/gpu/model_states/default.py:196-201). The Triton
    indexer splits its decode top-k by that bound (chunk_blocks from
    max_block = cdiv(max_seq_len, 128), common/ops/index_topk.py:922,951-958,
    via common/indexer.py:452-467). These rows build decode metadata with the
    case's own max_seq_len, so below max_model_len the top-k work is spread
    over more CTAs than serving's (rows faster than serving there; equal at
    max_model_len). The SM100 indexer ignores max_seq_len (index_decode_score
    .py:476; fixed MAX_K_TILES, nvidia/indexer_msa.py:62), and so do the
    attends. Retire by building decode metadata at serving's max_model_len
    once its deployment value is settled.
    """
    torch_device = torch.device(device)
    block_size = _serving_kv_block_size(vllm_config, attn_module)

    prefix_len = int(prefix_len) if is_context else 0

    if is_context:
        batch_spec = BatchSpec(
            seq_lens=[prefix_len + seq_len] * batch_size,
            query_lens=[seq_len] * batch_size,
        )
    else:
        # Generation KV coordinate contract: `seq_len` is the CACHED length
        # (the getter's kv_cache_len), the current token decodes at position
        # seq_len, and the measured total is seq_len + 1 — the coordinate the
        # loader keys the row at (s = isl + step = 1 + seq_len).
        # create_common_attn_metadata derives num_computed_tokens =
        # seq_lens - query_lens, so seq_lens must carry the TOTAL. An earlier
        # revision passed seq_lens=[seq_len] (cached seq_len - 1, total
        # seq_len) while persisting the seq_len + 1 coordinate (one-token key
        # skew; shipped rows from that revision were re-keyed step -> step-1).
        batch_spec = BatchSpec(
            seq_lens=[seq_len + 1] * batch_size,
            query_lens=[1] * batch_size,
        )

    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size, torch_device, arange_block_indices=True)
    # Serving passes the step's token positions (cached-token count plus the
    # query-local offset, _prepare_pos_seq_lens_kernel, v1/worker/gpu/
    # input_batch.py:348-381) into CommonAttentionMetadata.positions
    # (v1/worker/gpu/attn_utils.py:540). The SM100 indexer builder derives each
    # token's causal page count from it (nvidia/indexer_msa.py:168-172).
    common_attn_metadata.positions = torch.tensor(
        _msa_query_positions(batch_size, seq_len, is_context, prefix_len),
        dtype=torch.long,
        device=torch_device,
    )
    num_blocks = _rebase_block_table_and_slots(common_attn_metadata, block_size)

    # Both caches in serving's physical layout (_serving_kv_cache_layout). The
    # main cache's spec is FullAttentionSpec (nvidia/model.py:562-571), so the
    # element view is [num_blocks, num_kv_heads, block_size, 2*head_size] —
    # merged K|V on the last dim: the MSA impl splits it back with
    # k_cache, v_cache = kv_cache.split(head_size, dim=-1)
    # (nvidia/sparse_attention_msa.py:386) and the fused insert kernel reads
    # the block size off dim 2 (kv_cache.size(2), nvidia/model.py:642).
    # spec.dtype is bf16 for "auto" and uint8 storage for "fp8"
    # (kv_cache_dtype_str_to_dtype, nvidia/model.py:496-498); the impls
    # reinterpret uint8 as e4m3 at use (nvidia/sparse_attention_msa.py:277-278,
    # common/sparse_attention.py:479-481).
    layout = _serving_kv_cache_layout(vllm_config, attn_module)
    backend_cls = attn_module.get_attn_backend()
    kv_cache_spec = attn_module.get_kv_cache_spec(vllm_config)
    kv_cache = _layer_kv_cache(kv_cache_spec, num_blocks, layout, torch_device)

    # Indexer key side-cache: MLAAttentionSpec, key-only — one head slot, one
    # vector per token (common/indexer.py:145-153), logical
    # [num_blocks, 1, block_size, head_size] after the dtype view. Serving
    # hands this 4D view to the layer's bind_kv_cache, which squeezes the
    # H=1 dim (common/indexer.py:141-143); binding below mirrors that.
    index_cache_layer = attn_module.indexer.index_cache
    index_backend_cls = index_cache_layer.get_attn_backend()
    index_spec = index_cache_layer.get_kv_cache_spec(vllm_config)
    index_kv_cache = _layer_kv_cache(index_spec, num_blocks, layout, torch_device)

    # common_prefix_len 0 as serving's AttentionGroup.build_metadata passes
    # (v1/worker/utils.py:341-352); the M3 builders do not read it.
    main_builder = backend_cls.get_builder_cls()(kv_cache_spec, [_ATTN_LAYER_NAME], vllm_config, torch_device)
    attn_metadata = main_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    index_builder = index_backend_cls.get_builder_cls()(
        index_spec, [_INDEX_CACHE_LAYER_NAME], vllm_config, torch_device
    )
    index_metadata = index_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    return kv_cache, index_kv_cache, attn_metadata, index_metadata, common_attn_metadata


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


@with_exit_stack
def run_msa_module(
    exit_stack,
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
):
    """Run a single MiniMax-M3 MSA module-level benchmark point."""
    if kv_cache_dtype not in ("bfloat16", "fp8") or compute_dtype != "bfloat16":
        raise ValueError(
            f"MSA combos are declared with bf16 compute and bf16/fp8 KV cache at "
            f"vllm {vllm_version}; "
            f"got compute={compute_dtype}, kv={kv_cache_dtype}"
        )

    is_context = "context" in perf_filename
    prefix_len = int(prefix_len) if is_context else 0
    phase = "context" if is_context else "generation"
    if not is_context:
        # The plan does not depend on the device: refuse a point whose KV cache cannot fit it.
        reason = _kv_does_not_fit(
            batch_size * seq_len, num_heads, kv_cache_dtype, torch.cuda.get_device_properties(device).total_memory
        )
        if reason is not None:
            raise RuntimeError(reason)

    setup_distributed(device)
    torch.cuda.set_device(device)
    enable_engine_fused_ops()
    init_workspace_manager(torch.device(device))
    print(
        f"\n[MSA module] {phase} b={batch_size}, s={seq_len}, "
        f"prefix={prefix_len}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    # 1. Create attention module (framework builds and dispatches the impls).
    attn_module, vllm_config, original_architecture = _create_msa_attention_module(
        model_path=model_path,
        num_heads=num_heads,
        gemm_type=gemm_type,
        kv_cache_dtype=kv_cache_dtype,
        max_seq_len=prefix_len + seq_len,
        max_batch_size=batch_size,
        max_query_len=seq_len if is_context else 1,
        is_context=is_context,
        device=device,
    )

    # 2. Create KV caches + metadata via the framework's builders.
    with set_current_vllm_config(vllm_config):
        kv_cache, index_kv_cache, attn_metadata, index_metadata, common_attn_metadata = _create_kv_caches_and_metadata(
            vllm_config=vllm_config,
            attn_module=attn_module,
            batch_size=batch_size,
            seq_len=seq_len,
            is_context=is_context,
            prefix_len=prefix_len,
            device=device,
        )

    # 2b. Bind the caches to the registered layers exactly as serving does:
    # forward_context[layer_name].bind_kv_cache(kv_cache)
    # (v1/worker/utils.py:880). The main layer's bind is the
    # AttentionLayerBase default (stores the view as-is,
    # model_executor/layers/attention_layer_base.py:27-33); the indexer's
    # override squeezes the H=1 head dim, [B,1,N,C] -> [B,N,C]
    # (common/indexer.py:141-143).
    forward_ctx = vllm_config.compilation_config.static_forward_context
    forward_ctx[_ATTN_LAYER_NAME].bind_kv_cache(kv_cache)
    forward_ctx[_INDEX_CACHE_LAYER_NAME].bind_kv_cache(index_kv_cache)

    # 2c. Prove the requested KV precision actually reached the framework
    # path — a benchmark that silently ran bf16 under an "fp8" label would be
    # wrong data, worse than a crash. For "fp8": the layer must have been
    # built with cache_dtype "fp8" (nvidia/model.py:493-498), the
    # cache storage must be the fp8 uint8 layout (utils/torch_utils.py:39),
    # and the framework-selected impl must be in fp8 mode (use_fp8_kv,
    # common/sparse_attention.py:412). The indexer side-cache stays bf16
    # in both combos (indexer_kv_dtype "auto" resolves to the layer's "bf16",
    # config/attention.py:88,163-167; nvidia/model.py:507-509).
    want_fp8_kv = kv_cache_dtype == "fp8"
    expected_cache_dtype = torch.uint8 if want_fp8_kv else torch.bfloat16
    impl_use_fp8_kv = bool(getattr(attn_module.impl, "use_fp8_kv", False))
    if (
        kv_cache.dtype != expected_cache_dtype
        or impl_use_fp8_kv != want_fp8_kv
        or index_kv_cache.dtype != torch.bfloat16
    ):
        raise RuntimeError(
            f"KV-cache precision mismatch: requested kv={kv_cache_dtype}, allocated "
            f"main cache {kv_cache.dtype}, impl use_fp8_kv={impl_use_fp8_kv}, "
            f"index cache {index_kv_cache.dtype} "
            f"(layer cache_dtype={attn_module.kv_cache_dtype!r})"
        )

    # 2d. The cached history's index keys (bf16 rows, checked above).
    _write_index_key_history(index_kv_cache)

    # 3. Input tensors. Positions are absolute (cached prefix precedes the
    # current chunk); generation decodes position seq_len with seq_len
    # cached tokens (the KV coordinate contract in
    # _create_kv_caches_and_metadata).
    hidden_size = vllm_config.model_config.hf_text_config.hidden_size
    torch_device = torch.device(device)
    if is_context:
        num_tokens = seq_len * batch_size
        positions = (
            torch.arange(prefix_len, prefix_len + seq_len, device=torch_device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        num_tokens = batch_size
        positions = torch.full((batch_size,), seq_len, device=torch_device, dtype=torch.long)

    hidden_states = torch.full(
        (num_tokens, hidden_size),
        0.01,
        dtype=torch.bfloat16,
        device=torch_device,
    )

    # 4. Forward context: metadata + slot-mapping dicts keyed by layer name,
    # exactly the structures serving passes into set_forward_context
    # (v1/worker/gpu/model_runner.py:2015-2024 with
    # build_slot_mappings_by_layer, v1/worker/gpu/attn_utils.py:446-455).
    # The model consumes slot_mapping[self.layer_name] and
    # slot_mapping[indexer.index_cache.prefix] (nvidia/model.py:605-614) and
    # attn_metadata[layer_name] inside the impls
    # (nvidia/sparse_attention_msa.py:253-256, common/sparse_attention.py:
    # 458-461, common/indexer.py:415-418, nvidia/indexer_msa.py:268-271).
    attn_metadata_dict = {
        _ATTN_LAYER_NAME: attn_metadata,
        _INDEX_CACHE_LAYER_NAME: index_metadata,
    }
    slot_mapping_dict = {
        _ATTN_LAYER_NAME: common_attn_metadata.slot_mapping,
        _INDEX_CACHE_LAYER_NAME: common_attn_metadata.slot_mapping,
    }
    exit_stack.enter_context(set_current_vllm_config(vllm_config))
    exit_stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config, slot_mapping=slot_mapping_dict))

    # Steps 5-7 run under one try/finally: collect.py reuses this worker
    # process for later cases, so the workspace singleton and CUDA
    # allocations must be released on EVERY exit path — a leak here turns
    # one failure into misleading follow-on OOMs.
    try:
        # 5. Dry run
        try:
            with torch.inference_mode():
                attn_module.forward(positions, hidden_states)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  Dry run OOM: {e}")
            raise
        except Exception as e:
            print(f"  Dry run failed: {e}")
            traceback.print_exc()
            # Propagate to collect.py's worker so the failure is recorded in
            # the error queue (no silent skip).
            raise

        # 6. Benchmark. CUDA graph capture is mandatory (allow_graph_fail=False):
        # every builder declares AttentionCGSupport.UNIFORM_BATCH
        # (common/sparse_attention.py:242, common/indexer.py:212,
        # nvidia/indexer_msa.py:116) and the forward has no host syncs (the
        # SM100 prefill plan's .tolist() runs in the builder,
        # nvidia/indexer_msa.py:198-221). The SM100 prefill kernels come from
        # the fmha_sm100 package, pinned at 968061ab (f355c37e at 0.30.0;
        # cmake/external_projects/fmha_sm100.cmake:20), whose capture safety
        # only a GPU run shows.
        #
        # Deviation (graph mode): serving runs MiniMaxM3SparseForCausalLM on
        # breakable CUDA graphs with torch.compile off (config/vllm.py:74-102,
        # 812-840). A step with prefill tokens of up to the capture ceiling
        # (min(2 * max_num_seqs, 1024 on the SM100 family else 512),
        # config/vllm.py:2354-2366) replays graph segments around an eager
        # break at _run_attention (indexer + attend, @eager_break_during_capture,
        # nvidia/model.py:657-675; compilation/breakable_cudagraph.py:101-118,
        # 198-211), and a larger step runs eagerly. A uniform decode step of up
        # to the ceiling replays one FULL graph, attention included
        # (breakable_cudagraph.py:108-111); a larger one runs eagerly. This
        # collector times every case as one CUDA graph, so context rows leave
        # out the host launch gaps of the eager indexer and attend, and
        # generation rows above the ceiling (batch 1024 off the SM100 family)
        # leave out the module's launch gaps: those rows are lower than
        # serving by the launch overhead, largest for small steps. The kernels
        # and their arguments match (no capture-dependent branch in the M3
        # path). Retired by the pending measurement-method decision.
        def kernel_func():
            attn_module.forward(positions, hidden_states)

        with benchmark_with_power(
            device=torch_device,
            kernel_func=kernel_func,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=False,
        ) as results:
            pass

        latency = results["latency_ms"]

        # 7. Log results — schema aligned with TRT-LLM's collect_msa_module.
        if is_context:
            isl = seq_len
            step = prefix_len
        else:
            isl = 1
            step = seq_len

        op_name = f"msa_{phase}_module"

        # Ground truth: the impl class the framework's own dispatch selected at
        # construction (select_main_backend_and_impl_cls — Triton off the
        # SM100 family, MSA on it; common/sparse_attention.py:525-587).
        kernel_source = type(attn_module.impl).__name__

        _persist_msa_row(
            item={
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
            },
            vllm_version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name=op_name,
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )

        print(
            f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
            f"prefix={prefix_len}, gemm={gemm_type}, kv={kv_cache_dtype} "
            f"(cache storage {kv_cache.dtype}), backend={kernel_source}: {latency:.4f} ms"
        )

        return latency
    finally:
        _cleanup()


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
    # Serving executes model forward under inference mode
    # (v1/worker/gpu/model_runner.py:1718-1719, execute_model). Keep module
    # initialization, dry runs, graph warmup and timing in the SAME mode: a
    # cached workspace created by the dry run is mutated later.
    with torch.inference_mode():
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


def _cleanup():
    # Release vLLM's WorkspaceManager singleton scratch buffers between
    # tasks (same rationale as collect_mla_module._cleanup: the manager
    # only grows and pins its high-water mark for the worker's lifetime).
    import vllm.v1.worker.workspace as _ws_mod

    _ws_mod._manager = None
    gc.collect()
    torch.cuda.empty_cache()


def _persist_msa_row(
    *,
    item: dict,
    vllm_version: str,
    device_name: str,
    op_name: str,
    kernel_source: str,
    perf_filename: str,
    power_stats: dict | None,
) -> None:
    """Persist one MSA perf row, failing the case when the write fails.

    ``log_perf`` returns False after lock exhaustion or a write/fsync
    exception; discarding that would let the worker return normally and the
    checkpoint advance with no row persisted (a silently shrunk dataset).
    Mirrors the TRT-LLM and SGLang collectors."""
    if not log_perf(
        item_list=[item],
        framework="VLLM",
        version=vllm_version,
        device_name=device_name,
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise RuntimeError(f"failed to persist MSA row to {perf_filename}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _perf_file_for(mode: str) -> str:
    return PerfFile.MSA_CONTEXT_MODULE if mode == "context" else PerfFile.MSA_GENERATION_MODULE


def main():
    all_model_specs = get_mla_module_model_specs(attention_type="msa", apply_model_filter=False)
    model_names = [spec.model_path for spec in all_model_specs]

    parser = argparse.ArgumentParser(description="MiniMax-M3 MSA module-level collector for vLLM")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--model", type=str, default=None, choices=model_names)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument("--gemm-type", type=str, choices=["bfloat16", "fp8_block", "nvfp4"], default=None)
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["bfloat16", "fp8"],
        default=None,
        help="KV cache dtype (default: run both bfloat16 and fp8)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    perf_filename = _perf_file_for(args.mode)
    if not args.quick:
        # The full sweep is the op plan's cases; --model targets that checkpoint, as collect.py --model-path does.
        if args.model:
            os.environ["COLLECTOR_MODEL_PATH"] = args.model
        test_cases = [
            case
            for case in plan(f"msa_{args.mode}_module", "vllm", get_sm_version()).rows
            if (args.num_heads is None or case.num_heads == args.num_heads)
            and (args.kv_cache_dtype is None or case.kv_cache_dtype == args.kv_cache_dtype)
            and (args.gemm_type is None or case.gemm_type == args.gemm_type)
            and (args.seq_len is None or case.seq_len == args.seq_len)
            and (args.batch_size is None or case.batch_size == args.batch_size)
            and (args.prefix_len is None or case.prefix_len == args.prefix_len)
        ]
        print(f"Running {len(test_cases)} {args.mode} MSA module test cases...")
        num_failed = 0
        for i, case in enumerate(test_cases):
            print(f"[{i + 1}/{len(test_cases)}]", end="")
            try:
                run_msa_module(
                    seq_len=case.seq_len,
                    batch_size=case.batch_size,
                    num_heads=case.num_heads,
                    kv_cache_dtype=case.kv_cache_dtype,
                    compute_dtype=case.compute_dtype,
                    gemm_type=case.gemm_type,
                    prefix_len=case.prefix_len,
                    perf_filename=perf_filename,
                    model_path=case.model_path,
                    device=args.device,
                )
            except Exception as e:
                print(f"  FAILED: {case}: {e}")
                traceback.print_exc()
                num_failed += 1
                torch.cuda.empty_cache()
                gc.collect()
        if num_failed:
            # The registry/executor path records classified failures via the
            # raising worker; this standalone repro CLI at least exits
            # non-zero so a failed run is visible without scraping the log.
            print(f"{num_failed}/{len(test_cases)} cases failed")
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
            kv_cache_dtype=args.kv_cache_dtype or "bfloat16",
            compute_dtype="bfloat16",
            gemm_type=args.gemm_type or "bfloat16",
            prefix_len=args.prefix_len or 0,
            perf_filename=perf_filename,
            model_path=model_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()
